from __future__ import annotations

import json
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models import Topic
from app.services.llm_client import LLMClient
from app.services.rag_retrieval_service import RetrievedChunk

log = get_logger(__name__)


# Difficulty-band thresholds mirror difficulty_calibration_service._band
# so the LLM's self-reported estimate lands in the same band the calibrator
# will check against.
_EASY_UPPER: float = -0.5
_HARD_LOWER: float = 0.5

# Concrete, per-band generation guidance — keeps the LLM from drifting toward
# easy/medium items when the target student is high-ability (theta > 0.5).
_DIFFICULTY_BAND_GUIDANCE: dict[str, str] = {
    "easy": (
        "Design the question to be EASY for this student: single-step recall or "
        "direct application of one concept. Keep the distractors clearly wrong."
    ),
    "medium": (
        "Design the question to be MEDIUM for this student: combine two concepts "
        "or apply knowledge in a familiar scenario. Distractors should be "
        "plausible but clearly distinguishable from the correct answer."
    ),
    "hard": (
        "Design the question to be HARD for this student — it must challenge "
        "a strong, high-ability student. Requirements:\n"
        "- It must NOT be a recall question: do not ask what a term or "
        "concept is, and do not write 'according to the provided context' "
        "style questions.\n"
        "- It must require the student to APPLY or COMBINE at least two "
        "distinct concepts from the material (e.g. analyze a scenario, "
        "compare or rank approaches, diagnose a flaw, or reason through a "
        "multi-step implication).\n"
        "- Use precise technical detail and edge-case conditions.\n"
        "- Make the distractors highly plausible and only subtly wrong — "
        "each should tempt a strong student who reasons too quickly. Avoid "
        "obviously wrong or generic distractors."
    ),
}


def _difficulty_band_label(theta: float) -> str:
    if theta < _EASY_UPPER:
        return "easy"
    if theta > _HARD_LOWER:
        return "hard"
    return "medium"


@dataclass
class GenerationInput:
    topic_id: int
    topic_name: str
    theta: float
    recent_streak: int
    avg_theta: float | None
    retrieved_chunks: list[RetrievedChunk]
    extra_instructions: str | None = None


@dataclass
class GenerationOutput:
    question_text: str
    options: list[dict]
    explanation: str
    difficulty_estimate: float | None
    evidence_citations: list[str] | None = None
    distractor_rationale: dict[str, object] | None = None
    raw_response: str | None = None


class GeneratedQuestionService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.settings = get_settings()

        self._llm = LLMClient(provider=self.settings.RAG_GENERATION_PROVIDER)
        if not self._llm.is_available:
            log.warning("No LLM client available — generation disabled")

    def generate(self, inp: GenerationInput) -> GenerationOutput | None:
        if not self._llm.is_available:
            log.error("Cannot generate: no LLM client configured")
            return None

        prompt = self._build_prompt(inp)
        raw = self._llm.generate_chat(
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        if not raw:
            log.error("Empty generation response")
            return None

        try:
            parsed = json.loads(raw)
            raw_diff = parsed.get("difficulty_estimate")
            difficulty_estimate = float(raw_diff) if raw_diff is not None else None

            evidence_citations: list[str] | None = None
            raw_citations = parsed.get("evidence_citations")
            if isinstance(raw_citations, list):
                evidence_citations = raw_citations

            distractor_rationale: dict[str, object] | None = None
            raw_rationale = parsed.get("distractor_rationale")
            if isinstance(raw_rationale, dict):
                distractor_rationale = {str(k): v for k, v in raw_rationale.items()}

            return GenerationOutput(
                question_text=parsed["question_text"],
                options=parsed["options"],
                explanation=parsed.get("explanation", ""),
                difficulty_estimate=difficulty_estimate,
                evidence_citations=evidence_citations,
                distractor_rationale=distractor_rationale,
                raw_response=raw,
            )
        except json.JSONDecodeError as exc:
            log.error("Generation response was not valid JSON: %s", exc)
            return None
        except (KeyError, ValueError, TypeError) as exc:
            log.error("Generation response missing required fields: %s", exc)
            return None

    def _system_prompt(self) -> str:
        parts = [
            "You are an expert exam question generator for the American Board of AI certification.",
            "Generate exactly one SingleChoice multiple-choice question.",
            "",
            "RULES:",
            "- Base the question ONLY on the provided context material. Do not use external knowledge.",
            "- The correct answer must be directly supported by the context.",
            "- All distractors must be plausible but clearly wrong based on the context.",
            "",
            "Return valid JSON with these exact keys:",
            '  "question_text": string (the question)',
            '  "options": array of {"text": string, "is_correct": boolean} (exactly 4 options, exactly 1 correct)',
            '  "explanation": string (explain why the correct answer is right, referencing the context)',
            '  "difficulty_estimate": float (IRT logit scale: negative=easy, 0.0=medium, positive=hard)',
        ]
        if self.settings.GENERATED_MCQ_REQUIRE_CITATIONS:
            parts.append('  "evidence_citations": array of strings (exact quotes or close paraphrases from the context material)')
        if self.settings.GENERATED_MCQ_REQUIRE_DISTRACTOR_RATIONALE:
            parts.append('  "distractor_rationale": object mapping each wrong option INDEX (0-based) to a brief explanation of why it is wrong')
        return "\n".join(parts)

    def _build_prompt(self, inp: GenerationInput) -> str:
        context_parts = [f"--- Context Material (topic: {inp.topic_name}) ---"]
        for i, c in enumerate(inp.retrieved_chunks):
            context_parts.append(f"[{i+1}] {c.text[:600]}")
            if i >= 4:
                break
        context = "\n\n".join(context_parts)

        avg_theta_str = f"{inp.avg_theta:.2f}" if inp.avg_theta is not None else "N/A"
        performance = (
            f"Student ability (theta): {inp.theta:.2f}\n"
            f"Recent same-topic streak: {inp.recent_streak}\n"
            f"Average theta in this topic: {avg_theta_str}"
        )

        band = _difficulty_band_label(inp.theta)
        band_guidance = _DIFFICULTY_BAND_GUIDANCE[band]

        prompt = (
            f"Generate an MCQ based on the following course material.\n\n"
            f"{context}\n\n"
            f"--- Student Performance Context ---\n"
            f"{performance}\n\n"
            f"The difficulty should be appropriate for a student with ability "
            f"theta={inp.theta:.2f} (difficulty band: {band}).\n"
            f"{band_guidance}\n\n"
            f"Your 'difficulty_estimate' must match this band: "
            f"easy ~ -1.0, medium ~ 0.0, hard ~ +1.5."
        )

        if inp.extra_instructions:
            prompt += f"\n\n--- Additional Instructions ---\n{inp.extra_instructions}"

        return prompt
