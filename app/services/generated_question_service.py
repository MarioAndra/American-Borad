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


@dataclass
class GenerationInput:
    topic_id: int
    topic_name: str
    theta: float
    recent_streak: int
    avg_theta: float | None
    retrieved_chunks: list[RetrievedChunk]


@dataclass
class GenerationOutput:
    question_text: str
    options: list[dict]
    explanation: str
    difficulty_estimate: float
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
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        if not raw:
            log.error("Empty generation response")
            return None

        try:
            parsed = json.loads(raw)
            return GenerationOutput(
                question_text=parsed["question_text"],
                options=parsed["options"],
                explanation=parsed.get("explanation", ""),
                difficulty_estimate=float(parsed.get("difficulty_estimate", 0.5)),
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
            "Return valid JSON with these exact keys:",
            '  "question_text": string (the question)',
            '  "options": array of {"text": string, "is_correct": boolean} (exactly 4 options, exactly 1 correct)',
            '  "explanation": string (explain why the correct answer is right)',
            '  "difficulty_estimate": float (0.0=very easy, 1.0=very hard)',
        ]
        if self.settings.GENERATED_MCQ_REQUIRE_CITATIONS:
            parts.append('  "evidence_citations": array of strings (specific references to the material)')
        if self.settings.GENERATED_MCQ_REQUIRE_DISTRACTOR_RATIONALE:
            parts.append('  "distractor_rationale": object mapping each wrong option index to a brief explanation of why it is wrong')
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

        return (
            f"Generate an MCQ based on the following course material.\n\n"
            f"{context}\n\n"
            f"--- Student Performance Context ---\n"
            f"{performance}\n\n"
            f"The difficulty should be appropriate for a student with ability theta={inp.theta:.2f}."
        )
