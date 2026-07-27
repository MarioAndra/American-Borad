from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import tiktoken
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models import Question
from app.services.llm_client import LLMClient

log = get_logger(__name__)
_ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class ValidationReport:
    valid: bool
    issues: list[str] = field(default_factory=list)
    schema_ok: bool = False
    single_correct: bool = False
    non_duplicate: bool = True
    judge_feedback: str | None = None
    judge_ok: bool | None = None
    judge_ambiguity: bool | None = None
    judge_factual_error: bool | None = None
    max_similarity: float = 0.0


class GeneratedQuestionValidationService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.settings = get_settings()
        self._llm = LLMClient(provider=self.settings.RAG_GENERATION_PROVIDER)

    def validate(
        self,
        question_text: str,
        options: list[dict],
        explanation: str,
    ) -> ValidationReport:
        report = ValidationReport(valid=True)

        self._check_schema(question_text, options, explanation, report)
        self._check_single_correct(options, report)

        if self.settings.RAG_REVIEW_REQUIRED or self.settings.RAG_ENABLED:
            self._llm_judge(question_text, options, explanation, report)

        report.valid = all([
            report.schema_ok,
            report.single_correct,
            report.non_duplicate,
            report.judge_ok is not False,
        ])
        return report

    def _check_schema(self, question_text: str, options: list[dict], explanation: str, report: ValidationReport) -> None:
        if not question_text or not isinstance(question_text, str):
            report.issues.append("question_text missing or not a string")
            return
        if not options or not isinstance(options, list):
            report.issues.append("options missing or not a list")
            return
        if len(options) != self.settings.GENERATED_MCQ_OPTION_COUNT:
            report.issues.append(f"Expected {self.settings.GENERATED_MCQ_OPTION_COUNT} options, got {len(options)}")
            return
        for i, opt in enumerate(options):
            if not isinstance(opt, dict) or "text" not in opt or "is_correct" not in opt:
                report.issues.append(f"Option {i} has invalid structure")
                return
        if not explanation or not isinstance(explanation, str):
            report.issues.append("explanation missing or not a string")
            return
        report.schema_ok = True

    def _check_single_correct(self, options: list[dict], report: ValidationReport) -> None:
        correct_count = sum(1 for o in options if o.get("is_correct"))
        if correct_count == 1:
            report.single_correct = True
        else:
            report.issues.append(f"Expected exactly 1 correct option, found {correct_count}")
            report.single_correct = False

    def _llm_judge(self, question_text: str, options: list[dict], explanation: str, report: ValidationReport) -> None:
        if not self._llm.is_available:
            report.judge_feedback = "LLM judge skipped: no API key"
            report.judge_ok = False
            report.issues.append("LLM judge unavailable: no API key configured")
            return

        correct_text = next((o["text"] for o in options if o.get("is_correct")), "")
        options_text = "\n".join(f'  {"✓" if o.get("is_correct") else "○"} {o["text"]}' for o in options)

        prompt = (
            f"Review the following SingleChoice MCQ for quality:\n\n"
            f"Question: {question_text}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Explanation: {explanation}\n"
            f"Correct answer: {correct_text}\n\n"
            f"Check for:\n"
            f"1. Ambiguity: is there more than one plausible correct answer?\n"
            f"2. Factual consistency: does the explanation match the correct answer?\n"
            f"3. Clarity: is the question clearly worded?\n\n"
            f"Respond with a JSON object:\n"
            f'{{"valid": true/false, "feedback": "brief explanation", "ambiguity_found": true/false, "factual_error": true/false}}'
        )

        try:
            content = self._llm.generate_chat(
                messages=[
                    {"role": "system", "content": "You are an MCQ quality reviewer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            if content:
                parsed = json.loads(content)
                report.judge_feedback = parsed.get("feedback", "")
                report.judge_ok = parsed.get("valid", True)
                report.judge_ambiguity = parsed.get("ambiguity_found", False)
                report.judge_factual_error = parsed.get("factual_error", False)
                if not parsed.get("valid", True) or parsed.get("ambiguity_found") or parsed.get("factual_error"):
                    report.issues.append(f"LLM judge: {parsed.get('feedback', 'quality issue detected')}")
        except Exception as exc:
            log.warning("LLM judge call failed: %s", exc)
            report.judge_feedback = f"Judge call failed: {exc}"
            report.judge_ok = False
            report.issues.append(f"LLM judge call failed: {exc}")
