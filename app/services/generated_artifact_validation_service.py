from __future__ import annotations

from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.services.rag_retrieval_service import RetrievedChunk

log = get_logger(__name__)


@dataclass
class ArtifactValidationReport:
    """Deterministic validation of evidence citations and distractor rationale."""

    citations_valid: bool
    rationale_valid: bool
    issues: list[str] = field(default_factory=list)


class GeneratedArtifactValidationService:
    """Deterministic validator for evidence citations and distractor rationale.

    Validates structural correctness only — no LLM judge.  When the
    corresponding config flags (``GENERATED_MCQ_REQUIRE_CITATIONS``,
    ``GENERATED_MCQ_REQUIRE_DISTRACTOR_RATIONALE``) are ``False``,
    the respective check passes silently.
    """

    def __init__(
        self,
        *,
        require_citations: bool = True,
        require_rationale: bool = True,
    ) -> None:
        self._require_citations = require_citations
        self._require_rationale = require_rationale

    def validate(
        self,
        *,
        evidence_citations: list[str] | None,
        distractor_rationale: dict[str, object] | None,
        options: list[dict],
        retrieved_chunks: list[RetrievedChunk],
    ) -> ArtifactValidationReport:
        issues: list[str] = []

        citations_valid = self._validate_citations(
            evidence_citations=evidence_citations,
            retrieved_chunks=retrieved_chunks,
            issues=issues,
        )
        rationale_valid = self._validate_rationale(
            distractor_rationale=distractor_rationale,
            options=options,
            issues=issues,
        )

        return ArtifactValidationReport(
            citations_valid=citations_valid,
            rationale_valid=rationale_valid,
            issues=issues,
        )

    # -- citation validation ---------------------------------------------------

    def _validate_citations(
        self,
        *,
        evidence_citations: list[str] | None,
        retrieved_chunks: list[RetrievedChunk],
        issues: list[str],
    ) -> bool:
        if not self._require_citations:
            return True

        if evidence_citations is None:
            issues.append("evidence_citations is required but missing")
            return False

        if not isinstance(evidence_citations, list):
            issues.append("evidence_citations must be a list")
            return False

        if len(evidence_citations) == 0:
            issues.append("evidence_citations must be non-empty")
            return False

        valid_citations: list[str] = []
        for i, citation in enumerate(evidence_citations):
            if not isinstance(citation, str):
                issues.append(f"citation[{i}] must be a string")
                continue
            stripped = citation.strip()
            if not stripped:
                issues.append(f"citation[{i}] must be non-empty")
                continue
            if len(stripped) < 5:
                issues.append(f"citation[{i}] too short (<5 chars)")
                continue
            valid_citations.append(stripped)

        if not valid_citations:
            issues.append("no valid citations found")
            return False

        if len(valid_citations) != len(evidence_citations):
            issues.append("all citation entries must be valid")
            return False

        # Fuzzy check: at least one citation should share words with
        # the retrieved chunk text (avoids obviously fabricated citations).
        evidence_text = " ".join(c.text.lower() for c in retrieved_chunks)
        if evidence_text:
            evidence_words = set(evidence_text.split())
            matched = 0
            for c in valid_citations:
                citation_words = set(c.lower().split())
                overlap = citation_words & evidence_words
                if len(overlap) >= 2:
                    matched += 1
            if matched == 0:
                issues.append(
                    "no citation overlaps with retrieved evidence text"
                )
                return False

        return True

    # -- distractor rationale validation --------------------------------------

    def _validate_rationale(
        self,
        *,
        distractor_rationale: dict[str, object] | None,
        options: list[dict],
        issues: list[str],
    ) -> bool:
        if not self._require_rationale:
            return True

        if distractor_rationale is None:
            issues.append("distractor_rationale is required but missing")
            return False

        if not isinstance(distractor_rationale, dict):
            issues.append("distractor_rationale must be an object")
            return False

        wrong_indices: list[str] = []
        for i, opt in enumerate(options):
            if not opt.get("is_correct"):
                wrong_indices.append(str(i))

        if not wrong_indices:
            issues.append("no distractor options found to validate rationale against")
            return False

        missing = [idx for idx in wrong_indices if idx not in distractor_rationale]
        if missing:
            issues.append(
                f"distractor_rationale missing for option indices: {missing}"
            )
            return False

        # Correct answer should NOT have a rationale entry
        for i, opt in enumerate(options):
            if opt.get("is_correct") and str(i) in distractor_rationale:
                issues.append(
                    f"distractor_rationale includes correct option index {i}"
                )
                return False

        valid_count = 0
        for idx in wrong_indices:
            text = distractor_rationale.get(idx, "")
            if not isinstance(text, str):
                issues.append(f"rationale[{idx}] must be a string")
                continue
            stripped = text.strip()
            if not stripped:
                issues.append(f"rationale[{idx}] must be non-empty")
                continue
            if len(stripped) < 3:
                issues.append(f"rationale[{idx}] too short (<3 chars)")
                continue
            valid_count += 1

        if valid_count == 0:
            issues.append("no valid distractor rationale entries")
            return False

        if valid_count != len(wrong_indices):
            issues.append("all distractor rationale entries must be valid")
            return False

        return True
