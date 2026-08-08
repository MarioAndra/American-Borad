from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.enums import CognitiveLevel
from app.services.llm_client import LLMClient

log = get_logger(__name__)

# Maximum sequence length used at training time (see model report §2.2).
MAX_SEQUENCE_LENGTH = 280

# Trained label map — report §2.7 documents 0->easy, 1->medium, 2->hard.
# The exported config.json only carries generic LABEL_0..2, so the mapping
# is hardcoded here to stay consistent with the training run.
LABEL_MAP: dict[int, str] = {0: "easy", 1: "medium", 2: "hard"}

# IRT logit centers for each class.  Predictions are reported as a soft
# probability-weighted blend of these so the stored difficulty_estimate stays
# on the same continuous logit scale the calibration/repair gates expect.
# The centers are widened to +/-2 so a high-confidence "hard" prediction can
# reach the calibrator's MAX_DELTA=1.5 window for high-ability students
# (theta up to ~3.5) instead of capping at +1.
LABEL_TO_LOGIT: dict[str, float] = {"easy": -2.0, "medium": 0.0, "hard": 2.0}

# Training-time conventions used when a feature is absent from the runtime.
DEFAULT_BLOOM_CODE = "BT1"
DEFAULT_CONCEPT_COUNT = 1
DEFAULT_SIMILARITY = 0.4

# Stored CognitiveLevel (5-level taxonomy) -> BT1..BT6 code the model expects.
# RememberUnderstand covers BT1/BT2; the training default for absent levels is BT1.
COGNITIVE_TO_BT: dict[str, str] = {
    CognitiveLevel.RememberUnderstand.value: "BT1",
    CognitiveLevel.Apply.value: "BT3",
    CognitiveLevel.Analyze.value: "BT4",
    CognitiveLevel.Evaluate.value: "BT5",
    CognitiveLevel.Create.value: "BT6",
}


class MCQDifficultyError(RuntimeError):
    """Raised when the RoBERTa difficulty model cannot load or inference fails."""


@dataclass(frozen=True)
class MCQDifficultyPrediction:
    label: str
    logit: float
    confidence: float
    probabilities: dict[str, float]


def _clean(text: object) -> str:
    """Match the report's cleaning protocol: strip, drop NULs, N/A on empty."""
    if text is None:
        return "N/A"
    value = str(text).replace("\x00", "").strip()
    return value or "N/A"


def build_difficulty_prompt(
    question_text: str,
    options: list[str],
    correct_index: int,
    bloom_code: str | None = None,
    concept_count: int | None = None,
    similarity_mean: float | None = None,
) -> str:
    """Assemble the enriched prompt the classifier was trained on (report §2.4)."""
    bloom = bloom_code or DEFAULT_BLOOM_CODE
    concept_count = concept_count if concept_count is not None else DEFAULT_CONCEPT_COUNT
    similarity_mean = similarity_mean if similarity_mean is not None else DEFAULT_SIMILARITY
    letters = ["A", "B", "C", "D"]
    options_str = " ".join(
        f"{letters[i]}) {_clean(options[i])}" for i in range(min(len(options), 4))
    )
    correct_letter = letters[correct_index] if 0 <= correct_index < len(letters) else letters[0]
    return (
        f"Task: MCQ Difficulty Evaluation | "
        f"Cognitive Bloom Taxonomy Level: {bloom} | "
        f"Concept Count: {concept_count} | "
        f"Distractor Similarity Score: {similarity_mean:.2f} | "
        f"Question Text: {_clean(question_text)} | "
        f"Options: {options_str} | "
        f"Correct Option: {correct_letter}"
    )


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def answer_similarity_mean(
    option_texts: list[str],
    correct_index: int,
    embeddings: list[list[float] | None] | None,
) -> float | None:
    """Mean cosine similarity between the correct option and the distractors.

    Mirrors the training-time feature (report §2.3.2).  Returns ``None`` when
    the embeddings are unusable so callers can fall back to a default.
    """
    if not embeddings or correct_index < 0 or correct_index >= len(embeddings):
        return None
    correct_vec = embeddings[correct_index]
    if correct_vec is None:
        return None
    similarities = [
        _cosine_similarity(correct_vec, vec)
        for i, vec in enumerate(embeddings)
        if i != correct_index and vec is not None
    ]
    if not similarities:
        return None
    return float(sum(similarities) / len(similarities))


class MCQDifficultyService:
    """Loads the local RoBERTa MCQ difficulty classifier once and reuses it.

    Heavy libraries (torch, transformers) and the model itself are loaded
    lazily on the first prediction so importing this module stays cheap.
    """

    def __init__(self, model_path: str | None, device: str = "auto") -> None:
        self.model_path = model_path
        self.device = device
        self.tokenizer = None
        self.model = None
        self._resolved_device: str | None = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _ensure_loaded(self) -> None:
        if self.is_loaded:
            return
        with self._lock:
            if self.is_loaded:
                return
            started = time.perf_counter()
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self._resolved_device = self._resolve_device()
                path = self.model_path
                if not path or not str(path).strip():
                    raise MCQDifficultyError(
                        "MCQ difficulty model path is not configured "
                        "(MCQ_DIFFICULTY_MODEL_PATH)"
                    )
                if not Path(str(path)).exists():
                    raise MCQDifficultyError(f"MCQ difficulty model path not found: {path}")
                self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    path, local_files_only=True
                )
                self.model.eval()
                self.model.to(self._resolved_device)
                elapsed = time.perf_counter() - started
                log.info(
                    "MCQ difficulty model loaded from %s on %s in %.2fs",
                    path,
                    self._resolved_device,
                    elapsed,
                )
            except MCQDifficultyError:
                raise
            except Exception as exc:
                log.error("MCQ difficulty model load failed: %s", exc)
                raise MCQDifficultyError("Failed to load MCQ difficulty model") from exc

    def predict(
        self,
        question_text: str,
        options: list[str],
        correct_index: int,
        bloom_code: str | None = None,
        concept_count: int | None = None,
        similarity_mean: float | None = None,
    ) -> MCQDifficultyPrediction:
        """Classify a question and return a soft probability-weighted logit.

        Raises ``ValueError`` for empty input and ``MCQDifficultyError`` for
        any model load or inference failure.
        """
        if not question_text or not question_text.strip():
            raise ValueError("question_text must not be empty")
        if correct_index < 0 or correct_index >= len(options):
            raise ValueError(f"correct_index {correct_index} out of range")

        self._ensure_loaded()

        try:
            import torch

            if self.tokenizer is None or self.model is None:
                raise MCQDifficultyError("MCQ difficulty model is not loaded")

            prompt = build_difficulty_prompt(
                question_text=question_text,
                options=options,
                correct_index=correct_index,
                bloom_code=bloom_code,
                concept_count=concept_count,
                similarity_mean=similarity_mean,
            )
            started = time.perf_counter()
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
                padding=True,
            )
            inputs = {k: v.to(self._resolved_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            probs = torch.softmax(outputs.logits[0], dim=-1).detach().cpu().numpy()
            idx = int(probs.argmax())
            label = LABEL_MAP.get(idx)
            if label is None:
                raise MCQDifficultyError(f"Unexpected label index: {idx}")
            confidence = float(probs[idx])
            probabilities = {
                LABEL_MAP[i]: float(prob)
                for i, prob in enumerate(probs)
                if i in LABEL_MAP
            }
            logit = float(
                sum(probabilities[cls] * LABEL_TO_LOGIT[cls] for cls in probabilities)
            )
            elapsed = time.perf_counter() - started
            log.info(
                "MCQ difficulty prediction: label=%s logit=%.3f confidence=%.4f "
                "latency=%.3fs",
                label, logit, confidence, elapsed,
            )
            return MCQDifficultyPrediction(
                label=label,
                logit=logit,
                confidence=confidence,
                probabilities=probabilities,
            )
        except MCQDifficultyError:
            raise
        except Exception as exc:
            log.error("MCQ difficulty inference failed: %s", exc)
            raise MCQDifficultyError("MCQ difficulty inference failed") from exc

    def check_ready(self) -> bool:
        try:
            self._ensure_loaded()
            return True
        except MCQDifficultyError:
            return False


_service: MCQDifficultyService | None = None
_service_lock = threading.Lock()


def get_service() -> MCQDifficultyService:
    """Return the process-wide classifier singleton (lazy creation).

    Uses the bundled ``app/services/mcq_difficulty_roberta/`` directory unless
    ``MCQ_DIFFICULTY_MODEL_PATH`` points elsewhere.
    """
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                settings = get_settings()
                bundled = Path(__file__).with_name("mcq_difficulty_roberta")
                _service = MCQDifficultyService(
                    model_path=settings.MCQ_DIFFICULTY_MODEL_PATH or str(bundled),
                    device=settings.MCQ_DIFFICULTY_MODEL_DEVICE,
                )
    return _service


def predict(
    question_text: str,
    options: list[str],
    correct_index: int,
    bloom_code: str | None = None,
    concept_count: int | None = None,
    similarity_mean: float | None = None,
) -> MCQDifficultyPrediction:
    """Classify ``question_text`` using the singleton classifier service."""
    return get_service().predict(
        question_text,
        options,
        correct_index,
        bloom_code=bloom_code,
        concept_count=concept_count,
        similarity_mean=similarity_mean,
    )


def estimate_generated_difficulty(
    gen_output,
    cognitive_level: str | None = None,
) -> MCQDifficultyPrediction | None:
    """Predict difficulty for a ``GenerationOutput`` and override its estimate.

    Computes the distractor-similarity feature via the configured embedding
    provider (the same ``text-embedding-3-small`` family the classifier was
    trained with), assembles the enriched prompt, and returns a
    probability-weighted IRT logit.

    Fail-open by default: any error logs and returns ``None`` so callers keep
    the LLM-reported estimate.  With ``MCQ_DIFFICULTY_MODEL_FAIL_OPEN=false``
    a failure re-raises :class:`MCQDifficultyError` so callers can block.
    """
    try:
        settings = get_settings()
        if not settings.MCQ_DIFFICULTY_MODEL_ENABLED:
            return None
        if gen_output is None or not getattr(gen_output, "options", None):
            return None

        options = [str(o.get("text", "")) for o in gen_output.options]
        correct_index = next(
            (i for i, o in enumerate(gen_output.options) if o.get("is_correct")),
            None,
        )
        if correct_index is None:
            log.warning("MCQ difficulty: no correct option found — skipping override")
            return None

        similarity_mean = None
        try:
            llm = LLMClient(provider=settings.RAG_EMBEDDING_PROVIDER)
            embeddings = llm.embed(options)
            similarity_mean = answer_similarity_mean(options, correct_index, embeddings)
        except Exception:
            log.exception("MCQ difficulty: embedding failed — using default similarity")

        bloom_code = COGNITIVE_TO_BT.get(cognitive_level) if cognitive_level else None
        prediction = get_service().predict(
            gen_output.question_text,
            options,
            correct_index,
            bloom_code=bloom_code,
            similarity_mean=similarity_mean,
        )
        return prediction
    except MCQDifficultyError:
        if get_settings().MCQ_DIFFICULTY_MODEL_FAIL_OPEN:
            log.exception("MCQ difficulty model failed — keeping LLM estimate")
            return None
        raise
    except Exception:
        log.exception("MCQ difficulty estimation failed — keeping LLM estimate")
        return None
