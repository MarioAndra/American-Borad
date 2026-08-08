from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.enums import CognitiveLevel

log = get_logger(__name__)

# Maximum sequence length used at training time (see model report).
MAX_SEQUENCE_LENGTH = 256

# Model codes (from the fine-tuned DeBERTa checkpoint) mapped to the stored
# 5-level taxonomy values. Order follows the model's label dictionary.
BT_CODE_TO_LABEL: dict[str, CognitiveLevel] = {
    "BT3": CognitiveLevel.Apply,
    "BT4": CognitiveLevel.Analyze,
    "BT5": CognitiveLevel.Evaluate,
    "BT6": CognitiveLevel.Create,
    "BT7": CognitiveLevel.RememberUnderstand,
}


class BloomClassifierError(RuntimeError):
    """Raised when the Bloom model cannot be loaded or an inference fails."""


@dataclass(frozen=True)
class BloomPrediction:
    label: CognitiveLevel
    label_code: str
    confidence: float
    probabilities: dict[str, float] | None = None


def label_from_code(code: str) -> CognitiveLevel:
    try:
        return BT_CODE_TO_LABEL[code]
    except KeyError:
        raise ValueError(f"Unknown Bloom label code: {code}") from None


class BloomClassifierService:
    """Loads the local DeBERTa Bloom classifier once and reuses it.

    The heavy libraries (torch, transformers) and the model itself are
    loaded lazily on the first prediction so that importing this module is
    cheap and the rest of the app can boot even without them installed.
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

    @staticmethod
    def _id2label_code(config, idx: int) -> str:
        """Look up a label code from the model config, tolerating int or str
        keys (transformers 4.x stores str keys, 5.x stores int keys)."""
        try:
            return config.id2label[idx]
        except (KeyError, TypeError):
            return config.id2label[str(idx)]

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
                    raise BloomClassifierError(
                        "Bloom model path is not configured (BLOOM_MODEL_PATH)"
                    )
                if not Path(str(path)).exists():
                    raise BloomClassifierError(f"Bloom model path not found: {path}")
                self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    path, local_files_only=True
                )
                self.model.eval()
                self.model.to(self._resolved_device)
                elapsed = time.perf_counter() - started
                log.info(
                    "Bloom model loaded from %s on %s in %.2fs",
                    path,
                    self._resolved_device,
                    elapsed,
                )
            except BloomClassifierError:
                raise
            except Exception as exc:
                log.error("Bloom model load failed: %s", exc)
                raise BloomClassifierError("Failed to load Bloom model") from exc

    def predict(self, question_text: str) -> BloomPrediction:
        """Classify ``question_text`` and return the predicted taxonomy level.

        Raises ``ValueError`` for empty/blank input and ``BloomClassifierError``
        for any model load or inference failure.
        """
        if not question_text or not question_text.strip():
            raise ValueError("question_text must not be empty")

        self._ensure_loaded()

        try:
            import torch

            if self.tokenizer is None or self.model is None:
                raise BloomClassifierError("Bloom model is not loaded")

            started = time.perf_counter()
            inputs = self.tokenizer(
                question_text.strip(),
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
                padding=True,
            )
            inputs = {k: v.to(self._resolved_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            idx = int(probs.argmax())
            code = self._id2label_code(self.model.config, idx)
            label = label_from_code(code)
            confidence = float(probs[idx])
            elapsed = time.perf_counter() - started

            probabilities: dict[str, float] | None = None
            if get_settings().BLOOM_MODEL_RETURN_PROBABILITIES:
                probabilities = {
                    self._id2label_code(self.model.config, i): float(prob)
                    for i, prob in enumerate(probs)
                }

            log.info(
                "Bloom prediction: label=%s code=%s confidence=%.4f latency=%.3fs",
                label.value,
                code,
                confidence,
                elapsed,
            )
            return BloomPrediction(
                label=label,
                label_code=code,
                confidence=confidence,
                probabilities=probabilities,
            )
        except BloomClassifierError:
            raise
        except Exception as exc:
            log.error("Bloom inference failed: %s", exc)
            raise BloomClassifierError("Bloom inference failed") from exc

    def check_ready(self) -> bool:
        """Return True when the model can be loaded from disk."""
        try:
            self._ensure_loaded()
            return True
        except BloomClassifierError:
            return False


_service: BloomClassifierService | None = None
_service_lock = threading.Lock()


def get_service() -> BloomClassifierService:
    """Return the process-wide classifier singleton (lazy creation)."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                settings = get_settings()
                _service = BloomClassifierService(
                    model_path=settings.BLOOM_MODEL_PATH,
                    device=settings.BLOOM_MODEL_DEVICE,
                )
    return _service


def predict(question_text: str) -> BloomPrediction:
    """Classify ``question_text`` using the singleton classifier service."""
    return get_service().predict(question_text)
