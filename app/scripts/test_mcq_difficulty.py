"""Verification tests for the RoBERTa MCQ difficulty integration.

Run:  python -m app.scripts.test_mcq_difficulty

Pure tests (prompt building, similarity features, fail-open behavior) run
without torch/transformers and without a database. Model-backed and
LangGraph-node tests are skipped with a note when the heavy dependencies
(torch/transformers/langgraph) are unavailable — run those in the Docker
app container.
"""
from __future__ import annotations

from app.services import mcq_difficulty_service
from app.services.mcq_difficulty_service import (
    DEFAULT_BLOOM_CODE,
    DEFAULT_CONCEPT_COUNT,
    LABEL_MAP,
    LABEL_TO_LOGIT,
    MCQDifficultyError,
    MCQDifficultyService,
    MCQDifficultyPrediction,
    answer_similarity_mean,
    build_difficulty_prompt,
    _cosine_similarity,
)

try:
    from app.services.generated_question_service import GenerationOutput

    _HAS_GEN_OUTPUT = True
except Exception:
    _HAS_GEN_OUTPUT = False

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    import langgraph  # noqa: F401

    _HAS_LANGGRAPH = True
except Exception:
    _HAS_LANGGRAPH = False

_RESULTS: list[tuple[str, bool, str]] = []


def _ok(name: str) -> None:
    _RESULTS.append((name, True, ""))


def _fail(name: str, msg: str) -> None:
    _RESULTS.append((name, False, msg))


def _skip(name: str, msg: str) -> None:
    _RESULTS.append((name, True, f"SKIPPED: {msg}"))


def _sample_options() -> list[str]:
    return ["Water", "Carbon dioxide", "NADPH", "Oxygen"]


# ---------------------------------------------------------------------------
# Pure unit tests (no model, no DB)
# ---------------------------------------------------------------------------


def test_label_maps_consistent() -> None:
    name = "1. label maps consistent"
    try:
        assert LABEL_MAP == {0: "easy", 1: "medium", 2: "hard"}
        assert LABEL_TO_LOGIT == {"easy": -2.0, "medium": 0.0, "hard": 2.0}
        assert set(LABEL_MAP.values()) == set(LABEL_TO_LOGIT)
        _ok(name)
    except Exception as exc:
        _fail(name, str(exc))


def test_prompt_template_matches_report() -> None:
    name = "2. prompt template matches report"
    try:
        prompt = build_difficulty_prompt(
            question_text="In photosynthesis, which molecule serves as the primary electron donor?",
            options=_sample_options(),
            correct_index=0,
            bloom_code="BT4",
            concept_count=3,
            similarity_mean=0.55,
        )
        assert "Task: MCQ Difficulty Evaluation" in prompt
        assert "Cognitive Bloom Taxonomy Level: BT4" in prompt
        assert "Concept Count: 3" in prompt
        assert "Distractor Similarity Score: 0.55" in prompt
        assert "Correct Option: A" in prompt
        assert "B) Carbon dioxide" in prompt
        assert "C) NADPH" in prompt
        assert "D) Oxygen" in prompt
        _ok(name)
    except Exception as exc:
        _fail(name, str(exc))


def test_prompt_defaults_when_features_absent() -> None:
    name = "3. prompt defaults when features absent"
    try:
        prompt = build_difficulty_prompt(
            question_text="What is the capital of France?",
            options=["Berlin", "Paris", "Rome", "Madrid"],
            correct_index=1,
        )
        assert DEFAULT_BLOOM_CODE in prompt
        assert f"Concept Count: {DEFAULT_CONCEPT_COUNT}" in prompt
        assert "Distractor Similarity Score: 0.40" in prompt
        assert "Correct Option: B" in prompt
        _ok(name)
    except Exception as exc:
        _fail(name, str(exc))


def test_cosine_similarity() -> None:
    name = "4. cosine similarity"
    try:
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
        _ok(name)
    except Exception as exc:
        _fail(name, str(exc))


def test_answer_similarity_mean() -> None:
    name = "5. answer similarity mean"
    try:
        # correct = [1,0]; distractors identical, opposite, orthogonal
        sims = answer_similarity_mean(
            ["A", "B", "C", "D"],
            correct_index=0,
            embeddings=[[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
        )
        assert sims is not None
        assert abs(sims - ((1.0 - 1.0 + 0.0) / 3)) < 1e-9
        assert answer_similarity_mean(["A", "B"], 0, None) is None
        assert answer_similarity_mean(["A", "B"], 5, [[1.0], [1.0]]) is None
        assert answer_similarity_mean(["A", "B"], 0, [[1.0], None]) is None
        _ok(name)
    except Exception as exc:
        _fail(name, str(exc))


def test_empty_text_rejected() -> None:
    name = "6. empty text rejected"
    try:
        service = MCQDifficultyService(model_path="/nonexistent/model")
        try:
            service.predict("", _sample_options(), 0)
            raise AssertionError("empty question was not rejected")
        except ValueError:
            pass
        _ok(name)
    except Exception as exc:
        _fail(name, str(exc))


def test_bad_correct_index_rejected() -> None:
    name = "7. bad correct index rejected"
    try:
        service = MCQDifficultyService(model_path="/nonexistent/model")
        try:
            service.predict("What is X?", _sample_options(), 9)
            raise AssertionError("out-of-range index was not rejected")
        except ValueError:
            pass
        _ok(name)
    except Exception as exc:
        _fail(name, str(exc))


def test_missing_model_path_handled() -> None:
    name = "8. missing model path handled"
    try:
        service = MCQDifficultyService(model_path="/nonexistent/mcq_roberta")
        try:
            service.predict("What is X?", _sample_options(), 0)
            raise AssertionError("missing model path did not raise")
        except MCQDifficultyError:
            pass
        _ok(name)
    except Exception as exc:
        _fail(name, str(exc))


def test_estimate_disabled_returns_none() -> None:
    name = "9. estimation skipped when model disabled"
    try:
        settings = mcq_difficulty_service.get_settings()
        prev = settings.MCQ_DIFFICULTY_MODEL_ENABLED
        settings.MCQ_DIFFICULTY_MODEL_ENABLED = False
        try:
            if not _HAS_GEN_OUTPUT:
                _skip(name, "GenerationOutput import unavailable")
                return
            out = GenerationOutput(
                question_text="What is the capital of France?",
                options=[
                    {"text": "Berlin", "is_correct": False},
                    {"text": "Paris", "is_correct": True},
                    {"text": "Rome", "is_correct": False},
                    {"text": "Madrid", "is_correct": False},
                ],
                explanation="Paris is the capital.",
                difficulty_estimate=0.1,
            )
            assert mcq_difficulty_service.estimate_generated_difficulty(out) is None
            _ok(name)
        finally:
            settings.MCQ_DIFFICULTY_MODEL_ENABLED = prev
    except Exception as exc:
        _fail(name, str(exc))


def test_generation_prompt_difficulty_bands() -> None:
    name = "13. generation prompt difficulty-band guidance"
    try:
        from app.services.generated_question_service import (
            _DIFFICULTY_BAND_GUIDANCE,
            _difficulty_band_label,
        )

        assert _difficulty_band_label(-1.0) == "easy"
        assert _difficulty_band_label(0.0) == "medium"
        assert _difficulty_band_label(2.1) == "hard"
        assert "combine at least two distinct concepts" in _DIFFICULTY_BAND_GUIDANCE["hard"].lower()
        assert "not be a recall question" in _DIFFICULTY_BAND_GUIDANCE["hard"].lower()
        assert "single-step recall" in _DIFFICULTY_BAND_GUIDANCE["easy"]
        _ok(name)
    except Exception as exc:
        _fail(name, str(exc))


# ---------------------------------------------------------------------------
# Model-backed tests (skipped when torch/transformers are unavailable)
# ---------------------------------------------------------------------------


def test_bundled_model_ready_and_predicts() -> None:
    name = "10. bundled model loads and predicts"
    if not _HAS_TORCH:
        _skip(name, "torch/transformers not installed locally")
        return
    try:
        settings = mcq_difficulty_service.get_settings()
        prev_path = settings.MCQ_DIFFICULTY_MODEL_PATH
        settings.MCQ_DIFFICULTY_MODEL_PATH = None  # use bundled dir
        try:
            service = mcq_difficulty_service.get_service()
            assert service.check_ready(), "bundled model did not load"
            pred = service.predict(
                "In photosynthesis, which molecule serves as the primary electron donor?",
                _sample_options(),
                correct_index=0,
                bloom_code="BT4",
                concept_count=3,
                similarity_mean=0.55,
            )
            assert isinstance(pred, MCQDifficultyPrediction)
            assert pred.label in {"easy", "medium", "hard"}
            assert -1.0 <= pred.logit <= 1.0
            assert 0.0 <= pred.confidence <= 1.0
            assert set(pred.probabilities) == {"easy", "medium", "hard"}
            _ok(name)
        finally:
            settings.MCQ_DIFFICULTY_MODEL_PATH = prev_path
    except Exception as exc:
        _fail(name, str(exc))


# ---------------------------------------------------------------------------
# LangGraph node tests (skipped when langgraph is unavailable)
# ---------------------------------------------------------------------------


def test_langgraph_node_overrides_estimate() -> None:
    name = "11. LangGraph difficulty_estimator overrides gen_output"
    if not _HAS_LANGGRAPH or not _HAS_GEN_OUTPUT:
        _skip(name, "langgraph or GenerationOutput unavailable")
        return
    try:
        from unittest.mock import patch

        from app.services import langgraph_rag_workflow

        out = GenerationOutput(
            question_text="Question?",
            options=[{"text": "A", "is_correct": False}, {"text": "B", "is_correct": True}],
            explanation="x",
            difficulty_estimate=0.3,
        )
        state = {"gen_output": out}
        fake = MCQDifficultyPrediction(
            label="hard", logit=0.8, confidence=0.9,
            probabilities={"easy": 0.05, "medium": 0.1, "hard": 0.85},
        )
        with patch(
            "app.services.mcq_difficulty_service.estimate_generated_difficulty",
            return_value=fake,
        ):
            result = langgraph_rag_workflow.difficulty_estimator(state)
        assert out.difficulty_estimate == 0.8
        assert result["difficulty_model_report"]["label"] == "hard"
        assert result["difficulty_model_report"]["logit"] == 0.8
        _ok(name)
    except Exception as exc:
        _fail(name, str(exc))


def test_langgraph_node_fail_open() -> None:
    name = "12. LangGraph difficulty_estimator is best-effort on None"
    if not _HAS_LANGGRAPH or not _HAS_GEN_OUTPUT:
        _skip(name, "langgraph or GenerationOutput unavailable")
        return
    try:
        from unittest.mock import patch

        from app.services import langgraph_rag_workflow

        out = GenerationOutput(
            question_text="Question?",
            options=[{"text": "A", "is_correct": False}, {"text": "B", "is_correct": True}],
            explanation="x",
            difficulty_estimate=0.3,
        )
        state = {"gen_output": out}
        with patch(
            "app.services.mcq_difficulty_service.estimate_generated_difficulty",
            return_value=None,
        ):
            result = langgraph_rag_workflow.difficulty_estimator(state)
        assert out.difficulty_estimate == 0.3  # untouched
        assert "difficulty_model_report" not in result
        assert not result.get("failure_reason")
        _ok(name)
    except Exception as exc:
        _fail(name, str(exc))


def test_calibration_target_saturation() -> None:
    name = "14. calibration target saturates at reachable ceiling"
    try:
        from app.services.difficulty_calibration_service import (
            DifficultyCalibrationService,
        )

        service = DifficultyCalibrationService()

        # Ceiling student (theta 3.5): a hard item (logit 2.0) must pass.
        report = service.calibrate(target_theta=3.5, difficulty_estimate=2.0)
        assert report.aligned is True, f"hard item at ceiling should pass: {report}"
        assert report.effective_target_theta == 2.0

        # A mid-hard item (logit 0.6, hard band) passes at ceiling too.
        report = service.calibrate(target_theta=3.5, difficulty_estimate=0.6)
        assert report.aligned is True, f"mid-hard item at ceiling should pass: {report}"

        # A medium item is still rejected for a ceiling student.
        report = service.calibrate(target_theta=3.5, difficulty_estimate=0.05)
        assert report.aligned is False, f"medium item at ceiling should block: {report}"

        # Floor student (theta -3.5): easy item passes, medium blocks.
        report = service.calibrate(target_theta=-3.5, difficulty_estimate=-2.0)
        assert report.aligned is True, f"easy item at floor should pass: {report}"
        report = service.calibrate(target_theta=-3.5, difficulty_estimate=0.0)
        assert report.aligned is False, f"medium item at floor should block: {report}"

        # Mid-range thetas are unaffected by saturation.
        report = service.calibrate(target_theta=1.0, difficulty_estimate=-1.5)
        assert report.aligned is False
        assert report.effective_target_theta == 1.0

        _ok(name)
    except Exception as exc:
        _fail(name, str(exc))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all() -> None:
    tests = [
        test_label_maps_consistent,
        test_prompt_template_matches_report,
        test_prompt_defaults_when_features_absent,
        test_cosine_similarity,
        test_answer_similarity_mean,
        test_empty_text_rejected,
        test_bad_correct_index_rejected,
        test_missing_model_path_handled,
        test_estimate_disabled_returns_none,
        test_generation_prompt_difficulty_bands,
        test_calibration_target_saturation,
        test_bundled_model_ready_and_predicts,
        test_langgraph_node_overrides_estimate,
        test_langgraph_node_fail_open,
    ]
    for t in tests:
        t()

    passed = 0
    for name, ok, note in _RESULTS:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}" + (f" ({note})" if note else ""))
        if ok:
            passed += 1
    print(f"\n{passed}/{len(_RESULTS)} passed")
    if passed != len(_RESULTS):
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()
