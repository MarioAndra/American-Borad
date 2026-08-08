"""Verification tests for the Bloom classifier integration.

Run:  python -m app.scripts.test_bloom_classifier

Pure tests (label mapping, empty-text rejection, missing-model handling) run
without torch/transformers and without a database. Database-backed API tests
are skipped with a note when the configured DATABASE_URL is unreachable.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

from app.models.enums import CognitiveLevel, DifficultyLevel, QuestionType
from app.schemas.question import QuestionCreate
from app.services import bloom_classifier_service
from app.services.bloom_classifier_service import (
    BloomClassifierError,
    BloomClassifierService,
    BloomPrediction,
    label_from_code,
)

_RESULTS: list[tuple[str, bool, str]] = []


def _ok(name: str) -> None:
    _RESULTS.append((name, True, ""))


def _fail(name: str, msg: str) -> None:
    _RESULTS.append((name, False, msg))


def _skip(name: str, msg: str) -> None:
    _RESULTS.append((name, True, f"SKIPPED: {msg}"))


def _db_available() -> bool:
    try:
        from app.db.session import SessionLocal

        db = SessionLocal()
        try:
            db.connection()
        finally:
            db.close()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Pure unit tests (no model, no DB)
# ---------------------------------------------------------------------------

def test_label_mapping() -> None:
    name = "1. label mapping"
    try:
        assert label_from_code("BT7") is CognitiveLevel.RememberUnderstand
        assert label_from_code("BT3") is CognitiveLevel.Apply
        assert label_from_code("BT4") is CognitiveLevel.Analyze
        assert label_from_code("BT5") is CognitiveLevel.Evaluate
        assert label_from_code("BT6") is CognitiveLevel.Create
        _ok(name)
    except Exception as exc:
        _fail(name, str(exc))


def test_empty_text_rejected() -> None:
    name = "2. empty text rejected"
    try:
        for text in ("", "   ", "\t\n"):
            try:
                bloom_classifier_service.predict(text)
                raise AssertionError(f"predict({text!r}) was not rejected")
            except ValueError:
                pass
        _ok(name)
    except Exception as exc:
        _fail(name, str(exc))


def test_unknown_code_rejected() -> None:
    name = "3. unknown label code rejected"
    try:
        try:
            label_from_code("BT0")
            raise AssertionError("BT0 was accepted")
        except ValueError:
            pass
        _ok(name)
    except Exception as exc:
        _fail(name, str(exc))


def test_missing_model_path_handled() -> None:
    name = "4. missing model path handled"
    try:
        service = BloomClassifierService(model_path="/nonexistent/bloom_model")
        try:
            service.predict("What is the function of the mitochondria?")
            raise AssertionError("missing model path did not raise")
        except BloomClassifierError:
            pass
        _ok(name)
    except Exception as exc:
        _fail(name, str(exc))


# ---------------------------------------------------------------------------
# Database-backed API tests (skipped when DB is unreachable)
# ---------------------------------------------------------------------------

@dataclass
class _Fixture:
    user_id: int
    subtopic_id: int
    topic_id: int
    phase_id: int


def _create_user(db, email: str, role) -> object:
    from app.core.security import get_password_hash
    from app.models import User, UserRole

    user = User(
        full_name="Bloom Test",
        email=email,
        hashed_password=get_password_hash("TestPass!123"),
        role=role,
        is_active=True,
        is_verified=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _create_hierarchy(db) -> tuple[int, int, int]:
    from app.models import Phase, SubTopic, Topic

    phase = Phase(name="__bloom_test_phase__")
    db.add(phase)
    db.flush()
    topic = Topic(phase_id=phase.id, name="__bloom_test_topic__")
    db.add(topic)
    db.flush()
    subtopic = SubTopic(topic_id=topic.id, name="__bloom_test_subtopic__")
    db.add(subtopic)
    db.flush()
    db.commit()
    return subtopic.id, topic.id, phase.id


def _enable_bloom_classifier() -> tuple[object, object]:
    """Enable the Bloom classifier for the test process.

    Returns the previous setting values so the caller can restore them.
    """
    from app.core.config import get_settings

    settings = get_settings()
    prev_enabled = settings.BLOOM_MODEL_ENABLED
    prev_path = settings.BLOOM_MODEL_PATH
    settings.BLOOM_MODEL_ENABLED = True
    settings.BLOOM_MODEL_PATH = "dummy"
    return prev_enabled, prev_path


def _restore_bloom_classifier(prev_enabled: object, prev_path: object) -> None:
    from app.core.config import get_settings

    settings = get_settings()
    settings.BLOOM_MODEL_ENABLED = prev_enabled
    settings.BLOOM_MODEL_PATH = prev_path


def _cleanup(db, question_ids: list[int], fixture: _Fixture) -> None:
    from app.models import Choice, Phase, Question, SubTopic, Topic, User

    for qid in question_ids:
        db.query(Choice).filter(Choice.question_id == qid).delete()
        db.query(Question).filter(Question.id == qid).delete()
    db.query(SubTopic).filter(SubTopic.id == fixture.subtopic_id).delete()
    db.query(Topic).filter(Topic.id == fixture.topic_id).delete()
    db.query(Phase).filter(Phase.id == fixture.phase_id).delete()
    db.query(User).filter(User.id == fixture.user_id).delete()
    db.commit()


def test_admin_can_create_and_store_prediction() -> None:
    name = "5. admin creates question, stores predicted cognitive_level"
    if not _db_available():
        _skip(name, "database unavailable")
        return

    from fastapi.testclient import TestClient
    from app.core.security import create_access_token
    from app.db.session import SessionLocal
    from app.main import app
    from app.models import Question, UserRole

    db = SessionLocal()
    question_ids: list[int] = []
    prev_enabled, prev_path = _enable_bloom_classifier()
    try:
        admin = _create_user(db, "bloom_admin_create@example.com", UserRole.Admin)
        subtopic_id, topic_id, phase_id = _create_hierarchy(db)
        fixture = _Fixture(admin.id, subtopic_id, topic_id, phase_id)
        token = create_access_token(subject=str(admin.id), role=admin.role.value)["access_token"]

        def _fake_predict(text: str) -> BloomPrediction:
            return BloomPrediction(
                label=CognitiveLevel.Analyze,
                label_code="BT4",
                confidence=0.9,
            )

        original = bloom_classifier_service.predict
        bloom_classifier_service.predict = _fake_predict
        client = TestClient(app)
        try:
            response = client.post(
                "/api/v1/questions",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "text": "Which TCP/IP layer guarantees reliable delivery?",
                    "difficulty": "Medium",
                    "question_type": "SingleChoice",
                    "subtopic_id": subtopic_id,
                    "choices": [
                        {"text": "Transport", "is_correct": True},
                        {"text": "Application", "is_correct": False},
                        {"text": "Physical", "is_correct": False},
                        {"text": "Session", "is_correct": False},
                    ],
                },
            )
            assert response.status_code == 201, response.text
            body = response.json()
            assert body["cognitive_level"] == "Analyze", body
            row = db.query(Question).filter(Question.id == body["id"]).first()
            assert row is not None and row.cognitive_level is CognitiveLevel.Analyze
            assert row.created_by == admin.id
            question_ids.append(body["id"])
            _ok(name)
        finally:
            bloom_classifier_service.predict = original
    except Exception as exc:
        _fail(name, str(exc))
    finally:
        _restore_bloom_classifier(prev_enabled, prev_path)
        _cleanup(db, question_ids, fixture)


def test_service_layer_persists_prediction() -> None:
    name = "6. service layer persists classifier output"
    if not _db_available():
        _skip(name, "database unavailable")
        return

    from app.db.session import SessionLocal
    from app.models import UserRole
    from app.services import question_service

    db = SessionLocal()
    question_ids: list[int] = []
    prev_enabled, prev_path = _enable_bloom_classifier()
    try:
        admin = _create_user(db, "bloom_admin_service@example.com", UserRole.Admin)
        subtopic_id, topic_id, phase_id = _create_hierarchy(db)
        fixture = _Fixture(admin.id, subtopic_id, topic_id, phase_id)

        def _fake_predict(text: str) -> BloomPrediction:
            return BloomPrediction(
                label=CognitiveLevel.Create,
                label_code="BT6",
                confidence=0.95,
            )

        original = bloom_classifier_service.predict
        bloom_classifier_service.predict = _fake_predict
        try:
            payload = QuestionCreate(
                text="Design an algorithm to detect anomalous credit card transactions.",
                difficulty=DifficultyLevel.Hard,
                question_type=QuestionType.SingleChoice,
                subtopic_id=subtopic_id,
                choices=[
                    {"text": "Train a classifier", "is_correct": True},
                    {"text": "Store logs", "is_correct": False},
                    {"text": "Email users", "is_correct": False},
                    {"text": "Print reports", "is_correct": False},
                ],
            )
            question = question_service.create_question(db, payload, created_by=admin)
            assert question.cognitive_level is CognitiveLevel.Create
            question_ids.append(question.id)
            _ok(name)
        finally:
            bloom_classifier_service.predict = original
    except Exception as exc:
        _fail(name, str(exc))
    finally:
        _restore_bloom_classifier(prev_enabled, prev_path)
        _cleanup(db, question_ids, fixture)


def test_non_admin_cannot_create() -> None:
    name = "7. non-admin cannot create a question"
    if not _db_available():
        _skip(name, "database unavailable")
        return

    from fastapi.testclient import TestClient
    from app.core.dependencies import require_roles
    from app.core.security import create_access_token
    from app.db.session import SessionLocal
    from app.main import app
    from app.models import UserRole

    db = SessionLocal()
    try:
        student = _create_user(db, "bloom_student@example.com", UserRole.Student)
        subtopic_id, topic_id, phase_id = _create_hierarchy(db)
        fixture = _Fixture(student.id, subtopic_id, topic_id, phase_id)

        guard = require_roles("Admin")
        try:
            guard(user=student)
            raise AssertionError("student passed admin guard")
        except Exception:
            pass

        token = create_access_token(subject=str(student.id), role=student.role.value)["access_token"]
        client = TestClient(app)
        response = client.post(
            "/api/v1/questions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "text": "Non-admin attempt question?",
                "difficulty": "Easy",
                "question_type": "SingleChoice",
                "subtopic_id": subtopic_id,
                "choices": [
                    {"text": "A", "is_correct": True},
                    {"text": "B", "is_correct": False},
                    {"text": "C", "is_correct": False},
                    {"text": "D", "is_correct": False},
                ],
            },
        )
        assert response.status_code == 403, response.text
        _ok(name)
    except Exception as exc:
        _fail(name, str(exc))
    finally:
        _cleanup(db, [], fixture)


_ALL_TESTS = [
    test_label_mapping,
    test_empty_text_rejected,
    test_unknown_code_rejected,
    test_missing_model_path_handled,
    test_admin_can_create_and_store_prediction,
    test_service_layer_persists_prediction,
    test_non_admin_cannot_create,
]


def main() -> int:
    for fn in _ALL_TESTS:
        fn()

    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    failed = sum(1 for _, ok, _ in _RESULTS if not ok)

    print()
    for name, ok, msg in _RESULTS:
        tag = "PASS" if ok else "FAIL"
        line = f"  [{tag}] {name}"
        if msg:
            line += f"  --  {msg}"
        print(line)

    print()
    print("-" * 60)
    print(f"  {passed} passed, {failed} failed, {len(_RESULTS)} total")
    print("-" * 60)

    if failed:
        print("\nSOME TESTS FAILED")
        return 1

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
