from __future__ import annotations

"""Run inside the app container: docker compose exec app python -m app.scripts.clear_rag_data"""

from app.db.session import SessionLocal
from app.models import KnowledgeChunk, KnowledgeDocument, GeneratedQuestion, GeneratedQuestionEvidence, GeneratedQuestionReview, StudentTopicProgress
from app.core.logging import get_logger

log = get_logger(__name__)

TABLES = [
    GeneratedQuestionReview,
    GeneratedQuestionEvidence,
    GeneratedQuestion,
    StudentTopicProgress,
    KnowledgeChunk,
    KnowledgeDocument,
]


def main() -> None:
    db = SessionLocal()
    try:
        for table in TABLES:
            count = db.query(table).delete()
            log.info("Cleared %d rows from %s", count, table.__tablename__)
        db.commit()
        log.info("Done. PG knowledge tables cleared.")
    except Exception as e:
        db.rollback()
        log.error("Failed to clear: %s", e)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
