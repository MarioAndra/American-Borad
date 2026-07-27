from __future__ import annotations

from app.core.logging import get_logger
from app.services.langgraph_rag_state import RAGGraphState

log = get_logger(__name__)


def build_langgraph_trace(
    state: RAGGraphState,
    settings: object,  # noqa: ANN401 — settings duck-type
) -> dict:
    """Build a structured trace dict from the final LangGraph state.

    Called once at persist time.  The returned dict is stored inside
    ``validation_report["langgraph_trace"]`` so it is automatically
    visible through the admin ``GET /rag/questions`` API without any
    schema migration.

    The trace captures:
    - ``trace_id`` — unique id for correlating logs
    - ``retry_count`` — retrieval-repair attempts
    - ``evidence`` — chunk count, average similarity, high-quality count
    - ``confidence`` — raw/effective route, score, reasons
    - ``validation`` — schema validity, judge participation, any issues
    - ``failure_code`` — stable failure code if the graph aborted
    - ``repair`` — repair attempts and routing info
    """
    trace_id: str | None = state.get("trace_id")  # type: ignore[assignment]
    chunks = state.get("retrieved_chunks") or []
    cr = state.get("confidence_report")
    vr = state.get("validation_report")

    trace: dict = {
        "trace_id": trace_id,
        "retry_count": state.get("retry_count", 0),
    }

    # --- Failure code (only if graph did not persist) ---
    failure_code = state.get("failure_code")
    failure_reason = state.get("failure_reason")
    if failure_code:
        trace["failure_code"] = failure_code
        trace["failure_reason"] = failure_reason

    # --- Repair metadata ---
    repair_attempts = state.get("repair_attempt_count", 0)
    trace["repair"] = {
        "attempt_count": repair_attempts,
        "routing": state.get("repair_report").failure_type if state.get("repair_report") else None,
    }

    # --- Evidence quality ---
    if chunks:
        avg_sim = sum((c.similarity or 0.0) for c in chunks) / len(chunks)
        trace["evidence"] = {
            "chunk_count": len(chunks),
            "avg_similarity": round(avg_sim, 4),
            "high_quality_count": sum(
                1 for c in chunks if (c.similarity or 0.0) >= 0.5
            ),
        }

    # --- Confidence routing ---
    if cr:
        raw_route = cr.route
        if raw_route == "human_review" or getattr(settings, "RAG_REVIEW_REQUIRED", False):
            effective_route = "human_review"
        else:
            effective_route = "auto_approve"
        trace["confidence"] = {
            "route_raw": raw_route,
            "route_effective": effective_route,
            "score": cr.score,
            "reasons": list(cr.reasons),
        }

    # --- Validation ---
    if vr:
        trace["validation"] = {
            "schema_ok": vr.schema_ok,
            "single_correct": vr.single_correct,
            "judge_participated": vr.judge_ok is not None,
            "judge_ok": vr.judge_ok,
            "issues": list(vr.issues),
        }

    log.debug("LangGraph trace built: trace_id=%s", trace_id)
    return trace
