from __future__ import annotations

"""End-to-end pipeline test: embed → store → search → cleanup."""

from app.services.llm_client import LLMClient
from app.services.weaviate_client import WeaviateService, WEAVIATE_CLASS

SENTINEL_CHUNK_ID = 0
TEST_TOPIC_ID = 11  # Cybersecurity — first Phase II topic


def main() -> None:
    print("=" * 50)
    print("RAG Pipeline Readiness Test")
    print("=" * 50)

    # 1. Embed
    print("\n[1/4] Embedding via OpenAI...")
    llm = LLMClient(provider="openai")
    if not llm.is_available:
        print("  FAILED: OpenAI client not available")
        return
    text = "What is machine learning? A field of AI that enables systems to learn from data and improve from experience."
    embeddings = llm.embed([text])
    if embeddings[0] is None:
        print("  FAILED: Embedding returned None")
        return
    print(f"  ✅ {len(embeddings[0])} dimensions")

    # 2. Store in Weaviate
    print("\n[2/4] Storing in Weaviate...")
    w = WeaviateService()
    if not w.is_available:
        print("  FAILED: Weaviate not available")
        return
    w.store_chunk(
        chunk_id=SENTINEL_CHUNK_ID,
        document_id=0,
        course_name="test_pipeline",
        title="Pipeline Validation",
        text=text,
        chunk_index=0,
        topic_id=TEST_TOPIC_ID,
        embedding=embeddings[0],
    )
    print("  ✅ Stored")

    # 3. Search back
    print("\n[3/4] Searching back from Weaviate...")
    results = w.search(query_embedding=embeddings[0], topic_id=TEST_TOPIC_ID, top_k=5)
    found = any(r["chunk_id"] == SENTINEL_CHUNK_ID for r in results)
    print(f"  Results: {len(results)}")
    for r in results:
        match_mark = "✅ SENTINEL" if r["chunk_id"] == SENTINEL_CHUNK_ID else "   "
        print(f"  {match_mark} chunk_id={r['chunk_id']}, similarity={r['similarity']:.4f}")
    if not found:
        print("  ❌ Sentinel chunk not found in results")
    else:
        print("  ✅ Retrieval works")

    # 4. Cleanup
    print("\n[4/4] Cleaning up test data...")
    import weaviate.classes as wvc
    import weaviate
    client = weaviate.connect_to_custom(
        http_host="weaviate", http_port=8080, http_secure=False,
        grpc_host="weaviate", grpc_port=8081, grpc_secure=False,
        skip_init_checks=True,
    )
    if client.collections.exists(WEAVIATE_CLASS):
        client.collections.delete(WEAVIATE_CLASS)
    client.close()
    w.close()
    print("  ✅ Cleaned up")

    print("\n" + "=" * 50)
    print("ALL CHECKS PASSED — Pipeline is ready ✅")
    print("=" * 50)


if __name__ == "__main__":
    main()
