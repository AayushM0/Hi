"""Tests for ChromaDB vector store operations."""

import pytest
from pathlib import Path
from lace.memory.models import make_memory, MemoryCategory
from lace.retrieval.embeddings import embed_text
from lace.retrieval.vector import (
    upsert_memory,
    vector_search,
    delete_from_vector_store,
    get_collection_stats,
    _scope_to_collection_name,
)


@pytest.fixture
def vector_db(tmp_path):
    return tmp_path / "vector_db"


def _make_embedded_memory(content: str, **kwargs):
    memory = make_memory(content, **kwargs)
    memory.embedding = embed_text(content)
    return memory


def test_collection_name_global():
    assert _scope_to_collection_name("global") == "lace-global"


def test_collection_name_project():
    name = _scope_to_collection_name("project:my-api")
    assert name == "lace-project-my-api"
    assert len(name) <= 63


def test_upsert_and_search(vector_db):
    memory = _make_embedded_memory(
        "Use connection pooling with asyncpg",
        category=MemoryCategory.PATTERN,
    )
    upsert_memory(memory, vector_db)

    query_embedding = embed_text("database connection pool")
    results = vector_search(query_embedding, "global", vector_db)

    assert len(results) >= 1
    assert results[0]["id"] == memory.id


def test_upsert_no_embedding_is_noop(vector_db):
    """Memory without embedding should not be stored."""
    memory = make_memory("No embedding here")
    assert memory.embedding is None
    upsert_memory(memory, vector_db)  # Should not raise

    stats = get_collection_stats("global", vector_db)
    assert stats["count"] == 0


def test_delete_from_vector_store(vector_db):
    memory = _make_embedded_memory("Temporary memory to delete")
    upsert_memory(memory, vector_db)

    stats_before = get_collection_stats("global", vector_db)
    assert stats_before["count"] == 1

    delete_from_vector_store(memory.id, "global", vector_db)

    stats_after = get_collection_stats("global", vector_db)
    assert stats_after["count"] == 0


def test_empty_collection_returns_empty_search(vector_db):
    query_embedding = embed_text("anything")
    results = vector_search(query_embedding, "global", vector_db)
    assert results == []


def test_multiple_memories_ranked_by_similarity(vector_db):
    m1 = _make_embedded_memory("PostgreSQL connection pooling with asyncpg")
    m2 = _make_embedded_memory("Baking sourdough bread at home")
    m3 = _make_embedded_memory("Database query optimization techniques")

    for m in [m1, m2, m3]:
        upsert_memory(m, vector_db)

    query_embedding = embed_text("database connection")
    results = vector_search(query_embedding, "global", vector_db, n_results=3)

    ids = [r["id"] for r in results]
    # Database-related memories should rank before bread
    assert m2.id != ids[0]