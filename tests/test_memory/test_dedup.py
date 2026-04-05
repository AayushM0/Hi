"""Tests for deduplication logic."""

import pytest
from lace.memory.models import make_memory, MemoryCategory
from lace.memory.dedup import (
    DedupAction,
    check_duplicate,
    merge_memories,
    cosine_similarity,
)
from lace.retrieval.embeddings import embed_text


def _embedded(content: str, **kwargs):
    m = make_memory(content, **kwargs)
    m.embedding = embed_text(content)
    return m


def test_cosine_similarity_identical():
    v = [1.0, 0.0, 0.0]
    assert abs(cosine_similarity(v, v) - 1.0) < 0.001


def test_cosine_similarity_orthogonal():
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    assert abs(cosine_similarity(v1, v2)) < 0.001


def test_check_duplicate_no_existing():
    candidate = _embedded("Use asyncpg for PostgreSQL connections")
    result = check_duplicate(candidate, [])
    assert result.action == DedupAction.STORE


def test_check_duplicate_novel_content():
    existing  = _embedded("Use Redis for caching")
    candidate = _embedded("Always use type hints in Python")
    result = check_duplicate(candidate, [existing])
    assert result.action == DedupAction.STORE


def test_check_duplicate_identical_content():
    content   = "Always use connection pooling with asyncpg"
    existing  = _embedded(content)
    candidate = _embedded(content)
    result = check_duplicate(candidate, [existing])
    assert result.action == DedupAction.SKIP
    assert result.similarity > 0.95


def test_check_duplicate_very_similar():
    existing  = _embedded(
        "Always use connection pooling with asyncpg. Max pool = 2x workers."
    )
    candidate = _embedded(
        "Use connection pooling with asyncpg. Pool size should be 2x worker count."
    )
    result = check_duplicate(candidate, [existing])
    # Should be SKIP or MERGE — not STORE
    assert result.action in (DedupAction.SKIP, DedupAction.MERGE)


def test_check_duplicate_different_category_not_compared():
    """Memories of different categories are not compared."""
    existing  = _embedded(
        "Always use connection pooling",
        category=MemoryCategory.PATTERN,
    )
    candidate = _embedded(
        "Always use connection pooling",
        category=MemoryCategory.DECISION,
    )
    result = check_duplicate(candidate, [existing])
    # Different categories → stored as separate memories
    assert result.action == DedupAction.STORE


def test_merge_memories_combines_tags():
    existing  = _embedded("Use asyncpg for db", category=MemoryCategory.PATTERN)
    existing.tags = ["db", "asyncpg"]

    candidate = _embedded("Use asyncpg for db", category=MemoryCategory.PATTERN)
    candidate.tags = ["asyncpg", "performance"]

    merged = merge_memories(existing, candidate)
    assert "db" in merged.tags
    assert "asyncpg" in merged.tags
    assert "performance" in merged.tags


def test_merge_memories_boosts_confidence():
    existing           = _embedded("Use asyncpg")
    existing.confidence = 0.8
    candidate          = _embedded("Use asyncpg for PostgreSQL")

    merged = merge_memories(existing, candidate)
    assert merged.confidence > 0.8


def test_check_duplicate_no_embedding():
    candidate = make_memory("No embedding here")
    existing  = _embedded("Something else entirely")
    result = check_duplicate(candidate, [existing])
    assert result.action == DedupAction.STORE
    assert "no embedding" in result.reason