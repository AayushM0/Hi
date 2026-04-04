"""Tests for embedding generation."""

import pytest
from lace.retrieval.embeddings import embed_text, embed_batch, cosine_similarity


def test_embed_text_returns_list_of_floats():
    result = embed_text("hello world")
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(v, float) for v in result)


def test_embed_text_consistent_dimensions():
    """Same model always returns same dimension."""
    r1 = embed_text("short text")
    r2 = embed_text("a much longer piece of text with many more words")
    assert len(r1) == len(r2)


def test_embed_batch_matches_individual():
    """Batch embedding should match individual embeddings."""
    texts = ["hello world", "foo bar"]
    batch = embed_batch(texts)
    individual = [embed_text(t) for t in texts]

    assert len(batch) == len(individual)
    for b, ind in zip(batch, individual):
        # Should be very close (floating point differences possible)
        sim = cosine_similarity(b, ind)
        assert sim > 0.9999


def test_cosine_similarity_identical_vectors():
    """Identical vectors should have similarity ~1.0."""
    vec = embed_text("test sentence")
    sim = cosine_similarity(vec, vec)
    assert abs(sim - 1.0) < 0.0001


def test_cosine_similarity_related_texts():
    """Semantically related texts should have high similarity."""
    v1 = embed_text("database connection pooling")
    v2 = embed_text("connection pool for database")
    sim = cosine_similarity(v1, v2)
    assert sim > 0.8


def test_cosine_similarity_unrelated_texts():
    """Semantically unrelated texts should have lower similarity."""
    v1 = embed_text("database connection pooling asyncpg postgresql")
    v2 = embed_text("baking bread sourdough recipe yeast flour")
    sim = cosine_similarity(v1, v2)
    assert sim < 0.6


def test_embed_batch_empty_returns_empty():
    result = embed_batch([])
    assert result == []