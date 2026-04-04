"""Tests for multi-signal ranking."""

import pytest
from datetime import datetime, timezone, timedelta
from lace.memory.models import make_memory
from lace.retrieval.ranking import (
    RankingWeights,
    semantic_score,
    recency_score,
    frequency_score,
    scope_score,
    compute_relevance_score,
    rank_candidates,
)


def test_semantic_score_zero_distance():
    """Distance 0 = identical = score 1.0."""
    assert semantic_score(0.0) == 1.0


def test_semantic_score_max_distance():
    """Distance 2.0 = opposite = score 0.0."""
    assert semantic_score(2.0) == 0.0


def test_semantic_score_midpoint():
    """Distance 1.0 = score 0.5."""
    assert abs(semantic_score(1.0) - 0.5) < 0.001


def test_recency_score_today():
    """Accessed today = score ~1.0."""
    now = datetime.now(timezone.utc)
    score = recency_score(now)
    assert score > 0.99


def test_recency_score_half_life():
    """Accessed exactly half_life days ago = score ~0.5."""
    half_life = 30
    past = datetime.now(timezone.utc) - timedelta(days=half_life)
    score = recency_score(past, half_life_days=half_life)
    assert abs(score - 0.5) < 0.01


def test_recency_score_old_memory():
    """Very old memory should have low recency score."""
    old = datetime.now(timezone.utc) - timedelta(days=365)
    score = recency_score(old)
    assert score < 0.05


def test_frequency_score_zero_accesses():
    assert frequency_score(0) == 0.0


def test_frequency_score_increases_with_count():
    s1 = frequency_score(1)
    s10 = frequency_score(10)
    s100 = frequency_score(100)
    assert s1 < s10 < s100


def test_scope_score_exact_match():
    assert scope_score("project:my-api", "project:my-api") == 1.0


def test_scope_score_global_memory_in_project():
    score = scope_score("global", "project:my-api")
    assert score == 0.5


def test_scope_score_different_project():
    score = scope_score("project:other", "project:my-api")
    assert score == 0.2


def test_ranking_weights_sum_to_one():
    weights = RankingWeights()
    weights.validate()  # Should not raise


def test_rank_candidates_filters_by_threshold():
    m1 = make_memory("Highly relevant memory")
    m2 = make_memory("Less relevant memory")

    # m1 has distance 0.1 (very similar), m2 has distance 1.5 (not similar)
    results = rank_candidates(
        candidates=[(m1, 0.1), (m2, 1.5)],
        active_scope="global",
        threshold=0.6,
    )

    # Only m1 should pass the threshold
    assert len(results) == 1
    assert results[0].memory.id == m1.id


def test_rank_candidates_sorted_by_score():
    m1 = make_memory("Very relevant")
    m2 = make_memory("Less relevant")

    results = rank_candidates(
        candidates=[(m1, 0.05), (m2, 0.8)],
        active_scope="global",
        threshold=0.0,
    )

    assert len(results) == 2
    assert results[0].memory.id == m1.id
    assert results[0].rank == 1
    assert results[1].rank == 2


def test_rank_candidates_assigns_ranks():
    memories = [make_memory(f"Memory {i}") for i in range(5)]
    candidates = [(m, 0.3) for m in memories]

    results = rank_candidates(candidates, threshold=0.0)
    ranks = [r.rank for r in results]
    assert ranks == list(range(1, len(results) + 1))