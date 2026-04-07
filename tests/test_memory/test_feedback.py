"""Tests for memory rating and feedback loop."""

import pytest
from pathlib import Path
from lace.core.config import LaceConfig, init_lace_home
from lace.memory.store import MemoryStore
from lace.memory.models import MemoryLifecycle


@pytest.fixture
def store(tmp_path):
    lace_home = tmp_path / ".lace"
    init_lace_home(lace_home)
    config = LaceConfig()
    return MemoryStore(lace_home=lace_home, config=config)


def test_rate_helpful_boosts_confidence(store):
    memory = store.add("Test pattern")
    store.rate(memory.id, "helpful")

    updated = store.get(memory.id)
    assert updated.confidence == pytest.approx(0.85)
    assert updated.access_count == 1


def test_rate_outdated_halves_confidence(store):
    memory = store.add("Old pattern")
    store.rate(memory.id, "outdated")

    updated = store.get(memory.id)
    assert updated.confidence == pytest.approx(0.4)
    assert updated.metadata.get("flagged") == "outdated"


def test_rate_wrong_drops_confidence_heavily(store):
    memory = store.add("Incorrect info")
    store.rate(memory.id, "wrong")

    updated = store.get(memory.id)
    assert updated.confidence <= 0.16
    assert updated.metadata.get("flagged") == "incorrect"


def test_rate_unknown_signal_returns_false(store):
    memory = store.add("Test")
    result = store.rate(memory.id, "invalid")
    assert result is False


def test_rate_missing_memory_returns_false(store):
    result = store.rate("mem_nonexistent", "helpful")
    assert result is False


def test_rate_caps_confidence_at_1_0(store):
    memory = store.add("High confidence", confidence=0.98)
    store.rate(memory.id, "helpful")
    updated = store.get(memory.id)
    assert updated.confidence == pytest.approx(1.0)


def test_rate_floor_confidence_at_0_05(store):
    memory = store.add("Very wrong", confidence=0.1)
    store.rate(memory.id, "wrong")
    updated = store.get(memory.id)
    assert updated.confidence >= 0.05


def test_get_review_candidates_returns_low_confidence(store):
    store.add("Good memory", confidence=0.9)
    store.add("Shaky memory", confidence=0.5)

    candidates = store.get_review_candidates(min_confidence=0.7, include_zero_access=False)
    ids = {m.id for m in candidates}
    assert len(candidates) == 1
    assert any("Shaky" in m.content for m in candidates)


def test_get_review_candidates_excludes_archived(store):
    m = store.add("Archived", confidence=0.1)
    store.forget(m.id)

    candidates = store.get_review_candidates(min_confidence=0.5, include_zero_access=False)
    assert len(candidates) == 0


def test_get_review_candidates_includes_never_accessed(store):
    m1 = store.add("Accessed", confidence=0.9)
    m1.access_count = 5
    store.save(m1)
    
    store.add("Never accessed", confidence=0.9)

    candidates = store.get_review_candidates(min_confidence=0.9, include_zero_access=True)
    assert len(candidates) == 1
    assert "Never accessed" in candidates[0].content
