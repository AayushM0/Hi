"""Tests for MemoryObject model."""

import pytest
from datetime import datetime, timezone
from lace.memory.models import (
    MemoryCategory,
    MemoryLifecycle,
    MemoryObject,
    MemorySource,
    make_memory,
)


def test_memory_creation_defaults():
    m = make_memory("Test content")
    assert m.content == "Test content"
    assert m.category == MemoryCategory.PATTERN
    assert m.source == MemorySource.MANUAL
    assert m.lifecycle == MemoryLifecycle.CAPTURED
    assert m.confidence == 0.8
    assert m.project_scope == "global"
    assert m.id.startswith("mem_")
    assert m.access_count == 0


def test_memory_id_is_unique():
    m1 = make_memory("Content A")
    m2 = make_memory("Content B")
    assert m1.id != m2.id


def test_memory_empty_content_raises():
    with pytest.raises(ValueError, match="content cannot be empty"):
        MemoryObject(content="   ", category=MemoryCategory.PATTERN)


def test_memory_invalid_confidence_raises():
    with pytest.raises(ValueError, match="confidence"):
        MemoryObject(content="test", category=MemoryCategory.PATTERN, confidence=1.5)


def test_memory_touch_updates_access():
    m = make_memory("Test")
    original_count = m.access_count
    m.touch()
    assert m.access_count == original_count + 1


def test_memory_archive():
    m = make_memory("Test")
    assert m.is_active()
    m.archive()
    assert not m.is_active()
    assert m.lifecycle == MemoryLifecycle.ARCHIVED


def test_memory_validate_promotes_lifecycle():
    m = make_memory("Test")
    original_confidence = m.confidence
    m.validate()
    assert m.lifecycle == MemoryLifecycle.VALIDATED
    assert m.confidence > original_confidence


def test_display_summary_uses_summary_field():
    m = make_memory("Long content here")
    m.summary = "Short summary"
    assert m.display_summary() == "Short summary"


def test_display_summary_truncates_long_content():
    m = make_memory("x" * 100)
    assert m.display_summary().endswith("...")
    assert len(m.display_summary()) <= 83


def test_make_memory_with_tags():
    m = make_memory("Test", tags=["pattern", "db"])
    assert "pattern" in m.tags
    assert "db" in m.tags