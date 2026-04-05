"""Tests for memory extraction pipeline."""

import pytest
from lace.memory.extractor import (
    _parse_extraction_response,
    should_attempt_extraction,
    ExtractionCandidate,
)


def test_parse_valid_json_array():
    raw = """
    Here are the extractions:
    [
      {
        "content": "Always use connection pooling with asyncpg",
        "category": "pattern",
        "tags": ["asyncpg", "db"],
        "confidence": 0.9,
        "reasoning": "Specific actionable pattern"
      }
    ]
    """
    candidates = _parse_extraction_response(raw)
    assert len(candidates) == 1
    assert candidates[0].content == "Always use connection pooling with asyncpg"
    assert candidates[0].category == "pattern"
    assert "asyncpg" in candidates[0].tags
    assert candidates[0].confidence == 0.9


def test_parse_empty_array():
    raw = "[]"
    candidates = _parse_extraction_response(raw)
    assert candidates == []


def test_parse_empty_response():
    candidates = _parse_extraction_response("")
    assert candidates == []


def test_parse_no_json():
    raw = "Nothing useful was extracted from this conversation."
    candidates = _parse_extraction_response(raw)
    assert candidates == []


def test_parse_caps_at_max_extractions():
    raw = """[
      {"content": "Memory 1 with enough content to pass", "category": "pattern", "tags": [], "confidence": 0.8, "reasoning": "r"},
      {"content": "Memory 2 with enough content to pass", "category": "pattern", "tags": [], "confidence": 0.8, "reasoning": "r"},
      {"content": "Memory 3 with enough content to pass", "category": "pattern", "tags": [], "confidence": 0.8, "reasoning": "r"},
      {"content": "Memory 4 with enough content to pass", "category": "pattern", "tags": [], "confidence": 0.8, "reasoning": "r"}
    ]"""
    candidates = _parse_extraction_response(raw, max_extractions=2)
    assert len(candidates) == 2


def test_parse_invalid_category_defaults_to_pattern():
    raw = """[
      {"content": "Valid content here that is long enough",
       "category": "invalid_category",
       "tags": [],
       "confidence": 0.8,
       "reasoning": "test"}
    ]"""
    candidates = _parse_extraction_response(raw)
    assert len(candidates) == 1
    assert candidates[0].category == "pattern"


def test_parse_clips_confidence():
    raw = """[
      {"content": "Valid content that is long enough to pass filter",
       "category": "pattern",
       "tags": [],
       "confidence": 1.5,
       "reasoning": "test"}
    ]"""
    candidates = _parse_extraction_response(raw)
    assert len(candidates) == 1
    assert candidates[0].confidence <= 1.0


def test_parse_skips_too_short_content():
    raw = """[
      {"content": "short", "category": "pattern", "tags": [], "confidence": 0.8, "reasoning": "r"}
    ]"""
    candidates = _parse_extraction_response(raw)
    assert len(candidates) == 0


def test_should_attempt_extraction_normal():
    query    = "How do I set up asyncpg connection pooling?"
    response = "Use asyncpg.create_pool() with max_size set to 2x your worker count. " * 5
    assert should_attempt_extraction(query, response) is True


def test_should_attempt_extraction_too_short():
    assert should_attempt_extraction("hi", "Hello!") is False


def test_should_attempt_extraction_greeting():
    assert should_attempt_extraction("hello", "Hi there! How can I help?") is False


def test_should_attempt_extraction_error_response():
    query    = "help"
    response = "[Error: connection refused to Ollama]"
    assert should_attempt_extraction(query, response) is False