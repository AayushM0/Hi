"""Tests for ask engine."""

import pytest
from lace.utils.ask import build_system_prompt, build_user_message
from lace.memory.models import make_memory, RetrievalResult


def _make_result(content: str, score: float = 0.8) -> RetrievalResult:
    return RetrievalResult(
        memory=make_memory(content),
        relevance_score=score,
        match_type="vector",
        rank=1,
    )


def test_build_system_prompt_no_memories():
    prompt = build_system_prompt(
        identity="You are a helpful assistant.",
        preferences={},
        memories=[],
        scope="global",
    )
    assert "helpful assistant" in prompt
    assert "Memory" not in prompt


def test_build_system_prompt_with_memories():
    memories = [_make_result("Use asyncpg for PostgreSQL")]
    prompt = build_system_prompt(
        identity="You are a helpful assistant.",
        preferences={},
        memories=memories,
        scope="global",
    )
    assert "asyncpg" in prompt
    assert "Relevant Context from Memory" in prompt


def test_build_system_prompt_with_preferences():
    preferences = {"coding": {"language": "python", "style": "functional"}}
    prompt = build_system_prompt(
        identity="You are a helpful assistant.",
        preferences=preferences,
        memories=[],
        scope="global",
    )
    assert "python" in prompt
    assert "functional" in prompt


def test_build_user_message_no_memories():
    msg = build_user_message("How do I do X?", [])
    assert msg == "How do I do X?"


def test_build_user_message_with_memories():
    memories = [_make_result("Some memory")]
    msg = build_user_message("How do I do X?", memories)
    assert "How do I do X?" in msg
    assert "1 relevant memories" in msg


def test_build_system_prompt_with_project_scope():
    prompt = build_system_prompt(
        identity="You are a helpful assistant.",
        preferences={},
        memories=[],
        scope="project:my-api",
    )
    assert "project:my-api" in prompt