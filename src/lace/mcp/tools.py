"""MCP tool definitions for LACE.

These are the functions that Antigravity/Cursor/Claude Desktop
will call autonomously when they need memory operations.

Each tool has:
  - A clear docstring (becomes the tool description the LLM sees)
  - Typed parameters (becomes the tool schema)
  - A focused return value (JSON-serializable)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from lace.core.config import get_lace_home, load_config
from lace.core.identity import compose_identity
from lace.core.scope import detect_current_project, get_active_scope
from lace.memory.store import MemoryStore


# ── Store factory ─────────────────────────────────────────────────────────────

def _get_store(scope: str | None = None) -> tuple[MemoryStore, str]:
    """Return a configured MemoryStore and resolved scope."""
    lace_home = get_lace_home()
    config = load_config(lace_home)
    store = MemoryStore(lace_home=lace_home, config=config)

    if scope is None or scope == "auto":
        resolved_scope = get_active_scope(lace_home)
    else:
        resolved_scope = scope

    return store, resolved_scope


# ── Tool implementations ──────────────────────────────────────────────────────

async def search_memory(
    query: str,
    scope: str = "auto",
    max_results: int = 5,
    category: str = "all",
) -> list[dict]:
    """Search your knowledge base for memories relevant to a query.

    Use this when the user asks about something they might have
    encountered, decided, or learned before. Returns memories ranked
    by relevance with confidence scores.

    Args:
        query: What to search for (natural language).
        scope: "auto" uses active project, or specify "global" / "project:<name>".
        max_results: How many memories to return (default 5, max 20).
        category: Filter by type — "all", "pattern", "decision", "debug",
                  "reference", "preference".

    Returns:
        List of memory objects with relevance scores.
    """
    store, resolved_scope = _get_store(scope)

    results = store.search(
        query=query,
        scope=resolved_scope,
        max_results=min(max_results, 20),
    )

    # Filter by category if specified
    if category != "all":
        results = [r for r in results if r.memory.category.value == category]

    # Serialize to JSON-safe dicts
    output = []
    for result in results:
        m = result.memory

        # Update access tracking
        m.touch()
        store.save(m)

        output.append({
            "id":             m.id,
            "content":        m.content,
            "summary":        m.display_summary(),
            "confidence":     m.confidence,
            "category":       m.category.value,
            "source":         m.source.value,
            "scope":          m.project_scope,
            "relevance_score": result.relevance_score,
            "match_type":     result.match_type,
            "last_accessed":  m.last_accessed.isoformat(),
            "access_count":   m.access_count,
            "tags":           m.tags,
        })

    return output


async def get_project_context() -> dict:
    """Get the current project's identity, preferences, rules, and conventions.

    Use this at the start of every conversation to understand:
    - Who you are and how to behave
    - The user's coding preferences (language, style, tools)
    - Project-specific rules and conventions
    - What the current project is about

    Returns:
        Dict with identity string, preferences, and project metadata.
    """
    lace_home = get_lace_home()
    resolved_scope = get_active_scope(lace_home)

    identity, preferences = compose_identity(lace_home, scope=resolved_scope)

    # Get project name
    project_name = None
    if resolved_scope.startswith("project:"):
        project_name = resolved_scope.removeprefix("project:")

    return {
        "project_name":  project_name,
        "scope":         resolved_scope,
        "identity":      identity,
        "preferences":   preferences,
    }


async def remember(
    content: str,
    category: str = "pattern",
    tags: list[str] | None = None,
    scope: str = "auto",
) -> dict:
    """Store a new piece of knowledge for future retrieval.

    Use this when the user discovers something worth remembering:
    - A coding pattern or best practice
    - An architectural decision and its rationale
    - A debugging insight or root cause
    - A preference or convention they want to enforce

    Don't store:
    - Trivial observations
    - Information already in the knowledge base
    - Temporary or session-specific context

    Args:
        content: The knowledge to store (be specific and actionable).
        category: Type of knowledge — "pattern", "decision", "debug",
                  "reference", "preference".
        tags: Optional tags for filtering (e.g. ["fastapi", "performance"]).
        scope: "auto" uses active project scope.

    Returns:
        The created memory object with its ID.
    """
    store, resolved_scope = _get_store(scope)

    memory = store.add(
        content=content,
        category=category,
        tags=tags or [],
        scope=resolved_scope,
        source="conversation",
        confidence=0.8,
    )

    return {
        "id":       memory.id,
        "content":  memory.content,
        "summary":  memory.display_summary(),
        "category": memory.category.value,
        "scope":    memory.project_scope,
        "tags":     memory.tags,
        "stored":   True,
    }


async def list_memories(
    category: str = "all",
    scope: str = "auto",
    limit: int = 20,
    lifecycle: str = "all",
) -> list[dict]:
    """List stored memories with optional filtering.

    Use when the user wants to browse or review what the system remembers.
    For searching by content, use search_memory instead.

    Args:
        category: Filter by type — "all", "pattern", "decision", "debug",
                  "reference", "preference".
        scope: "auto" uses active project scope.
        limit: Maximum number of memories to return.
        lifecycle: Filter by lifecycle — "all", "captured", "validated",
                   "consolidated", "archived".

    Returns:
        List of memory summaries.
    """
    store, resolved_scope = _get_store(scope)

    cat = None if category == "all" else category
    lc = None if lifecycle == "all" else lifecycle
    sc = resolved_scope if resolved_scope != "global" else None

    memories = store.list(
        category=cat,
        scope=sc,
        lifecycle=lc,
        include_archived=(lifecycle == "archived"),
        limit=limit,
    )

    return [
        {
            "id":           m.id,
            "summary":      m.display_summary(),
            "category":     m.category.value,
            "scope":        m.project_scope,
            "confidence":   m.confidence,
            "lifecycle":    m.lifecycle.value,
            "tags":         m.tags,
            "access_count": m.access_count,
            "last_accessed": m.last_accessed.isoformat(),
        }
        for m in memories
    ]


async def forget_memory(memory_id: str) -> dict:
    """Archive a memory so it no longer appears in search results.

    The memory is NOT deleted — it is archived and can be restored.
    Use when the user says a memory is outdated, wrong, or no longer relevant.

    Args:
        memory_id: The ID of the memory to archive (e.g. "mem_abc123").

    Returns:
        Confirmation with the archived memory's details.
    """
    store, _ = _get_store()

    memory = store.get(memory_id)
    if memory is None:
        return {
            "success": False,
            "error":   f"Memory not found: {memory_id}",
        }

    summary = memory.display_summary()
    store.forget(memory_id)

    return {
        "success":   True,
        "id":        memory_id,
        "summary":   summary,
        "lifecycle": "archived",
        "message":   f"Memory archived. It will no longer appear in search results.",
    }