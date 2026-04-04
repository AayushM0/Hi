"""Memory CRUD operations — the interface between the CLI/MCP and the vault."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from lace.core.config import LaceConfig, get_lace_home, load_config
from lace.memory.markdown import (
    load_all_memories,
    markdown_to_memory,
    save_memory_to_file,
)
from lace.memory.models import (
    MemoryCategory,
    MemoryLifecycle,
    MemoryObject,
    MemorySource,
    make_memory,
)


# ── MemoryStore ───────────────────────────────────────────────────────────────

class MemoryStore:
    """Primary interface for reading and writing memories.

    All operations go through here. The store talks to the vault
    (markdown files) directly. Vector operations are added in Chunk 3.
    """

    def __init__(
        self,
        lace_home: Path | None = None,
        config: LaceConfig | None = None,
    ) -> None:
        self.lace_home = lace_home or get_lace_home()
        self.config = config or load_config(self.lace_home)
        self.vault_path = self.config.vault_path(self.lace_home)

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(
        self,
        content: str,
        category: str | MemoryCategory = MemoryCategory.PATTERN,
        tags: list[str] | None = None,
        scope: str = "global",
        source: str | MemorySource = MemorySource.MANUAL,
        confidence: float = 0.8,
        summary: str | None = None,
    ) -> MemoryObject:
        """Create and persist a new memory.

        Returns the created MemoryObject.
        """
        memory = make_memory(
            content=content,
            category=category,
            tags=tags or [],
            scope=scope,
            source=source,
            confidence=confidence,
        )
        if summary:
            memory.summary = summary

        save_memory_to_file(memory, self.vault_path)
        return memory

    def save(self, memory: MemoryObject) -> Path:
        """Persist an existing MemoryObject (update its file)."""
        return save_memory_to_file(memory, self.vault_path)

    def forget(self, memory_id: str) -> bool:
        """Archive a memory so it no longer appears in active searches.

        Returns True if the memory was found and archived, False otherwise.
        Note: never deletes — only archives.
        """
        memory = self.get(memory_id)
        if memory is None:
            return False

        memory.archive()
        save_memory_to_file(memory, self.vault_path)
        return True

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, memory_id: str) -> MemoryObject | None:
        """Fetch a single memory by ID. Returns None if not found."""
        for md_file in self.vault_path.rglob(f"{memory_id}.md"):
            return markdown_to_memory(md_file)
        return None

    def list(
        self,
        category: str | MemoryCategory | None = None,
        scope: str | None = None,
        lifecycle: str | MemoryLifecycle | None = None,
        include_archived: bool = False,
        limit: int = 100,
    ) -> list[MemoryObject]:
        """Return memories with optional filtering.

        By default excludes archived memories.
        Results are sorted by last_accessed descending (most recent first).
        """
        memories = load_all_memories(self.vault_path)

        # Filter archived unless explicitly requested
        if not include_archived:
            memories = [m for m in memories if m.is_active()]

        # Filter by category
        if category is not None:
            cat = MemoryCategory(category) if isinstance(category, str) else category
            memories = [m for m in memories if m.category == cat]

        # Filter by scope
        if scope is not None:
            memories = [m for m in memories if m.project_scope == scope]

        # Filter by lifecycle
        if lifecycle is not None:
            lc = MemoryLifecycle(lifecycle) if isinstance(lifecycle, str) else lifecycle
            memories = [m for m in memories if m.lifecycle == lc]

        # Sort by last_accessed descending
        memories.sort(key=lambda m: m.last_accessed, reverse=True)

        return memories[:limit]

    def search_keyword(self, query: str, limit: int = 20) -> list[MemoryObject]:
        """Basic keyword search across memory content.

        This is the fallback when vector search is unavailable.
        Simple case-insensitive substring match.
        """
        query_lower = query.lower()
        memories = self.list(include_archived=False, limit=10_000)

        matches: list[tuple[MemoryObject, int]] = []
        for memory in memories:
            score = 0
            text = (memory.content + " " + " ".join(memory.tags) + " " + memory.category.value).lower()

            # Exact phrase match scores highest
            if query_lower in text:
                score += 10

            # Individual word matches
            for word in query_lower.split():
                if word in text:
                    score += 1

            if score > 0:
                matches.append((memory, score))

        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in matches[:limit]]

    def stats(self) -> dict[str, int | dict]:
        """Return memory statistics."""
        all_memories = load_all_memories(self.vault_path)

        by_category: dict[str, int] = {}
        by_lifecycle: dict[str, int] = {}
        by_scope: dict[str, int] = {}

        for memory in all_memories:
            by_category[memory.category.value] = by_category.get(memory.category.value, 0) + 1
            by_lifecycle[memory.lifecycle.value] = by_lifecycle.get(memory.lifecycle.value, 0) + 1
            by_scope[memory.project_scope] = by_scope.get(memory.project_scope, 0) + 1

        return {
            "total": len(all_memories),
            "active": sum(1 for m in all_memories if m.is_active()),
            "archived": sum(1 for m in all_memories if not m.is_active()),
            "by_category": by_category,
            "by_lifecycle": by_lifecycle,
            "by_scope": by_scope,
        }