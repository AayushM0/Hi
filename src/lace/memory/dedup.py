"""Deduplication logic for LACE memory system.

Before storing a new memory, we check if it's too similar
to an existing one. Three outcomes:

  - STORE  → novel content, add it
  - MERGE  → very similar, update existing
  - SKIP   → nearly identical, discard

Thresholds (configurable):
  > 0.95 cosine similarity → SKIP  (nearly identical)
  > 0.85 cosine similarity → MERGE (combine into one)
  ≤ 0.85                  → STORE (novel)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from lace.memory.models import MemoryObject


class DedupAction(str, Enum):
    STORE = "store"   # Novel — add it
    MERGE = "merge"   # Similar — update existing
    SKIP  = "skip"    # Duplicate — discard


@dataclass
class DedupResult:
    action:          DedupAction
    candidate:       MemoryObject
    existing:        MemoryObject | None = None
    similarity:      float               = 0.0
    reason:          str                 = ""


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import numpy as np
    a = np.array(vec_a)
    b = np.array(vec_b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def check_duplicate(
    candidate: MemoryObject,
    existing_memories: list[MemoryObject],
    skip_threshold:  float = 0.95,
    merge_threshold: float = 0.85,
) -> DedupResult:
    if candidate.embedding is None:
        return DedupResult(
            action=DedupAction.STORE,
            candidate=candidate,
            reason="no embedding available for comparison",
        )

    best_similarity = 0.0
    best_match: MemoryObject | None = None

    for existing in existing_memories:
        # Explicitly skip archived memories — they are dead
        if existing.lifecycle.value == "archived":
            continue
        if existing.embedding is None:
            continue
        if existing.category != candidate.category:
            continue

        sim = cosine_similarity(candidate.embedding, existing.embedding)
        if sim > best_similarity:
            best_similarity = sim
            best_match = existing

    if best_similarity > skip_threshold:
        return DedupResult(
            action=DedupAction.SKIP,
            candidate=candidate,
            existing=best_match,
            similarity=best_similarity,
            reason=f"nearly identical to existing memory (sim={best_similarity:.3f})",
        )

    if best_similarity > merge_threshold:
        return DedupResult(
            action=DedupAction.MERGE,
            candidate=candidate,
            existing=best_match,
            similarity=best_similarity,
            reason=f"very similar to existing memory (sim={best_similarity:.3f})",
        )

    return DedupResult(
        action=DedupAction.STORE,
        candidate=candidate,
        existing=best_match,
        similarity=best_similarity,
        reason="novel content",
    )


def merge_memories(
    existing: MemoryObject,
    candidate: MemoryObject,
) -> MemoryObject:
    """Merge candidate into existing memory.

    Strategy:
    - Keep existing ID (stable reference)
    - Append candidate content if it adds new information
    - Merge tags
    - Boost confidence slightly
    - Update last_accessed
    """
    from datetime import datetime, timezone

    # Merge tags
    merged_tags = list(set(existing.tags + candidate.tags))

    # Append content only if candidate adds something new
    if candidate.content.strip() not in existing.content:
        existing.content = (
            existing.content.rstrip()
            + "\n\n"
            + candidate.content.strip()
        )

    existing.tags = merged_tags
    existing.confidence = min(1.0, existing.confidence + 0.05)
    existing.last_accessed = datetime.now(timezone.utc)
    existing.metadata["merged_from"] = existing.metadata.get(
        "merged_from", []
    ) + [candidate.id]

    return existing