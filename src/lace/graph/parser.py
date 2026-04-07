"""Parse [[wikilinks]] from markdown memory files.

Wikilinks are the connective tissue of the knowledge graph.
Every [[link]] in a memory file becomes an edge in the graph.

Examples:
  "Use asyncpg with [[connection-pooling]]"
  → edge: (memory_id, "connection-pooling")

  "See [[postgresql]] and [[alembic]] for migrations"
  → edges: (memory_id, "postgresql"), (memory_id, "alembic")
"""

from __future__ import annotations

import re
from pathlib import Path

# Matches [[anything in here]] including spaces and hyphens
WIKILINK_PATTERN = re.compile(r"\[\[([^\[\]]+)\]\]")


def extract_wikilinks(text: str) -> list[str]:
    """Extract all [[wikilink]] targets from a text string.

    Args:
        text: Markdown text to parse.

    Returns:
        List of link targets (the text inside [[ ]]).
        Normalized to lowercase with spaces replaced by hyphens.
    """
    matches = WIKILINK_PATTERN.findall(text)
    normalized = []
    for match in matches:
        # Normalize: lowercase, strip whitespace, spaces → hyphens
        clean = match.strip().lower().replace(" ", "-")
        if clean:
            normalized.append(clean)
    return normalized


def extract_wikilinks_from_file(file_path: Path) -> list[str]:
    """Extract all [[wikilinks]] from a markdown file.

    Reads the full file content (frontmatter + body) and extracts links.

    Args:
        file_path: Path to the markdown file.

    Returns:
        List of normalized link targets.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        return extract_wikilinks(content)
    except Exception:
        return []


def extract_tags_as_links(tags: list[str]) -> list[str]:
    """Convert memory tags into implicit wikilink targets.

    Tags act as implicit connections between memories.
    A memory tagged "asyncpg" is implicitly linked to the "asyncpg" concept.

    Args:
        tags: List of tag strings.

    Returns:
        Normalized tag strings suitable for graph edges.
    """
    return [tag.lower().replace(" ", "-") for tag in tags if tag]
