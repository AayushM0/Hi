"""Knowledge graph for LACE.

Nodes:
  - Memory nodes: represent stored MemoryObjects
  - Concept nodes: represent tags and [[wikilink]] targets

Edges:
  - memory → concept (via tags)
  - memory → concept (via [[wikilinks]] in content)
  - concept → concept (via co-occurrence in same memory)

The graph is built from the vault on demand and serialized
to ~/.lace/memory/graph/graph.json for persistence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

from lace.graph.parser import (
    extract_tags_as_links,
    extract_wikilinks_from_file,
)
from lace.memory.models import MemoryObject


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(memories: list[MemoryObject]) -> nx.DiGraph:
    """Build a directed knowledge graph from a list of MemoryObjects.

    Node types:
      - type="memory"  → a stored MemoryObject
      - type="concept" → a tag or wikilink target

    Edge types:
      - "tagged_with"   → memory uses this tag
      - "links_to"      → memory contains [[wikilink]]
      - "co_occurs"     → two concepts appear in the same memory

    Args:
        memories: List of MemoryObjects to build graph from.

    Returns:
        Directed graph (DiGraph).
    """
    G = nx.DiGraph()

    for memory in memories:
        if not memory.is_active():
            continue

        # Add memory node
        G.add_node(
            memory.id,
            type="memory",
            label=memory.display_summary()[:50],
            category=memory.category.value,
            confidence=memory.confidence,
            scope=memory.project_scope,
        )

        # Collect all concepts for this memory
        concepts: list[str] = []

        # 1. Tags → concept nodes
        tag_concepts = extract_tags_as_links(memory.tags)
        for concept in tag_concepts:
            if not G.has_node(concept):
                G.add_node(concept, type="concept", label=concept)
            G.add_edge(memory.id, concept, relation="tagged_with")
            concepts.append(concept)

        # 2. Wikilinks from file → concept nodes
        if memory.file_path:
            file_path = Path(memory.file_path)
            if file_path.exists():
                wikilinks = extract_wikilinks_from_file(file_path)
                for link in wikilinks:
                    if not G.has_node(link):
                        G.add_node(link, type="concept", label=link)
                    G.add_edge(memory.id, link, relation="links_to")
                    concepts.append(link)

        # 3. Wikilinks from content (in-memory)
        from lace.graph.parser import extract_wikilinks
        content_links = extract_wikilinks(memory.content)
        for link in content_links:
            if not G.has_node(link):
                G.add_node(link, type="concept", label=link)
            if not G.has_edge(memory.id, link):
                G.add_edge(memory.id, link, relation="links_to")
            concepts.append(link)

        # 4. Co-occurrence edges between concepts in same memory
        unique_concepts = list(set(concepts))
        for i, c1 in enumerate(unique_concepts):
            for c2 in unique_concepts[i + 1:]:
                if not G.has_edge(c1, c2):
                    G.add_edge(c1, c2, relation="co_occurs", weight=1)
                else:
                    # Increment weight for repeated co-occurrence
                    G[c1][c2]["weight"] = G[c1][c2].get("weight", 1) + 1

    return G


# ── Persistence ───────────────────────────────────────────────────────────────

def save_graph(G: nx.DiGraph, graph_path: Path) -> None:
    """Serialize graph to JSON file.

    Args:
        G: The graph to save.
        graph_path: Path to write the JSON file.
    """
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(G, edges="links")
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_graph(graph_path: Path) -> nx.DiGraph:
    """Load graph from JSON file.

    Returns an empty graph if file doesn't exist.

    Args:
        graph_path: Path to the JSON file.

    Returns:
        Loaded DiGraph, or empty DiGraph if file not found.
    """
    if not graph_path.exists():
        return nx.DiGraph()

    try:
        with open(graph_path, encoding="utf-8") as f:
            data = json.load(f)
        return nx.node_link_graph(data, directed=True, edges="links")
    except Exception:
        return nx.DiGraph()


def get_graph_stats(G: nx.DiGraph) -> dict[str, Any]:
    """Return basic statistics about the graph.

    Args:
        G: The graph to analyze.

    Returns:
        Dict with node/edge counts and type breakdowns.
    """
    memory_nodes  = [n for n, d in G.nodes(data=True) if d.get("type") == "memory"]
    concept_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "concept"]

    edge_types: dict[str, int] = {}
    for _, _, data in G.edges(data=True):
        rel = data.get("relation", "unknown")
        edge_types[rel] = edge_types.get(rel, 0) + 1

    return {
        "total_nodes":    G.number_of_nodes(),
        "memory_nodes":   len(memory_nodes),
        "concept_nodes":  len(concept_nodes),
        "total_edges":    G.number_of_edges(),
        "edge_types":     edge_types,
        "is_empty":       G.number_of_nodes() == 0,
    }
