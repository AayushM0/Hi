"""Graph traversal queries for LACE knowledge graph.

Given a concept or memory ID, find related nodes
within N hops. This enables associative retrieval —
surfacing memories connected to the query concept
even if not semantically similar.
"""

from __future__ import annotations

from typing import Any

import networkx as nx


# ── Traversal ─────────────────────────────────────────────────────────────────

def get_neighbors(
    G: nx.DiGraph,
    node_id: str,
    depth: int = 2,
    max_nodes: int = 20,
) -> list[dict[str, Any]]:
    """Get all nodes within `depth` hops of `node_id`.

    Uses BFS (breadth-first search) to find neighbors.

    Args:
        G: The knowledge graph.
        node_id: Starting node (memory ID or concept name).
        depth: How many hops to traverse.
        max_nodes: Maximum nodes to return.

    Returns:
        List of dicts with node info and hop distance.
        Sorted by distance (closest first).
    """
    if node_id not in G:
        return []

    visited: dict[str, int] = {node_id: 0}
    queue   = [node_id]
    results = []

    while queue and len(results) < max_nodes:
        current = queue.pop(0)
        current_depth = visited[current]

        if current_depth >= depth:
            continue

        # Get both successors and predecessors (undirected traversal)
        neighbors = set(G.successors(current)) | set(G.predecessors(current))

        for neighbor in neighbors:
            if neighbor not in visited:
                hop = current_depth + 1
                visited[neighbor] = hop

                node_data = G.nodes[neighbor]
                results.append({
                    "id":       neighbor,
                    "type":     node_data.get("type", "unknown"),
                    "label":    node_data.get("label", neighbor),
                    "category": node_data.get("category"),
                    "distance": hop,
                    "relation": _get_edge_relation(G, current, neighbor),
                })
                queue.append(neighbor)

    results.sort(key=lambda x: x["distance"])
    return results[:max_nodes]


def find_memories_near_concept(
    G: nx.DiGraph,
    concept: str,
    depth: int = 2,
    max_memories: int = 10,
) -> list[dict[str, Any]]:
    """Find memory nodes connected to a concept within N hops.

    Args:
        G: The knowledge graph.
        concept: Concept name to search from (e.g. "asyncpg").
        depth: How many hops to traverse.
        max_memories: Maximum memory nodes to return.

    Returns:
        List of memory node dicts sorted by proximity.
    """
    # Normalize concept name
    concept_normalized = concept.lower().replace(" ", "-")

    # Try exact match first, then partial match
    start_node = None
    if concept_normalized in G:
        start_node = concept_normalized
    else:
        # Find closest matching node
        for node in G.nodes():
            if concept_normalized in str(node).lower():
                start_node = node
                break

    if start_node is None:
        return []

    neighbors = get_neighbors(G, start_node, depth=depth, max_nodes=50)

    # Filter to memory nodes only
    memory_nodes = [
        n for n in neighbors
        if n["type"] == "memory"
    ]

    return memory_nodes[:max_memories]


def get_concept_connections(
    G: nx.DiGraph,
    concept: str,
) -> list[dict[str, Any]]:
    """Get all concepts directly connected to a given concept.

    Args:
        G: The knowledge graph.
        concept: Concept name.

    Returns:
        List of connected concept dicts with relation type.
    """
    concept_normalized = concept.lower().replace(" ", "-")

    if concept_normalized not in G:
        return []

    connections = []
    neighbors = set(G.successors(concept_normalized)) | set(G.predecessors(concept_normalized))

    for neighbor in neighbors:
        node_data = G.nodes[neighbor]
        connections.append({
            "id":       neighbor,
            "type":     node_data.get("type", "unknown"),
            "label":    node_data.get("label", neighbor),
            "relation": _get_edge_relation(G, concept_normalized, neighbor),
        })

    return connections


def _get_edge_relation(G: nx.DiGraph, source: str, target: str) -> str:
    """Get the relation type of an edge, checking both directions."""
    if G.has_edge(source, target):
        return G[source][target].get("relation", "connected")
    elif G.has_edge(target, source):
        return G[target][source].get("relation", "connected")
    return "connected"


# ── Graph-augmented retrieval ──────────────────────────────────────────────────

def augment_with_graph(
    vector_results: list[str],
    query_concepts: list[str],
    G: nx.DiGraph,
    boost_factor: float = 0.1,
    depth: int = 2,
) -> dict[str, float]:
    """Compute graph-based score boosts for memory IDs.

    Memories that are graph-neighbors of query concepts get a boost.
    This is added on top of the vector similarity score.

    Args:
        vector_results: List of memory IDs from vector search.
        query_concepts: Concepts extracted from the query.
        G: The knowledge graph.
        boost_factor: How much to boost graph-connected memories.
        depth: Hop depth for graph traversal.

    Returns:
        Dict mapping memory_id → boost_score (0.0 to boost_factor).
    """
    boosts: dict[str, float] = {}

    for concept in query_concepts:
        nearby_memories = find_memories_near_concept(G, concept, depth=depth)
        for mem in nearby_memories:
            mem_id = mem["id"]
            distance = mem["distance"]
            # Closer = bigger boost
            boost = boost_factor * (1.0 / distance)
            boosts[mem_id] = max(boosts.get(mem_id, 0), boost)

    return boosts
