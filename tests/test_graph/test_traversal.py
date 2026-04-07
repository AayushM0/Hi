"""Tests for graph traversal."""

import pytest
from lace.memory.models import make_memory
from lace.graph.graph import build_graph
from lace.graph.traversal import (
    get_neighbors,
    find_memories_near_concept,
    get_concept_connections,
)


def _make_tagged_memory(content: str, tags: list[str]):
    m = make_memory(content)
    m.tags = tags
    return m


@pytest.fixture
def sample_graph():
    m1 = _make_tagged_memory("Use asyncpg for PostgreSQL", ["asyncpg", "postgresql"])
    m2 = _make_tagged_memory("Connection pooling with asyncpg", ["asyncpg", "pooling"])
    m3 = _make_tagged_memory("Alembic for migrations", ["alembic", "postgresql"])
    return build_graph([m1, m2, m3]), [m1, m2, m3]


def test_get_neighbors_concept(sample_graph):
    G, memories = sample_graph
    neighbors = get_neighbors(G, "asyncpg", depth=1)
    assert len(neighbors) > 0
    ids = [n["id"] for n in neighbors]
    # asyncpg is connected to memories and other concepts
    assert any(m.id in ids for m in memories)


def test_get_neighbors_unknown_node(sample_graph):
    G, _ = sample_graph
    result = get_neighbors(G, "nonexistent_concept", depth=2)
    assert result == []


def test_get_neighbors_depth_1_vs_2(sample_graph):
    G, _ = sample_graph
    d1 = get_neighbors(G, "asyncpg", depth=1)
    d2 = get_neighbors(G, "asyncpg", depth=2)
    # Depth 2 should return at least as many as depth 1
    assert len(d2) >= len(d1)


def test_find_memories_near_concept(sample_graph):
    G, memories = sample_graph
    results = find_memories_near_concept(G, "asyncpg", depth=2)
    assert len(results) > 0
    assert all(r["type"] == "memory" for r in results)


def test_find_memories_near_concept_unknown(sample_graph):
    G, _ = sample_graph
    results = find_memories_near_concept(G, "unknown_concept_xyz", depth=2)
    assert results == []


def test_get_concept_connections(sample_graph):
    G, _ = sample_graph
    connections = get_concept_connections(G, "asyncpg")
    assert len(connections) > 0


def test_neighbors_sorted_by_distance(sample_graph):
    G, _ = sample_graph
    neighbors = get_neighbors(G, "asyncpg", depth=2)
    distances = [n["distance"] for n in neighbors]
    assert distances == sorted(distances)
    