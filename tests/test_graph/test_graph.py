"""Tests for knowledge graph building."""

import pytest
import networkx as nx
from lace.memory.models import make_memory, MemoryCategory
from lace.graph.graph import build_graph, get_graph_stats, save_graph, load_graph


def _make_tagged_memory(content: str, tags: list[str], **kwargs):
    m = make_memory(content, **kwargs)
    m.tags = tags
    return m


def test_build_graph_empty():
    G = build_graph([])
    assert G.number_of_nodes() == 0


def test_build_graph_single_memory():
    memory = _make_tagged_memory("Use asyncpg", ["asyncpg", "db"])
    G = build_graph([memory])

    assert memory.id in G
    assert "asyncpg" in G
    assert "db" in G


def test_build_graph_memory_edges_to_tags():
    memory = _make_tagged_memory("Use asyncpg for db", ["asyncpg", "db"])
    G = build_graph([memory])

    assert G.has_edge(memory.id, "asyncpg")
    assert G.has_edge(memory.id, "db")


def test_build_graph_co_occurrence_edges():
    memory = _make_tagged_memory("Use asyncpg for db", ["asyncpg", "db"])
    G = build_graph([memory])

    # asyncpg and db co-occur → should have edge between them
    assert G.has_edge("asyncpg", "db") or G.has_edge("db", "asyncpg")


def test_build_graph_excludes_archived():
    memory = _make_tagged_memory("Archived memory", ["asyncpg"])
    memory.archive()
    G = build_graph([memory])

    assert memory.id not in G


def test_build_graph_multiple_memories():
    m1 = _make_tagged_memory("Use asyncpg", ["asyncpg", "db"])
    m2 = _make_tagged_memory("Use PostgreSQL", ["postgresql", "db"])
    G = build_graph([m1, m2])

    assert m1.id in G
    assert m2.id in G
    # Both share "db" tag
    assert "db" in G


def test_get_graph_stats():
    memory = _make_tagged_memory("Use asyncpg", ["asyncpg", "db"])
    G = build_graph([memory])
    stats = get_graph_stats(G)

    assert stats["memory_nodes"] == 1
    assert stats["concept_nodes"] >= 2
    assert stats["total_edges"] > 0
    assert not stats["is_empty"]


def test_save_and_load_graph(tmp_path):
    memory = _make_tagged_memory("Use asyncpg", ["asyncpg", "db"])
    G = build_graph([memory])

    graph_path = tmp_path / "graph.json"
    save_graph(G, graph_path)

    loaded = load_graph(graph_path)
    assert loaded.number_of_nodes() == G.number_of_nodes()
    assert loaded.number_of_edges() == G.number_of_edges()
    assert memory.id in loaded


def test_load_graph_missing_file(tmp_path):
    G = load_graph(tmp_path / "nonexistent.json")
    assert G.number_of_nodes() == 0