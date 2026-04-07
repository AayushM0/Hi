"""Graph manager — builds and caches the knowledge graph.

The graph is rebuilt from the vault on demand.
Cached in memory for the process lifetime.
Persisted to disk between processes.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from lace.core.config import LaceConfig, get_lace_home, load_config
from lace.graph.graph import build_graph, load_graph, save_graph
from lace.memory.markdown import load_all_memories


class GraphManager:
    """Manages the LACE knowledge graph lifecycle."""

    def __init__(
        self,
        lace_home: Path | None = None,
        config: LaceConfig | None = None,
    ) -> None:
        self.lace_home  = lace_home or get_lace_home()
        self.config     = config or load_config(self.lace_home)
        self.vault_path = self.config.vault_path(self.lace_home)
        self.graph_path = self.lace_home / "memory" / "graph" / "graph.json"
        self._graph: nx.DiGraph | None = None

    def get_graph(self, rebuild: bool = False) -> nx.DiGraph:
        """Return the knowledge graph.

        Loads from disk if available, builds from vault if not.

        Args:
            rebuild: Force rebuild from vault even if cached.

        Returns:
            The knowledge graph.
        """
        if self._graph is not None and not rebuild:
            return self._graph

        # Try loading from disk first
        if self.graph_path.exists() and not rebuild:
            self._graph = load_graph(self.graph_path)
            if self._graph.number_of_nodes() > 0:
                return self._graph

        # Build from vault
        self._graph = self._build_and_save()
        return self._graph

    def _build_and_save(self) -> nx.DiGraph:
        """Build graph from vault and persist to disk."""
        memories = load_all_memories(self.vault_path)
        G = build_graph(memories)
        save_graph(G, self.graph_path)
        return G

    def rebuild(self) -> nx.DiGraph:
        """Force rebuild from vault."""
        self._graph = self._build_and_save()
        return self._graph

    def add_memory_to_graph(self, memory) -> None:
        """Add a single memory to the existing graph."""
        from lace.graph.graph import build_graph
        G = self.get_graph()
        single_memory_graph = build_graph([memory])

        # Merge nodes and edges
        G.add_nodes_from(single_memory_graph.nodes(data=True))
        G.add_edges_from(single_memory_graph.edges(data=True))

        self._graph = G
        save_graph(G, self.graph_path)
