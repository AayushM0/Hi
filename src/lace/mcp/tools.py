"""MCP tool implementations — the actual functions exposed to AI agents."""

from __future__ import annotations
import sys

from lace.core.config import get_lace_home, load_config
from lace.core.scope import get_active_scope, get_projects
from lace.memory.models import MemoryCategory
from lace.memory.store import MemoryStore


def _debug_log(msg: str) -> None:
    """Debug logging to stderr (MCP uses stdout for JSON-RPC)."""
    print(f"[LACE DEBUG] {msg}", file=sys.stderr, flush=True)


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


def _multi_scope_search(
    store: MemoryStore,
    query: str,
    primary_scope: str,
    max_results: int,
) -> list:
    """Search across multiple scopes intelligently."""
    all_results = []
    
    if primary_scope.startswith("session:"):
        lace_home = get_lace_home()
        projects = get_projects(lace_home)
        
        for project in projects:
            project_results = store.search(
                query=query,
                scope=project["scope"],
                max_results=max_results,
            )
            all_results.extend(project_results)
        
        global_results = store.search(
            query=query,
            scope="global",
            max_results=max_results,
        )
        all_results.extend(global_results)
    
    elif primary_scope.startswith("project:"):
        primary_results = store.search(
            query=query,
            scope=primary_scope,
            max_results=max_results,
        )
        all_results.extend(primary_results)
        
        global_results = store.search(
            query=query,
            scope="global",
            max_results=max_results,
        )
        all_results.extend(global_results)
    
    else:
        global_results = store.search(
            query=query,
            scope="global",
            max_results=max_results,
        )
        all_results.extend(global_results)
    
    # Deduplicate
    seen = set()
    unique_results = []
    for result in all_results:
        if result.memory.id not in seen:
            seen.add(result.memory.id)
            unique_results.append(result)
    
    unique_results.sort(key=lambda r: r.relevance_score, reverse=True)
    
    return unique_results[:max_results]


# ── Tool implementations ──────────────────────────────────────────────────────

async def search_memory(
    query: str,
    scope: str = "auto",
    max_results: int = 5,
    category: str = "all",
    **kwargs  # Accept any extra args from MCP
) -> list[dict]:
    """Search your knowledge base for memories relevant to a query."""
    store, resolved_scope = _get_store(scope)

    results = _multi_scope_search(
        store=store,
        query=query,
        primary_scope=resolved_scope,
        max_results=min(max_results, 20),
    )

    if category != "all":
        try:
            cat = MemoryCategory(category)
            results = [r for r in results if r.memory.category == cat]
        except ValueError:
            pass

    return [
        {
            "id": r.memory.id,
            "content": r.memory.content,
            "summary": r.memory.display_summary(),
            "category": r.memory.category.value,
            "tags": r.memory.tags,
            "scope": r.memory.project_scope,
            "confidence": r.memory.confidence,
            "relevance_score": round(r.relevance_score, 3),
            "created_at": r.memory.created_at.isoformat(),
            "access_count": r.memory.access_count,
        }
        for r in results
    ]


async def get_project_context(
    project_name: str | None = None,
    **kwargs
) -> dict:
    """Get memories and metadata for a specific project."""
    store, active_scope = _get_store()

    if project_name:
        scope = f"project:{project_name}"
    else:
        scope = active_scope if active_scope.startswith("project:") else "global"

    memories = store.list(scope=scope, limit=50, include_archived=False)
    patterns = [m for m in memories if m.category == MemoryCategory.PATTERN][:10]
    decisions = [m for m in memories if m.category == MemoryCategory.DECISION][:10]

    # Load identity and preferences
    from lace.core.identity import compose_identity
    from lace.core.config import get_lace_home
    lace_home = get_lace_home()
    identity_text, preferences = compose_identity(lace_home, scope=scope)

    return {
        "scope": scope,
        "total_memories": len(memories),
        "identity": identity_text or "",
        "preferences": preferences or {},
        "patterns": [
            {"id": m.id, "summary": m.display_summary(), "tags": m.tags, "confidence": m.confidence}
            for m in patterns
        ],
        "decisions": [
            {"id": m.id, "summary": m.display_summary(), "tags": m.tags, "confidence": m.confidence}
            for m in decisions
        ],
    }


async def remember(
    content: str,
    category: str = "pattern",
    tags: list[str] | None = None,
    confidence: float = 0.7,
    scope: str = "auto",
    **kwargs
) -> dict:
    """Store a new memory from this interaction."""
    store, resolved_scope = _get_store(scope)
    
    # If resolved to a session, default to global instead (sessions are ephemeral)
    if resolved_scope.startswith("session:"):
        resolved_scope = "global"

    try:
        cat = MemoryCategory(category)
    except ValueError:
        cat = MemoryCategory.PATTERN

    memory = store.add(
        content=content,
        category=cat,
        tags=tags or [],
        scope=resolved_scope,
        source="mcp",
        confidence=max(0.0, min(1.0, confidence)),
    )

    return {
        "stored": True,
        "status": "stored",
        "id": memory.id,
        "scope": memory.project_scope,
        "category": memory.category.value,
    }


async def list_memories(
    scope: str = "auto",
    category: str = "all",
    limit: int = 20,
    **kwargs  # Accept lifecycle and other args
) -> list[dict]:
    """List recent memories, optionally filtered by category or scope."""
    store, resolved_scope = _get_store(scope)

    cat_filter = None
    if category != "all":
        try:
            cat_filter = MemoryCategory(category)
        except ValueError:
            pass

    # Handle lifecycle filter (ignore if "all" or invalid)
    lifecycle_filter = None
    if "lifecycle" in kwargs and kwargs["lifecycle"] != "all":
        try:
            from lace.memory.models import MemoryLifecycle
            lifecycle_filter = MemoryLifecycle(kwargs["lifecycle"])
        except (ValueError, KeyError):
            pass
    
    memories = store.list(
        category=cat_filter,
        scope=resolved_scope if resolved_scope != "global" else None,
        limit=limit,
        lifecycle=lifecycle_filter,
        include_archived=False,
    )

    return [
        {
            "id": m.id,
            "summary": m.display_summary(),
            "category": m.category.value,
            "tags": m.tags,
            "scope": m.project_scope,
            "confidence": m.confidence,
            "last_accessed": m.last_accessed.isoformat(),
        }
        for m in memories
    ]


async def forget_memory(
    memory_id: str,
    **kwargs
) -> dict:
    """Archive a memory."""
    store, _ = _get_store()
    success = store.forget(memory_id)

    return {
        "success": success,
        "status": "archived" if success else "not_found",
        "lifecycle": "archived" if success else "not_found",
        "id": memory_id,
        **({"error": f"Memory {memory_id} not found"} if not success else {}),
    }


async def get_related_concepts(
    concept: str,
    depth: int = 2,
    **kwargs  # Accept memories_only and other args
) -> list[dict]:
    """Find memories and concepts related to a given concept via the knowledge graph."""
    from lace.core.engine import GraphManager
    from lace.graph.traversal import find_memories_near_concept

    lace_home = get_lace_home()
    manager = GraphManager(lace_home=lace_home)
    G = manager.get_graph()

    if G.number_of_nodes() == 0:
        return []

    related = find_memories_near_concept(G, concept, depth=min(depth, 3))

    return [
        {
            "type": node["type"],
            "id": node["id"],
            "label": node.get("label", ""),
            "distance": node["distance"],
        }
        for node in related[:20]
    ]
