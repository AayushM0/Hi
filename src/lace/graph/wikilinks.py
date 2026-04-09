"""Generate and inject wikilinks into memory markdown files."""

from __future__ import annotations

from pathlib import Path
from typing import Set
import logging

from lace.core.config import get_lace_home, load_config
from lace.graph.graph import build_graph
from lace.memory.markdown import load_all_memories, markdown_to_memory
from lace.memory.models import MemoryObject

logger = logging.getLogger(__name__)


def extract_existing_wikilinks(content: str) -> Set[str]:
    """Extract all [[wikilink]] references from markdown content."""
    import re
    pattern = r"\[\[([^\]]+)\]\]"
    matches = re.findall(pattern, content)
    return set(matches)


def get_related_concepts_for_memory(
    memory_id: str,
    graph,
    include_memory_links: bool = False,
) -> list[str]:
    """Get all concepts related to a memory node via the knowledge graph.
    
    Args:
        memory_id: The memory node ID
        graph: NetworkX DiGraph
        include_memory_links: If True, also return related memory IDs
        
    Returns:
        List of concept names (and optionally memory IDs) to link
    """
    if memory_id not in graph:
        return []
    
    related = []
    visited = {memory_id}
    
    # Direct outgoing edges (tags and links)
    for target in graph.successors(memory_id):
        target_data = graph.nodes[target]
        if target_data.get("type") == "concept":
            related.append(target)
            visited.add(target)
    
    # Co-occurring concepts (via other memories)
    for target in graph.successors(memory_id):
        if target in visited:
            continue
        if graph.nodes[target].get("type") == "concept":
            # Find other memories that share this concept
            for other_memory in graph.predecessors(target):
                if (other_memory != memory_id and 
                    graph.nodes[other_memory].get("type") == "memory"):
                    related.append(other_memory)
                    visited.add(other_memory)
    
    return sorted(list(set(related)))


def inject_wikilinks_into_memory(
    memory: MemoryObject,
    graph,
    markdown_path: Path,
) -> bool:
    """Inject wikilinks into a memory markdown file.
    
    Adds [[concept]] links for:
    - All tags (already in tags field, but adds wikilinks)
    - All tagged concepts from graph
    - Related concepts via co-occurrence
    
    Preserves existing wikilinks.
    
    Args:
        memory: MemoryObject
        graph: NetworkX graph
        markdown_path: Path to the .md file
        
    Returns:
        True if file was modified, False otherwise
    """
    if not markdown_path.exists():
        logger.warning(f"File not found: {markdown_path}")
        return False
    
    content = markdown_path.read_text(encoding="utf-8")
    original_content = content
    
    # Split frontmatter and content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) < 3:
            logger.warning(f"Invalid frontmatter in {markdown_path}")
            return False
        frontmatter = parts[1]
        body = parts[2].lstrip("\n")
    else:
        frontmatter = ""
        body = content
    
    # Extract existing wikilinks to preserve them
    existing_links = extract_existing_wikilinks(body)
    
    # Get related concepts from graph
    related = get_related_concepts_for_memory(memory.id, graph)
    
    # Build wikilinks section
    wikilinks = set()
    
    # Add tag-based wikilinks
    for tag in memory.tags:
        wikilinks.add(f"[[{tag}]]")
    
    # Add graph-derived wikilinks
    for concept in related:
        if concept not in memory.tags:  # Avoid duplication
            wikilinks.add(f"[[{concept}]]")
    
    # Preserve existing wikilinks
    for link in existing_links:
        wikilinks.add(f"[[{link}]]")
    
    if not wikilinks:
        return False  # Nothing to add
    
    # Create wikilinks section
    wikilinks_text = "\n\n**Related:**\n" + " ".join(sorted(wikilinks))
    
    # Check if wikilinks section already exists
    if "**Related:**" in body:
        # Replace existing section
        import re
        pattern = r"\n\*\*Related:\*\*.*?(?=\n\n|$)"
        new_body = re.sub(pattern, wikilinks_text, body, flags=re.DOTALL)
    else:
        # Append new section
        new_body = body.rstrip() + wikilinks_text
    
    # Reconstruct file
    if frontmatter:
        new_content = f"---{frontmatter}---\n{new_body}"
    else:
        new_content = new_body
    
    # Write back if changed
    if new_content != original_content:
        markdown_path.write_text(new_content, encoding="utf-8")
        logger.info(f"Added wikilinks to: {markdown_path.name}")
        return True
    
    return False


def inject_wikilinks_all(lace_home: Path | None = None) -> dict:
    """Inject wikilinks into all memory files.
    
    Returns:
        Dict with counts of updated files
    """
    lace_home = lace_home or get_lace_home()
    config = load_config(lace_home)
    vault_path = config.vault_path(lace_home)
    
    # Load graph
    from lace.core.engine import GraphManager
    gm = GraphManager(lace_home=lace_home, config=config)
    graph = gm.get_graph()
    
    if graph.number_of_nodes() == 0:
        logger.warning("Graph is empty")
        return {"updated": 0, "total": 0}
    
    # Load all memories
    memories = load_all_memories(vault_path)
    
    updated = 0
    total = 0
    
    # Find all .md files recursively in vault
    for md_path in vault_path.rglob("mem_*.md"):
        total += 1
        
        # Extract memory ID from filename
        import re
        match = re.search(r"mem_([a-f0-9]{12})", md_path.name)
        if not match:
            continue
        
        memory_id = "mem_" + match.group(1)
        
        # Find corresponding memory object
        memory = next((m for m in memories if m.id == memory_id), None)
        if not memory:
            logger.warning(f"Memory {memory_id} not found in store")
            continue
        
        if inject_wikilinks_into_memory(memory, graph, md_path):
            updated += 1
    
    return {
        "updated": updated,
        "total": total,
    }
