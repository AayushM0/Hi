"""Add wikilinks and related memories to all vault files for Obsidian."""

from pathlib import Path
import frontmatter
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lace.core.config import get_lace_home, load_config
from lace.core.engine import GraphManager
from lace.graph.traversal import get_neighbors


def add_wikilinks_to_content(content: str, tags: list[str]) -> str:
    """Add [[wikilinks]] for tags that appear in the content."""
    updated = content
    
    for tag in tags:
        if tag in updated and f"[[{tag}]]" not in updated:
            import re
            pattern = r'\b' + re.escape(tag) + r'\b'
            updated = re.sub(pattern, f"[[{tag}]]", updated, count=1, flags=re.IGNORECASE)
    
    return updated


def get_tags_section(tags: list[str]) -> str:
    """Generate a tags section with wikilinks."""
    if not tags:
        return ""
    return "\n\n**Tags**: " + " · ".join(f"[[{tag}]]" for tag in tags)


def get_related_section(memory_id: str, graph, max_related: int = 5) -> str:
    """Generate a 'Related' section using 2-hop graph traversal."""
    # Get neighbors at depth=2 to find memories that share tags
    neighbors = get_neighbors(graph, memory_id, depth=2)
    
    # Filter to only memory nodes (not concepts), exclude self
    memory_neighbors = [
        n for n in neighbors 
        if n["type"] == "memory" and n["id"] != memory_id
    ][:max_related]
    
    if not memory_neighbors:
        return ""
    
    lines = ["\n\n## Related"]
    for neighbor in memory_neighbors:
        label = neighbor.get("label", neighbor["id"])[:60]
        # Use Obsidian alias syntax: [[id|display text]]
        lines.append(f"- [[{neighbor['id']}|{label}]]")
    
    return "\n".join(lines)


def process_vault(vault_path: Path, graph, force: bool = False) -> tuple[int, int]:
    """Add wikilinks to all memory files in the vault."""
    updated = 0
    skipped = 0
    
    for md_file in vault_path.rglob("*.md"):
        try:
            post = frontmatter.load(str(md_file))
            
            if "id" not in post.metadata:
                skipped += 1
                continue
            
            memory_id = post.metadata["id"]
            tags = post.metadata.get("tags", [])
            content = post.content.strip()
            
            # Skip if already processed (unless force)
            if not force and ("## Related" in content or "**Tags**:" in content):
                skipped += 1
                continue
            
            # Build new content
            updated_content = add_wikilinks_to_content(content, tags)
            
            # Add tags section
            tags_section = get_tags_section(tags)
            
            # Add related memories (2-hop traversal)
            related_section = get_related_section(memory_id, graph, max_related=5)
            
            # Combine
            new_content = updated_content + tags_section + related_section
            
            if new_content != content:
                post.content = new_content
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(frontmatter.dumps(post))
                updated += 1
                print(f"✓ {md_file.name}")
            else:
                skipped += 1
        
        except Exception as e:
            print(f"✗ {md_file.name}: {e}")
            skipped += 1
    
    return updated, skipped


def main():
    lace_home = get_lace_home()
    config = load_config(lace_home)
    vault_path = config.vault_path(lace_home)
    
    print("Building knowledge graph...")
    manager = GraphManager(lace_home=lace_home)
    graph = manager.rebuild()
    
    print(f"\nProcessing vault: {vault_path}")
    # Use force=True to re-process files that already have Tags sections
    updated, skipped = process_vault(vault_path, graph, force=True)
    
    print(f"\n✓ Updated: {updated} files")
    print(f"⊘ Skipped: {skipped} files")
    print("\nOpen your vault in Obsidian to see the connections!")


if __name__ == "__main__":
    main()
