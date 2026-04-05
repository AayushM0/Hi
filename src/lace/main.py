"""LACE CLI entry point."""

from __future__ import annotations

import os
import warnings

# Suppress ALL noisy warnings — must be before any other imports
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.expanduser("~/.cache/sentence_transformers")

from pathlib import Path
from typing import Annotated, Optional



import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import print as rprint


from lace.core.config import (
    get_lace_home,
    init_lace_home,
    load_config,
    set_config_value,
)
from lace.core.scope import (
    get_active_scope,
    detect_current_project,
    get_projects,
    create_project,
    set_project_last_used,
    get_active_session,
    create_new_session,
)
from lace.core.identity import compose_identity

from lace.core.config import (
    get_lace_home,
    init_lace_home,
    load_config,
    set_config_value,
)

app = typer.Typer(
    name="lace",
    help="LACE — Local AI Context Engine",
    add_completion=False,
    rich_markup_mode="rich",
)

config_app = typer.Typer(help="Manage LACE configuration.")
app.add_typer(config_app, name="config")

memory_app = typer.Typer(help="Manage memories.")
app.add_typer(memory_app, name="memory")

project_app = typer.Typer(help="Manage projects.")
app.add_typer(project_app, name="project")

mcp_app = typer.Typer(help="MCP server management.")
app.add_typer(mcp_app, name="mcp")

console = Console()


# ── lace init ─────────────────────────────────────────────────────────────────

@app.command()
def init(
    home: Annotated[
        Optional[str],
        typer.Option("--home", help="Custom LACE home directory."),
    ] = None,
) -> None:
    """Initialize LACE — create ~/.lace directory structure."""
    lace_home = Path(home).expanduser() if home else get_lace_home()

    with console.status("[bold green]Initializing LACE...[/bold green]"):
        path, already_existed = init_lace_home(lace_home)

    if already_existed:
        console.print(
            Panel(
                f"[yellow]LACE was already initialized.[/yellow]\n\n"
                f"Home: [bold]{path}[/bold]\n\n"
                f"Any missing files/directories have been created.",
                title="[bold yellow]LACE[/bold yellow]",
                border_style="yellow",
            )
        )
    else:
        console.print(
            Panel(
                f"[bold green]✓ LACE initialized successfully![/bold green]\n\n"
                f"Home: [bold]{path}[/bold]\n\n"
                f"[dim]Next steps:[/dim]\n"
                f"  1. Edit [bold]{path}/config/identity.md[/bold]\n"
                f"  2. Edit [bold]{path}/config/preferences.yaml[/bold]\n"
                f"  3. Run [bold]lace memory add \"your first memory\"[/bold]\n"
                f"  4. Run [bold]lace mcp start[/bold] — connect to Antigravity",
                title="[bold green]LACE[/bold green]",
                border_style="green",
            )
        )


# ── lace version ──────────────────────────────────────────────────────────────

@app.command()
def version() -> None:
    """Show LACE version."""
    from lace import __version__
    console.print(f"[bold]LACE[/bold] v{__version__}")


# ── config commands ───────────────────────────────────────────────────────────

@config_app.command("show")
def config_show() -> None:
    """Show current LACE configuration."""
    lace_home = get_lace_home()
    config = load_config(lace_home)

    table = Table(title="LACE Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Key", style="bold")
    table.add_column("Value")

    def flatten(d: dict, prefix: str = "") -> list[tuple[str, str]]:
        rows = []
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                rows.extend(flatten(v, full_key))
            else:
                rows.append((full_key, str(v)))
        return rows

    for key, value in flatten(config.model_dump()):
        table.add_row(key, value)

    console.print(table)
    console.print(f"\n[dim]Config file: {lace_home}/config/lace.yaml[/dim]")


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="Config key (e.g. memory.decay_half_life_days)")],
    value: Annotated[str, typer.Argument(help="New value")],
) -> None:
    """Set a configuration value."""
    try:
        set_config_value(key, value)
        console.print(f"[green]✓[/green] Set [bold]{key}[/bold] = [bold]{value}[/bold]")
    except KeyError as e:
        console.print(f"[red]✗ Unknown config key:[/red] {e}")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]✗ Invalid value:[/red] {e}")
        raise typer.Exit(1)


# ── memory commands ───────────────────────────────────────────────────────────

def _get_store(scope: str | None = None):
    """Get a MemoryStore instance with active scope."""
    from lace.memory.store import MemoryStore
    store = MemoryStore()
    # Set active scope for store to use in searches
    if scope is None:
        scope = get_active_scope()
    # We'll add active scope to store in Chunk 5
    return store


@memory_app.command("add")
def memory_add(
    content: Annotated[str, typer.Argument(help="The memory content to store.")],
    tag: Annotated[
        Optional[list[str]],
        typer.Option("--tag", "-t", help="Tags (repeatable: --tag=pattern --tag=db)"),
    ] = None,
    category: Annotated[
        str,
        typer.Option("--category", "-c", help="Category: pattern, decision, debug, reference, preference"),
    ] = "pattern",
    scope: Annotated[
        str,
        typer.Option("--scope", "-s", help="Scope: global or project:<name>"),
    ] = "global",
    summary: Annotated[
        Optional[str],
        typer.Option("--summary", help="One-line summary for display."),
    ] = None,
) -> None:
    """Store a new memory."""
    store = _get_store()

    try:
        memory = store.add(
            content=content,
            category=category,
            tags=tag or [],
            scope=scope,
            summary=summary,
        )
    except ValueError as e:
        console.print(f"[red]✗ Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold green]✓ Memory stored[/bold green]\n\n"
            f"ID:       [bold]{memory.id}[/bold]\n"
            f"Category: {memory.category.value}\n"
            f"Scope:    {memory.project_scope}\n"
            f"Tags:     {', '.join(memory.tags) if memory.tags else '[dim]none[/dim]'}\n\n"
            f"[dim]{memory.display_summary()}[/dim]",
            title="[bold]Memory Added[/bold]",
            border_style="green",
        )
    )


@memory_app.command("list")
def memory_list(
    category: Annotated[
        Optional[str],
        typer.Option("--category", "-c", help="Filter by category."),
    ] = None,
    scope: Annotated[
        Optional[str],
        typer.Option("--scope", "-s", help="Filter by scope."),
    ] = None,
    include_archived: Annotated[
        bool,
        typer.Option("--archived", help="Include archived memories."),
    ] = False,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Max results to show."),
    ] = 20,
) -> None:
    """List stored memories."""
    store = _get_store()
    memories = store.list(
        category=category,
        scope=scope,
        include_archived=include_archived,
        limit=limit,
    )

    if not memories:
        console.print("[yellow]No memories found.[/yellow]")
        console.print("[dim]Try: lace memory add \"your first memory\"[/dim]")
        return

    table = Table(
        title=f"Memories ({len(memories)} shown)",
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("ID", style="dim", width=16)
    table.add_column("Category", width=10)
    table.add_column("Scope", width=14)
    table.add_column("Conf", width=5)
    table.add_column("Tags", width=20)
    table.add_column("Summary")

    for memory in memories:
        lifecycle_color = {
            "captured": "white",
            "validated": "green",
            "consolidated": "blue",
            "archived": "red",
        }.get(memory.lifecycle.value, "white")

        table.add_row(
            memory.id,
            memory.category.value,
            memory.project_scope,
            f"{memory.confidence:.2f}",
            ", ".join(memory.tags[:3]) if memory.tags else "[dim]—[/dim]",
            Text(memory.display_summary(), style=lifecycle_color),
        )

    console.print(table)


@memory_app.command("show")
def memory_show(
    memory_id: Annotated[str, typer.Argument(help="Memory ID to show.")],
) -> None:
    """Show full details of a memory."""
    store = _get_store()
    memory = store.get(memory_id)

    if memory is None:
        console.print(f"[red]✗ Memory not found:[/red] {memory_id}")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold]{memory.display_summary()}[/bold]\n\n"
            f"{memory.content}\n\n"
            f"[dim]─────────────────────────────────[/dim]\n"
            f"ID:           [bold]{memory.id}[/bold]\n"
            f"Category:     {memory.category.value}\n"
            f"Source:       {memory.source.value}\n"
            f"Lifecycle:    {memory.lifecycle.value}\n"
            f"Confidence:   {memory.confidence:.2f}\n"
            f"Scope:        {memory.project_scope}\n"
            f"Tags:         {', '.join(memory.tags) if memory.tags else '[dim]none[/dim]'}\n"
            f"Access count: {memory.access_count}\n"
            f"Created:      {memory.created_at.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"Last access:  {memory.last_accessed.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"File:         [dim]{memory.file_path}[/dim]",
            title=f"[bold]Memory — {memory.id}[/bold]",
            border_style="cyan",
        )
    )


@memory_app.command("forget")
def memory_forget(
    memory_id: Annotated[str, typer.Argument(help="Memory ID to archive.")],
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation."),
    ] = False,
) -> None:
    """Archive a memory (removes from search, never deletes)."""
    store = _get_store()

    memory = store.get(memory_id)
    if memory is None:
        console.print(f"[red]✗ Memory not found:[/red] {memory_id}")
        raise typer.Exit(1)

    if not yes:
        console.print(f"Archive memory: [bold]{memory.display_summary()}[/bold]")
        confirmed = typer.confirm("This will remove it from search results. Continue?")
        if not confirmed:
            console.print("[dim]Cancelled.[/dim]")
            return

    store.forget(memory_id)
    console.print(f"[green]✓[/green] Memory [bold]{memory_id}[/bold] archived.")


@memory_app.command("search")
def memory_search(
    query: Annotated[str, typer.Argument(help="Search query.")],
    limit: Annotated[int, typer.Option("--limit", "-n")] = 10,
    scope: Annotated[
        Optional[str],
        typer.Option("--scope", "-s", help="Scope to search (defaults to active scope)."),
    ] = None,
    show_scores: Annotated[bool, typer.Option("--scores")] = False,
) -> None:
    """Semantic search across memories."""
    if scope is None:
        scope = get_active_scope()
    store = _get_store(scope=scope)

    with console.status(f"[bold green]Searching in {scope}:[/bold green] {query}"):
        results = store.search(query, scope=scope, max_results=limit)

    if not results:
        console.print(f"[yellow]No memories found for:[/yellow] {query}")
        return

    table = Table(
        title=f"Search in {scope}: '{query}' ({len(results)} results)",
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("Rank", width=4)
    table.add_column("ID", style="dim", width=16)
    table.add_column("Category", width=10)
    table.add_column("Tags", width=20)
    if show_scores:
        table.add_column("Score", width=6)
    table.add_column("Summary")

    for result in results:
        m = result.memory
        row = [
            str(result.rank),
            m.id,
            m.category.value,
            ", ".join(m.tags[:3]) if m.tags else "[dim]—[/dim]",
        ]
        if show_scores:
            row.append(f"{result.relevance_score:.3f}")
        row.append(m.display_summary())
        table.add_row(*row)

    console.print(table)
    console.print(f"[dim]Match type: {results[0].match_type if results else '—'}[/dim]")


@memory_app.command("reindex")
def memory_reindex() -> None:
    """Re-embed all memories into the vector store."""
    store = _get_store()
    with console.status("[bold green]Re-indexing all memories...[/bold green]"):
        success, failure = store.reindex_all()
    console.print(f"[green]✓[/green] Indexed {success} memories. Failures: {failure}")



@memory_app.command("stats")
def memory_stats() -> None:
    """Show memory system statistics."""
    store = _get_store()
    stats = store.stats()

    by_cat = stats["by_category"]
    by_lc = stats["by_lifecycle"]

    console.print(
        Panel(
            f"[bold]Total memories:[/bold] {stats['total']}\n"
            f"  Active:   {stats['active']}\n"
            f"  Archived: {stats['archived']}\n\n"
            f"[bold]By category:[/bold]\n"
            + "\n".join(f"  {k}: {v}" for k, v in by_cat.items()) +
            f"\n\n[bold]By lifecycle:[/bold]\n"
            + "\n".join(f"  {k}: {v}" for k, v in by_lc.items()),
            title="[bold]Memory Statistics[/bold]",
            border_style="cyan",
        )
    )

@memory_app.command("search")
def memory_search(
    query: Annotated[str, typer.Argument(help="Search query.")],
    limit: Annotated[int, typer.Option("--limit", "-n")] = 10,
    scope: Annotated[str, typer.Option("--scope", "-s")] = "global",
    show_scores: Annotated[bool, typer.Option("--scores")] = False,
) -> None:
    """Semantic search across memories."""
    store = _get_store()

    with console.status(f"[bold green]Searching for:[/bold green] {query}"):
        results = store.search(query, scope=scope, max_results=limit)

    if not results:
        console.print(f"[yellow]No memories found for:[/yellow] {query}")
        return

    table = Table(
        title=f"Search: '{query}' ({len(results)} results)",
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("Rank", width=4)
    table.add_column("ID", style="dim", width=16)
    table.add_column("Category", width=10)
    table.add_column("Tags", width=20)
    if show_scores:
        table.add_column("Score", width=6)
    table.add_column("Summary")

    for result in results:
        m = result.memory
        row = [
            str(result.rank),
            m.id,
            m.category.value,
            ", ".join(m.tags[:3]) if m.tags else "[dim]—[/dim]",
        ]
        if show_scores:
            row.append(f"{result.relevance_score:.3f}")
        row.append(m.display_summary())
        table.add_row(*row)

    console.print(table)
    console.print(f"[dim]Match type: {results[0].match_type if results else '—'}[/dim]")


# Session commands

session_app = typer.Typer(help="Session management.")
app.add_typer(session_app, name="session")


@session_app.command("start")
def session_start() -> None:
    """Start a new session (temporary memory scope)."""
    session_id = create_new_session()
    console.print(f"[green]✓[/green] Started session: [bold]{session_id}[/bold]")


@session_app.command("info")
def session_info() -> None:
    """Show current active session."""
    session = get_active_session()
    if session:
        console.print(f"[bold]Active session:[/bold] {session}")
    else:
        console.print("[yellow]No active session.[/yellow]")


@session_app.command("stop")
def session_stop() -> None:
    """Stop current active session."""
    lace_home = get_lace_home()
    session_file = lace_home / "sessions" / "active"
    if session_file.exists():
        session_file.unlink()
        console.print("[green]✓[/green] Stopped active session.")
    else:
        console.print("[yellow]No active session to stop.[/yellow]")



# ── project commands ───────────────────────────────────────────────────────────

@project_app.command("create")
def project_create(
    name: Annotated[str, typer.Argument(help="Project name.")],
    description: Annotated[
        Optional[str],
        typer.Option("--description", "-d", help="Project description."),
    ] = None,
) -> None:
    """Create a new project."""
    lace_home = get_lace_home()
    created = create_project(name, description, lace_home)

    if created:
        console.print(
            Panel(
                f"[bold green]✓ Project created[/bold green]\n\n"
                f"Name:        [bold]{name}[/bold]\n"
                f"Scope:       project:{name}\n"
                f"Description: {description or '[dim]none[/dim]'}\n\n"
                f"[dim]Add project-specific memories with:[/dim]\n"
                f"  lace memory add \"...\" --scope=project:{name}",
                title="[bold]Project Created[/bold]",
                border_style="green",
            )
        )
    else:
        console.print(f"[yellow]Project [bold]{name}[/bold] already exists.[/yellow]")


@project_app.command("list")
def project_list() -> None:
    """List all configured projects."""
    projects = get_projects()

    if not projects:
        console.print("[yellow]No projects found.[/yellow]")
        console.print("[dim]Try: lace project create \"my-api\"[/dim]")
        return

    table = Table(
        title=f"Projects ({len(projects)} total)",
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("Name", style="bold")
    table.add_column("Scope", style="dim")
    table.add_column("Description", width=40)
    table.add_column("Last Used", width=20)

    for project in sorted(projects, key=lambda p: p.get("last_used") or "", reverse=True):
        last_used = project.get("last_used")
        if last_used:
            last_used = last_used.split("T")[0]  # Just date, not time
        else:
            last_used = "[dim]never[/dim]"

        table.add_row(
            project["name"],
            project["scope"],
            project["description"] or "[dim]none[/dim]",
            last_used,
        )

    console.print(table)


@project_app.command("switch")
def project_switch(
    name: Annotated[str, typer.Argument(help="Project name to switch to.")],
) -> None:
    """Switch to a project as active scope."""
    lace_home = get_lace_home()
    projects = get_projects()
    project_names = {p["name"] for p in projects}

    if name not in project_names:
        console.print(f"[red]✗ Project not found:[/red] {name}")
        raise typer.Exit(1)

    set_project_last_used(name, lace_home)
    console.print(f"[green]✓[/green] Switched to project: [bold]{name}[/bold]")


@project_app.command("info")
def project_info() -> None:
    """Show current active project info."""
    active_scope = get_active_scope()
    if active_scope == "global":
        console.print("[bold]Active scope:[/bold] global")
        return

    if active_scope.startswith("session:"):
        session_id = active_scope.removeprefix("session:")
        console.print(f"[bold]Active scope:[/bold] session:{session_id}")
        return

    if active_scope.startswith("project:"):
        project_name = active_scope.removeprefix("project:")
        lace_home = get_lace_home()
        projects = get_projects()
        project = next((p for p in projects if p["name"] == project_name), None)

        if project:
            console.print(
                Panel(
                    f"[bold]Project:[/bold] {project['name']}\n"
                    f"[bold]Scope:[/bold] {project['scope']}\n"
                    f"[bold]Description:[/bold] {project['description'] or '[dim]none[/dim]'}\n"
                    f"[bold]Created:[/bold] {project.get('created_at', '[dim]unknown[/dim]').split('T')[0]}\n"
                    f"[bold]Last Used:[/bold] {project.get('last_used', '[dim]never[/dim]').split('T')[0]}",
                    title="[bold]Active Project[/bold]",
                    border_style="cyan",
                )
            )
        else:
            console.print(f"[bold]Active scope:[/bold] {active_scope}")
        return


@project_app.command("detect")
def project_detect() -> None:
    """Auto-detect current project from working directory."""
    detected = detect_current_project()
    if detected:
        console.print(f"[green]✓ Detected project:[/green] [bold]{detected}[/bold]")
    else:
        console.print("[yellow]No project detected in current directory.[/yellow]")


# ── mcp placeholder ───────────────────────────────────────────────────────────

@mcp_app.command("start")
def mcp_start(
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Enable debug logging."),
    ] = False,
) -> None:
    """Start the LACE MCP server (stdio mode for Antigravity/Cursor)."""
    import asyncio
    import warnings
    import os

    # Suppress all warnings in MCP mode — they corrupt stdio JSON-RPC
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from lace.mcp.server import run_server
    asyncio.run(run_server(debug=debug))



# ── lace ask ──────────────────────────────────────────────────────────────────

@app.command()
def ask(
    query: Annotated[str, typer.Argument(help="Your question.")],
    show_context: Annotated[
        bool,
        typer.Option("--show-context", help="Show retrieved memories before response."),
    ] = False,
    no_memory: Annotated[
        bool,
        typer.Option("--no-memory", help="Skip memory retrieval entirely."),
    ] = False,
    scope: Annotated[
        Optional[str],
        typer.Option("--scope", "-s", help="Override active scope."),
    ] = None,
    max_memories: Annotated[
        int,
        typer.Option("--max-memories", "-m", help="Max memories to inject."),
    ] = 5,
    provider: Annotated[
        Optional[str],
        typer.Option("--provider", "-p", help="Override provider: ollama, openai, anthropic."),
    ] = None,
) -> None:
    """Ask a question with your memory injected automatically."""
    import time
    import warnings
    warnings.filterwarnings("ignore")

    from lace.core.config import load_config, get_lace_home
    from lace.utils.ask import ask as ask_engine

    lace_home = get_lace_home()
    config = load_config(lace_home)

    # Override provider if specified
    if provider:
        config.provider.default = provider

    start_time = time.time()

    # Run the ask engine
    try:
        memories, stream, llm_provider = ask_engine(
            query=query,
            use_memory=not no_memory,
            scope=scope,
            max_memories=max_memories,
            lace_home=lace_home,
            config=config,
        )
    except ValueError as e:
        console.print(f"[red]✗ Configuration error:[/red] {e}")
        raise typer.Exit(1)

    # Show context panel if requested
    if show_context:
        if memories:
            memory_lines = []
            for i, result in enumerate(memories, 1):
                m = result.memory
                memory_lines.append(
                    f"  [{i}] [bold]{m.display_summary()[:60]}[/bold]\n"
                    f"      scope: {m.project_scope} | "
                    f"conf: {m.confidence:.2f} | "
                    f"score: {result.relevance_score:.3f}"
                )

            retrieval_time = int((time.time() - start_time) * 1000)
            console.print(
                Panel(
                    "\n".join(memory_lines) +
                    f"\n\n[dim]Retrieved {len(memories)} memories in {retrieval_time}ms[/dim]",
                    title="[bold cyan]Context Retrieved[/bold cyan]",
                    border_style="cyan",
                )
            )
        else:
            if no_memory:
                console.print(Panel(
                    "[dim]Memory retrieval disabled (--no-memory)[/dim]",
                    title="[bold cyan]Context[/bold cyan]",
                    border_style="dim",
                ))
            else:
                console.print(Panel(
                    "[dim]No relevant memories found for this query.[/dim]",
                    title="[bold cyan]Context Retrieved[/bold cyan]",
                    border_style="dim",
                ))

    # Stream the response
    console.print()
    response_chunks: list[str] = []

    try:
        for chunk in stream:
            print(chunk, end="", flush=True)
            response_chunks.append(chunk)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        return

    # Footer
    total_time = int((time.time() - start_time) * 1000)
    full_response = "".join(response_chunks)
    token_estimate = len(full_response) // 4

    console.print(f"\n\n[dim]Provider: {llm_provider.provider_name} | "
                  f"Model: {llm_provider.model_name} | "
                  f"~{token_estimate} tokens | "
                  f"{total_time}ms total[/dim]")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()