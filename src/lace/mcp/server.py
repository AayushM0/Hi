"""LACE MCP Server.

Exposes LACE memory operations as MCP tools and resources.
Communicates via stdio (JSON-RPC) — no HTTP server needed.

Usage:
    lace mcp start         # Started by MCP client (Antigravity/Cursor)
    lace mcp start --debug # With debug logging
"""

from __future__ import annotations

import asyncio
import logging
import sys
import warnings

# Suppress noisy warnings before any imports
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from lace.mcp.tools import (
    search_memory,
    get_project_context,
    remember,
    list_memories,
    forget_memory,
)
from lace.mcp.resources import (
    get_patterns_resource,
    get_decisions_resource,
    get_project_context_resource,
    get_debug_log_resource,
)


# ── Server setup ──────────────────────────────────────────────────────────────

def create_server() -> Server:
    """Create and configure the LACE MCP server."""
    server = Server("lace")

    # ── Tool definitions ──────────────────────────────────────────────────────

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="search_memory",
                description=(
                    "Search your knowledge base for memories relevant to a query. "
                    "Use when the user asks about something they might have encountered, "
                    "decided, or learned before. Returns memories ranked by relevance."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for (natural language).",
                        },
                        "scope": {
                            "type": "string",
                            "description": "auto, global, or project:<name>",
                            "default": "auto",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Max memories to return (default 5).",
                            "default": 5,
                        },
                        "category": {
                            "type": "string",
                            "description": "all, pattern, decision, debug, reference, preference",
                            "default": "all",
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="get_project_context",
                description=(
                    "Get the current project's identity, preferences, rules, and conventions. "
                    "Use at the start of a conversation to understand the user's context, "
                    "coding standards, and project-specific decisions."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            types.Tool(
                name="remember",
                description=(
                    "Store a new piece of knowledge for future retrieval. "
                    "Use when the user discovers something worth remembering: "
                    "a pattern, a decision rationale, or a debugging insight."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The knowledge to store (be specific and actionable).",
                        },
                        "category": {
                            "type": "string",
                            "description": "pattern, decision, debug, reference, preference",
                            "default": "pattern",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags for filtering.",
                            "default": [],
                        },
                        "scope": {
                            "type": "string",
                            "description": "auto, global, or project:<name>",
                            "default": "auto",
                        },
                    },
                    "required": ["content"],
                },
            ),
            types.Tool(
                name="list_memories",
                description=(
                    "List stored memories with optional filtering. "
                    "Use when the user wants to browse what the system remembers. "
                    "For searching by content, use search_memory instead."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "all, pattern, decision, debug, reference, preference",
                            "default": "all",
                        },
                        "scope": {
                            "type": "string",
                            "description": "auto, global, or project:<name>",
                            "default": "auto",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max memories to return.",
                            "default": 20,
                        },
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="forget_memory",
                description=(
                    "Archive a memory so it no longer appears in search results. "
                    "The memory is NOT deleted — just deprioritized. "
                    "Use when the user says a memory is outdated or wrong."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "The ID of the memory to archive (e.g. mem_abc123).",
                        },
                    },
                    "required": ["memory_id"],
                },
            ),
        ]

    # ── Tool call handler ─────────────────────────────────────────────────────

    @server.call_tool()
    async def call_tool(
        name: str,
        arguments: dict,
    ) -> list[types.TextContent]:
        import json

        try:
            if name == "search_memory":
                result = await search_memory(
                    query=arguments["query"],
                    scope=arguments.get("scope", "auto"),
                    max_results=arguments.get("max_results", 5),
                    category=arguments.get("category", "all"),
                )
            elif name == "get_project_context":
                result = await get_project_context()
            elif name == "remember":
                result = await remember(
                    content=arguments["content"],
                    category=arguments.get("category", "pattern"),
                    tags=arguments.get("tags", []),
                    scope=arguments.get("scope", "auto"),
                )
            elif name == "list_memories":
                result = await list_memories(
                    category=arguments.get("category", "all"),
                    scope=arguments.get("scope", "auto"),
                    limit=arguments.get("limit", 20),
                    lifecycle=arguments.get("lifecycle", "all"),
                )
            elif name == "forget_memory":
                result = await forget_memory(
                    memory_id=arguments["memory_id"],
                )
            else:
                result = {"error": f"Unknown tool: {name}"}

        except Exception as e:
            result = {"error": str(e), "tool": name}

        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str),
        )]

    # ── Resource definitions ──────────────────────────────────────────────────

    @server.list_resources()
    async def list_resources() -> list[types.Resource]:
        return [
            types.Resource(
                uri="memory://patterns",
                name="Stored Patterns",
                description="All stored coding patterns and best practices.",
                mimeType="text/markdown",
            ),
            types.Resource(
                uri="memory://decisions",
                name="Architectural Decisions",
                description="All stored architectural decisions and rationale.",
                mimeType="text/markdown",
            ),
            types.Resource(
                uri="memory://project-context",
                name="Project Context",
                description="Current project identity, preferences, and rules.",
                mimeType="text/markdown",
            ),
            types.Resource(
                uri="memory://debug-log",
                name="Debug Log",
                description="Past debugging insights and solutions.",
                mimeType="text/markdown",
            ),
        ]

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        uri_str = str(uri)
        if uri_str == "memory://patterns":
            return await get_patterns_resource()
        elif uri_str == "memory://decisions":
            return await get_decisions_resource()
        elif uri_str == "memory://project-context":
            return await get_project_context_resource()
        elif uri_str == "memory://debug-log":
            return await get_debug_log_resource()
        else:
            return f"Unknown resource: {uri_str}"

    return server


# ── Entry point ───────────────────────────────────────────────────────────────

async def run_server(debug: bool = False) -> None:
    """Run the MCP server over stdio."""
    if debug:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    server = create_server()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )