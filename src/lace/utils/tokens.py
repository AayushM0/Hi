"""Token counting utilities for context window management.

Keeps prompts within model context limits before sending to LLM.
Uses character-based estimation when tiktoken is not available.
"""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Estimate token count for a string.

    Uses the rough heuristic: 1 token ≈ 4 characters for English text.
    Good enough for context window safety checks.

    Args:
        text: Input text to estimate.

    Returns:
        Estimated token count.
    """
    return max(1, len(text) // 4)


def truncate_to_token_limit(
    text: str,
    max_tokens: int,
    truncation_marker: str = "\n\n[... truncated for context window ...]",
) -> str:
    """Truncate text to fit within a token limit.

    Args:
        text: Text to potentially truncate.
        max_tokens: Maximum allowed tokens.
        truncation_marker: Appended when truncation occurs.

    Returns:
        Original text if within limit, truncated version otherwise.
    """
    if estimate_tokens(text) <= max_tokens:
        return text

    max_chars = max_tokens * 4
    marker_chars = len(truncation_marker)
    truncated = text[: max_chars - marker_chars]
    return truncated + truncation_marker


def fits_in_context(
    system_prompt: str,
    memories_text: str,
    user_query: str,
    context_window: int,
    reserved_for_response: int = 1024,
) -> bool:
    """Check if all components fit within the context window.

    Args:
        system_prompt: The identity/system prompt.
        memories_text: The formatted memory context.
        user_query: The user's question.
        context_window: Model's max context window in tokens.
        reserved_for_response: Tokens to reserve for the model's response.

    Returns:
        True if everything fits.
    """
    total = (
        estimate_tokens(system_prompt) +
        estimate_tokens(memories_text) +
        estimate_tokens(user_query) +
        reserved_for_response
    )
    return total <= context_window