"""LLM provider abstraction for lace ask.

Supports:
  - Ollama (local, free, default)
  - OpenAI (paid, cloud)
  - Anthropic (paid, cloud)

Each provider implements stream_response() which yields
text chunks as the model generates them.
"""

from __future__ import annotations

from typing import Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from lace.core.config import LaceConfig


# ── Base interface ─────────────────────────────────────────────────────────────

class LLMProvider:
    """Abstract base for LLM providers."""

    def stream_response(
        self,
        system_prompt: str,
        user_message: str,
    ) -> Iterator[str]:
        """Stream response chunks from the LLM.

        Args:
            system_prompt: The identity/context prompt.
            user_message: The user's question with memory context injected.

        Yields:
            Text chunks as they are generated.
        """
        raise NotImplementedError

    def complete(
        self,
        system_prompt: str,
        user_message: str,
    ) -> str:
        """Get a complete (non-streaming) response.

        Default implementation collects stream chunks.
        """
        return "".join(self.stream_response(system_prompt, user_message))

    @property
    def model_name(self) -> str:
        raise NotImplementedError

    @property
    def provider_name(self) -> str:
        raise NotImplementedError


# ── Ollama provider ───────────────────────────────────────────────────────────

class OllamaProvider(LLMProvider):
    """Local Ollama provider — free, private, no API key needed."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.2",
        temperature: float = 0.7,
    ) -> None:
        self.host = host
        self.model = model
        self.temperature = temperature

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def provider_name(self) -> str:
        return "ollama"

    def stream_response(
        self,
        system_prompt: str,
        user_message: str,
    ) -> Iterator[str]:
        """Stream response from local Ollama instance."""
        import ollama

        client = ollama.Client(host=self.host)

        try:
            stream = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                options={"temperature": self.temperature},
                stream=True,
            )

            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content

        except Exception as e:
            error_msg = str(e)
            if "connection refused" in error_msg.lower():
                yield "\n[Error: Ollama is not running. Start it with: ollama serve]\n"
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                yield f"\n[Error: Model '{self.model}' not found. Pull it with: ollama pull {self.model}]\n"
            else:
                yield f"\n[Error communicating with Ollama: {error_msg}]\n"

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            import ollama
            client = ollama.Client(host=self.host)
            models = client.list()
            model_names = [m.model for m in models.models]
            return any(self.model in name for name in model_names)
        except Exception:
            return False


# ── OpenAI provider ───────────────────────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    """OpenAI provider — requires OPENAI_API_KEY environment variable."""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.7,
    ) -> None:
        self.model = model
        self.temperature = temperature

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def provider_name(self) -> str:
        return "openai"

    def stream_response(
        self,
        system_prompt: str,
        user_message: str,
    ) -> Iterator[str]:
        """Stream response from OpenAI API."""
        try:
            from openai import OpenAI
            client = OpenAI()

            stream = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                temperature=self.temperature,
                stream=True,
            )

            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        except ImportError:
            yield "\n[Error: openai package not installed. Run: uv add openai]\n"
        except Exception as e:
            yield f"\n[Error: {e}]\n"


# ── Anthropic provider ────────────────────────────────────────────────────────

class AnthropicProvider(LLMProvider):
    """Anthropic provider — requires ANTHROPIC_API_KEY environment variable."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
    ) -> None:
        self.model = model
        self.temperature = temperature

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def stream_response(
        self,
        system_prompt: str,
        user_message: str,
    ) -> Iterator[str]:
        """Stream response from Anthropic API."""
        try:
            import anthropic
            client = anthropic.Anthropic()

            with client.messages.stream(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                for text in stream.text_stream:
                    yield text

        except ImportError:
            yield "\n[Error: anthropic package not installed. Run: uv add anthropic]\n"
        except Exception as e:
            yield f"\n[Error: {e}]\n"


# ── Provider factory ──────────────────────────────────────────────────────────

def get_provider(config: "LaceConfig") -> LLMProvider:
    """Return the configured LLM provider.

    Reads from config.provider.default to determine which provider to use.
    """
    provider_name = config.provider.default

    if provider_name == "ollama":
        cfg = config.provider.ollama
        return OllamaProvider(
            host=cfg.host,
            model=cfg.model,
            temperature=cfg.temperature,
        )
    elif provider_name == "openai":
        cfg = config.provider.openai
        return OpenAIProvider(
            model=cfg.model,
            temperature=cfg.temperature,
        )
    elif provider_name == "anthropic":
        cfg = config.provider.anthropic
        return AnthropicProvider(
            model=cfg.model,
            temperature=cfg.temperature,
        )
    else:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Valid options: ollama, openai, anthropic"
        )