"""
LLM provider abstraction — swap between Anthropic and OpenAI via environment variable.

Usage:
    Set LLM_PROVIDER=anthropic (default) or LLM_PROVIDER=openai

    from llm_provider import get_provider
    provider = get_provider()
    answer = provider.complete(system_prompt, messages, max_tokens=1024)

Messages format (same for both providers):
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
"""

import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv
load_dotenv()
from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


# --- Default models ---
# ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-6"
OPENAI_DEFAULT_MODEL = "gpt-5-mini"

# --- Fast models for cheap tasks (validation, filtering) ---
# ANTHROPIC_FAST_MODEL = "claude-haiku-4-5-20251001"
OPENAI_FAST_MODEL = "gpt-5-nano"


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, messages: list, max_tokens: int = 1024) -> str:
        """
        Send a chat completion request.

        Args:
            system_prompt: The system prompt (persona profile + grounding rules).
            messages: List of {"role": ..., "content": ...} dicts (conversation history).
            max_tokens: Maximum tokens to generate.

        Returns:
            The assistant's response text.
        """

    @abstractmethod
    def complete_structured(
        self,
        system_prompt: str,
        messages: list,
        response_model: Type[T],
        max_tokens: int = 256,
    ) -> T:
        """
        Like complete(), but returns a validated Pydantic model instance.
        Uses tool use (Anthropic) or structured outputs (OpenAI) under the hood.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name for display."""


class AnthropicProvider(LLMProvider):
    """
    Uses the Anthropic Messages API.
    Applies prompt caching to the system prompt (stable per session).
    """

    def __init__(self, model: str = None):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("ANTHROPIC_MODEL", ANTHROPIC_DEFAULT_MODEL)

    @property
    def name(self):
        return f"Anthropic ({self.model})"

    def complete(self, system_prompt: str, messages: list, max_tokens: int = 1024) -> str:
        # System prompt as a cached block — stable for the whole session
        system = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return next((b.text for b in response.content if b.type == "text"), "")

    def complete_structured(self, system_prompt, messages, response_model, max_tokens=256):
        tool_name = response_model.__name__
        system = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=[{
                "name": tool_name,
                "description": f"Return a structured {tool_name}",
                "input_schema": response_model.model_json_schema(),
            }],
            tool_choice={"type": "tool", "name": tool_name},
        )
        tool_block = next(b for b in response.content if b.type == "tool_use")
        return response_model(**tool_block.input)

    def complete_stream(self, system_prompt: str, messages: list, max_tokens: int = 1024):
        """Yield text tokens as the model generates them."""
        system = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text

    def complete_with_usage(self, system_prompt: str, messages: list, max_tokens: int = 1024):
        """Like complete() but also returns cache usage stats."""
        system = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        u = response.usage
        usage = {
            "cache_write": u.cache_creation_input_tokens,
            "cache_read": u.cache_read_input_tokens,
            "uncached": u.input_tokens,
        }
        return text, usage


class OpenAIProvider(LLMProvider):
    """
    Uses the OpenAI Chat Completions API.
    System prompt is passed as a system message (no native prompt caching).
    """

    def __init__(self, model: str = None):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model or os.environ.get("OPENAI_MODEL", OPENAI_DEFAULT_MODEL)

    @property
    def name(self):
        return f"OpenAI ({self.model})"

    def _tokens_kwarg(self, max_tokens: int) -> dict:
        # Some models don't support either parameter — return empty dict to use model default
        NO_TOKEN_LIMIT_MODELS = {"gpt-5-mini", "gpt-5-nano"}
        if self.model in NO_TOKEN_LIMIT_MODELS:
            return {}
        if self.model.startswith("gpt-5") or self.model.startswith("o1") or self.model.startswith("o3"):
            return {"max_completion_tokens": max_tokens}
        return {"max_tokens": max_tokens}

    def complete(self, system_prompt: str, messages: list, max_tokens: int = 1024) -> str:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response = self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            **self._tokens_kwarg(max_tokens),
        )
        return response.choices[0].message.content or ""

    def complete_structured(self, system_prompt, messages, response_model, max_tokens=256):
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=full_messages,
            response_format=response_model,
            **self._tokens_kwarg(max_tokens),
        )
        return response.choices[0].message.parsed

    def complete_stream(self, system_prompt: str, messages: list, max_tokens: int = 1024):
        """Yield text tokens as the model generates them."""
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            stream=True,
            **self._tokens_kwarg(max_tokens),
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def complete_with_usage(self, system_prompt: str, messages: list, max_tokens: int = 1024):
        """Like complete() but returns token usage (no cache stats for OpenAI)."""
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response = self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            **self._tokens_kwarg(max_tokens),
        )
        text = response.choices[0].message.content or ""
        u = response.usage
        usage = {
            "cache_write": 0,
            "cache_read": 0,
            "uncached": u.prompt_tokens if u else 0,
        }
        return text, usage


def get_provider(provider_name: str = None) -> LLMProvider:
    """
    Return a provider instance based on LLM_PROVIDER env var (or explicit argument).
    Defaults to Anthropic.

    LLM_PROVIDER=anthropic  →  AnthropicProvider
    LLM_PROVIDER=openai     →  OpenAIProvider
    """
    name = (provider_name or os.environ.get("LLM_PROVIDER", "openai")).lower()
    if name == "anthropic":
        return AnthropicProvider()
    elif name == "openai":
        return OpenAIProvider()
    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{name}'. Choose 'anthropic' or 'openai'.")


def get_fast_provider(provider_name: str = None) -> LLMProvider:
    """Like get_provider() but uses a smaller/faster model for quick tasks like validation."""
    name = (provider_name or os.environ.get("LLM_PROVIDER", "openai")).lower()
    if name == "anthropic":
        model = os.environ.get("ANTHROPIC_FAST_MODEL", ANTHROPIC_FAST_MODEL)
        return AnthropicProvider(model=model)
    elif name == "openai":
        model = os.environ.get("OPENAI_FAST_MODEL", OPENAI_FAST_MODEL)
        return OpenAIProvider(model=model)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{name}'. Choose 'anthropic' or 'openai'.")
