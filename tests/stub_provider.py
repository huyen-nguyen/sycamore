"""Stub LLMProvider for plumbing tests.

Implements the same interface as `llm_provider.LLMProvider` so it can be
swapped in for AnthropicProvider / OpenAIProvider in tests. Returns
deterministic placeholder text and minimal Pydantic objects.
"""
from __future__ import annotations

import hashlib
from typing import Type, TypeVar

from pydantic import BaseModel

from llm_provider import LLMProvider

T = TypeVar("T", bound=BaseModel)


class StubProvider(LLMProvider):
    @property
    def name(self) -> str:
        return "Stub"

    def complete(self, system_prompt: str, messages: list, max_tokens: int = 1024) -> str:
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        digest = hashlib.sha256((system_prompt + "||" + last_user).encode()).hexdigest()[:8]
        # If the prompt asked for JSON, return parseable JSON the runner can consume.
        if "Return ONLY a JSON object" in last_user:
            if "ranking" in last_user:
                return (
                    '{"ranking": {"Text": 3, "Image": 2, "Spec": 1}, '
                    '"rationale": "Stub rationale (not real)."}'
                )
            return '{"queries": ["stub query A", "stub query B", "stub query C"]}'
        snippet = last_user.strip().splitlines()[0] if last_user else "(empty)"
        return (
            f"[stub:{digest}] In response to: {snippet[:160]}\n"
            "This is a deterministic placeholder produced by StubProvider."
        )

    def complete_structured(
        self,
        system_prompt: str,
        messages: list,
        response_model: Type[T],
        max_tokens: int = 256,
    ) -> T:
        # Try to satisfy the validator's "ValidationResult" model (relevant_indices).
        if hasattr(response_model, "model_fields") and "relevant_indices" in response_model.model_fields:
            return response_model(relevant_indices=[1, 2, 3])
        # Fallback: instantiate with empty / default values where possible.
        try:
            return response_model()
        except Exception:
            # Last resort: pass a hash digest in a string field if the model has one
            return response_model.model_validate({})  # type: ignore
