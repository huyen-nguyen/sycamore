"""Build the fleet of evaluators for a condition.

GROUNDED:    one evaluator per entry in step8_evaluators.EVALUATORS
             (7 total, distribution per the Sycamore manuscript).
UNGROUNDED:  N generic evaluators (default 7, to match the published
             Geranium user study sample size).
"""
from __future__ import annotations

from llm_provider import LLMProvider

from step8_evaluators import EVALUATORS

from ..evaluator import Evaluator, GroundedEvaluator, UngroundedEvaluator


def build_evaluators(
    condition: str,
    provider: LLMProvider,
    *,
    fast_provider: LLMProvider | None = None,
    n_ungrounded: int = 7,
    only_abbr: list[str] | None = None,
) -> list[Evaluator]:
    if condition == "ungrounded":
        return [
            UngroundedEvaluator(index=i + 1, provider=provider)
            for i in range(n_ungrounded)
        ]
    if condition == "grounded":
        evals: list[Evaluator] = []
        for definition in EVALUATORS:
            if only_abbr and definition["abbr"] not in only_abbr:
                continue
            evals.append(
                GroundedEvaluator(
                    definition=definition,
                    provider=provider,
                    fast_provider=fast_provider,
                )
            )
        return evals
    raise ValueError(f"unknown condition: {condition!r}")
