"""Ungrounded evaluator.

Sycamore's UNGROUNDED condition (Section 3.3.2 of the manuscript): synthetic
personas instantiated from generic LLM priors. Same protocol as the grounded
condition, but no retrieval, no profile block, no validation. The only
shared piece is the LLM provider abstraction so that generation costs and
behaviour are otherwise comparable.
"""
from __future__ import annotations

from llm_provider import LLMProvider

from .base import Evaluator, EvaluatorIdentity, EvaluatorTurn

UNGROUNDED_SYSTEM_PROMPT = """\
You're a genomics researcher participating in a user study of a genomics
data visualization retrieval system. Respond in first person, naturally,
as a researcher would in a think-aloud interview.

About you:
- You work with genome-mapped data and the visualizations that go with it
- You have some familiarity with visualization tools, but the level of
  expertise is yours to choose and stay consistent with
- You speak from direct experience, not as a list of credentials

Be concrete: describe specific tools, file formats, chart types, friction
points, and audiences when they're relevant to the question. Don't enumerate
your skills. Don't describe yourself as a "synthetic persona" or "AI";
you're just a researcher answering questions. If asked to rank items,
give numeric scores explicitly. If asked for JSON, return only JSON."""

MAX_TOKENS = 1024


class UngroundedEvaluator(Evaluator):
    """One synthetic evaluator with no grounding artefacts."""

    def __init__(
        self,
        index: int,
        provider: LLMProvider,
        *,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        self.index = index
        self.provider = provider
        self.max_tokens = max_tokens
        self.identity = EvaluatorIdentity(
            eid=f"ungrounded:U{index}",
            condition="ungrounded",
            persona_group=None,
            instance_id=None,
            abbr=None,
            label=f"Ungrounded evaluator U{index}",
        )
        self.history: list[dict] = []
        self.transcript: list[EvaluatorTurn] = []

    def respond(self, phase: str, prompt: str, *, note: str = "") -> EvaluatorTurn:
        api_messages = list(self.history) + [{"role": "user", "content": prompt}]
        answer = self.provider.complete(
            UNGROUNDED_SYSTEM_PROMPT, api_messages, self.max_tokens
        )
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": answer})
        turn = EvaluatorTurn(
            phase=phase,
            prompt=prompt,
            response=answer.strip(),
            evidence=[],
            abstained=False,
            note=note,
        )
        self.transcript.append(turn)
        return turn
