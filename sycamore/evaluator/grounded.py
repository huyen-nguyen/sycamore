"""Grounded evaluator.

Wraps the persona-generator pipeline (step3 retrieval -> step4b validation ->
step4 prompt assembly -> generation via llm_provider) without modification.
This is the same flow used by `server.py:/api/chat` in the reference repo,
applied to one question at a time.

The evaluator preserves a conversation history across turns so that the
hands-on exploration phase reads like one continuous session, matching how
the published Geranium user study was conducted with each participant.

Phase classification
--------------------
Phases are either "recall" (the persona describes their own past experience —
evidence is load-bearing) or "reaction" (the persona responds to something new,
like a tool description or search results — evidence provides contextual backing
for experiential claims but is NOT a gate that blocks the response).

  recall  : workflow
  reaction: tool_description, gallery, exploration:*, modality_ranking, closing

For recall phases the existing hard-abstention logic is preserved.
For reaction phases:
  - A short background query is used for retrieval instead of the full prompt,
    so relevant past-experience excerpts surface at higher similarity.
  - The MIN_SIMILARITY threshold is bypassed.
  - Evidence is injected as background context when available; if nothing
    survives validation the persona still generates from profile alone.
  - REACTION_GROUNDING_RULES apply: react freely, but cite evidence for any
    specific experiential claim.
"""
from __future__ import annotations

import re

from llm_provider import LLMProvider, get_fast_provider

from step3_retrieve import (
    ABSTENTION_RESPONSE,
    MIN_SIMILARITY,
    get_evidence_with_scores,
)
from step4_build_prompts import make_evidence_block
from step4b_validate import validate_evidence
from step8_evaluators import (
    EVALUATORS,
    make_evaluator_prompt,
    make_evaluator_profile_block,
)

from .base import Evaluator, EvaluatorIdentity, EvaluatorTurn

# How many evidence items to retrieve per turn (matches server.py default).
EVIDENCE_K = 5
# Generation cap (matches server.py).
MAX_TOKENS = 1024

# Phases where the persona is recalling their own past experience.
# All other phases are treated as reactions to new content.
RECALL_PHASES = frozenset({"workflow"})


def find_evaluator_definition(eid_or_abbr: str) -> dict:
    """Look up an entry in step8_evaluators.EVALUATORS by id or abbr."""
    for ev in EVALUATORS:
        if ev["id"] == eid_or_abbr or ev["abbr"] == eid_or_abbr:
            return ev
    raise KeyError(
        f"Unknown evaluator '{eid_or_abbr}'. "
        f"Available ids: {[ev['id'] for ev in EVALUATORS]}"
    )


def _background_query(phase: str, prompt: str) -> str:
    """
    Return a short, general retrieval query for reaction phases.

    The full prompt contains Geranium-specific language that has low cosine
    similarity with the interview evidence pool.  A shorter, experience-focused
    query surfaces relevant background excerpts at higher similarity.
    """
    if phase == "tool_description":
        return "visualization tool evaluation, usability, what tools I use, software criteria"
    if phase == "gallery":
        return "browsing visualizations, chart types, what catches my eye, visual exploration"
    if phase.startswith("exploration:"):
        # Extract the user's query from "QUERY: {query}" in the reaction prompt.
        m = re.search(r"QUERY:\s*(.+?)(?:\n|$)", prompt)
        if m:
            return m.group(1).strip()
    if phase == "modality_ranking":
        return "comparing data modalities, text image spec preference, data format"
    if phase == "closing":
        return "overall tool assessment, blockers to adoption, what I wish existed"
    return prompt  # fallback: use the full prompt as-is


class GroundedEvaluator(Evaluator):
    """One synthetic evaluator backed by the persona-generator pipeline."""

    def __init__(
        self,
        definition: dict,
        provider: LLMProvider,
        *,
        fast_provider: LLMProvider | None = None,
        evidence_k: int = EVIDENCE_K,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        self.definition = definition
        self.provider = provider
        self.fast_provider = fast_provider or get_fast_provider()
        self.evidence_k = evidence_k
        self.max_tokens = max_tokens

        self.identity = EvaluatorIdentity(
            eid=f"grounded:{definition['abbr']}",
            condition="grounded",
            persona_group=definition["persona"],
            instance_id=definition["id"],
            abbr=definition["abbr"],
            label=definition["label"],
        )

        # Two system prompts — one per mode — cached at construction.
        self.system_prompt_recall: str = make_evaluator_prompt(definition, mode="recall")
        self.system_prompt_reaction: str = make_evaluator_prompt(definition, mode="reaction")
        self.profile_block: str = make_evaluator_profile_block(definition)

        # Conversation history (excluding the system prompt). Each user turn
        # carries the evidence block alongside the question, matching server.py.
        self.history: list[dict] = []
        self.transcript: list[EvaluatorTurn] = []

    def respond(self, phase: str, prompt: str, *, note: str = "") -> EvaluatorTurn:
        is_reaction = phase not in RECALL_PHASES

        # Choose retrieval query and system prompt based on phase type.
        retrieval_query = _background_query(phase, prompt) if is_reaction else prompt
        system_prompt = self.system_prompt_reaction if is_reaction else self.system_prompt_recall

        # Step 3: retrieve
        quotes, top_sim = get_evidence_with_scores(
            self.definition["persona"], retrieval_query, k=self.evidence_k
        )

        # Hard abstention for recall phases only.
        if not is_reaction and (not quotes or top_sim < MIN_SIMILARITY):
            return self._record(
                phase, prompt, ABSTENTION_RESPONSE, [], abstained=True, note=note,
            )

        # Step 4b: validate using the same focused retrieval query so the
        # validator judges relevance against the experience topic, not the
        # Geranium-specific prompt text.
        if quotes:
            quotes = validate_evidence(self.fast_provider, retrieval_query, quotes)

        # For recall phases, empty validation → abstain.
        # For reaction phases, proceed without evidence (profile-only generation).
        if not quotes and not is_reaction:
            return self._record(
                phase, prompt, ABSTENTION_RESPONSE, [], abstained=True, note=note,
            )

        # Step 4: assemble user content with profile + evidence (if any).
        evidence_block = make_evidence_block(quotes, reaction_mode=is_reaction)
        if evidence_block:
            user_content = (
                f"{self.profile_block}\n\n{evidence_block}\n\n---\n\n"
                f"Question: {prompt}"
            )
        else:
            user_content = f"{self.profile_block}\n\n---\n\nQuestion: {prompt}"

        # Build api messages: prior history (already evidence-augmented) +
        # this turn's evidence-augmented question. This matches server.py.
        api_messages = list(self.history) + [
            {"role": "user", "content": user_content}
        ]

        # Generate
        answer = self.provider.complete(
            system_prompt, api_messages, self.max_tokens
        )

        # Update conversation history.
        self.history.append({"role": "user", "content": user_content})
        self.history.append({"role": "assistant", "content": answer})

        evidence_recorded = [
            {
                "participant": q["participant"],
                "codes": q.get("codes", []),
                "quotation": q.get("quotation", ""),
            }
            for q in quotes
        ]
        return self._record(
            phase, prompt, answer, evidence_recorded, abstained=False, note=note,
        )

    def _record(self, phase, prompt, response, evidence, *, abstained, note):
        turn = EvaluatorTurn(
            phase=phase,
            prompt=prompt,
            response=response.strip(),
            evidence=evidence,
            abstained=abstained,
            note=note,
        )
        self.transcript.append(turn)
        return turn
