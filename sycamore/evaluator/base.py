"""Common interface shared by ungrounded and grounded evaluators.

Both implementations are passed an LLMProvider from `llm_provider.get_provider()`
(the persona-generator's provider abstraction) and answer one question at a
time. The protocol runner doesn't need to know which kind it's holding.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvaluatorIdentity:
    """Identifies the evaluator for logging and analysis."""
    eid: str                          # e.g., "grounded:CB1" or "ungrounded:U3"
    condition: str                    # "grounded" | "ungrounded"
    persona_group: Optional[str]      # "Computational Biologists", etc.; None for ungrounded
    instance_id: Optional[str]        # e.g., "comp_bio_2"; None for ungrounded
    abbr: Optional[str]               # e.g., "CB2"; None for ungrounded
    label: str                        # human-readable


@dataclass
class EvaluatorTurn:
    """One protocol turn produced by an evaluator."""
    phase: str                                      # e.g., "workflow", "exploration:Text:q1"
    prompt: str
    response: str
    evidence: list[dict] = field(default_factory=list)  # {participant, codes, quotation}
    abstained: bool = False
    note: str = ""


class Evaluator(ABC):
    identity: EvaluatorIdentity

    @abstractmethod
    def respond(self, phase: str, prompt: str, *, note: str = "") -> EvaluatorTurn:
        """Produce one turn of output. Implementations append to `transcript`."""
        ...

    transcript: list[EvaluatorTurn]
