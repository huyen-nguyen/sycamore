"""Smoke tests for Sycamore-on-top-of-persona-generator.

Requires data/personas.json and data/evidence.json to exist (run step1 and
step2 once before testing). Embeddings are built lazily by step3 on first
retrieval; in offline test environments this can be skipped by stubbing
get_evidence_with_scores.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow tests to import the reference-engine modules at the project root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.stub_provider import StubProvider  # noqa: E402

from step8_evaluators import EVALUATORS  # noqa: E402
from sycamore.evaluator import UngroundedEvaluator  # noqa: E402
from sycamore.evaluator.grounded import find_evaluator_definition  # noqa: E402


def test_evaluators_count_and_distribution():
    """Manuscript distribution: 1 Bio + 2 CB + 2 BIF + 2 SE = 7."""
    assert len(EVALUATORS) == 7
    by_persona = {}
    for ev in EVALUATORS:
        by_persona.setdefault(ev["persona"], 0)
        by_persona[ev["persona"]] += 1
    assert by_persona == {
        "Biologists": 1,
        "Computational Biologists": 2,
        "Bioinformaticians": 2,
        "Software Engineers": 2,
    }


def test_find_evaluator_definition_by_id_and_abbr():
    a = find_evaluator_definition("comp_bio_2")
    b = find_evaluator_definition("CB2")
    assert a is b


def test_ungrounded_evaluator_runs_with_stub():
    provider = StubProvider()
    ev = UngroundedEvaluator(index=1, provider=provider)
    turn = ev.respond("workflow", "Describe your typical workflow.")
    assert turn.response.startswith("[stub:")
    assert ev.identity.condition == "ungrounded"
    assert ev.identity.eid == "ungrounded:U1"


if __name__ == "__main__":
    test_evaluators_count_and_distribution()
    test_find_evaluator_definition_by_id_and_abbr()
    test_ungrounded_evaluator_runs_with_stub()
    print("All smoke tests passed.")
