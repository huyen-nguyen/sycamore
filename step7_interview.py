"""
Step 7: Structured persona interview for verification.

Inspired by Park et al. (2023) "Generative Agents" evaluation protocol.
Asks 25 questions across 5 categories per persona and scores responses automatically.

Categories:
  1. Self-knowledge  — profile attributes, role, skills
  2. Memory          — past experiences, tools used, challenges faced
  3. Planning        — how the persona would approach tasks
  4. Reactions       — responses to hypothetical scenarios
  5. Reflections     — opinions, wishes, deeper insights

Scoring per response (automatic):
  - has_citation:     answer contains a participant ID citation (P\d+) or (Persona Profile)
  - abstained:        answer contains an explicit abstention phrase
  - cross_contamination: answer cites a participant from a different persona group

Usage:
    python3 step7_interview.py                        # all personas, anthropic
    LLM_PROVIDER=openai python3 step7_interview.py   # openai
    python3 step7_interview.py --persona "Biologists" # single persona
    python3 step7_interview.py --out results.json     # save to file
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

from llm_provider import get_provider
from step3_retrieve import get_evidence, get_persona_profile, list_personas
from step4_build_prompts import make_evidence_block, make_profile_block, make_system_prompt

MAX_TOKENS = 512
EVIDENCE_K = 6

# ── Interview questions ────────────────────────────────────────────────────────

QUESTIONS = {
    "self_knowledge": [
        "Give a brief introduction of yourself and your role.",
        "What is your primary research or work focus?",
        "How would you describe your programming skill level?",
        "Who is the primary audience for the visualizations you create?",
        "How automated are the data workflows you use day to day?",
    ],
    "memory": [
        "Describe a type of data visualization you have created.",
        "What tools or libraries do you rely on most for visualization?",
        "Describe a challenge you have encountered when making a visualization.",
        "What kinds of biological or genomic data do you typically visualize?",
        "Have you collaborated with others on visualization tasks? Describe what that looks like.",
    ],
    "planning": [
        "Walk me through how you would approach creating a new visualization from scratch.",
        "If you needed to visualize a type of data you had never worked with before, what would your process be?",
        "How do you decide which type of visualization to use for a given dataset?",
        "What steps do you take to clean or prepare data before visualizing it?",
        "How would you go about sharing a visualization with collaborators or in a publication?",
    ],
    "reactions": [
        "How would you react if a drag-and-drop tool could generate publication-ready figures automatically, without any coding?",
        "If a colleague asked you to reproduce a visualization you made six months ago, how would you respond?",
        "How would you react if the visualization library you rely on most was no longer maintained?",
        "If you discovered an error in a figure you had already included in a paper, what would you do?",
        "How would you respond if asked to create an interactive visualization for a general public audience?",
    ],
    "reflections": [
        "What do you wish was different about your current visualization workflow?",
        "What is the most important skill for someone in your role to develop for data visualization?",
        "How has your approach to visualization changed as you have gained experience?",
        "What is the biggest gap between what current tools offer and what you actually need?",
        "If you could design your ideal visualization tool, what would it look like?",
    ],
}

ABSTENTION_PHRASES = [
    "don't have information",
    "not discussed",
    "no information",
    "wasn't discussed",
    "not covered",
    "not mentioned",
    "no evidence",
    "not in",
    "outside",
    "i don't",
    "we don't",
    "not available",
    "not part of",
    "no quotes",
    "not addressed",
]


# ── Core ask function ──────────────────────────────────────────────────────────

def ask(provider, persona_name, question):
    system_prompt = make_system_prompt(persona_name)
    quotes = get_evidence(persona_name, question, k=EVIDENCE_K)
    profile_block = make_profile_block(persona_name)
    evidence_block = make_evidence_block(quotes)
    user_content = f"{profile_block}\n\n{evidence_block}\n\n---\n\nQuestion: {question}"
    answer = provider.complete(system_prompt, [{"role": "user", "content": user_content}], MAX_TOKENS)
    return answer, quotes


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_response(answer, quotes, persona_name):
    profile = get_persona_profile(persona_name)
    own_participants = set(profile["participants"]) if profile else set()

    # Citations now appear in a "Sources: (P4, P8)" line at the end
    citations = set(re.findall(r"P\d+", answer))
    has_citation = bool(citations) or "profile" in answer.lower()
    abstained = any(ph in answer.lower() for ph in ABSTENTION_PHRASES)

    # Cross-contamination: cited a participant not in this persona's group
    cross = citations - own_participants - {""}
    cross_contamination = bool(cross)

    return {
        "has_citation": has_citation,
        "abstained": abstained,
        "cross_contamination": cross_contamination,
        "cross_cited": sorted(cross) if cross else [],
    }


# ── Interview one persona ─────────────────────────────────────────────────────

def interview_persona(provider, persona_name):
    print(f"\n{'='*60}")
    print(f"  PERSONA: {persona_name}")
    print(f"{'='*60}")

    results = {}

    for category, questions in QUESTIONS.items():
        print(f"\n--- {category.replace('_', ' ').title()} ---")
        results[category] = []

        for q in questions:
            print(f"\nQ: {q}")
            answer, quotes = ask(provider, persona_name, q)
            scores = score_response(answer, quotes, persona_name)

            print(f"A: {answer[:300]}{'...' if len(answer) > 300 else ''}")

            flags = []
            if not scores["has_citation"]:
                flags.append("NO CITATION")
            if scores["abstained"]:
                flags.append("ABSTAINED")
            if scores["cross_contamination"]:
                flags.append(f"CROSS-CONTAMINATION: {scores['cross_cited']}")

            if flags:
                print(f"   ⚠  {' | '.join(flags)}")

            results[category].append({
                "question": q,
                "answer": answer,
                "n_quotes": len(quotes),
                **scores,
            })

    return results


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(all_results):
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    total_q = 0
    total_cited = 0
    total_abstained = 0
    total_cross = 0

    for persona, categories in all_results.items():
        p_q = p_cited = p_abs = p_cross = 0
        for items in categories.values():
            for r in items:
                p_q += 1
                if r["has_citation"]:   p_cited += 1
                if r["abstained"]:      p_abs += 1
                if r["cross_contamination"]: p_cross += 1

        total_q      += p_q
        total_cited  += p_cited
        total_abstained += p_abs
        total_cross  += p_cross

        pct_cited = p_cited / p_q * 100 if p_q else 0
        pct_abs   = p_abs   / p_q * 100 if p_q else 0
        print(f"\n  {persona} ({p_q} questions):")
        print(f"    Citations present : {p_cited}/{p_q}  ({pct_cited:.0f}%)")
        print(f"    Abstentions       : {p_abs}/{p_q}  ({pct_abs:.0f}%)")
        if p_cross:
            print(f"    ⚠ Cross-contamination: {p_cross} response(s)")

    print(f"\n  TOTAL ({total_q} questions across all personas):")
    print(f"    Citations present : {total_cited}/{total_q}  ({total_cited/total_q*100:.0f}%)")
    print(f"    Abstentions       : {total_abstained}/{total_q}  ({total_abstained/total_q*100:.0f}%)")
    if total_cross:
        print(f"    ⚠ Cross-contamination: {total_cross} response(s)")
    else:
        print(f"    ✓ No cross-contamination detected")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Structured persona interview (generative-agents style)")
    parser.add_argument("--persona", help="Interview a single persona by name")
    parser.add_argument("--out", help="Save full results to a JSON file")
    args = parser.parse_args()

    provider = get_provider()
    print(f"Provider: {provider.name}")
    print(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")

    personas = [args.persona] if args.persona else list_personas()

    # Validate persona name if given
    if args.persona and args.persona not in list_personas():
        print(f"Error: unknown persona '{args.persona}'. Choose from: {list_personas()}")
        sys.exit(1)

    all_results = {}
    for persona in personas:
        all_results[persona] = interview_persona(provider, persona)

    print_summary(all_results)

    if args.out:
        payload = {
            "provider": provider.name,
            "timestamp": datetime.now().isoformat(),
            "personas": all_results,
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nFull results saved to {args.out}")


if __name__ == "__main__":
    main()
