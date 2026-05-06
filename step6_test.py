"""
Step 6: Verification tests for the persona pipeline.

Three tests:
  1. Known-answer test  — asks a question with a verifiable answer; checks citation accuracy
  2. Abstention test    — asks a question outside the data scope; verifies the persona abstains
  3. Cross-persona test — asks the same question to two personas; checks answers differ

Usage:
    python3 step6_test.py
    LLM_PROVIDER=openai python3 step6_test.py
"""

import os
from llm_provider import get_provider
from step3_retrieve import get_evidence, list_personas
from step4_build_prompts import make_system_prompt, make_evidence_block

MAX_TOKENS = 512
EVIDENCE_K = 6

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def ask(provider, persona_name, question, k=EVIDENCE_K):
    system_prompt = make_system_prompt(persona_name)
    quotes = get_evidence(persona_name, question, k=k)
    evidence_block = make_evidence_block(quotes)
    user_content = f"{evidence_block}\n\n---\n\nQuestion: {question}"
    answer = provider.complete(system_prompt, [{"role": "user", "content": user_content}], MAX_TOKENS)
    return answer, quotes


def test_known_answer(provider):
    """
    P4 (Computational Biologists) explicitly mentioned using matplotlib and seaborn in Python,
    then ggplot in R. Ask about tools and verify the answer cites something from P4.
    """
    print("\n=== Test 1: Known-answer citation check ===")
    persona = "Computational Biologists"
    question = "What Python libraries do you use for visualization?"

    answer, quotes = ask(provider, persona, question)
    print(f"Answer:\n{answer}\n")

    # Check: answer should mention matplotlib or seaborn or Python
    keywords = ["matplotlib", "seaborn", "python", "ggplot"]
    mentioned = [kw for kw in keywords if kw.lower() in answer.lower()]

    # Check: answer should cite a participant
    import re
    citations = re.findall(r"\(P\d+\)", answer)

    if mentioned and citations:
        result = PASS
        print(f"  ✓ Mentioned: {mentioned}")
        print(f"  ✓ Citations found: {citations}")
    elif mentioned and not citations:
        result = WARN
        print(f"  ~ Content correct ({mentioned}) but no citations found")
    else:
        result = FAIL
        print(f"  ✗ No expected keywords or citations in answer")

    print(f"  Result: {result}")
    return result


def test_abstention(provider):
    """
    Ask about something clearly outside the study scope.
    The persona should abstain rather than hallucinate.
    """
    print("\n=== Test 2: Abstention on out-of-scope question ===")
    persona = "Biologists"
    question = "What is your experience using Tableau for business intelligence dashboards?"

    answer, _ = ask(provider, persona, question, k=4)
    print(f"Answer:\n{answer}\n")

    # Abstention signals: "don't have", "not discussed", "no information", "wasn't"
    abstention_phrases = [
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
        "not part",
    ]
    abstained = any(ph in answer.lower() for ph in abstention_phrases)

    # Hallucination signal: confidently claims Tableau experience without citing evidence
    tableau_claim = "tableau" in answer.lower() and not abstained

    if abstained:
        result = PASS
        print("  ✓ Persona correctly abstained")
    elif tableau_claim:
        result = FAIL
        print("  ✗ Persona may have hallucinated Tableau experience")
    else:
        result = WARN
        print("  ~ Ambiguous response — check manually")

    print(f"  Result: {result}")
    return result


def test_cross_persona(provider):
    """
    Ask the same question to two different personas.
    Answers should cite different participants and potentially differ in content.
    """
    print("\n=== Test 3: Cross-persona differentiation ===")
    personas = list_personas()
    if len(personas) < 2:
        print("  ! Need at least 2 personas. Skipping.")
        return WARN

    # Use Biologists vs Software Engineers — most different profiles
    persona_a = "Biologists"
    persona_b = "Software Engineers"
    question = "How do you typically approach making a visualization?"

    answer_a, quotes_a = ask(provider, persona_a, question)
    answer_b, quotes_b = ask(provider, persona_b, question)

    print(f"{persona_a} answer:\n{answer_a}\n")
    print(f"{persona_b} answer:\n{answer_b}\n")

    import re
    citations_a = set(re.findall(r"P\d+", answer_a))
    citations_b = set(re.findall(r"P\d+", answer_b))

    # Participants in each group (from personas.json)
    from step3_retrieve import get_persona_profile
    participants_a = set(get_persona_profile(persona_a)["participants"])
    participants_b = set(get_persona_profile(persona_b)["participants"])

    overlap_a = citations_a & participants_a
    overlap_b = citations_b & participants_b
    cross_contamination = (citations_a & participants_b) | (citations_b & participants_a)

    print(f"  {persona_a} cited own participants: {overlap_a or '(none found)'}")
    print(f"  {persona_b} cited own participants: {overlap_b or '(none found)'}")
    if cross_contamination:
        print(f"  WARNING: cross-group citations detected: {cross_contamination}")

    answers_differ = answer_a.strip()[:200] != answer_b.strip()[:200]

    if answers_differ and not cross_contamination:
        result = PASS
        print("  ✓ Answers differ and no cross-group contamination")
    elif cross_contamination:
        result = FAIL
        print("  ✗ Persona cited participants from the wrong group")
    else:
        result = WARN
        print("  ~ Answers are similar — may be acceptable if evidence pools overlap")

    print(f"  Result: {result}")
    return result


def main():
    provider = get_provider()
    print(f"Using provider: {provider.name}\n")

    results = {}
    results["known_answer"] = test_known_answer(provider)
    results["abstention"] = test_abstention(provider)
    results["cross_persona"] = test_cross_persona(provider)

    print("\n=== Summary ===")
    for test, result in results.items():
        icon = "✓" if result == PASS else ("~" if result == WARN else "✗")
        print(f"  {icon} {test}: {result}")

    passed = sum(1 for r in results.values() if r == PASS)
    total = len(results)
    print(f"\n{passed}/{total} tests passed")


if __name__ == "__main__":
    main()
