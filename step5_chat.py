"""
Step 5: Chat loop — interview a persona.

Usage:
    python3 step5_chat.py

Provider selection (default: anthropic):
    LLM_PROVIDER=anthropic python3 step5_chat.py
    LLM_PROVIDER=openai    python3 step5_chat.py

Model override (optional):
    ANTHROPIC_MODEL=claude-opus-4-7 python3 step5_chat.py
    OPENAI_MODEL=gpt-5-mini        python3 step5_chat.py

Requires:
    - ANTHROPIC_API_KEY or OPENAI_API_KEY depending on provider
    - data/personas.json (from step 1)
    - data/evidence.json (from step 2)
"""

import os
from llm_provider import get_provider
from step3_retrieve import (
    get_evidence_with_scores, list_personas, MIN_SIMILARITY, ABSTENTION_RESPONSE
)
from step4_build_prompts import make_system_prompt, make_evidence_block, make_profile_block
from step4b_validate import validate_evidence

MAX_TOKENS = 1024
EVIDENCE_K = 8


def choose_persona():
    personas = list_personas()
    print("\nAvailable personas:")
    for i, p in enumerate(personas, 1):
        print(f"  {i}. {p}")
    while True:
        raw = input("\nChoose persona (number or name): ").strip()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(personas):
                return personas[idx]
        elif raw in personas:
            return raw
        print("Invalid choice, try again.")


def chat(persona_name, provider):
    system_prompt = make_system_prompt(persona_name)
    messages = []

    print(f"\n--- Interviewing persona: {persona_name} [{provider.name}] ---")
    print("Type 'quit' or press Ctrl+C to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEnding interview.")
            break

        if question.lower() in {"quit", "exit", "q"}:
            print("Ending interview.")
            break

        if not question:
            continue

        quotes, top_sim = get_evidence_with_scores(persona_name, question, k=EVIDENCE_K)

        if not quotes or top_sim < MIN_SIMILARITY:
            print(f"\n{persona_name}: {ABSTENTION_RESPONSE}\n")
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": ABSTENTION_RESPONSE})
            continue

        quotes = validate_evidence(provider, question, quotes)
        if not quotes:
            print(f"\n{persona_name}: {ABSTENTION_RESPONSE}\n")
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": ABSTENTION_RESPONSE})
            continue

        profile_block = make_profile_block(persona_name)
        evidence_block = make_evidence_block(quotes)
        user_content = f"{profile_block}\n\n{evidence_block}\n\n---\n\nQuestion: {question}"
        messages.append({"role": "user", "content": user_content})

        answer, usage = provider.complete_with_usage(system_prompt, messages, MAX_TOKENS)
        messages.append({"role": "assistant", "content": answer})

        # Show token usage on the first turn
        if len(messages) == 2:
            print(
                f"[cache write: {usage['cache_write']} | "
                f"cache read: {usage['cache_read']} | "
                f"uncached: {usage['uncached']}]"
            )

        print(f"\n{persona_name}: {answer}\n")


def main():
    provider = get_provider()

    # Check that the relevant API key is set
    if "anthropic" in provider.name.lower() and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.")
        return
    if "openai" in provider.name.lower() and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set.")
        return

    persona = choose_persona()
    chat(persona, provider)


if __name__ == "__main__":
    main()
