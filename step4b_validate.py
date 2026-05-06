"""
Step 4b: Pre-generation evidence validation with structured output.

A separate LLM call filters retrieved excerpts to only those that actually
support answering the query, before they are passed to the persona agent.
Uses Pydantic for structured output validation (tool use on Anthropic,
response_format on OpenAI) — no fragile text parsing.

Returns the filtered quote list. If the validator finds none of the excerpts
relevant, returns an empty list (triggering hard abstention in the caller).
"""

from typing import List
from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """Structured output for the evidence validation step."""
    relevant_indices: List[int] = Field(
        description="1-based indices of excerpts that directly support answering the question. Empty list if none are relevant."
    )


VALIDATION_SYSTEM = "You are a relevance filter for a retrieval-augmented system. Follow instructions exactly and return structured output."

VALIDATION_PROMPT = """Given a question and a numbered list of interview excerpts, identify which excerpts directly support answering the question.

An excerpt is relevant if it contains information a person could draw on to answer the question — even partially.

Return the 1-based indices of relevant excerpts. Return an empty list if none are relevant.

Question: {question}

Excerpts:
{excerpts}"""


def validate_evidence(provider, question, quotes, max_tokens=256):
    """
    Filter quotes to those the validator judges as relevant to the question.

    Args:
        provider:   an LLMProvider instance
        question:   the user's question string
        quotes:     list of quote dicts from get_evidence()
        max_tokens: keep low — response is a short structured object

    Returns:
        Filtered list of quote dicts. Empty list if none are deemed relevant.
    """
    if not quotes:
        return []

    excerpt_lines = []
    for i, q in enumerate(quotes, 1):
        codes_str = " | ".join(q["codes"][:2]) if q["codes"] else ""
        text = q["quotation"][:200]
        excerpt_lines.append(f"{i}. [{codes_str}] \"{text}\"")

    prompt = VALIDATION_PROMPT.format(
        question=question,
        excerpts="\n".join(excerpt_lines),
    )

    try:
        result = provider.complete_structured(
            system_prompt=VALIDATION_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            response_model=ValidationResult,
            max_tokens=max_tokens,
        )
    except Exception:
        # On error, pass all quotes through rather than silently dropping evidence
        return quotes

    valid_indices = [i - 1 for i in result.relevant_indices if 1 <= i <= len(quotes)]
    return [quotes[i] for i in sorted(set(valid_indices))]
