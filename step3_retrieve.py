"""
Step 3: Retrieve relevant evidence quotes for a given persona group and query.

Retrieval strategy: semantic similarity (sentence-transformers all-MiniLM-L6-v2)
with a small code-category bonus as a tie-breaker.

The embedding index is built from data/evidence.json on first use and cached to
data/embeddings.npy. Re-run with --rebuild to force a fresh index.

Usage as a module:
    from step3_retrieve import get_evidence, list_personas

Usage as a script (interactive test / index rebuild):
    python3 step3_retrieve.py
    python3 step3_retrieve.py --rebuild
"""

import json
import os
import re
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

PERSONAS_FILE = "data/personas.json"
EVIDENCE_FILE = "data/evidence.json"
EMBEDDINGS_FILE = "data/embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"

# Code-category bonus: keeps topical relevance as a tie-breaker (max 0.15)
KEYWORD_TO_CATEGORIES = {
    "tool": ["TOOLS & LANGUAGES"],
    "library": ["TOOLS & LANGUAGES"],
    "software": ["TOOLS & LANGUAGES"],
    "language": ["TOOLS & LANGUAGES"],
    "python": ["TOOLS & LANGUAGES"],
    "r ": ["TOOLS & LANGUAGES"],
    "workflow": ["VIS AUTHORING WORKFLOW & TASKS", "WORKFLOW DATA/INFO COMPONENTS"],
    "process": ["VIS AUTHORING WORKFLOW & TASKS", "WORKFLOW DATA/INFO COMPONENTS"],
    "pipeline": ["WORKFLOW DATA/INFO COMPONENTS"],
    "goal": ["GOAL"],
    "purpose": ["GOAL"],
    "why": ["GOAL"],
    "challenge": ["Challenges & Limitations"],
    "problem": ["Challenges & Limitations"],
    "difficult": ["Challenges & Limitations"],
    "struggle": ["Challenges & Limitations"],
    "skill": ["EXPERIENCE & SKILLS"],
    "experience": ["EXPERIENCE & SKILLS"],
    "expertise": ["EXPERIENCE & SKILLS"],
    "collaborat": ["*Collaboration"],
    "team": ["*Collaboration"],
    "colleague": ["*Collaboration"],
    "share": ["*Share / Save / Export"],
    "export": ["*Share / Save / Export"],
    "visual": ["VIS AUTHORING WORKFLOW & TASKS", "TARGET VIS"],
    "design": ["VIS AUTHORING WORKFLOW & TASKS", "*Reflections on Design"],
    "reuse": ["*Reuse"],
    "template": ["*Reuse"],
    "context": ["CONTEXT"],
    "criteria": ["CRITERIA", "VIS SUCCESS CRITERIA"],
    "aesthetic": ["VIS AUTHORING WORKFLOW & TASKS"],
    "feedback": ["VIS AUTHORING WORKFLOW & TASKS"],
    "iteration": ["PROCESS TYPE"],
    "modality": ["MODALITY"],
    "audience": ["*Collaboration", "GOAL"],
    "data": ["WORKFLOW DATA/INFO COMPONENTS"],
    "explor": ["GOAL", "VISUAL ANALYTICS TASKS"],
    "analys": ["GOAL", "VISUAL ANALYTICS TASKS"],
    "present": ["GOAL"],
    "communicat": ["GOAL"],
    "validat": ["GOAL"],
    "automat": ["MODALITY", "WORKFLOW DATA/INFO COMPONENTS"],
}

# Module-level cache — loaded once per process
_embedder: Optional[SentenceTransformer] = None
_embeddings: Optional[np.ndarray] = None   # shape (N_evidence, dim)
_evidence_cache: Optional[list] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path):
    with open(path) as f:
        return json.load(f)


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(MODEL_NAME)
    return _embedder


def _get_evidence() -> list:
    global _evidence_cache
    if _evidence_cache is None:
        _evidence_cache = _load(EVIDENCE_FILE)
    return _evidence_cache


def _embed_text(record):
    """Text fed to the encoder: code labels prepended to the quote so both are represented."""
    code_str = " | ".join(record["codes"]) if record["codes"] else ""
    return f"{code_str}. {record['quotation']}" if code_str else record["quotation"]


def _load_or_build_embeddings(force_rebuild=False):
    """
    Return embeddings array of shape (N_evidence, dim).
    Loads from cache if available and consistent; builds otherwise.
    """
    global _embeddings
    if _embeddings is not None and not force_rebuild:
        return _embeddings

    evidence = _get_evidence()

    if not force_rebuild and os.path.exists(EMBEDDINGS_FILE):
        arr = np.load(EMBEDDINGS_FILE)
        if arr.shape[0] == len(evidence):
            _embeddings = arr
            return _embeddings
        print(f"Index size mismatch ({arr.shape[0]} vs {len(evidence)} records) — rebuilding.")

    embedder = _get_embedder()
    texts = [_embed_text(r) for r in evidence]
    print(f"Building embedding index for {len(texts)} quotes (one-time, ~30s)…")
    arr = embedder.encode(texts, show_progress_bar=True, batch_size=64, convert_to_numpy=True)
    os.makedirs("data", exist_ok=True)
    np.save(EMBEDDINGS_FILE, arr)
    print(f"Index saved to {EMBEDDINGS_FILE}")

    _embeddings = arr
    return _embeddings


def _cosine_sim(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity. Embeddings from sentence-transformers are already unit-normalised."""
    import warnings
    q = query_vec.astype(np.float32)
    q /= np.linalg.norm(q) + 1e-9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return matrix.astype(np.float32) @ q


def _category_bonus(record: dict, target_categories: set) -> float:
    """Small additive bonus for code-category match (max 0.15)."""
    hits = sum(1 for c in record["code_categories"] if c in target_categories)
    return min(hits * 0.05, 0.15)


# Minimum cosine similarity for evidence to be considered relevant.
# Queries whose top result falls below this threshold trigger hard abstention.
MIN_SIMILARITY = 0.25

# Fixed abstention string returned when no relevant evidence exists.
ABSTENTION_RESPONSE = "I don't have enough information to answer that based on my experiences."


# ── Public API ────────────────────────────────────────────────────────────────

def list_personas():
    return list(_load(PERSONAS_FILE).keys())


def get_persona_profile(persona_name):
    return _load(PERSONAS_FILE).get(persona_name)


def get_evidence_with_scores(persona_name, query, k=8):
    """
    Return (quotes, top_similarity_score) for the given persona and query.
    quotes is a list of up to k evidence dicts sorted by relevance.
    top_similarity_score is the raw cosine similarity of the best match (0–1),
    used to decide whether any evidence is relevant enough to proceed.
    """
    personas = _load(PERSONAS_FILE)
    if persona_name not in personas:
        raise ValueError(f"Unknown persona '{persona_name}'. Available: {list(personas.keys())}")

    evidence = _get_evidence()
    all_embeddings = _load_or_build_embeddings()

    participants = set(personas[persona_name]["participants"])
    pool_indices = [i for i, r in enumerate(evidence) if r["participant"] in participants]

    if not pool_indices:
        return [], 0.0

    embedder = _get_embedder()
    query_vec = embedder.encode([query], convert_to_numpy=True)[0]
    pool_matrix = all_embeddings[pool_indices]
    sims = _cosine_sim(query_vec, pool_matrix)

    query_lower = query.lower()
    target_categories = set()
    for keyword, cats in KEYWORD_TO_CATEGORIES.items():
        if keyword in query_lower:
            target_categories.update(cats)

    scored = []
    for j, idx in enumerate(pool_indices):
        score = float(sims[j]) + _category_bonus(evidence[idx], target_categories)
        scored.append((idx, score, float(sims[j])))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_sim = scored[0][2] if scored else 0.0
    return [evidence[idx] for idx, _, _ in scored[:k]], top_sim


def get_evidence(persona_name, query, k=8):
    """Convenience wrapper — returns quotes only (no score)."""
    quotes, _ = get_evidence_with_scores(persona_name, query, k=k)
    return quotes


# ── Script mode ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of embedding index")
    args = parser.parse_args()

    _load_or_build_embeddings(force_rebuild=args.rebuild)

    print("Available personas:", list_personas())
    print()

    persona = input("Enter persona name: ").strip()
    query = input("Enter query: ").strip()

    results = get_evidence(persona, query, k=6)
    print(f"\nTop {len(results)} quotes for '{persona}' on '{query}':\n")
    for i, r in enumerate(results, 1):
        codes_str = " | ".join(r["codes"][:3])
        print(f"[{i}] {r['participant']} — {codes_str}")
        print(f"    \"{r['quotation'][:200]}\"")
        print()
