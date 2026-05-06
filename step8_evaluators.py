"""
Step 8: Define and instantiate the 7 synthetic evaluators.

Each evaluator is a distinct instantiation of one of the four persona groups,
differentiated by attribute overrides that reflect natural variation within the
group (skill levels, role classifications). The evidence pool remains the full
group — overrides only affect the profile description in the system prompt.

Distribution:
    1 x Biologist
    2 x Computational Biologist  (moderate skills vs. high programming & vis)
    2 x Bioinformatician         (expert Scientist+Engineer vs. tool-builder Engineer)
    2 x Software Engineer        (moderate genomics vs. domain-aware high vis)

Usage:
    from step8_evaluators import EVALUATORS, make_evaluator_prompt
    for ev in EVALUATORS:
        prompt = make_evaluator_prompt(ev)
"""

from step4_build_prompts import make_system_prompt, make_profile_block as _make_profile_block
from step3_retrieve import get_evidence as _get_evidence, get_persona_profile


# ── Evaluator definitions ─────────────────────────────────────────────────────

EVALUATORS = [
    # ── Biologists (1) ────────────────────────────────────────────────────────
    {
        "id": "biologist_1",
        "label": "Biologist",
        "abbr": "Bio",
        "persona": "Biologists",
        "overrides": {},  # group average: Bio-focus, Low-automation, Self-audience
    },

    # ── Computational Biologists (2) ──────────────────────────────────────────
    {
        "id": "comp_bio_1",
        "label": "Computational Biologist — moderate skills",
        "abbr": "CB1",
        "persona": "Computational Biologists",
        "overrides": {
            "skills": {"genomics": 2, "data_prep": 2, "programming": 2, "vis": 2},
        },
    },
    {
        "id": "comp_bio_2",
        "label": "Computational Biologist — high programming & visualization",
        "abbr": "CB2",
        "persona": "Computational Biologists",
        "overrides": {
            "skills": {"genomics": 2, "data_prep": 2, "programming": 3, "vis": 3},
            "bio_persona": ["Scientist"],
            "ds_persona": ["DSh/DSt", "D Eng", "RS", "E"],
        },
    },

    # ── Bioinformaticians (2) ─────────────────────────────────────────────────
    {
        "id": "bioinfo_1",
        "label": "Bioinformatician — expert (Scientist + Engineer)",
        "abbr": "BIF1",
        "persona": "Bioinformaticians",
        "overrides": {
            "skills": {"genomics": 3, "data_prep": 3, "programming": 3, "vis": 3},
            "bio_persona": ["Scientist", "Engineer"],
            "ds_persona": ["D Eng", "*Eng", "RS", "E"],
            "audience": "Self + Other",
        },
    },
    {
        "id": "bioinfo_2",
        "label": "Bioinformatician — tool-builder (Engineer)",
        "abbr": "BIF2",
        "persona": "Bioinformaticians",
        "overrides": {
            "skills": {"genomics": 1, "data_prep": 2, "programming": 2, "vis": 2},
            "bio_persona": ["Engineer"],
            "ds_persona": ["DSh/DSt", "D Eng", "*Eng", "E"],
            "audience": "Self + Other",
        },
    },

    # ── Software Engineers (2) ────────────────────────────────────────────────
    {
        "id": "se_1",
        "label": "Software Engineer — moderate genomics knowledge",
        "abbr": "SE1",
        "persona": "Software Engineers",
        "overrides": {
            "skills": {"genomics": 2, "data_prep": 2, "programming": 3, "vis": 2},
            "bio_persona": ["Engineer"],
            "ds_persona": ["D Eng", "E"],
        },
    },
    {
        "id": "se_2",
        "label": "Software Engineer — domain-aware, high visualization skill",
        "abbr": "SE2",
        "persona": "Software Engineers",
        "overrides": {
            "skills": {"genomics": 3, "data_prep": 3, "programming": 3, "vis": 3},
            "bio_persona": ["Engineer"],
            "ds_persona": ["D Eng", "E"],
        },
    },
]


# ── Public helpers ────────────────────────────────────────────────────────────

def make_evaluator_prompt(evaluator, mode="recall"):
    """Return the system prompt for a given evaluator definition."""
    return make_system_prompt(evaluator["persona"], overrides=evaluator["overrides"], mode=mode)


def get_evaluator_evidence(evaluator, query, k=8):
    """Retrieve evidence for a query; pool is the full persona group."""
    return _get_evidence(evaluator["persona"], query, k=k)


def make_evaluator_profile_block(evaluator):
    """Return the injectable profile block for a given evaluator."""
    from step4_build_prompts import get_persona_profile as _gpp
    # Build profile block text using the same overrides logic
    profile = get_persona_profile(evaluator["persona"])
    if not profile:
        return ""
    members = profile["members"]
    overrides = evaluator.get("overrides", {})

    from step4_build_prompts import (
        SKILL_LABELS, _skill_summary, _bio_persona_behavioral, _ds_persona_behavioral
    )

    positions     = sorted(set(m["position"]   for m in members.values() if m["position"]))
    orgs          = sorted(set(m["org"]        for m in members.values() if m["org"]))
    focus_vals    = sorted(set(m["focus"]      for m in members.values() if m["focus"]))
    auto_vals     = sorted(set(m["automation"] for m in members.values() if m["automation"]))
    aud_vals      = sorted(set(m["audience"]   for m in members.values() if m["audience"]))

    skill_keys = ["genomics", "data_prep", "programming", "vis"]
    skills = {}
    for k in skill_keys:
        if "skills" in overrides and k in overrides["skills"]:
            skills[k] = overrides["skills"][k]
        else:
            vals = [m["skills"][k] for m in members.values() if m["skills"].get(k) is not None]
            if vals:
                skills[k] = round(sum(vals) / len(vals), 1)

    bio_personas = overrides.get(
        "bio_persona",
        sorted(set(bp for m in members.values() for bp in m["bio_persona"]))
    )
    ds_personas  = overrides.get(
        "ds_persona",
        sorted(set(dp for m in members.values() for dp in m["ds_persona"]))
    )
    focus_str = overrides.get("focus",      " / ".join(focus_vals))
    auto_str  = overrides.get("automation", " / ".join(auto_vals))
    aud_str   = overrides.get("audience",   " / ".join(aud_vals))

    lines = [
        f"[YOUR PROFILE — cite as: (profile)]",
        f"Persona: {evaluator['label']}",
        f"Group: {evaluator['persona']}",
        f"Position types: {', '.join(positions) if positions else 'varied'}",
        f"Organisation settings: {', '.join(orgs) if orgs else 'varied'}",
        f"Work focus: {focus_str}",
        f"Automation level: {auto_str}",
        f"Primary audience: {aud_str}",
        f"Bioinformatics role: {_bio_persona_behavioral(bio_personas)}",
        f"Data science role: {_ds_persona_behavioral(ds_personas)}",
    ]
    return "\n".join(lines)


# ── Script: print all evaluator profiles ─────────────────────────────────────

if __name__ == "__main__":
    for ev in EVALUATORS:
        print(f"\n{'='*60}")
        print(f"  {ev['id']}  |  {ev['label']}")
        print(f"{'='*60}")
        print(make_evaluator_prompt(ev))
