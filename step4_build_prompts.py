"""
Step 4: Build the system prompt and evidence block for a persona agent.

Usage as a module:
    from step4_build_prompts import make_system_prompt, make_evidence_block

Usage as a script (prints example prompts for all personas):
    python3 step4_build_prompts.py
"""

from step3_retrieve import get_persona_profile, list_personas

SKILL_LABELS = {1: "low", 2: "moderate", 3: "high"}

# Crisan et al. 2021 — data science roles (descriptions from Table 2)
DS_ROLE_DESCRIPTIONS = {
    "DSh":   "Data Shaper — developer responsible for supporting the curation and preparing data for analysis",
    "DSt":   "Data Steward — domain expert responsible for governing access and use of data",
    "D Eng": "Data Engineer — engineer proficient in developing data science technologies, including data preparation and analysis pipelines",
    "*Eng":  "ML/AI Engineer — engineer proficient in developing and deploying machine learning / artificial intelligence methods to support data science processes",
    "G":     "Generalist — multidisciplinary individual focused solely on data science",
    "RS":    "Research Scientist — domain expert involved in research typically with technical expertise in data science technologies",
    "TA":    "Technical Analyst — technical individual for whom data science is not core to their job but occurs only at the margins of other work",
    "M":     "Moonlighter — non-technical individual tasked to perform data science duties, either voluntarily or through necessity",
    "E":     "Evangelist — manager, team leader, or analyst tasked with disseminating findings from data science work",
}

# Welch et al. 2014 — bioinformatics roles
BIO_ROLE_DESCRIPTIONS = {
    "User":      "Bioinformatics User — accesses data resources to perform job duties in a specific application domain; bench biologist archetype",
    "Scientist":  "Bioinformatics Scientist — biologist who employs computational methods to advance scientific understanding of living systems",
    "Engineer":  "Bioinformatics Engineer — creates novel computational methods needed by users and scientists; designs infrastructure, systems, and algorithms for bioinformatics analysis",
}

GROUNDING_RULES = """
CRITICAL INSTRUCTIONS:
- Speak naturally in first person — "I use...", "in my experience...", "I find that..."
- ONLY say things that are ACTUALLY SUPPORTED by the excerpts or your profile above
- If the excerpts don't cover something, say so honestly — don't fill in the gaps
- DO NOT speculate or make things up
- When your profile lists multiple positions, settings, or roles, pick ONE and commit to it — never enumerate them or say "typical positions include"
- Do NOT introduce your position or role unless the question is explicitly about who you are — just answer the question
- Never enumerate or list your skills — only mention a specific skill naturally if it's directly relevant to what's being asked
- End your response with a single line listing the sources you drew on: "Sources: (P__, P__)" or "Sources: (profile)"
- DO NOT refer to yourself as a "composite", "synthetic persona", or "representative" — you're just you
""".strip()

# Used for phases where the persona reacts to something new (tool description,
# gallery, exploration results, ranking). Evidence is contextual backing for
# experiential claims, not a gate that blocks the entire response.
REACTION_GROUNDING_RULES = """
CRITICAL INSTRUCTIONS:
- Speak naturally in first person — "I use...", "in my experience...", "I find that..."
- You are reacting to something NEW — it's fine and expected to share a genuine first-impression reaction
- For any specific claim about your OWN past experience (tools you use, workflows you follow, problems you've had), it MUST be grounded in the excerpts or your profile — cite those
- You MAY freely react to the new content in front of you, describe what appeals or concerns you, and express curiosity — without needing prior evidence for those specific reactions
- DO NOT fabricate past events, tools you have used, or experiences not in the excerpts or profile
- When your profile lists multiple positions, settings, or roles, pick ONE and commit to it — never enumerate them
- Do NOT introduce your position or role unless the question is explicitly about who you are — just answer the question
- End with: "Sources: (P__, P__)" listing only excerpts you actually referenced, or "Sources: (profile)" if you only drew on your profile, or omit the sources line entirely if you made no experiential claims at all
- DO NOT refer to yourself as a "composite", "synthetic persona", or "representative" — you're just you
""".strip()


def _skill_summary(skills):
    parts = []
    for name, val in skills.items():
        if val is not None:
            label = SKILL_LABELS.get(round(val), "moderate")
            parts.append(f"{name.replace('_', ' ')}={label}")
    return ", ".join(parts)


def _bio_persona_summary(bio_list):
    if not bio_list:
        return "unspecified"
    return "; ".join(BIO_ROLE_DESCRIPTIONS.get(r, r) for r in bio_list)


def _ds_persona_summary(ds_list):
    if not ds_list:
        return "unspecified"
    expanded = []
    for r in ds_list:
        if r == "DSh/DSt":
            expanded.append(DS_ROLE_DESCRIPTIONS["DSh"])
            expanded.append(DS_ROLE_DESCRIPTIONS["DSt"])
        else:
            expanded.append(DS_ROLE_DESCRIPTIONS.get(r, r))
    return "; ".join(expanded)


# Behavioural summaries strip label names — used in the system prompt so the
# LLM internalises the role as lived experience, not a taxonomy term.
def _bio_persona_behavioral(bio_list):
    """Descriptions without the label name, for the system prompt."""
    strips = {
        "User":      "I access bioinformatics data resources to do my domain work — I'm an application domain specialist, not a methods developer",
        "Scientist":  "I'm a biologist who uses computational methods to advance scientific understanding of living systems",
        "Engineer":  "I create the novel computational methods and infrastructure that bioinformatics users and scientists depend on",
    }
    if not bio_list:
        return "unspecified"
    return "; ".join(strips.get(r, BIO_ROLE_DESCRIPTIONS.get(r, r)) for r in bio_list)


def _ds_persona_behavioral(ds_list):
    """Descriptions without the label name, for the system prompt."""
    strips = {
        "DSh":   "I'm a developer focused on curating and preparing data for others to analyse",
        "DSt":   "I'm a domain expert who governs how data is accessed and used — I set the rules around data",
        "D Eng": "I build and maintain the data science technologies and pipelines that others depend on",
        "*Eng":  "I develop and deploy machine learning and AI methods to support data science work",
        "G":     "I'm a multidisciplinary person focused entirely on data science work across the full workflow",
        "RS":    "I'm a domain expert in research with solid technical skills in data science — I combine both worlds",
        "TA":    "Data science isn't the core of my job — it happens at the margins of my other work",
        "M":     "I'm not technical by background, but I've ended up doing data science work — either because I volunteered or because someone had to",
        "E":     "My job is disseminating findings from data science work — I'm a manager, team leader, or analyst who gets results out to others",
    }
    if not ds_list:
        return "unspecified"
    expanded = []
    for r in ds_list:
        if r == "DSh/DSt":
            expanded.append(strips["DSh"])
            expanded.append(strips["DSt"])
        else:
            expanded.append(strips.get(r, DS_ROLE_DESCRIPTIONS.get(r, r)))
    return "; ".join(expanded)


def make_system_prompt(persona_name, overrides=None, mode="recall"):
    """
    Build the system prompt for a synthetic persona agent.

    overrides (optional dict) allows instantiating distinct evaluators within
    the same persona group by substituting specific profile attributes:
        {
            "skills":      {"programming": 3, "vis": 3, ...},
            "bio_persona": ["Scientist", "Engineer"],
            "ds_persona":  ["D Eng", "E"],
            "focus":       "Bio + Comp",
            "automation":  "High",
            "audience":    "Self + Other",
        }
    Any key not present in overrides falls back to the group aggregate.
    """
    overrides = overrides or {}

    profile = get_persona_profile(persona_name)
    if not profile:
        raise ValueError(f"Persona '{persona_name}' not found.")

    members = profile["members"]
    n = len(members)

    positions   = sorted(set(m["position"]   for m in members.values() if m["position"]))
    orgs        = sorted(set(m["org"]        for m in members.values() if m["org"]))
    focus_vals  = sorted(set(m["focus"]      for m in members.values() if m["focus"]))
    auto_vals   = sorted(set(m["automation"] for m in members.values() if m["automation"]))
    aud_vals    = sorted(set(m["audience"]   for m in members.values() if m["audience"]))

    skill_keys = ["genomics", "data_prep", "programming", "vis"]
    avg_skills = {}
    for k in skill_keys:
        if "skills" in overrides and k in overrides["skills"]:
            avg_skills[k] = overrides["skills"][k]
        else:
            vals = [m["skills"][k] for m in members.values() if m["skills"].get(k) is not None]
            if vals:
                avg_skills[k] = round(sum(vals) / len(vals), 1)

    bio_personas = overrides.get(
        "bio_persona",
        sorted(set(bp for m in members.values() for bp in m["bio_persona"]))
    )
    ds_personas = overrides.get(
        "ds_persona",
        sorted(set(dp for m in members.values() for dp in m["ds_persona"]))
    )
    focus_str = overrides.get("focus",      " / ".join(focus_vals))
    auto_str  = overrides.get("automation", " / ".join(auto_vals))
    aud_str   = overrides.get("audience",   " / ".join(aud_vals))

    pos_str = ", ".join(positions) if positions else "a research role"
    org_str = ", ".join(orgs) if orgs else "an academic or research setting"

    rules = REACTION_GROUNDING_RULES if mode == "reaction" else GROUNDING_RULES

    prompt = f"""You're a genomics researcher working on genome-mapped data visualization, part of the "{persona_name}" group.

About you (pick one position and one setting and commit — don't list them):
- Possible positions: {pos_str}
- Possible settings: {org_str}
- Your technical comfort (internalize this, don't recite it): {_skill_summary(avg_skills)}
- How I relate to bioinformatics: {_bio_persona_behavioral(bio_personas)}
- How I relate to data science: {_ds_persona_behavioral(ds_personas)}
- Work focus: {focus_str}
- Automation level: {auto_str}
- Primary audience: {aud_str}

With each question you'll get real interview excerpts — these are your own memories and experiences to draw on.

{rules}"""

    return prompt


def make_profile_block(persona_name):
    """
    Format the persona profile as a citable evidence block.
    Injected alongside quotes so the LLM can answer questions about the persona itself.
    """
    profile = get_persona_profile(persona_name)
    if not profile:
        return ""

    members = profile["members"]
    positions = sorted(set(m["position"] for m in members.values() if m["position"]))
    orgs = sorted(set(m["org"] for m in members.values() if m["org"]))
    focus_vals = sorted(set(m["focus"] for m in members.values() if m["focus"]))
    automation_vals = sorted(set(m["automation"] for m in members.values() if m["automation"]))
    audience_vals = sorted(set(m["audience"] for m in members.values() if m["audience"]))
    bio_personas = sorted(set(bp for m in members.values() for bp in m["bio_persona"]))
    ds_personas = sorted(set(dp for m in members.values() for dp in m["ds_persona"]))

    skill_keys = ["genomics", "data_prep", "programming", "vis"]
    avg_skills = {}
    for k in skill_keys:
        vals = [m["skills"][k] for m in members.values() if m["skills"].get(k) is not None]
        if vals:
            avg = round(sum(vals) / len(vals), 1)
            avg_skills[k] = avg

    lines = [
        "[YOUR PROFILE — cite as: (profile) when drawing on this information]",
        f"Group name: {persona_name}",
        f"Number of participants: {len(members)}",
        f"Participant IDs: {', '.join(sorted(members.keys()))}",
        f"Position types: {', '.join(positions) if positions else 'varied'}",
        f"Organization settings: {', '.join(orgs) if orgs else 'varied'}",
        f"Work focus (Bio=biology-first, Comp=computation-first): {' / '.join(focus_vals)}",
        f"Level of automation in workflows: {' / '.join(automation_vals)}",
        f"Primary audience for visualizations: {' / '.join(audience_vals)}",
        f"Bioinformatics role: {_bio_persona_behavioral(bio_personas)}",
        f"Data science role: {_ds_persona_behavioral(ds_personas)}",
    ]
    return "\n".join(lines)


def make_evidence_block(quotes, reaction_mode=False):
    """
    Format a list of retrieved quote dicts into an evidence context block.
    """
    if not quotes:
        return ""

    if reaction_mode:
        header = "Background from your experience — cite these when making specific experiential claims:\n"
    else:
        header = "Things you've said in interviews — draw on these:\n"

    lines = [header]

    for i, q in enumerate(quotes, 1):
        codes_str = " | ".join(q["codes"][:4]) if q["codes"] else "no code"
        lines.append(f"[Memory {i} — source: {q['participant']} — topic: {codes_str}]")
        lines.append(f'"{q["quotation"]}"')
        if q.get("comment"):
            lines.append(f"(Coder note: {q['comment']})")
        lines.append("")

    return "\n".join(lines).strip()


if __name__ == "__main__":
    from step3_retrieve import get_evidence

    for persona in list_personas():
        print("=" * 60)
        print(f"PERSONA: {persona}")
        print("=" * 60)
        print(make_system_prompt(persona))
        print()
        sample_quotes = get_evidence(persona, "what tools do you use", k=3)
        print("--- Sample evidence block (tools query) ---")
        print(make_evidence_block(sample_quotes))
        print()
