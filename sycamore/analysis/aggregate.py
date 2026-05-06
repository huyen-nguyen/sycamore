"""Cross-condition analysis utilities.

Aligns synthetic-evaluator outputs to the five themes synthesized in the
Geranium user study (Section 3.3.1 of the Sycamore manuscript) and produces
a per-run report that the researcher can read alongside the published
expert findings.

The alignment here is deliberately mechanical: a keyword-anchored mapping
that makes the structure visible and reproducible. The manuscript's
qualitative coding remains a manual step performed by the researcher; this
module produces the scaffolding for that step, not a substitute for it.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

from ..protocol.runner import EvaluationRecord


GERANIUM_THEMES = {
    "usability_and_usefulness": [
        "useful", "usability", "usable", "easy", "difficult",
        "intuitive", "confusing", "fit", "adopt", "blocker",
    ],
    "modality_preference_rationale": [
        "modality", "text query", "image query", "spec", "ranking",
        "prefer", "preference", "fastest", "specificity",
    ],
    "balancing_variety_and_similarity": [
        "variety", "similar", "similarity", "diverse", "redundant",
        "near-duplicate", "duplicate", "different chart types", "off-target",
    ],
    "gallery_browsing_for_orientation": [
        "gallery", "browse", "browsing", "scroll", "overview",
        "what is available", "see what",
    ],
    "onboarding_and_user_intent": [
        "onboarding", "first time", "intent", "what i want", "how to start",
        "guidance", "example query", "starter",
    ],
}


def aggregate_modality_rankings(records: Iterable[EvaluationRecord]) -> dict:
    rows: list[tuple[str, str, int]] = []
    for r in records:
        if not r.modality_ranking:
            continue
        for modality, score in r.modality_ranking.items():
            rows.append((r.condition, modality, score))

    by_cond: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for cond, mod, score in rows:
        by_cond[cond][mod].append(score)

    summary: dict[str, dict] = {}
    for cond, mods in by_cond.items():
        summary[cond] = {
            mod: {
                "n": len(scores),
                "mean": round(mean(scores), 3) if scores else None,
                "scores": scores,
            }
            for mod, scores in mods.items()
        }
    return summary


def align_to_geranium_themes(records: Iterable[EvaluationRecord]) -> dict:
    out: dict[str, dict] = {}
    for r in records:
        text_blobs: list[tuple[str, str]] = []
        if r.workflow:
            text_blobs.append(("workflow", r.workflow.response))
        if r.tool_reaction:
            text_blobs.append(("tool_reaction", r.tool_reaction.response))
        if r.gallery_reaction:
            text_blobs.append(("gallery_reaction", r.gallery_reaction.response))
        for q in r.queries:
            text_blobs.append((f"reaction:{q.modality}", q.reaction))
        if r.closing_summary:
            text_blobs.append(("closing", r.closing_summary.response))

        per_theme: dict[str, list[dict]] = {t: [] for t in GERANIUM_THEMES}
        for source, text in text_blobs:
            for theme, kws in GERANIUM_THEMES.items():
                hits = [kw for kw in kws if re.search(rf"\b{re.escape(kw)}\b", text, re.I)]
                if hits:
                    per_theme[theme].append({"source": source, "matches": hits})

        out[r.evaluator_id] = {
            "condition": r.condition,
            "label": r.label,
            "theme_hits": per_theme,
        }
    return out


def write_run_report(records: list[EvaluationRecord], out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    (out_dir / "records").mkdir(parents=True, exist_ok=True)

    record_paths: list[Path] = []
    for r in records:
        fname = r.evaluator_id.replace(":", "_").replace("/", "_") + ".json"
        p = out_dir / "records" / fname
        p.write_text(json.dumps(r.to_dict(), indent=2, ensure_ascii=False))
        record_paths.append(p)

    rankings = aggregate_modality_rankings(records)
    themes = align_to_geranium_themes(records)
    summary = {
        "n_evaluators": len(records),
        "by_condition": {
            cond: sum(1 for r in records if r.condition == cond)
            for cond in {r.condition for r in records}
        },
        "modality_rankings": rankings,
        "theme_alignment": themes,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    md_path = out_dir / "summary.md"
    md_path.write_text(_render_markdown(summary, records))
    return {
        "summary_json": summary_path,
        "summary_md": md_path,
        "records_dir": out_dir / "records",
    }


def _render_markdown(summary: dict, records: list[EvaluationRecord]) -> str:
    lines: list[str] = []
    lines.append("# Sycamore run summary\n")
    lines.append(f"- Total evaluators: **{summary['n_evaluators']}**")
    lines.append("- By condition: " + ", ".join(
        f"{c}={n}" for c, n in summary["by_condition"].items()
    ))
    lines.append("")

    lines.append("## Modality preference (3 = most preferred)\n")
    for cond, mods in summary["modality_rankings"].items():
        lines.append(f"### {cond}")
        for mod in ("Text", "Image", "Spec"):
            info = mods.get(mod)
            if not info:
                lines.append(f"- {mod}: (no data)")
            else:
                lines.append(
                    f"- {mod}: mean={info['mean']} (n={info['n']}, scores={info['scores']})"
                )
        lines.append("")

    lines.append("## Theme alignment (heuristic; manual coding still required)\n")
    for eid, info in summary["theme_alignment"].items():
        lines.append(f"### {eid} ({info['condition']}) - {info['label']}")
        for theme, hits in info["theme_hits"].items():
            if hits:
                src_list = ", ".join(
                    f"{h['source']} [{', '.join(h['matches'])}]" for h in hits
                )
                lines.append(f"- **{theme}**: {src_list}")
        lines.append("")
    return "\n".join(lines)
