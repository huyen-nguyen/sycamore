"""Three-part Geranium user-study protocol, applied to one evaluator.

Mirrors the published study (Section 3.2 of the Sycamore manuscript):

  Part 1. Workflow description
  Part 2. Tool description (replaces the live demo)
  Part 3. Hands-on exploration: 3 modalities x N queries each, plus a
          1-3 modality preference ranking and a closing summary.

Each turn goes through the evaluator's `respond(phase, prompt)` method, so
this runner is condition-agnostic. For grounded evaluators that pulls in
RAG + validation + first-person citation. For ungrounded evaluators it's a
plain LLM call. In both cases, the conversation history is preserved by
the evaluator across turns.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ..evaluator import Evaluator, EvaluatorTurn
from ..geranium_client import GeraniumClient, RetrievedItem
from .tool_description import GERANIUM_TOOL_DESCRIPTION


# ── Prompts ───────────────────────────────────────────────────────────────────

WORKFLOW_PROMPT = (
    "Part 1: Please describe your typical workflow when you need to create "
    "or find a genomic data visualization. Talk about the kinds of charts "
    "you usually need, the tools you currently reach for, the data you "
    "work with, and the audience for the visualizations you produce. "
    "Be specific about steps and friction points. (Aim for 4-8 sentences.)"
)

TOOL_DESCRIPTION_PROMPT = (
    "Part 2: Below is a textual description of a system called Geranium. "
    "Read it, then give your first-impression reaction: what stands out "
    "as potentially useful for your work, what is unclear, and what you "
    "would want to try first.\n\n"
    "------ TOOL DESCRIPTION ------\n"
    f"{GERANIUM_TOOL_DESCRIPTION}\n"
    "------ END DESCRIPTION ------"
)

GALLERY_PROMPT_TEMPLATE = (
    "Before issuing your queries, you also browsed the Geranium gallery, "
    "which contains representative visualizations across the index. Here "
    "are the gallery items shown to you (truncated):\n"
    "{gallery}\n\n"
    "What did you notice from the gallery? Did anything inform what you "
    "want to search for? (3-5 sentences.)"
)

QUERY_GENERATION_PROMPT_TEMPLATE = (
    "Part 3 (hands-on exploration). You will issue {n} queries to Geranium "
    "in the {modality} modality.\n\n"
    "{modality_specific_instructions}\n\n"
    "Return ONLY a JSON object of the form:\n"
    "  {{\"queries\": [\"query 1\", \"query 2\", ...]}}\n"
    "with exactly {n} queries. Do not include any other text."
)

MODALITY_INSTRUCTIONS = {
    "Text": (
        "Each query is a natural-language description of a visualization "
        "you would actually want to find given your workflow. Make the "
        "queries varied: different chart types, different biological "
        "questions, different levels of specificity."
    ),
    "Image": (
        "Each 'query' here is a SHORT description of an example chart you "
        "would upload as a reference image. (In a real session you would "
        "upload a PNG; for this study, name the chart you would upload, "
        "e.g., 'a Circos plot from a published paper showing inter-"
        "chromosomal translocations'.) The runner will substitute a "
        "representative image from the gallery for each description."
    ),
    "Spec": (
        "Each 'query' here is a SHORT description of a Gosling spec you "
        "would paste in (e.g., 'a stacked bar track showing read coverage'). "
        "The runner will substitute a representative spec from the gallery "
        "matching your description as the actual query content."
    ),
}

REACTION_PROMPT_TEMPLATE = (
    "You issued the following {modality} query to Geranium:\n"
    "  QUERY: {query}\n\n"
    "Geranium returned the following top-{k} results:\n"
    "{results}\n\n"
    "React in first person, as you would in a think-aloud session. Comment "
    "on whether the results match your intent, what is useful, what is "
    "missing or off-target, and which (if any) you would adapt as a "
    "starting template. Keep it to 3-6 sentences."
)

MODALITY_RANKING_PROMPT = (
    "Final part: rank the three query modalities (Text, Image, Spec) on a "
    "1-3 scale, where 3 = most preferred and 1 = least preferred. Use "
    "each value exactly once. Provide a brief rationale (2-4 sentences) "
    "for your ranking based on the queries you just issued.\n\n"
    "Return ONLY a JSON object of the form:\n"
    "  {\"ranking\": {\"Text\": <int>, \"Image\": <int>, \"Spec\": <int>}, "
    "\"rationale\": \"<text>\"}\n"
    "with no additional commentary."
)


# ── Records ──────────────────────────────────────────────────────────────────

@dataclass
class ExplorationQuery:
    modality: str
    query_text: str
    actual_content: str            # what was sent to Geranium (truncated)
    results: list[dict]            # serialized RetrievedItem (image_b64 dropped)
    reaction: str
    reaction_evidence: list[dict] = field(default_factory=list)
    abstained: bool = False


@dataclass
class EvaluationRecord:
    evaluator_id: str
    condition: str
    persona_group: str | None
    instance_id: str | None
    abbr: str | None
    label: str

    workflow: EvaluatorTurn | None = None
    tool_reaction: EvaluatorTurn | None = None
    gallery_reaction: EvaluatorTurn | None = None
    queries: list[ExplorationQuery] = field(default_factory=list)
    modality_ranking: dict[str, int] | None = None
    ranking_rationale: str = ""
    closing_summary: EvaluatorTurn | None = None

    def to_dict(self) -> dict[str, Any]:
        def _t(turn: EvaluatorTurn | None) -> dict | None:
            return asdict(turn) if turn else None
        return {
            "evaluator_id": self.evaluator_id,
            "condition": self.condition,
            "persona_group": self.persona_group,
            "instance_id": self.instance_id,
            "abbr": self.abbr,
            "label": self.label,
            "workflow": _t(self.workflow),
            "tool_reaction": _t(self.tool_reaction),
            "gallery_reaction": _t(self.gallery_reaction),
            "queries": [asdict(q) for q in self.queries],
            "modality_ranking": self.modality_ranking,
            "ranking_rationale": self.ranking_rationale,
            "closing_summary": _t(self.closing_summary),
        }


# ── Runner ───────────────────────────────────────────────────────────────────

class ProtocolRunner:
    def __init__(
        self,
        client: GeraniumClient,
        *,
        queries_per_modality: int = 3,
        modalities: list[str] | None = None,
        k: int = 12,
        collect_modality_ranking: bool = True,
    ) -> None:
        self.client = client
        self.queries_per_modality = queries_per_modality
        self.modalities = modalities or ["Text", "Image", "Spec"]
        self.k = k
        self.collect_modality_ranking = collect_modality_ranking
        self._gallery: list[RetrievedItem] | None = None

    def run(self, evaluator: Evaluator) -> EvaluationRecord:
        """Run the full protocol synchronously, returning the final record.
        Internally consumes the streaming generator.
        """
        rec: EvaluationRecord | None = None
        for event in self.run_streaming(evaluator):
            if event["type"] == "done":
                rec = event["record"]
        assert rec is not None, "run_streaming did not emit a 'done' event"
        return rec

    def run_streaming(self, evaluator: Evaluator):
        """Generator: yield protocol events as they happen.

        Event shape: dict with a 'type' key. Types:
          - 'session_start'   {evaluator_id, condition, label, persona_group, abbr}
          - 'turn'            {phase, prompt, response, evidence, abstained}
          - 'gallery'         {n_items, items: [{name, text}, ...]}
          - 'query_issued'    {modality, query_text, actual_content_kind, n_results}
          - 'query_results'   {modality, query_text, results: [...]}  -- raw Geranium triplets (image dropped)
          - 'reaction'        same shape as 'turn' but with extra {modality, query_text}
          - 'modality_rank'   {ranking, rationale}
          - 'error'           {phase, error}
          - 'done'            {record: EvaluationRecord}
        """
        rec = EvaluationRecord(
            evaluator_id=evaluator.identity.eid,
            condition=evaluator.identity.condition,
            persona_group=evaluator.identity.persona_group,
            instance_id=evaluator.identity.instance_id,
            abbr=evaluator.identity.abbr,
            label=evaluator.identity.label,
        )
        yield {
            "type": "session_start",
            "evaluator_id": rec.evaluator_id,
            "condition": rec.condition,
            "label": rec.label,
            "persona_group": rec.persona_group,
            "abbr": rec.abbr,
        }

        # Part 1: workflow
        rec.workflow = evaluator.respond("workflow", WORKFLOW_PROMPT)
        yield self._turn_event(rec.workflow)

        # Part 2: tool description
        rec.tool_reaction = evaluator.respond("tool_description", TOOL_DESCRIPTION_PROMPT)
        yield self._turn_event(rec.tool_reaction)

        # Gallery browsing
        gallery = self._get_gallery()
        if gallery:
            yield {
                "type": "gallery",
                "n_items": len(gallery),
                "items": [
                    {"name": g.name, "text": (g.text or "")[:200]}
                    for g in gallery[:12]
                ],
            }
            gallery_summary = self.client.summarize_results(gallery[: min(12, len(gallery))])
            rec.gallery_reaction = evaluator.respond(
                "gallery", GALLERY_PROMPT_TEMPLATE.format(gallery=gallery_summary)
            )
            yield self._turn_event(rec.gallery_reaction)

        # Part 3: hands-on exploration
        for modality in self.modalities:
            queries = self._elicit_queries(evaluator, modality)
            yield {
                "type": "queries_elicited",
                "modality": modality,
                "queries": queries,
            }
            for i, q_text in enumerate(queries, 1):
                actual_content = self._materialize_query(modality, q_text)
                yield {
                    "type": "query_issued",
                    "modality": modality,
                    "query_text": q_text,
                    "actual_content_kind": (
                        "text" if modality == "Text" else
                        "image_b64" if modality == "Image" else
                        "spec_json"
                    ),
                }

                try:
                    results = self.client.search(modality, actual_content, k=self.k)
                except Exception as e:
                    yield {"type": "error", "phase": f"exploration:{modality}:q{i}",
                           "error": str(e)}
                    turn = evaluator.respond(
                        f"exploration:{modality}:q{i}",
                        f"You issued the following {modality} query to Geranium:\n"
                        f"  QUERY: {q_text}\n\n"
                        f"The system returned an error: {e}\n\n"
                        "React briefly in first person.",
                        note="geranium_error",
                    )
                    rec.queries.append(
                        ExplorationQuery(
                            modality=modality,
                            query_text=q_text,
                            actual_content=str(actual_content)[:200],
                            results=[],
                            reaction=turn.response,
                            reaction_evidence=turn.evidence,
                            abstained=turn.abstained,
                        )
                    )
                    yield {**self._turn_event(turn),
                           "modality": modality, "query_text": q_text}
                    continue

                serialized = [self._serialize_item(r) for r in results]
                # Live event includes the base64 thumbnail for the UI; the
                # persisted record (rec.queries[].results) keeps the lean
                # version without image bytes.
                live_serialized = [
                    {**s, "image_b64": getattr(r, "image_b64", "")}
                    for s, r in zip(serialized, results)
                ]
                yield {
                    "type": "query_results",
                    "modality": modality,
                    "query_text": q_text,
                    "n_results": len(results),
                    "results": live_serialized,
                }

                summary = self.client.summarize_results(results)
                reaction_prompt = REACTION_PROMPT_TEMPLATE.format(
                    modality=modality, query=q_text, k=len(results), results=summary,
                )
                turn = evaluator.respond(f"exploration:{modality}:q{i}", reaction_prompt)
                rec.queries.append(
                    ExplorationQuery(
                        modality=modality,
                        query_text=q_text,
                        actual_content=self._truncate_for_log(actual_content),
                        results=serialized,
                        reaction=turn.response,
                        reaction_evidence=turn.evidence,
                        abstained=turn.abstained,
                    )
                )
                yield {**self._turn_event(turn),
                       "modality": modality, "query_text": q_text}

        # Modality ranking
        if self.collect_modality_ranking:
            ranking_turn = evaluator.respond("modality_ranking", MODALITY_RANKING_PROMPT)
            rec.modality_ranking, rec.ranking_rationale = self._parse_ranking(
                ranking_turn.response
            )
            yield {**self._turn_event(ranking_turn)}
            yield {"type": "modality_rank",
                   "ranking": rec.modality_ranking,
                   "rationale": rec.ranking_rationale}

        # Closing
        rec.closing_summary = evaluator.respond(
            "closing",
            "Wrap up: in 4-6 sentences, give your overall assessment of "
            "Geranium for your work. What would you change first? What is "
            "the single biggest blocker, if any, to adopting it?",
        )
        yield self._turn_event(rec.closing_summary)

        yield {"type": "done", "record": rec}

    @staticmethod
    def _turn_event(turn) -> dict:
        return {
            "type": "turn",
            "phase": turn.phase,
            "prompt": turn.prompt,
            "response": turn.response,
            "evidence": turn.evidence,
            "abstained": turn.abstained,
        }

    # ── helpers ─────────────────────────────────────────────────────────── #

    def _get_gallery(self) -> list[RetrievedItem]:
        if self._gallery is None:
            try:
                self._gallery = self.client.gallery()
            except Exception:
                self._gallery = []
        return self._gallery

    def _elicit_queries(self, evaluator: Evaluator, modality: str) -> list[str]:
        prompt = QUERY_GENERATION_PROMPT_TEMPLATE.format(
            n=self.queries_per_modality,
            modality=modality,
            modality_specific_instructions=MODALITY_INSTRUCTIONS[modality],
        )
        turn = evaluator.respond(f"query_gen:{modality}", prompt, note="elicit_queries")
        try:
            obj = self._extract_json(turn.response)
            queries = [str(q).strip() for q in obj.get("queries", []) if str(q).strip()]
        except Exception:
            queries = []
        if len(queries) < self.queries_per_modality:
            queries += [
                f"(fallback {modality} query {i+1})"
                for i in range(self.queries_per_modality - len(queries))
            ]
        return queries[: self.queries_per_modality]

    # Geranium's embedding model (sentence-transformers) works best with
    # short keyword-style queries.  The evaluator LLM tends to produce long
    # descriptive sentences; we strip those down to the first clause before
    # the first comma/dash/semicolon, then hard-cap at 120 characters so we
    # never exceed the model's effective token window (~128 tokens).
    _TEXT_QUERY_MAX_CHARS: int = 120

    @classmethod
    def _shorten_text_query(cls, q_text: str) -> str:
        """Return a short, keyword-focused version of a verbose text query."""
        # Take only the first sentence/clause (split on . , ; — or —)
        import re
        # Split on: commas, semicolons, em/en dashes (—–), spaced hyphens ( - ),
        # or common filler words that start detail clauses.
        # Do NOT split on hyphens inside words (e.g. RNA-seq, BAM-file).
        first_clause = re.split(r"[,;]|[—–]| - |\s+using\s+|\s+with\s+|\s+from\s+", q_text)[0].strip()
        # Hard cap
        if len(first_clause) > cls._TEXT_QUERY_MAX_CHARS:
            first_clause = first_clause[: cls._TEXT_QUERY_MAX_CHARS].rsplit(" ", 1)[0]
        return first_clause or q_text[: cls._TEXT_QUERY_MAX_CHARS]

    def _materialize_query(self, modality: str, q_text: str) -> str:
        """Convert the evaluator's described query into something the
        Geranium API can consume.

        - Text:  shorten to a keyword-style query (first clause, ≤120 chars).
        - Image: pick the gallery item whose description has the most token
                 overlap with the description, send its base64 PNG.
        - Spec:  pick the same way, send its Gosling spec JSON.
        """
        if modality == "Text":
            return self._shorten_text_query(q_text)
        gallery = self._get_gallery()
        if not gallery:
            return ""
        chosen = self._best_gallery_match(gallery, q_text)
        if modality == "Image":
            b64 = chosen.image_b64
            if b64 and not b64.startswith("data:"):
                b64 = f"data:image/png;base64,{b64}"
            return b64
        if modality == "Spec":
            return chosen.spec_json
        return q_text

    @staticmethod
    def _best_gallery_match(gallery: list[RetrievedItem], q_text: str) -> RetrievedItem:
        q_tokens = set(re.findall(r"[a-zA-Z0-9]+", q_text.lower()))
        if not q_tokens:
            return gallery[0]
        best, best_score = gallery[0], -1
        for item in gallery:
            tokens = set(re.findall(r"[a-zA-Z0-9]+", (item.text or "").lower()))
            score = len(q_tokens & tokens)
            if score > best_score:
                best, best_score = item, score
        return best

    @staticmethod
    def _serialize_item(it: RetrievedItem) -> dict:
        return {"name": it.name, "text": it.text, "spec_json": it.spec_json}

    @staticmethod
    def _truncate_for_log(s: str, n: int = 200) -> str:
        if not isinstance(s, str):
            s = str(s)
        return s if len(s) <= n else s[: n - 3] + "..."

    @staticmethod
    def _extract_json(text: str) -> dict:
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError(f"No JSON object found in response: {text[:120]!r}")
        return json.loads(m.group(0))

    @staticmethod
    def _parse_ranking(response: str) -> tuple[dict[str, int] | None, str]:
        try:
            obj = ProtocolRunner._extract_json(response)
            ranking = obj.get("ranking", {})
            ranking = {k: int(v) for k, v in ranking.items() if k in ("Text", "Image", "Spec")}
            rationale = str(obj.get("rationale", "")).strip()
            if set(ranking.keys()) != {"Text", "Image", "Spec"}:
                return None, rationale
            return ranking, rationale
        except Exception:
            return None, response.strip()