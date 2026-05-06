"""Sycamore session interface — a researcher viewer for one synthetic
evaluator driving the real (or mock) Geranium server.

Architecture:
  GET  /                     -> static/sycamore.html (single-page viewer)
  GET  /api/evaluators       -> list of available personas
  GET  /api/sessions/stream  -> Server-Sent Events stream of one full
                                protocol run; query params select condition,
                                persona, Geranium URL, and exploration size.
  POST /api/sessions/save    -> persist a finished EvaluationRecord to disk

The stream emits per-turn events as they complete (workflow, tool reaction,
gallery, query elicitation, query results, reaction, modality ranking,
closing, done). The viewer renders each event as it arrives.

Run:
    uvicorn sycamore.interface.app:app --reload --port 8001
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from llm_provider import get_fast_provider, get_provider

from step8_evaluators import EVALUATORS

from ..analysis import write_run_report
from ..evaluator.grounded import GroundedEvaluator, find_evaluator_definition
from ..evaluator.ungrounded import UngroundedEvaluator
from ..geranium_client import GeraniumClient
from ..protocol import ProtocolRunner

_INTERFACE_DIR = Path(__file__).parent
_STATIC_DIR = _INTERFACE_DIR / "static"

# Local Geranium image folder. Hardcoded for now — when retrieval results
# come back with a `name` (e.g. "foo.png"), the UI loads the thumbnail from
# `/imgs/foo.png`, which is mounted as a static directory below.
_GERANIUM_IMGS_DIR = Path("/Users/huyennguyen/Huyen/GitHub/geranium/data/unified/imgs")


class SaveRequest(BaseModel):
    record: dict
    out_dir: str = "data/sycamore_outputs"


def _json_default(obj):
    """Make EvaluationRecord-tree objects JSON-serializable for SSE."""
    try:
        return asdict(obj)
    except TypeError:
        return str(obj)


def _sse(event: dict) -> str:
    payload = json.dumps(event, default=_json_default, ensure_ascii=False)
    return f"data: {payload}\n\n"


def create_app() -> FastAPI:
    app = FastAPI(title="Sycamore", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    @app.get("/")
    def index():
        return FileResponse(_STATIC_DIR / "sycamore.html")

    # Serve Geranium image files directly out of the local imgs folder so the
    # UI can render thumbnails as <img src="/imgs/<name>">. If the path is
    # absent (e.g. running on a different machine), skip the mount silently.
    if _GERANIUM_IMGS_DIR.is_dir():
        app.mount(
            "/imgs",
            StaticFiles(directory=str(_GERANIUM_IMGS_DIR)),
            name="geranium-imgs",
        )

    @app.get("/api/evaluators")
    def list_evaluators():
        return {
            "grounded": [
                {"abbr": ev["abbr"], "id": ev["id"], "label": ev["label"],
                 "group": ev["persona"]}
                for ev in EVALUATORS
            ],
            "ungrounded": {
                "label": "Ungrounded synthetic genomics researcher",
                "description": (
                    "No grounding artefacts; persona instantiated from generic "
                    "LLM priors under a minimal role description."
                ),
            },
        }

    @app.get("/api/sessions/stream")
    def stream_session(
        condition: str = Query(..., regex="^(ungrounded|grounded)$"),
        abbr: str | None = Query(None,
                                 description="Required when condition='grounded'"),
        geranium_url: str = Query("http://localhost:5001"),
        queries_per_modality: int = Query(2, ge=1, le=6),
        k: int = Query(5, ge=1, le=24),
    ):
        # Validate up front so errors come back as a normal HTTP error,
        # not a stream that immediately closes.
        if condition == "grounded":
            if not abbr:
                raise HTTPException(400, "abbr is required for grounded condition")
            try:
                find_evaluator_definition(abbr)
            except KeyError as e:
                raise HTTPException(404, str(e))

        return StreamingResponse(
            _generate_session_stream(
                condition=condition,
                abbr=abbr,
                geranium_url=geranium_url,
                queries_per_modality=queries_per_modality,
                k=k,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.post("/api/sessions/save")
    def save_session(req: SaveRequest):
        # Reconstitute a minimal EvaluationRecord-shaped object for write_run_report.
        # The viewer sends back the record dict it accumulated client-side.
        from ..protocol.runner import EvaluationRecord, ExplorationQuery
        from ..evaluator.base import EvaluatorTurn

        def _turn(d):
            return EvaluatorTurn(**d) if d else None

        rec = EvaluationRecord(
            evaluator_id=req.record["evaluator_id"],
            condition=req.record["condition"],
            persona_group=req.record.get("persona_group"),
            instance_id=req.record.get("instance_id"),
            abbr=req.record.get("abbr"),
            label=req.record.get("label", ""),
            workflow=_turn(req.record.get("workflow")),
            tool_reaction=_turn(req.record.get("tool_reaction")),
            gallery_reaction=_turn(req.record.get("gallery_reaction")),
            queries=[ExplorationQuery(**q) for q in req.record.get("queries", [])],
            modality_ranking=req.record.get("modality_ranking"),
            ranking_rationale=req.record.get("ranking_rationale", ""),
            closing_summary=_turn(req.record.get("closing_summary")),
        )
        paths = write_run_report([rec], req.out_dir)
        return {"saved": {k: str(v) for k, v in paths.items()}}

    return app


def _generate_session_stream(
    *, condition: str, abbr: str | None,
    geranium_url: str, queries_per_modality: int, k: int,
):
    """Drive one evaluator through the protocol and yield SSE events."""
    sid = uuid.uuid4().hex[:12]
    yield _sse({"type": "stream_start", "session_id": sid,
                "condition": condition, "abbr": abbr,
                "geranium_url": geranium_url})

    try:
        provider = get_provider()
        fast_provider = get_fast_provider()
        yield _sse({"type": "info",
                    "message": f"LLM: main={provider.name}; fast={fast_provider.name}"})

        if condition == "ungrounded":
            evaluator = UngroundedEvaluator(index=1, provider=provider)
        else:
            definition = find_evaluator_definition(abbr)  # type: ignore[arg-type]
            evaluator = GroundedEvaluator(
                definition=definition,
                provider=provider,
                fast_provider=fast_provider,
            )

        client = GeraniumClient(base_url=geranium_url)
        runner = ProtocolRunner(
            client=client,
            queries_per_modality=queries_per_modality,
            k=k,
        )

        for event in runner.run_streaming(evaluator):
            if event["type"] == "done":
                # Replace the EvaluationRecord with its dict form for transport.
                rec_dict = event["record"].to_dict()
                yield _sse({"type": "done", "record": rec_dict})
            else:
                yield _sse(event)

    except Exception as e:  # noqa: BLE001
        yield _sse({"type": "fatal_error", "error": str(e)})


# Convenience module-level app for `uvicorn sycamore.interface.app:app`
app = create_app()