"""
Persona chat server.

Run with:
    python server.py

Then open http://localhost:8000 in your browser.
API docs at http://localhost:8000/docs
"""

import json
import os
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from llm_provider import get_provider, get_fast_provider
from step3_retrieve import (
    get_evidence_with_scores, MIN_SIMILARITY, ABSTENTION_RESPONSE
)
from step4_build_prompts import make_evidence_block
from step4b_validate import validate_evidence
from step8_evaluators import EVALUATORS, make_evaluator_prompt, make_evaluator_profile_block

MAX_TOKENS = 1024

import re

SOCIAL_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening)|"
    r"thanks?|thank\s+you|cheers|bye|goodbye|see\s+you|how\s+are\s+you)\W*\s*$",
    re.IGNORECASE,
)

SOCIAL_SYSTEM = (
    "You are playing a synthetic research persona being interviewed. "
    "Respond naturally to casual social messages — greetings, thanks, farewells — "
    "in one short sentence, in character, without citing sources."
)

def is_social(text: str) -> bool:
    return bool(SOCIAL_PATTERNS.match(text.strip()))

app = FastAPI(title="Persona Interview", description="RAG-grounded persona chat interface")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Request / response models ─────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    evaluator_id: str
    provider: str = "anthropic"
    question: str
    history: List[Message] = Field(default_factory=list)
    k: int = Field(default=5, ge=0, le=15)


class QuoteOut(BaseModel):
    participant: str
    quotation: str
    codes: List[str]


class ChatResponse(BaseModel):
    answer: str
    quotes: List[QuoteOut]
    abstained: bool


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def index():
    return FileResponse("static/index.html")


@app.get("/api/evaluators")
def api_evaluators():
    return [{"id": ev["id"], "label": ev["label"], "abbr": ev["abbr"], "group": ev["persona"]} for ev in EVALUATORS]


@app.get("/api/info")
def api_info(provider: Optional[str] = None):
    p = get_provider(provider)
    return {"provider": p.name}


@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    evaluator = next((ev for ev in EVALUATORS if ev["id"] == req.evaluator_id), None)
    if evaluator is None:
        raise HTTPException(status_code=400, detail=f"Unknown evaluator_id: {req.evaluator_id!r}")

    try:
        provider = get_provider(req.provider)

        # 1. Retrieve with similarity score
        quotes, top_sim = get_evidence_with_scores(evaluator["persona"], req.question, k=req.k)

        # 2. Hard abstention: no relevant evidence found
        if not quotes or top_sim < MIN_SIMILARITY:
            return ChatResponse(answer=ABSTENTION_RESPONSE, quotes=[], abstained=True)

        # 3. Validate: filter to excerpts that actually support the question
        quotes = validate_evidence(get_fast_provider(req.provider), req.question, quotes)
        if not quotes:
            return ChatResponse(answer=ABSTENTION_RESPONSE, quotes=[], abstained=True)

        # 4. Generate
        system_prompt = make_evaluator_prompt(evaluator)
        profile_block = make_evaluator_profile_block(evaluator)
        evidence_block = make_evidence_block(quotes)
        user_content = f"{profile_block}\n\n{evidence_block}\n\n---\n\nQuestion: {req.question}"

        api_messages = [{"role": m.role, "content": m.content} for m in req.history]
        api_messages.append({"role": "user", "content": user_content})

        answer = provider.complete(system_prompt, api_messages, MAX_TOKENS)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        answer=answer,
        quotes=[QuoteOut(participant=q["participant"], quotation=q["quotation"], codes=q["codes"]) for q in quotes],
        abstained=False,
    )


@app.post("/api/chat/stream")
def api_chat_stream(req: ChatRequest):
    evaluator = next((ev for ev in EVALUATORS if ev["id"] == req.evaluator_id), None)
    if evaluator is None:
        raise HTTPException(status_code=400, detail=f"Unknown evaluator_id: {req.evaluator_id!r}")

    def generate():
        try:
            if is_social(req.question):
                provider = get_provider(req.provider)
                system_prompt = make_evaluator_prompt(evaluator)
                api_messages = [{"role": m.role, "content": m.content} for m in req.history]
                api_messages.append({"role": "user", "content": req.question})
                yield f"data: {json.dumps({'type': 'quotes', 'quotes': []})}\n\n"
                for token in provider.complete_stream(system_prompt, api_messages, 80):
                    yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            quotes, top_sim = get_evidence_with_scores(evaluator["persona"], req.question, k=req.k)

            if not quotes or top_sim < MIN_SIMILARITY:
                yield f"data: {json.dumps({'type': 'abstain', 'answer': ABSTENTION_RESPONSE})}\n\n"
                return

            quotes_out = [
                {"participant": q["participant"], "quotation": q["quotation"], "codes": q["codes"]}
                for q in quotes
            ]
            yield f"data: {json.dumps({'type': 'quotes', 'quotes': quotes_out})}\n\n"

            provider = get_provider(req.provider)
            system_prompt = make_evaluator_prompt(evaluator)
            profile_block = make_evaluator_profile_block(evaluator)
            evidence_block = make_evidence_block(quotes)
            user_content = f"{profile_block}\n\n{evidence_block}\n\n---\n\nQuestion: {req.question}"

            api_messages = [{"role": m.role, "content": m.content} for m in req.history]
            api_messages.append({"role": "user", "content": user_content})

            for token in provider.complete_stream(system_prompt, api_messages, MAX_TOKENS):
                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
