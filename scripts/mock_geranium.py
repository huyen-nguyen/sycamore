"""A tiny mock Geranium server that mirrors the two endpoints used by Sycamore.

Run:  python scripts/mock_geranium.py
Use it for pipeline plumbing tests when the real Geranium server is not
available. It returns canned triplets regardless of query content.
"""
from __future__ import annotations

import base64
import json

from fastapi import FastAPI, Request
import uvicorn

# A 1x1 transparent PNG so the API surface is faithful (image is base64 of bytes).
_TINY_PNG = base64.b64encode(
    bytes.fromhex(
        "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C4"
        "890000000A49444154789C6300010000000500010D0A2DB40000000049454E44"
        "AE426082"
    )
).decode("utf-8")

_FAKE_SPEC = json.dumps({
    "title": "Mock chart",
    "tracks": [{"data": {"url": "..."}, "mark": "line"}],
})

_FAKE_TEMPLATES = [
    {"name": f"mock_template_{i:02d}",
     "text": f"Mock visualization #{i}: a placeholder description for retrieval testing.",
     "image": _TINY_PNG, "spec": _FAKE_SPEC}
    for i in range(20)
]

app = FastAPI()


@app.post("/api/get_inference")
async def get_inference(req: Request):
    body = await req.json()
    k = int(body.get("k", 12))
    return {"data": _FAKE_TEMPLATES[:k]}


@app.get("/api/get_db")
async def get_db():
    return {"data": _FAKE_TEMPLATES[:12]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5001)