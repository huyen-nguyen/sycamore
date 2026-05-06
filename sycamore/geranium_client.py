"""HTTP client for the Geranium retrieval server.

Wraps the two endpoints exposed by `server/app.py` in the Geranium repo:

  POST /api/get_inference   -> top-k retrieval given (type, content, k)
  GET  /api/get_db          -> a fixed gallery used for browsing/orientation
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

Modality = Literal["Text", "Image", "Spec"]


@dataclass
class RetrievedItem:
    """One result triplet returned by Geranium."""
    name: str
    text: str
    image_b64: str          # raw base64-encoded PNG bytes
    spec_json: str          # Gosling spec as a JSON string

    @classmethod
    def from_api(cls, payload: dict) -> "RetrievedItem":
        return cls(
            name=payload["name"],
            text=payload.get("text", ""),
            image_b64=payload.get("image", ""),
            spec_json=payload.get("spec", ""),
        )


class GeraniumClient:
    """Thin HTTP wrapper. Synthetic evaluators interact with Geranium only
    through this object, so the protocol applies identically across conditions.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:5001",
        inference_path: str = "/api/get_inference",
        db_path: str = "/api/get_db",
        timeout_s: float = 120.0,
        modality_field: str = "type",   # some Geranium builds use "modality" instead
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.inference_url = f"{self.base_url}{inference_path}"
        self.db_url = f"{self.base_url}{db_path}"
        self.modality_field = modality_field
        self._client = httpx.Client(timeout=timeout_s)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GeraniumClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def search(self, modality: Modality, content: str, k: int = 12) -> list[RetrievedItem]:
        payload = {self.modality_field: modality, "content": content, "k": k}
        resp = self._client.post(self.inference_url, json=payload)
        if not resp.is_success:
            msg = f"Geranium {resp.status_code} for modality={modality!r}: {resp.text[:300]}"
            import sys; print(f"[GeraniumClient] {msg}", file=sys.stderr, flush=True)
            raise httpx.HTTPStatusError(msg, request=resp.request, response=resp)
        data = resp.json().get("data", [])
        return [RetrievedItem.from_api(item) for item in data]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def gallery(self) -> list[RetrievedItem]:
        resp = self._client.get(self.db_url)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return [RetrievedItem.from_api(item) for item in data]

    @staticmethod
    def encode_image_file(path: str | Path) -> str:
        return base64.b64encode(Path(path).read_bytes()).decode("utf-8")

    @staticmethod
    def summarize_results(items: Sequence[RetrievedItem]) -> str:
        """Compact textual summary of a result set, suitable for an LLM prompt.
        The full base64 image is omitted; only names and short text snippets
        are passed to the evaluator. Specs are truncated.
        """
        lines = []
        for i, it in enumerate(items, 1):
            text = (it.text or "").strip().replace("\n", " ")
            if len(text) > 240:
                text = text[:237] + "..."
            spec_preview = (it.spec_json or "").strip().replace("\n", " ")
            if len(spec_preview) > 200:
                spec_preview = spec_preview[:197] + "..."
            lines.append(
                f"  [{i}] {it.name}\n"
                f"      text: {text}\n"
                f"      spec: {spec_preview}"
            )
        return "\n".join(lines) if lines else "  (no results)"