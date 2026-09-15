"""Inline-media client for the separate physical perception API."""
from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path

import httpx

from .model import ModelConfig
from .physical_model import prepare_physical_request, validate_physical_audio
from .physical_schema import validate_physical_decision
from .physical_serve import PhysicalResponse


class PhysicalRemoteModel:
    def __init__(self, url: str, api_token: str, *, dataset_root: str | Path = ".",
                 config: ModelConfig | None = None, timeout: float = 180, client=None):
        if not api_token:
            raise ValueError("physical remote inference requires a token")
        self.url = url.rstrip("/")
        self.api_token = api_token
        self.config = config or ModelConfig(dataset_root=str(dataset_root), max_frames=32,
                                           max_input_tokens=16384, merge_adapter=False)
        self.client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def _post(self, request: dict) -> dict:
        clean = prepare_physical_request(request, self.config)
        validate_physical_audio(clean, self.config)
        media = {}
        for item in [*[frame for clip in clean["window"]["clips"] for frame in clip["frames"]],
                     *clean["window"]["audio"]]:
            path = Path(item["path"])
            media_type = mimetypes.guess_type(path.name)[0]
            if media_type == "audio/x-flac":
                media_type = "audio/flac"
            content = path.read_bytes()
            media[item["evidence_id"]] = {"content_base64": base64.b64encode(content).decode("ascii"),
                                          "media_type": media_type or "application/octet-stream"}
            item["path"] = "inline-upload"  # Original local paths never cross the HTTP boundary.
        payload = {"request": clean, "media": media}
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        if len(encoded) > 64 * 1024 * 1024:
            raise ValueError("physical remote request exceeds transport byte budget")
        response = self.client.post(self.url + "/observe", content=encoded,
                                    headers={"Authorization": "Bearer " + self.api_token, "Content-Type": "application/json"})
        if response.status_code not in (200, 422, 503):
            response.raise_for_status()
        result = PhysicalResponse.model_validate(response.json())
        if result.decision is not None:
            validate_physical_decision(result.decision, clean["window"],
                                      known_track_ids={item["track_id"] for item in clean["tracks"]})
        return result.model_dump()

    def observe(self, request: dict) -> dict:
        """Return preserved raw/error details even when model output is invalid."""
        return self._post(request)

    def generate_text(self, request: dict) -> tuple[str, dict]:
        result = self._post(request)
        if result["error"] and not result["raw_output"]:
            raise ValueError(result["error"])
        return result["raw_output"], result["metrics"]

    def close(self):
        if self._owns_client:
            self.client.close()
