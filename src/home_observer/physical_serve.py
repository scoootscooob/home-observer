"""Authenticated native physical perception with bounded inline-media transport."""
from __future__ import annotations

import argparse
import base64
import binascii
import copy
import hmac
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import Field, model_validator

from .model import ModelConfig
from .physical_model import PhysicalInferenceRequest, PhysicalNativeModel
from .physical_schema import PhysicalDecision, PhysicalModel, PhysicalWindow, validate_physical_decision

MEDIA_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
               "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/flac": ".flac"}


class InlinePhysicalMedia(PhysicalModel):
    content_base64: str = Field(min_length=1)
    media_type: str


class PhysicalUpload(PhysicalModel):
    request: PhysicalInferenceRequest
    media: dict[str, InlinePhysicalMedia] = Field(max_length=544)


class PhysicalResponse(PhysicalModel):
    decision: PhysicalDecision | None = None
    raw_output: str = ""
    metrics: dict = Field(default_factory=dict)
    error: str | None = None

    @model_validator(mode="after")
    def result_or_error(self):
        if (self.decision is None) == (self.error is None):
            raise ValueError("physical response must contain either a decision or an error")
        return self


def constrained_physical_schema(window: PhysicalWindow | dict, known_track_ids=()) -> dict:
    """JSON grammar aid: free labels, current evidence, known identity choices.

    Grammar cannot prove identity, physical truth or cross-field temporal/box
    consistency. Always run validate_physical_decision after constrained decoding.
    This helper does not silently enable an unverified Transformers grammar backend.
    """
    window = PhysicalWindow.model_validate(window)
    schema = copy.deepcopy(PhysicalDecision.model_json_schema())
    definitions = schema["$defs"]
    evidence = sorted(window.evidence_ids())
    frames = sorted(frame.evidence_id for clip in window.clips for frame in clip.frames)
    for name, keys in (("PhysicalObject", ("evidence_ids",)), ("PhysicalObservation", ("evidence_ids",)),
                       ("TemporalEvent", ("pre_evidence_ids", "post_evidence_ids"))):
        for key in keys:
            if evidence:
                definitions[name]["properties"][key]["items"] = {"type": "string", "enum": evidence}
    if frames:
        definitions["PhysicalObject"]["properties"]["frame_evidence_id"] = {"type": "string", "enum": frames}
    else:
        schema["properties"]["objects"]["maxItems"] = 0
    if not evidence:
        for key in ("observations", "events"):
            schema["properties"][key]["maxItems"] = 0
    tracks = sorted(set(known_track_ids))
    if any(not isinstance(identifier, str) or not identifier.startswith("phys_") for identifier in tracks):
        raise ValueError("constrained physical schema requires separate physical track IDs")
    definitions["PhysicalObject"]["properties"]["track_id"] = (
        {"anyOf": [{"type": "string", "enum": tracks}, {"type": "null"}], "default": None}
        if tracks else {"type": "null", "default": None})
    return schema


def _write_inline_media(upload: PhysicalUpload, directory: Path, max_file_bytes: int) -> dict:
    from PIL import Image
    request = upload.request.model_dump()
    frames = [frame for clip in request["window"]["clips"] for frame in clip["frames"]]
    audio = request["window"]["audio"]
    expected = {item["evidence_id"] for item in [*frames, *audio]}
    if set(upload.media) != expected:
        raise ValueError("provide exactly one inline upload for every frame/audio evidence ID")
    image_ids = {frame["evidence_id"] for frame in frames}
    for index, item in enumerate([*frames, *audio]):
        media = upload.media[item["evidence_id"]]
        if media.media_type not in MEDIA_TYPES:
            raise ValueError("unsupported inline physical media type")
        is_image = item["evidence_id"] in image_ids
        if is_image != media.media_type.startswith("image/"):
            raise ValueError("media type disagrees with frame/audio evidence")
        if len(media.content_base64) > ((max_file_bytes + 2) // 3) * 4:
            raise ValueError("inline media exceeds per-file byte budget")
        try:
            content = base64.b64decode(media.content_base64, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("invalid inline media base64") from exc
        if not content or len(content) > max_file_bytes:
            raise ValueError("inline media exceeds per-file byte budget")
        # Client paths/evidence IDs never become filesystem names.
        path = directory / (f"media-{index}" + MEDIA_TYPES[media.media_type])
        path.write_bytes(content)
        if is_image:
            with Image.open(path) as image:
                if image.width * image.height > 16_777_216 or image.width < 1 or image.height < 1:
                    raise ValueError("inline image exceeds pixel budget")
                image.verify()
        else:
            import soundfile as sf
            info = sf.info(path)
            if not 0 < info.frames / info.samplerate <= 30 or not 1 <= info.channels <= 2:
                raise ValueError("inline audio exceeds channel/duration budget")
        item["path"] = str(path)
    return request


def create_physical_app(backend, *, api_token: str, media_root: str | Path,
                        max_body_bytes: int = 64 * 1024 * 1024, max_file_bytes: int = 8 * 1024 * 1024):
    if not api_token or len(api_token) < 16:
        raise ValueError("physical API requires an explicit token of at least 16 characters")
    if not 1 <= max_file_bytes <= max_body_bytes <= 128 * 1024 * 1024:
        raise ValueError("invalid physical API body/file budget")
    media_root = Path(media_root).resolve()
    media_root.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="Physical Perception", docs_url=None, redoc_url=None, openapi_url=None)

    def authorize(request):
        expected = "Bearer " + api_token
        if not hmac.compare_digest(request.headers.get("authorization", ""), expected):
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/health")
    def health(request: Request):
        authorize(request)
        return {"ok": True, "task": "physical_perception", "media_transport": "inline_base64",
                "max_body_bytes": max_body_bytes, "max_file_bytes": max_file_bytes}

    @app.post("/observe")
    async def observe(request: Request):
        authorize(request)
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > max_body_bytes:
                raise HTTPException(status_code=413, detail="physical request exceeds body budget")
            body.extend(chunk)
        raw, metrics = "", {}
        try:
            upload = PhysicalUpload.model_validate_json(bytes(body))
            with tempfile.TemporaryDirectory(prefix="physical-", dir=media_root) as temporary:
                clean = _write_inline_media(upload, Path(temporary), max_file_bytes)
                # NativeModel owns its serialized generation lock. Keep HTTP health
                # and other requests responsive while one GPU request runs.
                from starlette.concurrency import run_in_threadpool
                raw, metrics = await run_in_threadpool(backend.generate_text, clean)
                decision = validate_physical_decision(PhysicalDecision.model_validate_json(raw), clean["window"],
                    known_track_ids={item["track_id"] for item in clean["tracks"]})
                response = PhysicalResponse(decision=decision, raw_output=raw, metrics=metrics)
                return response.model_dump()
        except (ValueError, OSError, KeyError) as exc:
            response = PhysicalResponse(raw_output=raw, metrics=metrics, error=str(exc))
            return JSONResponse(status_code=422, content=response.model_dump())
        except Exception as exc:
            response = PhysicalResponse(raw_output=raw, metrics=metrics, error=f"physical backend failed: {type(exc).__name__}: {exc}")
            return JSONResponse(status_code=503, content=response.model_dump())

    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter")
    parser.add_argument("--backend", choices=["native", "vllm"], default="native")
    parser.add_argument("--engine-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--served-model", default="physical-observer")
    parser.add_argument("--media-root", default="artifacts/physical-upload-media")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--quantization", choices=["none", "4bit"], default="none")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    args = parser.parse_args()
    token = os.environ.get("PHYSICAL_API_TOKEN", "")
    if len(token) < 16:
        parser.error("set PHYSICAL_API_TOKEN to an explicit token of at least16 characters")
    Path(args.media_root).mkdir(parents=True, exist_ok=True)
    config = ModelConfig(adapter_path=args.adapter, dataset_root=str(Path(args.media_root).resolve()),
        quantization=args.quantization, max_frames=args.max_frames, max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens, merge_adapter=False)
    if args.backend == "vllm":
        from .physical_vllm import PhysicalVLLMModel
        backend = PhysicalVLLMModel(config, base_url=args.engine_url, served_model=args.served_model)
        if not backend.ready():
            parser.error("physical vLLM engine is not ready")
    else:
        backend = PhysicalNativeModel(config)
    uvicorn.run(create_physical_app(backend, api_token=token, media_root=config.dataset_root),
                host=args.host, port=args.port)


if __name__ == "__main__":
    main()
