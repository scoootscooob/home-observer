"""Authenticated local-file inference API. The endpoint only proposes Decisions."""
from __future__ import annotations

import argparse
import asyncio
import hmac
import ipaddress
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import httpx

from .model import DEFAULT_MODEL, DEFAULT_REVISION, ModelConfig, NativeModel, prepare_request
from .transport import MAX_BODY_BYTES, unpack_media


def is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def validate_bind(host: str, token: str | None) -> None:
    if not is_loopback(host) and not token:
        raise ValueError("non-loopback bind requires HOME_OBSERVER_API_TOKEN")


def authorized(authorization: str | None, token: str | None) -> bool:
    if token is None:
        return True
    if authorization is None or not authorization.startswith("Bearer "):
        return False
    return hmac.compare_digest(authorization[7:].encode(), token.encode())


def create_app(config: ModelConfig | None = None, *, backend: Any = None,
               host: str = "127.0.0.1", api_token: str | None = None):
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException, Request
    from starlette.concurrency import run_in_threadpool

    from .schema import InferenceRequest, InferenceResponse
    validate_bind(host, api_token)
    config = config or ModelConfig()

    @asynccontextmanager
    async def lifespan(app):
        app.state.backend = backend if backend is not None else NativeModel(config)
        if hasattr(app.state.backend, "ready") and not await run_in_threadpool(app.state.backend.ready):
            raise RuntimeError("inference engine is not ready")
        try:
            yield
        finally:
            if hasattr(app.state.backend, "close"):
                await run_in_threadpool(app.state.backend.close)

    app = FastAPI(title="Home Observer", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.backend = backend
    upload_slots = asyncio.Semaphore(2)

    @app.middleware("http")
    async def access_guard(request: Request, call_next):
        from fastapi.responses import JSONResponse
        if not authorized(request.headers.get("authorization"), api_token):
            return JSONResponse({"detail": "unauthorized"}, status_code=401,
                                headers={"WWW-Authenticate": "Bearer"})
        declared = request.headers.get("content-length")
        if declared:
            try:
                maximum = MAX_BODY_BYTES if request.url.path == "/observe_upload" else 1024 * 1024
                if int(declared) > maximum:
                    return JSONResponse({"detail": "request too large"}, status_code=413)
            except ValueError:
                return JSONResponse({"detail": "invalid content length"}, status_code=400)
        return await call_next(request)

    @app.get("/health")
    async def health():
        ready = app.state.backend is not None
        if ready and hasattr(app.state.backend, "ready"):
            ready = await run_in_threadpool(app.state.backend.ready)
        return {"status": "ready" if ready else "unavailable",
                "model_id": config.model_id, "model_revision": config.revision,
                "adapter_configured": bool(config.adapter_path),
                "inference_backend": type(app.state.backend).__name__,
                "schema_constraints": getattr(app.state.backend, "schema_constraints", False),
                "schema_contract": getattr(app.state.backend, "schema_contract", None),
                "brightness_enabled": getattr(app.state.backend, "allow_brightness", None),
                "decision_field_order": getattr(app.state.backend, "decision_field_order", None),
                "model_identity_source": "deployment_configuration",
                "context_mode": "bounded_window"}

    # Concrete annotation assignment avoids postponed local annotations in FastAPI.
    async def observe(payload):
        try:
            request = prepare_request(payload.model_dump(), config)
            response = await run_in_threadpool(app.state.backend.observe, request)
            return InferenceResponse.model_validate(response)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail="inference engine request failed") from exc
    observe.__annotations__["payload"] = InferenceRequest
    app.post("/observe", response_model=InferenceResponse)(observe)

    async def observe_upload(http_request):
        async with upload_slots:
            body = bytearray()
            async for chunk in http_request.stream():
                if len(body) + len(chunk) > MAX_BODY_BYTES:
                    raise HTTPException(status_code=413, detail="request too large")
                body.extend(chunk)
            root = Path(config.dataset_root).resolve(strict=True)
            try:
                payload = json.loads(body)
                with tempfile.TemporaryDirectory(prefix=".incoming-", dir=root) as directory:
                    request, size = unpack_media(payload, directory)
                    request = prepare_request(request, config)
                    response = await run_in_threadpool(app.state.backend.observe, request)
                    validated = InferenceResponse.model_validate(response)
                    validated.metrics["received_media_bytes"] = size
                    return validated
            except (ValueError, OSError, KeyError, TypeError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except httpx.HTTPError as exc:
                raise HTTPException(status_code=502, detail="inference engine request failed") from exc
    observe_upload.__annotations__["http_request"] = Request
    app.post("/observe_upload", response_model=InferenceResponse)(observe_upload)
    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--adapter")
    merging = parser.add_mutually_exclusive_group()
    merging.add_argument('--merge-adapter', dest='merge_adapter', action='store_const', const=True,
                         help='Explicit experimental BF16 merge; validate behavior before use')
    merging.add_argument('--no-merge-adapter', dest='merge_adapter', action='store_const', const=False)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--quantization", choices=["4bit", "none"], default="4bit")
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--backend", choices=["native", "vllm"], default="native")
    parser.add_argument("--engine-endpoint", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--engine-model", default="home-observer")
    parser.add_argument("--no-schema-constraints", action="store_true")
    parser.add_argument("--allow-brightness", action="store_true",
                        help="Advertise optional brightness for dimmable lights; default is on/off actions")
    parser.add_argument("--decision-field-order", choices=['summary-first', 'actions-first'], default='summary-first')
    args = parser.parse_args()
    token = os.environ.get("HOME_OBSERVER_API_TOKEN") or None
    validate_bind(args.host, token)
    config = ModelConfig(model_id=args.model, revision=args.revision, adapter_path=args.adapter,
                         merge_adapter=args.merge_adapter,
                         dataset_root=args.dataset_root, quantization=args.quantization,
                         max_input_tokens=args.max_input_tokens, max_new_tokens=args.max_new_tokens)
    backend = None
    if args.backend == "vllm":
        from .vllm_backend import VLLMModel
        backend = VLLMModel(config, base_url=args.engine_endpoint, served_model=args.engine_model,
                            schema_constraints=not args.no_schema_constraints,
                            allow_brightness=args.allow_brightness, decision_field_order=args.decision_field_order)
    uvicorn.run(create_app(config, backend=backend, host=args.host, api_token=token), host=args.host, port=args.port,
                proxy_headers=False, access_log=False, limit_concurrency=8)


if __name__ == "__main__":
    main()
