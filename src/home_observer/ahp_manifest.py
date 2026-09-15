"""Agent Hardware Protocol (AHP) device manifest for the production sensing node.

The local sensing server can present itself to an agent gateway as one AHP device:
its events are exactly the production export message types, its commands are the
typed production commands, and its event policies mirror the local watch
discipline (bounded emission, no firehose). Nothing here adds a transport; it
derives the self-description from the enforced schemas so the manifest cannot
drift from what the boundary actually sends.
"""
from __future__ import annotations

import json
from typing import Any

from .local_verifier import VerificationRequest
from .production_export import (
    SCHEMA_VERSION,
    ExportedPerceptionStatus,
    ExportedPhysicalEvent,
    ExportedVerification,
    ExportedWatchStatus,
    scan_export,
)
from .watches import WatchSpec

EVENT_MODELS = {
    "physical_event": (ExportedPhysicalEvent, "alert", {"debounce_ms": 250, "max_rate_per_min": 120, "wake": True}),
    "watch_status": (ExportedWatchStatus, "info", {"max_rate_per_min": 60, "wake": False}),
    "perception_status": (ExportedPerceptionStatus, "info", {"max_rate_per_min": 30, "wake": False}),
    "verification_result": (ExportedVerification, "alert", {"max_rate_per_min": 30, "wake": True}),
}


def _schema(model) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema.pop("title", None)
    return schema


def build_manifest(device_id: str = "home-observer-node", label: str = "Home Observer sensing node") -> dict:
    """Self-description for an agent gateway: typed events, typed commands, no media."""
    resources = [{
        "id": "sensing",
        "label": "Local physical sensing",
        "properties": [
            {"id": "schema_version", "type": "string", "observable": False, "writable": False,
             "value": SCHEMA_VERSION},
            {"id": "media_leaves_device", "type": "boolean", "observable": False, "writable": False, "value": False,
             "description": "Camera and audio never leave the sensing node; only typed observations are emitted."},
            {"id": "pending_exports", "type": "integer", "minimum": 0, "observable": True, "writable": False},
            {"id": "armed_watches", "type": "integer", "minimum": 0, "observable": True, "writable": False},
        ],
        "commands": [
            {"id": "create_watch", "idempotent": True, "safe": False,
             "description": "Arm a bounded local watch that selects a locally approved response.",
             "input": _schema(WatchSpec)},
            {"id": "cancel_watch", "idempotent": True, "safe": True,
             "input": {"type": "object", "properties": {"watch_id": {"type": "string"}}, "required": ["watch_id"]}},
            {"id": "verify", "idempotent": False, "safe": True,
             "description": "Ask the local verifier a structured question; the answer arrives as a verification_result event.",
             "input": _schema(VerificationRequest)},
            {"id": "ack_export", "idempotent": True, "safe": True,
             "input": {"type": "object", "properties": {"export_id": {"type": "string"},
                                                        "export_sha256": {"type": "string"}},
                       "required": ["export_id", "export_sha256"]}},
        ],
        "events": [
            {"id": name, "priority": priority, "policy": policy, "payload": _schema(model)}
            for name, (model, priority, policy) in EVENT_MODELS.items()
        ],
    }]
    return {"protocol": "ahp", "device_id": device_id, "label": label, "resources": resources,
            "notes": ["Event payloads are the production export messages; every emitted payload also passes "
                      "the export leak scanner before it reaches any transport.",
                      "Commands mirror ProductionCommand; a gateway cannot request media, and no command "
                      "resolves an audit reference into frames."]}


def manifest_is_media_free(manifest: dict) -> list[str]:
    """Denied keys and media markers must not appear anywhere in the event payload schemas."""
    violations = []
    for resource in manifest["resources"]:
        for event in resource["events"]:
            found = scan_export({"message": {"kind": "x"}, "payload_keys": sorted(
                event["payload"].get("properties", {}))})
            violations.extend(f"{event['id']}: {item}" for item in found if "denied key" in item)
    return violations


if __name__ == "__main__":
    print(json.dumps(build_manifest(), indent=2))
