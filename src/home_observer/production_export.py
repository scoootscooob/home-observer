"""Enforced production export boundary between local sensing and the cloud frontier.

Everything the frontier receives in production passes through this module. Each
message is a versioned, typed allowlist built from explicitly selected fields.
Nothing here forwards a private outbox envelope, a journal row, model output, a
media path, a recording manifest or free text. The sanitized payload is hashed
after sanitization; that exact payload is persisted, retried and acknowledged,
and an opaque audit reference maps it back to the private local record.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import sqlite3
import uuid
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, ValidationError, model_validator

from .schema import StrictModel
from .watches import ID, WatchEngine, WatchSpec, _digest, _json

SCHEMA_VERSION = "home-observer.production-export/1"
TOKEN = r"^[a-z][a-z0-9_-]{0,47}$"
EVENT_ID = r"^pevt_[a-f0-9]{32}$"
TRACK_ID = r"^phys_[a-f0-9]{32}$"
EXPORT_ID = r"^export_[a-f0-9]{32}$"
AUDIT_REF = r"^audit_[a-f0-9]{24}$"
VERIFICATION_ID = r"^pver_[a-f0-9]{32}$"
CLASS_NAME = r"^[A-Za-z][A-Za-z0-9]{0,63}$"
SHA256 = r"^[a-f0-9]{64}$"

WATCH_STATUS_BY_TOPIC = {
    "watch.created": "armed",
    "watch.cancelled": "cancelled",
    "watch.expired": "expired",
    "watch.matched": "matched",
    "watch.triggered": "awaiting_delivery",
    "watch.response_verified": "response_verified",
    "watch.response_uncertain": "response_uncertain",
    "watch.missed_deadline": "missed_deadline",
    "watch.outcome_verified": "outcome_verified",
    "watch.outcome_unknown": "outcome_unknown",
}
PERCEPTION_TOPICS = {"perception.completed", "perception.failed"}
EXPORT_TOPICS = ("physical.event", *WATCH_STATUS_BY_TOPIC, *sorted(PERCEPTION_TOPICS), "verification.result")
VERIFICATION_QUESTIONS = ("moved_from_initial_location", "clear_of_support", "still_present", "pickup_completed")
VERIFICATION_REASONS = (
    "geometric_displacement", "geometric_and_learned_agreement", "no_displacement_observed",
    "learned_evidence_contradicts", "localization_uncertain", "insufficient_tracking_history",
    "requires_learned_evidence", "learned_model_unavailable", "learned_output_invalid",
    "subject_unknown", "reference_event_unknown", "visible_in_window", "not_observed_in_window",
    "ambiguous_displacement",
)

Token = Annotated[str, Field(pattern=TOKEN)]
TrackRef = Annotated[str, Field(pattern=TRACK_ID)]
Wall = Annotated[float, Field(strict=True)]


class ExportBlocked(ValueError):
    """A private record could not be turned into a safe export. Nothing is sent."""


class ProductionModeError(RuntimeError):
    """The runtime is configured in a way production must refuse."""


class ExportPolicy(StrictModel):
    """Local allowlists. Values outside the vocabulary export as ``other``; never as free text."""

    schema_version: Literal["home-observer.production-export/1"] = SCHEMA_VERSION
    event_kinds: list[Token] = Field(min_length=1, max_length=256)
    object_categories: list[Token] = Field(min_length=1, max_length=512)
    areas: dict[Annotated[str, Field(pattern=ID)], Token] = Field(default_factory=dict, max_length=64)
    time_decimals: int = Field(default=3, ge=0, le=6)
    confidence_decimals: int = Field(default=2, ge=0, le=3)

    @model_validator(mode="after")
    def reserved_tokens(self):
        for values in (self.event_kinds, self.object_categories):
            if len(values) != len(set(values)):
                raise ValueError("export vocabulary contains duplicates")
            if "other" in values:
                raise ValueError("'other' is reserved for values outside the vocabulary")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "ExportPolicy":
        return cls.model_validate(json.loads(Path(path).read_text()))


def normalize_token(value) -> str | None:
    if not isinstance(value, str):
        return None
    token = re.sub(r"[^a-z0-9_-]+", "-", value.strip().lower()).strip("-_")[:48]
    return token if re.fullmatch(TOKEN, token) else None


def vocabulary_token(value, vocabulary) -> tuple[str, bool]:
    token = normalize_token(value)
    if token is not None and token in vocabulary:
        return token, True
    return "other", False


def uncertainty_level(confidence: float, origin: str) -> str:
    """Bounded bands, not calibration. Heuristic geometric scores never export as low."""
    level = "low" if confidence >= 0.8 else "medium" if confidence >= 0.5 else "high"
    if origin == "geometric" and level == "low":
        level = "medium"
    return level


class ExportedPhysicalEvent(StrictModel):
    message_type: Literal["physical_event"] = "physical_event"
    event_id: str = Field(pattern=EVENT_ID)
    origin: Literal["learned", "geometric"]
    kind: Token
    kind_in_vocabulary: bool
    object_category: Token | None = None
    category_in_vocabulary: bool
    subject_ids: list[TrackRef] = Field(default_factory=list, max_length=8)
    area: Token | None = None
    started_at: Wall
    ended_at: Wall
    available_at: Wall
    confidence: float = Field(strict=True, ge=0, le=1)
    uncertainty_level: Literal["low", "medium", "high"]
    evidence_count: int = Field(ge=0, le=128)
    audit_ref: str = Field(pattern=AUDIT_REF)

    @model_validator(mode="after")
    def ordered(self):
        if self.ended_at < self.started_at or self.available_at < self.ended_at:
            raise ValueError("exported event clocks are out of order")
        return self


class ExportedWatchStatus(StrictModel):
    message_type: Literal["watch_status"] = "watch_status"
    watch_id: str = Field(pattern=ID)
    commitment_id: str | None = Field(default=None, pattern=ID)
    response_id: str | None = Field(default=None, pattern=ID)
    status: Literal[
        "armed", "cancelled", "expired", "matched", "awaiting_delivery", "response_verified",
        "response_uncertain", "missed_deadline", "outcome_verified", "outcome_unknown",
    ]
    event_id: str | None = Field(default=None, pattern=EVENT_ID)
    outcome_event_id: str | None = Field(default=None, pattern=EVENT_ID)
    eligible_at: Wall | None = None
    matched_at: Wall | None = None
    response_reserved_at: Wall | None = None
    response_sent_at: Wall | None = None
    response_verified_at: Wall | None = None
    outcome_verified_at: Wall | None = None
    reported_at: Wall | None = None
    response_timely: bool | None = None
    verification_kind: Literal["ha_state_readback", "coordinator_delivery_ack"] | None = None
    response_result: Literal["verified", "uncertain", "not_sent"] | None = None
    readback_state: Literal["on", "off", "unknown"] | None = None
    failure_class: str | None = Field(default=None, pattern=CLASS_NAME)
    audit_ref: str = Field(pattern=AUDIT_REF)


class ExportedPerceptionStatus(StrictModel):
    message_type: Literal["perception_status"] = "perception_status"
    status: Literal["completed", "invalid", "failed"]
    failure_class: str | None = Field(default=None, pattern=CLASS_NAME)
    area: Token | None = None
    source_started_at: Wall
    source_ended_at: Wall
    inference_started_at: Wall | None = None
    emitted_at: Wall
    latency_s: float | None = Field(default=None, strict=True, ge=0)
    sampled_frames: int = Field(ge=0, le=512)
    full_sequence_frames: int = Field(ge=0, le=100000)
    objects_accepted: int = Field(ge=0, le=128)
    observations_accepted: int = Field(ge=0, le=128)
    events_accepted: int = Field(ge=0, le=64)
    audit_ref: str = Field(pattern=AUDIT_REF)


class ExportedVerification(StrictModel):
    message_type: Literal["verification_result"] = "verification_result"
    verification_id: str = Field(pattern=VERIFICATION_ID)
    request_id: str = Field(pattern=ID)
    question: Literal[VERIFICATION_QUESTIONS]
    subject_id: TrackRef
    verdict: Literal["confirmed", "contradicted", "unknown"]
    confidence: float = Field(strict=True, ge=0, le=1)
    uncertainty_level: Literal["low", "medium", "high"]
    reason_code: Literal[VERIFICATION_REASONS]
    evaluated_at: Wall
    evidence_started_at: Wall | None = None
    evidence_ended_at: Wall | None = None
    frames_examined: int = Field(ge=0, le=100000)
    audit_ref: str = Field(pattern=AUDIT_REF)


ExportMessage = Annotated[
    ExportedPhysicalEvent | ExportedWatchStatus | ExportedPerceptionStatus | ExportedVerification,
    Field(discriminator="message_type"),
]


class ExportEnvelope(StrictModel):
    """The only shape that leaves the host. ``export_sha256`` covers the sanitized content."""

    schema_version: Literal["home-observer.production-export/1"] = SCHEMA_VERSION
    export_id: str = Field(pattern=EXPORT_ID)
    topic: Literal[EXPORT_TOPICS]
    message: ExportMessage
    created_at: Wall
    export_sha256: str = Field(pattern=SHA256)
    attempt: int = Field(ge=1, le=100000)
    exported_at: Wall

    @model_validator(mode="after")
    def hash_matches_content(self):
        if self.export_sha256 != export_content_sha256(self.export_id, self.topic, self.message.model_dump()):
            raise ValueError("export hash does not match its sanitized content")
        return self


def export_content_sha256(export_id: str, topic: str, message: dict) -> str:
    """Hash of the canonical validated message dump, so defaults never change the hash."""
    content = TypeAdapter(ExportMessage).validate_python(message).model_dump()
    return _digest({"schema_version": SCHEMA_VERSION, "export_id": export_id, "topic": topic, "message": content})


# --- Defense in depth: a scanner that runs on the serialized export -------------------------------

DENIED_KEYS = frozenset({
    "path", "paths", "url", "urls", "uri", "href", "manifest", "full_sequence_manifest", "raw_output",
    "content_base64", "base64", "media", "jpeg", "jpg", "png", "image", "images", "frame", "frames",
    "audio", "audio_path", "embedding", "embeddings", "features", "transcript", "ocr", "description",
    "uncertainty", "text", "message", "error", "summary", "spec", "result", "window", "clips",
    "payload", "context", "prompt", "narration", "evidence_ids", "pre_evidence_ids", "post_evidence_ids",
    "box", "boxes", "detection", "detections", "objects", "observations", "events", "tracks",
    "adapter", "model_id", "engine_url", "camera_id", "attributes", "data", "traceback", "exception",
})
PATH_LIKE = re.compile(r"[/\\]|://|^data:|base64|\.\.", re.IGNORECASE)
EXTENSION = re.compile(
    r"\.(jpe?g|png|webp|gif|bmp|tiff?|heic|mp4|m4v|mov|avi|mkv|webm|wav|flac|ogg|mp3|m4a|aac|json|jsonl"
    r"|sqlite3?|db|txt|log|csv|npy|npz|pt|pth|safetensors|bin|gguf|tar|gz|zip)(?![a-z0-9])",
    re.IGNORECASE,
)
BASE64_RUN = re.compile(r"[A-Za-z0-9+/=]{48,}")
KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
MAX_STRING, MAX_DEPTH, MAX_ITEMS, MAX_BYTES = 128, 6, 64, 16384


def scan_export(value) -> list[str]:
    """Return every violation found in a serialized export; an empty list means clean."""
    violations = []
    try:
        serialized = json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return ["export is not finite, serializable JSON"]
    if len(serialized.encode()) > MAX_BYTES:
        violations.append("export exceeds byte budget")

    def check_string(text, where):
        if len(text) > MAX_STRING:
            violations.append(f"{where}: string exceeds {MAX_STRING} characters")
        if re.search(r"\s", text):
            violations.append(f"{where}: free text with whitespace")
        if not text.isascii() or not text.isprintable():
            violations.append(f"{where}: non-printable or non-ASCII text")
        if PATH_LIKE.search(text):
            violations.append(f"{where}: path, URL or encoded media marker")
        if EXTENSION.search(text):
            violations.append(f"{where}: media/file extension")
        if BASE64_RUN.search(text) and not re.fullmatch(r"[a-f0-9]{1,64}", text):
            violations.append(f"{where}: base64-like run")

    def walk(node, where, depth):
        if depth > MAX_DEPTH:
            violations.append(f"{where}: nesting exceeds {MAX_DEPTH}")
            return
        if isinstance(node, dict):
            if len(node) > MAX_ITEMS:
                violations.append(f"{where}: too many keys")
            for key, item in node.items():
                if not isinstance(key, str) or not KEY_PATTERN.fullmatch(key):
                    violations.append(f"{where}: invalid key")
                    continue
                if key in DENIED_KEYS and not (depth == 0 and key == "message"):
                    violations.append(f"{where}.{key}: denied key")
                    continue
                if key == "schema_version" and depth == 0:
                    if item != SCHEMA_VERSION:
                        violations.append(f"{where}.{key}: unknown schema version")
                    continue
                walk(item, f"{where}.{key}", depth + 1)
        elif isinstance(node, list):
            if len(node) > MAX_ITEMS:
                violations.append(f"{where}: too many items")
            for index, item in enumerate(node):
                walk(item, f"{where}[{index}]", depth + 1)
        elif isinstance(node, str):
            check_string(node, where)
        elif isinstance(node, bool) or node is None:
            return
        elif isinstance(node, (int, float)):
            if not math.isfinite(node):
                violations.append(f"{where}: non-finite number")
        else:
            violations.append(f"{where}: unsupported type {type(node).__name__}")

    walk(value, "$", 0)
    return violations


# --- Private store for exports, acknowledgements and the audit mapping ----------------------------

class ExportStore:
    """Exports, their leases/acks and the private audit mapping, in local SQLite only."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS production_exports(
          export_id TEXT PRIMARY KEY, topic TEXT NOT NULL, private_delivery_id TEXT UNIQUE NOT NULL,
          private_payload_sha256 TEXT NOT NULL, export_json TEXT NOT NULL, export_sha256 TEXT NOT NULL,
          created_at REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, next_attempt_at REAL NOT NULL,
          lease_until REAL, lease_token TEXT, delivered_at REAL, acked_at REAL, last_error TEXT);
        CREATE TABLE IF NOT EXISTS production_export_audit(
          audit_ref TEXT PRIMARY KEY, export_id TEXT NOT NULL, private_delivery_id TEXT NOT NULL,
          topic TEXT NOT NULL, local_refs_json TEXT NOT NULL, created_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS production_export_blocked(
          private_delivery_id TEXT NOT NULL, topic TEXT NOT NULL, reason_class TEXT NOT NULL,
          violations_json TEXT NOT NULL, blocked_at REAL NOT NULL);
        """)

    def close(self):
        self.db.close()

    def record(self, *, export_id, topic, private_delivery_id, private_payload_sha256, export_json,
               export_sha256, audit_ref, local_refs, now) -> bool:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            inserted = self.db.execute(
                """INSERT OR IGNORE INTO production_exports(export_id,topic,private_delivery_id,
                private_payload_sha256,export_json,export_sha256,created_at,next_attempt_at)
                VALUES(?,?,?,?,?,?,?,?)""",
                (export_id, topic, private_delivery_id, private_payload_sha256, export_json, export_sha256,
                 now, now),
            ).rowcount
            if inserted:
                self.db.execute(
                    "INSERT OR IGNORE INTO production_export_audit VALUES(?,?,?,?,?,?)",
                    (audit_ref, export_id, private_delivery_id, topic, _json(local_refs), now),
                )
            self.db.commit()
            return bool(inserted)
        except BaseException:
            self.db.rollback()
            raise

    def block(self, private_delivery_id, topic, reason_class, violations, now):
        self.db.execute(
            "INSERT INTO production_export_blocked VALUES(?,?,?,?,?)",
            (private_delivery_id, topic, reason_class, _json(violations), now),
        )

    def claim(self, *, now, limit=32, lease_s=30) -> list[dict]:
        if not 1 <= limit <= 1000 or not 0 < lease_s <= 3600:
            raise ValueError("invalid export lease bounds")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            rows = self.db.execute(
                """SELECT * FROM production_exports WHERE acked_at IS NULL AND next_attempt_at<=?
                AND (lease_until IS NULL OR lease_until<=?) ORDER BY created_at,export_id LIMIT ?""",
                (now, now, limit),
            ).fetchall()
            claimed = []
            for row in rows:
                token = uuid.uuid4().hex
                self.db.execute(
                    "UPDATE production_exports SET attempts=attempts+1,lease_until=?,lease_token=? WHERE export_id=?",
                    (now + lease_s, token, row["export_id"]),
                )
                export = json.loads(row["export_json"])
                claimed.append({**export, "attempt": row["attempts"] + 1, "exported_at": now, "lease_token": token})
            self.db.commit()
            return claimed
        except BaseException:
            self.db.rollback()
            raise

    def delivery_result(self, export_id, lease_token, *, now, delivered, error=None) -> bool:
        row = self.db.execute(
            "SELECT attempts FROM production_exports WHERE export_id=? AND lease_token=?", (export_id, lease_token)
        ).fetchone()
        if row is None:
            return False
        delay = min(300, 2 ** min(row["attempts"], 8))
        self.db.execute(
            """UPDATE production_exports SET delivered_at=CASE WHEN ? THEN COALESCE(delivered_at,?) ELSE delivered_at END,
            next_attempt_at=?,lease_until=NULL,lease_token=NULL,last_error=? WHERE export_id=? AND lease_token=?""",
            (delivered, now, now + delay, error, export_id, lease_token),
        )
        return True

    def acknowledge(self, export_id, export_sha256, *, now) -> dict:
        """Bind the ACK to the exact exported hash; return the private mapping for the local ACK."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT * FROM production_exports WHERE export_id=?", (export_id,)).fetchone()
            if row is None or not hmac.compare_digest(row["export_sha256"], export_sha256):
                raise ValueError("acknowledgement must name an exported ID and its exact export hash")
            if row["delivered_at"] is None:
                raise ValueError("export was not yet delivered by a transport")
            changed = row["acked_at"] is None
            if changed:
                self.db.execute(
                    "UPDATE production_exports SET acked_at=?,lease_until=NULL,lease_token=NULL WHERE export_id=?",
                    (now, export_id),
                )
            self.db.commit()
            return {"changed": changed, "private_delivery_id": row["private_delivery_id"],
                    "private_payload_sha256": row["private_payload_sha256"], "topic": row["topic"]}
        except BaseException:
            self.db.rollback()
            raise

    def audit(self, audit_ref) -> dict | None:
        row = self.db.execute("SELECT * FROM production_export_audit WHERE audit_ref=?", (audit_ref,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["local_refs"] = json.loads(result.pop("local_refs_json"))
        return result

    def exports(self) -> list[dict]:
        rows = self.db.execute("SELECT * FROM production_exports ORDER BY created_at,export_id").fetchall()
        return [dict(row) for row in rows]

    def blocked(self) -> list[dict]:
        rows = self.db.execute("SELECT * FROM production_export_blocked ORDER BY blocked_at").fetchall()
        return [{**dict(row), "violations": json.loads(row["violations_json"])} for row in rows]

    def stats(self) -> dict:
        row = self.db.execute(
            """SELECT COUNT(*) AS exports, SUM(delivered_at IS NOT NULL) AS delivered,
            SUM(acked_at IS NOT NULL) AS acked, (SELECT COUNT(*) FROM production_export_blocked) AS blocked,
            (SELECT COUNT(*) FROM production_export_audit) AS audit_records FROM production_exports"""
        ).fetchone()
        return {key: (row[key] or 0) for key in row.keys()}


# --- Building exports from private envelopes ---------------------------------------------------

def load_export_key(path: str | Path) -> bytes:
    """A private local key for opaque references; created with mode 600 on first use."""
    path = Path(path)
    if path.is_symlink():
        raise ProductionModeError("export key path cannot be a symlink")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(secrets.token_hex(32) + "\n")
    key = bytes.fromhex(path.read_text().strip())
    if len(key) < 32:
        raise ProductionModeError("export key must contain at least 32 bytes")
    return key


def _number(value, *, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ExportBlocked("expected a finite number")
    return float(value)


def _class_name(value):
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("type") or value.get("error_type")
    if not isinstance(value, str):
        return None
    # An exception *class name* is a bounded token; exception text never crosses.
    name = value.split(":")[0].split(".")[-1].strip()
    return name if re.fullmatch(CLASS_NAME, name) else "UnknownError"


class ProductionExporter:
    """Turns private outbox envelopes into typed exports; fails closed on anything else."""

    def __init__(self, engine: WatchEngine, store: ExportStore, policy: ExportPolicy, key: bytes):
        if not isinstance(policy, ExportPolicy):
            raise ProductionModeError("production export requires an explicit ExportPolicy")
        if not isinstance(key, (bytes, bytearray)) or len(key) < 32:
            raise ProductionModeError("production export requires a private key of at least 32 bytes")
        self.engine, self.store, self.policy, self._key = engine, store, policy, bytes(key)

    def _ref(self, prefix, purpose, value, length):
        return prefix + hmac.new(self._key, (purpose + "\0" + value).encode(), hashlib.sha256).hexdigest()[:length]

    def _round_time(self, value):
        return round(_number(value), self.policy.time_decimals)

    def _optional_time(self, value):
        return None if value is None else self._round_time(value)

    def _confidence(self, value):
        number = _number(value)
        if not 0 <= number <= 1:
            raise ExportBlocked("confidence outside [0, 1]")
        return round(number, self.policy.confidence_decimals)

    def _area(self, evidence_ids) -> str | None:
        cameras = set()
        for identifier in evidence_ids:
            if not isinstance(identifier, str) or ":" not in identifier:
                return None
            camera, _, sequence = identifier.rpartition(":")
            if not sequence.isdigit():
                return None
            cameras.add(camera)
        if len(cameras) != 1:
            return None
        return self.policy.areas.get(cameras.pop())

    def _watch_context(self, watch_id):
        row = self.engine.store.get(watch_id) if isinstance(watch_id, str) else None
        if not row:
            return None, None
        try:
            spec = WatchSpec.model_validate(row["spec"])
        except ValidationError:
            return None, None
        return spec.context.commitment_id, spec.response_id

    def build(self, topic: str, payload: dict, audit_ref: str):
        """Explicit field selection per topic. Unknown topics are blocked, never forwarded."""
        if not isinstance(payload, dict):
            raise ExportBlocked("private payload must be an object")
        if topic == "physical.event":
            origin = payload.get("origin")
            if origin not in ("learned", "geometric"):
                raise ExportBlocked("only learned or geometric events are exportable")
            kind, kind_known = vocabulary_token(payload.get("kind"), self.policy.event_kinds)
            label = payload.get("object_label")
            category, category_known = (None, False) if label is None else vocabulary_token(
                label, self.policy.object_categories)
            subjects = payload.get("subject_ids") or []
            evidence = [*(payload.get("pre_evidence_ids") or []), *(payload.get("post_evidence_ids") or [])]
            confidence = self._confidence(payload.get("confidence"))
            message = ExportedPhysicalEvent(
                event_id=str(payload.get("event_id")), origin=origin, kind=kind, kind_in_vocabulary=kind_known,
                object_category=category, category_in_vocabulary=category_known,
                subject_ids=[str(item) for item in subjects], area=self._area(evidence),
                started_at=self._round_time(payload.get("started_at")),
                ended_at=self._round_time(payload.get("ended_at")),
                available_at=self._round_time(payload.get("event_available_at", payload.get("available_at"))),
                confidence=confidence, uncertainty_level=uncertainty_level(confidence, origin),
                evidence_count=len(evidence), audit_ref=audit_ref)
            refs = {"event_id": payload.get("event_id"), "window_id": payload.get("window_id"),
                    "source_event_id": payload.get("source_event_id"), "evidence_ids": evidence}
            return message, refs
        if topic in WATCH_STATUS_BY_TOPIC:
            watch_id = payload.get("watch_id")
            commitment_id, response_id = self._watch_context(watch_id)
            result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
            response_result = None
            if topic == "watch.response_verified":
                response_result = "verified"
            elif topic == "watch.response_uncertain":
                response_result = "uncertain"
            elif topic == "watch.missed_deadline" and payload.get("not_sent_at") is not None:
                response_result = "not_sent"
            readback = result.get("state") if result.get("state") in ("on", "off") else (
                "unknown" if result else None)
            message = ExportedWatchStatus(
                watch_id=str(watch_id), commitment_id=commitment_id, response_id=response_id,
                status=WATCH_STATUS_BY_TOPIC[topic],
                event_id=payload.get("event_id") if isinstance(payload.get("event_id"), str) else None,
                outcome_event_id=payload.get("outcome_event_id") if isinstance(payload.get("outcome_event_id"), str) else None,
                eligible_at=self._optional_time(payload.get("eligible_at")),
                matched_at=self._optional_time(payload.get("matched_at")),
                response_reserved_at=self._optional_time(payload.get("response_reserved_at")),
                response_sent_at=self._optional_time(payload.get("response_sent_at")),
                response_verified_at=self._optional_time(payload.get("response_verified_at")),
                outcome_verified_at=self._optional_time(payload.get("outcome_verified_at")),
                reported_at=self._optional_time(payload.get("reported_at", payload.get("expired_at",
                    payload.get("cancelled_at", payload.get("not_sent_at", payload.get("created_at")))))),
                response_timely=payload.get("response_timely") if isinstance(payload.get("response_timely"), bool) else None,
                verification_kind=payload.get("verification_kind") if payload.get("verification_kind") in (
                    "ha_state_readback", "coordinator_delivery_ack") else None,
                response_result=response_result, readback_state=readback,
                failure_class=_class_name(result.get("error_type")) if result else None, audit_ref=audit_ref)
            refs = {"watch_id": watch_id, "event_id": payload.get("event_id"),
                    "response_key": payload.get("response_key"), "evidence_ids": payload.get("evidence_ids")}
            return message, refs
        if topic in PERCEPTION_TOPICS:
            window = payload.get("window") if isinstance(payload.get("window"), dict) else {}
            result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
            mapping = payload.get("clock_mapping") if isinstance(payload.get("clock_mapping"), dict) else None
            if not mapping:
                raise ExportBlocked("perception status requires an explicit clock mapping")
            source_origin = _number(mapping.get("source_origin"))
            wall_origin, scale = _number(mapping.get("wall_origin")), _number(mapping.get("scale", 1.0))
            if scale <= 0:
                raise ExportBlocked("clock mapping scale must be positive")

            def wall(source_time):
                return self._round_time(wall_origin + (_number(source_time) - source_origin) * scale)

            decision = result.get("decision") if isinstance(result.get("decision"), dict) else None
            error = result.get("error")
            raw_output = result.get("raw_output")
            if topic == "perception.completed" and decision is not None and not error:
                status, failure_class = "completed", None
            elif isinstance(error, dict) and _class_name(error) in ("ValueError", "ValidationError"):
                status, failure_class = "invalid", _class_name(error)
            elif isinstance(error, dict):
                status, failure_class = "failed", _class_name(error)
            elif isinstance(error, str) and isinstance(raw_output, str) and raw_output.strip():
                # A generation exists but failed validation; the text itself never crosses.
                status, failure_class = "invalid", "ValidationError"
            else:
                status, failure_class = "failed", "BackendError"
            counts = {name: len(decision.get(name, [])) if status == "completed" and isinstance(
                decision.get(name, []), list) else 0 for name in ("objects", "observations", "events")}
            frame_ids = [frame.get("evidence_id") for clip in window.get("clips", []) if isinstance(clip, dict)
                         for frame in clip.get("frames", []) if isinstance(frame, dict)]
            metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
            latency = metrics.get("latency_s")
            latency = round(_number(latency), 3) if isinstance(latency, (int, float)) and not isinstance(
                latency, bool) and math.isfinite(latency) and latency >= 0 else None
            message = ExportedPerceptionStatus(
                status=status, failure_class=failure_class,
                area=self._area(frame_ids), source_started_at=wall(window.get("started_at")),
                source_ended_at=wall(window.get("ended_at")),
                inference_started_at=self._optional_time(payload.get("inference_started_at")),
                emitted_at=self._round_time(payload.get("emitted_at")), latency_s=latency,
                sampled_frames=int(payload.get("sampled_frames", len(frame_ids))),
                full_sequence_frames=int(payload.get("full_sequence_frames", len(frame_ids))),
                objects_accepted=counts["objects"], observations_accepted=counts["observations"],
                events_accepted=counts["events"], audit_ref=audit_ref)
            refs = {"window_id": window.get("window_id"), "frame_evidence_ids": frame_ids}
            return message, refs
        if topic == "verification.result":
            confidence = self._confidence(payload.get("confidence"))
            span = payload.get("evidence_span") if isinstance(payload.get("evidence_span"), dict) else {}
            message = ExportedVerification(
                verification_id=str(payload.get("verification_id")), request_id=str(payload.get("request_id")),
                question=payload.get("question"), subject_id=str(payload.get("subject_id")),
                verdict=payload.get("verdict"), confidence=confidence,
                uncertainty_level=uncertainty_level(confidence, "learned"),
                reason_code=payload.get("reason_code"), evaluated_at=self._round_time(payload.get("evaluated_at")),
                evidence_started_at=self._optional_time(span.get("started_at")),
                evidence_ended_at=self._optional_time(span.get("ended_at")),
                frames_examined=int(payload.get("frames_examined", 0)), audit_ref=audit_ref)
            refs = {"verification_id": payload.get("verification_id"), "request_id": payload.get("request_id"),
                    "audit": payload.get("audit")}
            return message, refs
        raise ExportBlocked("topic is not in the production export allowlist: " + str(topic)[:80])

    def export_envelope(self, private_envelope: dict, *, now: float) -> tuple[dict, str, dict]:
        """Sanitize first, then hash the sanitized content. Never the private payload."""
        delivery_id, topic = str(private_envelope["delivery_id"]), str(private_envelope["topic"])
        export_id = self._ref("export_", "export:" + topic, delivery_id, 32)
        audit_ref = self._ref("audit_", "audit:" + topic, delivery_id, 24)
        message, refs = self.build(topic, private_envelope.get("payload"), audit_ref)
        content = message.model_dump()
        digest = export_content_sha256(export_id, topic, content)
        export = {"schema_version": SCHEMA_VERSION, "export_id": export_id, "topic": topic,
                  "message": content, "created_at": round(now, 6), "export_sha256": digest}
        violations = scan_export(export)
        if violations:
            raise ExportBlocked("; ".join(violations)[:2000])
        return export, audit_ref, refs

    def export_pending(self, *, now=None, limit=32) -> dict:
        now = self.engine.clock() if now is None else now
        exported, blocked = [], []
        for envelope in self.engine.store.claim_events(now=now, limit=limit):
            try:
                export, audit_ref, refs = self.export_envelope(envelope, now=now)
                self.store.record(
                    export_id=export["export_id"], topic=export["topic"],
                    private_delivery_id=envelope["delivery_id"],
                    private_payload_sha256=envelope["payload_sha256"], export_json=_json(export),
                    export_sha256=export["export_sha256"], audit_ref=audit_ref, local_refs=refs, now=now)
                # The private envelope is consumed by the boundary, not delivered anywhere.
                self.engine.store.delivery_result(envelope["delivery_id"], envelope["lease_token"], now=now,
                                                  delivered=True)
                exported.append(export["export_id"])
            except (ExportBlocked, ValidationError, KeyError, TypeError, ValueError) as exc:
                reason = type(exc).__name__
                self.store.block(envelope["delivery_id"], envelope["topic"], reason, [str(exc)[:2000]], now)
                self.engine.store.delivery_result(envelope["delivery_id"], envelope["lease_token"], now=now,
                                                  delivered=False, error="ExportBlocked:" + reason)
                blocked.append({"delivery_id": envelope["delivery_id"], "topic": envelope["topic"],
                                "reason_class": reason})
        return {"exported": exported, "blocked": blocked}

    def acknowledge(self, export_id: str, export_sha256: str, *, now=None) -> dict:
        """One frontier ACK of an export becomes the local ACK of the exact private delivery."""
        now = self.engine.clock() if now is None else now
        mapping = self.store.acknowledge(export_id, export_sha256, now=now)
        already = True
        if mapping["changed"]:
            already = not self.engine.acknowledge(mapping["private_delivery_id"],
                                                  mapping["private_payload_sha256"], now=now)
        return {"acknowledged": True, "already_acknowledged": (not mapping["changed"]) or already,
                "topic": mapping["topic"]}


# --- Production mode guards ------------------------------------------------------------------------

FORBIDDEN_MODEL_CLASSES = ("PhysicalRemoteModel", "PhysicalVLLMModel", "NativeModel", "RemoteModel")
NETWORK_ATTRIBUTES = ("url", "base_url", "engine_url", "endpoint", "client", "api_token")


def assert_production_model(model) -> dict:
    """Fail closed: only a backend that declares local, non-uploading inference is accepted."""
    name = type(model).__name__
    if name in FORBIDDEN_MODEL_CLASSES or any(
            base.__name__ in FORBIDDEN_MODEL_CLASSES for base in type(model).__mro__):
        raise ProductionModeError(f"{name} performs remote or experimental sensory inference")
    if getattr(model, "local_inference", None) is not True:
        raise ProductionModeError(f"{name} does not declare local_inference=True")
    if getattr(model, "sensory_media_leaves_host", None) is not False:
        raise ProductionModeError(f"{name} does not declare sensory_media_leaves_host=False")
    exposed = [attr for attr in NETWORK_ATTRIBUTES if hasattr(model, attr)]
    if exposed:
        raise ProductionModeError(f"{name} exposes network transport attributes: {', '.join(exposed)}")
    return {"model_class": name, "local_inference": True, "sensory_media_leaves_host": False}


def assert_production_coordinator(coordinator) -> dict:
    from .coordinator import CoordinatorBridge
    if isinstance(coordinator, CoordinatorBridge) or not getattr(coordinator, "exports_only", False):
        raise ProductionModeError(type(coordinator).__name__ + " can publish private envelopes")
    return {"coordinator_class": type(coordinator).__name__, "exports_only": True}


class ProductionRuntimeConfig(StrictModel):
    mode: Literal["production"]
    inference_backend: Literal["local-mps", "local-mlx", "local-test"]
    model_id: str = Field(min_length=1, max_length=200)
    revision: str = Field(min_length=1, max_length=64)
    adapter_path: str | None = Field(default=None, max_length=4096)
    export_policy: str = Field(min_length=1, max_length=4096)
    export_key_file: str = Field(min_length=1, max_length=4096)
    max_frames: int = Field(default=8, ge=2, le=32)
    max_new_tokens: int = Field(default=768, ge=16, le=4096)
    output_contract: Literal["strict", "judged-fields"] = "strict"

    @classmethod
    def load(cls, path: str | Path) -> "ProductionRuntimeConfig":
        raw = json.loads(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ProductionModeError("production runtime config must be an object")
        forbidden = {"url", "engine_url", "base_url", "endpoint", "token_file", "api_token", "bridge_media"}
        found = sorted(forbidden & set(raw))
        if found:
            raise ProductionModeError("production runtime config names remote inference: " + ", ".join(found))
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            raise ProductionModeError("invalid production runtime config: " + str(exc)[:500]) from None
