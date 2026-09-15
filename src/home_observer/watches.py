"""Durable declarative physical watches and one-shot approved responses.

Perception contributes physical events only. Response plans are local approvals,
not model output. A committed reservation precedes every device command.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from .schema import Action, StrictModel

ID = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _delivery_id(topic, key):
    return "delivery_" + hashlib.sha256((topic + "\0" + key).encode()).hexdigest()[:32]


class DigitalContext(StrictModel):
    commitment_id: str = Field(pattern=ID)
    text: str = Field(min_length=1, max_length=12000)
    coordinator_id: str = Field(min_length=1, max_length=300)
    source_ref: str | None = Field(default=None, max_length=2000)


class EventCondition(StrictModel):
    type: Literal["event"] = "event"
    kind: str = Field(min_length=1, max_length=160)
    object_label: str | None = Field(default=None, max_length=200)
    subject_ids: list[str] = Field(default_factory=list, max_length=16)
    origin: Literal["any", "learned", "geometric"] = "any"
    min_confidence: float = Field(default=0.8, ge=0, le=1)
    min_duration_s: float = Field(default=0, ge=0, le=86400)
    max_gap_s: float = Field(default=0.5, ge=0, le=60)
    reset_kinds: list[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def physical_identifiers(self):
        if any(not re.fullmatch(r"phys_[A-Za-z0-9_-]+", value) for value in self.subject_ids):
            raise ValueError("watch subjects must be persistent phys_* IDs, never Home Assistant entities")
        return self


class DeadlineCondition(StrictModel):
    type: Literal["deadline"] = "deadline"
    at: float


class OutcomeSpec(StrictModel):
    condition: EventCondition
    within_s: float = Field(default=60, gt=0, le=86400)


class WatchSpec(StrictModel):
    watch_id: str = Field(pattern=ID)
    context: DigitalContext
    condition: EventCondition | DeadlineCondition = Field(discriminator="type")
    response_id: str = Field(pattern=ID)
    created_at: float
    ttl_s: float = Field(default=3600, gt=0, le=604800)
    not_before: float | None = None
    deadline_at: float | None = None
    response_deadline_s: float = Field(default=10, gt=0, le=3600)
    max_event_age_s: float = Field(default=30, gt=0, le=3600)
    outcome: OutcomeSpec | None = None
    max_responses: Literal[1] = 1

    @property
    def starts_at(self):
        return self.created_at if self.not_before is None else self.not_before

    @property
    def expires_at(self):
        return min(
            self.created_at + self.ttl_s, self.deadline_at if self.deadline_at is not None else float("inf")
        )

    @model_validator(mode="after")
    def ordered_times(self):
        if not self.created_at <= self.starts_at < self.expires_at:
            raise ValueError("watch start and expiry must follow creation")
        if (
            isinstance(self.condition, DeadlineCondition)
            and not self.starts_at <= self.condition.at < self.expires_at
        ):
            raise ValueError("deadline trigger must lie inside the watch lifetime")
        return self


class ApprovedResponse(StrictModel):
    response_id: str = Field(pattern=ID)
    kind: Literal["notify", "ha_service"]
    message: str = Field(default="", max_length=4000)
    ha_entity_id: str | None = None
    domain: Literal["light", "switch"] | None = None
    service: Literal["turn_on", "turn_off"] | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    approval_ref: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def locally_approved_action(self):
        if self.kind == "notify":
            if self.ha_entity_id or self.domain or self.service or self.data:
                raise ValueError("notification response cannot contain a device command")
        else:
            if not self.domain or not self.service or not self.ha_entity_id:
                raise ValueError("device response requires its exact domain, service and HA entity")
            if not re.fullmatch(self.domain + r"\.[a-z0-9_]+", self.ha_entity_id):
                raise ValueError("HA entity must match its approved domain; physical IDs are separate")
            if set(self.data) - {"brightness"}:
                raise ValueError("unsupported approved service argument")
            if "brightness" in self.data:
                value = self.data["brightness"]
                if (
                    self.domain != "light"
                    or self.service != "turn_on"
                    or type(value) is not int
                    or not 0 <= value <= 255
                ):
                    raise ValueError("brightness requires light.turn_on and an integer from 0 to 255")
        return self


class WatchEvent(StrictModel):
    event_id: str = Field(pattern=r"^pevt_[A-Za-z0-9_-]+$")
    source_event_id: str = ""
    window_id: str = ""
    kind: str = Field(min_length=1, max_length=160)
    object_label: str | None = None
    subject_ids: list[str] = Field(default_factory=list, max_length=32)
    started_at: float
    ended_at: float
    description: str = Field(default="", max_length=6000)
    confidence: float = Field(ge=0, le=1)
    uncertainty: str = Field(default="", max_length=3000)
    pre_evidence_ids: list[str] = Field(default_factory=list, max_length=64)
    post_evidence_ids: list[str] = Field(default_factory=list, max_length=64)
    origin: Literal["learned", "geometric", "unspecified"] = "unspecified"
    available_at: float | None = None

    @model_validator(mode="after")
    def valid_physical_event(self):
        if self.ended_at < self.started_at:
            raise ValueError("event ends before it starts")
        if any(not re.fullmatch(r"phys_[A-Za-z0-9_-]+", x) for x in self.subject_ids):
            raise ValueError("physical event subjects must be resolved phys_* IDs")
        if not self.pre_evidence_ids and not self.post_evidence_ids:
            raise ValueError("event needs concrete source evidence")
        if self.available_at is not None and self.available_at < self.ended_at:
            raise ValueError("event cannot be available before its source interval ends")
        return self


class WatchStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(
            """
        CREATE TABLE IF NOT EXISTS local_watches(
          watch_id TEXT PRIMARY KEY, spec_json TEXT NOT NULL, response_json TEXT NOT NULL,
          digest TEXT NOT NULL, status TEXT NOT NULL, progress_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS watch_input_events(
          event_id TEXT PRIMARY KEY, digest TEXT NOT NULL, event_json TEXT NOT NULL,
          available_at REAL NOT NULL, received_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS watch_seen_events(
          watch_id TEXT NOT NULL,event_id TEXT NOT NULL,PRIMARY KEY(watch_id,event_id));
        CREATE TABLE IF NOT EXISTS watch_responses(
          response_key TEXT PRIMARY KEY,watch_id TEXT UNIQUE NOT NULL,status TEXT NOT NULL,
          reserved_at REAL NOT NULL,sent_at REAL,verified_at REAL,result_json TEXT);
        CREATE TABLE IF NOT EXISTS watch_outbox(
          delivery_id TEXT PRIMARY KEY,topic TEXT NOT NULL,payload_json TEXT NOT NULL,
          payload_sha256 TEXT NOT NULL,created_at REAL NOT NULL,attempts INTEGER NOT NULL DEFAULT0,
          next_attempt_at REAL NOT NULL,lease_until REAL,lease_token TEXT,
          delivered_at REAL,acked_at REAL,last_error TEXT);
        CREATE TABLE IF NOT EXISTS coordinator_inbox(
          command_id TEXT PRIMARY KEY,digest TEXT NOT NULL,status TEXT NOT NULL,result_json TEXT);
        """.replace("DEFAULT0", "DEFAULT 0")
        )

    def close(self):
        self.db.close()

    def _emit(self, topic, key, payload, now):
        delivery_id = _delivery_id(topic, key)
        self.db.execute(
            """INSERT OR IGNORE INTO watch_outbox
          (delivery_id,topic,payload_json,payload_sha256,created_at,next_attempt_at) VALUES(?,?,?,?,?,?)""",
            (delivery_id, topic, _json(payload), _digest(payload), now, now),
        )
        return delivery_id

    def get(self, watch_id):
        row = self.db.execute("SELECT * FROM local_watches WHERE watch_id=?", (watch_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        for name in ("spec", "response", "progress"):
            result[name] = json.loads(result.pop(name + "_json"))
        response = self.db.execute("SELECT * FROM watch_responses WHERE watch_id=?", (watch_id,)).fetchone()
        result["execution"] = dict(response) if response else None
        if result["execution"] and result["execution"]["result_json"]:
            result["execution"]["result"] = json.loads(result["execution"].pop("result_json"))
        progress = result["progress"]
        clocks = {
            key: progress.get(key)
            for key in (
                "eligible_at",
                "event_source_started_at",
                "event_source_ended_at",
                "event_available_at",
                "matched_at",
                "response_reserved_at",
                "response_queued_at",
                "response_sent_at",
                "response_verified_at",
                "outcome_source_ended_at",
                "outcome_available_at",
                "outcome_verified_at",
            )
        }
        for prefix, event_id in (
            ("event", progress.get("event_id")),
            ("outcome", progress.get("outcome_event_id")),
        ):
            delivery = (
                self.db.execute(
                    "SELECT delivered_at,acked_at FROM watch_outbox WHERE delivery_id=?",
                    (_delivery_id("physical.event", event_id),),
                ).fetchone()
                if event_id
                else None
            )
            clocks[prefix + "_delivered_at"] = delivery["delivered_at"] if delivery else None
            clocks[prefix + "_acked_at"] = delivery["acked_at"] if delivery else None
        result["clocks"] = clocks
        result["latencies_s"] = {
            "source_to_available": self._elapsed(clocks, "event_source_ended_at", "event_available_at"),
            "available_to_match": self._elapsed(clocks, "event_available_at", "matched_at"),
            "source_to_delivery": self._elapsed(clocks, "event_source_ended_at", "event_delivered_at"),
            "source_to_response": self._elapsed(clocks, "event_source_ended_at", "response_sent_at"),
            "source_to_verified_response": self._elapsed(
                clocks, "event_source_ended_at", "response_verified_at"
            ),
            "response_to_verified_outcome": self._elapsed(clocks, "response_sent_at", "outcome_verified_at"),
        }
        return result

    @staticmethod
    def _elapsed(clocks, start, end):
        return clocks[end] - clocks[start] if clocks[start] is not None and clocks[end] is not None else None

    def claim_events(self, *, now, limit=32, lease_s=30):
        if not 1 <= limit <= 1000 or not 0 < lease_s <= 3600:
            raise ValueError("invalid outbox lease bounds")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            rows = self.db.execute(
                """SELECT * FROM watch_outbox WHERE acked_at IS NULL
              AND next_attempt_at<=? AND (lease_until IS NULL OR lease_until<=?)
              ORDER BY created_at,delivery_id LIMIT ?""",
                (now, now, limit),
            ).fetchall()
            result = []
            for row in rows:
                token = uuid.uuid4().hex
                self.db.execute(
                    """UPDATE watch_outbox SET attempts=attempts+1,lease_until=?,lease_token=?
                  WHERE delivery_id=?""",
                    (now + lease_s, token, row["delivery_id"]),
                )
                result.append(
                    {
                        "delivery_id": row["delivery_id"],
                        "topic": row["topic"],
                        "payload": json.loads(row["payload_json"]),
                        "payload_sha256": row["payload_sha256"],
                        "created_at": row["created_at"],
                        "attempt": row["attempts"] + 1,
                        "lease_token": token,
                        "delivered_at": row["delivered_at"],
                    }
                )
            self.db.commit()
            return result
        except BaseException:
            self.db.rollback()
            raise

    def delivery_result(self, delivery_id, lease_token, *, now, delivered, error=None):
        row = self.db.execute(
            "SELECT attempts FROM watch_outbox WHERE delivery_id=? AND lease_token=?",
            (delivery_id, lease_token),
        ).fetchone()
        if row is None:
            return False
        delay = min(300, 2 ** min(row["attempts"], 8))
        self.db.execute(
            """UPDATE watch_outbox SET delivered_at=CASE WHEN ? THEN COALESCE(delivered_at,?)
          ELSE delivered_at END,next_attempt_at=?,lease_until=NULL,lease_token=NULL,last_error=?
          WHERE delivery_id=? AND lease_token=?""",
            (delivered, now, now + delay, error, delivery_id, lease_token),
        )
        return True


class WatchEngine:
    def __init__(self, store: WatchStore, approved_responses, executor=None, *, clock=time.time):
        self.store, self.executor, self.clock = store, executor, clock
        self.approved = {}
        for value in approved_responses:
            response = (
                value if isinstance(value, ApprovedResponse) else ApprovedResponse.model_validate(value)
            )
            if response.response_id in self.approved:
                raise ValueError("local response registry contains a duplicate response ID")
            self.approved[response.response_id] = response

    def create(self, spec: WatchSpec | dict, *, now=None):
        spec = spec if isinstance(spec, WatchSpec) else WatchSpec.model_validate(spec)
        now = self.clock() if now is None else now
        if spec.response_id not in self.approved:
            raise ValueError("response_id is not in the local approved registry")
        response = self.approved[spec.response_id]
        digest = _digest({"spec": spec.model_dump(), "response": response.model_dump()})
        db = self.store.db
        db.execute("BEGIN IMMEDIATE")
        try:
            old = db.execute("SELECT digest FROM local_watches WHERE watch_id=?", (spec.watch_id,)).fetchone()
            if old and old["digest"] != digest:
                raise ValueError("watch ID already belongs to a different immutable specification")
            if not old:
                if spec.created_at > now or spec.expires_at <= now:
                    raise ValueError("watch creation must be current and unexpired")
                db.execute(
                    "INSERT INTO local_watches VALUES(?,?,?,?,?,?)",
                    (
                        spec.watch_id,
                        _json(spec.model_dump()),
                        _json(response.model_dump()),
                        digest,
                        "armed",
                        "{}",
                    ),
                )
                self.store._emit(
                    "watch.created",
                    spec.watch_id,
                    {"watch_id": spec.watch_id, "spec": spec.model_dump(), "created_at": now},
                    now,
                )
            db.commit()
            return self.store.get(spec.watch_id)
        except BaseException:
            db.rollback()
            raise

    def cancel(self, watch_id, *, now=None):
        now = self.clock() if now is None else now
        # Cancellation never retracts or repeats a command already reserved/sent.
        db = self.store.db
        db.execute("BEGIN IMMEDIATE")
        try:
            row = db.execute("SELECT status FROM local_watches WHERE watch_id=?", (watch_id,)).fetchone()
            cancelled = bool(row and row["status"] in ("armed", "cancelled"))
            if row and row["status"] == "armed":
                db.execute("UPDATE local_watches SET status='cancelled' WHERE watch_id=?", (watch_id,))
                self.store._emit(
                    "watch.cancelled", watch_id, {"watch_id": watch_id, "cancelled_at": now}, now
                )
            db.commit()
            return cancelled
        except BaseException:
            db.rollback()
            raise

    @staticmethod
    def _matches(condition, event):
        return (
            event.kind == condition.kind
            and event.confidence >= condition.min_confidence
            and (condition.object_label is None or event.object_label == condition.object_label)
            and set(condition.subject_ids).issubset(event.subject_ids)
            and (condition.origin == "any" or condition.origin == event.origin)
        )

    @staticmethod
    def _accumulate(condition, event, progress, lower_bound):
        key = _json(sorted(event.subject_ids) or [event.object_label])
        start, end = max(lower_bound, event.started_at), event.ended_at
        old_end = progress.get("candidate_end")
        if end < start:
            return False, None
        if old_end is None or start > old_end + condition.max_gap_s or key != progress.get("candidate_key"):
            progress.update(candidate_start=start, candidate_key=key, candidate_observed_s=0)
            old_end = start
        # Count the union of evidenced intervals, never the unobserved gaps
        # tolerated between them. Point detections add zero observed duration.
        covered_start = max(start, old_end)
        observed_before = progress.get("candidate_observed_s", 0)
        progress["candidate_observed_s"] = observed_before + max(0, end - covered_start)
        progress["candidate_end"] = max(end, old_end)
        eligible = covered_start + max(0, condition.min_duration_s - observed_before)
        if condition.min_duration_s == 0:
            eligible = end
        return progress["candidate_observed_s"] >= condition.min_duration_s, eligible

    def consume(self, value: WatchEvent | dict, *, now=None):
        now = self.clock() if now is None else now
        event = value if isinstance(value, WatchEvent) else WatchEvent.model_validate(value)
        if event.ended_at > now or (event.available_at is not None and event.available_at > now):
            raise ValueError("future events cannot trigger a local watch")
        payload = event.model_dump(exclude={"available_at"})
        digest = _digest(payload)
        available = event.available_at if event.available_at is not None else now
        self.store.db.execute("BEGIN IMMEDIATE")
        try:
            old = self.store.db.execute(
                "SELECT digest FROM watch_input_events WHERE event_id=?", (event.event_id,)
            ).fetchone()
            if old and old["digest"] != digest:
                raise ValueError("event ID was reused with different physical evidence")
            if not old:
                self.store.db.execute(
                    "INSERT OR IGNORE INTO watch_input_events VALUES(?,?,?,?,?)",
                    (event.event_id, digest, _json(payload), available, now),
                )
                self.store._emit(
                    "physical.event",
                    event.event_id,
                    {**payload, "event_available_at": available, "received_at": now},
                    now,
                )
            self.store.db.commit()
        except BaseException:
            self.store.db.rollback()
            raise
        return self.advance(now=now)

    def advance(self, *, now=None):
        """Local clock/queue drain; independent of the sensing model and cloud agent."""
        now = self.clock() if now is None else now
        changes, pending = [], []
        ids = [
            r[0]
            for r in self.store.db.execute(
                "SELECT watch_id FROM local_watches WHERE status IN ('armed','awaiting_outcome','uncertain')"
            )
        ]
        for watch_id in ids:
            db = self.store.db
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute("SELECT * FROM local_watches WHERE watch_id=?", (watch_id,)).fetchone()
                spec, response = (
                    WatchSpec.model_validate_json(row["spec_json"]),
                    ApprovedResponse.model_validate_json(row["response_json"]),
                )
                progress, status = json.loads(row["progress_json"]), row["status"]
                trigger = None
                if status == "armed" and now >= spec.expires_at:
                    status = "expired"
                    self.store._emit(
                        "watch.expired", watch_id, {"watch_id": watch_id, "expired_at": now}, now
                    )
                elif (
                    status == "armed"
                    and isinstance(spec.condition, DeadlineCondition)
                    and now >= spec.condition.at
                ):
                    trigger = {
                        "eligible_at": spec.condition.at,
                        "event_id": None,
                        "event_available_at": None,
                        "evidence_ids": [f"watch:{watch_id}:deadline"],
                    }
                if (
                    status in ("awaiting_outcome", "uncertain")
                    and spec.outcome
                    and now
                    > min(spec.expires_at, progress.get("response_sent_at", now) + spec.outcome.within_s)
                ):
                    status = "outcome_unknown"
                    self.store._emit(
                        "watch.outcome_unknown",
                        watch_id,
                        {"watch_id": watch_id, **progress, "reported_at": now},
                        now,
                    )
                if status in ("armed", "awaiting_outcome", "uncertain"):
                    events = db.execute(
                        """SELECT e.* FROM watch_input_events e LEFT JOIN watch_seen_events s
                      ON s.event_id=e.event_id AND s.watch_id=? WHERE s.event_id IS NULL AND e.received_at<=?
                      ORDER BY json_extract(e.event_json,'$.ended_at'),e.event_id LIMIT 1000""",
                        (watch_id, now),
                    ).fetchall()
                    for item in events:
                        event = WatchEvent.model_validate_json(item["event_json"])
                        db.execute(
                            "INSERT OR IGNORE INTO watch_seen_events VALUES(?,?)", (watch_id, event.event_id)
                        )
                        if now - event.ended_at > spec.max_event_age_s or event.ended_at < spec.starts_at:
                            continue
                        if status in ("awaiting_outcome", "uncertain"):
                            if spec.outcome and event.started_at >= progress.get(
                                "response_sent_at", float("inf")
                            ):
                                condition = spec.outcome.condition
                                outcome_progress = progress.setdefault("outcome_progress", {})
                                if self._reset_matches(condition, event, outcome_progress):
                                    outcome_progress.clear()
                                if self._matches(condition, event):
                                    if event.ended_at < outcome_progress.get("candidate_end", float("-inf")):
                                        continue
                                    matched, _ = self._accumulate(
                                        condition, event, outcome_progress, progress["response_sent_at"]
                                    )
                                    if matched:
                                        status = "verified"
                                        progress.update(
                                            outcome_event_id=event.event_id,
                                            outcome_verified_at=now,
                                            outcome_source_ended_at=event.ended_at,
                                            outcome_available_at=item["available_at"],
                                        )
                                        self.store._emit(
                                            "watch.outcome_verified",
                                            watch_id,
                                            {"watch_id": watch_id, **progress},
                                            now,
                                        )
                                        break
                            continue
                        if not isinstance(spec.condition, EventCondition):
                            continue
                        condition = spec.condition
                        if self._reset_matches(condition, event, progress):
                            for key in (
                                "candidate_start",
                                "candidate_end",
                                "candidate_key",
                                "candidate_observed_s",
                            ):
                                progress.pop(key, None)
                        if self._matches(condition, event):
                            if event.ended_at < progress.get("candidate_end", float("-inf")):
                                continue
                            matched, eligible = self._accumulate(condition, event, progress, spec.starts_at)
                            if matched:
                                trigger = {
                                    "eligible_at": eligible,
                                    "event_id": event.event_id,
                                    "event_source_started_at": event.started_at,
                                    "event_source_ended_at": event.ended_at,
                                    "event_available_at": item["available_at"],
                                    "evidence_ids": list(
                                        dict.fromkeys(event.pre_evidence_ids + event.post_evidence_ids)
                                    ),
                                }
                                break
                if trigger and status == "armed":
                    progress.update(trigger, matched_at=now)
                    if now > trigger["eligible_at"] + spec.response_deadline_s:
                        status = "missed_deadline"
                        self.store._emit(
                            "watch.missed_deadline", watch_id, {"watch_id": watch_id, **progress}, now
                        )
                    else:
                        key = "response_" + hashlib.sha256(watch_id.encode()).hexdigest()[:32]
                        inserted = db.execute(
                            "INSERT OR IGNORE INTO watch_responses VALUES(?,?,?,?,?,?,?)",
                            (
                                key,
                                watch_id,
                                "pending",
                                now,
                                None,
                                None,
                                None,
                            ),
                        ).rowcount
                        if inserted:
                            status = "reserved"
                            progress.update(response_key=key, response_reserved_at=now)
                            self.store._emit(
                                "watch.matched", watch_id, {"watch_id": watch_id, **progress}, now
                            )
                            if response.kind == "notify":
                                delivery_id = self.store._emit(
                                    "watch.triggered",
                                    watch_id,
                                    {
                                        "watch_id": watch_id,
                                        "response_key": key,
                                        "message": response.message,
                                        "context": spec.context.model_dump(),
                                        **progress,
                                    },
                                    now,
                                )
                                progress.update(response_delivery_id=delivery_id, response_queued_at=now)
                                status = "awaiting_delivery"
                            else:
                                pending.append((watch_id, key, response, progress.copy()))
                db.execute(
                    "UPDATE local_watches SET status=?,progress_json=? WHERE watch_id=?",
                    (status, _json(progress), watch_id),
                )
                db.commit()
                changes.append(self.store.get(watch_id))
            except BaseException:
                db.rollback()
                raise
        for watch_id, key, response, progress in pending:
            self._respond(watch_id, key, response, progress)
        return [self.store.get(x["watch_id"]) for x in changes]

    @staticmethod
    def _reset_matches(condition, event, progress):
        return (
            event.kind in condition.reset_kinds
            and event.confidence >= condition.min_confidence
            and (condition.origin == "any" or condition.origin == event.origin)
            and (condition.object_label is None or event.object_label == condition.object_label)
            and set(condition.subject_ids).issubset(event.subject_ids)
            and event.ended_at >= progress.get("candidate_end", float("-inf"))
            and (
                progress.get("candidate_key") is None
                or progress["candidate_key"] == _json(sorted(event.subject_ids) or [event.object_label])
            )
        )

    def _respond(self, watch_id, key, response, progress):
        now = self.clock()
        spec = WatchSpec.model_validate(self.store.get(watch_id)["spec"])
        if now > min(spec.expires_at, progress["eligible_at"] + spec.response_deadline_s):
            self.store.db.execute("BEGIN IMMEDIATE")
            try:
                self.store.db.execute(
                    "UPDATE watch_responses SET status='not_sent_deadline' WHERE response_key=?", (key,)
                )
                self.store.db.execute(
                    "UPDATE local_watches SET status='missed_deadline' WHERE watch_id=?", (watch_id,)
                )
                self.store._emit(
                    "watch.missed_deadline",
                    watch_id,
                    {"watch_id": watch_id, **progress, "not_sent_at": now},
                    now,
                )
                self.store.db.commit()
            except BaseException:
                self.store.db.rollback()
                raise
            return
        self.store.db.execute("UPDATE watch_responses SET sent_at=? WHERE response_key=?", (now, key))
        progress["response_sent_at"] = now
        result, verified = {}, False
        try:
            if self.executor is None:
                raise RuntimeError("approved HA response requires the configured verified executor")
            action = Action(
                domain=response.domain,
                service=response.service,
                entity_id=response.ha_entity_id,
                data=response.data,
                reason="Approved physical watch: " + watch_id,
                evidence_ids=progress["evidence_ids"],
            )
            result = self.executor.execute(action, key)
            verified = result.get("verified") is True
        except Exception as exc:
            result = {
                "verified": False,
                "error_type": type(exc).__name__,
                "outcome": "uncertain; command will not be repeated",
            }
        finished = self.clock()
        progress.update(
            response_verified_at=finished if verified else None,
            response_timely=finished <= progress["eligible_at"] + spec.response_deadline_s,
            verification_kind="ha_state_readback",
        )
        status = ("awaiting_outcome" if spec.outcome else "verified") if verified else "uncertain"
        self.store.db.execute("BEGIN IMMEDIATE")
        try:
            self.store.db.execute(
                "UPDATE watch_responses SET status=?,verified_at=?,result_json=? WHERE response_key=?",
                ("verified" if verified else "uncertain", finished if verified else None, _json(result), key),
            )
            self.store.db.execute(
                "UPDATE local_watches SET status=?,progress_json=? WHERE watch_id=?",
                (status, _json(progress), watch_id),
            )
            self.store._emit(
                "watch.response_verified" if verified else "watch.response_uncertain",
                watch_id,
                {"watch_id": watch_id, "result": result, **progress},
                finished,
            )
            self.store.db.commit()
        except BaseException:
            self.store.db.rollback()
            raise

    def acknowledge(self, delivery_id, payload_sha256, *, now=None):
        now = self.clock() if now is None else now
        db = self.store.db
        db.execute("BEGIN IMMEDIATE")
        try:
            row = db.execute("SELECT * FROM watch_outbox WHERE delivery_id=?", (delivery_id,)).fetchone()
            if not row or row["payload_sha256"] != payload_sha256:
                raise ValueError("acknowledgement must name the exact delivered payload")
            if row["acked_at"] is not None:
                db.commit()
                return False
            if row["delivered_at"] is None:
                raise ValueError("event was not yet delivered by a transport")
            db.execute(
                "UPDATE watch_outbox SET acked_at=?,lease_until=NULL,lease_token=NULL WHERE delivery_id=?",
                (now, delivery_id),
            )
            if row["topic"] == "watch.triggered":
                payload = json.loads(row["payload_json"])
                watch = db.execute(
                    "SELECT * FROM local_watches WHERE watch_id=?", (payload["watch_id"],)
                ).fetchone()
                progress, spec = (
                    json.loads(watch["progress_json"]),
                    WatchSpec.model_validate_json(watch["spec_json"]),
                )
                progress.update(
                    response_sent_at=row["delivered_at"],
                    response_verified_at=now,
                    response_timely=now <= progress["eligible_at"] + spec.response_deadline_s,
                    verification_kind="coordinator_delivery_ack",
                )
                db.execute(
                    "UPDATE watch_responses SET status='verified',sent_at=?,verified_at=?,result_json=? WHERE response_key=?",
                    (
                        row["delivered_at"],
                        now,
                        _json({"verified": True, "verification_kind": "coordinator_delivery_ack"}),
                        payload["response_key"],
                    ),
                )
                db.execute(
                    "UPDATE local_watches SET status=?,progress_json=? WHERE watch_id=?",
                    (
                        "awaiting_outcome" if spec.outcome else "verified",
                        _json(progress),
                        payload["watch_id"],
                    ),
                )
                self.store._emit(
                    "watch.response_verified",
                    payload["watch_id"],
                    {"watch_id": payload["watch_id"], **progress},
                    now,
                )
            db.commit()
            return True
        except BaseException:
            db.rollback()
            raise
