"""Durable machine bridge for a real frontier coordinator, without an API-key dependency.

This module transports typed messages; it does not simulate an agent or generate
watch plans. A frontier agent reads published event files and writes inbox commands.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .schema import StrictModel
from .watches import ID, WatchEngine, WatchSpec, _digest, _json


class CoordinatorCommand(StrictModel):
    command_id: str = Field(pattern=ID)
    coordinator_id: str = Field(min_length=1, max_length=300)
    type: Literal["create_watch", "cancel_watch", "ack_event"]
    watch: WatchSpec | None = None
    watch_id: str | None = Field(default=None, pattern=ID)
    delivery_id: str | None = Field(default=None, pattern=r"^delivery_[a-f0-9]{32}$")
    payload_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def exact_command_fields(self):
        fields = {
            "watch": self.watch,
            "watch_id": self.watch_id,
            "delivery_id": self.delivery_id,
            "payload_sha256": self.payload_sha256,
        }
        expected = {
            "create_watch": {"watch"},
            "cancel_watch": {"watch_id"},
            "ack_event": {"delivery_id", "payload_sha256"},
        }[self.type]
        if {key for key, value in fields.items() if value is not None} != expected:
            raise ValueError("command fields must match its exact type")
        if self.watch and self.watch.context.coordinator_id != self.coordinator_id:
            raise ValueError("watch digital context must identify the actual coordinating agent")
        return self


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class CoordinatorBridge:
    def __init__(self, engine: WatchEngine):
        self.engine = engine

    def accept(self, value, *, now=None):
        command = value if isinstance(value, CoordinatorCommand) else CoordinatorCommand.model_validate(value)
        now = self.engine.clock() if now is None else now
        digest, db = _digest(command.model_dump()), self.engine.store.db
        # Individual operations below are idempotent. An interrupted command is
        # retried with its same ID/digest, never translated into a new watch ID.
        db.execute("BEGIN IMMEDIATE")
        try:
            old = db.execute(
                "SELECT * FROM coordinator_inbox WHERE command_id=?", (command.command_id,)
            ).fetchone()
            if old and old["digest"] != digest:
                raise ValueError("coordinator command ID reused with different content")
            if old and old["status"] == "complete":
                db.commit()
                return json.loads(old["result_json"])
            db.execute(
                "INSERT OR IGNORE INTO coordinator_inbox VALUES(?,?,?,?)",
                (command.command_id, digest, "pending", None),
            )
            db.commit()
        except BaseException:
            db.rollback()
            raise
        if command.type == "create_watch":
            result = self.engine.create(command.watch, now=now)
        elif command.type == "cancel_watch":
            result = {"cancelled": self.engine.cancel(command.watch_id, now=now)}
        else:
            changed = self.engine.acknowledge(command.delivery_id, command.payload_sha256, now=now)
            result = {"acknowledged": True, "already_acknowledged": not changed}
        response = {
            "command_id": command.command_id,
            "coordinator_id": command.coordinator_id,
            "accepted_at": now,
            "result": result,
        }
        db.execute(
            "UPDATE coordinator_inbox SET status='complete',result_json=? WHERE command_id=?",
            (_json(response), command.command_id),
        )
        return response


class FileCoordinator(CoordinatorBridge):
    """Outbox files are retryable deliveries; acknowledgements arrive through inbox."""

    def __init__(self, engine: WatchEngine, directory: str | Path):
        super().__init__(engine)
        self.directory = Path(directory)
        for name in ("inbox", "outbox", "results", "rejected"):
            (self.directory / name).mkdir(parents=True, exist_ok=True)

    def publish(self, *, now=None, limit=32):
        now = self.engine.clock() if now is None else now
        published = []
        for envelope in self.engine.store.claim_events(now=now, limit=limit):
            try:
                # The stable file name and stable payload hash allow a coordinator
                # to deduplicate retries before making any new planning decision.
                path = self.directory / "outbox" / (envelope["delivery_id"] + ".json")
                if path.is_symlink():
                    raise ValueError("outbox target cannot be a symlink")
                envelope["transport_delivered_at"] = now
                atomic_json(path, envelope)
                self.engine.store.delivery_result(
                    envelope["delivery_id"], envelope["lease_token"], now=now, delivered=True
                )
                published.append(envelope)
            except Exception as exc:
                self.engine.store.delivery_result(
                    envelope["delivery_id"],
                    envelope["lease_token"],
                    now=now,
                    delivered=False,
                    error=type(exc).__name__,
                )
        return published

    def receive(self, *, now=None, limit=32):
        now = self.engine.clock() if now is None else now
        results = []
        for path in sorted((self.directory / "inbox").glob("*.json"))[:limit]:
            identity = hashlib.sha256(path.name.encode()).hexdigest()[:16]
            original_stat = None
            try:
                original_stat = path.lstat()
                if path.is_symlink() or not path.is_file() or original_stat.st_size > 1024 * 1024:
                    raise ValueError("inbox packet must be a bounded regular JSON file")
                content = path.read_text()
                value = json.loads(content)
                command = CoordinatorCommand.model_validate(value)
                response = self.accept(command, now=now)
                atomic_json(self.directory / "results" / (command.command_id + ".json"), response)
                # Remove only the exact packet just handled; a changed file is kept.
                if path.read_text() == content:
                    path.unlink()
                results.append(response)
            except Exception as exc:
                atomic_json(
                    self.directory / "rejected" / (identity + ".json"),
                    {
                        "input_filename": path.name,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:3000],
                        "received_at": now,
                    },
                )
                # Quarantine the exact rejected file so a fixed prefix of invalid
                # commands cannot starve the valid inbox. Never follow a symlink.
                if original_stat is not None and path.exists() and path.lstat() == original_stat:
                    os.replace(path, self.directory / "rejected" / (identity + "-packet.json"))
        return results

    def tick(self, *, now=None, limit=32):
        now = self.engine.clock() if now is None else now
        received = self.receive(now=now, limit=limit)
        watches = self.engine.advance(now=now)
        published = self.publish(now=now, limit=limit)
        return {"commands": received, "watches": watches, "published": published}


class HTTPCoordinator(CoordinatorBridge):
    """Optional explicitly configured event delivery; default integration is files."""

    def __init__(self, engine: WatchEngine, client, endpoint: str):
        super().__init__(engine)
        self.client, self.endpoint = client, endpoint

    def publish(self, *, now=None, limit=32):
        now = self.engine.clock() if now is None else now
        delivered = []
        for envelope in self.engine.store.claim_events(now=now, limit=limit):
            try:
                response = self.client.post(
                    self.endpoint, json=envelope, headers={"Idempotency-Key": envelope["delivery_id"]}
                )
                response.raise_for_status()
                self.engine.store.delivery_result(
                    envelope["delivery_id"], envelope["lease_token"], now=now, delivered=True
                )
                # Successful transport is separate from semantic acknowledgement.
                delivered.append(envelope["delivery_id"])
            except Exception as exc:
                self.engine.store.delivery_result(
                    envelope["delivery_id"],
                    envelope["lease_token"],
                    now=now,
                    delivered=False,
                    error=type(exc).__name__,
                )
        return delivered
