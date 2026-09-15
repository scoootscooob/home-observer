"""Production coordinator bridge: typed commands in, sanitized exports out, nothing else.

The directory handed to the frontier contains only export envelopes, sanitized
command results and bounded rejection classes. Private outbox envelopes, journal
rows, watch records, approved-response text and exception text never appear in
it. Acknowledgements name exports, never private delivery IDs.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .coordinator import atomic_json
from .local_verifier import VerificationRequest
from .production_export import (
    EXPORT_ID,
    SHA256,
    ExportStore,
    ProductionExporter,
    ProductionModeError,
    scan_export,
)
from .schema import StrictModel
from .watches import ID, WatchEngine, WatchSpec, _digest, _json


class ProductionCommand(StrictModel):
    command_id: str = Field(pattern=ID)
    coordinator_id: str = Field(min_length=1, max_length=300)
    type: Literal["create_watch", "cancel_watch", "ack_export", "verify"]
    watch: WatchSpec | None = None
    watch_id: str | None = Field(default=None, pattern=ID)
    export_id: str | None = Field(default=None, pattern=EXPORT_ID)
    export_sha256: str | None = Field(default=None, pattern=SHA256)
    verification: VerificationRequest | None = None

    @model_validator(mode="after")
    def exact_command_fields(self):
        fields = {"watch": self.watch, "watch_id": self.watch_id, "export_id": self.export_id,
                  "export_sha256": self.export_sha256, "verification": self.verification}
        expected = {"create_watch": {"watch"}, "cancel_watch": {"watch_id"},
                    "ack_export": {"export_id", "export_sha256"}, "verify": {"verification"}}[self.type]
        if {key for key, value in fields.items() if value is not None} != expected:
            raise ValueError("command fields must match its exact type")
        if self.watch and self.watch.context.coordinator_id != self.coordinator_id:
            raise ValueError("watch digital context must identify the actual coordinating agent")
        return self


class ProductionCoordinator:
    """File bridge whose outbox holds sanitized exports only."""

    exports_only = True

    def __init__(self, engine: WatchEngine, exporter: ProductionExporter, directory: str | Path, *,
                 verifier=None, private_directory: str | Path | None = None, async_verification: bool = False):
        if not isinstance(exporter, ProductionExporter) or not isinstance(exporter.store, ExportStore):
            raise ProductionModeError("production coordinator requires the production exporter")
        if exporter.engine is not engine:
            raise ProductionModeError("exporter and coordinator must share one watch engine")
        self.engine, self.exporter, self.verifier = engine, exporter, verifier
        self.directory = Path(directory)
        for name in ("inbox", "outbox", "results", "rejected"):
            (self.directory / name).mkdir(parents=True, exist_ok=True)
        self.private_directory = Path(private_directory) if private_directory else self.directory.parent / (
            self.directory.name + "-private")
        if self.private_directory.resolve() == self.directory.resolve() or self.private_directory.resolve(
        ).is_relative_to(self.directory.resolve()):
            raise ProductionModeError("private diagnostics cannot live inside the frontier-visible bridge")
        self.private_directory.mkdir(parents=True, exist_ok=True)
        self.async_verification = bool(async_verification)
        self._verifications = queue.Queue()
        self._verification_threads = []

    # -- inbound commands --------------------------------------------------------------------

    def accept(self, value, *, now=None):
        command = value if isinstance(value, ProductionCommand) else ProductionCommand.model_validate(value)
        now = self.engine.clock() if now is None else now
        digest, db = _digest(command.model_dump()), self.engine.store.db
        db.execute("BEGIN IMMEDIATE")
        try:
            old = db.execute("SELECT * FROM coordinator_inbox WHERE command_id=?", (command.command_id,)).fetchone()
            if old and old["digest"] != digest:
                raise ValueError("coordinator command ID reused with different content")
            if old and old["status"] == "complete":
                db.commit()
                return json.loads(old["result_json"])
            db.execute("INSERT OR IGNORE INTO coordinator_inbox VALUES(?,?,?,?)",
                       (command.command_id, digest, "pending", None))
            db.commit()
        except BaseException:
            db.rollback()
            raise
        if command.type == "create_watch":
            watch = self.engine.create(command.watch, now=now)
            result = {"watch_id": watch["watch_id"], "status": watch["status"]}
        elif command.type == "cancel_watch":
            result = {"cancelled": self.engine.cancel(command.watch_id, now=now)}
        elif command.type == "ack_export":
            result = self.exporter.acknowledge(command.export_id, command.export_sha256, now=now)
        else:
            if self.verifier is None:
                raise ValueError("no local verifier is configured")
            verification_id = "pver_" + uuid.uuid4().hex
            if self.async_verification:
                # Slow local verification never blocks capture or the watch loop; the
                # result is emitted on a later tick and exported like every other message.
                thread = threading.Thread(target=self._verify_in_background,
                                          args=(command.verification, verification_id, now), daemon=True)
                thread.start()
                self._verification_threads.append(thread)
                result = {"verification_id": verification_id, "status": "pending"}
            else:
                verification = self.verifier.verify(command.verification, now=now, verification_id=verification_id)
                # The private result (with frame references) enters the private outbox and is
                # exported through the same sanitizing boundary as every other message.
                self.engine.store._emit("verification.result", verification_id, verification, now)
                result = {"verification_id": verification_id, "verdict": verification["verdict"],
                          "reason_code": verification["reason_code"], "confidence": verification["confidence"]}
        response = {"command_id": command.command_id, "coordinator_id": command.coordinator_id,
                    "accepted_at": now, "outcome": result}
        if scan_export(result):
            response = {"command_id": command.command_id, "coordinator_id": command.coordinator_id,
                        "accepted_at": now, "outcome": {"accepted": True, "detail_suppressed": True}}
        db.execute("UPDATE coordinator_inbox SET status='complete',result_json=? WHERE command_id=?",
                   (_json(response), command.command_id))
        return response

    def _verify_in_background(self, request, verification_id, now):
        try:
            verification = self.verifier.verify(request, now=now, verification_id=verification_id)
        except Exception as exc:
            verification = {"verification_id": verification_id, "request_id": request.request_id,
                            "question": request.question, "subject_id": request.subject_id,
                            "reference_event_id": request.reference_event_id, "verdict": "unknown",
                            "confidence": 0.0, "reason_code": "learned_output_invalid", "evaluated_at": time.time(),
                            "frames_examined": 0, "evidence_span": {},
                            "audit": {"error_type": type(exc).__name__, "error": str(exc)[:1000]}}
        self._verifications.put(verification)

    def drain_verifications(self, *, now=None):
        now = self.engine.clock() if now is None else now
        emitted = []
        while True:
            try:
                verification = self._verifications.get_nowait()
            except queue.Empty:
                break
            self.engine.store._emit("verification.result", verification["verification_id"], verification, now)
            emitted.append(verification["verification_id"])
        self._verification_threads = [thread for thread in self._verification_threads if thread.is_alive()]
        return emitted

    @property
    def pending_verifications(self) -> int:
        return sum(thread.is_alive() for thread in self._verification_threads) + self._verifications.qsize()

    def receive(self, *, now=None, limit=32):
        now = self.engine.clock() if now is None else now
        results = []
        for path in sorted((self.directory / "inbox").glob("*.json"))[:limit]:
            identity = hashlib.sha256(path.name.encode()).hexdigest()[:16]
            original_stat = None
            try:
                original_stat = path.lstat()
                if path.is_symlink() or not path.is_file() or original_stat.st_size > 256 * 1024:
                    raise ValueError("inbox packet must be a bounded regular JSON file")
                content = path.read_text()
                command = ProductionCommand.model_validate(json.loads(content))
                response = self.accept(command, now=now)
                atomic_json(self.directory / "results" / (command.command_id + ".json"), response)
                if path.read_text() == content:
                    path.unlink()
                results.append(response)
            except Exception as exc:
                # The frontier sees the failure class only; detail stays in the private directory.
                atomic_json(self.directory / "rejected" / (identity + ".json"),
                            {"packet_id": identity, "error_type": type(exc).__name__, "received_at": now})
                atomic_json(self.private_directory / ("rejected-" + identity + ".json"),
                            {"input_filename": path.name, "error_type": type(exc).__name__,
                             "error": str(exc)[:3000], "received_at": now})
                if original_stat is not None and path.exists() and path.lstat() == original_stat:
                    os.replace(path, self.private_directory / ("rejected-" + identity + "-packet.json"))
        return results

    # -- outbound exports --------------------------------------------------------------------

    def publish(self, *, now=None, limit=32):
        now = self.engine.clock() if now is None else now
        boundary = self.exporter.export_pending(now=now, limit=limit)
        published = []
        for export in self.exporter.store.claim(now=now, limit=limit):
            lease = export.pop("lease_token")
            try:
                path = self.directory / "outbox" / (export["export_id"] + ".json")
                if path.is_symlink():
                    raise ValueError("outbox target cannot be a symlink")
                if scan_export({key: value for key, value in export.items()}):
                    raise ValueError("export failed the outbound scan")
                atomic_json(path, export)
                self.exporter.store.delivery_result(export["export_id"], lease, now=now, delivered=True)
                published.append(export)
            except Exception as exc:
                self.exporter.store.delivery_result(export["export_id"], lease, now=now, delivered=False,
                                                    error=type(exc).__name__)
        return {"boundary": boundary, "published": published}

    def tick(self, *, now=None, limit=32):
        now = self.engine.clock() if now is None else now
        received = self.receive(now=now, limit=limit)
        watches = self.engine.advance(now=now)
        self.drain_verifications(now=now)
        published = self.publish(now=now, limit=limit)
        return {"commands": received, "watches": watches, "published": published["published"],
                "blocked": published["boundary"]["blocked"]}
