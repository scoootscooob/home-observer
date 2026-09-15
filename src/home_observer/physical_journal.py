"""Append-only physical evidence and tracks in tables separate from HA state.

The registry stores proposed identities; it does not infer associations, fabricate
disappearances from missing detections, or execute actions. Queries are source-time
bounded, including when windows arrive out of order.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import uuid
from dataclasses import asdict
from pathlib import Path

from .physical_schema import PhysicalDecision, PhysicalWindow, validate_physical_decision


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class PhysicalJournal:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        self.db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS physical_windows(
          window_id TEXT PRIMARY KEY, started_at REAL NOT NULL, ended_at REAL NOT NULL,
          content_sha256 TEXT NOT NULL, input_json TEXT NOT NULL, decision_json TEXT NOT NULL,
          mapping_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS physical_tracks(
          track_id TEXT PRIMARY KEY, first_seen_at REAL NOT NULL, first_window_id TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS physical_evidence(
          window_id TEXT NOT NULL REFERENCES physical_windows(window_id), evidence_id TEXT NOT NULL,
          payload_json TEXT NOT NULL, PRIMARY KEY(window_id,evidence_id));
        CREATE TABLE IF NOT EXISTS physical_detections(
          window_id TEXT NOT NULL REFERENCES physical_windows(window_id), detection_id TEXT NOT NULL,
          track_id TEXT NOT NULL REFERENCES physical_tracks(track_id), timestamp REAL NOT NULL,
          payload_json TEXT NOT NULL, PRIMARY KEY(window_id,detection_id));
        CREATE INDEX IF NOT EXISTS physical_detections_track_time ON physical_detections(track_id,timestamp);
        CREATE TABLE IF NOT EXISTS physical_observations(
          id TEXT PRIMARY KEY, window_id TEXT NOT NULL REFERENCES physical_windows(window_id),
          source_id TEXT NOT NULL, subject_id TEXT, timestamp REAL NOT NULL, payload_json TEXT NOT NULL,
          UNIQUE(window_id,source_id));
        CREATE INDEX IF NOT EXISTS physical_observations_time ON physical_observations(timestamp);
        CREATE TABLE IF NOT EXISTS physical_events(
          id TEXT PRIMARY KEY, window_id TEXT NOT NULL REFERENCES physical_windows(window_id),
          source_id TEXT NOT NULL, timestamp REAL NOT NULL, payload_json TEXT NOT NULL,
          UNIQUE(window_id,source_id));
        CREATE INDEX IF NOT EXISTS physical_events_time ON physical_events(timestamp);
        """)
        self.db.commit()

    def ingest(self, window: PhysicalWindow | dict, decision: PhysicalDecision | dict) -> dict[str, str]:
        """Atomically persist one validated window; identical replay returns its mapping.

        A reused window ID with different content fails. References to known tracks
        are validated as of the source window, not the machine's current time.
        """
        window = PhysicalWindow.model_validate(window)
        decision = PhysicalDecision.model_validate(decision)
        input_json, decision_json = _json(window.model_dump()), _json(decision.model_dump())
        if len(input_json) + len(decision_json) > 4 * 1024 * 1024:
            raise ValueError("physical window/decision exceeds journal byte budget")
        digest = hashlib.sha256((input_json + "\0" + decision_json).encode()).hexdigest()
        with self._lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                prior = self.db.execute("SELECT * FROM physical_windows WHERE window_id=?",
                                        (window.window_id,)).fetchone()
                if prior:
                    if prior["content_sha256"] != digest:
                        raise ValueError("physical window ID already contains different content")
                    self.db.commit()
                    return json.loads(prior["mapping_json"])
                local_ids = {item.detection_id for item in decision.objects}
                referenced = {item.track_id for item in decision.objects if item.track_id}
                referenced.update(item.subject_id for item in decision.observations if item.subject_id)
                referenced.update(subject for item in decision.events for subject in item.subject_ids)
                referenced -= local_ids
                known = set()
                for identifier in referenced:
                    record = self.db.execute("SELECT first_seen_at FROM physical_tracks WHERE track_id=?",
                                             (identifier,)).fetchone()
                    if record and record["first_seen_at"] <= window.ended_at:
                        known.add(identifier)
                validate_physical_decision(decision, window, known_track_ids=known)
                mapping = {item.detection_id: item.track_id or "phys_" + uuid.uuid4().hex
                           for item in decision.objects}
                self.db.execute("INSERT INTO physical_windows VALUES(?,?,?,?,?,?,?)",
                                (window.window_id, window.started_at, window.ended_at, digest,
                                 input_json, decision_json, _json(mapping)))
                for reference in window.evidence().values():
                    self.db.execute("INSERT INTO physical_evidence VALUES(?,?,?)",
                                    (window.window_id, reference.evidence_id, _json(asdict(reference))))
                for item in decision.objects:
                    track = mapping[item.detection_id]
                    self.db.execute("INSERT OR IGNORE INTO physical_tracks VALUES(?,?,?)",
                                    (track, item.timestamp, window.window_id))
                    payload = {**item.model_dump(), "track_id": track}
                    self.db.execute("INSERT INTO physical_detections VALUES(?,?,?,?,?)",
                                    (window.window_id, item.detection_id, track, item.timestamp, _json(payload)))
                for item in decision.observations:
                    subject = mapping.get(item.subject_id, item.subject_id)
                    payload = {**item.model_dump(), "subject_id": subject}
                    self.db.execute("INSERT INTO physical_observations VALUES(?,?,?,?,?,?)",
                                    ("pobs_" + uuid.uuid4().hex, window.window_id, item.observation_id,
                                     subject, item.observed_at, _json(payload)))
                for item in decision.events:
                    payload = {**item.model_dump(), "subject_ids": [mapping.get(x, x) for x in item.subject_ids]}
                    self.db.execute("INSERT INTO physical_events VALUES(?,?,?,?,?)",
                                    ("pevt_" + uuid.uuid4().hex, window.window_id, item.event_id,
                                     item.ended_at, _json(payload)))
                self.db.commit()
                return mapping
            except BaseException:
                self.db.rollback()
                raise

    @staticmethod
    def _bounds(as_of, limit):
        if not math.isfinite(as_of) or not 1 <= limit <= 1000:
            raise ValueError("require finite as_of and a limit from 1 to 1000")

    def tracks(self, as_of: float, limit: int = 64, stale_after: float = 300) -> list[dict]:
        """Latest actual detection per identity as of source time; stale is not absent."""
        self._bounds(as_of, limit)
        if not math.isfinite(stale_after) or stale_after < 0:
            raise ValueError("stale_after must be finite and nonnegative")
        with self._lock:
            rows = self.db.execute("""SELECT * FROM (
              SELECT d.*,t.first_seen_at,ROW_NUMBER() OVER(
                PARTITION BY d.track_id ORDER BY d.timestamp DESC,d.window_id DESC,d.detection_id DESC) AS n
              FROM physical_detections d JOIN physical_tracks t ON d.track_id=t.track_id
              WHERE d.timestamp<=?) WHERE n=1 ORDER BY timestamp DESC,track_id LIMIT ?""",
                                   (as_of, limit)).fetchall()
        return [{"track_id": row["track_id"], "first_seen_at": row["first_seen_at"],
                 "last_seen_at": row["timestamp"], "window_id": row["window_id"],
                 "age_seconds": as_of - row["timestamp"], "stale": as_of - row["timestamp"] > stale_after,
                 "last_detection": json.loads(row["payload_json"])} for row in rows]

    def recent_events(self, as_of: float, limit: int = 32, since: float | None = None) -> list[dict]:
        self._bounds(as_of, limit)
        if since is not None and (not math.isfinite(since) or since > as_of):
            raise ValueError("since must be finite and at or before as_of")
        with self._lock:
            rows = self.db.execute("""SELECT * FROM physical_events WHERE timestamp<=? AND timestamp>=?
              ORDER BY timestamp DESC,id DESC LIMIT ?""", (as_of, since if since is not None else -1e300, limit)).fetchall()
        return [{**json.loads(row["payload_json"]), "event_id": row["id"], "source_event_id": row["source_id"],
                 "window_id": row["window_id"]} for row in reversed(rows)]

    def observations(self, as_of: float, limit: int = 64, subject_id: str | None = None) -> list[dict]:
        self._bounds(as_of, limit)
        with self._lock:
            rows = self.db.execute("""SELECT * FROM physical_observations WHERE timestamp<=?
              AND (? IS NULL OR subject_id=?) ORDER BY timestamp DESC,id DESC LIMIT ?""",
                                   (as_of, subject_id, subject_id, limit)).fetchall()
        return [{**json.loads(row["payload_json"]), "observation_id": row["id"],
                 "source_observation_id": row["source_id"], "window_id": row["window_id"]} for row in reversed(rows)]

    def evidence(self, window_id: str, evidence_ids: list[str] | None = None) -> list[dict]:
        with self._lock:
            rows = self.db.execute("SELECT payload_json FROM physical_evidence WHERE window_id=? ORDER BY evidence_id",
                                   (window_id,)).fetchall()
        requested = set(evidence_ids) if evidence_ids is not None else None
        return [item for row in rows if (item := json.loads(row["payload_json"]))
                and (requested is None or item["evidence_id"] in requested)]

    def track_detections(self, track_id: str, limit: int = 64) -> list[dict]:
        """Chronological learned/bootstrap detections for one persistent track."""
        if not isinstance(track_id, str) or not track_id.startswith("phys_") or not 1 <= limit <= 1000:
            raise ValueError("track_detections requires a phys_* ID and a limit from 1 to 1000")
        with self._lock:
            rows = self.db.execute("""SELECT * FROM physical_detections WHERE track_id=?
              ORDER BY timestamp,window_id,detection_id LIMIT ?""", (track_id, limit)).fetchall()
        return [{**json.loads(row["payload_json"]), "window_id": row["window_id"]} for row in rows]

    def stats(self) -> dict:
        with self._lock:
            return {table.removeprefix("physical_"): self.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    for table in ("physical_windows", "physical_tracks", "physical_detections",
                                  "physical_observations", "physical_events", "physical_evidence")}

    def close(self):
        with self._lock:
            self.db.close()
