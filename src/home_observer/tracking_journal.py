"""Append-only current tracking evidence alongside the physical object registry.

Create PhysicalJournal first, then pass its SQLite path here. This helper owns a
separate connection protected by a lock, so capture can write while model context
reads. Queries use source time; missing updates never imply disappearance.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import time
from pathlib import Path

from .temporal_buffer import CapturedFrame


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _physical_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"phys_[A-Za-z0-9_-]{1,128}", value):
        raise ValueError("tracking updates require a physical phys_* ID, never an HA entity ID")
    return value


class TrackingJournal:
    """Persist one bounded batch per frame and return source-time current states.

    `latest` returns the most recent explicit update per physical track, retaining
    camera identity. `box=None` means localization is unknown, not object absence.
    Historical reads never substitute a newer box for an earlier source timestamp.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False, timeout=10, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        if not self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='physical_tracks'"
        ).fetchone():
            self.db.close()
            raise ValueError("initialize PhysicalJournal first and use its existing file-backed SQLite path")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS physical_track_updates(
          track_id TEXT NOT NULL REFERENCES physical_tracks(track_id),
          timestamp REAL NOT NULL, camera_id TEXT NOT NULL, frame_evidence_id TEXT NOT NULL,
          source_sequence INTEGER NOT NULL, box_json TEXT, visibility TEXT NOT NULL,
          heuristic_confidence REAL, method TEXT NOT NULL, evidence_sha256 TEXT NOT NULL,
          details_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
          captured_at REAL NOT NULL, recorded_at REAL NOT NULL,
          PRIMARY KEY(track_id,camera_id,frame_evidence_id));
        CREATE INDEX IF NOT EXISTS physical_track_updates_time
          ON physical_track_updates(track_id,timestamp DESC);
        """)

    @staticmethod
    def _validated(state, frame, method, evidence_sha256):
        if not isinstance(state, dict):
            raise ValueError("tracking state must be a dictionary")
        track_id = _physical_id(state.get("track_id"))
        if state.get("timestamp") != frame.timestamp or state.get("evidence_id") != frame.evidence_id:
            raise ValueError("state timestamp and evidence ID must name the supplied source frame")
        visibility = state.get("visibility")
        if visibility not in ("visible", "uncertain", "lost", "unknown"):
            raise ValueError("tracking visibility must be explicit")
        box = state.get("box")
        if visibility == "visible":
            if (
                not isinstance(box, (list, tuple))
                or len(box) != 4
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in box)
                or not 0 <= box[0] < box[2] <= 1
                or not 0 <= box[1] < box[3] <= 1
            ):
                raise ValueError("visible tracking state requires a finite normalized xyxy box")
            box = [float(value) for value in box]
        elif box is not None:
            raise ValueError("uncertain localization must use box=None, not a stale visible box")
        confidence = state.get("confidence")
        if confidence is not None:
            if (
                type(confidence) not in (int, float)
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                raise ValueError("heuristic confidence must be finite and in [0,1], or unknown")
            confidence = float(confidence)
        details = {
            key: state[key]
            for key in (
                "label",
                "confidence_kind",
                "failure_reason",
                "appearance_distance",
                "visible_fraction",
                "relative_motion",
                "camera_compensated",
                "total_area_ratio",
            )
            if key in state
        }
        # Never promote a tracker score to a calibrated model probability.
        details["confidence_kind"] = "uncalibrated_heuristic"
        details["calibrated_confidence"] = None
        payload = {
            "track_id": track_id,
            "timestamp": float(frame.timestamp),
            "camera_id": frame.camera_id,
            "frame_evidence_id": frame.evidence_id,
            "source_sequence": frame.sequence,
            "box": box,
            "visibility": visibility,
            "heuristic_confidence": confidence,
            "method": method,
            "evidence_sha256": evidence_sha256,
            "details": details,
        }
        serialized = _json(payload)
        if len(serialized.encode()) > 32768:
            raise ValueError("tracking state exceeds its byte budget")
        return payload, hashlib.sha256(serialized.encode()).hexdigest()

    def record(self, states: list[dict], frame: CapturedFrame, *, method="opencv_csrt") -> int:
        """Commit all updates for one actual frame, returning newly inserted rows.

        Replays preserve the first capture/recording clocks. Same identity plus
        different source time, box, method or image digest is a conflict. Empty
        batches add no state and do not mark unmentioned objects absent.
        """
        if not isinstance(frame, CapturedFrame):
            raise ValueError("record requires the actual CapturedFrame, including its image bytes")
        if not isinstance(states, (list, tuple)) or len(states) > 64:
            raise ValueError("one tracking batch must contain at most 64 states")
        if not isinstance(method, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}", method):
            raise ValueError("tracking method must be a bounded method identifier")
        if not states:
            return 0
        digest = hashlib.sha256(frame.jpeg).hexdigest()
        validated = {}
        for state in states:
            payload, payload_sha = self._validated(state, frame, method, digest)
            key = payload["track_id"]
            if key in validated and validated[key][1] != payload_sha:
                raise ValueError("one frame contains conflicting states for the same physical track")
            validated[key] = payload, payload_sha
        with self._lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                placeholders = ",".join("?" for _ in validated)
                known = {
                    row["track_id"]: row["first_seen_at"]
                    for row in self.db.execute(
                        f"SELECT track_id,first_seen_at FROM physical_tracks WHERE track_id IN ({placeholders})",
                        list(validated),
                    )
                }
                inserted = 0
                recorded_at = time.time()
                for track_id, (payload, payload_sha) in validated.items():
                    if track_id not in known or known[track_id] > frame.timestamp:
                        raise ValueError(
                            "tracking update references an unknown or future-born physical track"
                        )
                    old = self.db.execute(
                        """SELECT payload_sha256 FROM physical_track_updates
                        WHERE track_id=? AND camera_id=? AND frame_evidence_id=?""",
                        (track_id, frame.camera_id, frame.evidence_id),
                    ).fetchone()
                    if old:
                        if old["payload_sha256"] != payload_sha:
                            raise ValueError("tracking frame identity was reused with conflicting content")
                        continue
                    self.db.execute(
                        "INSERT INTO physical_track_updates VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            track_id,
                            frame.timestamp,
                            frame.camera_id,
                            frame.evidence_id,
                            frame.sequence,
                            _json(payload["box"]) if payload["box"] is not None else None,
                            payload["visibility"],
                            payload["heuristic_confidence"],
                            method,
                            digest,
                            _json(payload["details"]),
                            payload_sha,
                            frame.captured_at,
                            recorded_at,
                        ),
                    )
                    inserted += 1
                self.db.commit()
                return inserted
            except BaseException:
                self.db.rollback()
                raise

    def latest(
        self, as_of: float, track_ids: list[str] | None = None, *, limit=64, camera_id: str | None = None
    ) -> list[dict]:
        """Most recent explicit state per track at/before `as_of`, never future state.

        With multiple cameras, one latest update per track is returned together
        with its camera ID; optional camera_id selects a single camera. This query
        performs no cross-camera identity inference or visibility fusion.
        """
        if not math.isfinite(as_of) or type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("as_of must be finite and limit must be an integer from 1 to 1000")
        filters, values = ["timestamp<=?"], [as_of]
        if track_ids is not None:
            if not isinstance(track_ids, (list, tuple)) or len(track_ids) > 1000:
                raise ValueError("track_ids must be a bounded list")
            identifiers = sorted({_physical_id(value) for value in track_ids})
            if not identifiers:
                return []
            filters.append("track_id IN (" + ",".join("?" for _ in identifiers) + ")")
            values.extend(identifiers)
        if camera_id is not None:
            filters.append("camera_id=?")
            values.append(camera_id)
        with self._lock:
            rows = self.db.execute(
                f"""SELECT * FROM (
                SELECT *, ROW_NUMBER() OVER(PARTITION BY track_id
                  ORDER BY timestamp DESC,camera_id,frame_evidence_id DESC) AS position
                FROM physical_track_updates WHERE {" AND ".join(filters)})
                WHERE position=1 ORDER BY timestamp DESC,track_id LIMIT ?""",
                (*values, limit),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item.pop("position")
            item["box"] = json.loads(item.pop("box_json")) if item["box_json"] is not None else None
            item.pop("box_json", None)
            item["details"] = json.loads(item.pop("details_json"))
            item["age_seconds"] = as_of - item["timestamp"]
            result.append(item)
        return result

    def history(self, track_id: str, *, since: float, until: float, limit: int = 2000) -> list[dict]:
        """Chronological explicit states for one track within a closed source-time span.

        Missing rows mean no update was recorded, never that the object was absent.
        """
        if not math.isfinite(since) or not math.isfinite(until) or since > until:
            raise ValueError("history requires a finite, ordered source-time span")
        if type(limit) is not int or not 1 <= limit <= 100000:
            raise ValueError("history limit must be an integer from 1 to 100000")
        identifier = _physical_id(track_id)
        with self._lock:
            rows = self.db.execute(
                """SELECT * FROM physical_track_updates WHERE track_id=? AND timestamp>=? AND timestamp<=?
                ORDER BY timestamp,camera_id,frame_evidence_id LIMIT ?""",
                (identifier, since, until, limit),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["box"] = json.loads(item.pop("box_json")) if item["box_json"] is not None else None
            item.pop("box_json", None)
            item["details"] = json.loads(item.pop("details_json"))
            result.append(item)
        return result

    def stats(self):
        with self._lock:
            row = self.db.execute("""SELECT COUNT(*) AS updates,COUNT(DISTINCT track_id) AS tracks,
                MIN(timestamp) AS first_source_time,MAX(timestamp) AS last_source_time
                FROM physical_track_updates""").fetchone()
            return dict(row)

    def close(self):
        with self._lock:
            self.db.close()
