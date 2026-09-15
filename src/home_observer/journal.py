"""Durable event log and bounded, expiring materialized state."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class Journal:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS windows(
          id TEXT PRIMARY KEY, started_at REAL, ended_at REAL, status TEXT NOT NULL,
          input_json TEXT NOT NULL, result_json TEXT, error TEXT);
        CREATE TABLE IF NOT EXISTS facts(
          entity_id TEXT, attribute TEXT, value_json TEXT, confidence REAL,
          observed_at REAL, evidence_json TEXT, source TEXT,
          PRIMARY KEY(entity_id, attribute));
        CREATE TABLE IF NOT EXISTS events(
          id INTEGER PRIMARY KEY AUTOINCREMENT, window_id TEXT, timestamp REAL,
          kind TEXT, payload_json TEXT);
        CREATE INDEX IF NOT EXISTS events_time ON events(timestamp);
        CREATE TABLE IF NOT EXISTS actions(
          id TEXT PRIMARY KEY, window_id TEXT, timestamp REAL, signature TEXT,
          status TEXT NOT NULL, action_json TEXT, result_json TEXT);
        CREATE INDEX IF NOT EXISTS actions_sig_time ON actions(signature, timestamp);
        """)
        self.db.commit()

    def begin(self, window: dict) -> bool:
        """An interrupted window cannot silently retry potentially executed actions."""
        cur = self.db.execute(
            "INSERT OR IGNORE INTO windows VALUES(?,?,?,?,?,?,?)",
            (window["window_id"], window["started_at"], window["ended_at"], "processing",
             json.dumps(window), None, None),
        )
        self.db.commit()
        return cur.rowcount == 1

    def latest_timestamp(self) -> float:
        value = self.db.execute("SELECT MAX(ended_at) FROM windows").fetchone()[0]
        return float(value) if value is not None else float("-inf")

    def finish(self, window_id: str, result: dict, error: str | None = None):
        self.db.execute("UPDATE windows SET status=?, result_json=?, error=? WHERE id=?",
                        ("error" if error else "complete", json.dumps(result), error, window_id))
        self.db.commit()

    def get_window(self, window_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM windows WHERE id=?", (window_id,)).fetchone()
        if not row:
            return None
        return {"window_id": row["id"], "status": row["status"], "error": row["error"],
                "result": json.loads(row["result_json"]) if row["result_json"] else None}

    def event(self, window_id: str, timestamp: float, kind: str, payload: dict):
        self.db.execute("INSERT INTO events(window_id,timestamp,kind,payload_json) VALUES(?,?,?,?)",
                        (window_id, timestamp, kind, json.dumps(payload)))
        self.db.commit()

    def update_fact(self, entity: str, attribute: str, value: Any, confidence: float,
                    timestamp: float, evidence: list[str], source: str = "model"):
        self.db.execute("""INSERT INTO facts VALUES(?,?,?,?,?,?,?)
          ON CONFLICT(entity_id,attribute) DO UPDATE SET value_json=excluded.value_json,
          confidence=excluded.confidence, observed_at=excluded.observed_at,
          evidence_json=excluded.evidence_json,source=excluded.source
          WHERE excluded.observed_at >= facts.observed_at""",
                        (entity, attribute, json.dumps(value), confidence, timestamp, json.dumps(evidence), source))
        self.db.commit()

    def state(self, now: float, ttl: float = 300, limit: int = 64) -> dict:
        rows = self.db.execute("SELECT * FROM facts WHERE observed_at<=? ORDER BY observed_at DESC LIMIT ?",
                               (now, limit)).fetchall()
        result: dict[str, dict] = {}
        for row in rows:
            stale = now - row["observed_at"] > ttl
            result.setdefault(row["entity_id"], {})[row["attribute"]] = {
                "value": None if stale else json.loads(row["value_json"]),
                "confidence": 0 if stale else row["confidence"],
                "observed_at": row["observed_at"], "stale": stale, "source": row["source"],
                "evidence_ids": json.loads(row["evidence_json"]),
            }
        return result

    def recent(self, now: float, limit: int = 8, max_chars: int = 8000) -> list[dict]:
        rows = self.db.execute("SELECT * FROM events WHERE timestamp<=? AND kind='decision' ORDER BY timestamp DESC,id DESC LIMIT ?",
                               (now, limit)).fetchall()
        result, size = [], 0
        for row in rows:
            item = {"window_id": row["window_id"], "timestamp": row["timestamp"],
                    "kind": row["kind"], "payload": json.loads(row["payload_json"])}
            length = len(json.dumps(item))
            if size + length > max_chars:
                break
            result.append(item)
            size += length
        return list(reversed(result))

    def search(self, query: str, before: float, limit: int = 20) -> list[dict]:
        # LIKE wildcards are escaped; the query is never executable SQL.
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        rows = self.db.execute("SELECT * FROM events WHERE timestamp<=? AND payload_json LIKE ? ESCAPE '\\' ORDER BY timestamp DESC LIMIT ?",
                               (before, f"%{escaped}%", min(limit, 100))).fetchall()
        return [{"timestamp": r["timestamp"], "window_id": r["window_id"], "kind": r["kind"],
                 "payload": json.loads(r["payload_json"])} for r in rows]

    def reserve_action(self, action_id: str, window_id: str, timestamp: float,
                       signature: str, action: dict, cooldown: float) -> bool:
        # Reserve before sending: timeout/crash leaves status pending, not retryable.
        with self.db:
            prior = self.db.execute("""SELECT 1 FROM actions WHERE signature=? AND
                (status IN ('pending','uncertain') OR
                 (status='complete' AND timestamp BETWEEN ? AND ?))""",
                                    (signature, timestamp - cooldown, timestamp)).fetchone()
            if prior:
                return False
            cur = self.db.execute("INSERT OR IGNORE INTO actions VALUES(?,?,?,?,?,?,?)",
                                  (action_id, window_id, timestamp, signature, "pending", json.dumps(action), None))
            return cur.rowcount == 1

    def complete_action(self, action_id: str, status: str, result: dict):
        with self.db:
            self.db.execute("UPDATE actions SET status=?,result_json=? WHERE id=?",
                            (status, json.dumps(result), action_id))

    def stats(self) -> dict:
        return {
            "windows": dict(self.db.execute("SELECT status,COUNT(*) FROM windows GROUP BY status").fetchall()),
            "actions": dict(self.db.execute("SELECT status,COUNT(*) FROM actions GROUP BY status").fetchall()),
            "events": self.db.execute("SELECT COUNT(*) FROM events").fetchone()[0],
            "facts": self.db.execute("SELECT COUNT(*) FROM facts").fetchone()[0],
        }

    def close(self):
        self.db.close()
