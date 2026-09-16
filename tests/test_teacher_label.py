"""Pure parts of the teacher labeling pass: tick grids, reply validation, prompt layout, seed rows."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import teacher_label as tl  # noqa: E402


def test_tick_grid_counts_and_spacing():
    grid = tl.tick_grid(0.0, 8.51, 2.0)
    assert len(grid) == 17 and grid[0] == 0.0 and grid[-1] == 8.0
    assert tl.tick_grid(40.0, 69.0, 1.0) == [float(t) for t in range(40, 69)]


def good_reply(times):
    ticks = [{"t": t, "decision": "silent"} for t in times]
    ticks[1] = {"t": times[1], "decision": "context", "why": "lid lifted",
                "observation": {"kind": "object_taken", "subject": "lid", "location": "sink", "detail": "", "confidence": 0.8}}
    return {"ticks": ticks, "chunk_summary": "one lid taken"}


def test_parse_reply_accepts_prose_around_json_and_checks_grid():
    times = tl.tick_grid(0.0, 2.0, 2.0)
    text = "Here you go:\n" + json.dumps(good_reply(times)) + "\nDone."
    reply = tl.parse_reply(text, times)
    assert [t.decision for t in reply.ticks] == ["silent", "context", "silent", "silent"]
    with pytest.raises(ValueError, match="grid"):
        tl.parse_reply(json.dumps(good_reply(times)), tl.tick_grid(0.0, 2.5, 2.0))


@pytest.mark.parametrize("mutate, message", [
    (lambda tick: tick.update(decision="context", observation=None), "needs an observation"),
    (lambda tick: tick["observation"].update(kind="teleport"), "known kind"),
    (lambda tick: tick.update(decision="act", observation=None), "needs an action"),
    (lambda tick: tick.update(decision="silent"), "must not carry a payload"),
    (lambda tick: tick.update(decision="maybe"), "unknown decision"),
])
def test_parse_reply_rejects_inconsistent_ticks(mutate, message):
    times = tl.tick_grid(0.0, 2.0, 2.0)
    reply = good_reply(times)
    mutate(reply["ticks"][1])
    with pytest.raises(ValueError, match=message):
        tl.parse_reply(json.dumps(reply), times)


def test_parse_reply_rejects_unknown_fields():
    times = tl.tick_grid(0.0, 1.0, 2.0)
    reply = good_reply(times)
    reply["ticks"][0]["frame_path"] = "/etc/passwd"
    with pytest.raises(Exception):
        tl.parse_reply(json.dumps(reply), times)


def test_user_content_labels_lead_in_frames_and_lists_decision_ticks():
    frames = [b"jpeg"] * 6
    times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    content = tl.build_user_content(frames, times, context_frames=2, context={"goal": "g"}, schema_hint={"ticks": []})
    images = [block for block in content if block["type"] == "image"]
    labels = [block["text"] for block in content if block["type"] == "text"]
    assert len(images) == 6
    assert sum(label.startswith("[lead-in") for label in labels) == 2
    order = labels[-1].split("in order:")[1].split("Allowed")[0]
    assert "1.00, 1.50, 2.00, 2.50" in order and "0.50" not in order
    assert all("data" in block["source"] for block in images)


def test_committed_seed_rows_validate():
    path = ROOT / "data" / "teacher-seed" / "interactive-2026-09-16.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == 5
    for row in rows:
        assert row["schema_version"] == tl.SCHEMA_VERSION
        times = [tick["t"] for tick in row["ticks"]]
        assert times == tl.tick_grid(row["source"]["start_s"], row["source"]["end_s"], row["tick_hz"])
        for tick in row["ticks"]:
            tl.TickDecision.model_validate(tick).check()
        assert row["teacher"]["mode"] == "interactive" and row["teacher"]["narrations_hidden_during_pass"] is True
    pushes = sum(tick["decision"] == "context" for row in rows for tick in row["ticks"])
    total = sum(len(row["ticks"]) for row in rows)
    assert pushes / total < 0.2, "silence must dominate the seed"
