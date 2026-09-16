#!/usr/bin/env python3
"""Write the interactive teacher pass of 2026-09-16 (Claude reviewing tick sheets in a session,
narrations hidden, then checked) as stream-target rows. Corrected decisions are the targets;
the blind reading is kept in the audit block where it differed."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teacher_label import SCHEMA_VERSION, TickDecision, sha256_file, tick_grid  # noqa: E402

GOAL = "Keep the household coordinator aware of what people do with objects, appliances and food in this room."
CLIPS = Path("data/continuous-v2-workflow")
VIDEOS = Path("data/continuous-v2/videos")


def obs(kind, subject, location, detail, confidence):
    return {"kind": kind, "subject": subject, "location": location, "detail": detail, "confidence": confidence}


SOURCES = [
    {"video": CLIPS / "positive-take-lid-P27_03.mp4", "video_id": "P27_03", "offset": 44.5, "hz": 2.0,
     "pushes": {48.0: obs("object_taken", "lid", "sink and drainer area", "right hand lifts a lid from beside the sink", 0.8)},
     "audit": {"narration_checked": ["take lid 47.68-49.90 s matches", "put down lid 48.98-55.03 s ends in the next clip; no placement visible here"],
               "blind_pass_differences": []}},
    {"video": CLIPS / "positive-take-pan-P27_03.mp4", "video_id": "P27_03", "offset": 52.5, "hz": 2.0,
     "pushes": {54.5: obs("container_state", "frying pan", "drawer under the counter", "glass lid placed onto the pan in the drawer", 0.7),
                57.5: obs("object_taken", "frying pan", "drawer under the counter", "pan lifted out and carried toward the hob", 0.75),
                59.0: obs("object_placed", "frying pan", "hob", "pan set down on the hob", 0.7)},
     "audit": {"narration_checked": ["put down lid 48.98-55.03 s", "take pan 56.54-57.99 s", "put down pan 57.90-62.17 s"],
               "blind_pass_differences": [{"t": 53.0, "blind": obs("object_taken", "frying pan", "drawer under the counter", "read as the pan being lifted", 0.6),
                                           "corrected": "the first two seconds are the lid being placed on the pan; the take happens during the camera swing at 56.5-58.0 s"}]}},
    {"video": CLIPS / "positive-take-spoon-P28_07.mp4", "video_id": "P28_07", "offset": 16.0, "hz": 2.0,
     "pushes": {16.5: obs("appliance_state", "oven", "oven", "door open, roasting tray being pulled out with an oven glove", 0.85),
                22.5: obs("object_taken", "spoon", "counter next to the oven", "left hand picks a spoon up from the counter", 0.75),
                24.5: obs("activity_started", "stirring vegetables in the roasting tray", "oven", "spoon turning the roasted vegetables", 0.85)},
     "audit": {"narration_checked": ["take spoon 22.15-22.90 s matches", "stir food 23.99-30.73 s matches", "tray pull-out at 16.0-21.5 s has no narration inside the clip window"],
               "blind_pass_differences": []}},
    {"video": CLIPS / "negative-move-plate-P22_105.mp4", "video_id": "P22_105", "offset": 55.0, "hz": 2.0,
     "pushes": {61.0: obs("food_state", "mozzarella slices", "dining table, right plate", "slices moved from the cutting plate onto the second plate with a fork", 0.8)},
     "audit": {"narration_checked": ["move plate 57.95-59.00 s and 64.27-65.25 s: silent under the default goal (repositioned, not lifted)",
                                     "pick up mozzarella 59.76-60.99 s and put down mozzarella 61.02-61.83 s: one served observation",
                                     "put down cutlery 62.08-63.94 s: silent"],
               "goal_dependence": "under a plate-related recipe goal the 58.0 s slide would be one object_moved line",
               "blind_pass_differences": []}},
    {"video": VIDEOS / "P22_105.mp4", "video_id": "P22_105", "offset": 40.0, "end": 69.0, "hz": 1.0,
     "pushes": {42.0: obs("object_placed", "plate", "dining table", "plate set down on the table (narration-corrected; not seen at 1 Hz)", 0.6),
                44.0: obs("activity_started", "cutting mozzarella", "dining table, left plate", "knife cutting mozzarella on the plate", 0.85),
                61.0: obs("food_state", "mozzarella slices", "dining table, right plate", "slices served onto the second plate with a fork", 0.8),
                67.0: obs("person_presence", "person", "kitchen counter", "left the dining table for the kitchen counter", 0.75)},
     "audit": {"narration_checked": ["put down plate 41.66-42.65 s: missed in the blind pass at 1 Hz, added from narration",
                                     "pick up knife 42.67-43.29 s: folded into the activity start", "cut mozzarella 43.25-57.44 s matches",
                                     "plate slides 57.95-59.00 s and 64.27-65.25 s silent", "presence changes have no narration"],
               "blind_pass_differences": [{"t": 42.0, "blind": "person_presence: arrived at the dining table",
                                           "corrected": "kept as the plate placement; the arrival is real but was 40-41 s and outside the narrated set"}]}},
]


def main():
    out = Path("data/teacher-seed/interactive-2026-09-16.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for src in SOURCES:
        import imageio_ffmpeg
        meta = next(imageio_ffmpeg.read_frames(str(src["video"]), pix_fmt="rgb24", output_params=["-t", "0.05"]))
        duration = float(meta["duration"])
        start = src["offset"]
        end = src.get("end", start + duration)
        grid = tick_grid(start, end, src["hz"])
        ticks = []
        journal = []
        for t in grid:
            push = src["pushes"].get(round(t, 3))
            if push:
                tick = TickDecision(t=t, decision="context", observation=push, why="first tick where the state change is visibly established")
                journal.append({"t": t, **push})
            else:
                tick = TickDecision(t=t, decision="silent")
            ticks.append(tick.check().model_dump(exclude_none=True))
        sha = sha256_file(src["video"])
        rows.append({
            "schema_version": SCHEMA_VERSION, "prompt_version": "teacher-v1-interactive",
            "chunk_id": hashlib.sha256(f"{sha}:{start:.3f}:{end:.3f}:{src['hz']}".encode()).hexdigest()[:20],
            "source": {"video_id": src["video_id"], "sha256": sha, "start_s": start, "end_s": round(end, 3),
                       "camera": "ego", "license": "CC BY-NC 4.0", "path": str(src["video"])},
            "tick_hz": src["hz"],
            "context": {"goal": GOAL, "action_policy": [], "device_state": {}, "journal_prior": [], "camera": "ego"},
            "ticks": ticks, "chunk_summary": "",
            "teacher": {"model": "claude-fable-5-1", "mode": "interactive", "frames_seen": "tick sheets, 400 px tiles",
                        "narrations_hidden_during_pass": True, "second_looks": []},
            "audit": src["audit"], "created_at": datetime.now(timezone.utc).isoformat()})
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    pushes = sum(1 for r in rows for t in r["ticks"] if t["decision"] == "context")
    total = sum(len(r["ticks"]) for r in rows)
    print(json.dumps({"rows": len(rows), "ticks": total, "context": pushes, "silent": total - pushes, "out": str(out)}))


if __name__ == "__main__":
    main()
