#!/usr/bin/env python3
"""Cut a demonstration segment from a benchmark video with full provenance.

The clip is re-encoded from the 480p benchmark re-encode for frame-accurate
bounds. The sidecar records the source video hash, the offsets, the frame rate,
and every official narration overlapping the segment, so the digital instruction
and the annotated actions stay separate and inspectable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seconds(value: str) -> float:
    hours, minutes, rest = value.split(":")
    return round(int(hours) * 3600 + int(minutes) * 60 + float(rest), 6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="854x480 benchmark re-encode")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True, help="Official EPIC_100_train.csv")
    parser.add_argument("--role", choices=["positive", "negative"], required=True)
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite an existing workflow clip")
    if not 0 <= args.start < args.end or args.end - args.start > 120:
        raise ValueError("segment must be positive and at most 120 s")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    duration = args.end - args.start
    command = [ffmpeg, "-y", "-loglevel", "error", "-ss", f"{args.start:.3f}", "-i", str(args.video),
               "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-an",
               "-pix_fmt", "yuv420p", str(args.output)]
    started = time.time()
    subprocess.run(command, check=True)
    rows = [row for row in csv.DictReader(args.annotations.open()) if row["video_id"] == args.video_id]
    overlapping = []
    for row in rows:
        start, stop = seconds(row["start_timestamp"]), seconds(row["stop_timestamp"])
        if stop >= args.start and start <= args.end:
            overlapping.append({"narration_id": row["narration_id"], "narration": row["narration"],
                                "verb": row["verb"], "noun": row["noun"], "start": start, "stop": stop,
                                "clip_relative_start": round(start - args.start, 3),
                                "clip_relative_stop": round(stop - args.start, 3)})
    sidecar = {"role": args.role, "source_video": str(args.video), "source_video_sha256": sha(args.video),
               "video_id": args.video_id, "source_start_s": args.start, "source_end_s": args.end,
               "duration_s": duration, "clip_sha256": sha(args.output), "encode_seconds": round(time.time() - started, 2),
               "annotations_sha256": sha(args.annotations), "overlapping_narrations": overlapping,
               "note": args.note, "time_basis": "clip-relative seconds; add source_start_s for recording time",
               "selection": "chosen by inspecting footage and official narrations only; no model predictions used"}
    args.output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(json.dumps({k: v for k, v in sidecar.items() if k != "overlapping_narrations"}, indent=2))
    print(json.dumps(overlapping, indent=1))


if __name__ == "__main__":
    main()
