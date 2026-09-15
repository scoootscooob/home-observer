#!/usr/bin/env python3
"""Prepare four temporally offset inputs from ONE public household recording.

These are not synchronized multiview cameras or four-room coverage. Distinct
decoded frames avoid identical-image inputs in a bounded transport benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, output):
    source = source.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    if any(output.iterdir()):
        raise ValueError("use an empty output directory; existing inputs are not overwritten")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    _, source_duration = imageio_ffmpeg.count_frames_and_secs(str(source))
    if source_duration < 17:
        raise ValueError("source must contain at least 17 seconds")
    slots = [("kitchen", 0), ("entry", 3), ("lounge", 6), ("garage", 9)]

    def segment(slot):
        camera, offset = slot
        path = output / f"{camera}-offset-{offset}s.mp4"
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", str(offset),
                   "-i", str(source), "-t", "8", "-vf", "scale=384:-2", "-r", "10",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-threads", "1",
                   "-c:a", "aac", "-ar", "16000", "-ac", "1", "-movflags", "+faststart", str(path)]
        subprocess.run(command, check=True, capture_output=True, timeout=60)
        frames, duration = imageio_ffmpeg.count_frames_and_secs(str(path))
        if abs(duration - 8) > .15 or frames < 79:
            raise ValueError(f"unexpected decoded duration/frame count for {camera}: {duration}/{frames}")
        decoded = {}
        for timestamp in (0, 4):
            pixels = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
                "-ss", str(timestamp), "-i", str(path), "-frames:v", "1", "-f", "rawvideo",
                "-pix_fmt", "rgb24", "pipe:1"], check=True, capture_output=True, timeout=15).stdout
            if not pixels:
                raise ValueError(f"no decoded frame for {camera} at {timestamp}s")
            decoded[str(timestamp)] = hashlib.sha256(pixels).hexdigest()
        return {"camera_id": camera, "path": str(path), "source_offset_s": offset,
                "requested_duration_s": 8, "decoded_duration_s": duration, "decoded_frames": frames,
                "sha256": sha256(path), "bytes": path.stat().st_size,
                "decoded_rgb24_sha256_by_segment_second": decoded}

    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(segment, slots))
    distinct = {str(timestamp): len({r["decoded_rgb24_sha256_by_segment_second"][str(timestamp)]
                                   for r in records}) for timestamp in (0, 4)}
    if any(value != 4 for value in distinct.values()):
        raise ValueError("decoded camera inputs are not all different at the sampled positions")
    sources = {record["camera_id"]: record["path"] for record in records}
    sources_path = output / "sources.json"
    sources_path.write_text(json.dumps(sources, indent=2) + "\n")
    report = {"source": str(source), "source_sha256": sha256(source),
              "source_decoded_duration_s": source_duration, "segments": records,
              "distinct_decoded_frames_per_sample": distinct, "sources_sha256": sha256(sources_path),
              "scope": "Four temporally offset segments from ONE EgoLife recording. Camera names are routing labels only; not genuine simultaneous multiview or four-room coverage.",
              "transforms": "8-second segments at offsets 0/3/6/9 seconds; video resized to384px width at10fps; source audio retained as16kHz mono AAC.",
              "verification": "All four RGB24 decoded frame hashes differ at segment seconds0 and4; every clip decodes to about8seconds. This verifies distinct inputs, not model quality."}
    (output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    return {"sources": str(sources_path), "provenance": str(output / "provenance.json"),
            "audio_source": records[0]["path"], "segments": len(records),
            "distinct_frames": distinct, "durations": [r["decoded_duration_s"] for r in records]}


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=project / "data/public-egolife/raw/A1_JAKE/DAY1/DAY1_A1_JAKE_11094208.mp4")
    parser.add_argument("--output", type=Path, default=project.parents[1] / "work/live-distinct-inputs")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
