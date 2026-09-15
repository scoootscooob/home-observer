#!/usr/bin/env python3
"""Download a small pinned EPIC action pilot, with honest partial physical labels.

Uses the public Hugging Face SDK without credentials. No generated labels,
Home Assistant fields, teacher models, or train/test participant overlap.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image, ImageDraw

from home_observer.physical_schema import PhysicalWindow

REPO = "lightly-ai/epic-kitchens-100-clips"
REVISION = "e1739df818cbfb7e528fb1ef5005dbb51e96aacd"
OFFICIAL_REVISION = "ea8b40457a400c3fffa1c7f406ef3dc169cc2522"
OFFICIAL_URL = "https://github.com/epic-kitchens/epic-kitchens-100-annotations"
ANNOTATION_HASHES = {
    "train": "a3a7ef2e397bd8af12bd0496b09d61b1f2c1f3f6e1726fa86af50a27f4855e87",
    "validation": "35f7932ba0a1127a96cac215a98d35398946f343e3cea9ad6688ed17eee9d75d",
}
PARTICIPANTS = {"train": ["P01", "P02", "P04", "P06"], "validation": ["P11"], "test": ["P03"]}
FAMILIES = {
    "pickup": {"take", "pick-up"},
    "placement": {"put", "put-down", "put-on", "put-in", "put-into"},
    "opening": {"open"},
    "closing": {"close"},
}
NOUNS = {"bowl", "cup", "mug", "pan", "pot", "spoon", "knife", "plate", "drawer", "cupboard"}
WORKFLOW_IDS = ["P03_106_10", "P03_106_11"]
MASK = {"summary": True, "events": {
    "kind": True, "object_label": True, "started_at": True, "ended_at": True, "description": True,
}}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def opaque(value: str) -> str:
    return hashlib.sha256(("epic-temporal-v1:" + value).encode()).hexdigest()[:20]


def seconds(value: str) -> float:
    h, m, s = value.split(":")
    return round(int(h) * 3600 + int(m) * 60 + float(s), 6)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def select_rows(rows: list[dict], per_family: int) -> list[dict]:
    """Fixed participant split; hash selection only within participant/family.

    Two demonstration clips are selected explicitly and remain test-only.
    Selection never uses model results. Other choices are sorted by a fixed hash.
    """
    chosen = []
    for split, participants in PARTICIPANTS.items():
        for participant in participants:
            candidates = [r for r in rows if r["participant_id"] == participant
                          and len(r["video_id"].split("_")[-1]) == 3
                          and r["noun"] in NOUNS
                          and 0.8 <= seconds(r["stop_timestamp"]) - seconds(r["start_timestamp"]) <= 6]
            for family, verbs in FAMILIES.items():
                pool = [r for r in candidates if r["verb"] in verbs]
                pool.sort(key=lambda r: (r["narration_id"] not in WORKFLOW_IDS, opaque(r["narration_id"])))
                if len(pool) < per_family:
                    raise ValueError(f"Insufficient {participant}/{family}: {len(pool)}")
                chosen.extend(dict(r, split=split, interaction_family=family) for r in pool[:per_family])
    ids = [r["narration_id"] for r in chosen]
    if len(ids) != len(set(ids)) or not set(WORKFLOW_IDS).issubset(ids):
        raise ValueError("Selection duplicates IDs or omits the recipe demonstration")
    return chosen


def make_row(source: dict, clip: Path, output: Path, frames: int) -> dict:
    # Decode all frames sequentially: source-time = action start + frame_index/fps.
    reader = imageio_ffmpeg.read_frames(str(clip), pix_fmt="rgb24")
    metadata = next(reader)
    decoded = list(reader)
    fps = metadata["fps"]
    source_start = seconds(source["start_timestamp"])
    source_end = seconds(source["stop_timestamp"])
    duration = len(decoded) / fps
    if abs(duration - (source_end - source_start)) > max(0.08, 2 / fps):
        raise ValueError(f"Clip/annotation duration mismatch: {source['narration_id']} {duration}")
    valid_count = min(len(decoded), int((source_end - source_start) * fps + 1e-5))
    if valid_count < frames:
        raise ValueError("Too few source frames")
    indices = [round(i * (valid_count - 1) / (frames - 1)) for i in range(frames)]
    clip_key = opaque(source["narration_id"])
    relative = Path("media") / clip_key
    (output / relative).mkdir(parents=True, exist_ok=True)
    window_frames, hashes = [], []
    for index in indices:
        frame_key = opaque(f"{source['narration_id']}:{index}")
        path = relative / f"{frame_key}.jpg"
        Image.frombytes("RGB", metadata["size"], decoded[index]).save(output / path, quality=90)
        stamp = round(source_start + index / fps, 6)
        window_frames.append({"evidence_id": f"ev_{frame_key}", "timestamp": stamp, "path": str(path)})
        hashes.append({"path": str(path), "sha256": sha(output / path), "decoded_frame_index": index,
                       "timestamp": stamp})
    # The original clip has no artificial timestamps; times are recording-relative seconds.
    window = {"window_id": f"win_{clip_key}", "started_at": source_start, "ended_at": source_end,
              "clips": [{"clip_id": f"clip_{clip_key}", "camera_id": "ego", "started_at": source_start,
                         "ended_at": source_end, "frames": window_frames}], "audio": []}
    event = {"kind": source["verb"], "object_label": source["noun"], "started_at": source_start,
             "ended_at": source_end, "description": source["narration"]}
    original = {k: v for k, v in source.items() if k not in {"split", "interaction_family", "original_split"}}
    return {"id": f"epic_{clip_key}", "split": source["split"], "group_id": source["video_id"],
            "source": "EPIC-KITCHENS-100 public human action annotations",
            "task_type": "physical_temporal_partial", "window": window,
            "target": {"summary": source["narration"], "events": [event]}, "supervision_mask": MASK,
            "annotation_coverage": {"exhaustive": False, "negative_intervals_available": False,
                "covered_intervals": [[source_start, source_end]], "evaluated_event_classes": [source["verb"]],
                "event_count_exhaustive": False, "objects_exhaustive": False,
                "confidence_annotated": False, "uncertainty_annotated": False,
                "intervals_are_pretrimmed": True, "false_events_per_hour_supported": False},
            "provenance": {"kind": "real_video_existing_human_annotations", "repo_id": REPO,
                "revision": REVISION, "license": "CC-BY-NC-4.0", "participant_id": source["participant_id"],
                "recording_id": source["video_id"], "home_id": None, "original_split": source["original_split"],
                "original_narration_id": source["narration_id"], "original_annotation": original,
                "interaction_family": source["interaction_family"], "time_basis": "recording_relative_seconds",
                "clip": str(clip.relative_to(output)), "clip_sha256": sha(clip),
                "decoded_frames": len(decoded), "fps": fps, "decoded_duration_s": duration,
                "decoded_frames_distinct": len(set(h["sha256"] for h in hashes)), "frames": hashes,
                "audio_in_model_input": False, "transform": "Mirror action trim +854x480 H.264; local JPEG sampling",
                "source_annotation_labels_are_not_model_input": True}}


def validate_rows(rows: list[dict], output: Path) -> dict:
    participants, recordings, clips, frame_hashes = {}, {}, {}, {}
    for row in rows:
        split = row["split"]
        p = row["provenance"]
        for seen, value in [(participants, p["participant_id"]), (recordings, p["recording_id"]),
                            (clips, p["clip_sha256"])]:
            if value in seen and seen[value] != split:
                raise ValueError("Cross-split participant, recording or clip leakage")
            seen[value] = split
        window = row["window"]
        PhysicalWindow.model_validate(window)
        for frame in p["frames"]:
            previous_split = frame_hashes.setdefault(frame["sha256"], split)
            if previous_split != split:
                raise ValueError("Cross-split frame hash overlap")
        last = window["started_at"] - 1
        for frame in window["clips"][0]["frames"]:
            assert window["started_at"] <= frame["timestamp"] <= window["ended_at"]
            assert frame["timestamp"] > last
            assert (output / frame["path"]).is_file()
            last = frame["timestamp"]
        assert set(row["target"]) == {"summary", "events"}
        assert set(row["target"]["events"][0]) == set(MASK["events"])
        assert not {"actions", "noop", "device_states", "policy"}.intersection(window.keys() | row["target"].keys())
    return {"passed": True, "rows": len(rows), "participant_disjoint": True, "recording_disjoint": True,
            "clip_hash_disjoint": True, "frame_hash_disjoint": True, "physical_schema_valid": True,
            "held_out_homes_verified": False,
            "participants": {s: sorted(p for p, v in participants.items() if v == s) for s in PARTICIPANTS},
            "row_counts": dict(Counter(r["split"] for r in rows)), "all_timestamps_causal": True,
            "all_targets_partial": True, "home_assistant_fields": False}


def prepare(output: Path, cache: Path, per_family: int, frames: int, max_bytes: int) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to replace {output}; choose a new output directory")
    api = HfApi(token=False)
    info = api.dataset_info(REPO, revision=REVISION)
    if info.private or info.gated:
        raise RuntimeError("Public unauthenticated dataset access is no longer available")
    output.mkdir(parents=True)
    (output / "sources").mkdir()
    all_rows = []
    downloaded_bytes = 0
    for split, expected_sha in ANNOTATION_HASHES.items():
        name = f"epic-kitchens-100-annotations/EPIC_100_{split}.csv"
        path = Path(hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION,
                                    token=False, local_dir=cache))
        if sha(path) != expected_sha:
            raise ValueError("Pinned annotation file differs from verified official source")
        downloaded_bytes += path.stat().st_size
        all_rows.extend(dict(r, original_split=split) for r in csv.DictReader(path.open()))
    for name in ["README.md", "license.txt", "cut_clips.py"]:
        source = Path(hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION,
                                      token=False, local_dir=cache))
        shutil.copy2(source, output / "sources" / name)
        downloaded_bytes += source.stat().st_size
    selected = select_rows(all_rows, per_family)
    paths = [f"clips/{r['participant_id']}/{r['narration_id']}.mp4" for r in selected]
    hub_files = list(api.get_paths_info(REPO, paths, repo_type="dataset", revision=REVISION))
    sizes = {f.path: f.size for f in hub_files}
    hashes = {f.path: f.lfs.sha256 for f in hub_files if f.lfs is not None}
    if set(sizes) != set(paths) or downloaded_bytes + sum(sizes.values()) > max_bytes:
        raise ValueError("Missing clip or download byte budget exceeded before media download")

    def download(pair: tuple[dict, str]) -> tuple[dict, Path]:
        row, path = pair
        downloaded = Path(hf_hub_download(REPO, path, repo_type="dataset", revision=REVISION,
                                          token=False, local_dir=cache))
        if downloaded.stat().st_size != sizes[path] or sha(downloaded) != hashes.get(path):
            raise ValueError("Downloaded clip differs from pinned Hub size/SHA256")
        target = output / "clips" / f"{opaque(row['narration_id'])}.mp4"
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(downloaded, target)
        return row, target

    with ThreadPoolExecutor(max_workers=6) as pool:
        assets = list(pool.map(download, zip(selected, paths)))
    rows = []
    for index, (source, clip) in enumerate(assets):
        rows.append(make_row(source, clip, output, frames))
        print(f"Prepared {index + 1}/{len(assets)}", flush=True)
    for split in PARTICIPANTS:
        write_jsonl(output / f"{split}.jsonl", [r for r in rows if r["split"] == split])
    validation = validate_rows(rows, output)
    write_json(output / "validation-report.json", validation)
    # One continuous source clip; the separate pickup clip is not stitched into this replay.
    by_id = {r["provenance"]["original_narration_id"]: r for r in rows}
    workflow = [by_id["P03_106_11"]]
    workflow.sort(key=lambda r: r["window"]["started_at"])
    write_jsonl(output / "recipe-demo.jsonl", workflow)
    write_json(output / "recipe-demo.json", {
        "name": "Put the bowl down", "recipe_step": "Put the bowl down on the work surface.",
        "recipe_step_origin": "Authored demonstration instruction; not an original recipe annotation",
        "rows": [r["id"] for r in workflow], "split": "test", "recording_id": "P03_106",
        "unobserved_gap_s": 0,
        "continuous_video": True, "workflow_completion_ground_truth": False,
        "supported_ground_truth": "One source-labeled bowl placement and its source interval; recipe wording is authored",
        "source_target_labels_are_not_model_input": True,
    })
    canvas = Image.new("RGB", (4 * 320, len(workflow) * 204), "white")
    draw = ImageDraw.Draw(canvas)
    for row_i, row in enumerate(workflow):
        fs = row["window"]["clips"][0]["frames"]
        for col, idx in enumerate([0, len(fs) // 3, 2 * len(fs) // 3, len(fs) - 1]):
            frame = fs[idx]
            canvas.paste(Image.open(output / frame["path"]).resize((320, 180)), (col * 320, row_i * 204))
            draw.text((col * 320 + 4, row_i * 204 + 183), f"Source t={frame['timestamp']:.2f}s", fill="black")
    canvas.save(output / "recipe-contact-sheet.jpg", quality=90)
    manifest = {
        "version": 1, "complete": True, "dataset": "EPIC-KITCHENS-100 bounded physical temporal pilot",
        "repo_id": REPO, "revision": REVISION, "access": "Public unauthenticated Hugging Face SDK",
        "license": "CC-BY-NC-4.0", "license_path": "sources/license.txt",
        "official_annotations": {"url": OFFICIAL_URL, "revision": OFFICIAL_REVISION,
            "sha256": ANNOTATION_HASHES, "mirror_byte_identical_verified": True},
        "preparer_sha256": sha(Path(__file__)), "seed": "epic-temporal-v1 hash ordering",
        "split_policy": "Fixed participant-disjoint custom split; not the official benchmark split",
        "split_participants": PARTICIPANTS, "counts": validation["row_counts"],
        "recordings": {s: sorted({r['group_id'] for r in rows if r['split'] == s}) for s in PARTICIPANTS},
        "interaction_families": {s: dict(Counter(r["provenance"]["interaction_family"] for r in rows
                                                 if r["split"] == s)) for s in PARTICIPANTS},
        "frames_per_clip": frames, "download_bytes": downloaded_bytes + sum(sizes.values()),
        "download_budget_bytes": max_bytes, "output_bytes_before_manifest": sum(p.stat().st_size
                                                                                  for p in output.rglob("*")
                                                                                  if p.is_file()),
        "source_video_seconds": sum(r["provenance"]["decoded_duration_s"] for r in rows),
        "files": {f"{s}.jsonl": sha(output / f"{s}.jsonl") for s in PARTICIPANTS},
        "coverage": {"positive_action_trimmed_clips_only": True, "exhaustive": False,
            "verified_home_ids": False, "false_events_per_hour_supported": False,
            "supports_free_stream_temporal_localization": False, "unlabeled_absence_is_negative": False},
        "preserved_baseline": True, "no_home_assistant_training_fields": True,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/temporal-pilot"))
    parser.add_argument("--cache", type=Path, required=True, help="Scratch SDK download directory")
    parser.add_argument("--per-family", type=int, default=4)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--max-download-mb", type=int, default=200)
    args = parser.parse_args()
    if not 1 <= args.per_family <= 12 or not 2 <= args.frames <= 16:
        parser.error("per-family must be 1..12 and frames 2..16")
    result = prepare(args.output.resolve(), args.cache.resolve(), args.per_family, args.frames,
                     args.max_download_mb * 1_000_000)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
