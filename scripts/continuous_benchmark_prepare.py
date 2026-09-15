#!/usr/bin/env python3
"""Prepare the frozen continuous EPIC-KITCHENS-100 benchmark (continuous-v2).

Untrimmed official videos from participants disjoint from the temporal pilot are
re-encoded to 854x480 at their source frame rate, tiled into fixed 3.0 s windows
with a 1.5 s stride, and labelled from the official dense narrations. The split
is written before any download or decoding and is never changed from content.
No credentials, no model predictions and no Home Assistant fields are involved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import io
import json
import math
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import imageio_ffmpeg
from PIL import Image

from home_observer.physical_schema import PhysicalWindow, validate_physical_decision

NAMESPACE = "epic-continuous-v2"
USER_AGENT = "home-observer-continuous-v2-preparer"
OFFICIAL_URL = "https://github.com/epic-kitchens/epic-kitchens-100-annotations"
OFFICIAL_REVISION = "ea8b40457a400c3fffa1c7f406ef3dc169cc2522"
ANNOTATION_FILES = {
    "EPIC_100_train.csv": "a3a7ef2e397bd8af12bd0496b09d61b1f2c1f3f6e1726fa86af50a27f4855e87",
    "EPIC_100_validation.csv": "35f7932ba0a1127a96cac215a98d35398946f343e3cea9ad6688ed17eee9d75d",
    "EPIC_100_video_info.csv": "75fd040f6662cb407b4ca2eed4811ca94df280a1aba00b6bf8a5cd286deff382",
    "EPIC_100_train_missing_timestamp_narrations.csv":
        "dce53e33f68178293f8bd339e657bfa78d39ea02a06bd7c217ee7d9b939f8450",
    "EPIC_100_validation_missing_timestamp_narrations.csv":
        "395773f385f21e39154fcee2c53050f5c4ae6fab743ed0d0f3d406ab64215657",
    "Extension_Participants.csv": "896b36886807ac7e64da767b0445d70292c5be3aa2d4b4a74b01679cdc65fdb1",
    "license.txt": "acb7384c3f73cc96367ac132bc0bd57efa38c4495c17781032a2cf1444b65ebe",
}
NARRATION_FILES = ("EPIC_100_train.csv", "EPIC_100_validation.csv")
MISSING_FILES = ("EPIC_100_train_missing_timestamp_narrations.csv",
                 "EPIC_100_validation_missing_timestamp_narrations.csv")
PILOT_PARTICIPANTS = ("P01", "P02", "P03", "P04", "P06", "P11")
FROZEN_SPLIT = {"test": ["P28_10", "P30_01", "P22_105", "P27_03"], "validation": ["P26_16"],
                "train": ["P09_04", "P25_12", "P14_05", "P07_01", "P23_01", "P13_06"]}
# Alternates in the stated priority order. Same-participant alternates are preferred for a replacement,
# "avoid" entries are a last resort, and "others" expands to that participant's remaining videos, shortest
# first. Participant disjointness across splits (which covers the P26/P25 conditions) is always enforced.
ALTERNATES = {
    "test": [{"video": "P26_17", "unless": "P26 is used in validation"},
             {"video": "P25_01", "unless": "P25 is used in train"}, {"video": "P30_112"}, {"video": "P28_07"},
             {"video": "P22_101", "avoid": "long"}, {"others": "P27"}],
    "validation": [{"video": "P26_01"}, {"video": "P26_11"}],
    "train": [{"video": "P07_02"}, {"video": "P14_01"}, {"video": "P14_07"}, {"video": "P09_105"},
              {"others": "P13"}, {"others": "P23"}, {"video": "P31_04"}, {"video": "P19_02"}, {"video": "P15_12"}],
}
OLD_URL = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/{subset}/{participant}/{video}.MP4"
NEW_URL = "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/{participant}/videos/{video}.MP4"
FPS_RATIONAL = {"59.9400599400599": Fraction(60000, 1001), "50.0": Fraction(50)}
EVALUATED_CLASSES = ["take", "pick-up", "put", "put-down", "put-on", "put-in", "put-into", "open", "close"]
DEMO_NOUNS = ["bowl", "cup", "mug", "pan", "pot", "plate", "jar", "bottle", "container", "glass", "lid", "board"]
DEMO_POSITIVE_VERBS = ["take", "pick-up"]
DEMO_NEGATIVE_VERBS = ["wash", "stir", "move", "wipe", "cut", "mix", "shake", "scoop"]
WINDOW = Fraction(3)
STRIDE = Fraction(3, 2)
FRAMES = 8
TARGET_SIZE = (854, 480)
DURATION_TOLERANCE_S = 0.5
MIN_ALTERNATE_DURATION_S = 30
LICENSE = "CC-BY-NC-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc/4.0/"
ATTRIBUTION = "Dima Damen and the EPIC-KITCHENS authors (University of Bristol)"
EVENT_FIELDS = ("kind", "object_label", "started_at", "ended_at", "description")
CITATION_FIELDS = ("pre_evidence_ids", "post_evidence_ids")
HOME_IDENTITY_STATEMENT = (
    "Held-out home identity is asserted from the EPIC-KITCHENS one-participant-one-kitchen recording protocol "
    "(each participant recorded in their own kitchen) together with the Extension_Participants.csv "
    "returning/changing-kitchen notes for the 2020 extension recordings. It was not independently verified: "
    "the dataset publishes no shared-home identifiers, so only participant and recording separation is checked.")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def opaque(key: str) -> str:
    return hashlib.sha256(f"{NAMESPACE}:{key}".encode()).hexdigest()[:20]


def parse_timestamp(value: str) -> Fraction:
    hours, minutes, seconds = value.strip().split(":")
    return int(hours) * 3600 + int(minutes) * 60 + Fraction(seconds)


def f6(value: Fraction) -> float:
    return round(float(value), 6)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def participant_of(video_id: str) -> str:
    return video_id.split("_")[0]


def candidate_urls(video_id: str) -> list[str]:
    participant, suffix = video_id.split("_")
    if int(suffix) < 100:
        return [OLD_URL.format(subset=subset, participant=participant, video=video_id) for subset in ("train", "test")]
    return [NEW_URL.format(participant=participant, video=video_id)]


def load_annotations(directory: Path) -> dict:
    """Verify every pinned annotation file, then index narrations, video info, missing lists and notes."""
    files = {}
    for name, expected in ANNOTATION_FILES.items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing annotation file {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{name} SHA-256 {actual} differs from the pinned value {expected}")
        files[name] = {"sha256": actual, "bytes": path.stat().st_size}
    narrations: dict[str, list[dict]] = {}
    for name in NARRATION_FILES:
        with (directory / name).open(newline="") as handle:
            for row in csv.DictReader(handle):
                narrations.setdefault(row["video_id"], []).append(dict(row, source_file=name))
    for rows in narrations.values():
        rows.sort(key=lambda r: (parse_timestamp(r["start_timestamp"]), parse_timestamp(r["stop_timestamp"]),
                                 int(r["narration_id"].rsplit("_", 1)[1])))
    with (directory / "EPIC_100_video_info.csv").open(newline="") as handle:
        video_info = {row["video_id"]: row for row in csv.DictReader(handle)}
    missing: dict[str, list[str]] = {}
    for name in MISSING_FILES:
        with (directory / name).open(newline="") as handle:
            for row in csv.DictReader(handle):
                narration_id = row["narration_id"]
                missing.setdefault(narration_id.rsplit("_", 1)[0], []).append(f"{narration_id} ({name})")
    with (directory / "Extension_Participants.csv").open(newline="") as handle:
        extension = {row["Participant"]: row for row in csv.DictReader(handle)}
    return {"directory": directory, "files": files, "narrations": narrations, "video_info": video_info,
            "missing": missing, "extension": extension}


def http_head(url: str) -> tuple[int, int | None]:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    error: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                length = response.headers.get("Content-Length")
                return response.status, int(length) if length else None
        except urllib.error.HTTPError as failure:
            if failure.code in (403, 404, 410):
                return failure.code, None
            error = failure
        except (urllib.error.URLError, OSError, http.client.HTTPException) as failure:
            error = failure
        time.sleep(2 ** attempt)
    raise RuntimeError(f"HEAD request failed for {url}: {error}")


def check_video(video_id: str, split: str, assigned: dict[str, list[str | None]], annotations: dict) -> dict:
    """Content-independent eligibility: annotation coverage, participant disjointness and URL availability."""
    participant = participant_of(video_id)
    info = annotations["video_info"].get(video_id)
    check = {"video_id": video_id, "participant_id": participant, "split": split, "eligible": False,
             "missing_timestamp_narrations": annotations["missing"].get(video_id, []),
             "narrations": len(annotations["narrations"].get(video_id, [])), "head_checks": []}
    conflict = next((other for other, videos in assigned.items() if other != split
                     and any(video and participant_of(video) == participant for video in videos)), None)
    if info is None:
        check["reason"] = "not listed in EPIC_100_video_info.csv"
    elif participant in PILOT_PARTICIPANTS:
        check["reason"] = "participant was used by the temporal pilot"
    elif conflict:
        check["reason"] = f"participant {participant} is already assigned to the {conflict} split"
    elif check["missing_timestamp_narrations"]:
        check["reason"] = ("video has narrations with missing timestamps (incomplete timing coverage): "
                           + ", ".join(check["missing_timestamp_narrations"]))
    elif info["fps"] not in FPS_RATIONAL:
        check["reason"] = f"unsupported source frame rate {info['fps']}"
    elif check["narrations"] == 0:
        check["reason"] = "no official narrations in the train/validation annotation files"
    else:
        check["official_video_info"] = dict(info)
        for url in candidate_urls(video_id):
            status, length = http_head(url)
            check["head_checks"].append({"url": url, "http_status": status, "content_length": length})
            if status == 200 and length:
                check.update(eligible=True, url=url, content_length=length)
                break
        else:
            check["reason"] = "not available from the official public server (HEAD did not return 200)"
    return check


def alternates_for(split: str, excluded: str, annotations: dict, assigned: dict[str, list[str | None]]) -> list[str]:
    participant = participant_of(excluded)
    taken = {video for videos in assigned.values() for video in videos if video}
    ordered, avoided = [], []
    for item in ALTERNATES[split]:
        if "others" in item:
            videos = [video for video, info in annotations["video_info"].items()
                      if participant_of(video) == item["others"]
                      and float(info["duration"]) >= MIN_ALTERNATE_DURATION_S]
            videos.sort(key=lambda video: (float(annotations["video_info"][video]["duration"]), video))
        else:
            videos = [item["video"]]
        for video in videos:
            if video not in taken and video != excluded:
                (avoided if item.get("avoid") else ordered).append(video)
    same = [video for video in ordered if participant_of(video) == participant]
    rest = [video for video in ordered if participant_of(video) != participant]
    return same + rest + avoided


def resolve_split(annotations: dict) -> dict:
    """Apply the frozen assignment; substitute only for missing-timestamp or unavailable videos."""
    assigned: dict[str, list[str | None]] = {split: list(videos) for split, videos in FROZEN_SPLIT.items()}
    checks, substitutions, considered = {}, [], []
    for split in ("validation", "train", "test"):
        for position, requested in enumerate(FROZEN_SPLIT[split]):
            check = check_video(requested, split, assigned, annotations)
            checks[requested] = check
            if check["eligible"]:
                log(f"{requested} ({split}): available, {check['content_length']} bytes at {check['url']}")
                continue
            log(f"{requested} ({split}): excluded, {check['reason']}")
            assigned[split][position] = None
            replacement = None
            for candidate in alternates_for(split, requested, annotations, assigned):
                candidate_check = check_video(candidate, split, assigned, annotations)
                considered.append(dict(candidate_check, replacing=requested))
                if candidate_check["eligible"]:
                    replacement = candidate
                    checks[candidate] = candidate_check
                    break
                log(f"  alternate {candidate} rejected: {candidate_check['reason']}")
            if replacement is None:
                raise RuntimeError(f"No eligible replacement for {requested} in the {split} split")
            assigned[split][position] = replacement
            substitutions.append({"split": split, "requested": requested, "replacement": replacement,
                                  "reason": check["reason"],
                                  "same_participant": participant_of(replacement) == participant_of(requested)})
            log(f"{requested} ({split}): replaced by {replacement}")
    final = {split: [video for video in videos if video] for split, videos in assigned.items()}
    participants = {split: sorted({participant_of(video) for video in videos}) for split, videos in final.items()}
    for split, group in participants.items():
        for other, other_group in participants.items():
            if split < other and set(group) & set(other_group):
                raise RuntimeError("Participants overlap across splits")
        if set(group) & set(PILOT_PARTICIPANTS):
            raise RuntimeError("A pilot participant entered the split")
    return {
        "frozen_at_utc": utc_now(), "frozen_before_download_and_decoding": True,
        "policy": ("Fixed participant-disjoint and recording-disjoint assignment, disjoint from the temporal pilot "
                   "participants. Substitutions are allowed only for videos listed in the official "
                   "missing-timestamp narration files or unavailable from the official server, decided by "
                   "annotation lists and HTTP HEAD checks before any download or decoding. The split is never "
                   "changed from video content, labels or model results."),
        "pilot_participants_excluded": list(PILOT_PARTICIPANTS), "requested": FROZEN_SPLIT, "assigned": final,
        "participants": participants, "substitutions": substitutions, "alternates_policy": ALTERNATES,
        "alternates_considered": considered, "video_checks": checks,
    }


class Budget:
    def __init__(self, limit: int) -> None:
        self.limit, self.received, self.lock = limit, 0, threading.Lock()

    def add(self, count: int) -> None:
        with self.lock:
            self.received += count
            if self.received > self.limit:
                raise RuntimeError(f"Download budget of {self.limit} bytes exceeded; aborting")


def download(url: str, destination: Path, expected: int, budget: Budget, label: str) -> dict:
    """Resumable streaming download; the file is renamed into place only at the verified size."""
    started = time.monotonic()
    if destination.exists():
        if destination.stat().st_size != expected:
            raise RuntimeError(f"{destination} exists with an unexpected size; remove it to download again")
        log(f"{label}: reusing the complete earlier download ({expected} bytes)")
        return {"reused": True, "bytes_received": 0, "seconds": 0.0, "attempts": 0}
    part = destination.with_name(destination.name + ".part")
    received, error, attempt = 0, None, 0
    for attempt in range(1, 9):
        have = part.stat().st_size if part.exists() else 0
        if have > expected:
            raise RuntimeError(f"{part} is larger than the announced size; remove it to download again")
        if have == expected:
            break
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
                if have and response.status != 206:
                    log(f"{label}: server ignored the range request; restarting from byte 0")
                    have = 0
                with part.open("ab" if have else "wb") as handle:
                    next_report = have + (64 << 20)
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        handle.write(chunk)
                        budget.add(len(chunk))
                        received += len(chunk)
                        have += len(chunk)
                        if have >= next_report:
                            log(f"{label}: {have / 1e6:.0f}/{expected / 1e6:.0f} MB")
                            next_report += 64 << 20
            if have == expected:
                break
            error = RuntimeError(f"connection ended at {have}/{expected} bytes")
        except (urllib.error.URLError, OSError, http.client.HTTPException) as failure:
            error = failure
        log(f"{label}: attempt {attempt} incomplete ({error}); retrying")
        time.sleep(min(60, 2 ** attempt))
    else:
        raise RuntimeError(f"Download failed for {url}: {error}")
    if part.stat().st_size != expected:
        raise RuntimeError(f"{part} has {part.stat().st_size} bytes, expected {expected}")
    part.replace(destination)
    seconds = round(time.monotonic() - started, 3)
    log(f"{label}: download complete, {expected} bytes in {seconds} s")
    return {"reused": False, "bytes_received": received, "seconds": seconds, "attempts": attempt}


def reencode(ffmpeg: str, source: Path, target: Path, fps: Fraction) -> list[str]:
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    rate = str(fps.numerator) if fps.denominator == 1 else f"{fps.numerator}/{fps.denominator}"
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(source),
               "-vf", f"scale={TARGET_SIZE[0]}:{TARGET_SIZE[1]}", "-r", rate, "-fps_mode", "cfr",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-threads", "1",
               "-an", "-pix_fmt", "yuv420p", str(target)]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0 or not target.is_file():
        raise RuntimeError(f"ffmpeg failed for {source.name}: {completed.stderr[-2000:]}")
    return command


def plan_windows(frame_count: int, fps: Fraction) -> list[dict]:
    """Fixed 3.0 s windows with a 1.5 s stride over the decoded duration; 8 uniformly sampled frames each."""
    duration = Fraction(frame_count) / fps
    windows = []
    index = 0
    while index * STRIDE + WINDOW <= duration:
        start = index * STRIDE
        first = math.ceil(start * fps)
        last = math.ceil((start + WINDOW) * fps) - 1
        count = last - first + 1
        if count < FRAMES:
            raise RuntimeError(f"Window {index} holds only {count} frames")
        sampled = [first + round(Fraction(position * (count - 1), FRAMES - 1)) for position in range(FRAMES)]
        windows.append({"index": index, "start": start, "end": start + WINDOW, "first_frame": first,
                        "last_frame": last, "frame_count": count, "sampled": sampled})
        index += 1
    return windows


def extract_frames(video: Path, fps: Fraction, windows: list[dict], output: Path,
                   video_id: str) -> tuple[int, dict]:
    """One sequential decode of the re-encode; frame index i has timestamp i / fps."""
    needed: dict[int, list[int]] = {}
    for window in windows:
        for index in window["sampled"]:
            needed.setdefault(index, []).append(window["index"])
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
    metadata = next(reader)
    size = tuple(metadata["size"])
    if size != TARGET_SIZE:
        raise RuntimeError(f"Re-encode of {video_id} decodes at {size}, expected {TARGET_SIZE}")
    if abs(float(metadata["fps"]) - float(fps)) > 0.05:
        raise RuntimeError(f"Re-encode of {video_id} reports {metadata['fps']} fps, expected {float(fps)}")
    frames: dict[tuple[int, int], dict] = {}
    decoded = 0
    for index, data in enumerate(reader):
        decoded += 1
        if index not in needed:
            continue
        buffer = io.BytesIO()
        Image.frombytes("RGB", size, data).save(buffer, format="JPEG", quality=90)
        blob = buffer.getvalue()
        digest = hashlib.sha256(blob).hexdigest()
        for window_index in needed[index]:
            directory = Path("media") / opaque(f"{video_id}:{window_index}")
            (output / directory).mkdir(parents=True, exist_ok=True)
            path = directory / f"{opaque(f'{video_id}:{index}')}.jpg"
            (output / path).write_bytes(blob)
            frames[(window_index, index)] = {"path": str(path), "sha256": digest}
    missing = [key for window in windows for key in ((window["index"], i) for i in window["sampled"])
               if key not in frames]
    if missing:
        raise RuntimeError(f"{video_id}: {len(missing)} sampled frames were not decoded")
    return decoded, frames


def subtract_intervals(start: Fraction, end: Fraction, holes: list[tuple[Fraction, Fraction]]) -> list:
    covered = []
    cursor = start
    for hole_start, hole_end in sorted(holes):
        hole_start, hole_end = max(hole_start, start), min(hole_end, end)
        if hole_end <= hole_start:
            continue
        if hole_start > cursor:
            covered.append((cursor, hole_start))
        cursor = max(cursor, hole_end)
    if cursor < end:
        covered.append((cursor, end))
    return covered


def window_labels(start: Fraction, end: Fraction, narrations: list[tuple[dict, Fraction, Fraction]]) -> tuple:
    """Targets: narrations with at least half their duration inside the window; others that overlap are listed."""
    targets, unselected = [], []
    for narration, first, last in narrations:
        duration = last - first
        overlap = min(last, end) - max(first, start)
        if duration > 0:
            overlaps = overlap > 0
            selected = overlaps and overlap * 2 >= duration
        else:
            overlaps = start <= first < end
            selected = overlaps
        if not overlaps:
            continue
        clipped = (max(first, start), min(last, end))
        fraction = f6(overlap / duration) if duration > 0 else 1.0
        (targets if selected else unselected).append((narration, clipped, fraction))
    return targets, unselected


def event_mask(has_events: bool, derivable: bool) -> bool | dict:
    """Per-field mask for windows with target events; boolean mask (abstention) for empty windows."""
    if not has_events:
        return True
    return {**{field: True for field in EVENT_FIELDS}, **{field: derivable for field in CITATION_FIELDS}}


def build_rows(meta: dict, windows: list[dict], frames: dict, narrations: list[dict]) -> list[dict]:
    video_id, fps = meta["video_id"], meta["fps_fraction"]
    parsed = [(n, parse_timestamp(n["start_timestamp"]), parse_timestamp(n["stop_timestamp"])) for n in narrations]
    rows = []
    for window in windows:
        start, end, index = window["start"], window["end"], window["index"]
        key = opaque(f"{video_id}:{index}")
        sampled = []
        for frame_index in window["sampled"]:
            info = frames[(index, frame_index)]
            sampled.append({"index": frame_index, "stamp": Fraction(frame_index) / fps, "path": info["path"],
                            "sha256": info["sha256"], "evidence_id": f"ev_{opaque(f'{video_id}:{frame_index}')}"})
        targets, unselected = window_labels(start, end, parsed)
        events, citations, derivable = [], [], True
        for narration, (first, last), _ in targets:
            pre = [frame for frame in sampled if frame["stamp"] <= first]
            post = [frame for frame in sampled if first < frame["stamp"] <= last]
            citations.append((pre[-1]["evidence_id"] if pre else None, post[-1]["evidence_id"] if post else None))
            derivable = derivable and bool(pre) and bool(post)
            events.append({"kind": narration["verb"], "object_label": narration["noun"], "started_at": f6(first),
                           "ended_at": f6(last), "description": narration["narration"]})
        if derivable:
            for event, (pre_id, post_id) in zip(events, citations, strict=True):
                event["pre_evidence_ids"], event["post_evidence_ids"] = [pre_id], [post_id]
        holes = [clipped for narration, clipped, _ in unselected if narration["verb"] in EVALUATED_CLASSES]
        covered = [[f6(a), f6(b)] for a, b in subtract_intervals(start, end, holes)]
        quiet = not targets and not unselected
        mask = {"summary": True, "events": event_mask(bool(events), derivable)}
        clip_frames = [{"evidence_id": frame["evidence_id"], "timestamp": f6(frame["stamp"]), "path": frame["path"]}
                       for frame in sampled]
        window_input = {"window_id": f"win_{key}", "started_at": f6(start), "ended_at": f6(end),
                        "clips": [{"clip_id": f"clip_{key}", "camera_id": "ego", "started_at": f6(start),
                                   "ended_at": f6(end), "frames": clip_frames}], "audio": []}
        rows.append({
            "id": f"cont_{key}", "split": meta["split"], "group_id": video_id,
            "source": "EPIC-KITCHENS-100 official untrimmed video with public dense human narrations",
            "task_type": "physical_temporal_continuous", "window": window_input,
            "target": {"summary": "; ".join(n["narration"] for n, _, _ in targets) or "no annotated action",
                       "events": events},
            "supervision_mask": mask,
            "annotation_coverage": {
                "exhaustive": True, "event_count_exhaustive": True, "negative_intervals_available": True,
                "timeline_id": video_id, "covered_intervals": covered,
                "evaluated_event_classes": list(EVALUATED_CLASSES), "intervals_are_pretrimmed": False,
                "false_events_per_hour_supported": bool(covered), "objects_exhaustive": False,
                "boxes_annotated": False, "identity_annotated": False, "confidence_annotated": False,
                "uncertainty_annotated": False, "narration_is_exhaustive_for_actions": True,
                "quiet_window": quiet, "evidence_citations_derivable": derivable,
                "unscored_intervals": [[f6(a), f6(b)] for a, b in sorted(holes) if b > a],
                "target_selection_rule": "narration with at least 50% of its duration inside the window",
            },
            "provenance": {
                "kind": "real_video_existing_human_annotations", "dataset": "EPIC-KITCHENS-100",
                "source_url": meta["source_url"], "source_video_sha256": meta["source_sha256"],
                "source_video_bytes": meta["source_bytes"], "official_video_info": meta["official_video_info"],
                "participant_id": meta["participant_id"], "recording_id": video_id, "home_id": None,
                "extension_participant_row": meta["extension_participant_row"],
                "official_annotation_revision": OFFICIAL_REVISION,
                "original_split": meta["original_split"], "license": LICENSE, "license_url": LICENSE_URL,
                "attribution": ATTRIBUTION, "time_basis": "recording_relative_seconds",
                "window_index": index, "window_s": float(WINDOW), "stride_s": float(STRIDE),
                "window_frame_range": [window["first_frame"], window["last_frame"]],
                "reencoded_video": meta["reencoded_video"], "reencoded_video_sha256": meta["reencoded_sha256"],
                "fps": float(fps), "fps_rational": f"{fps.numerator}/{fps.denominator}",
                "decoded_frames_total": meta["decoded_frames"], "decoded_duration_s": meta["decoded_duration_s"],
                "frames": [{"path": frame["path"], "sha256": frame["sha256"], "decoded_frame_index": frame["index"],
                            "timestamp": f6(frame["stamp"])} for frame in sampled],
                "transform": meta["transform"], "ffmpeg_command": meta["ffmpeg_command"],
                "audio_in_model_input": False, "source_annotation_labels_are_not_model_input": True,
                "original_annotations": [dict(n, clipped_interval=[f6(a), f6(b)], fraction_inside_window=fraction)
                                         for n, (a, b), fraction in targets],
                "unselected_overlapping_annotations": [
                    dict(n, clipped_interval=[f6(a), f6(b)], fraction_inside_window=fraction,
                         reason="less than 50% of the narration interval lies inside the window",
                         subtracted_from_covered_intervals=n["verb"] in EVALUATED_CLASSES)
                    for n, (a, b), fraction in unselected],
            },
        })
    return rows


def process_video(entry: dict, annotations: dict, output: Path, source: Path, ffmpeg: str,
                  transfer: dict) -> tuple[list[dict], dict]:
    video_id, split = entry["video_id"], entry["split"]
    info = annotations["video_info"][video_id]
    fps = FPS_RATIONAL[info["fps"]]
    size = source.stat().st_size
    if size != entry["content_length"]:
        raise RuntimeError(f"{video_id}: downloaded {size} bytes, HEAD announced {entry['content_length']}")
    log(f"{video_id}: hashing source ({size} bytes)")
    source_sha = sha256_file(source)
    target = output / "videos" / f"{video_id}.mp4"
    log(f"{video_id}: re-encoding to {TARGET_SIZE[0]}x{TARGET_SIZE[1]} at {info['fps']} fps")
    encode_started = time.monotonic()
    command = reencode(ffmpeg, source, target, fps)
    frame_count, counted_seconds = imageio_ffmpeg.count_frames_and_secs(str(target))
    if not frame_count:
        raise RuntimeError(f"{video_id}: could not count decoded frames")
    decoded_duration = Fraction(frame_count) / fps
    official = Fraction(info["duration"])
    difference = float(abs(decoded_duration - official))
    if difference > DURATION_TOLERANCE_S:
        raise RuntimeError(f"{video_id}: decoded {float(decoded_duration):.3f} s, official {float(official):.3f} s")
    windows = plan_windows(frame_count, fps)
    log(f"{video_id}: {frame_count} frames = {float(decoded_duration):.3f} s (official {float(official):.3f} s, "
        f"difference {difference:.3f} s), {len(windows)} windows, encode {time.monotonic() - encode_started:.0f} s")
    decoded, frames = extract_frames(target, fps, windows, output, video_id)
    if decoded != frame_count:
        raise RuntimeError(f"{video_id}: sequential decode produced {decoded} frames, counted {frame_count}")
    narrations = annotations["narrations"][video_id]
    original_splits = sorted({n["source_file"].split("_")[2].split(".")[0] for n in narrations})
    meta = {
        "video_id": video_id, "split": split, "participant_id": participant_of(video_id),
        "source_url": entry["url"], "source_bytes": size, "source_sha256": source_sha,
        "http_head_content_length": entry["content_length"], "download": transfer,
        "official_video_info": dict(info), "official_duration_s": float(official),
        "extension_participant_row": annotations["extension"].get(participant_of(video_id)),
        "original_split": original_splits[0] if len(original_splits) == 1 else original_splits,
        "fps_fraction": fps, "fps": float(fps), "reencoded_video": str(target.relative_to(output)),
        "reencoded_sha256": sha256_file(target), "reencoded_bytes": target.stat().st_size,
        "decoded_frames": frame_count, "decoded_duration_s": f6(decoded_duration),
        "ffmpeg_reported_seconds": counted_seconds, "duration_difference_s": round(difference, 6),
        "windows": len(windows), "tiled_seconds": f6(windows[-1]["end"]) if windows else 0.0,
        "narrations": len(narrations),
        "transform": (f"ffmpeg -vf scale={TARGET_SIZE[0]}:{TARGET_SIZE[1]} -r <source fps> -fps_mode cfr -c:v libx264 "
                      "-preset veryfast -crf 23 -threads 1 -an -pix_fmt yuv420p; one sequential rgb24 decode of the "
                      f"re-encode; {FRAMES} uniformly sampled frames per window saved as JPEG quality 90"),
        "ffmpeg_command": [part.replace(str(source), f"<download-dir>/{source.name}")
                           .replace(str(output), "<output>") for part in command],
    }
    rows = build_rows(meta, windows, frames, narrations)
    if len(rows) != len(windows) or sum(len(r["provenance"]["frames"]) for r in rows) != FRAMES * len(windows):
        raise RuntimeError(f"{video_id}: row/window bookkeeping mismatch")
    if len({n["narration_id"] for n in narrations}) != len(narrations):
        raise RuntimeError(f"{video_id}: duplicate narration IDs")
    write_json(output / "sources" / "annotations" / f"{video_id}.json", {
        "video_id": video_id, "participant_id": meta["participant_id"], "split": split,
        "official_annotation_revision": OFFICIAL_REVISION, "annotation_files": annotations["files"],
        "official_video_info": dict(info),
        "extension_participant_row": meta["extension_participant_row"],
        "missing_timestamp_narrations": annotations["missing"].get(video_id, []),
        "narrations": [dict(n) for n in narrations],
    })
    log(f"{video_id}: {len(rows)} rows, {sum(r['annotation_coverage']['quiet_window'] for r in rows)} quiet, "
        f"{sum(len(r['target']['events']) for r in rows)} target events")
    return rows, meta


def validate_rows(rows: list[dict], output: Path, split: dict) -> dict:
    """Leakage, schema, causal-citation and coverage checks over every prepared window."""
    seen: dict[str, dict[str, str]] = {name: {} for name in ("participant", "recording", "source_video_sha256",
                                                              "reencoded_video_sha256", "frame_sha256")}

    def note(kind: str, value: str, split_name: str) -> None:
        if seen[kind].setdefault(value, split_name) != split_name:
            raise ValueError(f"Cross-split {kind} leakage: {value}")

    ids: set[str] = set()
    counts: Counter = Counter()
    quiet: Counter = Counter()
    citation_rows: Counter = Counter()
    empty_coverage: Counter = Counter()
    busy_without_targets: Counter = Counter()
    classes: dict[str, Counter] = {name: Counter() for name in FROZEN_SPLIT}
    runtime_accepted = frames_checked = 0
    distinct_frame_hashes: set[str] = set()
    for row in rows:
        split_name, provenance, window = row["split"], row["provenance"], row["window"]
        if row["id"] in ids or row["id"] != "cont_" + window["window_id"][4:]:
            raise ValueError(f"Duplicate or inconsistent row id {row['id']}")
        ids.add(row["id"])
        if provenance["participant_id"] not in split["participants"][split_name]:
            raise ValueError("Row participant is outside the frozen split")
        note("participant", provenance["participant_id"], split_name)
        note("recording", provenance["recording_id"], split_name)
        note("source_video_sha256", provenance["source_video_sha256"], split_name)
        note("reencoded_video_sha256", provenance["reencoded_video_sha256"], split_name)
        PhysicalWindow.model_validate(window)
        if len(window["clips"]) != 1 or window["audio"] or window["clips"][0]["camera_id"] != "ego":
            raise ValueError("Window must hold exactly one ego clip and no audio")
        frames = window["clips"][0]["frames"]
        if len(frames) != FRAMES:
            raise ValueError("Unexpected sampled frame count")
        last = None
        for frame, record in zip(frames, provenance["frames"], strict=True):
            if not window["started_at"] <= frame["timestamp"] <= window["ended_at"]:
                raise ValueError("Frame outside its window")
            if last is not None and frame["timestamp"] <= last:
                raise ValueError("Frames are not strictly chronological")
            last = frame["timestamp"]
            path = output / frame["path"]
            if record["path"] != frame["path"] or not path.is_file() or sha256_file(path) != record["sha256"]:
                raise ValueError(f"Frame file/hash mismatch for {frame['path']}")
            if abs(record["decoded_frame_index"] / provenance["fps"] - frame["timestamp"]) > 1e-5:
                raise ValueError("Frame timestamp is not decoded_frame_index / fps")
            note("frame_sha256", record["sha256"], split_name)
            distinct_frame_hashes.add(record["sha256"])
            frames_checked += 1
        target, mask, coverage = row["target"], row["supervision_mask"], row["annotation_coverage"]
        derivable = coverage["evidence_citations_derivable"]
        fields = set(EVENT_FIELDS) | (set(CITATION_FIELDS) if derivable else set())
        if set(target) != {"summary", "events"} or mask != {
                "summary": True, "events": event_mask(bool(target["events"]), derivable)}:
            raise ValueError("Target or supervision mask has an unexpected shape")
        evidence = {frame["evidence_id"]: frame["timestamp"] for frame in frames}
        for event in target["events"]:
            if set(event) != fields:
                raise ValueError("Event fields do not match the supervision mask")
            if not window["started_at"] <= event["started_at"] <= event["ended_at"] <= window["ended_at"]:
                raise ValueError("Event interval is outside its window")
            if derivable:
                pre, post = event["pre_evidence_ids"], event["post_evidence_ids"]
                if (len(pre) != 1 or len(post) != 1 or pre[0] == post[0]
                        or not evidence[pre[0]] <= event["started_at"] < evidence[post[0]] <= event["ended_at"]):
                    raise ValueError("Derived evidence citations are not causal")
        if derivable and target["events"]:
            decision = {"summary": target["summary"], "events": [
                dict(event, event_id=f"evt_{position}", confidence=1.0,
                     uncertainty="Interval from the source narration; citations derived deterministically")
                for position, event in enumerate(target["events"])]}
            validate_physical_decision(decision, window)
            runtime_accepted += 1
        cursor = window["started_at"]
        for pair in coverage["covered_intervals"]:
            first, second = pair
            if not (window["started_at"] <= first < second <= window["ended_at"] and first >= cursor):
                raise ValueError("Covered intervals must be ordered, disjoint and inside the window")
            cursor = second
        unselected = provenance["unselected_overlapping_annotations"]
        if coverage["quiet_window"] != (not target["events"] and not unselected):
            raise ValueError("quiet_window disagrees with the overlapping narrations")
        if coverage["quiet_window"] and (coverage["covered_intervals"] != [[window["started_at"], window["ended_at"]]]
                                         or target["summary"] != "no annotated action"):
            raise ValueError("Quiet window must be fully covered with no annotated action")
        if coverage["false_events_per_hour_supported"] != bool(coverage["covered_intervals"]):
            raise ValueError("false_events_per_hour_supported must follow covered_intervals")
        if {"actions", "noop", "device_states", "policy"} & (window.keys() | target.keys()):
            raise ValueError("Home Assistant fields are not allowed in physical rows")
        counts[split_name] += 1
        quiet[split_name] += coverage["quiet_window"]
        citation_rows[split_name] += derivable
        empty_coverage[split_name] += not coverage["covered_intervals"]
        busy_without_targets[split_name] += not target["events"] and not coverage["quiet_window"]
        classes[split_name].update(event["kind"] for event in target["events"])
    if set(seen["participant"]) & set(PILOT_PARTICIPANTS):
        raise ValueError("A pilot participant appears in the continuous benchmark")
    return {
        "passed": True, "rows": len(rows), "participant_disjoint": True, "recording_disjoint": True,
        "video_hash_disjoint": True, "frame_hash_disjoint": True, "physical_schema_valid": True,
        "frame_files_verified": frames_checked, "distinct_frame_hashes": len(distinct_frame_hashes),
        "frame_timestamps_are_index_over_fps": True, "all_timestamps_causal": True,
        "runtime_validator_accepted_rows": runtime_accepted, "pilot_participants_disjoint": True,
        "held_out_homes_verified": False, "home_identity_statement": HOME_IDENTITY_STATEMENT,
        "participants": {name: sorted(p for p, s in seen["participant"].items() if s == name) for name in FROZEN_SPLIT},
        "recordings": {name: sorted(r for r, s in seen["recording"].items() if s == name) for name in FROZEN_SPLIT},
        "row_counts": {name: counts[name] for name in FROZEN_SPLIT},
        "quiet_windows": {name: quiet[name] for name in FROZEN_SPLIT},
        "evidence_citation_rows": {name: citation_rows[name] for name in FROZEN_SPLIT},
        "rows_without_covered_intervals": {name: empty_coverage[name] for name in FROZEN_SPLIT},
        "non_quiet_windows_without_targets": {name: busy_without_targets[name] for name in FROZEN_SPLIT},
        "target_events_per_class": {name: dict(sorted(classes[name].items())) for name in FROZEN_SPLIT},
        "all_targets_partial": True, "home_assistant_fields": False,
    }


def demo_candidates(split: dict, annotations: dict, rows: list[dict]) -> dict:
    """Test-split narrations a human can choose from for pickup / handling-without-pickup demonstrations."""
    as_target: dict[str, list[str]] = {}
    as_overlap: dict[str, list[str]] = {}
    for row in rows:
        if row["split"] != "test":
            continue
        for item in row["provenance"]["original_annotations"]:
            as_target.setdefault(item["narration_id"], []).append(row["id"])
        for item in row["provenance"]["unselected_overlapping_annotations"]:
            as_overlap.setdefault(item["narration_id"], []).append(row["id"])
    positives, negatives = [], []
    for video_id in split["assigned"]["test"]:
        for narration in annotations["narrations"][video_id]:
            head = narration["noun"].split(":")[0]
            if narration["noun"] not in DEMO_NOUNS and head not in DEMO_NOUNS:
                continue
            entry = {"narration_id": narration["narration_id"], "video_id": video_id,
                     "participant_id": narration["participant_id"], "narration": narration["narration"],
                     "verb": narration["verb"], "noun": narration["noun"], "noun_head": head,
                     "noun_match": "exact" if narration["noun"] in DEMO_NOUNS else "head",
                     "start_timestamp": narration["start_timestamp"], "stop_timestamp": narration["stop_timestamp"],
                     "start_s": f6(parse_timestamp(narration["start_timestamp"])),
                     "stop_s": f6(parse_timestamp(narration["stop_timestamp"])),
                     "duration_s": f6(parse_timestamp(narration["stop_timestamp"])
                                      - parse_timestamp(narration["start_timestamp"])),
                     "target_row_ids": as_target.get(narration["narration_id"], []),
                     "overlapping_unselected_row_ids": as_overlap.get(narration["narration_id"], [])}
            if narration["verb"] in DEMO_POSITIVE_VERBS:
                positives.append(entry)
            elif narration["verb"] in DEMO_NEGATIVE_VERBS:
                negatives.append(entry)
    return {"purpose": ("Human selection of one positive pickup and one negative handling-without-pickup "
                        "demonstration from the test split; no model output was consulted"),
            "test_videos": split["assigned"]["test"], "noun_set": DEMO_NOUNS,
            "noun_match_rule": "noun equals a listed noun, or its head before ':' does (e.g. board:chopping)",
            "positive_verbs": DEMO_POSITIVE_VERBS, "negative_verbs": DEMO_NEGATIVE_VERBS,
            "positive_candidates": positives, "negative_candidates": negatives,
            "time_basis": "recording_relative_seconds"}


def attribution_text(split: dict, videos: dict) -> str:
    lines = [
        "# Attribution", "",
        "This benchmark redistributes derived media from **EPIC-KITCHENS-100** by Dima Damen and the EPIC-KITCHENS "
        "authors (University of Bristol and collaborators), licensed under the Creative Commons "
        f"Attribution-NonCommercial 4.0 International License ({LICENSE_URL}). The license text is in "
        "`license.txt`. Credit: Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari, Evangelos "
        "Kazakos, Jian Ma, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price and Michael Wray, "
        "*Rescaling Egocentric Vision: Collection, Pipeline and Challenges for EPIC-KITCHENS-100*, IJCV 2022; "
        "and Damen et al., *Scaling Egocentric Vision: The EPIC-KITCHENS Dataset*, ECCV 2018.", "",
        "The material is used here for non-commercial research. No endorsement by the licensor is implied.", "",
        "## Changes made", "",
        f"- Videos were re-encoded to {TARGET_SIZE[0]}x{TARGET_SIZE[1]} H.264 at the source frame rate with the audio "
        "track removed; the originals are not redistributed.",
        f"- Each video was tiled into {float(WINDOW)} s windows with a {float(STRIDE)} s stride and {FRAMES} JPEG "
        "frames were sampled per window.",
        "- Official dense narrations (verb, noun, start/stop time, narration text) were converted into window "
        "labels. Narration text and labels are kept outside the model input.",
        f"- Annotations come from {OFFICIAL_URL} at revision {OFFICIAL_REVISION}.", "",
        "## Sources", "",
    ]
    for split_name, ids in split["assigned"].items():
        for video_id in ids:
            lines.append(f"- `{video_id}` ({split_name}): {videos[video_id]['source_url']}")
    return "\n".join(lines) + "\n"


def prepare(output: Path, download_dir: Path, annotations_dir: Path, budget_bytes: int, workers: int) -> dict:
    started_utc, started = utc_now(), time.monotonic()
    if output.exists():
        raise FileExistsError(f"Refusing to replace {output}; choose a new output directory")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    log("Verifying annotation files")
    annotations = load_annotations(annotations_dir)
    log("Resolving the frozen split (annotation lists and HTTP HEAD checks only)")
    split = resolve_split(annotations)
    if output.exists():
        raise FileExistsError(f"Refusing to replace {output}; choose a new output directory")
    output.mkdir(parents=True)
    for name in ("videos", "media", "sources", "sources/annotations"):
        (output / name).mkdir(parents=True)
    shutil.copyfile(annotations_dir / "license.txt", output / "sources" / "license.txt")
    write_json(output / "sources" / "annotation-files.json", {
        "url": OFFICIAL_URL, "revision": OFFICIAL_REVISION, "local_directory": str(annotations_dir),
        "files": annotations["files"]})
    write_json(output / "split.json", split)
    log(f"split.json frozen at {split['frozen_at_utc']}: {json.dumps(split['assigned'])}")
    entries = [dict(split["video_checks"][video_id], split=split_name)
               for split_name in ("test", "validation", "train") for video_id in split["assigned"][split_name]]
    planned = sum(entry["content_length"] for entry in entries)
    if planned > budget_bytes:
        raise RuntimeError(f"Planned downloads ({planned} bytes) exceed the budget ({budget_bytes} bytes); "
                           "aborting before any download")
    log(f"Planned downloads: {planned} bytes across {len(entries)} videos (budget {budget_bytes} bytes)")
    download_dir.mkdir(parents=True, exist_ok=True)
    budget = Budget(budget_bytes)
    downloads_started = time.monotonic()
    videos: dict[str, dict] = {}
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download, entry["url"], download_dir / f"{entry['video_id']}.MP4",
                               entry["content_length"], budget, entry["video_id"]): entry for entry in entries}
        for future in as_completed(futures):
            entry = futures[future]
            transfer = future.result()
            if transfer["reused"] is False:
                transfer["download_wall_seconds_since_start"] = round(time.monotonic() - downloads_started, 3)
            video_rows, meta = process_video(entry, annotations, output, download_dir / f"{entry['video_id']}.MP4",
                                             ffmpeg, transfer)
            rows.extend(video_rows)
            videos[entry["video_id"]] = meta
    download_seconds = round(time.monotonic() - downloads_started, 3)
    processing_finished_utc = utc_now()
    entry_ids = [entry["video_id"] for entry in entries]
    order = {video_id: position for position, video_id in enumerate(entry_ids)}
    rows.sort(key=lambda row: (order[row["group_id"]], row["provenance"]["window_index"]))
    if [row["group_id"] for row in rows if row["provenance"]["window_index"] == 0] != entry_ids:
        raise RuntimeError("Every assigned video must contribute windows")
    for split_name in FROZEN_SPLIT:
        write_jsonl(output / f"{split_name}.jsonl", [row for row in rows if row["split"] == split_name])
    log("Validating every row")
    validation = validate_rows(rows, output, split)
    validation["validation_script_sha256"] = sha256_file(Path(__file__))
    validation["validated_at_utc"] = utc_now()
    write_json(output / "validation-report.json", validation)
    write_json(output / "candidates.json", demo_candidates(split, annotations, rows))
    (output / "sources" / "attribution.md").write_text(attribution_text(split, videos))
    serializable = {video_id: {key: value for key, value in meta.items() if key != "fps_fraction"}
                    for video_id, meta in videos.items()}
    narration_ids_selected = {item["narration_id"] for row in rows for item in row["provenance"]["original_annotations"]}
    narrations_total = sum(meta["narrations"] for meta in videos.values())
    long_evaluated = sorted(n["narration_id"] for video_id in videos for n in annotations["narrations"][video_id]
                            if n["verb"] in EVALUATED_CLASSES
                            and parse_timestamp(n["stop_timestamp"]) - parse_timestamp(n["start_timestamp"]) > 2 * WINDOW)
    manifest = {
        "version": 2, "complete": True,
        "dataset": "EPIC-KITCHENS-100 continuous benchmark v2: untrimmed official videos with dense narrations",
        "opaque_namespace": NAMESPACE, "preparer_sha256": sha256_file(Path(__file__)),
        "timestamps_utc": {"preparation_started": started_utc, "split_frozen": split["frozen_at_utc"],
                           "processing_finished": processing_finished_utc, "manifest_written": utc_now()},
        "license": LICENSE, "license_url": LICENSE_URL, "license_path": "sources/license.txt",
        "attribution": ATTRIBUTION, "attribution_path": "sources/attribution.md",
        "official_annotations": {"url": OFFICIAL_URL, "revision": OFFICIAL_REVISION, "files": annotations["files"],
                                 "narration_files": list(NARRATION_FILES), "missing_timestamp_files": list(MISSING_FILES)},
        "access": "Official public download server, no credentials",
        "split_policy": split["policy"], "split_participants": split["participants"],
        "split_recordings": split["assigned"], "requested_split": FROZEN_SPLIT,
        "substitutions": split["substitutions"], "pilot_participants_excluded": list(PILOT_PARTICIPANTS),
        "videos": serializable,
        "download": {"budget_bytes": budget_bytes, "planned_bytes": planned,
                     "bytes_received_this_run": budget.received,
                     "bytes_reused_from_earlier_downloads": sum(meta["source_bytes"] for meta in videos.values()
                                                                if meta["download"]["reused"]),
                     "total_source_bytes": sum(meta["source_bytes"] for meta in videos.values()),
                     "wall_seconds_including_processing": download_seconds, "concurrent_downloads": workers,
                     "download_directory": str(download_dir)},
        "tiling": {"window_s": float(WINDOW), "stride_s": float(STRIDE), "frames_per_window": FRAMES,
                   "frame_sampling": "indices round(i * (N - 1) / 7) over frames with timestamp in [start, start + 3.0)",
                   "duration_basis": "decoded re-encode duration (frames / fps), checked against the official "
                                     f"duration within {DURATION_TOLERANCE_S} s",
                   "target_size": list(TARGET_SIZE), "time_basis": "recording_relative_seconds"},
        "labels": {"target_rule": "narration with at least 50% of its duration inside the window, clipped to it",
                   "summary_rule": "'; '.join of target narrations in chronological order, else 'no annotated action'",
                   "evaluated_event_classes": EVALUATED_CLASSES,
                   "covered_intervals_rule": "window minus clipped intervals of evaluated-class narrations that overlap "
                                             "the window without being selected as targets",
                   "quiet_window_rule": "no narration of any verb overlaps the window",
                   "evidence_citation_rule": "pre = latest sampled frame at or before started_at; post = latest "
                                             "sampled frame in (started_at, ended_at]; omitted for the whole row "
                                             "when any event lacks either",
                   "narrations_total": narrations_total,
                   "narrations_selected_as_target_at_least_once": len(narration_ids_selected),
                   "evaluated_class_narrations_longer_than_6s_never_targets": long_evaluated},
        "mask_policy": {
            "windows_with_target_events": "supervision_mask.events is a per-field dict: kind, object_label, "
                                          "started_at, ended_at and description true; pre_evidence_ids and "
                                          "post_evidence_ids true only when annotation_coverage."
                                          "evidence_citations_derivable is true",
            "windows_without_target_events": "supervision_mask.events is boolean true with an empty events list, "
                                             "so the trainer supervises the empty list as abstention (a per-field "
                                             "dict on an empty collection is rejected by physical_training_target)",
            "summary": "always supervised"},
        "counts": {split_name: {
            "videos": len(split["assigned"][split_name]),
            "windows": validation["row_counts"][split_name],
            "quiet_windows": validation["quiet_windows"][split_name],
            "evidence_citation_rows": validation["evidence_citation_rows"][split_name],
            "rows_without_covered_intervals": validation["rows_without_covered_intervals"][split_name],
            "non_quiet_windows_without_targets": validation["non_quiet_windows_without_targets"][split_name],
            "target_events": sum(validation["target_events_per_class"][split_name].values()),
            "target_events_per_class": validation["target_events_per_class"][split_name],
            "decoded_seconds": round(sum(meta["decoded_duration_s"] for meta in videos.values()
                                         if meta["split"] == split_name), 6),
            "tiled_seconds": round(sum(meta["tiled_seconds"] for meta in videos.values()
                                       if meta["split"] == split_name), 6),
            "narrations": sum(meta["narrations"] for meta in videos.values() if meta["split"] == split_name),
        } for split_name in FROZEN_SPLIT},
        "files": {f"{name}.jsonl": sha256_file(output / f"{name}.jsonl") for name in FROZEN_SPLIT},
        "output_bytes_before_manifest": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
        "coverage": {"exhaustive_narration_of_all_actions": True, "false_events_per_hour_supported": True,
                     "negative_intervals_available": True, "supports_free_stream_temporal_localization": True,
                     "intervals_are_pretrimmed": False, "objects_exhaustive": False, "boxes_annotated": False,
                     "identity_annotated": False, "verified_home_ids": False,
                     "unlabeled_absence_is_negative": "only inside covered_intervals for evaluated_event_classes"},
        "held_out_home_identity": HOME_IDENTITY_STATEMENT,
        "no_model_predictions_used": True, "split_changed_after_freeze": False,
        "no_home_assistant_training_fields": True, "preserved_existing_data": True,
        "total_seconds": round(time.monotonic() - started, 3),
    }
    write_json(output / "manifest.json", manifest)
    log(f"Done in {manifest['total_seconds']:.0f} s")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, required=True, help="Scratch directory for source MP4s")
    parser.add_argument("--annotations", type=Path, required=True,
                        help="Directory containing the official EPIC-KITCHENS-100 annotation CSVs")
    parser.add_argument("--max-download-gb", type=float, default=6.0)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 0 < args.max_download_gb <= 64 or not 1 <= args.workers <= 3:
        parser.error("max-download-gb must be in (0, 64] and workers in 1..3")
    manifest = prepare(args.output.resolve(), args.download_dir.resolve(), args.annotations.resolve(),
                       int(args.max_download_gb * 1_000_000_000), args.workers)
    print(json.dumps({"counts": manifest["counts"], "substitutions": manifest["substitutions"],
                      "download": manifest["download"]}, indent=2))


if __name__ == "__main__":
    main()
