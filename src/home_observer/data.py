"""Public media preparation and explicitly authored synthetic plumbing fixtures.

Public captions/QA are stored separately: absence of a public action label is
never converted into an assistant no-op. No network request occurs at import.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import wave
import zlib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .replay import assert_disjoint_splits, validate_window

CAMERAS = ("kitchen", "entry", "lounge", "garage")
FIXTURE_EPOCH = 1_750_000_000.0


def opaque_id(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()[:24]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def group_split(group_id: str, seed: str = "home-observer-v1") -> str:
    bucket = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def authored_decision(window: dict[str, Any]) -> dict[str, Any]:
    """Transparent fixture oracle, not a learned perception or reasoning model."""
    states = window["device_states"]
    observations, actions = [], []
    lux = float(states.get("sensor.ambient_lux", 100))
    for room in CAMERAS:
        motion, light = f"binary_sensor.motion_{room}", f"light.{room}"
        if motion not in states:
            continue
        evidence = [f"device:{motion}"]
        observations.append({"entity_id": motion, "attribute": "state", "value": states[motion],
                             "confidence": 1.0, "evidence_ids": evidence})
        if states[motion] == "on" and lux < 40 and states.get(light) == "off":
            actions.append({"domain": "light", "service": "turn_on", "entity_id": light, "data": {},
                            "reason": "Motion is on, ambient light is below 40 lux, and this light is off.",
                            "evidence_ids": evidence + ["device:sensor.ambient_lux", f"device:{light}"]})
    if states.get("binary_sensor.water_leak") == "on" and states.get("switch.buzzer") == "off":
        actions.append({"domain": "switch", "service": "turn_on", "entity_id": "switch.buzzer",
                        "data": {},
                        "reason": "An explicit leak sensor is active.",
                        "evidence_ids": ["device:binary_sensor.water_leak", "device:switch.buzzer"]})
    return {"summary": "Explicit fixture policy requires an action." if actions else "Explicit fixture policy requires no action.",
            "observations": observations if actions else [], "actions": actions, "noop": not actions}


def fixture_policy() -> dict[str, Any]:
    return {"entities": [*[f"light.{room}" for room in CAMERAS],
                *[f"room.{room}" for room in CAMERAS], *[f"binary_sensor.motion_{room}" for room in CAMERAS],
                "sensor.ambient_lux", "binary_sensor.water_leak", "switch.buzzer"],
            "allowed_services": ["light.turn_on", "switch.turn_on"],
            "rules": ["For each camera, room.<camera_id> occupied is true exactly when a person is visible; report uncertainty for ambiguous footage.",
                      "Turn on light.<room> when occupancy is supported by its camera frame or an explicit motion sensor, ambient_lux < 40, and that light is off.",
                      "Turn on switch.buzzer only when binary_sensor.water_leak is on and switch.buzzer is off.",
                      "Generated tones are fixture sounds; never interpret them as a real fire alarm.",
                      "Do not turn off lights or take any other action. Return noop=true with empty observations/actions when nothing should be recorded or executed."],
            "min_confidence": 0.8, "cooldown_seconds": 30, "state_ttl_seconds": 300,
            "max_actions_per_window": 2, "max_window_seconds": 60}


def visual_fixture_decision(window: dict[str, Any], occupied_room: str | None) -> dict[str, Any]:
    """Authored geometry labels; occupied_room is NEVER included in model inputs."""
    observations, actions = [], []
    for frame in window["frames"]:
        room = frame["camera_id"]
        occupied = room == occupied_room
        evidence = [frame["evidence_id"]]
        observations.append({"entity_id": f"room.{room}", "attribute": "occupied", "value": occupied,
                             "confidence": 1.0, "evidence_ids": evidence})
        light = f"light.{room}"
        if occupied and window["device_states"]["sensor.ambient_lux"] < 40 and window["device_states"][light] == "off":
            actions.append({"domain": "light", "service": "turn_on", "entity_id": light, "data": {},
                            "reason": "The camera shows an occupant; ambient light is below 40 lux and this light is off.",
                            "evidence_ids": evidence + ["device:sensor.ambient_lux", f"device:{light}"]})
    return {"summary": "Recorded occupancy from the current camera views.",
            "observations": observations, "actions": actions, "noop": False}


def _png(path: Path, room: int, occupied: bool, lit: bool, variant: int) -> None:
    """Small procedural scene. No answer, action, class text, or metadata is drawn."""
    width, height = 160, 120
    palette = hashlib.sha256(f"scene-{variant}-{room}".encode()).digest()
    tint = tuple(int(channel % 29) - 14 for channel in palette[:3])
    base = (178, 172, 153) if lit else (47, 52, 63)
    wall = tuple(channel + shift for channel, shift in zip(base, tint))
    pixels = bytearray(bytes(wall) * width * height)

    def rect(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        for y in range(max(0, y0), min(height, y1)):
            for x in range(max(0, x0), min(width, x1)):
                p = (y * width + x) * 3
                pixels[p:p + 3] = bytes(color)

    rect(0, 84, 160, 120, (84, 79, 70) if lit else (31, 33, 38))
    rect(108, 23, 145, 85, (101, 80, 57))
    rect(15, 58 + room * 2, 63 + room * 3, 89, (65, 85, 98))
    rect(25, 89, 30, 108, (48, 48, 50))
    rect(53, 89, 58, 108, (48, 48, 50))
    rect(15, 17, 55, 42, (107, 147, 169) if lit else (21, 32, 62))
    if occupied:
        x = 77 + variant % 13
        rect(x, 38, x + 12, 51, (180, 132, 98))
        rect(x - 5, 51, x + 17, 81, (70 + variant % 80, 114, 161))
        rect(x - 5, 81, x + 2, 104, (37, 42, 50))
        rect(x + 10, 81, x + 17, 104, (37, 42, 50))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)

    raw = b"".join(b"\0" + pixels[y * width * 3:(y + 1) * width * 3] for y in range(height))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _wav(path: Path, sound: bool, duration: float = 2, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = bytearray()
    for i in range(int(sample_rate * duration)):
        amplitude = 0.08 * math.sin(2 * math.pi * 440 * i / sample_rate) if sound and i < sample_rate / 3 else 0
        samples.extend(struct.pack("<h", int(amplitude * 32767)))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples)


def generate_fixtures(output_dir: str | Path, families_per_split: int = 2, variants: int = 2) -> dict[str, Any]:
    """Write 4-camera observations with authored action AND explicit negative labels.

    Entire scenario families, including all their variants/windows, stay in one
    split. The frames and non-speech audio are procedural illustrations, not
    evidence of real camera performance. Device readings drive the oracle.
    """
    if families_per_split < 1 or variants < 1:
        raise ValueError("families_per_split and variants must be positive")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for split_index, split in enumerate(("train", "validation", "test")):
        for family in range(families_per_split):
            group = f"synthetic:{split}-household-routine-{family}"
            for variant in range(variants):
                recording = f"{split}-{family}-{variant}"
                # Opaque randomized source clock; the absolute phase carries no
                # fixed event-class digit. Per-recording order remains causal.
                epoch = FIXTURE_EPOCH + int(opaque_id(f"clock:{recording}"), 16) % 10_000_000 / 10
                for step in range(4):
                    window_id = opaque_id(f"fixture-window:{recording}:{step}")
                    start, end = epoch + step * 10, epoch + (step + 1) * 10
                    active_room = CAMERAS[(family + variant + split_index) % len(CAMERAS)]
                    # Explicit negatives: empty room; occupied but bright; already-on light.
                    occupied, lux = step > 0, 100 if step == 2 else 12
                    states: dict[str, Any] = {"sensor.ambient_lux": lux, "binary_sensor.water_leak": "off"}
                    frames = []
                    for room_index, room in enumerate(CAMERAS):
                        motion = occupied and room == active_room
                        states[f"binary_sensor.motion_{room}"] = "on" if motion else "off"
                        states[f"light.{room}"] = "on" if step == 3 and room == active_room else "off"
                        relative = f"assets/{recording}/{step}-{room}.png"
                        _png(root / relative, room_index, motion, lux >= 40 or states[f"light.{room}"] == "on",
                             split_index * 10000 + family * 100 + variant)
                        frames.append({"camera_id": room, "timestamp": end - 1,
                                       "path": relative, "evidence_id": f"{window_id}:frame:{room}"})
                    audio_path = f"assets/{recording}/{step}-home.wav"
                    _wav(root / audio_path, sound=step == 1)
                    window = {"window_id": window_id, "started_at": start, "ended_at": end,
                              "frames": frames, "audio": [{"microphone_id": "home", "started_at": end - 2,
                              "ended_at": end, "path": audio_path, "evidence_id": f"{window_id}:audio:home"}],
                              "device_states": states}
                    validate_window(window)
                    rows.append({"id": window_id, "split": split, "source": "synthetic_authored_v1",
                                 "group_id": group, "recording_id": recording,
                                 "annotation_method": "synthetic_authored_rules_v1",
                                 "provenance": {"kind": "synthetic", "generator": "home_observer.data.generate_fixtures",
                                                "rule_version": "motion-lux-light-v1", "audio": "generated tone/silence; no speech"},
                                 "window": window, "target": authored_decision(window)})
    counts = assert_disjoint_splits(rows)
    # Visual counterpart has identical scene geometry but no occupancy/motion
    # readings. The original geometry-derived labels never enter the window.
    visual_rows = []
    for original in rows:
        row = copy.deepcopy(original)
        window = row["window"]
        occupied_room = next((room for room in CAMERAS if window["device_states"][f"binary_sensor.motion_{room}"] == "on"), None)
        window["device_states"] = {key: value for key, value in window["device_states"].items() if not key.startswith("binary_sensor.motion_")}
        window["window_id"] = opaque_id(f"visual:{window['window_id']}")
        row["id"] = window["window_id"]
        row["recording_id"] = f"visual-{row['recording_id']}"
        row["task_type"] = "visual_occupancy"
        row["annotation_method"] = "synthetic_geometry_and_authored_rules_v1"
        row["provenance"]["rule_version"] = "visual-occupancy-lux-light-v1"
        row["target"] = visual_fixture_decision(window, occupied_room)
        visual_rows.append(row)
        original["task_type"] = "device_control"
    rows.extend(visual_rows)
    counts = assert_disjoint_splits(rows)
    for split in counts:
        write_jsonl(root / f"{split}.jsonl", [row for row in rows if row["split"] == split])
        for task in ("device_control", "visual_occupancy"):
            write_jsonl(root / f"{split}-{task}.jsonl", [row for row in rows if row["split"] == split and row["task_type"] == task])
    write_jsonl(root / "all.jsonl", rows)
    manifest = {"kind": "synthetic", "counts": counts, "total_rows": len(rows),
                "split_unit": "scenario_family", "cameras": list(CAMERAS),
                "task_types": ["device_control", "visual_occupancy"],
                "limits": ["No human speech", "No real household footage", "Visual targets derived from authored geometry", "Does not establish real-world visual competence"]}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "policy.json").write_text(json.dumps(fixture_policy(), indent=2) + "\n")
    return manifest


def generate_counterfactual_pairs(output_dir: str | Path, pairs_per_room: int = 2) -> dict[str, Any]:
    """Held-out image-dependent pairs, not a chronological stream.

    Both members of each pair have byte-identical model-visible text and silent
    audio, including IDs/times/device readings. Only their image pixels differ.
    Call inference independently per row, with fresh/empty state and journal.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for room_index, room in enumerate(CAMERAS):
        for variant in range(pairs_per_room):
            pair = opaque_id(f"counterfactual-pair:{room}:{variant}")
            window_id = opaque_id(f"counterfactual-window:{pair}")
            epoch = FIXTURE_EPOCH + int(pair, 16) % 10_000_000 / 10
            common_audio = f"assets/{pair}/common.wav"
            _wav(root / common_audio, sound=False)
            for occupied in (False, True):
                frames = []
                for camera_index, camera in enumerate(CAMERAS):
                    relative = f"assets/{pair}/{int(occupied)}-{camera}.png"
                    _png(root / relative, camera_index, occupied and camera == room, False,
                         900000 + room_index * 100 + variant)
                    frames.append({"camera_id": camera, "timestamp": epoch + 9,
                                   "path": relative, "evidence_id": opaque_id(f"{pair}:frame:{camera}")})
                window = {"window_id": window_id, "started_at": epoch, "ended_at": epoch + 10,
                          "frames": frames, "audio": [{"microphone_id": "home", "started_at": epoch + 8,
                          "ended_at": epoch + 10, "path": common_audio, "evidence_id": opaque_id(f"{pair}:audio")}],
                          "device_states": {"sensor.ambient_lux": 12, **{f"light.{camera}": "off" for camera in CAMERAS}}}
                validate_window(window)
                rows.append({"id": f"{pair}-{int(occupied)}", "split": "test", "source": "synthetic_counterfactual_v1",
                             "group_id": f"counterfactual:{pair}", "recording_id": f"counterfactual:{pair}:{int(occupied)}",
                             "pair_id": pair, "counterfactual_member": "occupied" if occupied else "empty",
                             "task_type": "visual_counterfactual", "annotation_method": "synthetic_geometry_and_authored_rules_v1",
                             "provenance": {"kind": "synthetic", "evaluation_protocol": "independent inference, empty state/journal for each row",
                                            "input_text": "identical within pair; only images differ", "training_eligible": False},
                             "window": window, "target": visual_fixture_decision(window, room if occupied else None)})
    write_jsonl(root / "pairs.jsonl", rows)
    (root / "policy.json").write_text(json.dumps(fixture_policy(), indent=2) + "\n")
    manifest = {"kind": "synthetic_counterfactual", "pairs": len(rows) // 2, "rows": len(rows),
                "protocol": "Independent inference with empty state/journal per row; do not call chronological iter_replay on pairs.jsonl",
                "training_eligible": False}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def ffmpeg_executable() -> str:
    configured = os.environ.get("FFMPEG_BINARY")
    if configured:
        return configured
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("Install imageio-ffmpeg or ffmpeg to extract public video/audio") from exc


def _ffmpeg(args: list[str]) -> None:
    result = subprocess.run([ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args],
                            capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-2000:]}")


def media_info(path: str | Path) -> dict[str, Any]:
    """Probe with the same portable ffmpeg binary used for extraction."""
    result = subprocess.run([ffmpeg_executable(), "-hide_banner", "-i", str(path)],
                            capture_output=True, text=True, timeout=30)
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        raise ValueError(f"Could not read media duration: {result.stderr[-1000:]}")
    hours, minutes, seconds = map(float, match.groups())
    return {"duration_s": hours * 3600 + minutes * 60 + seconds, "has_audio": "Audio:" in result.stderr}


def prepare_recording(video_path: str | Path, output_dir: str | Path, *, recording_id: str,
                      camera_id: str = "ego", start_time: float = FIXTURE_EPOCH,
                      duration_s: float = 30, window_s: float = 10, frame_step_s: float = 5,
                      source: str = "user_supplied", provenance: dict[str, Any] | None = None,
                      include_audio: bool = True) -> list[dict[str, Any]]:
    """Extract bounded windows from ONE source recording; no training target is fabricated.

    start_time is the wall-clock origin of the source clip. A synthetic origin
    is explicit in provenance when the source does not supply UTC. Supply an
    accurate duration; extraction stops at the requested bound and fails on EOF.
    """
    if not 0 < window_s <= 120 or duration_s <= 0 or frame_step_s <= 0:
        raise ValueError("invalid duration, window size or frame stride")
    if any(x in recording_id for x in ("/", "\\", "..")):
        raise ValueError("recording_id must be a simple identifier")
    video_path, root = Path(video_path).resolve(), Path(output_dir)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    offset = 0.0
    while offset < duration_s - 1e-8:
        end = min(duration_s, offset + window_s)
        window_id = f"{recording_id}-{len(rows):05d}"
        assets = root / "assets" / window_id
        assets.mkdir(parents=True, exist_ok=True)
        frames = []
        # Sample interior times so a final frame cannot equal the next window.
        frame_offset = offset + min(0.1, (end - offset) / 2)
        while frame_offset < end - 1e-8:
            frame_file = assets / f"{len(frames):03d}.jpg"
            _ffmpeg(["-ss", str(frame_offset), "-i", str(video_path), "-t", str(end - frame_offset),
                     "-frames:v", "1", "-vf", "scale=384:-2", str(frame_file)])
            if not frame_file.is_file():
                raise ValueError(f"requested frame at {frame_offset}s is beyond source video")
            frames.append({"camera_id": camera_id, "timestamp": start_time + frame_offset,
                           "path": frame_file.relative_to(root).as_posix(), "evidence_id": f"{window_id}:frame:{len(frames)}"})
            frame_offset += frame_step_s
        audio = []
        if include_audio:
            audio_file = assets / "audio.wav"
            _ffmpeg(["-ss", str(offset), "-i", str(video_path), "-t", str(end - offset),
                     "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio_file)])
            with wave.open(str(audio_file), "rb") as sound:
                actual_duration = sound.getnframes() / sound.getframerate()
            if actual_duration <= 0:
                raise ValueError("source audio is empty")
            audio.append({"microphone_id": camera_id, "started_at": start_time + offset,
                          "ended_at": start_time + offset + min(actual_duration, end - offset),
                          "path": audio_file.relative_to(root).as_posix(), "evidence_id": f"{window_id}:audio"})
        window = {"window_id": window_id, "started_at": start_time + offset, "ended_at": start_time + end,
                  "frames": frames, "audio": audio, "device_states": {}}
        validate_window(window)
        group = (provenance or {}).get("group_id", f"{source}:{recording_id}")
        rows.append({"id": window_id, "source": source, "recording_id": recording_id, "group_id": group,
                     "split": group_split(group), "annotation_method": "unlabeled_public_replay",
                     "provenance": provenance or {"kind": "user_supplied", "timestamp_origin": "caller_supplied"}, "window": window})
        offset = end
    write_jsonl(root / "replay.jsonl", rows)
    return rows


def download_public_sample(output_dir: str | Path, *, max_bytes: int = 64 * 1024 * 1024,
                           revision: str = "main", prepare: bool = True) -> dict[str, Any]:
    """Fetch one size-bounded EgoLife MP4 using the official Hugging Face Hub SDK.

    Pins the resolved revision and records hash/license. Does not crawl YouTube,
    clone a full dataset, or treat first-person footage as four static cameras.
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface-hub for the public sample downloader") from exc
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    repo_id = "lmms-lab/EgoLife"
    api = HfApi()
    info = api.dataset_info(repo_id, revision=revision)
    pinned = info.sha
    candidates = []
    # One participant/day only: server-side prefix prevents listing 512GB of files.
    for item in api.list_repo_tree(repo_id, path_in_repo="A1_JAKE/DAY1", repo_type="dataset", revision=pinned):
        if getattr(item, "path", "").lower().endswith(".mp4") and 0 < getattr(item, "size", 0) <= max_bytes:
            candidates.append(item)
        if len(candidates) >= 10:
            break
    if not candidates:
        raise RuntimeError(f"No EgoLife A1_JAKE/DAY1 MP4 fits the {max_bytes}-byte limit at {pinned}")
    candidate = sorted(candidates, key=lambda x: x.path)[0]
    local = hf_hub_download(repo_id=repo_id, filename=candidate.path, repo_type="dataset", revision=pinned,
                            local_dir=root / "raw")
    video = Path(local)
    if video.stat().st_size > max_bytes:
        raise RuntimeError("download exceeded declared byte cap")
    card = info.card_data.to_dict() if info.card_data is not None else {}
    manifest: dict[str, Any] = {"kind": "public_real", "repo_id": repo_id, "revision": pinned,
        "source_url": f"https://huggingface.co/datasets/{repo_id}/blob/{pinned}/{candidate.path}",
        "path_in_repo": candidate.path, "local_path": str(video.resolve()), "bytes": video.stat().st_size,
        "sha256": sha256_file(video), "license": card.get("license", "see dataset card"),
        "timestamp_origin": "synthetic UTC origin; offsets within downloaded source clip are real",
        "view": "one egocentric camera", "group_id": "EgoLife:Beijing:shared-household-week",
        "annotation_method": "unlabeled_public_replay", "training_eligible": False,
        "limits": ["No Home Assistant action labels", "No silence labels", "One clip is only a media/replay smoke test",
                   "All cameras/participants/days from this shared recording remain in one split"]}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if prepare:
        properties = media_info(video)
        manifest["media_info"] = properties
        rows = prepare_recording(video, root, recording_id="egolife-a1-day1-sample",
                                 duration_s=min(20, properties["duration_s"]), include_audio=properties["has_audio"],
                                 source="EgoLife", provenance=manifest)
        manifest["prepared_windows"] = len(rows)
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


HOUSEHOLD_CAPTION_PATTERN = re.compile(
    r"\b(kitchen|bathroom|bedroom|living room|dining room|couch|sofa|refrigerator|toilet)\b", re.I)


def summary_supervised_row(*, image_id: int, image_path: str, caption: str, split: str,
                           provenance: dict[str, Any]) -> dict[str, Any]:
    """Partial supervision only; absent Decision fields are NOT negative labels."""
    window_id = f"coco-{image_id}"
    return {"id": window_id, "source": "COCO_human_captions", "split": split,
            "group_id": window_id, "recording_id": window_id, "task_type": "natural_image_summary",
            "annotation_method": "existing_human_COCO_caption",
            "supervision_mask": {"summary": True, "actions": False, "observations": False, "noop": False},
            "provenance": provenance,
            "window": {"window_id": window_id, "started_at": FIXTURE_EPOCH, "ended_at": FIXTURE_EPOCH + 1,
                       "frames": [{"camera_id": "household", "timestamp": FIXTURE_EPOCH + 0.5,
                                   "path": image_path, "evidence_id": f"{window_id}:frame"}],
                       "audio": [], "device_states": {}},
            "target": {"summary": caption.strip()}}


def _coco_download_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname != "images.cocodataset.org" or not parsed.path.startswith(("/train2014/", "/val2014/")):
        raise ValueError("Only official COCO train2014/val2014 image paths are accepted")
    # The dataset host's HTTPS certificate currently does not cover its hostname.
    # Use the same public S3 bucket through Amazon's valid path-style TLS endpoint.
    return "https://s3.amazonaws.com/images.cocodataset.org" + parsed.path


def prepare_coco_household(output_dir: str | Path, *, train_count: int = 64, validation_count: int = 16,
                           test_count: int = 16, max_bytes: int = 250_000_000,
                           revision: str = "96f5cf8784404ffd62ff541c5aeaea7b7a4d550d",
                           raw_dir: str | Path | None = None) -> dict[str, Any]:
    """Download small household-filtered COCO sample with existing human captions.

    Metadata is a pinned Hub packaging of original COCO image IDs/annotations.
    Images come from the official public COCO bucket; all variants/captions of
    an image stay in the same split. Only supported summary tokens are labeled.
    """
    from io import BytesIO

    import pyarrow.parquet as parquet
    from huggingface_hub import HfApi, hf_hub_download
    from PIL import Image

    counts = {"train": train_count, "validation": validation_count, "test": test_count}
    if any(count <= 0 for count in counts.values()) or sum(counts.values()) > 256:
        raise ValueError("request 1–256 total examples with nonempty splits")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "assets").mkdir(exist_ok=True)
    raw = Path(raw_dir) if raw_dir is not None else root / "raw"
    repo = "mlgym/coco-captioning"
    api = HfApi()
    pinned = api.dataset_info(repo, revision=revision).sha
    source_files = {"train": "train_images.parquet", "validation": "val_images.parquet", "test": "test_images.parquet"}
    sizes = {item.path: item.size for item in api.list_repo_tree(repo, repo_type="dataset", revision=pinned)
             if hasattr(item, "size")}
    downloaded = 0
    rows: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_hashes: set[str] = set()
    failures = []
    metadata = []
    for split, filename in source_files.items():
        size = sizes[filename]
        if downloaded + size > max_bytes:
            raise ValueError("metadata exceeds total download budget")
        local = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=pinned, local_dir=raw))
        downloaded += local.stat().st_size
        metadata.append({"filename": filename, "bytes": local.stat().st_size, "sha256": sha256_file(local)})
        candidates = sorted(parquet.read_table(local).to_pylist(), key=lambda row: row["id"])
        count = 0
        for item in candidates:
            image_id = int(item["id"])
            captions = [ann for ann in item["anns"] if HOUSEHOLD_CAPTION_PATTERN.search(ann["caption"])]
            if not captions or image_id in seen_ids:
                continue
            if count >= counts[split]:
                break
            annotation = captions[0]
            download_url = _coco_download_url(item["coco_url"])
            image_limit = min(3_000_000, max_bytes - downloaded)
            if image_limit <= 0:
                raise ValueError("total download budget exhausted")
            try:
                with urlopen(Request(download_url, headers={"User-Agent": "home-observer-research/0.1"}), timeout=20) as response:
                    image_bytes = response.read(image_limit + 1)
                downloaded += len(image_bytes)
                if len(image_bytes) > image_limit:
                    raise ValueError("image exceeds download size limit")
                digest = hashlib.sha256(image_bytes).hexdigest()
                if digest in seen_hashes:
                    continue
                path = root / "assets" / f"coco-{image_id}.jpg"
                with Image.open(BytesIO(image_bytes)) as image:
                    image = image.convert("RGB")
                    image.thumbnail((512, 512))
                    image.save(path, quality=90)
            except Exception as exc:
                failures.append({"image_id": image_id, "error": f"{type(exc).__name__}: {exc}"})
                if len(failures) >= 20:
                    raise RuntimeError(f"20 COCO image failures; latest: {failures[-1]}") from exc
                continue
            provenance = {"kind": "public_real", "dataset": "MS-COCO", "metadata_repo": repo,
                "metadata_revision": pinned, "metadata_file": filename, "image_id": image_id,
                "annotation_id": int(annotation["id"]), "annotation_image_id": int(annotation["image_id"]),
                "original_coco_url": item["coco_url"], "download_url": download_url,
                "image_license_id": int(item["license"]), "image_original_sha256": digest,
                "prepared_sha256": sha256_file(path), "source_captions": item["anns"],
                "timestamp_origin": "synthetic one-second wrapper around a still image; no temporal claim",
                "source_paper": "https://arxiv.org/abs/1504.00325",
                "supervision": "Existing human caption; no action, no-op or entity-state supervision"}
            if int(annotation["image_id"]) != image_id:
                raise ValueError("caption/image ID mismatch")
            rows.append(summary_supervised_row(image_id=image_id, image_path=path.relative_to(root).as_posix(),
                         caption=annotation["caption"], split=split, provenance=provenance))
            seen_ids.add(image_id)
            seen_hashes.add(digest)
            count += 1
        if count < counts[split]:
            raise ValueError(f"only prepared {count}/{counts[split]} {split} household images")
        write_jsonl(root / f"{split}.jsonl", [row for row in rows if row["split"] == split])
    assert_disjoint_splits(rows)
    write_jsonl(root / "all.jsonl", rows)
    # Inference-only replay excludes even the partial caption targets.
    write_jsonl(root / "heldout-replay.jsonl", [{key: value for key, value in row.items() if key != "target"}
                                                for row in rows if row["split"] == "test"])
    policy = {"entities": [], "allowed_services": [],
              "rules": ["Describe only what is supported by the input image. No devices are configured."]}
    (root / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    manifest = {"kind": "public_real", "dataset": "COCO_household_human_caption_subset", "counts": counts,
                "metadata_repo": repo, "metadata_revision": pinned, "metadata_files": metadata,
                "downloaded_bytes": downloaded, "download_budget_bytes": max_bytes,
                "split_unit": "COCO image ID and image-byte SHA256", "filtered_by": HOUSEHOLD_CAPTION_PATTERN.pattern,
                "supervision_mask": {"summary": True, "actions": False, "observations": False, "noop": False},
                "failures": failures,
                "limitations": ["Still images, not temporal video", "Household rooms/objects, not necessarily private-home camera views",
                                "Small human-caption subset; labels may contain normal annotation noise", "No action/no-op competence measured"]}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fixtures = subparsers.add_parser("fixtures", help="Generate explicitly labeled synthetic plumbing examples")
    fixtures.add_argument("output_dir")
    fixtures.add_argument("--families-per-split", type=int, default=2)
    fixtures.add_argument("--variants", type=int, default=2)
    public = subparsers.add_parser("public-sample", help="Download one bounded, real EgoLife clip (no policy labels)")
    public.add_argument("output_dir")
    public.add_argument("--max-mib", type=int, default=64)
    public.add_argument("--revision", default="main")
    public.add_argument("--no-prepare", action="store_true")
    coco = subparsers.add_parser("coco-household", help="Small real-image subset with human caption supervision only")
    coco.add_argument("output_dir")
    coco.add_argument("--train-count", type=int, default=64)
    coco.add_argument("--validation-count", type=int, default=16)
    coco.add_argument("--test-count", type=int, default=16)
    coco.add_argument("--raw-dir")
    paired = subparsers.add_parser("counterfactuals", help="Image-dependent held-out pairs with identical input text")
    paired.add_argument("output_dir")
    paired.add_argument("--pairs-per-room", type=int, default=2)
    args = parser.parse_args()
    if args.command == "fixtures":
        result = generate_fixtures(args.output_dir, args.families_per_split, args.variants)
    elif args.command == "public-sample":
        result = download_public_sample(args.output_dir, max_bytes=args.max_mib * 1024 * 1024,
                                        revision=args.revision, prepare=not args.no_prepare)
    elif args.command == "coco-household":
        result = prepare_coco_household(args.output_dir, train_count=args.train_count,
                                       validation_count=args.validation_count, test_count=args.test_count,
                                       raw_dir=args.raw_dir)
    else:
        result = generate_counterfactual_pairs(args.output_dir, pairs_per_room=args.pairs_per_room)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
