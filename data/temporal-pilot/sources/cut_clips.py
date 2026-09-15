"""Cut short action clips from full EPIC-KITCHENS videos using annotations."""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ANNOTATIONS_DIR = Path("epic-kitchens-100-annotations")
DATA_DIR = Path("EPIC-KITCHENS-100")
CLIPS_DIR = Path("clips")


def load_annotations(video_ids: list[str] | None) -> pd.DataFrame:
    train = pd.read_csv(ANNOTATIONS_DIR / "EPIC_100_train.csv")
    val = pd.read_csv(ANNOTATIONS_DIR / "EPIC_100_validation.csv")
    annotations = pd.concat([train, val], ignore_index=True)

    if not video_ids:
        return annotations

    filtered = annotations[annotations["video_id"].isin(video_ids)]

    found_ids = set(filtered["video_id"].unique())
    for vid in video_ids:
        if vid not in found_ids:
            print(f"WARNING: No annotations found for video {vid}")

    return filtered


def video_path(participant_id: str, video_id: str) -> Path:
    return DATA_DIR / participant_id / "videos" / f"{video_id}.MP4"


def build_ffmpeg_cmd(row: pd.Series) -> list[str]:
    input_path = video_path(row["participant_id"], row["video_id"])
    output_path = CLIPS_DIR / row["participant_id"] / f"{row['narration_id']}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        "ffmpeg",
        "-ss", str(row["start_timestamp"]),
        "-to", str(row["stop_timestamp"]),
        "-i", str(input_path),
        "-vf", "scale=854:480",
        "-c:v", "libx264", "-crf", "23", "-preset", "medium",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        "-y", str(output_path),
    ]


def main():
    parser = argparse.ArgumentParser(description="Cut action clips from EPIC-KITCHENS videos")
    parser.add_argument("video_ids", nargs="*", help="Video IDs to process (e.g. P04_26 P03_11). If omitted, process all.")
    parser.add_argument("--dry-run", action="store_true", help="Print ffmpeg commands without executing")
    parser.add_argument("--force", action="store_true", help="Reprocess clips even if they already exist")
    args = parser.parse_args()

    annotations = load_annotations(args.video_ids or None)
    if annotations.empty:
        print("No annotations found. Exiting.")
        sys.exit(1)

    CLIPS_DIR.mkdir(exist_ok=True)

    created = 0
    skipped = 0
    failed = 0

    rows = list(annotations.iterrows())
    progress = tqdm(rows, unit="clip")
    for _, row in progress:
        narration_id = row["narration_id"]
        progress.set_description(narration_id)

        input_file = video_path(row["participant_id"], row["video_id"])
        if not input_file.exists():
            tqdm.write(f"WARNING: Source video not found: {input_file}, skipping {narration_id}")
            failed += 1
            continue

        cmd = build_ffmpeg_cmd(row)

        if args.dry_run:
            tqdm.write(" ".join(cmd))
            continue

        output_path = CLIPS_DIR / f"{narration_id}.mp4"
        if output_path.exists() and not args.force:
            skipped += 1
            continue

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            tqdm.write(f"ERROR: ffmpeg failed for {narration_id}: {result.stderr.splitlines()[-1] if result.stderr else 'unknown error'}")
            failed += 1
        else:
            created += 1

    if not args.dry_run:
        print(f"\nDone: {created} created, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()

