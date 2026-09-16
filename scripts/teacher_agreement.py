#!/usr/bin/env python3
"""Compare teacher passes with each other and with hidden narrations.

Reads stream-target rows (API teacher and interactive teacher), matches pushes across passes
within an onset collar, and checks each push against the narrated events of the same source
window. Narrations record every manipulation, so narration recall is reported only for the
durable verbs the contract asks for (take, put, open, close, turn-on, turn-off, pour, serve).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DURABLE_VERBS = {"take", "pick-up", "put-down", "put", "put-in", "put-on", "open", "close", "turn-on", "turn-off",
                 "pour", "insert", "remove", "throw"}
# Which observation kinds legitimately report a narrated verb (kind-aware matching).
VERB_KINDS = {
    "take": {"object_taken", "food_state"}, "pick-up": {"object_taken", "food_state"}, "remove": {"object_taken", "container_state"},
    "put-down": {"object_placed", "food_state", "container_state"}, "put": {"object_placed", "food_state", "container_state"},
    "put-in": {"object_placed", "food_state", "container_state"}, "put-on": {"object_placed", "container_state"},
    "insert": {"object_placed", "container_state"}, "throw": {"object_placed"},
    "open": {"container_state", "appliance_state"}, "close": {"container_state", "appliance_state"},
    "turn-on": {"appliance_state"}, "turn-off": {"appliance_state"}, "pour": {"food_state"},
}


def load_rows(paths):
    rows = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pushes(row):
    return [(tick["t"], tick) for tick in row["ticks"] if tick["decision"] in ("context", "act")]


def narrations_for(row, annotation_root: Path, clip_root: Path):
    """Narrated events overlapping the row's source window, in source seconds."""
    video_id, start, end = row["source"]["video_id"], row["source"]["start_s"], row["source"]["end_s"]
    events = []
    annotation = annotation_root / f"{video_id}.json"
    if annotation.exists():
        payload = json.loads(annotation.read_text())
        if isinstance(payload, list):
            items = payload
        else:
            items = next((v for v in payload.values() if isinstance(v, list) and v and isinstance(v[0], dict)
                          and "start_timestamp" in v[0]), [])

        def sec(value):
            h, m, s = str(value).split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
        for item in items:
            a, b = sec(item["start_timestamp"]), sec(item["stop_timestamp"])
            if b >= start - 1.0 and a <= end + 1.0:
                events.append({"start": a, "stop": b, "narration": item["narration"], "verb": item["verb"]})
        return events
    for sidecar in clip_root.glob(f"*{video_id}.json"):
        meta = json.loads(sidecar.read_text())
        if abs(meta["source_start_s"] - start) < 0.6:
            for item in meta["overlapping_narrations"]:
                events.append({"start": item["start"], "stop": item["stop"], "narration": item["narration"], "verb": item["verb"]})
    return events


def match(a_times, b_times, collar):
    """Greedy one-to-one matching of two sorted time lists within a collar."""
    used, pairs = set(), []
    for t in a_times:
        best = None
        for index, u in enumerate(b_times):
            if index in used or abs(t - u) > collar:
                continue
            if best is None or abs(t - u) < abs(t - b_times[best]):
                best = index
        if best is not None:
            used.add(best)
            pairs.append((t, b_times[best]))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", nargs="+", required=True, help="stream-target JSONL files from the API teacher")
    parser.add_argument("--interactive", nargs="*", default=[], help="stream-target JSONL files from the interactive pass")
    parser.add_argument("--annotations", type=Path, default=Path("data/continuous-v2/sources/annotations"))
    parser.add_argument("--clips", type=Path, default=Path("data/continuous-v2-workflow"))
    parser.add_argument("--collar", type=float, default=1.5, help="seconds; a push counts as matched within this of an event onset or span")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    api_rows, inter_rows = load_rows(args.api), load_rows(args.interactive)
    report = {"collar_s": args.collar, "sources": [], "totals": {}}
    tot = {"api_ticks": 0, "api_pushes": 0, "api_pushes_in_narrated_span": 0, "durable_events": 0,
           "durable_events_with_push": 0, "inter_pushes": 0, "api_inter_matched": 0, "second_looks": 0,
           "durable_events_with_kind_matching_push": 0, "api_pushes_matching_a_durable_event_kind": 0}
    for row in api_rows:
        start, end = row["source"]["start_s"], row["source"]["end_s"]
        events = narrations_for(row, args.annotations, args.clips)
        api = pushes(row)
        in_span = 0
        for t, _ in api:
            if any(e["start"] - args.collar <= t <= e["stop"] + args.collar for e in events):
                in_span += 1
        durable = [e for e in events if e["verb"] in DURABLE_VERBS and e["stop"] >= start and e["start"] <= end]
        covered = sum(1 for e in durable if any(e["start"] - args.collar <= t <= e["stop"] + args.collar for t, _ in api))

        def kind_of(tick):
            return tick["observation"]["kind"] if tick.get("observation") else "act"
        covered_kind = sum(1 for e in durable if any(
            e["start"] - args.collar <= t <= e["stop"] + args.collar and kind_of(tick) in VERB_KINDS.get(e["verb"], set())
            for t, tick in api))
        in_span_kind = sum(1 for t, tick in api if any(
            e["start"] - args.collar <= t <= e["stop"] + args.collar and kind_of(tick) in VERB_KINDS.get(e["verb"], set())
            for e in events if e["verb"] in DURABLE_VERBS))
        inter = [r for r in inter_rows if r["source"]["video_id"] == row["source"]["video_id"]
                 and abs(r["source"]["start_s"] - start) < 0.6]
        inter_p = pushes(inter[0]) if inter else []
        pairs = match([t for t, _ in api], [t for t, _ in inter_p], args.collar) if inter else []
        entry = {"video_id": row["source"]["video_id"], "window_s": [start, end], "ticks": len(row["ticks"]),
                 "api_pushes": [(t, tick["observation"]["kind"] if tick.get("observation") else "act") for t, tick in api],
                 "api_pushes_in_narrated_span": in_span, "durable_events": [(e["start"], e["narration"]) for e in durable],
                 "durable_events_with_push": covered, "durable_events_with_kind_matching_push": covered_kind,
                 "api_pushes_matching_a_durable_event_kind": in_span_kind,
                 "interactive_pushes": [(t, tick["observation"]["kind"]) for t, tick in inter_p] if inter else None,
                 "api_interactive_matched": len(pairs), "second_looks": len(row["teacher"].get("second_looks", []))}
        report["sources"].append(entry)
        increments = {"api_ticks": len(row["ticks"]), "api_pushes": len(api), "api_pushes_in_narrated_span": in_span,
                      "durable_events": len(durable), "durable_events_with_push": covered, "inter_pushes": len(inter_p),
                      "api_inter_matched": len(pairs), "second_looks": entry["second_looks"],
                      "durable_events_with_kind_matching_push": covered_kind, "api_pushes_matching_a_durable_event_kind": in_span_kind}
        for key, value in increments.items():
            tot[key] += value
    tot["api_silence_rate"] = round(1 - tot["api_pushes"] / max(1, tot["api_ticks"]), 3)
    tot["api_push_precision_vs_narration"] = round(tot["api_pushes_in_narrated_span"] / max(1, tot["api_pushes"]), 3)
    tot["durable_recall"] = round(tot["durable_events_with_push"] / max(1, tot["durable_events"]), 3)
    tot["durable_recall_kind_aware"] = round(tot["durable_events_with_kind_matching_push"] / max(1, tot["durable_events"]), 3)
    tot["api_push_precision_kind_aware"] = round(tot["api_pushes_matching_a_durable_event_kind"] / max(1, tot["api_pushes"]), 3)
    tot["api_interactive_agreement"] = round(2 * tot["api_inter_matched"] / max(1, tot["api_pushes"] + tot["inter_pushes"]), 3)
    report["totals"] = tot
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
