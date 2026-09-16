#!/usr/bin/env python3
"""Teacher pass: a frontier model watches footage tick by tick and decides what a local
observer should do at each tick: stay silent, push one structured observation to the cloud
coordinator, or take a pre-authorized action. Writes stream-target rows for distillation.

Research tool only. It sends decoded frames to a frontier model through the local proxy, which
is authorized for research footage and never for production media. The token is read from
ANTHROPIC_AUTH_TOKEN and is never written anywhere.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

SCHEMA_VERSION = "home-observer.stream-target/1"
PROMPT_VERSION = "teacher-v1"
DECISIONS = ("silent", "context", "act", "unsure")
OBSERVATION_KINDS = (
    "person_presence", "activity_started", "activity_ended", "object_taken", "object_placed",
    "object_moved", "container_state", "appliance_state", "food_state", "hazard", "other",
)

SYSTEM_PROMPT = """You are the teacher for a small always-on home observer model. You watch a
camera stream as a sequence of frames at a fixed tick rate. At every tick the observer must
choose exactly one of:
- "silent": nothing worth telling the cloud coordinator. This is the default and should be the
  choice at most ticks. Ongoing activity that was already reported stays silent. Intermediate
  manipulation (each cut, each stir, each reach) stays silent. Camera motion stays silent.
- "context": push ONE structured observation the coordinator could not infer from the prior
  journal: an activity starting or ending, an object taken from or placed on a surface or in a
  container, a lid, door or appliance changing state, food served or moved between containers,
  a person entering, leaving or moving to another area, or a hazard. Report each event once,
  at the first tick where it is visibly established, never before it happens.
- "act": only when a rule in the action policy matches with high confidence and waiting for the
  coordinator would be wrong. Otherwise push context and let the coordinator decide.
- "unsure": you cannot tell at this frame rate or resolution; you will be shown more frames.
Be conservative: a missed minor event costs less than a false report, and a repeated report is
a false report. Use the goal to judge what is useful, but do not invent events to fit the goal.
Answer with strict JSON only, no prose, matching the schema in the user message."""


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str
    subject: str = Field(min_length=1, max_length=80)
    location: str = Field(default="", max_length=120)
    detail: str = Field(default="", max_length=200)
    confidence: float = Field(ge=0.0, le=1.0)


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: str = Field(min_length=1, max_length=80)
    args: dict = Field(default_factory=dict)
    rule: str = Field(min_length=1, max_length=120)


class TickDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    t: float
    decision: str
    observation: Observation | None = None
    action: Action | None = None
    why: str = Field(default="", max_length=200)

    def check(self):
        if self.decision not in DECISIONS:
            raise ValueError(f"unknown decision {self.decision!r}")
        if self.decision == "context" and (self.observation is None or self.observation.kind not in OBSERVATION_KINDS):
            raise ValueError(f"context decision at t={self.t} needs an observation with a known kind")
        if self.decision == "act" and self.action is None:
            raise ValueError(f"act decision at t={self.t} needs an action")
        if self.decision in ("silent", "unsure") and (self.observation or self.action):
            raise ValueError(f"{self.decision} decision at t={self.t} must not carry a payload")
        return self


class TeacherReply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticks: list[TickDecision]
    chunk_summary: str = Field(default="", max_length=300)


def tick_grid(start: float, end: float, hz: float) -> list[float]:
    count = int(round((end - start) * hz))
    return [round(start + k / hz, 3) for k in range(count)]


def decode_frames(video: Path, start: float, duration: float, fps: float, width: int) -> list[bytes]:
    """Decode frames at `fps` from `start` for `duration` seconds, JPEG-encoded at `width` px."""
    import imageio_ffmpeg
    from PIL import Image
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24", input_params=["-ss", f"{start:.3f}"],
                                        output_params=["-t", f"{duration:.3f}", "-vf", f"fps={fps}"])
    meta = next(reader)
    w, h = meta["size"]
    out = []
    for raw in reader:
        image = Image.frombytes("RGB", (w, h), raw)
        if width and w > width:
            image = image.resize((width, round(width * h / w)))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        out.append(buffer.getvalue())
    return out


def build_user_content(frames: list[bytes], times: list[float], context_frames: int, context: dict,
                       schema_hint: dict) -> list[dict]:
    content = [{"type": "text", "text": "Context for this chunk (JSON):\n" + json.dumps(context, indent=1)}]
    for index, (frame, t) in enumerate(zip(frames, times)):
        label = f"[lead-in frame, no decision] t={t:.2f}s" if index < context_frames else f"tick t={t:.2f}s"
        content.append({"type": "text", "text": label})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                     "data": base64.b64encode(frame).decode()}})
    decision_times = [f"{t:.2f}" for t in times[context_frames:]]
    content.append({"type": "text", "text": (
        "Return strict JSON with this shape:\n" + json.dumps(schema_hint, indent=1) +
        "\nInclude exactly one entry in `ticks` for each decision tick, in order: " + ", ".join(decision_times) +
        ". Allowed `decision` values: silent, context, act, unsure. Allowed observation `kind` values: " +
        ", ".join(OBSERVATION_KINDS) + ". Keep `why` under 20 words. Do not include lead-in frames.")})
    return content


SCHEMA_HINT = {
    "ticks": [{"t": 0.0, "decision": "silent"},
              {"t": 0.5, "decision": "context", "why": "short reason",
               "observation": {"kind": "object_taken", "subject": "frying pan", "location": "drawer under counter",
                               "detail": "lifted out by the right hand", "confidence": 0.85}},
              {"t": 1.0, "decision": "act", "why": "policy rule matched",
               "action": {"command": "notify", "args": {"message": "..."}, "rule": "rule id from the action policy"}},
              {"t": 1.5, "decision": "unsure"}],
    "chunk_summary": "one sentence",
}


class Frontier:
    def __init__(self, base_url: str, token: str, model: str, timeout: float = 300):
        self.base_url, self.token, self.model, self.timeout = base_url.rstrip("/"), token, model, timeout
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def complete(self, content: list[dict], max_tokens: int = 4000) -> str:
        body = json.dumps({"model": self.model, "max_tokens": max_tokens, "temperature": 0, "system": SYSTEM_PROMPT,
                           "messages": [{"role": "user", "content": content}]}).encode()
        request = urllib.request.Request(self.base_url + "/v1/messages", data=body, method="POST", headers={
            "content-type": "application/json", "anthropic-version": "2023-06-01",
            "x-api-key": self.token, "authorization": "Bearer " + self.token})
        last = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode())
                usage = payload.get("usage", {})
                self.usage["input_tokens"] += int(usage.get("input_tokens", 0))
                self.usage["output_tokens"] += int(usage.get("output_tokens", 0))
                self.usage["calls"] += 1
                return "".join(block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:300]
                last = f"HTTP {exc.code}: {detail}"
                if exc.code in (429, 500, 502, 503, 529):
                    time.sleep(5 * (attempt + 1))
                    continue
                raise RuntimeError(last) from None
            except (urllib.error.URLError, TimeoutError) as exc:
                last = type(exc).__name__
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"frontier unavailable after retries: {last}")


def extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in reply")
    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Common model slips: a trailing comma before a closing bracket or brace, or a code fence.
        import re
        repaired = re.sub(r",\s*([\]}])", r"\1", candidate.replace("```json", "").replace("```", ""))
        return json.loads(repaired)


def ask_json(frontier: "Frontier", content: list[dict], max_tokens: int) -> tuple[dict, str]:
    """Ask once, and once more with the rejection reason if the reply is not a JSON object."""
    text = frontier.complete(content, max_tokens=max_tokens)
    try:
        return extract_json(text), text
    except (ValueError, json.JSONDecodeError) as first:
        retry = content + [{"type": "text", "text": f"Your previous reply was not valid JSON ({str(first)[:200]}). "
                                                     "Reply again with one strict JSON object only, no prose, no code fence."}]
        text = frontier.complete(retry, max_tokens=max_tokens)
        return extract_json(text), text


def parse_reply(text: str, expected_times: list[float]) -> TeacherReply:
    reply = TeacherReply.model_validate(extract_json(text))
    for tick in reply.ticks:
        tick.check()
    got = [round(t.t, 2) for t in reply.ticks]
    want = [round(t, 2) for t in expected_times]
    if got != want:
        raise ValueError(f"tick times differ from the grid: got {got[:6]}... want {want[:6]}...")
    return reply


def label_chunk(frontier: Frontier, video: Path, start: float, end: float, hz: float, width: int,
                lead_in_s: float, context: dict) -> tuple[TeacherReply, dict]:
    lead_ticks = int(round(lead_in_s * hz))
    frame_start = max(0.0, start - lead_ticks / hz)
    frames = decode_frames(video, frame_start, end - frame_start, hz, width)
    times = [round(frame_start + k / hz, 3) for k in range(len(frames))]
    context_frames = sum(1 for t in times if t < start - 1e-6)
    decision_times = times[context_frames:]
    content = build_user_content(frames, times, context_frames, context, SCHEMA_HINT)
    payload, text = ask_json(frontier, content, 4000)
    try:
        reply = parse_reply(json.dumps(payload), decision_times)
    except (ValidationError, ValueError) as first:
        content.append({"type": "text", "text": f"Your previous reply was rejected: {str(first)[:300]}. Reply again with strict JSON only."})
        payload, text = ask_json(frontier, content, 4000)
        reply = parse_reply(json.dumps(payload), decision_times)
    record = {"frames_sent": len(frames), "lead_in_frames": context_frames, "width_px": width, "tick_hz": hz,
              "raw_text_sha256": hashlib.sha256(text.encode()).hexdigest()}
    return reply, record


def second_look(frontier: Frontier, video: Path, unsure: list[float], hz: float, width: int, context: dict,
                window_s: float = 2.0) -> list[TickDecision]:
    """Re-ask about unsure ticks with more frames at higher resolution; returns resolved decisions."""
    resolved = []
    for t in unsure:
        start, end = max(0.0, t - window_s), t + window_s
        frames = decode_frames(video, start, end - start, hz, width)
        times = [round(start + k / hz, 3) for k in range(len(frames))]
        content = [{"type": "text", "text": "Second look at higher frame rate and resolution. Context (JSON):\n" +
                    json.dumps(context, indent=1)}]
        for frame, ft in zip(frames, times):
            content.append({"type": "text", "text": f"frame t={ft:.2f}s"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                         "data": base64.b64encode(frame).decode()}})
        content.append({"type": "text", "text": (
            f"Decide for the single tick t={t:.2f}s only. Return strict JSON: {{\"ticks\": [one entry with t={t:.2f}], "
            "\"chunk_summary\": \"\"}}. `unsure` is not allowed now; choose silent, context or act.")})
        try:
            payload, _ = ask_json(frontier, content, 1200)
            reply = TeacherReply.model_validate(payload)
            tick = reply.ticks[0].check()
        except (ValidationError, ValueError, json.JSONDecodeError, IndexError) as exc:
            tick = TickDecision(t=t, decision="silent", why=f"second look unparseable: {type(exc).__name__}")
        if tick.decision == "unsure":
            tick = TickDecision(t=t, decision="silent", why="teacher could not resolve after second look")
        tick.t = t
        resolved.append(tick)
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None, help="default: end of video")
    parser.add_argument("--chunk-s", type=float, default=20.0)
    parser.add_argument("--tick-hz", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--lead-in-s", type=float, default=2.0)
    parser.add_argument("--second-look-hz", type=float, default=4.0)
    parser.add_argument("--second-look-width", type=int, default=896)
    parser.add_argument("--goal", default="Keep the household coordinator aware of what people do with objects, appliances and food in this room.")
    parser.add_argument("--action-policy", type=Path, default=None, help="JSON list of pre-authorized rules")
    parser.add_argument("--device-state", type=Path, default=None, help="JSON object of device state text")
    parser.add_argument("--camera", default="fixed", choices=["fixed", "ego"])
    parser.add_argument("--source-offset", type=float, default=0.0,
                        help="seconds to add to every reported time when the file is a clip cut from a longer source")
    parser.add_argument("--license", default="")
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--base-url", default="http://127.0.0.1:8317")
    parser.add_argument("--out", type=Path, required=True, help="JSONL of stream-target rows")
    parser.add_argument("--dry-run", action="store_true", help="build prompts, write a manifest, call nothing")
    args = parser.parse_args()

    token = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    if not args.dry_run and len(token) < 16:
        parser.error("ANTHROPIC_AUTH_TOKEN is not set")
    import imageio_ffmpeg
    meta = next(imageio_ffmpeg.read_frames(str(args.video), pix_fmt="rgb24", output_params=["-t", "0.05"]))
    video_end = float(meta["duration"]) if meta.get("duration") else None
    end = args.end if args.end is not None else video_end
    if end is None:
        parser.error("--end is required when the container has no duration")
    video_id = args.video_id or args.video.stem
    policy = json.loads(args.action_policy.read_text()) if args.action_policy else []
    device_state = json.loads(args.device_state.read_text()) if args.device_state else {}
    frontier = Frontier(args.base_url, token, args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    journal: list[dict] = []
    written = 0
    video_sha = sha256_file(args.video)
    started = datetime.now(timezone.utc).isoformat()
    with args.out.open("a") as sink:
        chunk_start = args.start
        while chunk_start < end - 1e-6:
            chunk_end = min(end, chunk_start + args.chunk_s)
            context = {"goal": args.goal, "action_policy": policy, "device_state": device_state,
                       "journal_prior": journal[-12:], "camera": args.camera,
                       "tick_hz": args.tick_hz, "chunk": [round(chunk_start, 3), round(chunk_end, 3)]}
            grid = tick_grid(chunk_start, chunk_end, args.tick_hz)
            if args.dry_run:
                print(json.dumps({"chunk": [chunk_start, chunk_end], "ticks": len(grid), "context_keys": list(context)}))
                chunk_start = chunk_end
                continue
            reply, record = label_chunk(frontier, args.video, chunk_start, chunk_end, args.tick_hz, args.width,
                                        args.lead_in_s, context)
            unsure = [t.t for t in reply.ticks if t.decision == "unsure"]
            looks = second_look(frontier, args.video, unsure, args.second_look_hz, args.second_look_width, context) if unsure else []
            by_time = {round(t.t, 2): t for t in looks}
            ticks = [by_time.get(round(t.t, 2), t) for t in reply.ticks]
            for tick in ticks:
                if tick.decision == "context" and tick.observation is not None:
                    journal.append({"t": tick.t, **tick.observation.model_dump()})
                elif tick.decision == "act" and tick.action is not None:
                    journal.append({"t": tick.t, "action": tick.action.model_dump()})
            shift = args.source_offset
            shifted = []
            for t in ticks:
                dumped = t.model_dump(exclude_none=True)
                dumped["t"] = round(dumped["t"] + shift, 3)
                shifted.append(dumped)
            row = {"schema_version": SCHEMA_VERSION, "prompt_version": PROMPT_VERSION,
                   "chunk_id": hashlib.sha256(f"{video_sha}:{chunk_start:.3f}:{chunk_end:.3f}:{args.tick_hz}".encode()).hexdigest()[:20],
                   "source": {"video_id": video_id, "sha256": video_sha, "start_s": round(chunk_start + shift, 3),
                              "end_s": round(chunk_end + shift, 3), "offset_s": shift, "file_start_s": round(chunk_start, 3),
                              "camera": args.camera, "license": args.license, "path": str(args.video)},
                   "tick_hz": args.tick_hz, "context": context,
                   "ticks": shifted,
                   "chunk_summary": reply.chunk_summary,
                   "teacher": {"model": args.model, "mode": "blind", "second_looks": unsure, **record},
                   "created_at": datetime.now(timezone.utc).isoformat()}
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            written += 1
            counts = {d: sum(1 for t in ticks if t.decision == d) for d in DECISIONS}
            print(json.dumps({"chunk": [round(chunk_start, 2), round(chunk_end, 2)], "decisions": counts,
                              "second_looks": len(unsure), "usage": frontier.usage}), flush=True)
            chunk_start = chunk_end
    manifest = {"video": str(args.video), "video_sha256": video_sha, "video_id": video_id, "range_s": [args.start, end],
                "chunk_s": args.chunk_s, "tick_hz": args.tick_hz, "width_px": args.width, "model": args.model,
                "prompt_version": PROMPT_VERSION, "rows_written": written, "usage": frontier.usage,
                "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(), "dry_run": args.dry_run}
    manifest_path = args.out.with_suffix(".manifest.json")
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"runs": []}
    existing.setdefault("runs", []).append(manifest)
    manifest_path.write_text(json.dumps(existing, indent=1))
    print(json.dumps({"rows_written": written, "manifest": str(manifest_path), "usage": frontier.usage}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
