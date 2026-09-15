#!/usr/bin/env python3
"""Production frontier coordinator: plans from sanitized exports only, standard library only.

This process is meant to run inside scripts/frontier_sandbox.py. It can see one
directory (the bridge) and reach one local inference proxy. It receives typed
exports, asks a frontier model for a plan, writes typed commands, acknowledges
consumed exports by export ID and hash, and records every turn. It has no way
to request or read media, and it never asks for any.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SYSTEM_PROMPT = """You are the cloud planning coordinator for a home assistant. A local sensing server keeps all
camera images, video and audio on the household's own device. You receive ONLY typed, sanitized exports:
physical events (opaque event and track IDs, a bounded kind and object category, an area, timestamps,
a confidence and an uncertainty band), watch status updates, perception status counts and local
verification verdicts. You never see images, descriptions, file paths or free text about the scene, there
is no tool to fetch media, and you must never ask for any. Low confidence never justifies requesting media.

Your job for the digital commitment you are given:
1. Create bounded local watches that react to physical events. A watch may only select a response_id from
   the approved response menu; it cannot name devices or code. Create at most one watch per response_id.
2. Acknowledge exports you have consumed with ack_export commands naming the exact export_id and export_sha256.
3. Ask the local verifier structured questions with verify commands. Allowed questions:
   moved_from_initial_location, clear_of_support, still_present, pickup_completed. subject_id must be a
   phys_ track ID that appeared in an export. Ask pickup_completed only after you have seen motion for that
   subject; do not repeat an identical question more than twice.
4. Decide whether the recipe step is complete. It is complete ONLY when a verification_result export says
   verdict "confirmed" for pickup_completed or clear_of_support on the relevant subject. Geometric motion,
   a verified light cue, or a learned event alone never complete the step. If evidence is missing or the
   verifier says unknown, the step stays pending and pickup_verified is "unknown". If the verifier says
   contradicted, pickup_verified is "contradicted".

Respond with ONE JSON object and nothing else:
{"commands": [...], "assessment": {"recipe_step_complete": false, "pickup_verified": "unknown",
 "rationale": "<= 300 characters"}, "wait": true}
Command shapes (coordinator_id and created_at are supplied to you; copy them exactly):
- {"command_id": "<unique>", "coordinator_id": "<given>", "type": "create_watch", "watch": {"watch_id": "<id>",
   "context": {"commitment_id": "<from digital context>", "text": "<short paraphrase>", "coordinator_id": "<given>"},
   "condition": {"type": "event", "kind": "motion_started", "object_label": "bowl", "origin": "geometric",
   "min_confidence": 0.5, "min_duration_s": 0, "max_gap_s": 0.5}, "response_id": "<approved id>",
   "created_at": <given now>, "ttl_s": 3600, "response_deadline_s": 1.0, "max_event_age_s": 2.0, "max_responses": 1}}
- {"command_id": "<unique>", "coordinator_id": "<given>", "type": "ack_export", "export_id": "<export_...>",
   "export_sha256": "<64 hex>"}
- {"command_id": "<unique>", "coordinator_id": "<given>", "type": "verify", "verification": {"request_id": "<id>",
   "question": "pickup_completed", "subject_id": "<phys_...>", "lookback_s": 30}}
- {"command_id": "<unique>", "coordinator_id": "<given>", "type": "cancel_watch", "watch_id": "<id>"}
IDs use letters, digits, '-', '_', '.' and ':' only. Event kinds and object categories in conditions must be
plain lowercase tokens. Watches trigger fast local responses; verification is slower and separate."""


def atomic_write(path: Path, value) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def read_json_files(directory: Path) -> dict[str, dict]:
    result = {}
    for path in sorted(directory.glob("*.json")):
        try:
            result[path.name] = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
    return result


class FrontierClient:
    def __init__(self, base_url: str, token: str, model: str, fallback_model: str | None, timeout: float = 180):
        self.base_url, self.token, self.model, self.fallback, self.timeout = base_url.rstrip("/"), token, model, fallback_model, timeout
        self.calls = []

    def complete(self, system: str, prompt: str, *, max_tokens: int = 2000) -> tuple[str, dict]:
        errors = []
        for model in [self.model, *([self.fallback] if self.fallback else [])]:
            body = json.dumps({"model": model, "max_tokens": max_tokens, "system": system, "temperature": 0,
                               "messages": [{"role": "user", "content": prompt}]}).encode()
            request = urllib.request.Request(self.base_url + "/v1/messages", data=body, method="POST", headers={
                "content-type": "application/json", "anthropic-version": "2023-06-01",
                "x-api-key": self.token, "authorization": "Bearer " + self.token,
                "user-agent": "home-observer-frontier-coordinator/1"})
            started = time.time()
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout) as response:
                        payload = json.loads(response.read())
                    text = "".join(part.get("text", "") for part in payload.get("content", [])
                                   if isinstance(part, dict) and part.get("type") == "text")
                    usage = payload.get("usage", {})
                    record = {"model": payload.get("model", model), "attempt": attempt + 1,
                              "latency_s": time.time() - started, "usage": usage}
                    self.calls.append(record)
                    return text, record
                except urllib.error.HTTPError as exc:
                    detail = exc.read()[:300].decode("utf-8", "replace")
                    errors.append({"model": model, "status": exc.code, "detail": detail})
                    if exc.code in (400, 401, 403, 404):
                        break
                    time.sleep(2 * (attempt + 1))
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    errors.append({"model": model, "error": type(exc).__name__})
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError("frontier model unavailable: " + json.dumps(errors)[:1000])


def parse_plan(text: str) -> dict:
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("frontier reply contained no JSON object")
    plan = json.loads(body[start:end + 1])
    if not isinstance(plan, dict) or not isinstance(plan.get("commands", []), list):
        raise ValueError("frontier plan must be an object with a commands list")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--coordinator-id", required=True)
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--fallback-model", default="claude-sonnet-5")
    parser.add_argument("--base-url", default=os.environ.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:8317"))
    parser.add_argument("--token-env", default="ANTHROPIC_AUTH_TOKEN")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--replan-seconds", type=float, default=20)
    parser.add_argument("--stop-file", default="coordinator/stop")
    args = parser.parse_args()
    token = os.environ.get(args.token_env, "")
    if not token:
        raise SystemExit("frontier coordinator requires the inference proxy token in the environment")
    bridge = args.bridge.resolve()
    state_dir = bridge / "coordinator"
    (state_dir / "turns").mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {
        "seen_exports": [], "acked_exports": [], "commands": [], "turns": 0, "assessment": None,
        "started_at": time.time(), "restarts": 0}
    if state_path.is_file():
        state["restarts"] = state.get("restarts", 0) + 1
    context = json.loads((bridge / "context/digital-context.json").read_text())
    responses = json.loads((bridge / "context/responses.json").read_text())
    client = FrontierClient(args.base_url, token, args.model, args.fallback_model)
    deadline = time.time() + args.max_seconds
    last_turn_at = 0.0
    stop_file = bridge / args.stop_file
    log = open(state_dir / "coordinator.log", "a")

    def note(message):
        log.write(json.dumps({"at": time.time(), **message}) + "\n")
        log.flush()

    note({"event": "start", "model": args.model, "base_url": args.base_url, "restart": state["restarts"]})
    while time.time() < deadline and not stop_file.exists():
        exports = read_json_files(bridge / "outbox")
        ordered = sorted(exports.values(), key=lambda e: (e.get("created_at", 0), e.get("export_id", "")))
        seen = set(state["seen_exports"])
        fresh = [e for e in ordered if e.get("export_id") not in seen]
        due = fresh or state["turns"] == 0 or (time.time() - last_turn_at >= args.replan_seconds
                                               and state["turns"] < args.max_turns and fresh == [] and state["turns"] < 3)
        if not due or state["turns"] >= args.max_turns:
            time.sleep(args.poll_seconds)
            continue
        now = time.time()
        results = read_json_files(bridge / "results")
        rejected = read_json_files(bridge / "rejected")
        recent = ordered[-80:]
        prompt = json.dumps({
            "now": round(now, 3), "coordinator_id": args.coordinator_id, "digital_context": context,
            "approved_responses": responses, "exports_new": fresh[-40:],
            "exports_all_recent": [{k: e[k] for k in ("topic", "export_id", "created_at")} for e in recent],
            "export_messages_recent": [{"export_id": e["export_id"], "topic": e["topic"], "message": e["message"]}
                                        for e in recent[-40:]],
            "command_results": list(results.values())[-40:], "rejected_commands": list(rejected.values())[-20:],
            "previous_assessment": state["assessment"], "already_acknowledged": state["acked_exports"][-80:],
            "turn": state["turns"] + 1}, indent=1)
        text, record, plan, error = "", {}, {"commands": [], "assessment": state["assessment"]}, None
        for attempt in range(2):
            try:
                text, record = client.complete(SYSTEM_PROMPT, prompt if attempt == 0 else prompt
                                               + "\n\nYour previous reply was not valid JSON. Reply with one valid JSON object only.")
                plan = parse_plan(text)
                error = None
                break
            except ValueError as exc:
                error = f"{type(exc).__name__}: {exc}"[:500]
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:500]
                break
        written, skipped = [], []
        existing = {c["command_id"] for c in state["commands"]}
        for command in plan.get("commands", []):
            if not isinstance(command, dict) or not isinstance(command.get("command_id"), str):
                skipped.append({"reason": "malformed", "command": str(command)[:200]})
                continue
            command["coordinator_id"] = args.coordinator_id
            if command.get("type") == "create_watch" and isinstance(command.get("watch"), dict):
                watch = command["watch"]
                watch.setdefault("context", {})["coordinator_id"] = args.coordinator_id
                created = watch.get("created_at")
                if not isinstance(created, (int, float)) or not now - 60 <= created <= now + 1:
                    watch["created_at"] = round(now, 3)
            if command["command_id"] in existing:
                skipped.append({"reason": "duplicate_command_id", "command_id": command["command_id"]})
                continue
            atomic_write(bridge / "inbox" / (command["command_id"] + ".json"), command)
            existing.add(command["command_id"])
            state["commands"].append({"command_id": command["command_id"], "type": command.get("type"),
                                      "written_at": time.time()})
            written.append(command)
            if command.get("type") == "ack_export" and command.get("export_id"):
                state["acked_exports"].append(command["export_id"])
        # Consumption acknowledgement: every export presented in this turn that the planner
        # did not acknowledge itself is acknowledged by this coordinator process now.
        for export in fresh:
            if export["export_id"] not in state["acked_exports"]:
                command_id = "ack-" + export["export_id"][7:23] + "-" + str(state["turns"] + 1)
                command = {"command_id": command_id, "coordinator_id": args.coordinator_id, "type": "ack_export",
                           "export_id": export["export_id"], "export_sha256": export["export_sha256"]}
                atomic_write(bridge / "inbox" / (command_id + ".json"), command)
                state["commands"].append({"command_id": command_id, "type": "ack_export", "written_at": time.time(),
                                          "auto": True})
                state["acked_exports"].append(export["export_id"])
        state["seen_exports"].extend(e["export_id"] for e in fresh)
        state["turns"] += 1
        if isinstance(plan.get("assessment"), dict):
            state["assessment"] = plan["assessment"]
        last_turn_at = time.time()
        atomic_write(state_dir / "turns" / f"turn-{state['turns']:03d}.json", {
            "turn": state["turns"], "at": now, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt": json.loads(prompt), "model_record": record, "model_text": text, "error": error,
            "commands_written": written, "skipped": skipped, "auto_acknowledged": [e["export_id"] for e in fresh],
            "assessment": state["assessment"]})
        atomic_write(state_path, state)
        note({"event": "turn", "turn": state["turns"], "new_exports": len(fresh), "commands": len(written),
              "error": error})
        time.sleep(args.poll_seconds)
    final = {"coordinator_id": args.coordinator_id, "finished_at": time.time(), "turns": state["turns"],
             "restarts": state["restarts"], "exports_seen": len(state["seen_exports"]),
             "exports_acknowledged": len(set(state["acked_exports"])), "commands_written": len(state["commands"]),
             "assessment": state["assessment"], "model_calls": client.calls,
             "stopped_by": "stop_file" if stop_file.exists() else "deadline",
             "media_access": "none: sandboxed to the bridge directory and the local inference proxy port"}
    atomic_write(state_dir / "final-assessment.json", final)
    atomic_write(state_path, state)
    note({"event": "stop", **{k: v for k, v in final.items() if k != "model_calls"}})
    print(json.dumps({k: v for k, v in final.items() if k != "model_calls"}))


if __name__ == "__main__":
    sys.exit(main())
