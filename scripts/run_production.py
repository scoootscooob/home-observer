#!/usr/bin/env python3
"""Production-shaped recipe workflow: local perception, enforced export boundary, sandboxed frontier.

Every sensory byte stays on this host. The perception model runs in-process on
local hardware, the coordinator bridge publishes sanitized exports only, and
the frontier coordinator runs in a deny-by-default sandbox that can read one
directory and reach one local proxy port. The run fails closed if configured
with remote inference or the experimental media-capable bridge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from itertools import chain
from pathlib import Path

from home_observer.ha import HomeAssistant
from home_observer.local_verifier import LocalVerifier, VerifierPolicy
from home_observer.model import ModelConfig
from home_observer.object_tracker import CSRTContinuousTracker
from home_observer.physical_journal import PhysicalJournal
from home_observer.physical_model import prepare_physical_request
from home_observer.physical_schema import PhysicalDecision, PhysicalWindow
from home_observer.physical_stream import PhysicalStream, video_frames
from home_observer.production_coordinator import ProductionCoordinator
from home_observer.production_export import (
    ExportPolicy,
    ExportStore,
    ProductionExporter,
    ProductionModeError,
    ProductionRuntimeConfig,
    assert_production_coordinator,
    assert_production_model,
    load_export_key,
    scan_export,
)
from home_observer.schema import Action
from home_observer.watches import ApprovedResponse, WatchEngine, WatchStore

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import frontier_sandbox  # noqa: E402

POSITIVE_KINDS = {"pick-up", "pickup", "pick_up", "take", "lift", "picked_up", "grab"}


class LocalTestModel:
    """Deterministic in-process stand-in for software tests; asserts nothing about the scene."""

    local_inference = True
    sensory_media_leaves_host = False
    backend = "local-test"

    def __init__(self, config: ModelConfig):
        self.config = config

    def observe(self, request):
        clean = prepare_physical_request(request, self.config)
        frames = sum(len(clip["frames"]) for clip in clean["window"]["clips"])
        return {"decision": {"summary": "local test backend; no physical assertion"},
                "metrics": {"backend": self.backend, "latency_s": 0.0, "sampled_frames": frames},
                "raw_output": "{}"}


def build_model(runtime: ProductionRuntimeConfig, output: Path):
    config = ModelConfig(model_id=runtime.model_id, revision=runtime.revision, adapter_path=runtime.adapter_path,
                         dataset_root=str(output), quantization="none", max_frames=runtime.max_frames,
                         max_input_tokens=8192, max_new_tokens=runtime.max_new_tokens, merge_adapter=False)
    if runtime.inference_backend == "local-test":
        return LocalTestModel(config)
    if runtime.inference_backend == "local-mps":
        from home_observer.physical_local import PhysicalLocalModel
        return PhysicalLocalModel(config, device="mps", contract=runtime.output_contract)
    raise ProductionModeError(f"inference backend {runtime.inference_backend} is not available in this iteration")


def bootstrap_seed(stream, first, seed):
    """Explicit recorded box bootstrap; never described as a model detection."""
    allowed = {"label", "box", "confidence", "provenance", "uncertainty"}
    if set(seed) != allowed:
        raise ValueError("seed requires exactly label, box, confidence, provenance and uncertainty")
    folder = stream.output / "bootstrap"
    folder.mkdir(exist_ok=True)
    path = folder / "first-frame.jpg"
    path.write_bytes(first.jpeg)
    window = PhysicalWindow.model_validate({"window_id": "bootstrap", "started_at": first.timestamp,
        "ended_at": first.timestamp, "clips": [{"clip_id": "bootstrap_clip", "camera_id": first.camera_id,
            "started_at": first.timestamp, "ended_at": first.timestamp, "frames": [
                {"evidence_id": first.evidence_id, "timestamp": first.timestamp, "path": str(path)}]}]})
    decision = PhysicalDecision.model_validate({"summary": "Explicit box bootstrap from recorded provenance",
        "objects": [{"detection_id": "bootstrap_object", "label": seed["label"], "box": seed["box"],
            "location": {"camera_id": first.camera_id}, "timestamp": first.timestamp,
            "frame_evidence_id": first.evidence_id, "visibility": "partially_occluded",
            "confidence": seed["confidence"], "uncertainty": seed["uncertainty"],
            "evidence_ids": [first.evidence_id], "attributes": {"bootstrap_provenance": seed["provenance"]}}]})
    mapping = stream.journal.ingest(window, decision)
    stream.seeds.put_nowait({**decision.objects[0].model_dump(), "track_id": mapping["bootstrap_object"]})
    (folder / "seed.json").write_text(json.dumps({"seed": seed, "window": window.model_dump(),
        "decision": decision.model_dump(), "mapping": mapping}, indent=2) + "\n")
    return mapping["bootstrap_object"]


def learned_check_factory(stream, model, output: Path):
    """Focused local re-observation over retained frames; the verifier gets a value, never media."""
    log = output / "verifications-private.jsonl"

    def learned_check(subject_id, question, source_as_of, lookback_s):
        candidates = {}
        for manifest_path in stream.spool.root.glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, ValueError):
                continue
            for frame in manifest["frames"]:
                if source_as_of - lookback_s <= frame["timestamp"] <= source_as_of and Path(frame["path"]).is_file():
                    candidates[frame["evidence_id"]] = frame
        frames = sorted(candidates.values(), key=lambda f: f["timestamp"])
        if len(frames) < 2:
            raise ValueError("insufficient retained frames for a learned check")
        budget = model.config.max_frames
        if len(frames) > budget:
            picks = sorted({round(i * (len(frames) - 1) / (budget - 1)) for i in range(budget)})
            frames = [frames[i] for i in picks]
        registry = stream.registry(frames[-1]["timestamp"])
        label = next((t["last_detection"]["label"] for t in registry if t["track_id"] == subject_id), "object")
        identity = "verify_" + hashlib.sha256(f"{subject_id}:{question}:{frames[-1]['evidence_id']}".encode()).hexdigest()[:20]
        window = {"window_id": identity, "started_at": frames[0]["timestamp"], "ended_at": frames[-1]["timestamp"],
                  "clips": [{"clip_id": identity + "_clip", "camera_id": frames[0]["camera_id"],
                             "started_at": frames[0]["timestamp"], "ended_at": frames[-1]["timestamp"],
                             "frames": [{k: f[k] for k in ("evidence_id", "timestamp", "path")} for f in frames]}]}
        request = {"window": window, "tracks": registry,
                   "recent_events": stream.journal.recent_events(frames[-1]["timestamp"]),
                   "focus": [f"Verification question '{question}' for track {subject_id} ({label}): report one "
                             f"observation with subject_id '{subject_id}', attribute 'clear_of_support' and value "
                             f"true, false or \"unknown\", stating whether the {label} is now lifted clear of the "
                             f"surface that previously supported it. Report a pick-up event only if visible."]}
        result = model.observe(request)
        value, confidence = None, 0.0
        for observation in result["decision"].get("observations", []):
            attribute = str(observation.get("attribute", "")).lower()
            if observation.get("subject_id") == subject_id and any(k in attribute for k in ("clear", "lift", "support")):
                raw = observation.get("value")
                text = str(raw).strip().lower()
                if raw is True or text in ("true", "yes", "lifted", "clear"):
                    value, confidence = True, float(observation.get("confidence", 0.0))
                elif raw is False or text in ("false", "no", "supported", "resting"):
                    value, confidence = False, float(observation.get("confidence", 0.0))
        if value is None:
            for event in result["decision"].get("events", []):
                kind = str(event.get("kind", "")).strip().lower().replace(" ", "-")
                if subject_id in event.get("subject_ids", []) and kind in POSITIVE_KINDS:
                    value, confidence = True, float(event.get("confidence", 0.0))
        audit = {"window_id": identity, "frame_evidence_ids": [f["evidence_id"] for f in frames],
                 "raw_output_sha256": hashlib.sha256(result.get("raw_output", "").encode()).hexdigest(),
                 "latency_s": result.get("metrics", {}).get("latency_s")}
        with log.open("a") as handle:
            handle.write(json.dumps({"subject_id": subject_id, "question": question, "source_as_of": source_as_of,
                                     "value": value, "confidence": confidence, "request": request,
                                     "result": result}, ensure_ascii=False) + "\n")
        return {"value": value, "confidence": confidence, "audit": audit}

    return learned_check


class FrontierProcess:
    """The sandboxed frontier coordinator subprocess with restart support."""

    def __init__(self, bridge: Path, output: Path, *, coordinator_id: str, model: str, max_seconds: float,
                 python: str, allow_port: int):
        self.bridge, self.output, self.coordinator_id = bridge, output, coordinator_id
        self.model, self.max_seconds, self.python, self.allow_port = model, max_seconds, python, allow_port
        self.process = self.profile_path = None
        self.launches = 0
        self.log = open(output / "frontier-coordinator.log", "a")

    def start(self):
        env = {name: os.environ[name] for name in ("ANTHROPIC_AUTH_TOKEN",) if name in os.environ}
        if "ANTHROPIC_AUTH_TOKEN" not in env:
            raise ProductionModeError("the sandboxed frontier needs ANTHROPIC_AUTH_TOKEN for the local proxy")
        self.process, self.profile_path = frontier_sandbox.popen_sandboxed(
            bridge=self.bridge, python=self.python, script=SCRIPTS / "frontier_coordinator.py",
            # The sandbox allows exactly one local port, so the proxy URL is explicit here and
            # never taken from the caller's environment.
            script_args=["--bridge", str(self.bridge), "--coordinator-id", self.coordinator_id, "--model", self.model,
                         "--max-seconds", str(self.max_seconds), "--base-url",
                         f"http://127.0.0.1:{self.allow_port}"],
            allow_ports=[self.allow_port], env=env, profile_out=self.output / "frontier-sandbox-profile.sb",
            stdout=self.log, stderr=subprocess.STDOUT)
        self.launches += 1

    def stop(self, timeout=45):
        stop_file = self.bridge / "coordinator" / "stop"
        stop_file.parent.mkdir(exist_ok=True)
        stop_file.write_text("stop\n")
        if self.process is not None:
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)
            if self.profile_path and Path(self.profile_path).exists():
                os.unlink(self.profile_path)
        self.log.close()

    def interrupt_and_restart(self, pause_s=3.0):
        """Simulated coordinator outage: kill without notice, then relaunch with resumed state."""
        if self.process is not None:
            self.process.kill()
            self.process.wait(timeout=10)
            if self.profile_path and Path(self.profile_path).exists():
                os.unlink(self.profile_path)
        time.sleep(pause_s)
        self.start()


def prepare_light(executor, response, output):
    """Isolated fixture preparation: record the state and make sure the cue starts from off."""
    entity = response.ha_entity_id
    before = executor.read_states([entity]).get(entity)
    prepared = {"entity_id": entity, "state_before": before, "prepared_at": time.time()}
    if before is None:
        raise RuntimeError("approved entity is not present in the isolated Home Assistant fixture")
    if before.get("state") != "off":
        result = executor.execute(Action(domain=response.domain, service="turn_off", entity_id=entity, data={},
                                         reason="Preparation: reset the isolated demo light before the run",
                                         evidence_ids=["preparation"]), "preparation")
        prepared["reset_result"] = result
        if not result.get("verified"):
            raise RuntimeError("could not reset the isolated demo light to off before the run")
    prepared["state_after"] = executor.read_states([entity]).get(entity)
    (output / "preparation.json").write_text(json.dumps(prepared, indent=2) + "\n")
    return prepared


def bridge_scan(bridge: Path) -> dict:
    """Every frontier-visible file must be scanner-clean JSON; context files are digital inputs."""
    files, violations = [], {}
    for path in sorted(bridge.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(bridge).as_posix()
        files.append(relative)
        if relative.startswith("context/") or relative.startswith("coordinator/"):
            continue
        try:
            found = scan_export(json.loads(path.read_text()))
        except (OSError, ValueError) as exc:
            found = ["unreadable: " + type(exc).__name__]
        if found:
            violations[relative] = found
    return {"files": files, "violations": violations, "frontier_visible_files": len(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--camera-id", default="recipe")
    parser.add_argument("--source-origin", type=float, default=0.0)
    parser.add_argument("--runtime", type=Path, default=Path("configs/production/runtime.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--digital-context", type=Path, required=True)
    parser.add_argument("--ha-credentials", type=Path)
    parser.add_argument("--ha-url", default="http://127.0.0.1:8124")
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--frontier", choices=["sandboxed", "none"], default="sandboxed")
    parser.add_argument("--frontier-model", default="claude-opus-5")
    parser.add_argument("--frontier-max-seconds", type=float, default=900)
    parser.add_argument("--frontier-python", default=None,
                        help="Base interpreter for the sandboxed coordinator (defaults to this interpreter's base)")
    parser.add_argument("--proxy-port", type=int, default=8317)
    parser.add_argument("--wait-for-watch-seconds", type=float, default=180)
    parser.add_argument("--post-source-watch-seconds", type=float, default=60)
    parser.add_argument("--verification-drain-seconds", type=float, default=120)
    parser.add_argument("--inference-delay-s", type=float, default=0)
    parser.add_argument("--interrupt-frontier-at-seconds", type=float, default=None)
    parser.add_argument("--tracker", choices=["csrt", "lk"], default="csrt")
    parser.add_argument("--motion-threshold", type=float, default=0.003,
                        help="Normalized per-frame relative motion needed for a moving frame (default 0.003)")
    parser.add_argument("--motion-confirm-frames", type=int, default=3,
                        help="Consecutive moving frames before motion_started is emitted (default 3)")
    parser.add_argument("--opencv-threads", type=int, default=1)
    parser.add_argument("--skip-light-preparation", action="store_true")
    args = parser.parse_args()
    import cv2
    cv2.setNumThreads(max(1, args.opencv_threads))
    runtime = ProductionRuntimeConfig.load(args.runtime)
    policy = ExportPolicy.load(runtime.export_policy)
    key = load_export_key(runtime.export_key_file)
    output = args.output.resolve()
    bridge = args.bridge.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("use a fresh output directory for a new recorded run")
    if bridge.exists() and any(bridge.iterdir()):
        raise ValueError("use a fresh bridge directory for a new recorded run")
    if bridge.is_relative_to(output):
        raise ProductionModeError("the frontier-visible bridge cannot live inside the private output directory")
    output.mkdir(parents=True, exist_ok=True)
    approved = [ApprovedResponse.model_validate(x) for x in json.loads(args.responses.read_text())]
    executor = None
    if args.ha_credentials:
        credentials = json.loads(args.ha_credentials.read_text())
        token = credentials.get("long_lived_token") or credentials.get("token") or credentials.get("access_token")
        executor = HomeAssistant(credentials.get("base_url", args.ha_url), token, execute_real=True)
        if not args.skip_light_preparation:
            for response in approved:
                if response.kind == "ha_service":
                    prepare_light(executor, response, output)
    store = WatchStore(output / "journal.sqlite3")
    engine = WatchEngine(store, approved, executor=executor)
    exporter = ProductionExporter(engine, ExportStore(output / "journal.sqlite3"), policy, key)
    coordinator = ProductionCoordinator(engine, exporter, bridge, private_directory=output / "bridge-private",
                                        async_verification=True)
    journal = PhysicalJournal(output / "journal.sqlite3")
    model = build_model(runtime, output)
    declaration = {**assert_production_model(model), **assert_production_coordinator(coordinator)}
    (bridge / "context").mkdir(exist_ok=True)
    shutil.copyfile(args.digital_context, bridge / "context/digital-context.json")
    (bridge / "context/responses.json").write_text(json.dumps([
        {"response_id": r.response_id, "kind": r.kind, "purpose": r.message} for r in approved], indent=2) + "\n")
    coordinator_id = "sandboxed-frontier-" + hashlib.sha256(str(output).encode()).hexdigest()[:12]
    (output / "runtime-ready.json").write_text(json.dumps({
        "mode": "production", "runtime": runtime.model_dump(), "declaration": declaration,
        "bridge": str(bridge), "coordinator_id": coordinator_id,
        "approved_response_ids": [x.response_id for x in approved], "ready_at": time.time(),
        "source_kind": "continuous_file_replay", "source_sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
        "source_origin": args.source_origin, "camera_id": args.camera_id, "frontier": args.frontier,
        "tracker": {"kind": args.tracker, "motion_threshold": args.motion_threshold,
                    "motion_confirm_frames": args.motion_confirm_frames}}, indent=2) + "\n")
    frontier = None
    if args.frontier == "sandboxed":
        base = args.frontier_python or str(Path(sys.base_prefix) / "bin" / f"python{sys.version_info.major}.{sys.version_info.minor}")
        frontier = FrontierProcess(bridge, output, coordinator_id=coordinator_id, model=args.frontier_model,
                                   max_seconds=args.frontier_max_seconds, python=base, allow_port=args.proxy_port)
        frontier.start()
    stream = None
    try:
        deadline = time.time() + args.wait_for_watch_seconds
        while not store.db.execute("SELECT count(*) FROM local_watches WHERE status='armed'").fetchone()[0]:
            coordinator.tick()
            if time.time() >= deadline:
                raise TimeoutError("No frontier-created watch arrived before the configured start deadline")
            time.sleep(0.05)
        wall_origin = time.time()
        tracker_settings = {"motion_threshold": args.motion_threshold,
                            "motion_confirm_frames": args.motion_confirm_frames}
        if args.tracker == "csrt":
            tracker = CSRTContinuousTracker(**tracker_settings)
        else:
            from home_observer.physical_tracking import ContinuousTracker
            tracker = ContinuousTracker(**tracker_settings)
        stream = PhysicalStream(output, model, journal=journal, coordinator=coordinator, tracker=tracker,
                                source_origin=args.source_origin, wall_origin=wall_origin,
                                inference_delay_s=args.inference_delay_s,
                                post_source_watch_seconds=args.post_source_watch_seconds)
        learned = None if runtime.inference_backend == "local-test" else learned_check_factory(stream, model, output)
        coordinator.verifier = LocalVerifier(
            journal, stream.tracking_journal, clock_mapping=stream.clock_mapping, policy=VerifierPolicy(),
            learned_check=learned,
            latest_source_time=lambda: stream.buffer.last_frame.timestamp if stream.buffer.last_frame else None)
        frames = iter(video_frames(args.video, camera_id=args.camera_id, source_origin=args.source_origin,
                                   wall_origin=wall_origin, realtime=True))
        seed_track = None
        if args.seed:
            first = next(frames)
            seed_track = bootstrap_seed(stream, first, json.loads(args.seed.read_text()))
            frames = chain([first], frames)
        interruptions = []
        if frontier is not None and args.interrupt_frontier_at_seconds is not None:
            def interrupt():
                interruptions.append({"killed_at": time.time()})
                frontier.interrupt_and_restart()
                interruptions[-1]["relaunched_at"] = time.time()
            threading.Timer(args.interrupt_frontier_at_seconds, interrupt).start()
        report = stream.run(frames)
        drain_deadline = time.time() + args.verification_drain_seconds
        while coordinator.pending_verifications and time.time() < drain_deadline:
            coordinator.tick()
            time.sleep(0.05)
        coordinator.tick()
        if frontier is not None:
            # Give the coordinator one last look at the final exports, then stop it.
            settle = time.time() + min(30, args.post_source_watch_seconds)
            while time.time() < settle:
                coordinator.tick()
                time.sleep(0.2)
            frontier.stop()
        scan = bridge_scan(bridge)
        final_assessment = None
        assessment_path = bridge / "coordinator/final-assessment.json"
        if assessment_path.is_file():
            final_assessment = json.loads(assessment_path.read_text())
        private_rows = [dict(r) for r in store.db.execute("SELECT topic,payload_json FROM watch_outbox")]
        private_contains_paths = any("/" in row["payload_json"] for row in private_rows)
        evidence = {
            "declaration": declaration, "runtime": runtime.model_dump(), "policy_sha256": hashlib.sha256(
                Path(runtime.export_policy).read_bytes()).hexdigest(),
            "bridge_scan": scan, "export_store": exporter.store.stats(), "blocked_exports": exporter.store.blocked(),
            "private_outbox_envelopes": len(private_rows), "private_outbox_contains_paths": private_contains_paths,
            "frontier": {"mode": args.frontier, "model": args.frontier_model if frontier else None,
                         "launches": frontier.launches if frontier else 0, "interruptions": interruptions,
                         "sandbox_profile": str(output / "frontier-sandbox-profile.sb") if frontier else None,
                         "allowed_port": args.proxy_port if frontier else None, "final_assessment": final_assessment},
            "seed_track": seed_track, "stream_errors": report["errors"], "captured_frames": report["captured_frames"],
            "verified_at": time.time(),
            "claim": "Sensory media stayed on this host: in-process local inference, exports-only bridge, "
                     "sandboxed frontier with no media/filesystem access. Field filtering and host isolation "
                     "are both enforced and tested; this is not a claim about model correctness.",
        }
        (output / "privacy-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
        summary = {"captured_frames": report["captured_frames"], "journal": report["journal"],
                   "errors": report["errors"], "exports": exporter.store.stats(),
                   "bridge_violations": scan["violations"], "final_assessment": final_assessment,
                   "watches": [{k: w[k] for k in ("watch_id", "status")} for w in report["watches_at_shutdown"]],
                   "output": str(output)}
        (output / "production-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary))
        if report["errors"] or scan["violations"]:
            raise SystemExit(1)
    finally:
        if frontier is not None and frontier.process is not None and frontier.process.poll() is None:
            frontier.stop()
        if stream is not None:
            stream.close()
        journal.close()
        store.close()
        if executor:
            executor.client.close()


if __name__ == "__main__":
    main()
