#!/usr/bin/env python3
"""Build a standalone, offline evidence dashboard from existing measured artifacts.

Run again after copying new artifacts. Missing reports remain explicitly pending.
No model calls, fabricated results, external web assets or network requests.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

REPORT_FILES = {"training_report.json", "report.json", "metrics.json", "benchmark_report.json",
                "batch_benchmark_report.json", "smoke_report.json", "hardware.json", "gpu.json",
                "cost_report.json", "costs.json", "live_report.json", "generation-report.json",
                "transport_summary.json", "summary.json", "interrupted.json", "comparison.json", "runtime.json",
                "annotation-metrics.json", "manifest.json", "provenance.json", "source-provenance.json",
                "selection.json", "training_manifest.json", "resume_manifest.json", "validation.json",
                "merge_report.json", "merge-metadata.json", "combination-provenance.json",
                "serving-response-evidence.json", "registered-models.json", "comparison-confound.json",
                "encoded-input-comparison.json", "canonicalization-verification.json"}
GROUPS = {"training": "artifacts/training-v1", "release": "artifacts/release-v1",
          "training_v2": "artifacts/training-v2", "release_v2": "artifacts/release-v2",
          "batch": "artifacts/h100-batch", "consumer": "artifacts/consumer-benchmark",
          "merge": "artifacts/h100-merge-v1", "context": "artifacts/h100-context-v1",
          "consumer_compile": "artifacts/consumer-compile", "base_fixed8": "reports/base-fixed8",
          "live": "reports/live-cloud-v1", "base_ablation": "reports/base-state-ablation",
          "checkpoint300_validation": "artifacts/checkpoint300-validation",
          "release_base_v2_policy": "artifacts/release-base-v2-policy",
          "release_checkpoint300_v2_policy": "artifacts/release-checkpoint300-v2-policy",
          "release_continuation_selected": "artifacts/release-continuation-selected",
          "release_continuation_merged_rejected": "artifacts/release-continuation-merged-rejected",
          "release_trained50_vllm_canonical": "artifacts/release-trained50-vllm-canonical",
          "training_v2_continuation": "artifacts/training-v2-continuation",
          "trained50_vllm_fixed8": "reports/trained50-vllm-fixed8",
          "vllm_lora50": "artifacts/vllm-lora-checkpoint50",
          "continuation_policy_order": "artifacts/continuation-policy-order",
          "policy_order_diagnostic": "artifacts/policy-order-diagnostic",
          "base_batch_parity": "artifacts/base-batch-parity",
          "vllm_fixed8": "reports/vllm-fixed8", "vllm_upload_cli": "reports/vllm-upload-cli",
          "learned-home-assistant-vllm": "reports/learned-home-assistant-vllm",
          "live_ha_vllm": "reports/live-home-assistant-vllm",
          "learned_ha": "reports/learned-home-assistant"}


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_jsonl(path, limit=250):
    try:
        with path.open(encoding="utf-8") as source:
            for index, line in enumerate(source):
                if index >= limit:
                    return
                if line.strip():
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue  # An artifact may still be copying/appending.
    except OSError:
        return


def collect(project: Path, output: Path):
    project = project.resolve()
    groups = dict(GROUPS)
    for folder in sorted((project / "reports").glob("learned-home-assistant-*")):
        if folder.is_dir():
            groups[folder.name] = folder.relative_to(project).as_posix()
    for folder in sorted((project / "reports").glob("*release*")):
        if folder.is_dir() and folder.relative_to(project).as_posix() not in groups.values():
            groups[folder.name] = folder.relative_to(project).as_posix()
    for folder in sorted((project / "artifacts").glob("release-*")):
        if folder.is_dir() and folder.relative_to(project).as_posix() not in groups.values():
            groups[folder.name] = folder.relative_to(project).as_posix()
    for pattern in ("vllm-fixed8-*", "vllm-upload-cli-*", "live-home-assistant-vllm-*"):
        for folder in sorted((project / "reports").glob(pattern)):
            if folder.is_dir() and folder.relative_to(project).as_posix() not in groups.values():
                groups[folder.name] = folder.relative_to(project).as_posix()
    result = {"built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "reports": [], "predictions": [], "media": {}, "datasets": [], "warnings": [],
              "groups": groups, "checkpoint_artifacts": [], "prediction_files": [],
              "readme_href": quote(os.path.relpath(project / "README.md", output.parent)),
              "evaluation_guide_href": quote(os.path.relpath(project / "docs/evaluation-guide.md", output.parent))}
    for group, relative in (("training_v2", "artifacts/training-v2/adapter"),
                            ("training_v2_continuation", "artifacts/training-v2-continuation/adapter/checkpoints")):
        for folder in sorted((project / relative).glob("checkpoint-*")):
            try:
                step = int(folder.name.removeprefix("checkpoint-"))
            except ValueError:
                continue
            result["checkpoint_artifacts"].append({"step": step, "group": group,
                "path": folder.relative_to(project).as_posix(),
                "report_present": read_json(folder / "generation-report.json") is not None,
                "generation_rows": sum(1 for _ in read_jsonl(folder / "generation-validation.jsonl")),
                "weights_present": (folder / "adapter_model.safetensors").is_file()})
    rows_by_id = {}
    for name, filenames in {"mixed": ["test.jsonl"], "counterfactuals": ["pairs.jsonl"],
                             "mixed-v2": ["validation.jsonl"],
                             "coco-household": ["test.jsonl"], "public-egolife": ["replay.jsonl"]}.items():
        folder = project / "data" / name
        manifest = read_json(folder / "manifest.json")
        if manifest is not None:
            result["datasets"].append({"name": name, "data": manifest,
                "href": quote(os.path.relpath(folder / "manifest.json", output.parent)),
                "sha256": hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest()})
        for filename in filenames:
            for row in read_jsonl(folder / filename, limit=2000):
                row["_media_root"] = project / "data" if name in ("mixed", "mixed-v2") else folder
                rows_by_id[str(row.get("id"))] = row

    def href(path):
        return quote(os.path.relpath(path, output.parent))

    def provenance(path):
        raw = path.read_bytes()
        return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")}

    def thumbnail(path_value, media_root):
        if not path_value:
            return None
        original = Path(path_value)
        possibilities = [original if original.is_absolute() else media_root / original]
        text_path = str(original)
        for directory in ("data", "artifacts", "reports"):
            if f"/{directory}/" in text_path:
                possibilities.append(project / directory / text_path.split(f"/{directory}/", 1)[1])
        path = next((p.resolve() for p in possibilities if p.is_file() and p.resolve().is_relative_to(project)), None)
        if path is None:
            return None
        key = hashlib.sha256(str(path).encode()).hexdigest()[:20]
        if key in result["media"]:
            return key
        if len(result["media"]) >= 600:
            return None
        try:
            from PIL import Image
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((360, 240))
                buffer = BytesIO()
                image.save(buffer, "JPEG", quality=72)
            result["media"][key] = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
        except (ImportError, OSError, ValueError):
            return None
        return key

    def prediction(item, path, group):
        engine_raw = path.name == "engine-outputs.jsonl"
        record_id = str(item.get("id", item.get("window_id", "unknown")))
        # Raw engine logs carry only window IDs, intentionally shared by members
        # of counterfactual pairs. Never guess a row/target join from those IDs.
        source_row = {} if engine_raw else rows_by_id.get(
            record_id, rows_by_id.get(item.get("result", {}).get("window_id"), {}))
        window = item.get("window", item.get("request", {}).get("window", source_row.get("window", {})))
        if item.get("device_readback") and window:
            window = {**window, "device_states": item["device_readback"]}
        if record_id == "unknown":
            record_id = str(window.get("window_id", "unknown"))
        record = item.get("result", item)
        decision = record.get("decision")
        raw = record.get("raw_output", item.get("raw_output"))
        target = None if engine_raw else item.get("target", source_row.get("target"))
        source = item.get("source", source_row.get("source", item.get("source_kind", "unspecified")))
        if isinstance(source, dict):
            source = source.get("type", source.get("kind", json.dumps(source, sort_keys=True)))
        frames = [{"camera": frame.get("camera_id", "camera"), "evidence_id": frame.get("evidence_id"),
                   "media": thumbnail(frame.get("path"), source_row.get("_media_root", project / "data"))}
                  for frame in window.get("frames", [])[:8]]
        result["predictions"].append({"id": record_id, "group": group, "file": path.relative_to(project).as_posix(),
            "variant": "original engine output" if engine_raw else item.get("mode", item.get("phase", path.parent.name if path.parent.name.startswith("checkpoint-") else "")),
            "href": href(path), "task": "engine_raw_generation" if engine_raw else item.get(
                "task_type", source_row.get("task_type", item.get("case", "unspecified"))),
            "source": source,
            "decision": decision, "target": target, "raw": raw, "frames": frames,
            "metrics": record.get("metrics", {}), "latency_s": record.get("total_latency_s", record.get("observed_service_s")),
            "error": record.get("error", record.get("validation_error")), "rejections": record.get("rejections", []),
            "supervision_mask": item.get("supervision_mask", source_row.get("supervision_mask", {})),
            "has_audio": bool(window.get("audio")), "device_states": window.get("device_states", {}),
            "split": source_row.get("split"), "group_id": source_row.get("group_id"),
            "input_context_source": ("recorded request" if item.get("request") else "recorded observer context"
                                     if "prior_context" in record else "source dataset row" if source_row else "unavailable"),
            "prior_state": item.get("request", {}).get("state", record.get("prior_context", {}).get(
                "state", source_row.get("runtime_context", {}).get("state"))),
            "media_integrity": record.get("media", {}),
            "provenance": source_row.get("provenance", {}),
            "recorded_quality_label": item.get("valid_grounded_decision")})

    for group, relative in groups.items():
        root = project / relative
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            if path.name not in REPORT_FILES or path.stat().st_size > 8_000_000:
                continue
            data = read_json(path)
            if data is None:
                result["warnings"].append(f"Report unreadable or still copying: {path.relative_to(project)}")
                continue
            result["reports"].append({"group": group, "name": path.name,
                                      "path": path.relative_to(project).as_posix(), "href": href(path),
                                      **provenance(path), "data": data})
            if "learned-home-assistant" in relative and path.name == "report.json":
                for case in data.get("cases", []):
                    prediction({**case, "target": {"actions": case.get("expected_actions", [])},
                        "supervision_mask": {"actions": True, "summary": False, "observations": False, "noop": False},
                        "task_type": "isolated_ha_execution"}, path, group)
            if group == "base_ablation" and path.name == "summary.json":
                for variant in data.get("variants", []):
                    request = read_json(root / f"{variant['variant']}-request.json") or {}
                    response = variant.get("response", {})
                    prediction({"id": request.get("window", {}).get("window_id", variant["variant"]),
                        "request": request, "result": {**response, "error": response.get("detail"),
                            "total_latency_s": variant.get("elapsed_s")},
                        "target": None, "mode": variant["variant"], "task_type": "prior_state_ablation",
                        "source": "old_prompt_controlled_request"}, path, group)
        for name in ("predictions.jsonl", "requests.jsonl", "generations.jsonl", "generation-validation.jsonl",
                     "engine-outputs.jsonl"):
            for path in sorted(root.rglob(name)):
                # Preserve full row availability separately from the bounded inspector.
                rows = list(read_jsonl(path, limit=100000))
                result["prediction_files"].append({"path": path.relative_to(project).as_posix(),
                    "group": group, "href": href(path), "rows": len(rows),
                    "count_bounded": len(rows) >= 100000, "inspected_rows": min(len(rows), 250),
                    **provenance(path)})
                for item in rows[:250]:
                    prediction(item, path, group)
        for path in sorted(root.rglob("comparisons.jsonl")):
            rows = list(read_jsonl(path, limit=250))
            if rows:
                result["reports"].append({"group": group, "name": path.name,
                    "path": path.relative_to(project).as_posix(), "href": href(path), **provenance(path),
                    "data": {"comparisons": rows, "partial": True,
                             "caveat": "Recorded comparison rows from a partial run; consult interrupted.json. No completed benchmark is inferred."}})
        # Release predictions already carry parsed output/labels. Do not join raw
        # outputs by window ID: counterfactual pairs intentionally reuse that ID.
        for path in sorted(root.rglob("batches.jsonl")):
            for batch in read_jsonl(path, limit=12):
                if batch.get("warmup"):
                    continue
                for item in batch.get("windows", []):
                    prediction({**item, "case": batch.get("case"), "source_kind": batch.get("source_kind"),
                                "mode": f"batch {batch.get('batch_size')}, iteration {batch.get('iteration')}"}, path, group)
    standalone_reports = [("home-assistant-integration.json", "ha_transport"),
                            ("vllm-integration-execution.json", "vllm_integration_execution"),
                            ("release-status.json", "release_status"),
                            ("h100-recovery.json", "h100_recovery"),
                            ("base-422-diagnostic.json", "base_diagnostic")]
    standalone_reports.extend((p.name, "vllm_integration_execution") for p in
                              sorted((project / "reports").glob("vllm-integration-execution-*.json")))
    standalone_reports.extend((p.name, "counterfactual_pairs") for p in
                              sorted((project / "reports").glob("counterfactual-*.json")))
    for filename, group in standalone_reports:
        standalone = project / "reports" / filename
        if not standalone.exists():
            continue
        data = read_json(standalone)
        if data is not None:
            result["reports"].append({"group": group, "name": standalone.name,
                                      "path": standalone.relative_to(project).as_posix(), "href": href(standalone),
                                      **provenance(standalone), "data": data})
        else:
            result["warnings"].append(f"Report unreadable or still copying: {standalone.relative_to(project)}")
    checkpoint_input = project / "artifacts/checkpoint300-evaluation-input-manifest.json"
    if checkpoint_input.is_file():
        data = read_json(checkpoint_input)
        if data is not None:
            result["reports"].append({"group": "checkpoint300_evaluation_input",
                "name": checkpoint_input.name, "path": checkpoint_input.relative_to(project).as_posix(),
                "href": href(checkpoint_input), **provenance(checkpoint_input), "data": data})
    return result


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Home Observer — Evidence review</title><link rel="icon" href="data:,">
<style>
:root{--bg:#f5f7fa;--paper:#fff;--ink:#172137;--muted:#657087;--line:#e3e8f0;--blue:#315cdc;--light:#edf2ff;--green:#15725b;--amber:#946300;--red:#a83e46}*{box-sizing:border-box}body{margin:0;background:var(--bg);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--ink)}a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}button,input,select{font:inherit}button,a,input,select{outline-offset:4px}button{cursor:pointer}.shell{display:grid;grid-template-columns:230px minmax(0,1fr);min-height:100vh}.sidebar{position:sticky;top:0;height:100vh;background:white;border-right:1px solid var(--line);padding:32px 22px;display:flex;flex-direction:column}.brand{display:flex;gap:11px;align-items:center;font-size:16px;font-weight:750;letter-spacing:-.4px}.mark{display:grid;grid-template-columns:repeat(2,9px);gap:3px;padding:9px;border-radius:12px;background:var(--light)}.mark i{width:9px;height:9px;border-radius:2px;background:var(--blue)}.sidebar small{color:var(--muted);display:block;margin:8px 0 32px}.nav{display:grid;gap:7px}.nav button{border:0;background:transparent;text-align:left;padding:11px 14px;border-radius:9px;color:var(--muted);font-weight:600}.nav button.active{background:var(--light);color:var(--blue)}.sidefoot{margin-top:auto;color:var(--muted);font-size:12px}.sidefoot a{display:block;margin-top:10px}.main{max-width:1440px;padding:36px 44px 64px;width:100%;margin:auto}.topline{display:flex;justify-content:space-between;gap:20px;color:var(--muted);font-size:12px;margin-bottom:28px}.eyebrow{text-transform:uppercase;letter-spacing:1.9px;font-size:11px;font-weight:750;color:var(--blue)}h1{font-size:36px;letter-spacing:-1.2px;font-weight:730;margin:8px 0}h2{font-size:20px;letter-spacing:-.4px;margin:0 0 5px}h3{font-size:15px;margin:0 0 9px}p{margin:6px 0 14px}.lead{font-size:15px;max-width:760px;color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin:26px 0}.card{background:var(--paper);border:1px solid var(--line);border-radius:13px;padding:22px;min-width:0}.card .label{font-size:12px;color:var(--muted)}.big{font-size:28px;letter-spacing:-1px;font-weight:700;margin:10px 0 4px}.sub{color:var(--muted);font-size:12px}.two{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);gap:20px;margin:22px 0}.pill{display:inline-block;vertical-align:middle;font-size:11px;font-weight:650;padding:4px 9px;border-radius:20px;background:#edf1f7;color:#586579}.pill.good{background:#e9f6ef;color:var(--green)}.pill.pending{background:#fff4db;color:var(--amber)}.pill.bad{background:#ffedf0;color:var(--red)}.section{display:none}.section.active{display:block}.muted{color:var(--muted)}.note{background:var(--light);border-left:3px solid #7a95ec;border-radius:0 9px 9px 0;padding:14px 17px;color:#3e5078;margin:18px 0;font-size:13px}.pendingbox{padding:35px 22px;border:1px dashed #cfd7e5;border-radius:12px;background:#fafbfd;color:var(--muted);margin:20px 0}.pendingbox b{display:block;color:#41506c;margin-bottom:6px}.statusrow{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:12px 0;border-bottom:1px solid var(--line)}.statusrow:last-child{border:0}.tablewrap{overflow:auto;margin-top:17px}table{width:100%;border-collapse:collapse;text-align:left;font-size:13px;white-space:nowrap}th{font-weight:600;color:var(--muted);padding:11px 14px;background:#f8fafd}td{padding:13px 14px;border-bottom:1px solid var(--line)}td:first-child{font-weight:600}tbody tr:last-child td{border:0}.bars{margin:30px 0 20px}.barrow{display:grid;grid-template-columns:63px 1fr 65px;gap:12px;align-items:center;margin:18px 0}.bartrack{height:18px;border-radius:5px;background:#f0f3f8;overflow:hidden}.bar{height:100%;border-radius:5px;background:#9eadcd}.bar.after{background:var(--blue)}.barval{text-align:right;font-weight:700}.kv{display:grid;grid-template-columns:1fr 1fr;gap:15px;margin-top:22px}.kv div{border-top:1px solid var(--line);padding-top:10px}.kv span{display:block;color:var(--muted);font-size:12px}.kv strong{font-weight:650}.toolbar{display:flex;gap:12px;flex-wrap:wrap;margin:22px 0}input,select{border:1px solid var(--line);background:white;border-radius:9px;padding:10px 12px;color:var(--ink)}input{flex:1;min-width:200px}.inspect{display:grid;grid-template-columns:290px minmax(0,1fr);gap:18px}.list{max-height:690px;overflow:auto;background:white;border:1px solid var(--line);border-radius:12px;padding:7px}.item{display:block;width:100%;text-align:left;border:0;background:transparent;border-radius:8px;padding:12px;white-space:normal;overflow-wrap:anywhere}.item.selected{background:var(--light)}.item span{display:block;font-size:11px;color:var(--muted);margin-top:4px}.item b{font-size:12px;font-weight:600}.previewgrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin:16px 0}.previewgrid figure{margin:0;background:#f4f6fa;border-radius:9px;overflow:hidden}.previewgrid img{width:100%;height:175px;object-fit:contain;display:block;background:#e9edf3}.previewgrid figcaption{font-size:11px;color:var(--muted);padding:7px 10px}.outputs{display:grid;grid-template-columns:1fr 1fr;gap:12px}.outputs>div{min-width:0}pre{font:12px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;margin:8px 0;background:#f5f7fa;border:1px solid var(--line);border-radius:8px;padding:13px;max-height:350px;overflow:auto}.source{display:flex;justify-content:space-between;align-items:center;gap:20px;border-bottom:1px solid var(--line);padding:14px 0}.source a{overflow-wrap:anywhere}.mini{font-size:11px;color:var(--muted)}details{margin:14px 0}summary{cursor:pointer;color:var(--muted);font-size:12px}.foot{margin-top:40px;padding-top:18px;border-top:1px solid var(--line);font-size:12px;color:var(--muted)}.scopechips{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}.emptyimg{height:120px;display:grid;place-items:center;color:var(--muted);font-size:12px;background:#f3f5f9;border-radius:8px}.statuspair{display:flex;gap:8px;align-items:center}.ha-case{padding:16px 0;border-bottom:1px solid var(--line)}@media(max-width:1150px){.main{padding:30px 25px}.grid{grid-template-columns:repeat(2,1fr)}.two{grid-template-columns:1fr}.inspect{grid-template-columns:245px minmax(0,1fr)}}@media(max-width:780px){.shell{display:block}.sidebar{position:relative;height:auto;border-right:0;border-bottom:1px solid var(--line);padding:18px}.sidebar small,.sidefoot{display:none}.nav{display:flex;overflow:auto;margin-top:16px}.nav button{white-space:nowrap;padding:9px 12px}.main{padding:24px 17px}h1{font-size:29px}.inspect{display:block}.list{max-height:230px;margin-bottom:16px}.outputs{grid-template-columns:1fr}.previewgrid img{height:145px}.topline{display:block}.card{padding:18px}.big{font-size:25px}}
</style></head><body>
<div class="shell"><aside class="sidebar"><div class="brand"><span class="mark"><i></i><i></i><i></i><i></i></span>Home Observer</div><small>Experiment evidence</small><nav class="nav" aria-label="Evidence sections"></nav><div class="sidefoot">A local evidence snapshot.<br>No network requests.<a id="readme">Project README ↗</a></div></aside>
<main class="main"><div class="topline"><span>RESEARCH BUILD / EVIDENCE REVIEW</span><span id="timestamp"></span></div><div id="content"></div><div class="foot">Measurements apply to their stated datasets, devices and test windows. Missing reports are shown as pending. A dash indicates an undefined or unavailable metric. No long-duration household reliability or completed endurance run is implied.</div></main></div>
<noscript><p>This offline report uses JavaScript to display embedded evidence.</p></noscript>
<script id="evidence-data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('evidence-data').textContent);
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=(v,n=2)=>typeof v==='number'&&Number.isFinite(v)?v.toLocaleString(undefined,{maximumFractionDigits:n}):'Unavailable';
const pct=v=>typeof v==='number'?num(v*100,1)+'%':'—';
const tag=(label,kind='')=>`<span class="pill ${kind}">${esc(label)}</span>`;
const link=r=>r?`<a href="${esc(r.href)}">Source JSON ↗</a>`:'';
const pending=(title,note)=>`<div class="pendingbox"><b>${esc(title)}</b>${esc(note)}</div>`;
const reports=g=>D.reports.filter(r=>r.group===g);
const find=(g,name)=>reports(g).find(r=>r.name===name);
const training=find('training','training_report.json');
const releaseCandidates=D.reports.filter(r=>r.name==='report.json'&&r.data?.suites);
const folderOf=r=>r.path.slice(0,r.path.lastIndexOf('/'));
const releaseComplete=r=>{if(!r||r.data.complete===false)return false;const folder=folderOf(r),manifest=D.reports.find(m=>m.path===folder+'/manifest.json')?.data;return Array.isArray(manifest?.suites)&&['heldout','counterfactual','public_video'].every(name=>{const s=manifest.suites.find(v=>v.name===name),raw=D.prediction_files.find(p=>p.path===folder+'/'+name+'/predictions.jsonl');return s&&s.selected_rows===s.source_rows&&r.data.suites?.[name]?.windows===s.source_rows&&raw&&!raw.count_bounded&&raw.rows===s.source_rows})};
const release=releaseCandidates.find(releaseComplete)||releaseCandidates[0];
const batches=find('batch','batch_benchmark_report.json')||find('batch','comparisons.jsonl');
const consumer=find('consumer','benchmark_report.json');
const learnedVersions=D.reports.filter(r=>(r.group==='learned_ha'||r.group.startsWith('learned-home-assistant-'))&&r.name==='report.json'&&Array.isArray(r.data?.cases));
const learned=learnedVersions[0];
const v1ha=find('learned-home-assistant-v1','report.json');
const baseAblation=find('base_ablation','summary.json');
const baseFixed=find('base_fixed8','metrics.json');
const baseRuntime=find('base_fixed8','runtime.json');
const vllmFixedReports=D.reports.filter(r=>r.name==='metrics.json'&&(r.group==='vllm_fixed8'||r.group==='trained50_vllm_fixed8'||r.group.startsWith('vllm-fixed8-')));
const vllmUploadReports=D.reports.filter(r=>r.name==='metrics.json'&&(r.group==='vllm_upload_cli'||r.group.startsWith('vllm-upload-cli-')));
const vllmExecutions=reports('vllm_integration_execution');
const vllmLiveReports=D.reports.filter(r=>r.name==='live_report.json'&&(r.group==='live_ha_vllm'||r.group.startsWith('live-home-assistant-vllm-')));
const recovered300=find('checkpoint300_validation','report.json');
const compileReport=find('consumer_compile','report.json');
const recovery=find('h100_recovery','h100-recovery.json');
const merge=find('merge','report.json');
const contextReport=find('context','report.json');
const live=find('live','transport_summary.json');
const selection=find('release_status','release-status.json');
const candidateSelection=find('training_v2_continuation','selection.json');
const policyOrderControl=find('continuation_policy_order','report.json');
const policyOrderVerification=find('policy_order_diagnostic','canonicalization-verification.json');
const allTraining=D.reports.filter(r=>r.name==='training_report.json');
const checkpointFiles=D.reports.filter(r=>r.name==='generation-report.json');
const checkpoints=checkpointFiles.length?checkpointFiles:allTraining.flatMap(r=>(r.data.generation_validation||[]).map(d=>({...r,data:d})));
if(recovered300)checkpoints.push({...recovered300,data:{step:300,checkpoint:'Recovered checkpoint 300 / separate validation run',quality:recovered300.data}});
const isContinuation=r=>r.group==='training_v2_continuation';
const checkpointLabel=r=>isContinuation(r)?'Continuation step '+r.data.step:r.group==='checkpoint300_validation'?'Recovered V2 step '+r.data.step:'V2 step '+r.data.step;
checkpoints.sort((a,b)=>Number(isContinuation(a))-Number(isContinuation(b))||(a.data.step||0)-(b.data.step||0)); 
const transport=reports('ha_transport')[0];
const sections=[['overview','Overview'],['experiments','Experiments'],['quality','Held-out results'],['performance','Hardware & speed'],['predictions','Inspect predictions'],['integration','Home Assistant'],['sources','Source files']];
document.querySelector('.nav').innerHTML=sections.map(([id,title],i)=>`<button type="button" data-tab="${id}" aria-selected="${i===0}" class="${i===0?'active':''}">${title}</button>`).join('');
document.getElementById('timestamp').textContent='Built '+new Date(D.built_at).toLocaleString();document.getElementById('readme').href=D.readme_href;
const title=(eyebrow,h,p)=>`<div class="eyebrow">${esc(eyebrow)}</div><h1>${esc(h)}</h1><p class="lead">${esc(p)}</p>`;
function card(label,value,sub){return `<div class="card"><div class="label">${esc(label)}</div><div class="big">${esc(value)}</div><div class="sub">${esc(sub)}</div></div>`}
function table(headers,rows){return `<div class="tablewrap"><table><thead><tr>${headers.map(x=>`<th>${esc(x)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(x=>`<td>${x}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`}
function runtimeMetadata(r){const rows=D.predictions.filter(p=>p.file.startsWith(folderOf(r)+'/')),unique=key=>[...new Set(rows.map(p=>p.metrics?.[key]).filter(v=>v!==undefined))];const merged=unique('adapter_merged'),adapters=unique('adapter'),backends=unique('backend');const health=r.data.engine_health||{},deployment=r.data.deployment||D.reports.find(m=>m.path===folderOf(r)+'/manifest.json')?.data?.deployment||{};let mode=merged.length?merged.map(v=>v?'Merged adapter':'Unmerged adapter').join(' / '):'Merge mode not recorded';if(adapters.length===1&&adapters[0]===null)mode='Base model; no adapter';if(!adapters.length&&health.adapter_configured===false)mode='Base model; no adapter (deployment config)';if(deployment.adapter_execution==='vllm_native_lora'||(backends.includes('vllm')&&adapters.some(Boolean)))mode='vLLM native LoRA (configured)';else if(adapters.some(Boolean)&&merged.length)mode='Native HF / '+mode.toLowerCase();return {mode,adapters:adapters.filter(Boolean),backends,batch_sizes:unique('batch_size'),context_modes:unique('context_mode'),structured_decode:unique('structured_decode'),engine_reported_models:unique('engine_reported_model'),inspected_records:rows.length,engine_health:health,deployment,execution:r.data.execution,manifest:D.reports.find(m=>m.path===folderOf(r)+'/manifest.json')?.data?.model_config}}
function runtimeBlock(r){const m=runtimeMetadata(r);return `<p>${tag(m.mode)} <span class="mini">${esc(m.adapters.join(', '))}</span></p><details><summary>Runtime identity and configuration evidence</summary><pre>${esc(JSON.stringify(m,null,2))}</pre><p class="mini">Merge flags above come from recorded generation metrics. Manifest settings are requested configuration. Backend, merge state, loading context, schema constraints, policy and prompt changes are distinct experiment conditions. A difference between runs does not isolate its cause.</p></details>`}
function candidateBanner(){if(!candidateSelection)return '';const d=candidateSelection.data,report=D.reports.find(r=>r.path===d.selected_checkpoint+'/generation-report.json');return `<div class="card" style="margin:18px 0"><h2>Training-callback candidate for further evaluation</h2><p>${tag('Candidate only','pending')} ${esc(d.selected_checkpoint||'Candidate path unavailable')}</p><p>${esc(d.status||'Deployment acceptance is separate.')}</p><p class="muted">${esc(d.selection_data||'Consult selection source for its evaluation split.')}</p>${report?runtimeBlock(report):''}<p>${link(candidateSelection)}</p><p class="mini">This training-selection file records callback behavior. Fresh-load parity and deployment acceptance are separate; consult the current canonical status above.</p></div>`}
function policyOrderBlock(){if(!policyOrderControl)return '';const d=policyOrderControl.data,comparisons=d.comparisons||[],verification=policyOrderVerification?.data;return `<div class="card" style="margin:18px 0"><h2>Policy serialization control</h2><p>${tag(comparisons.filter(c=>c.raw_exact_match===true).length+' / '+comparisons.length+' exact raw matches')}</p><p>The fresh-loaded single-request control preserves the training policy field order. Equivalent policy values can produce different text tokens when their JSON key order changes. Earlier reload, merge and batch comparisons used different text inputs and do not isolate merging or batching as the cause.</p>${runtimeBlock(policyOrderControl)}${table(['Example','Raw text matches reference','Semantic / validity matches'],comparisons.map(c=>[esc(c.id),esc(c.raw_exact_match),esc(c.semantic_validity_match)]))}${verification?'<p class="note">Canonical context matched training for '+verification.rows.filter(r=>r.matches_training_context===true).length+' / '+verification.rows.length+' rows; all encoded tensors matched for '+verification.rows.filter(r=>r.all_encoded_tensors_match_training===true).length+' / '+verification.rows.length+'. This control does not establish release accuracy.</p>'+link(policyOrderVerification):''}<p>${link(policyOrderControl)}</p></div>`}
function overview(){const t=training?.data,b=t?.before?.eval_loss,a=t?.after?.eval_loss;const changed=typeof a==='number'&&typeof b==='number'&&b!==0?(b-a)/b:null;
const status=[['V1 adapter training',training,'Report available','good'],['V2 checkpoint generation',checkpoints[0],'Metrics recorded','pending'],['Full release evaluation',release,releaseComplete(release)?'All suite reports present':'Partial suite report',releaseComplete(release)?'good':'pending'],['H100 batching',batches,batches?.data.partial?'Interrupted':'Report available',batches?.data.partial?'bad':'good'],['Consumer GPU benchmark',consumer,'Report available','good'],['Learned HA experiments',learned,learnedVersions.some(r=>r.data.passed===false)?'Failures recorded':'Reports available',learnedVersions.some(r=>r.data.passed===false)?'bad':'good'],['Live cloud capture / upload',live,'Transport recorded','good']];
let html=title('01 / Experiment snapshot','Measured results, including failures.','Training, model responses and execution checks are kept with their original sources. Each result is shown within the scope that was actually tested.');
html+=selectionBanner()+candidateBanner()+policyOrderBlock()+failureSummary()+fixedValidation();
html+=`<div class="grid">${card('V1 model',t?.model_id?.split('/').pop()||'Pending','Revision '+(t?.resolved_revision?.slice(0,12)||'unavailable'))}${card('V1 training examples',num(t?.train_rows,0),(t?num(t.summary_only_rows,0)+' human-caption rows; remaining rows synthetic':'Training report not yet available'))}${card('V1 validation loss',typeof a==='number'?num(a,3):'Pending',typeof b==='number'?'Before adaptation: '+num(b,3):'No measurement available')}${card('Evidence groups',status.filter(x=>x[1]).length+' / '+status.length,'Available report categories; not an acceptance score')}</div>`;
html+=`<div class="two"><div class="card"><h2>V1 learning signal</h2><p class="muted">Held-out validation loss before and after LoRA adaptation.</p>`;
if(t&&typeof b==='number'&&typeof a==='number'){const max=Math.max(a,b,0.01);html+=`<div class="bars"><div class="barrow"><span>Before</span><div class="bartrack"><div class="bar" style="width:${100*b/max}%"></div></div><div class="barval">${num(b,3)}</div></div><div class="barrow"><span>After</span><div class="bartrack"><div class="bar after" style="width:${100*a/max}%"></div></div><div class="barval">${num(a,3)}</div></div></div><p>${tag(num(Math.abs(changed)*100,1)+'% '+(changed>=0?'lower':'higher'),changed>=0?'good':'bad')} <span class="mini">Calculated directly from the two recorded losses.</span></p><div class="kv"><div><span>Optimization steps</span><strong>${num(t.steps,0)}</strong></div><div><span>Training time</span><strong>${num(t.train?.train_runtime,1)} s</strong></div><div><span>Validation rows</span><strong>${num(t.validation_rows,0)}</strong></div><div><span>Trainable parameters</span><strong>${num(t.trainable_parameters,0)}</strong></div></div><div class="note">Lower loss shows better fit to held-out labels. It does not establish reliable actions in real homes.</div>${link(training)}`}else html+=pending('Learning measurements pending','The adapter may be present, but a training report is required to display loss or speed.');
html+=`</div><div class="card"><h2>Evidence availability</h2><p class="muted">A report being present is distinct from its tests passing.</p>${status.map(([name,r,label,kind])=>`<div class="statusrow"><span>${esc(name)}</span>${tag(r?label:'Pending',r?kind:'pending')}</div>`).join('')}<div class="statusrow"><span>Long-duration household run</span>${tag('Not evidenced','pending')}</div><div class="note">Rebuild this file after new artifacts arrive:<br><code>python scripts/build_report.py</code></div></div></div>`;
html+=`<div class="card"><h2>Three evidence scopes</h2><div class="scopechips">${tag('Synthetic: authored actions')}${tag('Natural images: human captions')}${tag('Public video: unscored replay')}</div><p class="muted">Synthetic scenes test explicit rules and drawn occupancy. COCO images carry summary-only supervision. Public EgoLife video exercises real media input and journal flow; its missing action labels are never treated as silence targets.</p></div>`;return html}

function annotationFor(report){if(!report)return null;const folder=report.path.slice(0,report.path.lastIndexOf('/'));return D.reports.find(r=>r.name==='annotation-metrics.json'&&r.path===folder+'/annotation-metrics.json')}
function annotationBlock(d,r){const known=d.known_labels||{},camera=d.camera_occupancy||{};return `<h3 style="margin-top:18px">Known-label observation values</h3><p class="muted">Only annotated entity/attribute pairs have truth labels. Additional predicted facts remain unscored.</p>${table(['Scope','Annotated pairs','Correct','Incorrect','Missing','Coverage','Accuracy when covered'],[['All annotated facts',known],['Room occupied facts',camera]].map(([name,m])=>[esc(name),num(m.annotated_pairs,0),num(m.correct,0),num(m.incorrect,0),num(m.missing,0),pct(m.coverage),pct(m.accuracy_on_covered)]))}<div class="note">${num(d.unscored_extras?.pair_instances,0)} additional predicted pairs are unscored, not established false perception. Typed JSON values must match exactly. ${num(d.original_invalid_or_failed_windows,0)} original windows failed validation or execution checks; parsed fact-value agreement does not override those failures.</div><details><summary>Known-label results by entity</summary>${table(['Entity','Annotated','Correct','Incorrect','Missing','Coverage'],Object.entries(d.per_entity||{}).map(([entity,m])=>[esc(entity),num(m.annotated_pairs,0),num(m.correct,0),num(m.incorrect,0),num(m.missing,0),pct(m.coverage)]))}</details><p class="mini">Legacy strict-set observation scores below count unannotated extras as mismatches and are retained for compatibility only. Room occupancy scores do not themselves establish image dependence. ${link(r)}</p>`}

function metricRows(entries){return table(['Task / condition','Windows','Valid response','Action precision','Action recall','Legacy observation recall','False-action windows','p50 latency'],entries.map(([task,m])=>[esc(task),num(m.windows,0),pct(m.valid_response_rate),pct(m.action_precision),pct(m.action_recall),pct(m.observation_recall),pct(m.false_action_window_rate),num(m.latency_p50_s)+' s']))}
function caseDiagnosis(c){if(c.passed)return 'Case passed';if(c.result?.error)return 'Response rejected / error';if(c.expected_actions?.length&&!c.result?.decision?.actions?.length)return 'Missed required action';return 'Case failed'}
function selectionBanner(){if(!selection)return `<div class="note"><b>No checkpoint selected for deployment.</b> Canonical release-status.json has not been supplied. Training loss and diagnostic results do not select a deployment.</div>`;const d=selection.data,chosen=d.selected_checkpoint;const accepted=Boolean(chosen)&&d.deployment_accepted===true;return `<div class="card" style="margin:18px 0"><h2>${accepted?'Selected for deployment':chosen?'Checkpoint recorded; acceptance pending':'No checkpoint selected for deployment'}</h2><p>${tag(d.status||'Canonical status available',accepted?'good':'pending')} ${esc(chosen||'')}</p><p>${esc(d.reason||'Review the canonical report for acceptance scope.')}</p>${link(selection)}<details><summary>Canonical release status</summary><pre>${esc(JSON.stringify(d,null,2))}</pre></details></div>`}
function failureSummary(){let html='';if(v1ha){const cases=v1ha.data.cases||[],missed=cases.filter(c=>c.expected_actions?.length&&!c.result?.decision?.actions?.length&&!c.result?.error);if(missed.length)html+=`<div class="note" style="background:#fff0f1;border-color:#c65761;color:#7d3038"><b>V1 failed the action checks.</b> It returned no actions for ${missed.length} action-required HA cases. ${cases.filter(c=>c.result?.decision?.noop).length} / ${cases.length} responses were noop. ${link(v1ha)}</div>`;}return html}
function fixedValidation(){if(!baseFixed)return '';const entries=[['Base / corrected prompt',baseFixed.data,baseFixed],...vllmFixedReports.map(r=>['vLLM / '+r.group.replaceAll('_',' '),r.data,r]),...checkpoints.map(r=>[checkpointLabel(r),r.data.quality||{},r])];const hash=baseRuntime?.data.input_sha256;const matching=hash&&checkpoints.length&&checkpoints.every(r=>r.data.validation_sha256===hash)&&vllmFixedReports.every(r=>(r.data.input_sha256||find(r.group,'runtime.json')?.data?.input_sha256)===hash);return `<div class="card" style="margin:22px 0"><h2>Same eight validation examples: base vs adaptation</h2><p class="muted">Four labeled action cases and four labeled no-action cases. This small selection set is used to compare checkpoints; it is not the independent release test set.</p>${table(['Model','Valid','Action precision','Action recall','Missed actions','False actions / quiet','Known-label correct','Source'],entries.map(([name,m,r])=>[esc(name)+'<div class="mini">'+esc(runtimeMetadata(r).mode)+'</div>',pct(m.valid_response_rate),pct(m.action_precision),pct(m.action_recall),num(m.counts?.action_fn,0),tag(num(m.counts?.false_action_windows,0)+' / '+num(m.counts?.no_action_windows,0),(m.counts?.false_action_windows||0)>0?'bad':''),pct(annotationFor(r)?.data?.known_labels?.correct_fraction),link(r)]))}<div class="note">The base result has ${num(baseFixed.data.counts?.action_fp,0)} false action proposals and misses ${num(baseFixed.data.counts?.action_fn,0)} required actions. Compare these errors with the adapters; higher action recall alone can hide over-triggering. No model is accepted from this table.</div><p class="mini">${matching?'Matching input-file SHA-256 verified across the base and listed checkpoint reports.':'Input-file identity is not established for every listed report.'} ${esc(hash||'')} Different runtime hardware may apply; these timing results are not a speed comparison.</p>${link(baseRuntime)}</div>`}
function experiments(){let html=title('02 / Experiment history','Failures remain part of the evidence.','Training loss, response validity, policy decisions and device execution answer different questions. Each version keeps its original report.');html+=selectionBanner()+candidateBanner()+policyOrderBlock()+failureSummary();
html+=fixedValidation();
if(recovery)html+=`<div class="note"><b>H100 training artifacts recovered.</b> ${esc(recovery.data.interruption||'')} Recovery confirms archive integrity; it does not complete interrupted evaluation. ${link(recovery)}</div>`;
if(allTraining.length){html+=`<div class="card"><h2>Training runs</h2>${table(['Run','Training rows','Steps','Before loss','After loss','Training time','Source'],allTraining.map(r=>[esc(r.group)+'<div class="mini">'+esc(r.path)+'</div>',num(r.data.train_rows,0),num(r.data.steps,0),num(r.data.before?.eval_loss,3),num(r.data.after?.eval_loss,3),num(r.data.train?.train_runtime,1)+' s',link(r)]))}<p class="mini">Loss is label fit on its stated validation split. It is not action correctness or deployment acceptance.</p></div>`;}
html+=`<div class="card" style="margin-top:22px"><h2>V2 checkpoint generation checks</h2><p class="muted">Fixed validation examples used for checkpoint selection. Original held-out tests are separate.</p>`;
if(!checkpoints.length)html+=pending('Checkpoint generation metrics pending','No generation-report.json has been copied from training-v2 checkpoint directories.');else html+=table(['Step / checkpoint','Windows','Valid response','Action precision','Action recall','Known-label correct','False action / quiet','Source'],checkpoints.map(r=>{const d=r.data,m=d.quality||{};return [esc(checkpointLabel(r))+'<div class="mini">'+esc(runtimeMetadata(r).mode)+'<br>'+esc(d.checkpoint||r.path)+'</div>',num(m.windows,0),pct(m.valid_response_rate),pct(m.action_precision),pct(m.action_recall),pct(annotationFor(r)?.data?.known_labels?.correct_fraction),tag(num(m.counts?.false_action_windows,0)+' / '+num(m.counts?.no_action_windows,0),(m.counts?.false_action_windows||0)>0?'bad':'pending'),link(r)]}));const missingSteps=[100,200,300,400].filter(step=>!checkpoints.some(r=>!isContinuation(r)&&r.data.step===step));if(missingSteps.length)html+=`<p>${tag('Reports not yet supplied','pending')} Steps ${missingSteps.join(', ')}. No result or completion is inferred.</p>`;for(const checkpoint of D.checkpoint_artifacts||[]){if(!checkpoint.report_present)html+=`<div class="note"><b>${checkpoint.group==='training_v2_continuation'?'Continuation':'Original training-time'} step ${num(checkpoint.step,0)} evaluation incomplete.</b> Adapter weights present: ${esc(checkpoint.weights_present)}. ${num(checkpoint.generation_rows,0)} raw generation rows are available, but no completed generation report. No aggregate quality score is inferred for that interrupted attempt. Any recovered evaluation has its own source report above. <span class="mini">${esc(checkpoint.path)}</span></div>`;}
html+='<p class="mini">A dash means undefined or unavailable; precision is undefined when no positive actions are proposed. False actions are counted among labeled no-action windows. Valid JSON alone does not establish decision correctness; a checkpoint is selected only by the canonical release-status report.</p></div>';
if(baseAblation){const d=baseAblation.data;html+=`<div class="card" style="margin-top:22px"><h2>Base model with the old prompt: citation failure</h2>${tag('Controlled diagnostic, not an accepted deployment','pending')}<p>${esc(d.finding||'')}</p><p class="muted">${esc(d.scope||'')}</p>${table(['Prior-state condition','HTTP status','Elapsed','Recorded result'],(d.variants||[]).map(v=>[esc(v.variant),num(v.status_code,0),num(v.elapsed_s)+' s',esc(v.response?.detail||('Proposed '+(v.response?.decision?.actions?.length??0)+' action(s); no execution in this diagnostic.'))]))}<div class="note">${esc(d.interpretation||'')}</div>${link(baseAblation)}</div>`;}
if(merge){html+=`<div class="card" style="margin-top:22px"><h2>V1 merge experiment</h2><p>${esc(merge.data.parity_scope||'')}</p>${table(['Source ID','Greedy output identical','Unmerged','Merged'],(merge.data.comparisons||[]).map(c=>[esc(c.id),esc(c.raw_greedy_exact_match),num(c.unmerged_latency_s)+' s',num(c.merged_latency_s)+' s']))}<div class="note">Output parity checks consistency between implementations. V1 still missed required actions on the measured diagnostic slice.</div>${link(merge)}</div>`;}return html}
function quality(){let html=title('03 / Generalization checks','Held-out results by task.','Small diagnostic slices, full release suites and unscored public video are labeled separately. Checkpoint-selection validation is shown in Experiments.');
const releases=D.reports.filter(r=>r.name==='report.json'&&r.data?.suites);
if(!releases.length)html+=pending('Full release suite pending','No full release evaluation report is present. The measured diagnostic slices below do not substitute for a completed release suite.');
for(const [group,path] of Object.entries(D.groups).filter(([_,path])=>path.includes('release-'))){const partialReports=reports(group),files=D.prediction_files.filter(p=>p.group===group);if(!releases.some(r=>r.group===group)&&(partialReports.length||files.length)){const manifest=partialReports.find(r=>r.name==='manifest.json');html+=`<div class="card" style="margin-top:22px"><h2>Release evaluation: report incomplete</h2><p class="mini">${esc(path)}</p>${tag('No completed aggregate report','pending')}<p>Source files or prediction rows are available, but full evaluation completion is not established.</p>${manifest?runtimeBlock(manifest):''}${table(['Prediction file','Readable rows','Inspected rows'],files.map(p=>[esc(p.path),num(p.rows,0)+(p.count_bounded?' (capped)':''),num(p.inspected_rows,0)]))}${manifest?link(manifest):''}</div>`;}}
for(const releaseReport of releases){for(const [name,suite] of Object.entries(releaseReport.data.suites)){html+=`<div class="card" style="margin-top:22px"><h2>${esc(name.replaceAll('_',' '))}</h2><p class="mini">${esc(releaseReport.path)}</p><p>${tag(releaseComplete(releaseReport)?'Full suite files present':'Partial release evidence',releaseComplete(releaseReport)?'good':'pending')}</p>${runtimeBlock(releaseReport)}<p class="muted">${esc(suite.accuracy_interpretation||suite.scope||'See original report for evaluation scope.')}</p>`;
const annotations=annotationFor(releaseReport)?.data?.suites?.[name];if(annotations)html+=annotationBlock(annotations,annotationFor(releaseReport));const tasks=Object.entries(suite.by_task_type||{overall:suite});html+=metricRows(tasks);if(!releaseComplete(releaseReport))html+=`<p>${tag('Partial release report','pending')} Matching full source-row counts for all required suites are not yet evidenced by the manifest, reports and local prediction files. A bounded slice or incomplete run is not full-release completion.</p>`;
if(name.includes('counter'))html+=`<div class="note">Each empty/occupied pair has identical model-visible text, times and device states. Only image pixels differ. These are per-window results; separately sourced pair-level checks appear below when supplied.</div>`;
if(name.includes('public'))html+=`<div class="note">No ground-truth action labels are available. Schema validity and timing do not measure perception accuracy.</div>`;
if(tasks.some(([task])=>task.includes('summary')))html+=`<p class="mini">Natural-image summaries require caption/reference review. Unsupported action and observation fields are masked.</p>`;html+=link(releaseReport)+'</div>';}}
for(const r of reports('counterfactual_pairs')){const d=r.data,source=D.prediction_files.find(p=>p.sha256===d.predictions_sha256);html+=`<div class="card" style="margin-top:22px"><h2>Counterfactual pair checks</h2><p class="mini">${esc(r.path)}</p><p>${tag(num(d.passed_pairs,0)+' / '+num(d.pairs,0)+' pairs passed',d.passed_pairs===d.pairs?'good':'bad')}</p><p>${esc(d.scope||'')}</p>${table(['Pair','Identical text','Identical audio','Changed camera','Pair passed'],(d.results||[]).map(v=>[esc(v.pair_id),esc(v.identical_model_visible_text),esc(v.identical_audio),esc((v.changed_cameras||[]).join(', ')),esc(v.passed)]))}<details><summary>Member-level checks and scoring provenance</summary><pre>${esc(JSON.stringify(d,null,2))}</pre></details><p class="mini">${source?'Prediction-file SHA-256 matches a local source.':'Matching prediction source is not locally established.'} ${source?'<a href="'+esc(source.href)+'">Predictions ↗</a>':''}</p>${link(r)}</div>`;}
if(merge?.data.snapshot_quality)html+=`<div class="card" style="margin-top:22px"><h2>V1 merged adapter: diagnostic slice</h2>${tag('Missed required actions','bad')}<p class="muted">Four source windows only. Valid JSON and faster inference did not establish useful decisions.</p>${metricRows([['merged V1 snapshot',merge.data.snapshot_quality]])}${link(merge)}</div>`;
if(contextReport)html+=`<div class="card" style="margin-top:22px"><h2>V1 prior-state comparison</h2><p class="muted">Same four-window diagnostic slice under two context conditions. These are diagnostic inputs, including duplicated current telemetry.</p>${metricRows(Object.entries(contextReport.data).filter(([_,m])=>m&&typeof m.windows==='number'))}${link(contextReport)}</div>`;
if(baseFixed)html+=`<div class="card" style="margin-top:22px"><h2>Base model: selection-validation task breakdown</h2>${tag('Eight validation examples; not release test','pending')}<p class="muted">${esc(baseFixed.data.context_mode||'')}</p>${annotationFor(baseFixed)?annotationBlock(annotationFor(baseFixed).data,annotationFor(baseFixed)):''}${metricRows(Object.entries(baseFixed.data.by_task_type||{}))}<div class="note">Device-rule success does not establish visual occupancy understanding. Keep the task-specific failures visible.</div>${link(baseFixed)}</div>`;
for(const [label,r] of [...vllmFixedReports.map(r=>['vLLM fixed validation8 / '+r.group,r]),...vllmUploadReports.map(r=>['vLLM CLI upload evaluation / '+r.group,r])]){if(r)html+=`<div class="card" style="margin-top:22px"><h2>${esc(label)}</h2>${runtimeBlock(r)}<p class="muted">${esc(r.data.scope||'Recorded evaluation; completion is limited to the rows in this report.')}</p>${annotationFor(r)?annotationBlock(annotationFor(r).data,annotationFor(r)):''}${metricRows(Object.entries(r.data.by_task_type||{overall:r.data}))}<p class="mini">${esc(r.path)}. Backend/decoding settings and original predictions remain separate source evidence.</p>${link(r)}</div>`;}
return html}
function performance(){let html=title('04 / Runtime measurements','Latency and capacity, separately.','Consumer hardware results remain distinct from H100 batching. Batch throughput is not per-event responsiveness.');
const hardware=D.reports.filter(r=>['hardware.json','gpu.json'].includes(r.name));const costs=D.reports.filter(r=>['cost_report.json','costs.json'].includes(r.name));
html+=`<div class="grid">${card('Hardware inventory',hardware.length?'Recorded':'Unavailable',hardware.length?'See source files for device details':'No hardware-inventory report supplied')}${card('Measured costs',costs.length?'Recorded':'Unavailable',costs.length?'See original cost report; no cost extrapolation':'No cost report supplied')}${card('Consumer cases',consumer?num(consumer.data.cases?.length,0):'Pending','Measured cases only')}${card('H100 comparisons',batches?num(batches.data.comparisons?.length,0):'Pending','Ready-request batching benchmark')}</div>`;
if(live){const d=live.data,c=d.capture_report||{};html+=`<div class="card" style="margin:22px 0"><h2>Live cloud transport: V1</h2>${tag('Transport evidence only','pending')}<p class="muted">${esc(d.scope||'')}</p><div class="grid">${card('Captured run',num(d.actual_elapsed_s,2)+' s','Requested '+num(d.requested_duration_s,0)+' seconds')}${card('Model windows',num(d.windows,0),num(d.windows_with_audio,0)+' with audio')}${card('Observed throughput',num(c.actual_windows_per_s,3)+' / s','Requested '+num(c.requested_windows_per_s,1)+' / s')}${card('Mean round trip',num(d.roundtrip_s_mean,2)+' s','Maximum '+num(d.roundtrip_s_max,2)+' s')}</div>${table(['Camera feed','Captured frames','Model sampled','Not model sampled'],Object.entries(c.cameras||{}).map(([name,m])=>[esc(name),num(m.captured,0),num(m.model_sampled,0),num(m.not_model_sampled,0)]))}<p>Uploaded bytes matched received bytes: <b>${esc(d.uploaded_equals_received_per_window)}</b>. Recorded errors: <b>${num(d.all_results_error_count,0)}</b>. Remaining matching capture processes: <b>${Array.isArray(d.remaining_matching_capture_processes)?d.remaining_matching_capture_processes.length:'Unavailable'}</b>.</p><div class="note">${num(c.snapshot_retention?.expired_windows,0)} snapshot windows expired; ${num(c.snapshot_retention?.retained_completed_windows,0)} remain. ${num(d.noop_windows,0)} model responses were noop; unscored public footage does not establish whether those decisions were correct.</div>${link(live)}</div>`;}
if(compileReport){const d=compileReport.data;html+=`<div class="card" style="margin:22px 0"><h2>Consumer HF cache / compilation experiment</h2>${tag(d.adopt_optimization===true?'Optimization adopted in report':'Optimization not adopted',d.adopt_optimization===true?'good':'bad')}<p class="muted">${esc(d.scope||'')}</p>${table(['Phase','Case','Measurement condition','Completed','Wall time','Exact greedy parity'],(d.records||[]).map(r=>[esc(r.phase),esc(r.case),esc(r.temperature),esc(r.completed),num(r.wall_s)+' s',typeof r.greedy_output_matches_baseline==='boolean'?esc(r.greedy_output_matches_baseline):'Not measured']))}<div class="note">${esc(d.conclusion||'')} ${esc(d.cold_baseline_caveat||'')}</div>${link(compileReport)}</div>`;}
for(const vllmLive of vllmLiveReports){const d=vllmLive.data,liveValidation=D.reports.find(r=>r.group===vllmLive.group&&r.name==='validation.json'),liveRows=D.predictions.filter(p=>p.group===vllmLive.group),sourceProvenance=D.reports.find(r=>r.group===vllmLive.group&&['provenance.json','source-provenance.json'].includes(r.name));html+=`<div class="card" style="margin:22px 0"><h2>Live capture / isolated HA: vLLM</h2><p class="mini">${esc(vllmLive.path)}</p>${tag('Bounded run; no household accuracy labels','pending')}<p>${esc(d.scope||'')}</p><div class="grid">${card('Elapsed',num(d.elapsed_s)+' s','Measured run duration')}${card('Processed windows',num(d.model_windows,0),'Actual reported count')}${card('Observed throughput',num(d.actual_windows_per_s,3)+' / s','Requested '+num(d.requested_windows_per_s,1)+' / s')}${card('Errors',num(d.error_count,0),'See original report for capture and inference errors')}</div>${table(['Camera routing ID','Captured','Model sampled','Not sampled'],Object.entries(d.cameras||{}).map(([name,m])=>[esc(name),num(m.captured,0),num(m.model_sampled,0),num(m.not_model_sampled,0)]))}<div class="note">Camera names are routing labels. Source provenance determines whether footage is simultaneous or temporally offset; four inputs alone do not establish multiview coverage. Transport throughput does not establish decision quality or physical-household control.</div><p><b>${liveRows.filter(p=>p.rejections?.length).length} of ${liveRows.length} inspected prediction rows have recorded policy or validation rejections.</b> These are separate from the transport error count. The inspector is bounded to its stated source-file limit.</p>${liveValidation?'<div class="note"><b>'+num(liveValidation.data.actual_ha_verified_action_count??liveValidation.data.verified_action_calls,0)+' independently verified HA action calls.</b> '+esc(liveValidation.data.interpretation||liveValidation.data.scope||'Read the exact protocol checks below.')+'</div><details><summary>Recorded live protocol validation</summary><pre>'+esc(JSON.stringify(liveValidation.data,null,2))+'</pre>'+link(liveValidation)+'</details>':''}${sourceProvenance?'<p class="note">'+esc(sourceProvenance.data.scope||'See exact source scope below.')+'</p><details><summary>Exact input source provenance</summary><pre>'+esc(JSON.stringify(sourceProvenance.data,null,2))+'</pre>'+link(sourceProvenance)+'</details>':''}${link(vllmLive)}</div>`;}
html+=`<div class="card"><h2>Consumer sustained benchmark</h2><p class="muted">End-to-end service time, sustained observed throughput and actual encoded inputs.</p>`;
if(!consumer)html+=pending('Full consumer benchmark pending','The short cache diagnostic above is separate. No H100 number substitutes for a sustained consumer benchmark.');else{html+=table(['Case / data','Windows','p50 / p95 service','Windows / second','Peak memory','Audio tokens','Pending backlog','Source pace'],(consumer.data.cases||[]).map(c=>[esc(c.case)+'<div class="mini">'+esc(c.source_kind)+'</div>',num(c.completed_inference_windows,0),num(c.latency_s?.p50)+' / '+num(c.latency_s?.p95)+' s',num(c.observed_windows_per_second,3),num(c.max_peak_allocated_gb)+' GB',num(c.encoded_audio_tokens,0),num(c.pending_arrived_windows,0),tag(c.can_keep_pace?'Kept pace in this run':'Not established',c.can_keep_pace?'good':'pending')]));html+=`<div class="note">${esc(consumer.data.interpretation||'Measured replay; no indefinite-state claim.')}</div>${link(consumer)}`;}html+='</div>';
html+=`<div class="card" style="margin-top:22px"><h2>H100 batch comparison</h2><p class="muted">Each event waits for the whole batch to finish; amortized seconds show capacity only.</p>`;
if(batches?.data.partial)html+=`<p>${tag('Partial run; interrupted','bad')}</p>`;
if(!batches)html+=pending('Batch comparison pending','Measured comparisons will appear when batch_benchmark_report.json is present.');else{html+=table(['Case','Batch size','Windows / second','Event p50','Amortized / window','Peak allocated','Invalid windows'],(batches.data.comparisons||[]).map(c=>[esc(c.case),num(c.batch_size,0),num(c.observed_windows_per_second,3),num(c.per_event_service_latency_s?.p50)+' s',num(c.amortized_seconds_per_window)+' s',num(c.max_peak_allocated_gb)+' GB',num(c.invalid_windows,0)]));html+=`<div class="note">${esc(batches.data.caveat||'Ready independent requests; not a live deadline guarantee.')}</div>${link(batches)}`;}return html+'</div>'}
function predictions(){return title('05 / Inspect the evidence','See the inputs and responses.','Filter recorded predictions by task or search their source IDs. Missing local images remain unavailable; no replacement images are generated.')+`<div class="toolbar"><input id="search" type="search" placeholder="Search ID, task, source or response…" aria-label="Search predictions"><select id="taskfilter" aria-label="Filter task"><option value="">All tasks</option>${[...new Set(D.predictions.map(p=>p.task))].sort().map(t=>`<option>${esc(t)}</option>`).join('')}</select><span id="shown" class="muted"></span></div><div class="inspect"><div class="list" id="predictionlist"></div><div class="card" id="predictiondetail"></div></div>`}
function integration(){let html=title('06 / Device execution','Home Assistant evidence, by experiment.','The real Home Assistant server executes REST calls against isolated fixture entities backed by input_boolean. These tests do not control physical household appliances or validate a real home.');
for(const vllmExecution of vllmExecutions){html+=`<div class="card"><h2>vLLM integration execution evidence</h2><p>${tag('Recorded probe; deployment status remains separate','pending')}</p><p>${esc(vllmExecution.data.scope||'Consult the source for exact requests, Home Assistant fixture entities and verification scope.')}</p><pre>${esc(JSON.stringify(vllmExecution.data,null,2))}</pre>${link(vllmExecution)}</div>`;}
if(transport){const d=transport.data;html+=`<div class="card"><h2>REST execution & independent readback</h2><p>${tag(d.status==='passed'?'Transport checks passed':d.status||'Report available',d.status==='passed'?'good':'pending')}</p><p>${esc(d.method||'')}</p><p class="muted">${esc(d.device_kind||'')}</p><div class="kv"><div><span>HA version</span><strong>${esc(d.home_assistant_version||'Unavailable')}</strong></div><div><span>Recorded service calls</span><strong>${num(d.service_calls?.length,0)}</strong></div></div><p style="margin-top:18px">${link(transport)}</p></div>`}
if(!learnedVersions.length)html+=pending('Learned execution evidence pending','A successful REST transport check alone does not verify learned decisions.');
for(const r of learnedVersions){const d=r.data;html+=`<div class="card" style="margin-top:22px"><h2>${esc(r.group.replace('learned-home-assistant-','').replaceAll('-',' '))}</h2><p>${tag(d.passed===true?'Reported cases passed':d.passed===false?'Experiment failed':'Report available',d.passed===true?'good':d.passed===false?'bad':'pending')} <span class="mini">${d.cases.filter(c=>c.passed).length} / ${d.cases.length} recorded cases passed</span></p><p class="muted">${esc(d.scope||'')}</p>${runtimeBlock(r)}${d.cases.map((c,i)=>`<div class="ha-case"><div class="statuspair"><b>Case ${i+1}</b>${tag(caseDiagnosis(c),c.passed?'good':'bad')}</div><p class="mini">${esc(c.id)}</p><div>Duplicate replay skipped: ${esc(c.duplicate_replay_skipped)}</div><div class="outputs"><div><h3>Expected actions</h3><pre>${esc(JSON.stringify(c.expected_actions,null,2))}</pre></div><div><h3>Proposed / executed</h3><pre>${esc(JSON.stringify({decision:c.result?.decision,execution:c.result?.actions,error:c.result?.error},null,2))}</pre></div></div></div>`).join('')}<p>${link(r)}</p></div>`}
const pendingDirs=Object.entries(D.groups).filter(([g])=>g.startsWith('learned-home-assistant-')&&!reports(g).some(r=>r.name==='report.json'));
if(pendingDirs.length)html+=pending('Further HA experiments pending',pendingDirs.map(([g])=>g).join(', '));return html}
function sources(){let html=title('07 / Reproducibility','Source files behind this page.','Every displayed measurement comes from the files listed here. This HTML embeds a snapshot of their data and selected local thumbnails. The inspector includes up to 250 rows per prediction file and 12 batch records; report metrics retain their original full evaluation scope.');html+=`<div class="card"><h2>Measured reports</h2>${D.reports.map(r=>`<div class="source"><div><a href="${esc(r.href)}">${esc(r.path)} ↗</a><div class="mini">${esc(r.group)} · ${num(r.bytes,0)} bytes · ${esc(r.modified_at||'')}</div><div class="mini" style="overflow-wrap:anywhere">SHA-256 ${esc(r.sha256||'unavailable')}</div></div>${tag('Available','good')}</div>`).join('')||'<p>No reports available.</p>'}</div>`;
html+=`<div class="card" style="margin-top:22px"><h2>Not yet supplied</h2>${Object.entries(GROUPS_LABELS).filter(([g])=>!reports(g).length).map(([g,label])=>`<div class="statusrow"><span>${esc(label)}<div class="mini">${esc(D.groups[g])}</div></span>${tag('Pending','pending')}</div>`).join('')||'<p>All configured artifact groups contain at least one report. Review their outcomes and scope.</p>'}<div class="statusrow"><span>Consumer hardware inventory / billed cost</span>${tag('Shown only if source reports exist')}</div></div>`;
html+=`<div class="card" style="margin-top:22px"><h2>Prediction file provenance</h2><p><a href="${esc(D.evaluation_guide_href)}">Evaluation guide ↗</a></p>${(D.prediction_files||[]).map(r=>`<div class="source"><div><a href="${esc(r.href)}">${esc(r.path)} ↗</a><div class="mini">${r.rows} readable rows${r.count_bounded?' (count capped)':''}; ${r.inspected_rows} included in inspector</div><div class="mini" style="overflow-wrap:anywhere">SHA-256 ${esc(r.sha256)}</div></div></div>`).join('')}</div>`;
if(D.datasets.length)html+=`<div class="card" style="margin-top:22px"><h2>Dataset provenance</h2>${D.datasets.map(d=>`<details><summary>${esc(d.name)}</summary><a href="${esc(d.href)}">Original manifest ↗</a><p class="mini" style="overflow-wrap:anywhere">SHA-256 ${esc(d.sha256)}</p><pre>${esc(JSON.stringify(d.data,null,2))}</pre></details>`).join('')}</div>`;
if(D.warnings.length)html+=`<div class="note">${D.warnings.map(esc).join('<br>')}</div>`;return html}
const GROUPS_LABELS={training:'V1 training',training_v2:'V2 training / checkpoints',release:'V1 release evaluation',release_v2:'V2 release evaluation',batch:'H100 batch benchmark',consumer:'Consumer benchmark',merge:'V1 adapter merge',context:'V1 context comparison',live:'Live cloud transport',base_ablation:'Base-model prior-state ablation',base_fixed8:'Base model fixed validation8',consumer_compile:'Consumer cache / compile diagnostic',checkpoint300_validation:'Recovered checkpoint300 validation',release_base_v2_policy:'Base model full release / V2 policy',release_checkpoint300_v2_policy:'Checkpoint300 full release / V2 policy',release_continuation_selected:'Selected continuation candidate release',release_continuation_merged_rejected:'Rejected merged continuation experiment',release_trained50_vllm_canonical:'Trained50 native-LoRA vLLM full release',trained50_vllm_fixed8:'Trained50 native-LoRA vLLM fixed8',vllm_lora50:'vLLM registered LoRA response evidence',continuation_policy_order:'Continuation policy-order control',policy_order_diagnostic:'Policy-order tensor diagnostic',base_batch_parity:'Base single / batch control',training_v2_continuation:'V2 training continuation',vllm_fixed8:'vLLM fixed validation8',vllm_upload_cli:'vLLM CLI upload evaluation',live_ha_vllm:'vLLM live HA demo',"learned-home-assistant-vllm":'vLLM learned HA demo'};
document.getElementById('content').innerHTML=sections.map(([id])=>`<section id="${id}" class="section ${id==='overview'?'active':''}">${({overview,experiments,quality,performance,predictions,integration,sources})[id]()}</section>`).join('');
document.querySelectorAll('[data-tab]').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('[data-tab]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-selected',b===button)});document.querySelectorAll('.section').forEach(s=>s.classList.toggle('active',s.id===button.dataset.tab));window.scrollTo({top:0,behavior:'smooth'})}));
let selected=0;function detail(index){const p=D.predictions[index];if(!p){document.getElementById('predictiondetail').innerHTML=(D.predictions.length?pending('No matching predictions','Change the search or task filter to inspect another recorded example.'):pending('Predictions pending','Prediction JSONL files have not yet been supplied.'));return}selected=index;document.querySelectorAll('[data-pred]').forEach(b=>b.classList.toggle('selected',Number(b.dataset.pred)===index));
const ok=p.decision&&!p.error&&!p.rejections?.length;let html=`<div class="statuspair">${tag(p.task)}${tag(ok?'Parsed response':p.error?'Error recorded':p.rejections?.length?'Rejected response':'Raw / unavailable',ok?'good':p.error||p.rejections?.length?'bad':'pending')}</div><h3 style="margin-top:16px;overflow-wrap:anywhere">${esc(p.id)}</h3><p class="mini">${esc(p.source)} · ${esc(p.group)} ${esc(p.variant)} · ${num(p.latency_s??p.metrics.latency_s)} s</p>`;
html+=p.frames.length?`<div class="previewgrid">${p.frames.map(f=>`<figure>${f.media?`<img loading="lazy" src="${D.media[f.media]}" alt="${esc(f.camera)} input frame">`:'<div class="emptyimg">Local image unavailable</div>'}<figcaption>${esc(f.camera)}<br>${esc(f.evidence_id||'')}</figcaption></figure>`).join('')}</div>`:'<div class="emptyimg">No local image resolved for this prediction</div>';
html+=`<div class="outputs"><div><h3>Recorded model output</h3><pre>${esc(p.decision?JSON.stringify(p.decision,null,2):(p.raw||p.error||'No output supplied'))}</pre></div><div><h3>Source target / expectations</h3><pre>${esc(p.target?JSON.stringify(p.target,null,2):'No target supplied. This example is unscored.')}</pre></div></div>`;
if(p.supervision_mask?.summary===true&&p.supervision_mask?.actions===false)html+=`<div class="note">Summary-only supervision. Missing actions, observations and noop labels are not negative targets.</div>`;
if(p.error||p.rejections?.length)html+=`<div class="note">${esc(p.error||p.rejections.join('; '))}</div>`;
if(p.raw)html+=`<details><summary>Original raw generation text</summary><pre>${esc(p.raw)}</pre></details>`;
html+=`<details><summary>Metrics, causal context & source provenance</summary><pre>${esc(JSON.stringify({metrics:p.metrics,device_states:p.device_states,audio_present:p.has_audio,supervision_mask:p.supervision_mask,split:p.split,group_id:p.group_id,input_context_source:p.input_context_source,prior_state:p.prior_state,media_integrity:p.media_integrity,provenance:p.provenance},null,2))}</pre></details><a href="${esc(p.href)}">Original prediction file ↗</a>`;document.getElementById('predictiondetail').innerHTML=html}
function filter(){const q=document.getElementById('search').value.toLowerCase(),task=document.getElementById('taskfilter').value;const hits=D.predictions.map((p,i)=>({p,i})).filter(({p})=>(!task||p.task===task)&&(!q||JSON.stringify(p).toLowerCase().includes(q)));document.getElementById('shown').textContent=hits.length+' examples';document.getElementById('predictionlist').innerHTML=hits.map(({p,i})=>`<button class="item ${selected===i?'selected':''}" data-pred="${i}"><b>${esc(p.id)}</b><span>${esc(p.task)} · ${esc(p.group)} ${esc(p.variant)}</span></button>`).join('')||'<p class="muted" style="padding:12px">No matching predictions.</p>';document.querySelectorAll('[data-pred]').forEach(b=>b.addEventListener('click',()=>detail(Number(b.dataset.pred))));if(hits.length)detail(hits.some(h=>h.i===selected)?selected:hits[0].i);else detail(-1)}
document.getElementById('search').addEventListener('input',filter);document.getElementById('taskfilter').addEventListener('change',filter);filter();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    output = args.output.resolve() if args.output else project / "reports" / "index.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    data = collect(project, output)
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    output.write_text(HTML.replace("__DATA__", encoded), encoding="utf-8")
    print(json.dumps({"output": str(output), "reports": len(data["reports"]),
                      "predictions": len(data["predictions"]), "embedded_images": len(data["media"]),
                      "pending_groups": [g for g in GROUPS if not any(r["group"] == g for r in data["reports"])]}, indent=2))


if __name__ == "__main__":
    main()
