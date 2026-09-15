#!/usr/bin/env python3
"""Start and verify an isolated Home Assistant with template devices only."""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
# Support direct execution from a source checkout before an editable install.
from home_observer.ha import HomeAssistant  # noqa: E402
from home_observer.schema import Action  # noqa: E402

LABEL = "home-observer.isolated-ha-demo"
CONTAINER = "home-observer-ha-demo-20260915"
NETWORK = "home-observer-ha-demo-net-20260915"
PORT = 8124
BASE_URL = f"http://127.0.0.1:{PORT}"
DEFAULT_WORK = PROJECT.parents[1] / "work" / "ha-demo"
ENTITIES = ["light.kitchen", "light.entry", "light.lounge", "light.garage", "switch.buzzer"]


def private_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def docker(*args, check=True, stdin=None, timeout=60):
    result = subprocess.run(["docker", *args], input=stdin, capture_output=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f"Docker {args[0]} failed with exit {result.returncode}")
    return result


def inspect_owned(kind, name):
    result = docker(kind, "inspect", name, check=False)
    if result.returncode:
        return None
    obj = json.loads(result.stdout)[0]
    labels = obj.get("Config", {}).get("Labels", {}) if kind == "container" else obj.get("Labels", {})
    if labels.get(LABEL) != "true":
        raise RuntimeError(f"Name collision: {name} is not this project's isolated demo")
    return obj


def ensure_demo(work):
    work.mkdir(parents=True, exist_ok=True)
    work.chmod(0o700)
    config_dir = work / "config"
    config_dir.mkdir(exist_ok=True)
    config_dir.chmod(0o700)
    settings = json.loads((PROJECT / "configs/ha-demo.json").read_text())
    existing = inspect_owned("container", CONTAINER)
    if existing:
        if str(config_dir.resolve()) + ":/config" not in existing["HostConfig"]["Binds"]:
            raise RuntimeError("Existing demo uses a different work directory; use its recorded work directory")
        if not existing["State"]["Running"]:
            docker("start", CONTAINER)
        return
    architecture = docker("info", "--format", "{{.Architecture}}").stdout.decode().strip()
    key = "arm64" if architecture in ("aarch64", "arm64") else "amd64"
    image = settings["images"][key]
    config = json.loads((PROJECT / "configs/ha-demo-configuration.json").read_text())
    (config_dir / "configuration.yaml").write_text(json.dumps(config, indent=2) + "\n")
    if not inspect_owned("network", NETWORK):
        # A private bridge supports Docker VM localhost publishing; no host networking,
        # multicast discovery, hardware devices, or privileged mounts are enabled.
        docker("network", "create", "--label", LABEL + "=true", NETWORK)
    # Pull can take minutes; output is redirected to an ordinary task log without credentials.
    with (work / "image-pull.log").open("w") as log:
        pulled = subprocess.run(["docker", "pull", image], stdout=log, stderr=subprocess.STDOUT, timeout=600)
    if pulled.returncode:
        raise RuntimeError("Home Assistant image pull failed; see image-pull.log")
    docker("run", "-d", "--name", CONTAINER, "--label", LABEL + "=true",
           "--network", NETWORK, "--publish", f"127.0.0.1:{PORT}:8123",
           "--restart", "no", "--stop-timeout", "60", "--memory", "2g", "--cpus", "2",
           "--env", "TZ=UTC", "--volume", str(config_dir.resolve()) + ":/config", image,
           timeout=90)
    private_json(work / "instance.json", {"container": CONTAINER, "network": NETWORK,
                 "url": BASE_URL, "image": image, "version": settings["version"], "isolated": True})


def wait_ready(timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = httpx.get(BASE_URL + "/api/onboarding", timeout=5)
            if response.status_code in (200, 401):
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise RuntimeError("Home Assistant readiness deadline expired")


def check_response(response):
    if not 200 <= response.status_code < 300:
        raise RuntimeError(f"Home Assistant request failed with HTTP {response.status_code}; response body omitted")
    return response.json()


def onboard(work):
    credential_path = work / "credentials.json"
    if credential_path.exists():
        creds = json.loads(credential_path.read_text())
        if creds.get("long_lived_token"):
            return creds
    else:
        creds = {"username": "homeobserver_demo", "password": secrets.token_urlsafe(32), "client_id": BASE_URL + "/"}
        private_json(credential_path, creds)
    with httpx.Client(base_url=BASE_URL, timeout=30) as client:
        steps = check_response(client.get("/api/onboarding"))
        done = {item["step"] for item in steps if item["done"]}
        if "user" not in done:
            result = check_response(client.post("/api/onboarding/users", json={
                "name": "Home Observer Demo", "username": creds["username"], "password": creds["password"],
                "client_id": creds["client_id"], "language": "en"}))
            tokens = check_response(client.post("/auth/token", data={"grant_type": "authorization_code",
                "code": result["auth_code"], "client_id": creds["client_id"]}))
            creds.update(tokens)
            private_json(credential_path, creds)
        elif creds.get("refresh_token"):
            tokens = check_response(client.post("/auth/token", data={"grant_type": "refresh_token",
                "refresh_token": creds["refresh_token"], "client_id": creds["client_id"]}))
            creds.update(tokens)
            private_json(credential_path, creds)
        else:
            raise RuntimeError("Onboarding already started without saved refresh credentials; use the saved demo login")
        client.headers["Authorization"] = "Bearer " + creds["access_token"]
        for step in ("core_config", "analytics"):
            if step not in done:
                check_response(client.post("/api/onboarding/" + step, json={}))
        if "integration" not in done:
            check_response(client.post("/api/onboarding/integration", json={
                "client_id": creds["client_id"], "redirect_uri": creds["client_id"] + "?auth_callback=1"}))
    # Use Home Assistant's installed aiohttp client for the documented WebSocket auth flow.
    # The short-lived token is sent over docker exec stdin, never a process argument or log.
    ws_script = """import asyncio, json, sys, aiohttp
async def main():
    token = json.load(sys.stdin)['token']
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect('http://127.0.0.1:8123/api/websocket') as ws:
            await ws.receive_json()
            await ws.send_json({'type':'auth','access_token':token})
            if (await ws.receive_json())['type'] != 'auth_ok': raise RuntimeError('auth failed')
            await ws.send_json({'id':1,'type':'auth/long_lived_access_token','lifespan':30,'client_name':'Home Observer isolated demo'})
            result = await ws.receive_json()
            if not result.get('success'): raise RuntimeError('token creation failed')
            print(json.dumps({'long_lived_token':result['result']}))
asyncio.run(main())
"""
    result = docker("exec", "-i", CONTAINER, "python", "-c", ws_script,
                    stdin=json.dumps({"token": creds["access_token"]}).encode())
    creds.update(json.loads(result.stdout))
    private_json(credential_path, creds)
    return creds


def demo_credentials(work):
    owned = inspect_owned("container", CONTAINER)
    if not owned or not owned["State"]["Running"]:
        raise RuntimeError("This project's Home Assistant demo is not running")
    return json.loads((work / "credentials.json").read_text())


def verify(work, report_path=None):
    creds = demo_credentials(work)
    adapter = HomeAssistant(BASE_URL, creds["long_lived_token"], execute_real=True)
    states = adapter.client.get("/api/states")
    states.raise_for_status()
    by_id = {item["entity_id"]: item for item in states.json()}
    missing = set(ENTITIES) - set(by_id)
    if missing:
        raise RuntimeError(f"Demo entities missing: {sorted(missing)}")
    reports = []
    for entity in ENTITIES:
        for service in ("turn_off", "turn_on", "turn_off"):
            action = Action(domain=entity.split(".")[0], service=service, entity_id=entity,
                            data={}, reason="Isolated Home Assistant integration test", evidence_ids=["ha-demo-test"])
            result = adapter.execute(action, "ha-demo-" + uuid.uuid4().hex)
            # Independent API read confirms the service reached an actual Home Assistant entity.
            observed = check_response(adapter.client.get("/api/states/" + entity))
            result.update(service=service, observed_state=observed["state"], last_updated=observed["last_updated"])
            if result["state"] != ("on" if service == "turn_on" else "off") or not result["verified"]:
                raise RuntimeError(f"State verification failed for {entity} {service}")
            reports.append(result)
    config = check_response(adapter.client.get("/api/config"))
    adapter.close()
    instance = json.loads((work / "instance.json").read_text())
    output = {"status": "passed", "tested_at": time.time(), "home_assistant_version": config["version"],
              "instance": instance, "method": "HomeAssistant.execute -> real HA REST service -> independent HA state read",
              "device_kind": "isolated template lights/switch backed by input_boolean", "service_calls": reports}
    private_json(work / "verification.json", output)
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(output, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "verify", "stop"))
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    if args.action == "start":
        ensure_demo(work)
        wait_ready()
        onboard(work)
        output = verify(work, args.report)
        print(json.dumps({"status": output["status"], "url": BASE_URL, "container": CONTAINER,
                          "version": output["home_assistant_version"], "verified_service_calls": len(output["service_calls"]),
                          "credentials_file": str(work / "credentials.json")}, indent=2))
    elif args.action == "verify":
        output = verify(work, args.report)
        print(json.dumps({"status": output["status"], "verified_service_calls": len(output["service_calls"])}))
    elif args.action == "stop":
        if inspect_owned("container", CONTAINER):
            docker("stop", "--time", "60", CONTAINER, timeout=90)
        print(json.dumps({"status": "stopped", "container": CONTAINER}))
    else:
        owned = inspect_owned("container", CONTAINER)
        print(json.dumps({"url": BASE_URL, "running": bool(owned and owned["State"]["Running"]),
                          "container": CONTAINER, "credentials_present": (work / "credentials.json").is_file()}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Exception text must not contain headers, response bodies, or credentials.
        print(json.dumps({"error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        raise SystemExit(1)
