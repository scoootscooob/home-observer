"""Runpod control plane and bounded, recoverable SSH jobs (stdlib only)."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
REMOTE = "/workspace/home-observer"
POD_FIELDS = "id name desiredStatus costPerHr runtime { ports { ip isIpPublic privatePort publicPort type } }"


class CloudError(RuntimeError):
    pass


class AmbiguousCreate(CloudError):
    """The provider might have accepted the create request. Never retry it."""


class GraphQLError(CloudError):
    """Sanitized provider GraphQL error, without response headers or request bodies."""


class CreateRejected(CloudError):
    """Provider explicitly rejected capacity or request validation before creation."""


def credentials(environ=None, config=None):
    env = os.environ if environ is None else environ
    token = env.get("RUNPOD_API_KEY", "").strip()
    if token:
        return token
    path = Path(config or Path.home() / ".runpod/config.toml")
    if path.is_file():
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                raise CloudError(
                    "Use Python 3.11+ to read ~/.runpod/config.toml, or set RUNPOD_API_KEY"
                ) from None
        try:
            data = tomllib.loads(path.read_text())
        except Exception:
            raise CloudError("Runpod configuration is invalid TOML") from None
        for table in (data, data.get("default", {}), data.get("runpod", {})):
            for key in ("api_key", "apiKey", "apikey"):
                if isinstance(table.get(key), str) and table[key].strip():
                    return table[key].strip()
    raise CloudError("Runpod credentials missing: set RUNPOD_API_KEY or ~/.runpod/config.toml api_key")


def redact(value, secrets=()):
    if isinstance(value, dict):
        return {
            k: (
                "[REDACTED]"
                if re.search(r"token|secret|password|api.?key|authorization", k, re.I)
                else redact(v, secrets)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v, secrets) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            if isinstance(secret, str) and secret:
                value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"(?:rpa_|hf_)[A-Za-z0-9_-]{10,}", "[REDACTED]", value)
        value = re.sub(r"(?i)(\bauthorization\s*[:=]\s*(?:bearer\s+)?)[^\s\"']+", r"\1[REDACTED]", value)
        value = re.sub(
            r"(?i)(\b(?:api[_-]?key|token|password|secret)\s*[=:]\s*)[^\s\"']+", r"\1[REDACTED]", value
        )
    return value


class Runpod:
    def __init__(self, api_key, opener=urllib.request.urlopen):
        self._key, self._opener = api_key, opener

    def request(self, method, url, payload=None):
        # Never log response bodies, headers, URLs with credentials, or HTTP exception text.
        req = urllib.request.Request(
            url,
            data=None if payload is None else json.dumps(payload).encode(),
            method=method,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "User-Agent": "home-observer/0.1",
            },
        )
        try:
            with self._opener(req, timeout=25) as response:
                raw = response.read()
            return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            raise CloudError(
                f"Runpod HTTP {exc.code}; provider response omitted to protect credentials"
            ) from None
        except (OSError, ValueError):
            raise CloudError("Runpod transport or response error; response omitted") from None

    def graphql(self, query, variables=None):
        response = self.request("POST", GRAPHQL, {"query": query, "variables": variables or {}})
        if response.get("errors"):
            messages = [
                str(item.get("message", "GraphQL error"))
                for item in response["errors"]
                if isinstance(item, dict)
            ]
            detail = redact("; ".join(messages), (self._key,))[:1800]
            raise GraphQLError(f"Runpod GraphQL error: {detail}")
        if "data" not in response:
            raise CloudError("Runpod GraphQL response missing data")
        return response["data"]

    def account(self):
        return self.graphql("query { myself { id clientBalance currentSpendPerHr } }")["myself"]

    def quote(self, gpu_id, cloud_type):
        query = """query Quote($input: GpuTypeFilter, $secure: Boolean) {
          gpuTypes(input: $input) { id displayName memoryInGb
            lowestPrice(input: {gpuCount: 1, secureCloud: $secure}) {
              uninterruptablePrice stockStatus availableGpuCounts } } }"""
        rows = self.graphql(query, {"input": {"id": gpu_id}, "secure": cloud_type == "SECURE"})["gpuTypes"]
        if len(rows) != 1:
            raise CloudError("GPU type did not resolve uniquely")
        return rows[0]

    def create(self, payload):
        # No retry: even a transport failure or invalid response may hide a successfully created Pod.
        query = "mutation Create($input: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $input) { id name costPerHr desiredStatus } }"
        try:
            result = self.graphql(query, {"input": payload})["podFindAndDeployOnDemand"]
            if not result or not result.get("id"):
                raise CloudError("Missing Pod id")
            return result
        except GraphQLError as exc:
            # Explicit request validation / unavailable-capacity errors are distinguishable
            # from transport loss. An unrecognized server error remains ambiguous.
            message = str(exc)
            if re.search(
                r"cannot query field|unknown (?:argument|type|field)|syntax error|variable .*got invalid value|no instances (?:are )?currently available|not enough .*available|insufficient (?:balance|funds)|could not find .*deploy",
                message,
                re.I,
            ):
                raise CreateRejected(message) from None
            raise AmbiguousCreate(
                f"{message}. Create outcome uncertain; reconcile the recorded run before any new launch."
            ) from None
        except Exception:
            raise AmbiguousCreate(
                "Create outcome uncertain. Do not launch again; use reconcile with this run directory."
            ) from None

    def get(self, pod_id):
        return self.graphql(
            f"query Pod($input: PodFilter) {{ pod(input: $input) {{ {POD_FIELDS} }} }}",
            {"input": {"podId": pod_id}},
        )["pod"]

    def list(self):
        return self.graphql(f"query {{ myself {{ pods {{ {POD_FIELDS} }} }} }}")["myself"]["pods"]

    def stop(self, pod_id):
        self.request("POST", f"{REST}/pods/{checked_id(pod_id)}/stop")

    def delete(self, pod_id):
        self.request("DELETE", f"{REST}/pods/{checked_id(pod_id)}")


def checked_id(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise CloudError("Invalid Pod id")
    return value


@dataclass
class JobConfig:
    gpu_type: str = "NVIDIA RTX A6000"
    cloud_type: str = "SECURE"
    image: str = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
    volume_gb: int = 80
    container_gb: int = 30
    min_memory_gb: int = 64
    min_vcpu: int = 8
    max_spend_usd: float = 10.0
    max_hours: float = 2.0
    max_hourly_usd: float = 3.0
    cleanup_reserve_seconds: int = 600
    startup_timeout_seconds: int = 900
    # Storage quote is based on current documented $0.10/GB/month running rate.
    storage_usd_gb_month: float = 0.10
    command: str = "bash scripts/runpod_job.sh"

    def validate(self):
        if self.cloud_type not in ("SECURE", "COMMUNITY"):
            raise CloudError("cloud_type must be SECURE or COMMUNITY")
        for key in ("max_spend_usd", "max_hours", "max_hourly_usd", "storage_usd_gb_month"):
            val = getattr(self, key)
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val <= 0:
                raise CloudError(f"{key} must be finite and positive")
        for key in (
            "volume_gb",
            "container_gb",
            "min_memory_gb",
            "min_vcpu",
            "cleanup_reserve_seconds",
            "startup_timeout_seconds",
        ):
            if not isinstance(getattr(self, key), int) or getattr(self, key) <= 0:
                raise CloudError(f"{key} must be a positive integer")
        if self.max_hours * 3600 <= self.cleanup_reserve_seconds + 60:
            raise CloudError("Job time must exceed cleanup reserve by at least one minute")

    @property
    def storage_hourly(self):
        return (self.volume_gb + self.container_gb) * self.storage_usd_gb_month / (30 * 24)


def preflight(api, config):
    config.validate()
    account = api.account()
    quote = api.quote(config.gpu_type, config.cloud_type)
    lowest = quote.get("lowestPrice") or {}
    raw_price = lowest.get("uninterruptablePrice")
    if raw_price is None or not math.isfinite(float(raw_price)) or float(raw_price) <= 0:
        raise CloudError("GPU has no usable live on-demand quote")
    counts = lowest.get("availableGpuCounts")
    if lowest.get("stockStatus") == "None" or (isinstance(counts, list) and 1 not in counts):
        raise CloudError("No single-GPU capacity in the requested cloud")
    hourly = float(raw_price) + config.storage_hourly
    if hourly > config.max_hourly_usd:
        raise CloudError("Live GPU plus estimated storage rate exceeds max_hourly_usd")
    seconds = min(config.max_hours * 3600, config.max_spend_usd / hourly * 3600)
    if seconds <= config.cleanup_reserve_seconds + 60:
        raise CloudError("Budget leaves insufficient execution time after cleanup reserve")
    balance = account.get("clientBalance")
    if balance is None or float(balance) < max(hourly, hourly * seconds / 3600):
        raise CloudError("Account balance is unavailable or insufficient for the bounded job")
    return {
        "account": account,
        "gpu": quote,
        "estimated_hourly_usd": hourly,
        "maximum_seconds": int(seconds),
        "estimated_maximum_usd": hourly * int(seconds) / 3600,
        "storage_pricing_source": "https://docs.runpod.io/pods/pricing",
        "quoted_at": time.time(),
    }


def create_payload(config, name, public_key, stop_epoch):
    if not re.fullmatch(r"ssh-ed25519 [A-Za-z0-9+/=]+(?: [^\r\n]*)?", public_key.strip()):
        raise CloudError("A valid single-line Ed25519 public key is required")
    # Explicit bootstrap works with the pinned official image and never exposes Jupyter.
    boot = """set -eu
mkdir -p /root/.ssh /run/sshd
chmod 700 /root/.ssh
printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A
exec /usr/sbin/sshd -D -e -o PasswordAuthentication=no -o PermitRootLogin=prohibit-password
"""
    return {
        "name": name,
        "cloudType": config.cloud_type,
        "gpuTypeId": config.gpu_type,
        "gpuCount": 1,
        "imageName": config.image,
        "containerDiskInGb": config.container_gb,
        "volumeInGb": config.volume_gb,
        "volumeMountPath": "/workspace",
        "minMemoryInGb": config.min_memory_gb,
        "minVcpuCount": config.min_vcpu,
        "ports": "22/tcp",
        "supportPublicIp": True,
        "startSsh": True,
        "startJupyter": False,
        "env": [
            {"key": "PUBLIC_KEY", "value": public_key.strip()},
            {"key": "SSH_PUBLIC_KEY", "value": public_key.strip()},
        ],
        "dockerArgs": "bash -lc " + shlex.quote(boot),
        "stopAfter": datetime.fromtimestamp(stop_epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def save_state(run_dir, state):
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "state.json"
    tmp = run_dir / "state.json.tmp"
    tmp.write_text(json.dumps(redact(state), indent=2) + "\n")
    tmp.chmod(0o600)
    os.replace(tmp, path)


def package_project(project, destination):
    """Allowlist release payload; credentials, work, caches and previous runs stay local."""
    project = Path(project).resolve()
    include = (
        "src",
        "scripts",
        "configs",
        "docs",
        "tests",
        "datasets",
        "data",
        "pyproject.toml",
        "requirements-gpu.txt",
        "requirements-vllm.txt",
        "README.md",
        "CONTRACT.md",
    )
    with tarfile.open(destination, "w:gz") as archive:
        for root_name in include:
            root = project / root_name
            if not root.exists():
                continue
            files = [root] if root.is_file() else sorted(root.rglob("*"))
            for file in files:
                relative = file.relative_to(project)
                if (
                    file.is_symlink()
                    or not file.is_file()
                    or any(p.startswith(".") or p == "__pycache__" for p in relative.parts)
                ):
                    continue
                if re.search(
                    r"(?:\.pem|\.key|\.pyc|\.pyo)$|(?:^|/)(?:credentials|secrets)(?:[./]|$)",
                    str(relative),
                    re.I,
                ):
                    continue
                archive.add(file, arcname="home-observer/" + str(relative), recursive=False,
                            filter=_root_owned)
    return hashlib.sha256(Path(destination).read_bytes()).hexdigest()


def _root_owned(info):
    """Archive entries carry no local uid/gid, so extraction never needs chown on the Pod."""
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    return info


def ssh_address(pod):
    for port in (pod.get("runtime") or {}).get("ports") or []:
        if port.get("privatePort") == 22 and port.get("isIpPublic") and port.get("type") == "tcp":
            host = str(ipaddress.ip_address(port["ip"]))
            number = int(port["publicPort"])
            if not 0 < number < 65536:
                raise CloudError("Invalid SSH port")
            return host, number
    return None


class SSH:
    def __init__(self, host, port, key, run_dir):
        self.host, self.port = host, int(port)
        self.run_dir = Path(run_dir).resolve()
        self.options = [
            "-F",
            "/dev/null",
            "-i",
            str(Path(key).resolve()),
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=2",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={Path(run_dir).resolve() / 'known_hosts'}",
        ]

    def _record_error(self, operation, stderr, stdin=None):
        """Keep sanitized transport diagnostics privately, never commands or stdout."""
        secret_values = [
            v
            for k, v in os.environ.items()
            if re.search(r"token|secret|password|api.?key|authorization", k, re.I)
        ]
        if stdin:
            secret_values.append(stdin.decode(errors="replace") if isinstance(stdin, bytes) else str(stdin))
        detail = stderr.decode(errors="replace") if isinstance(stderr, bytes) else str(stderr or "")
        item = {
            "timestamp": time.time(),
            "operation": operation,
            "host": self.host,
            "port": self.port,
            "stderr": redact(detail, secret_values)[:16000],
        }
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / "ssh-errors.jsonl"
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600), "a") as stream:
            os.chmod(path, 0o600)
            stream.write(json.dumps(item) + "\n")

    def call(self, command, timeout=45, stdin=None):
        try:
            result = subprocess.run(
                ["ssh", *self.options, "-p", str(self.port), f"root@{self.host}", command],
                input=stdin,
                capture_output=True,
                timeout=max(1, timeout),
            )
        except subprocess.TimeoutExpired as exc:
            self._record_error("ssh_timeout", exc.stderr, stdin)
            raise CloudError("SSH operation timed out") from None
        if result.returncode:
            self._record_error("ssh_exit_" + str(result.returncode), result.stderr, stdin)
            raise CloudError(f"SSH operation failed with exit {result.returncode}; see retained remote logs")
        return result.stdout

    def copy(self, source, destination, upload=True, timeout=120):
        remote = f"root@{self.host}:"
        args = (
            [str(source), remote + str(destination)] if upload else [remote + str(source), str(destination)]
        )
        try:
            result = subprocess.run(
                ["scp", *self.options, "-P", str(self.port), *args],
                capture_output=True,
                timeout=max(1, timeout),
            )
        except subprocess.TimeoutExpired as exc:
            self._record_error("scp_timeout", exc.stderr)
            raise CloudError("SCP transfer timed out") from None
        if result.returncode:
            self._record_error("scp_exit_" + str(result.returncode), result.stderr)
            raise CloudError("SCP transfer failed; remote volume retained")

    def tunnel_command(self, local_port=8000):
        return shlex.join(
            [
                "ssh",
                *self.options,
                "-p",
                str(self.port),
                "-N",
                "-L",
                f"127.0.0.1:{local_port}:127.0.0.1:8000",
                f"root@{self.host}",
            ]
        )


def retrieve(ssh, run_dir, timeout):
    # The job is stopped before this function; the archive is an immutable snapshot.
    raw = ssh.call(
        f"cd {REMOTE} && tar -czf /workspace/result.tar.gz artifacts && sha256sum /workspace/result.tar.gz",
        timeout=timeout,
    )
    expected = raw.decode().strip().split()[0]
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise CloudError("Remote artifact checksum is invalid")
    local = run_dir / "artifacts.tar.gz"
    ssh.copy("/workspace/result.tar.gz", local, upload=False, timeout=timeout)
    actual = hashlib.sha256(local.read_bytes()).hexdigest()
    if actual != expected:
        raise CloudError("Artifact checksum mismatch; Pod must be retained")
    # Keep the archive as the definitive copy; extract only ordinary files under artifacts/.
    output = run_dir / "downloaded"
    output.mkdir(exist_ok=True)
    with tarfile.open(local) as archive:
        members = archive.getmembers()
        for member in members:
            parts = Path(member.name).parts
            if (
                not parts
                or parts[0] != "artifacts"
                or ".." in parts
                or not (member.isfile() or member.isdir())
            ):
                raise CloudError("Unsafe member in artifact archive")
        archive.extractall(output, members=members)
    return {"archive": str(local), "sha256": actual}


def finish_pod(api, pod_id, artifacts_verified):
    """Never delete the only copy of artifacts. Stop preserves mounted volume data."""
    if artifacts_verified:
        api.delete(pod_id)
        return "terminated"
    api.stop(pod_id)
    return "stopped_recovery_required"


def confirm_cleanup(api, pod_id, deleted, attempts=5, pause=time.sleep):
    """Mutations are asynchronous; confirm eventual provider state with bounded reads."""
    for attempt in range(attempts):
        pod = api.get(pod_id)
        if deleted and pod is None:
            return True
        if not deleted and pod and pod.get("desiredStatus") in ("EXITED", "STOPPED"):
            return True
        if attempt + 1 < attempts:
            pause(min(2**attempt, 8))
    return False


def poll_remote_job(
    api,
    ssh,
    state,
    run_dir,
    key,
    execution_deadline,
    *,
    grace_seconds=300,
    poll_seconds=15,
    now=time.time,
    pause=time.sleep,
):
    """Retry only read-only status checks, bounded by grace and the job deadline.

    Every transport failure refreshes the provider SSH endpoint. A response from
    the provider proving terminal state is distinct from an unavailable API.
    Create, upload, launch and cleanup mutations are never retried here.
    """
    failure_started = None
    command = f"""if [ -f {REMOTE}/artifacts/exit-code ]; then
cat {REMOTE}/artifacts/exit-code
elif [ -f {REMOTE}/artifacts/job.pid ]; then
pid=$(cat {REMOTE}/artifacts/job.pid)
case "$pid" in ''|*[!0-9]*) printf invalid_pid;;
*) if kill -0 -- -"$pid" 2>/dev/null; then printf running; else printf process_missing; fi;; esac
else printf running; fi"""
    while now() < execution_deadline:
        try:
            result = ssh.call(command, timeout=min(45, max(1, execution_deadline - now()))).decode().strip()
            if result == "process_missing":
                state["terminal_process_missing_at"] = now()
                save_state(run_dir, state)
                raise CloudError("Remote job process exited without an exit-code record")
            if result != "running" and not re.fullmatch(r"[0-9]{1,3}", result):
                raise CloudError("Remote status response is invalid")
        except (CloudError, UnicodeError) as exc:
            if state.get("terminal_process_missing_at"):
                raise
            stamp = now()
            failure_started = stamp if failure_started is None else failure_started
            state["poll_error_count"] = state.get("poll_error_count", 0) + 1
            state["poll_consecutive_errors"] = state.get("poll_consecutive_errors", 0) + 1
            state["poll_failure_since"] = failure_started
            state["last_poll_error_at"] = stamp
            state["last_poll_error"] = redact(str(exc), (getattr(api, "_key", ""),))
            try:
                pod = api.get(state["pod_id"])
            except CloudError as api_error:
                state["poll_api_error_count"] = state.get("poll_api_error_count", 0) + 1
                state["last_poll_api_error"] = redact(str(api_error), (getattr(api, "_key", ""),))
            else:
                if not pod or pod.get("desiredStatus") in ("EXITED", "TERMINATED", "STOPPED"):
                    state["terminal_provider_status"] = pod.get("desiredStatus") if pod else "absent"
                    save_state(run_dir, state)
                    raise CloudError("Provider confirms the running Pod is terminal or absent")
                address = ssh_address(pod)
                if address and address != (ssh.host, ssh.port):
                    ssh = SSH(*address, key, run_dir)
                    state["ssh"] = {"host": ssh.host, "port": ssh.port, "key": str(key)}
                    state["tunnel"] = ssh.tunnel_command()
                    state["ssh_endpoint_refreshes"] = state.get("ssh_endpoint_refreshes", 0) + 1
            save_state(run_dir, state)
            remaining = min(execution_deadline - now(), grace_seconds - (now() - failure_started))
            if remaining <= 0:
                raise CloudError("Read-only SSH polling grace or execution deadline exhausted") from None
            pause(min(poll_seconds, remaining))
            continue
        state["last_poll"] = now()
        if failure_started is not None:
            state["poll_recoveries"] = state.get("poll_recoveries", 0) + 1
            state["last_poll_recovery_at"] = now()
        state["poll_consecutive_errors"] = 0
        state["poll_failure_since"] = None
        save_state(run_dir, state)
        return result, ssh
    raise CloudError("Execution deadline reached during read-only status polling")


def run_job(api, config, project, run_dir, key, poll_seconds=15):
    config.validate()
    project, run_dir, key = Path(project).resolve(), Path(run_dir).resolve(), Path(key).resolve()
    if (run_dir / "state.json").exists():
        raise CloudError("Run directory already has a lifecycle record. Reconcile it; do not launch twice.")
    if not key.is_file() or not Path(str(key) + ".pub").is_file():
        raise CloudError("Task SSH keypair is missing; run the keygen subcommand")
    if not (project / "scripts/runpod_job.sh").is_file():
        raise CloudError("Remote job script is missing")
    quote = preflight(api, config)
    run_dir.mkdir(parents=True, exist_ok=True)
    bundle = run_dir / "project.tar.gz"
    digest = package_project(project, bundle)
    start = time.time()
    deadline = start + quote["maximum_seconds"]
    execution_deadline = deadline - config.cleanup_reserve_seconds
    state = {
        "name": "home-observer-" + uuid.uuid4().hex[:16],
        "status": "create_intent",
        "config": asdict(config),
        "quote": quote,
        "started_at": start,
        "deadline": deadline,
        "project_sha256": digest,
        "pod_id": None,
        "artifacts_verified": False,
    }
    save_state(run_dir, state)
    ssh, error = None, None
    try:
        pod = api.create(create_payload(config, state["name"], Path(str(key) + ".pub").read_text(), deadline))
        state.update(pod_id=pod["id"], status="created")
        save_state(run_dir, state)
        actual_rate = float(pod.get("costPerHr") or 0) + config.storage_hourly
        # Keep an additional storage allowance even if costPerHr already includes it.
        if (
            actual_rate <= config.storage_hourly
            or actual_rate > config.max_hourly_usd
            or actual_rate * quote["maximum_seconds"] / 3600 > config.max_spend_usd + 1e-6
        ):
            raise CloudError("Actual Pod rate is unknown or exceeds the configured budget")
        state["actual_estimated_hourly_usd"] = actual_rate
        ready_deadline = min(execution_deadline, start + config.startup_timeout_seconds)
        while time.time() < ready_deadline:
            try:
                pod = api.get(state["pod_id"])
            except CloudError as exc:
                state["startup_api_error_count"] = state.get("startup_api_error_count", 0) + 1
                state["last_startup_api_error"] = redact(str(exc), (getattr(api, "_key", ""),))
                save_state(run_dir, state)
                time.sleep(min(poll_seconds, max(0, ready_deadline - time.time())))
                continue
            if not pod or pod.get("desiredStatus") in ("EXITED", "TERMINATED", "STOPPED"):
                raise CloudError("Pod stopped before SSH became ready")
            address = ssh_address(pod)
            if address:
                candidate = SSH(*address, key, run_dir)
                try:
                    candidate.call("true", timeout=20)
                    ssh = candidate
                    break
                except CloudError:
                    pass
            time.sleep(min(poll_seconds, max(0, ready_deadline - time.time())))
        if not ssh:
            raise CloudError("Pod SSH startup deadline expired")
        state.update(
            status="uploading",
            ssh={"host": ssh.host, "port": ssh.port, "key": str(key)},
            tunnel=ssh.tunnel_command(),
        )
        save_state(run_dir, state)
        ssh.copy(bundle, "/workspace/project.tar.gz", timeout=min(300, execution_deadline - time.time()))
        ssh.call(
            f"printf '%s  /workspace/project.tar.gz\\n' {shlex.quote(digest)} | sha256sum -c - && tar --no-same-owner -xzf /workspace/project.tar.gz -C /workspace && mkdir -p {REMOTE}/artifacts",
            timeout=120,
        )
        # Optional gated-model credential travels via encrypted stdin into tmpfs, never argv or Pod env.
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            ssh.call("umask 077; cat > /dev/shm/home-observer-hf-token", stdin=hf_token.encode())
        inference_token = os.environ.get("HOME_OBSERVER_API_TOKEN")
        if inference_token:
            ssh.call(
                "umask 077; cat > /dev/shm/home-observer-inference-token", stdin=inference_token.encode()
            )
        remaining = int(execution_deadline - time.time())
        if remaining < 60:
            raise CloudError("Execution budget exhausted by startup/upload")
        remote_script = f"""#!/bin/bash
set +e
cd {REMOTE}
if [ -f /dev/shm/home-observer-hf-token ]; then export HF_TOKEN="$(cat /dev/shm/home-observer-hf-token)"; fi
setsid timeout --signal=TERM --kill-after=30 {remaining}s bash -lc {shlex.quote(config.command)} > artifacts/job.log 2>&1 &
job_pid=$!
printf '%s\n' "$job_pid" > artifacts/job.pid
wait "$job_pid"
job_exit=$?
rm -f /dev/shm/home-observer-hf-token
printf '%s\n' "$job_exit" > artifacts/exit-code
exit "$job_exit"
"""
        ssh.call(
            "cat > /workspace/home-observer-run.sh && chmod 700 /workspace/home-observer-run.sh",
            stdin=remote_script.encode(),
        )
        ssh.call(
            "nohup bash /workspace/home-observer-run.sh </dev/null >/workspace/home-observer-launch.log 2>&1 &"
        )
        state["status"] = "running"
        save_state(run_dir, state)
        while time.time() < execution_deadline:
            result, ssh = poll_remote_job(
                api, ssh, state, run_dir, key, execution_deadline, poll_seconds=poll_seconds
            )
            if result != "running":
                state["job_exit_code"] = int(result)
                if int(result):
                    raise CloudError(f"Remote job exited {result}; downloading failure evidence")
                break
            time.sleep(min(poll_seconds, max(0, execution_deadline - time.time())))
        else:
            raise CloudError("Execution deadline reached; preserving partial artifacts")
    except BaseException as exc:
        error = exc
        state["error"] = redact(str(exc), (getattr(api, "_key", ""), os.environ.get("HF_TOKEN", "")))
        state["status"] = (
            "create_ambiguous"
            if isinstance(exc, AmbiguousCreate)
            else "create_rejected"
            if isinstance(exc, CreateRejected)
            else "failed"
        )
    finally:
        if state.get("pod_id"):
            # A failed grace interval may still have refreshed the provider's
            # endpoint; use that last observed endpoint for artifact recovery.
            latest_ssh = state.get("ssh")
            if ssh and latest_ssh and (ssh.host, ssh.port) != (latest_ssh["host"], latest_ssh["port"]):
                ssh = SSH(latest_ssh["host"], latest_ssh["port"], key, run_dir)
            if ssh:
                try:
                    ssh.call(
                        f"""if [ -f {REMOTE}/artifacts/job.pid ]; then
pid=$(cat {REMOTE}/artifacts/job.pid)
case "$pid" in ''|*[!0-9]*) exit 1;; esac
kill -TERM -- -"$pid" 2>/dev/null || true
for attempt in 1 2 3 4 5; do
  kill -0 -- -"$pid" 2>/dev/null || break
  sleep 1
done
kill -KILL -- -"$pid" 2>/dev/null || true
fi
rm -f /dev/shm/home-observer-hf-token""",
                        timeout=20,
                    )
                    state["artifacts"] = retrieve(ssh, run_dir, min(180, max(1, deadline - time.time() - 30)))
                    state["artifacts_verified"] = True
                except Exception as exc:
                    state["artifact_error"] = str(exc)
            try:
                state["cleanup"] = finish_pod(api, state["pod_id"], state["artifacts_verified"])
                state["cleanup_at"] = time.time()
                # GET confirms disappearance or stop. Mutation receipt alone is not success evidence.
                state["cleanup_confirmed"] = confirm_cleanup(
                    api, state["pod_id"], state["artifacts_verified"]
                )
            except Exception as exc:
                state["cleanup_error"] = str(exc)
                state["cleanup_confirmed"] = False
        state["finished_at"] = time.time()
        if not error and state.get("artifacts_verified") and state.get("cleanup_confirmed"):
            state["status"] = "complete"
        elif not error:
            state["status"] = "recovery_required"
        save_state(run_dir, state)
    if error:
        raise CloudError(f"{state['error']} State saved in {run_dir / 'state.json'}") from None
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("plan", "preflight", "keygen", "run", "status", "reconcile", "stop", "recover", "cleanup"),
    )
    parser.add_argument("--config", default="configs/runpod-training.json")
    parser.add_argument("--project", default=".")
    parser.add_argument("--run-dir", default="work/runpod-training")
    parser.add_argument("--key", default="work/runpod_ssh")
    args = parser.parse_args(argv)
    run_dir, key = Path(args.run_dir).resolve(), Path(args.key).resolve()
    try:
        if args.action == "keygen":
            key.parent.mkdir(parents=True, exist_ok=True)
            if key.exists() or Path(str(key) + ".pub").exists():
                raise CloudError("Key path already exists")
            subprocess.run(
                ["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "home-observer", "-f", str(key)],
                check=True,
                capture_output=True,
            )
            print(json.dumps({"key": str(key), "public_key": str(key) + ".pub"}))
            return 0
        config = JobConfig(**json.loads(Path(args.config).read_text()))
        config.validate()
        if args.action == "plan":
            print(
                json.dumps(
                    {
                        "config": asdict(config),
                        "network": "22/tcp only; inference loopback + SSH tunnel",
                        "credential_present": bool(
                            os.environ.get("RUNPOD_API_KEY")
                            or (Path.home() / ".runpod/config.toml").is_file()
                        ),
                        "provisioned": False,
                        "next": "preflight, keygen, run",
                        "live_price_required": True,
                    },
                    indent=2,
                )
            )
            return 0
        api = Runpod(credentials())
        if args.action == "preflight":
            print(json.dumps(preflight(api, config), indent=2))
        elif args.action == "run":
            # Convert SIGTERM to an exception so the finally block saves outputs and stops billing.
            def terminated(signum, frame):
                raise CloudError("Local supervisor interrupted")

            signal.signal(signal.SIGTERM, terminated)
            print(json.dumps(run_job(api, config, args.project, run_dir, key), indent=2))
        else:
            state = json.loads((run_dir / "state.json").read_text())
            if args.action == "reconcile":
                matches = [p for p in api.list() if p["name"] == state["name"]]
                if len(matches) != 1:
                    raise CloudError(f"Found {len(matches)} Pods for recorded name. No new Pod was created.")
                state["pod_id"] = matches[0]["id"]
                state["status"] = "reconciled_recovery_required"
                save_state(run_dir, state)
            elif args.action == "stop":
                api.stop(state["pod_id"])
            elif args.action == "recover":
                pod = api.get(state["pod_id"])
                address = ssh_address(pod or {})
                if not address:
                    raise CloudError(
                        "Recovery needs a running Pod with SSH. Resume the recorded Pod first; do not create a replacement."
                    )
                connection = SSH(*address, key, run_dir)
                state["artifacts"] = retrieve(connection, run_dir, 300)
                state["artifacts_verified"] = True
                save_state(run_dir, state)
            elif args.action == "cleanup":
                if not state.get("artifacts_verified"):
                    raise CloudError(
                        "No verified local artifacts; use stop to preserve volume and recover through SSH first"
                    )
                api.delete(state["pod_id"])
                state["cleanup"] = "terminated"
                state["cleanup_confirmed"] = confirm_cleanup(api, state["pod_id"], True)
                save_state(run_dir, state)
            print(
                json.dumps(
                    {"state": state, "pod": api.get(state["pod_id"]) if state.get("pod_id") else None},
                    indent=2,
                )
            )
    except (CloudError, OSError, ValueError, TypeError) as exc:
        print(json.dumps({"error": redact(str(exc))}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
