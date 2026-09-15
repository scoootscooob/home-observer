#!/usr/bin/env python3
"""Run the production frontier coordinator under a deny-by-default macOS sandbox.

The sandboxed process can read and write only the frontier bridge directory and
its own interpreter, and can open TCP connections only to the configured local
inference proxy port. It cannot read camera media, journals, sequences or any
other project file, and it cannot reach any other host or port. Field
filtering is enforced separately by the export boundary; this wrapper is the
host isolation layer the privacy document requires.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SANDBOX_EXEC = "/usr/bin/sandbox-exec"


def quote(path: str) -> str:
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


def interpreter_prefix(python: str) -> Path:
    resolved = Path(python).resolve()
    # Python frameworks ship as <prefix>/bin/python3.x with libs under <prefix>/lib.
    return resolved.parent.parent


def interpreter_roots(python: str) -> list[Path]:
    """The resolved interpreter prefix plus a venv directory when the path is a venv launcher."""
    roots = [interpreter_prefix(python)]
    launcher_root = Path(python).absolute().parent.parent
    if (launcher_root / "pyvenv.cfg").is_file():
        roots.append(launcher_root)
    return roots


def build_profile(*, bridge: Path, python: str, script: Path, allow_ports: list[int],
                  extra_read: list[Path] = ()) -> str:
    prefix = interpreter_prefix(python)
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process-fork)",
        "(allow signal (target self))",
        "(allow sysctl-read)",
        "(allow file-read-metadata)",
        "(allow file-ioctl (subpath \"/dev\"))",
        "(allow mach-lookup (global-name \"com.apple.system.logger\") (global-name \"com.apple.system.notification_center\")"
        " (global-name \"com.apple.SecurityServer\") (global-name \"com.apple.system.opendirectoryd.libinfo\")"
        " (global-name \"com.apple.trustd.agent\") (global-name \"com.apple.networkd\") (global-name \"com.apple.nehelper\")"
        " (global-name \"com.apple.coreservices.launchservicesd\") (global-name \"com.apple.cfprefsd.daemon\")"
        " (global-name \"com.apple.cfprefsd.agent\") (global-name \"com.apple.dnssd.service\"))",
        f"(allow process-exec* (subpath {quote(prefix)}) (literal {quote(Path(python).resolve())}))",
        "(allow file-read* (subpath \"/usr/lib\") (subpath \"/usr/share\") (subpath \"/System\")"
        " (subpath \"/Library/Preferences\") (subpath \"/private/var/db\") (subpath \"/private/etc\")"
        " (subpath \"/dev\") (literal \"/private/tmp\") (literal \"/tmp\") (literal \"/\"))",
        *[f"(allow file-read* (subpath {quote(root)}))" for root in interpreter_roots(python)],
        f"(allow file-read* (literal {quote(script.resolve())}))",
        f"(allow file-read* file-write* (subpath {quote(bridge.resolve())}))",
        "(allow file-write* (subpath \"/dev\"))",
    ]
    for path in extra_read:
        lines.append(f"(allow file-read* (subpath {quote(Path(path).resolve())}))")
    for port in allow_ports:
        if not 1 <= int(port) <= 65535:
            raise ValueError("invalid port")
        lines.append(f"(allow network-outbound (remote tcp \"localhost:{int(port)}\"))")
    return "\n".join(lines) + "\n"


def popen_sandboxed(*, bridge: Path, python: str, script: Path, script_args: list[str], allow_ports: list[int],
                    env: dict | None = None, extra_read=(), profile_out: Path | None = None,
                    stdout=None, stderr=None):
    """Start the sandboxed process; returns (process, profile_path). Caller removes the profile."""
    if not Path(SANDBOX_EXEC).is_file():
        raise RuntimeError("macOS sandbox-exec is not available on this host")
    profile = build_profile(bridge=bridge, python=python, script=script, allow_ports=allow_ports,
                            extra_read=list(extra_read))
    if profile_out is not None:
        profile_out.write_text(profile)
    clean_env = {"PATH": "/usr/bin:/bin", "HOME": str(bridge.resolve()), "TMPDIR": str(bridge.resolve()),
                 "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1",
                 "LANG": "C.UTF-8"}
    clean_env.update(env or {})
    with tempfile.NamedTemporaryFile("w", suffix=".sb", delete=False, dir=str(bridge.resolve())) as handle:
        handle.write(profile)
        profile_path = handle.name
    # -I ignores environment/user site; -S skips site-packages entirely, so only the
    # standard library and the script are importable inside the sandbox.
    process = subprocess.Popen([SANDBOX_EXEC, "-f", profile_path, python, "-I", "-S", str(script.resolve()),
                                *script_args], env=clean_env, cwd=str(bridge.resolve()),
                               stdout=stdout if stdout is not None else subprocess.PIPE,
                               stderr=stderr if stderr is not None else subprocess.PIPE, text=True)
    return process, profile_path


def run_sandboxed(*, bridge: Path, python: str, script: Path, script_args: list[str], allow_ports: list[int],
                  env: dict | None = None, extra_read=(), timeout: float | None = None, profile_out: Path | None = None):
    process, profile_path = popen_sandboxed(bridge=bridge, python=python, script=script, script_args=script_args,
                                            allow_ports=allow_ports, env=env, extra_read=extra_read,
                                            profile_out=profile_out)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
    finally:
        os.unlink(profile_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--allow-port", type=int, action="append", default=[])
    parser.add_argument("--env", action="append", default=[], help="NAME=VALUE or NAME (copied from the caller)")
    parser.add_argument("--profile-out", type=Path)
    parser.add_argument("script_args", nargs="*")
    args = parser.parse_args()
    env = {}
    for item in args.env:
        name, _, value = item.partition("=")
        env[name] = value if _ else os.environ.get(name, "")
    if shutil.which("sandbox-exec") is None and not Path(SANDBOX_EXEC).is_file():
        raise SystemExit("sandbox-exec is required for the production frontier coordinator")
    result = run_sandboxed(bridge=args.bridge, python=args.python, script=args.script,
                           script_args=args.script_args, allow_ports=args.allow_port, env=env,
                           profile_out=args.profile_out)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    print(json.dumps({"returncode": result.returncode}), file=sys.stderr)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
