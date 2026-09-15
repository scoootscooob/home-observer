"""Host isolation: the sandboxed coordinator cannot read media or reach other ports."""
import json
import socket
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import frontier_sandbox  # noqa: E402

PROBE = '''
import json, os, socket, sys
bridge, outside, allowed_port, denied_port = sys.argv[1:5]
result = {}
try:
    open(outside, "rb").read(); result["outside_read"] = "allowed"
except Exception as exc:
    result["outside_read"] = type(exc).__name__
try:
    result["outside_list"] = "allowed" if os.listdir(os.path.dirname(outside)) else "empty"
except Exception as exc:
    result["outside_list"] = type(exc).__name__
try:
    open(os.path.join(bridge, "outbox", "probe-read.json")).read(); result["bridge_read"] = "allowed"
except Exception as exc:
    result["bridge_read"] = type(exc).__name__
try:
    open(os.path.join(bridge, "inbox", "probe-write.json"), "w").write("{}"); result["bridge_write"] = "allowed"
except Exception as exc:
    result["bridge_write"] = type(exc).__name__
try:
    open(os.path.join(os.path.dirname(outside), "written-by-probe.txt"), "w").write("x"); result["outside_write"] = "allowed"
except Exception as exc:
    result["outside_write"] = type(exc).__name__
for name, port in (("allowed_port", int(allowed_port)), ("denied_port", int(denied_port))):
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=2); s.close(); result[name] = "connected"
    except Exception as exc:
        result[name] = type(exc).__name__
print(json.dumps(result))
'''


def listener():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(4)
    port = server.getsockname()[1]

    def serve():
        while True:
            try:
                connection, _ = server.accept()
            except OSError:
                return
            connection.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return server, port


@pytest.mark.skipif(not Path(frontier_sandbox.SANDBOX_EXEC).is_file(), reason="macOS sandbox-exec unavailable")
def test_sandboxed_process_is_confined_to_bridge_and_allowed_port(tmp_path):
    bridge = tmp_path / "bridge"
    for name in ("inbox", "outbox"):
        (bridge / name).mkdir(parents=True)
    (bridge / "outbox/probe-read.json").write_text("{}")
    media = tmp_path / "run/sequences/clip_x"
    media.mkdir(parents=True)
    outside = media / "000001.jpg"
    outside.write_bytes(b"camera frame bytes")
    script = tmp_path / "probe.py"
    script.write_text(PROBE)
    allowed_server, allowed_port = listener()
    denied_server, denied_port = listener()
    try:
        result = frontier_sandbox.run_sandboxed(
            bridge=bridge, python=sys.executable, script=script,
            script_args=[str(bridge), str(outside), str(allowed_port), str(denied_port)],
            allow_ports=[allowed_port], timeout=60, profile_out=tmp_path / "profile.sb")
    finally:
        allowed_server.close()
        denied_server.close()
    assert result.returncode == 0, result.stderr[-2000:]
    probe = json.loads(result.stdout.strip().splitlines()[-1])
    assert probe["outside_read"] == "PermissionError"
    assert probe["outside_list"] == "PermissionError"
    assert probe["outside_write"] == "PermissionError"
    assert probe["bridge_read"] == "allowed"
    assert probe["bridge_write"] == "allowed"
    assert probe["allowed_port"] == "connected"
    assert probe["denied_port"] != "connected"
    assert not (media / "written-by-probe.txt").exists()
    profile = (tmp_path / "profile.sb").read_text()
    assert "(deny default)" in profile and str(bridge.resolve()) in profile
