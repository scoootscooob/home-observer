"""Safety-relevant orchestration tests. All provider calls are fake."""

import io
import json
import subprocess
import tarfile
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from home_observer.cloud import (
    SSH,
    AmbiguousCreate,
    CloudError,
    CreateRejected,
    GraphQLError,
    JobConfig,
    Runpod,
    confirm_cleanup,
    create_payload,
    credentials,
    finish_pod,
    package_project,
    poll_remote_job,
    preflight,
    redact,
    run_job,
    ssh_address,
)


def quoted_api(price=0.8, balance=100.0, counts=None):
    api = Mock()
    api.account.return_value = {"id": "account", "clientBalance": balance, "currentSpendPerHr": 0}
    api.quote.return_value = {
        "id": "NVIDIA RTX A6000",
        "memoryInGb": 48,
        "lowestPrice": {
            "uninterruptablePrice": price,
            "stockStatus": "High",
            "availableGpuCounts": [1] if counts is None else counts,
        },
    }
    return api


def test_environment_credentials_take_precedence(tmp_path):
    invalid = tmp_path / "config.toml"
    invalid.write_text("this is not TOML")
    assert credentials({"RUNPOD_API_KEY": "  token-secret  "}, invalid) == "token-secret"
    with pytest.raises(CloudError, match="missing"):
        credentials({}, tmp_path / "missing")


def test_redaction_covers_nested_keys_and_values():
    result = redact(
        {"authorization": "secret", "deep": [{"error": "url token123", "api_key": "secret"}]}, ("token123",)
    )
    assert "token123" not in json.dumps(result)
    assert "secret" not in json.dumps(result)


def test_http_exception_does_not_expose_provider_body_or_key():
    error = urllib.error.HTTPError(
        "https://api.runpod.io/?api_key=token-secret", 403, "secret response", {}, io.BytesIO(b"token-secret")
    )
    opener = Mock(side_effect=error)
    with pytest.raises(CloudError) as caught:
        Runpod("token-secret", opener).account()
    assert "token-secret" not in str(caught.value)
    req = opener.call_args.args[0]
    assert req.get_header("User-agent") == "home-observer/0.1"


def test_create_never_retries_ambiguous_response():
    api = Runpod("secret")
    api.graphql = Mock(side_effect=CloudError("network failed"))
    with pytest.raises(AmbiguousCreate):
        api.create({"name": "unique-job"})
    assert api.graphql.call_count == 1


def test_graphql_error_preserves_sanitized_diagnostic():
    api = Runpod("token-secret")
    api.request = Mock(return_value={"errors": [{"message": "Unknown type PodWrong; token-secret"}]})
    with pytest.raises(GraphQLError) as caught:
        api.account()
    assert "Unknown type PodWrong" in str(caught.value)
    assert "token-secret" not in str(caught.value)


def test_explicit_create_validation_is_rejected_not_ambiguous():
    api = Runpod("secret")
    api.graphql = Mock(side_effect=GraphQLError("Unknown argument requestedField"))
    with pytest.raises(CreateRejected, match="Unknown argument"):
        api.create({"name": "job"})
    assert api.graphql.call_count == 1


def test_unknown_gpu_counts_are_not_zero_capacity():
    api = quoted_api()
    api.quote.return_value["lowestPrice"]["availableGpuCounts"] = None
    assert preflight(api, JobConfig())["maximum_seconds"] > 0


def test_budget_includes_storage_and_cleanup_time():
    config = JobConfig(max_spend_usd=1, max_hours=2)
    result = preflight(quoted_api(), config)
    assert result["estimated_hourly_usd"] > 0.8
    assert result["maximum_seconds"] < 4500
    assert result["estimated_maximum_usd"] <= 1


@pytest.mark.parametrize(
    "config",
    [
        JobConfig(max_hours=float("nan")),
        JobConfig(max_spend_usd=-1),
        JobConfig(max_hours=0.1),
        JobConfig(volume_gb=0),
    ],
)
def test_invalid_budget_rejected_before_provider_calls(config):
    api = quoted_api()
    with pytest.raises(CloudError):
        preflight(api, config)
    api.account.assert_not_called()


@pytest.mark.parametrize("api", [quoted_api(price=10), quoted_api(balance=0), quoted_api(counts=[2])])
def test_no_capacity_or_balance_or_price_cannot_launch(api):
    with pytest.raises(CloudError):
        preflight(api, JobConfig())
    api.create.assert_not_called()


def test_payload_exposes_only_ssh_and_sets_provider_stop_deadline():
    payload = create_payload(JobConfig(), "job", "ssh-ed25519 AAAA test", 1800000000)
    assert payload["ports"] == "22/tcp"
    assert payload["stopAfter"] == "2027-01-15T08:00:00Z"
    assert payload["startJupyter"] is False
    assert "terminateAfter" not in payload  # preserves artifacts if local supervisor is unavailable
    assert {item["key"] for item in payload["env"]} == {"PUBLIC_KEY", "SSH_PUBLIC_KEY"}


def test_private_or_http_port_is_not_used_for_ssh():
    assert ssh_address({"runtime": {"ports": [{"privatePort": 22, "isIpPublic": False}]}}) is None
    assert ssh_address(
        {
            "runtime": {
                "ports": [
                    {
                        "privatePort": 22,
                        "isIpPublic": True,
                        "type": "tcp",
                        "ip": "192.0.2.1",
                        "publicPort": 22000,
                    }
                ]
            }
        }
    ) == ("192.0.2.1", 22000)


def test_release_bundle_excludes_secrets_and_previous_runs(tmp_path):
    project = tmp_path / "project"
    for name in ("src/home_observer", "work", "data", "artifacts"):
        (project / name).mkdir(parents=True)
    (project / "src/home_observer/code.py").write_text("pass")
    (project / "work/secret").write_text("token")
    (project / "data/.env").write_text("token")
    (project / "data/private.key").write_text("token")
    (project / "data/link").symlink_to(project / "work/secret")
    (project / "data/frames.jsonl").write_text("{}")
    (project / "requirements-vllm.txt").write_text("vllm[audio]==0.29.0\n")
    destination = tmp_path / "payload.tar.gz"
    assert len(package_project(project, destination)) == 64
    with tarfile.open(destination) as archive:
        assert set(archive.getnames()) == {
            "home-observer/src/home_observer/code.py",
            "home-observer/data/frames.jsonl",
            "home-observer/requirements-vllm.txt",
        }


def test_cleanup_deletes_only_after_artifacts_verified():
    api = Mock()
    assert finish_pod(api, "pod1", False) == "stopped_recovery_required"
    api.stop.assert_called_once_with("pod1")
    api.delete.assert_not_called()
    assert finish_pod(api, "pod1", True) == "terminated"
    api.delete.assert_called_once_with("pod1")


def test_cleanup_waits_for_provider_state_transition():
    api = Mock()
    api.get.side_effect = [{"desiredStatus": "RUNNING"}, {"desiredStatus": "EXITED"}]
    pause = Mock()
    assert confirm_cleanup(api, "pod1", False, pause=pause)
    assert api.get.call_count == 2
    pause.assert_called_once_with(1)


def test_cleanup_does_not_claim_termination_without_evidence():
    api = Mock()
    api.get.return_value = {"desiredStatus": "RUNNING"}
    assert not confirm_cleanup(api, "pod1", True, attempts=2, pause=lambda _: None)


def job_inputs(tmp_path):
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "scripts/runpod_job.sh").write_text("true")
    key = tmp_path / "key"
    key.write_text("private")
    Path(str(key) + ".pub").write_text("ssh-ed25519 AAAA test")
    return project, tmp_path / "run", key


def test_actual_rate_violation_stops_created_pod(tmp_path):
    project, directory, key = job_inputs(tmp_path)
    api = quoted_api()
    api.create.return_value = {"id": "pod1", "costPerHr": 20}
    api.get.return_value = {"id": "pod1", "desiredStatus": "EXITED"}
    with pytest.raises(CloudError, match="Actual Pod rate"):
        run_job(api, JobConfig(), project, directory, key)
    api.stop.assert_called_once_with("pod1")
    api.delete.assert_not_called()
    state = json.loads((directory / "state.json").read_text())
    assert state["cleanup_confirmed"] is True
    assert state["artifacts_verified"] is False


def test_ambiguous_create_is_durable_and_cannot_be_relaunched(tmp_path):
    project, directory, key = job_inputs(tmp_path)
    api = quoted_api()
    api.create.side_effect = AmbiguousCreate("uncertain")
    with pytest.raises(CloudError):
        run_job(api, JobConfig(), project, directory, key)
    assert json.loads((directory / "state.json").read_text())["status"] == "create_ambiguous"
    with pytest.raises(CloudError, match="already has"):
        run_job(api, JobConfig(), project, directory, key)
    api.create.assert_called_once()


def test_success_download_precedes_delete(tmp_path):
    project, directory, key = job_inputs(tmp_path)
    api = quoted_api()
    api.create.return_value = {"id": "pod1", "costPerHr": 0.8}
    api.get.side_effect = [
        {
            "id": "pod1",
            "desiredStatus": "RUNNING",
            "runtime": {
                "ports": [
                    {
                        "privatePort": 22,
                        "isIpPublic": True,
                        "type": "tcp",
                        "ip": "192.0.2.1",
                        "publicPort": 2222,
                    }
                ]
            },
        },
        None,
    ]
    fake_ssh = Mock()
    fake_ssh.host, fake_ssh.port = "192.0.2.1", 2222
    fake_ssh.tunnel_command.return_value = "ssh tunnel"
    fake_ssh.call.return_value = b"0"
    calls = []

    def retrieve_fake(*args):
        calls.append("download")
        return {"sha256": "0" * 64}

    api.delete.side_effect = lambda _: calls.append("delete")
    with (
        patch("home_observer.cloud.SSH", return_value=fake_ssh),
        patch("home_observer.cloud.retrieve", side_effect=retrieve_fake),
    ):
        result = run_job(api, JobConfig(), project, directory, key)
    assert calls == ["download", "delete"]
    assert result["status"] == "complete"
    assert result["cleanup_confirmed"] is True


def running_pod(host="192.0.2.1", port=2222):
    return {
        "id": "pod1",
        "desiredStatus": "RUNNING",
        "runtime": {
            "ports": [{"privatePort": 22, "isIpPublic": True, "type": "tcp", "ip": host, "publicPort": port}]
        },
    }


class FakeClock:
    def __init__(self):
        self.value = 1000.0

    def now(self):
        return self.value

    def pause(self, seconds):
        self.value += seconds


def status_ssh(*responses):
    connection = Mock()
    connection.host, connection.port = "192.0.2.1", 2222
    connection.call.side_effect = responses
    return connection


def test_transient_read_failure_recovers_without_any_mutation(tmp_path):
    api = Mock()
    api.get.return_value = running_pod()
    connection = status_ssh(CloudError("connection refused"), b"running")
    clock, state = FakeClock(), {"pod_id": "pod1"}
    result, returned = poll_remote_job(
        api, connection, state, tmp_path, "key", 2000, now=clock.now, pause=clock.pause
    )
    assert (result, returned) == ("running", connection)
    assert state["poll_error_count"] == state["poll_recoveries"] == 1
    assert state["poll_consecutive_errors"] == 0
    assert clock.value == 1015
    for call in connection.call.call_args_list:
        assert "kill -0" in call.args[0]
        assert "nohup" not in call.args[0] and "kill -TERM" not in call.args[0]
    api.create.assert_not_called()
    api.stop.assert_not_called()
    api.delete.assert_not_called()


def test_status_failure_refreshes_changed_ssh_endpoint(tmp_path):
    api = Mock()
    api.get.return_value = running_pod("192.0.2.2", 3333)
    old = status_ssh(CloudError("connection refused"))
    fresh = status_ssh(b"running")
    fresh.host, fresh.port = "192.0.2.2", 3333
    fresh.tunnel_command.return_value = "replacement tunnel"
    clock, state = FakeClock(), {"pod_id": "pod1"}
    with patch("home_observer.cloud.SSH", return_value=fresh) as factory:
        result, returned = poll_remote_job(
            api, old, state, tmp_path, "key", 2000, now=clock.now, pause=clock.pause
        )
    assert returned is fresh and result == "running"
    factory.assert_called_once_with("192.0.2.2", 3333, "key", tmp_path)
    assert state["ssh_endpoint_refreshes"] == 1
    assert state["ssh"]["port"] == 3333


def test_unavailable_provider_read_does_not_kill_job(tmp_path):
    api = Mock()
    api.get.side_effect = CloudError("transient API error")
    connection = status_ssh(CloudError("temporary SSH timeout"), b"running")
    clock, state = FakeClock(), {"pod_id": "pod1"}
    assert (
        poll_remote_job(api, connection, state, tmp_path, "key", 2000, now=clock.now, pause=clock.pause)[0]
        == "running"
    )
    assert state["poll_api_error_count"] == 1
    api.stop.assert_not_called()


@pytest.mark.parametrize("deadline, expected_time", [(2000, 1300), (1030, 1030)])
def test_status_retry_grace_is_bounded_by_deadline(tmp_path, deadline, expected_time):
    api = Mock()
    api.get.return_value = running_pod()
    connection = status_ssh()
    connection.call.side_effect = CloudError("connection refused")
    clock, state = FakeClock(), {"pod_id": "pod1"}
    with pytest.raises(CloudError, match="deadline|grace"):
        poll_remote_job(
            api,
            connection,
            state,
            tmp_path,
            "key",
            deadline,
            poll_seconds=50,
            now=clock.now,
            pause=clock.pause,
        )
    assert clock.value == expected_time
    assert state["poll_error_count"] >= 1
    api.stop.assert_not_called()  # Cleanup belongs to outer lifecycle, never the read retry helper.


def test_provider_terminal_state_is_not_hidden_by_retry_grace(tmp_path):
    api = Mock()
    api.get.return_value = {"id": "pod1", "desiredStatus": "EXITED"}
    connection = status_ssh(CloudError("connection refused"))
    clock, state = FakeClock(), {"pod_id": "pod1"}
    with pytest.raises(CloudError, match="terminal"):
        poll_remote_job(api, connection, state, tmp_path, "key", 2000, now=clock.now, pause=clock.pause)
    assert clock.value == 1000
    assert state["terminal_provider_status"] == "EXITED"


def test_missing_remote_job_process_is_terminal_evidence(tmp_path):
    api = Mock()
    connection = status_ssh(b"process_missing")
    clock, state = FakeClock(), {"pod_id": "pod1"}
    with pytest.raises(CloudError, match="process exited"):
        poll_remote_job(api, connection, state, tmp_path, "key", 2000, now=clock.now, pause=clock.pause)
    assert state["terminal_process_missing_at"] == 1000
    assert clock.value == 1000
    api.get.assert_not_called()


def test_ssh_diagnostic_retains_stderr_privately_and_redacts_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME_OBSERVER_API_TOKEN", "random-private-value")
    result = subprocess.CompletedProcess(
        [],
        255,
        stdout=b"stdout must not be recorded",
        stderr=b"Connection refused random-private-value; authorization: Bearer another-secret",
    )
    connection = SSH("192.0.2.1", 2222, "key", tmp_path)
    with patch("home_observer.cloud.subprocess.run", return_value=result):
        with pytest.raises(CloudError) as error:
            connection.call("private command must not be recorded")
    diagnostics = tmp_path / "ssh-errors.jsonl"
    text = diagnostics.read_text()
    assert "Connection refused" in text
    assert all(
        value not in text
        for value in ("random-private-value", "another-secret", "stdout must", "private command")
    )
    assert "Connection refused" not in str(error.value)
    assert diagnostics.stat().st_mode & 0o777 == 0o600


def test_lifecycle_does_not_relaunch_job_after_transient_status_failure(tmp_path):
    project, directory, key = job_inputs(tmp_path)
    api = quoted_api()
    api.create.return_value = {"id": "pod1", "costPerHr": 0.8}
    api.get.side_effect = [running_pod(), running_pod(), None]
    connection = status_ssh()
    connection.tunnel_command.return_value = "ssh tunnel"
    polls = []

    def call(command, **kwargs):
        if command.startswith("if [ -f /workspace/home-observer/artifacts/exit-code"):
            polls.append(command)
            if len(polls) == 1:
                raise CloudError("transient connection refused")
        return b"0"

    connection.call.side_effect = call
    with (
        patch("home_observer.cloud.SSH", return_value=connection),
        patch("home_observer.cloud.retrieve", return_value={"sha256": "0" * 64}),
    ):
        result = run_job(api, JobConfig(), project, directory, key, poll_seconds=0.001)
    assert result["status"] == "complete" and result["poll_recoveries"] == 1
    assert (
        sum("nohup bash /workspace/home-observer-run.sh" in c.args[0] for c in connection.call.call_args_list)
        == 1
    )
    api.create.assert_called_once()
    api.delete.assert_called_once()


def test_startup_api_read_error_does_not_delete_new_pod(tmp_path):
    project, directory, key = job_inputs(tmp_path)
    api = quoted_api()
    api.create.return_value = {"id": "pod1", "costPerHr": 0.8}
    api.get.side_effect = [CloudError("temporary control-plane error"), running_pod(), None]
    connection = status_ssh()
    connection.call.side_effect = None
    connection.call.return_value = b"0"
    connection.tunnel_command.return_value = "ssh tunnel"
    with (
        patch("home_observer.cloud.SSH", return_value=connection),
        patch("home_observer.cloud.retrieve", return_value={"sha256": "0" * 64}),
    ):
        result = run_job(api, JobConfig(), project, directory, key, poll_seconds=0.001)
    assert result["status"] == "complete" and result["startup_api_error_count"] == 1
    api.create.assert_called_once()
    api.delete.assert_called_once()
