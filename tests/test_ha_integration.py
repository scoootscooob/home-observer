"""Opt-in tests against this project's actual disposable Home Assistant server."""
import importlib.util
import os
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ha_demo", ROOT / "scripts/ha_demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)

@pytest.fixture(scope="module")
def work():
    if os.environ.get("HA_DEMO_TEST") != "1":
        pytest.skip("Set HA_DEMO_TEST=1 after scripts/ha_demo.py start; never targets a user's HA")
    path = Path(os.environ.get("HA_DEMO_WORK", demo.DEFAULT_WORK))
    demo.demo_credentials(path)  # verifies the exact container ownership label first
    return path


def test_demo_container_cannot_use_host_devices_or_public_listener(work):
    container = demo.inspect_owned("container", demo.CONTAINER)
    config = container["HostConfig"]
    assert config["Privileged"] is False
    assert not config.get("Devices")
    assert config["NetworkMode"] == demo.NETWORK
    assert config["PortBindings"] == {"8123/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8124"}]}
    assert len(config["Binds"]) == 1
    assert str(work.resolve() / "config") in config["Binds"][0]
    assert (work / "credentials.json").stat().st_mode & 0o777 == 0o600


def test_real_rest_light_and_switch_service_calls_and_state_reads(work):
    report = demo.verify(work)
    assert report["status"] == "passed"
    assert report["home_assistant_version"] == "2026.9.2"
    assert len(report["service_calls"]) == 15
    assert all(item["verified"] for item in report["service_calls"])
    assert {item["entity_id"] for item in report["service_calls"]} == set(demo.ENTITIES)
    assert all(item["simulated"] is False for item in report["service_calls"])


def test_http_api_requires_a_real_token_and_discovery_is_disabled(work):
    assert httpx.get(demo.BASE_URL + "/api/states", timeout=5).status_code == 401
    token = demo.demo_credentials(work)["long_lived_token"]
    response = httpx.get(demo.BASE_URL + "/api/config", headers={"Authorization": "Bearer " + token}, timeout=5)
    response.raise_for_status()
    components = set(response.json()["components"])
    assert not components & {"ssdp", "zeroconf", "bluetooth", "dhcp", "discovery", "homekit_controller"}


def test_readback_event_lag_never_repeats_the_service(monkeypatch):
    from home_observer.ha import HomeAssistant
    from home_observer.schema import Action
    adapter = HomeAssistant("http://127.0.0.1:8124", "test-token", execute_real=True)
    adapter.close()
    adapter.client = Mock()
    stale, fresh = Mock(), Mock()
    stale.json.return_value = {"state": "off"}
    fresh.json.return_value = {"state": "on"}
    adapter.client.get.side_effect = [stale, fresh]
    monkeypatch.setattr("home_observer.ha.time.sleep", lambda _: None)
    result = adapter.execute(Action(domain="light", service="turn_on", entity_id="light.kitchen",
        reason="test", evidence_ids=["test"]), "test")
    assert result["verified"] is True
    assert adapter.client.post.call_count == 1
    assert adapter.client.get.call_count == 2


def test_readback_deadline_records_unverified_without_retry():
    from home_observer.ha import HomeAssistant
    from home_observer.schema import Action
    adapter = HomeAssistant("http://127.0.0.1:8124", "test-token", execute_real=True, verification_timeout_s=0)
    adapter.close()
    adapter.client = Mock()
    adapter.client.get.return_value.json.return_value = {"state": "off"}
    result = adapter.execute(Action(domain="light", service="turn_on", entity_id="light.kitchen",
        reason="test", evidence_ids=["test"]), "test")
    assert result["verified"] is False
    assert adapter.client.post.call_count == adapter.client.get.call_count == 1


def test_state_reads_send_only_configured_entities_and_needed_attributes():
    from home_observer.ha import HomeAssistant
    adapter = HomeAssistant('http://127.0.0.1:8124', 'test-token', execute_real=True)
    adapter.close()
    adapter.client = Mock()
    adapter.client.get.return_value.json.return_value = [
        {'entity_id': 'light.kitchen', 'state': 'on', 'attributes': {'brightness': 100, 'friendly_name': 'private name'}},
        {'entity_id': 'sensor.ambient_lux', 'state': '12', 'attributes': {'unit_of_measurement': 'lx'}},
        {'entity_id': 'person.unconfigured', 'state': 'home', 'attributes': {'latitude': 1}},
    ]
    states = adapter.read_states(['light.kitchen', 'sensor.ambient_lux', 'light.missing'])
    assert set(states) == {'light.kitchen', 'sensor.ambient_lux'}
    assert states['light.kitchen'] == {'state': 'on', 'attributes': {'brightness': 100}}
    assert states['sensor.ambient_lux']['state'] == '12'
