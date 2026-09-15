# Real Home Assistant integration with isolated demo devices

`scripts/ha_demo.py` starts the official Home Assistant **2026.9.2** image pinned by platform digest, completes local onboarding, and exercises the project's `HomeAssistant` adapter through the actual REST service and state endpoints.

The server runs at [http://127.0.0.1:8124](http://127.0.0.1:8124). It contains four template lights—`light.kitchen`, `light.entry`, `light.lounge`, and `light.garage`—and `switch.buzzer`. Each has an `input_boolean` backing state. Fixture motion/leak sensors and ambient illuminance are also available. These are simulated devices hosted by the real Home Assistant software.

The container uses a project-specific Docker bridge and a loopback port binding. It has no host networking, privileged mode, device mounts, or discovery integrations. It does not connect to an existing Home Assistant account or real household equipment. The only bind mount is this demo's own configuration directory.

## Start and verify

Install the project's normal Python dependencies and ensure Docker is running. From this project:

```bash
python scripts/ha_demo.py start --work-dir work/ha-demo --report reports/home-assistant-integration.json
HA_DEMO_TEST=1 python -m pytest tests/test_ha_integration.py -q
```

`start` is idempotent and refuses to operate on a same-named container or network without the project's ownership label. It creates `home-observer-ha-demo-20260915`, its network, and local configuration. Both AMD64 and ARM64 images have verified pinned digests in `configs/ha-demo.json`.

The script generates a demo login, obtains an access token using Home Assistant's local onboarding/OAuth flow, and creates a 30-day token through its WebSocket API. Credentials stay in `work/ha-demo/credentials.json`, mode 600. The script prints the credential-file path, never the credentials themselves. Do not copy this file into the project bundle or reports.

Verification runs `turn_off`, `turn_on`, and `turn_off` on every demo light/switch using `HomeAssistant.execute`, then independently reads the entity's real Home Assistant state. The report contains the observed states, timestamps, software version, image digest, and service-call results without tokens. A `simulated:false` adapter result means the call used the actual HA API; the report separately identifies that the controlled entities are simulated devices.

## Connect the learned observer

Read the local demo token from the credential file in Python, then construct:

```python
import json
from pathlib import Path
from home_observer.ha import HomeAssistant

credentials = json.loads(Path("work/ha-demo/credentials.json").read_text())
ha = HomeAssistant("http://127.0.0.1:8124", credentials["long_lived_token"], execute_real=True)
```

Pass that adapter into the observer engine. The policy must still restrict the permitted entities and services. The integration test uses `light.turn_off` to reset devices. The default service allowlist includes it, but the model's rule text does not direct the observer to turn lights off merely because occupancy is unknown.

The deterministic checks enforce entity/service permissions, supported arguments,
evidence-ID existence, action reservations and state readback. They do not prove
that an image supports a claimed fact or execute the natural-language policy
rules as code. An accepted action can therefore be semantically wrong. The held-out
decision evaluations measure that separate failure mode; readback establishes
execution, not correctness of the model's reason for acting.

## Lifecycle

```bash
python scripts/ha_demo.py status --work-dir work/ha-demo
python scripts/ha_demo.py verify --work-dir work/ha-demo
python scripts/ha_demo.py stop --work-dir work/ha-demo
```

The container remains running after verification for the model-to-device demo. It has no restart policy and can be stopped with the final command. Configuration remains in the work directory for reproducibility.

## Sources

The implementation follows the official [Home Assistant container documentation](https://www.home-assistant.io/installation/linux), [template integration](https://www.home-assistant.io/integrations/template/), [REST API](https://developers.home-assistant.io/docs/api/rest/), and the pinned release's [onboarding source](https://github.com/home-assistant/core/blob/2026.9.2/homeassistant/components/onboarding/views.py) and [token implementation](https://github.com/home-assistant/core/blob/2026.9.2/homeassistant/components/auth/__init__.py). This Docker setup is an isolated software integration test; the project's real-home installation should follow Home Assistant's supported deployment guidance.
