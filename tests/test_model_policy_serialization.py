"""Policy object reconstruction must not change trained prompt token order."""
import copy
import json

from home_observer.policy import Policy
from home_observer.prompts import POLICY_FIELD_ORDER, build_context, ordered_policy


def test_runtime_policy_prompt_matches_authored_training_order():
    authored = {"entities": ["light.kitchen"], "allowed_services": ["light.turn_on"],
                "rules": ["Only when current occupancy is true."], "min_confidence": 0.8,
                "cooldown_seconds": 30, "state_ttl_seconds": 300,
                "max_actions_per_window": 2, "max_window_seconds": 60}
    runtime = Policy(**authored).as_dict()
    assert authored == runtime
    assert list(authored) != list(runtime)  # The historical drift exercised by this test.
    request = {"window": {"window_id": "one", "started_at": 1.0, "ended_at": 2.0},
               "policy": authored}
    other = {**request, "policy": runtime}
    snapshot = copy.deepcopy(other)
    assert build_context(request) == build_context(other)
    assert list(json.loads(build_context(other))["policy"]) == list(POLICY_FIELD_ORDER)
    assert other == snapshot


def test_unknown_policy_fields_are_preserved_in_stable_order():
    first = {"z_extension": ["leave intact"], "rules": [], "a_extension": {"enabled": True}}
    second = dict(reversed(list(first.items())))
    assert ordered_policy(first) == first
    assert list(ordered_policy(first)) == ["rules", "a_extension", "z_extension"]
    assert json.dumps(ordered_policy(first)) == json.dumps(ordered_policy(second))
