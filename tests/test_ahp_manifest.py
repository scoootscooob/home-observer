"""The AHP self-description is derived from the enforced production schemas and carries no media."""
import json

from home_observer.ahp_manifest import build_manifest, manifest_is_media_free
from home_observer.production_export import DENIED_KEYS


def test_manifest_events_and_commands_mirror_the_production_boundary():
    manifest = build_manifest()
    resource = manifest["resources"][0]
    assert {event["id"] for event in resource["events"]} == {
        "physical_event", "watch_status", "perception_status", "verification_result"}
    assert {command["id"] for command in resource["commands"]} == {"create_watch", "cancel_watch", "verify", "ack_export"}
    physical = next(event for event in resource["events"] if event["id"] == "physical_event")
    assert set(physical["payload"]["properties"]) >= {"event_id", "kind", "object_category", "subject_ids",
                                                       "confidence", "uncertainty_level", "audit_ref"}
    assert not (set(physical["payload"]["properties"]) & DENIED_KEYS)
    assert physical["policy"]["wake"] is True and physical["policy"]["max_rate_per_min"] <= 120
    assert manifest_is_media_free(manifest) == []
    text = json.dumps(manifest)
    for marker in ("content_base64", "\"path\"", "raw_output", "manifest.json", "/Users/"):
        assert marker not in text
    verify = next(command for command in resource["commands"] if command["id"] == "verify")
    assert set(verify["input"]["properties"]["question"]["enum"]) == {
        "moved_from_initial_location", "clear_of_support", "still_present", "pickup_completed"}
