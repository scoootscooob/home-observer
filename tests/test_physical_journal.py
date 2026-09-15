import copy

import pytest
from test_physical_schema import physical_decision, physical_window

from home_observer.journal import Journal
from home_observer.physical_journal import PhysicalJournal


def test_durable_registry_is_separate_and_replay_is_idempotent(tmp_path):
    path = tmp_path / "journal.sqlite"
    baseline = Journal(path)
    baseline.update_fact("light.kitchen", "state", "off", 1, 10, ["device:light.kitchen"], "device")
    registry = PhysicalJournal(path)
    mapping = registry.ingest(physical_window(), physical_decision())
    assert mapping["bowl-1"].startswith("phys_")
    assert registry.ingest(physical_window(), physical_decision()) == mapping
    assert registry.stats()["tracks"] == 1
    assert baseline.stats()["facts"] == 1
    assert baseline.stats()["events"] == 0
    event = registry.recent_events(11)[0]
    assert event["event_id"].startswith("pevt_")
    assert event["source_event_id"] == "placement"
    assert event["subject_ids"] == [mapping["bowl-1"]]
    assert event["kind"] == "put"
    assert registry.observations(11)[0]["subject_id"] == mapping["bowl-1"]
    assert registry.evidence(event["window_id"], ["after"])[0]["path"] == "after.jpg"
    registry.close()
    registry = PhysicalJournal(path)
    assert registry.tracks(11)[0]["track_id"] == mapping["bowl-1"]
    assert registry.recent_events(11)[0] == event
    registry.close()
    baseline.close()


def test_no_future_state_or_inferred_disappearance(tmp_path):
    registry = PhysicalJournal(tmp_path / "physical.sqlite")
    mapping = registry.ingest(physical_window(), physical_decision())
    later = physical_decision(12)
    later["objects"][0]["track_id"] = mapping["bowl-1"]
    later["objects"][0]["location"]["area"] = "sink"
    registry.ingest(physical_window("later", 12), later)
    assert registry.tracks(10) == []
    assert registry.tracks(11)[0]["last_detection"]["location"]["area"] == "counter"
    assert registry.tracks(13)[0]["last_detection"]["location"]["area"] == "sink"
    registry.ingest(physical_window("empty", 14), {"summary": "No additional physical claims."})
    track = registry.tracks(1000)[0]
    assert track["stale"] is True
    assert track["last_seen_at"] == 13
    assert track["last_detection"]["visibility"] == "partially_occluded"
    assert len(registry.recent_events(11)) == 1
    assert len(registry.recent_events(13)) == 2
    registry.close()


def test_conflicting_replay_and_failed_validation_leave_no_partial_state(tmp_path):
    registry = PhysicalJournal(tmp_path / "physical.sqlite")
    registry.ingest(physical_window(), physical_decision())
    before = registry.stats()
    altered = copy.deepcopy(physical_decision())
    altered["summary"] = "A conflicting interpretation."
    with pytest.raises(ValueError, match="different content"):
        registry.ingest(physical_window(), altered)
    invalid = physical_decision(12)
    invalid["events"][0]["post_evidence_ids"] = ["unknown"]
    with pytest.raises(ValueError, match="unknown physical evidence"):
        registry.ingest(physical_window("invalid", 12), invalid)
    assert registry.stats() == before
    registry.close()
