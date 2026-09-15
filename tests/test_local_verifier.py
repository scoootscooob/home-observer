"""The local verifier answers from local journals only; unseen is never gone."""
import pytest

from home_observer.local_verifier import LocalVerifier, VerificationRequest, VerifierPolicy
from home_observer.physical_journal import PhysicalJournal
from home_observer.temporal_buffer import CapturedFrame
from home_observer.tracking_journal import TrackingJournal

MAPPING = {"source_origin": 0.0, "wall_origin": 1000.0, "scale": 1.0}
INITIAL = [0.4, 0.4, 0.6, 0.7]


def frame(index):
    return CapturedFrame("cam", index, round(index * 0.1, 3), 1000 + index * 0.1, b"jpeg-bytes-%d" % index)


def journals(tmp_path):
    journal = PhysicalJournal(tmp_path / "journal.sqlite3")
    tracking = TrackingJournal(journal.path)
    window = {"window_id": "bootstrap", "started_at": 0.0, "ended_at": 0.0, "clips": [{
        "clip_id": "bootstrap_clip", "camera_id": "cam", "started_at": 0.0, "ended_at": 0.0,
        "frames": [{"evidence_id": "cam:0", "timestamp": 0.0, "path": "bootstrap/first.jpg"}]}]}
    decision = {"summary": "bootstrap", "objects": [{
        "detection_id": "seed", "label": "bowl", "box": dict(zip(("x_min", "y_min", "x_max", "y_max"), INITIAL)),
        "location": {"camera_id": "cam"}, "timestamp": 0.0, "frame_evidence_id": "cam:0",
        "visibility": "visible", "confidence": 0.9, "uncertainty": "seed", "evidence_ids": ["cam:0"]}]}
    track = journal.ingest(window, decision)["seed"]
    return journal, tracking, track


def record(tracking, track, boxes, *, start=1, lost_from=None):
    for offset, box in enumerate(boxes):
        index = start + offset
        state = {"track_id": track, "timestamp": frame(index).timestamp, "evidence_id": frame(index).evidence_id,
                 "visibility": "visible", "box": box, "confidence": 0.8}
        if lost_from is not None and index >= lost_from:
            state.update(visibility="uncertain", box=None, confidence=0.0, failure_reason="appearance_changed")
        tracking.record([state], frame(index), method="test")


def moved_boxes(count=12):
    return [[0.4 - 0.03 * i, 0.4 - 0.03 * i, 0.6 - 0.03 * i, 0.7 - 0.03 * i] for i in range(count)]


def learned_event(journal, track, kind, *, start=0.6, end=0.9, window_id="learned"):
    window = {"window_id": window_id, "started_at": 0.5, "ended_at": 0.9, "clips": [{
        "clip_id": window_id + "_clip", "camera_id": "cam", "started_at": 0.5, "ended_at": 0.9,
        "frames": [{"evidence_id": window_id + ":5", "timestamp": 0.5, "path": "a.jpg"},
                   {"evidence_id": window_id + ":9", "timestamp": 0.9, "path": "b.jpg"}]}]}
    decision = {"summary": kind, "events": [{
        "event_id": "e1", "kind": kind, "object_label": "bowl", "subject_ids": [track], "started_at": start,
        "ended_at": end, "description": kind + " bowl", "confidence": 0.7, "uncertainty": "model",
        "pre_evidence_ids": [window_id + ":5"], "post_evidence_ids": [window_id + ":9"]}]}
    journal.ingest(window, decision)


def request(track, question, **changes):
    return VerificationRequest(request_id="q-1", question=question, subject_id=track, **changes)


def test_geometric_displacement_confirms_movement_but_not_pickup(tmp_path):
    journal, tracking, track = journals(tmp_path)
    record(tracking, track, moved_boxes())
    verifier = LocalVerifier(journal, tracking, clock_mapping=MAPPING)
    moved = verifier.verify(request(track, "moved_from_initial_location"), now=1001.3)
    assert (moved["verdict"], moved["reason_code"]) == ("confirmed", "geometric_displacement")
    assert moved["audit"]["center_displacement"] > 0.15 and moved["frames_examined"] == 12
    assert moved["evidence_span"] == {"started_at": 1000.1, "ended_at": pytest.approx(1001.2)}
    assert moved["verification_id"].startswith("pver_")
    pickup = verifier.verify(request(track, "pickup_completed"), now=1001.3)
    assert (pickup["verdict"], pickup["reason_code"]) == ("unknown", "learned_model_unavailable")
    clear = verifier.verify(request(track, "clear_of_support"), now=1001.3)
    assert clear["verdict"] == "unknown"


def test_learned_agreement_contradiction_and_invalid_checks(tmp_path):
    journal, tracking, track = journals(tmp_path)
    record(tracking, track, moved_boxes())
    calls = []

    def check(subject, question, source_as_of, lookback):
        calls.append((subject, question))
        return {"value": True, "confidence": 0.6, "audit": {"frames": ["cam:11", "cam:12"]}}

    verifier = LocalVerifier(journal, tracking, clock_mapping=MAPPING, learned_check=check)
    result = verifier.verify(request(track, "clear_of_support"), now=1001.3)
    assert (result["verdict"], result["reason_code"]) == ("confirmed", "geometric_and_learned_agreement")
    assert result["confidence"] == pytest.approx(0.6)
    assert calls == [(track, "clear_of_support")]
    verifier.learned_check = lambda *args: {"value": False, "confidence": 0.7}
    result = verifier.verify(request(track, "pickup_completed"), now=1001.3)
    assert (result["verdict"], result["reason_code"]) == ("contradicted", "learned_evidence_contradicts")
    verifier.learned_check = lambda *args: {"value": None, "confidence": 0.0}
    result = verifier.verify(request(track, "pickup_completed"), now=1001.3)
    assert (result["verdict"], result["reason_code"]) == ("unknown", "requires_learned_evidence")

    def broken(*args):
        raise RuntimeError("model crashed reading /Users/private/frame.jpg")

    verifier.learned_check = broken
    result = verifier.verify(request(track, "pickup_completed"), now=1001.3)
    assert (result["verdict"], result["reason_code"]) == ("unknown", "learned_output_invalid")
    assert result["audit"]["learned_error_type"] == "RuntimeError"


def test_journaled_learned_events_decide_without_a_model_call(tmp_path):
    journal, tracking, track = journals(tmp_path)
    record(tracking, track, moved_boxes())
    learned_event(journal, track, "pick-up")
    verifier = LocalVerifier(journal, tracking, clock_mapping=MAPPING,
                             learned_check=lambda *a: pytest.fail("journal evidence must be used first"))
    result = verifier.verify(request(track, "pickup_completed"), now=1001.3)
    assert (result["verdict"], result["reason_code"]) == ("confirmed", "geometric_and_learned_agreement")
    assert result["confidence"] == pytest.approx(0.7 * 0 + min(0.72, 0.7), abs=0.05)
    learned_event(journal, track, "put-down", start=0.7, end=0.9, window_id="later")
    result = verifier.verify(request(track, "pickup_completed"), now=1001.3)
    assert (result["verdict"], result["reason_code"]) == ("contradicted", "learned_evidence_contradicts")


def test_no_displacement_contradicts_and_lost_localization_stays_unknown(tmp_path):
    journal, tracking, track = journals(tmp_path)
    record(tracking, track, [INITIAL] * 10)
    verifier = LocalVerifier(journal, tracking, clock_mapping=MAPPING)
    result = verifier.verify(request(track, "moved_from_initial_location"), now=1001.1)
    assert (result["verdict"], result["reason_code"]) == ("contradicted", "no_displacement_observed")
    result = verifier.verify(request(track, "pickup_completed"), now=1001.1)
    assert result["verdict"] == "contradicted"
    journal2, tracking2, track2 = journals(tmp_path / "second")
    record(tracking2, track2, moved_boxes(), lost_from=9)
    verifier2 = LocalVerifier(journal2, tracking2, clock_mapping=MAPPING)
    result = verifier2.verify(request(track2, "moved_from_initial_location"), now=1001.3)
    assert (result["verdict"], result["reason_code"]) == ("unknown", "localization_uncertain")
    result = verifier2.verify(request(track2, "pickup_completed"), now=1001.3)
    assert result["verdict"] == "unknown"


def test_presence_never_becomes_absence_from_missing_updates(tmp_path):
    journal, tracking, track = journals(tmp_path)
    record(tracking, track, [INITIAL] * 5)
    verifier = LocalVerifier(journal, tracking, clock_mapping=MAPPING, policy=VerifierPolicy(stale_after_s=1.0))
    recent = verifier.verify(request(track, "still_present"), now=1000.6)
    assert (recent["verdict"], recent["reason_code"]) == ("confirmed", "visible_in_window")
    stale = verifier.verify(request(track, "still_present"), now=1010.0)
    assert (stale["verdict"], stale["reason_code"]) == ("unknown", "not_observed_in_window")
    unknown = verifier.verify(request("phys_" + "f" * 32, "moved_from_initial_location"), now=1001.0)
    assert (unknown["verdict"], unknown["reason_code"]) == ("unknown", "subject_unknown")
    missing = verifier.verify(request(track, "moved_from_initial_location", reference_event_id="pevt_missing"),
                              now=1001.0)
    assert (missing["verdict"], missing["reason_code"]) == ("unknown", "reference_event_unknown")
    short = verifier.verify(request(track, "moved_from_initial_location", lookback_s=0.05), now=1000.5)
    assert short["verdict"] == "unknown"


def test_request_and_policy_are_typed_and_bounded():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        VerificationRequest(request_id="q", question="describe_scene", subject_id="phys_" + "a" * 32)
    with pytest.raises(ValidationError):
        VerificationRequest(request_id="q", question="still_present", subject_id="light.kitchen")
    with pytest.raises(ValidationError):
        VerificationRequest(request_id="q", question="still_present", subject_id="phys_a", lookback_s=10000)
    with pytest.raises(ValidationError):
        VerificationRequest(request_id="q", question="still_present", subject_id="phys_a", frames=["/x.jpg"])
    with pytest.raises(ValueError):
        LocalVerifier(None, None, clock_mapping={"source_origin": 0, "wall_origin": 1, "scale": 0})


def test_questions_after_the_newest_frame_use_latest_evidence_and_record_staleness(tmp_path):
    journal, tracking, track = journals(tmp_path)
    record(tracking, track, moved_boxes())
    verifier = LocalVerifier(journal, tracking, clock_mapping=MAPPING, latest_source_time=lambda: 1.2)
    late = verifier.verify(request(track, "moved_from_initial_location"), now=1061.2)
    assert (late["verdict"], late["reason_code"]) == ("confirmed", "geometric_displacement")
    assert late["audit"]["evidence_stale_s"] == pytest.approx(60.0)
    assert late["evidence_span"]["ended_at"] == pytest.approx(1001.2)
    without = LocalVerifier(journal, tracking, clock_mapping=MAPPING)
    result = without.verify(request(track, "moved_from_initial_location"), now=1061.2)
    assert (result["verdict"], result["reason_code"]) == ("unknown", "insufficient_tracking_history")
