import numpy as np
import pytest
import torch
import cv2

from app.services.fight_detector import FightActionDetector


class ScoresModel:
    def __call__(self, _tensor):
        return torch.tensor([[0.0, 2.0]])


def make_detector(tmp_path, **kwargs):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"fixture")
    loader = kwargs.pop("model_loader", lambda _path, _device: (ScoresModel(), torch, "cpu"))
    return FightActionDetector(str(checkpoint), model_loader=loader, **kwargs)


def test_fighting_detector_without_model_is_honest():
    detector = FightActionDetector("")
    assert detector.status == "not_configured"
    events, result = detector.finish()
    assert events == []
    assert result["status"] == "not_configured"


def test_missing_fighting_weights_are_reported_without_crashing(tmp_path):
    detector = FightActionDetector(str(tmp_path / "missing.pt"))
    assert detector.status == "unavailable"
    assert "not found" in detector.error_message.lower()


def test_model_initialization_failure_is_unavailable(tmp_path):
    def fail(_path, _device):
        raise RuntimeError("fixture model init")

    detector = make_detector(tmp_path, model_loader=fail)
    assert detector.status == "unavailable"
    assert detector.finish()[1]["candidate_event_count"] == 0


def test_actual_x3d_checkpoint_safe_loads_and_forward_passes():
    """Exercise real weights when installed; absent local weights remain optional."""
    from pathlib import Path
    model_path = Path(__file__).resolve().parents[1] / "models" / "final_x3d_realtime.pt"
    if not model_path.is_file():
        pytest.skip("Approved optional fighting checkpoint is not present")
    detector = FightActionDetector(str(model_path), device="cpu")
    assert detector.status == "ready", detector.error_message
    detector.buffer.extend((np.zeros((224, 224, 3), dtype=np.uint8), i * .25, i * 8)
                           for i in range(16))
    score = detector._infer()
    assert 0.0 <= score <= 1.0


def test_temporal_windows_require_confirmation_and_merge(tmp_path):
    detector = make_detector(tmp_path)
    # Drive the score aggregation directly, without representing mock values as
    # real model outputs or persisting mock events.
    def window(start, end, score):
        detector.buffer.clear()
        detector.buffer.extend((None, t, i) for i, t in enumerate([start, end]))
        detector._record_window(score)

    window(0, 4, .72)
    assert detector.events == []
    window(1, 5, .81)
    window(2, 6, .63)
    events, summary = detector.finish()
    assert len(events) == 1
    assert events[0]["event_type"] == "SUSPECTED_FIGHTING"
    assert events[0]["start_timestamp_seconds"] == 0
    assert events[0]["end_timestamp_seconds"] == 6
    assert events[0]["model_score"] == .81
    assert events[0]["confidence"] is None and events[0]["anomaly_score"] is None
    assert events[0]["track_id"] is None
    assert events[0]["details"]["zone_association_status"] == "unavailable"
    assert summary["candidate_event_count"] == 1


def test_single_positive_window_does_not_create_incident(tmp_path):
    detector = make_detector(tmp_path)
    detector.buffer.extend((None, i * .25, i) for i in range(16))
    detector._record_window(.9)
    events, _summary = detector.finish()
    assert events == []


def test_short_video_reports_insufficient_video_not_negative(tmp_path):
    detector = make_detector(tmp_path)
    events, summary = detector.finish()
    assert events == []
    assert summary["status"] == "insufficient_video"


def test_video_processor_feeds_fight_detector_without_changing_zone_semantics(tmp_path):
    from app.services.video_processor import analyze_video

    source = tmp_path / "temporal.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer unavailable")
    for index in range(50):
        writer.write(np.full((48, 64, 3), index, dtype=np.uint8))
    writer.release()

    class FixtureTemporalDetector:
        METHOD = "fixture temporal classifier"

        def __init__(self):
            self.frames = 0

        def observe_frame(self, _frame, _timestamp, _frame_number):
            self.frames += 1

        def finish(self):
            event = {"event_type": "SUSPECTED_FIGHTING", "timestamp_seconds": 1.0,
                     "start_timestamp_seconds": 1.0, "end_timestamp_seconds": 3.0,
                     "model_score": .8, "confidence": None, "anomaly_score": None,
                     "track_id": None, "details": {}}
            return [event], {"status": "complete", "windows_analyzed": 2,
                             "candidate_event_count": 1}

    detector = FixtureTemporalDetector()
    zone = {"zone_id": "restricted", "name": "Restricted", "enabled": True,
            "points": [[.1, .1], [.9, .1], [.9, .9], [.1, .9]]}
    result = analyze_video(str(source), .1, 30, zones=[zone], fight_detector=detector,
                           zone_sample_seconds=.5)
    assert detector.frames == 50
    event = next(row for row in result["events"] if row["event_type"] == "SUSPECTED_FIGHTING")
    assert event["zone_association_status"] == "unavailable"
    assert event["details"]["zone_association_status"] == "unavailable"
    assert result["fight_analysis"]["candidate_event_count"] == 1

