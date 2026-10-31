import cv2
import numpy as np
import pytest

from app.services.lighting import assess_capture, enhance_low_light_frame
from app.services.movement_analysis import MovementAnalyzer, classify_pose_observation


class FakeCapture:
    def __init__(self, luminances):
        self.frames = [np.full((24, 32, 3), value, dtype=np.uint8) for value in luminances]
        self.position = 0

    def set(self, _property, value):
        self.position = int(value)
        return True

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame.copy()


def test_lighting_vote_requires_a_majority_of_dim_representative_frames():
    day = assess_capture(FakeCapture([180] * 7), 7)
    night = assess_capture(FakeCapture([24] * 7), 7)
    mixed = assess_capture(FakeCapture([20, 180, 20, 180, 20, 180, 20]), 7)
    assert day["detected_mode"] == day["lighting_mode"] == "DAY"
    assert day["night_enhancement_applied"] is False
    assert night["detected_mode"] == "LOW_LIGHT"
    assert night["lighting_mode"] == "NIGHT_VISION" and night["night_enhancement_applied"] is True
    assert mixed["detected_mode"] == "DAY"  # one bright minority prevents one-frame switching
    assert mixed["dark_sample_fraction"] == pytest.approx(4 / 7, abs=0.001)


def test_lighting_threshold_and_manual_modes_are_configurable():
    capture = FakeCapture([90] * 4)
    assert assess_capture(capture, 4, threshold=100)["detected_mode"] == "LOW_LIGHT"
    assert assess_capture(FakeCapture([20] * 4), 4, requested_mode="day")["lighting_mode"] == "DAY"
    assert assess_capture(FakeCapture([190] * 4), 4, requested_mode="night_vision")["lighting_mode"] == "NIGHT_VISION"
    with pytest.raises(ValueError, match="Lighting mode"):
        assess_capture(FakeCapture([50] * 4), 4, requested_mode="twilight")


def test_lighting_assessment_fails_truthfully_without_decoded_frames():
    class EmptyCapture(FakeCapture):
        def read(self):
            return False, None
    with pytest.raises(ValueError, match="representative video frames"):
        assess_capture(EmptyCapture([]), 0)


def test_night_enhancement_preserves_dimensions_and_does_not_mutate_source():
    source = np.full((48, 64, 3), 28, dtype=np.uint8)
    before = source.copy()
    enhanced = enhance_low_light_frame(source)
    assert enhanced.shape == source.shape
    assert enhanced.dtype == np.uint8
    assert not np.array_equal(enhanced, source)
    assert np.array_equal(source, before)
    with pytest.raises(ValueError, match="gamma"):
        enhance_low_light_frame(source, gamma=1.5)


def person_pose(track_id=4, x=20, torso="upright", crouched=False):
    # COCO-17 ordering, deliberately simple synthetic geometry for heuristic tests.
    coords = {
        "nose": (x, 8), "left_eye": (x - 1, 7), "right_eye": (x + 1, 7),
        "left_ear": (x - 2, 8), "right_ear": (x + 2, 8),
        "left_shoulder": (x - 5, 20), "right_shoulder": (x + 5, 20),
        "left_elbow": (x - 10, 27), "right_elbow": (x + 10, 27),
        "left_wrist": (x - 12, 35), "right_wrist": (x + 12, 35),
        "left_hip": (x - 4, 38), "right_hip": (x + 4, 38),
        "left_knee": (x - 4, 55), "right_knee": (x + 4, 55),
        "left_ankle": (x - 4, 75), "right_ankle": (x + 4, 75),
    }
    if crouched:
        coords.update({"left_shoulder": (x - 5, 60), "right_shoulder": (x + 5, 60),
                       "left_hip": (x - 4, 65), "right_hip": (x + 4, 65),
                       "left_knee": (x - 9, 78), "right_knee": (x + 9, 78),
                       "left_ankle": (x - 24, 80), "right_ankle": (x + 24, 80)})
    if torso == "horizontal":
        coords.update({"left_shoulder": (x - 8, 38), "right_shoulder": (x - 3, 41),
                       "left_hip": (x + 8, 40), "right_hip": (x + 13, 43)})
    return {"track_id": track_id, "keypoints": {
        name: {"x": px, "y": py, "confidence": .9} for name, (px, py) in coords.items()}}


def test_movement_uses_normalized_temporal_motion_and_reports_limb_measures():
    standing = person_pose()
    first = classify_pose_observation(standing, 100, 100, 0)
    assert first["category"] == "insufficient_temporal_evidence"
    moved = person_pose(x=42)
    walking = classify_pose_observation(moved, 100, 100, 1, first)
    assert walking["category"] == "walking_like_movement"
    assert walking["mean_limb_motion_normalized_per_second"] > 0
    assert walking["center_x_normalized"] > first["center_x_normalized"]
    running = classify_pose_observation(person_pose(x=90), 100, 100, 1, first)
    assert running["category"] == "running_like_movement"


def test_movement_segments_require_repeated_evidence_and_split_after_track_gap():
    analyzer = MovementAnalyzer(min_observations=2, max_observation_gap_seconds=1.5)
    analyzer.observe([person_pose()], 100, 100, 0, "DAY")
    analyzer.observe([person_pose()], 100, 100, 1, "DAY")
    analyzer.observe([person_pose()], 100, 100, 2, "DAY")
    first = analyzer.finish()
    assert first
    assert first[0]["category"] in {"low_movement", "standing"}
    assert first[0]["track_segment_id"] == "4:movement1"
    analyzer = MovementAnalyzer(min_observations=2, max_observation_gap_seconds=1.5)
    analyzer.observe([person_pose()], 100, 100, 0, "DAY")
    analyzer.observe([], 100, 100, 1, "DAY")
    analyzer.observe([person_pose()], 100, 100, 5, "NIGHT_VISION")
    analyzer.observe([person_pose()], 100, 100, 6, "NIGHT_VISION")
    analyzer.observe([person_pose()], 100, 100, 7, "NIGHT_VISION")
    second = analyzer.finish()
    assert second and second[-1]["track_segment_id"] == "4:movement2"
    assert second[-1]["lighting_mode"] == "NIGHT_VISION"


def test_movement_needs_confident_keypoints_and_never_returns_person_without_track():
    person = person_pose()
    person["track_id"] = None
    assert classify_pose_observation(person, 100, 100, 0) is None
    low_confidence = person_pose()
    for point in low_confidence["keypoints"].values():
        point["confidence"] = .05
    assert classify_pose_observation(low_confidence, 100, 100, 0) is None


def test_movement_posture_transition_is_an_observation_not_an_anomaly_event():
    analyzer = MovementAnalyzer(min_observations=2)
    analyzer.observe([person_pose()], 100, 100, 0, "DAY")
    analyzer.observe([person_pose(torso="horizontal", x=35)], 100, 100, 1, "DAY")
    analyzer.observe([person_pose(torso="horizontal", x=50)], 100, 100, 2, "DAY")
    rows = analyzer.finish()
    assert any(row["category"] in {"sudden_posture_change", "possible_fall_related_movement"} for row in rows)
    assert all("event_type" not in row for row in rows)


def test_crouching_and_standing_are_pose_observations_not_anomaly_incidents():
    crouch = person_pose(crouched=True)
    first = classify_pose_observation(crouch, 100, 100, 0)
    second = classify_pose_observation(crouch, 100, 100, 1, first)
    assert second["category"] == "crouching"
    standing = person_pose()
    first_stand = classify_pose_observation(standing, 100, 100, 0)
    second_stand = classify_pose_observation(standing, 100, 100, 1, first_stand)
    assert second_stand["category"] in {"standing", "low_movement"}
    assert "event_type" not in second and "event_type" not in second_stand


def test_video_pipeline_enhances_night_frames_but_preserves_day_override_and_source(tmp_path):
    from app.services.video_processor import analyze_video

    def write_clip(path, luminance):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
        if not writer.isOpened():
            pytest.skip("OpenCV MJPG test encoder is unavailable")
        for _ in range(10):
            writer.write(np.full((48, 64, 3), luminance, dtype=np.uint8))
        writer.release()

    class InspectPixels:
        METHOD = "synthetic pixel fixture"
        def __init__(self): self.means = []
        def inspect_with_evidence(self, frame, timestamp):
            self.means.append(float(np.mean(frame)))
            return frame, None, {"timestamp_seconds": timestamp, "person_count": 0, "people": []}

    night_path = tmp_path / "synthetic-low-light.avi"
    write_clip(night_path, 24)
    original_bytes = night_path.read_bytes()
    night_detector = InspectPixels()
    night = analyze_video(str(night_path), .2, 30, night_detector,
                          requested_lighting_mode="auto", lighting_sample_frames=5)
    assert night["lighting"]["detected_mode"] == "LOW_LIGHT"
    assert night["lighting"]["lighting_mode"] == "NIGHT_VISION"
    assert night["night_enhancement_applied"] is True
    assert night_detector.means and min(night_detector.means) > 24
    assert night["stage_timings"]["night_enhancement_seconds"] > 0
    assert night_path.read_bytes() == original_bytes

    day_path = tmp_path / "manual-day.avi"
    write_clip(day_path, 24)
    day_detector = InspectPixels()
    day = analyze_video(str(day_path), .2, 30, day_detector,
                        requested_lighting_mode="day", lighting_sample_frames=5)
    assert day["lighting"]["detected_mode"] == "LOW_LIGHT"
    assert day["lighting"]["lighting_mode"] == "DAY"
    assert day["night_enhancement_applied"] is False
    assert max(day_detector.means) < 35


def test_low_light_validation_only_still_publishes_enhanced_output_without_pose_model(tmp_path):
    from app.services.video_processor import analyze_video

    source = tmp_path / "low-light-validation-only.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG test encoder is unavailable")
    for _ in range(5):
        writer.write(np.full((48, 64, 3), 22, dtype=np.uint8))
    writer.release()
    output = tmp_path / "enhanced-validation-only.webm"
    result = analyze_video(str(source), 1.0, 10, detector=None, output_path=str(output),
                           requested_lighting_mode="auto", lighting_sample_frames=5)
    assert result["lighting"]["lighting_mode"] == "NIGHT_VISION"
    assert result["pose_analysis"]["status"] == "not_configured"
    assert result["annotated_output"] is True
    assert result["annotated_output_metadata"]["container"] == ".webm"
    assert output.is_file() and output.stat().st_size > 0
