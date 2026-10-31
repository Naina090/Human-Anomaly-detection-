import math

import cv2
import numpy as np
import pytest

from app.services.motion_geometry import (
    JOINT_DEFINITIONS, angle_at, pose_geometry, segment_orientation,
    signed_angular_difference,
)
from app.services.motion_geometry import MotionHistory
from app.services.video_processor import _draw_motion_overlay
from app.services.fall_detector import _pose_overlay


def pose(track_id=1, shift_x=0, shift_y=0, elbow_bend=0, confidence=0.9):
    points = {
        "nose": (50, 5), "left_eye": (48, 4), "right_eye": (52, 4),
        "left_ear": (46, 5), "right_ear": (54, 5),
        "left_shoulder": (40 + shift_x, 20), "right_shoulder": (60 + shift_x, 20),
        "left_elbow": (30 + shift_x, 30), "right_elbow": (70 + shift_x, 30),
        "left_wrist": (20 + shift_x, 20 + elbow_bend), "right_wrist": (80 + shift_x, 20),
        "left_hip": (42 + shift_x, 50), "right_hip": (58 + shift_x, 50),
        "left_knee": (42 + shift_x, 70), "right_knee": (58 + shift_x, 70),
        "left_ankle": (42 + shift_x, 90), "right_ankle": (58 + shift_x, 90),
    }
    return {"track_id": track_id, "keypoints": {
        name: {"x": x, "y": y + shift_y, "confidence": confidence}
        for name, (x, y) in points.items()}}


def test_interior_joint_geometry_angles_and_invalid_values():
    assert angle_at((0, 0), (0, 1), (1, 1)) == pytest.approx(90)
    assert angle_at((0, 0), (1, 0), (2, 0)) == pytest.approx(180)
    assert angle_at((1, 0), (0, 0), (1, 1)) == pytest.approx(45)
    assert angle_at((1, 0), (0, 0), (-1, 1)) == pytest.approx(135)
    assert angle_at((0, 0), (0, 0), (1, 1)) is None
    assert angle_at((float("nan"), 0), (0, 0), (1, 1)) is None
    assert angle_at((0, 0), (1, 0), (1, 1)) == pytest.approx(90)


def test_wrapped_orientation_difference_uses_shortest_signed_path():
    assert segment_orientation((0, 0), (-1, 0)) == pytest.approx(180)
    assert signed_angular_difference(-179, 179) == pytest.approx(2)
    assert signed_angular_difference(179, -179) == pytest.approx(-2)
    first = pose_geometry(pose(), 100, 100, 0)[1]
    first["angle_history"] = {"shoulder_line": [179.0, -179.0]}
    wrapped = pose()
    wrapped["keypoints"]["left_shoulder"].update(x=60, y=20)
    wrapped["keypoints"]["right_shoulder"].update(x=40, y=19.65)
    current = pose_geometry(wrapped, 100, 100, 1, first)[0]
    assert abs(abs(current["joint_measurements"]["shoulder_line"]["smoothed_angle_degrees"]) - 180) < 2


def test_pose_geometry_maps_coco_joints_and_reports_2d_orientation():
    sample = pose_geometry(pose(), 100, 100, 0)[0]
    assert set(JOINT_DEFINITIONS).issubset(sample["joint_measurements"])
    assert "torso_inclination" in sample["joint_measurements"]
    assert "shoulder_line" in sample["joint_measurements"]
    assert sample["body_center_reference"] == "midpoint_between_visible_hips"
    assert sample["movement_speed_pixels_per_second"] is None
    assert sample["joint_measurements"]["left_elbow"]["raw_angle_degrees"] is not None


def test_pose_geometry_confidence_and_missing_keypoints_remain_unavailable():
    person = pose(confidence=0.1)
    row = pose_geometry(person, 100, 100, 0, confidence_threshold=0.3)
    assert row is None
    person = pose()
    del person["keypoints"]["left_wrist"]
    sample = pose_geometry(person, 100, 100, 0)[0]
    assert "left_elbow" not in sample["joint_measurements"]
    assert "right_elbow" in sample["joint_measurements"]


def test_speed_uses_video_time_and_image_coordinate_direction():
    first = pose_geometry(pose(shift_x=10), 100, 100, 1)[1]
    moved = pose_geometry(pose(shift_x=20), 100, 100, 3, first)[0]
    assert moved["elapsed_seconds"] == pytest.approx(2)
    assert moved["movement_speed_pixels_per_second"] == pytest.approx(5)
    assert moved["movement_direction_degrees"] == pytest.approx(0)
    vertical = pose_geometry(pose(), 100, 100, 4, pose_geometry(pose(), 100, 100, 1)[1])[0]
    assert vertical["movement_direction_degrees"] is None


def test_frame_overlay_draws_only_current_valid_joint_measurements_and_track_panel():
    frame = np.zeros((240, 200, 3), dtype=np.uint8)
    person = pose()
    person.update({"track_id": 7, "bbox_xyxy": [30, 15, 90, 110]})
    metric = {"track_id": 7, "joint_measurements": {
        "left_elbow": {"valid": True, "smoothed_angle_degrees": 142.5},
        "right_knee": {"valid": True, "smoothed_angle_degrees": 153.7},
        "right_hip": {"valid": False, "smoothed_angle_degrees": 90.0},
        "torso_inclination": {"valid": True, "smoothed_angle_degrees": 12.0},
    }, "movement_speed_pixels_per_second": 22.4,
        "movement_direction_degrees": 45.0}
    annotated, labels, panels = _draw_motion_overlay(
        frame.copy(), [person], [metric], {7: "walking"}, 1.25, "NIGHT_VISION")
    assert labels == 3  # left elbow, right knee, and torso inclination.
    assert panels == 1
    assert np.any(annotated != frame)


def test_frame_overlay_does_not_reuse_a_measurement_for_a_different_track():
    frame = np.zeros((240, 200, 3), dtype=np.uint8)
    person = pose()
    person.update({"track_id": 8, "bbox_xyxy": [30, 15, 90, 110]})
    stale = {"track_id": 7, "joint_measurements": {
        "left_elbow": {"valid": True, "smoothed_angle_degrees": 142.5}}}
    annotated, labels, panels = _draw_motion_overlay(frame.copy(), [person], [stale], {}, 0, "DAY")
    assert labels == 0
    assert panels == 0
    # The frame banner is still drawn, but no person measurement is fabricated.
    assert np.any(annotated != frame)


def test_person_box_and_track_id_remain_in_analysis_output_with_skeleton_only_setting():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    points = np.zeros((1, 17, 3), dtype=np.float32)
    confidences = np.zeros((1, 17), dtype=np.float32)
    annotated = _pose_overlay(frame, points, confidences,
                             np.array([[10, 10, 80, 90]], dtype=np.float32),
                             np.array([0.9]), np.array([4]), overlay_mode="skeleton_only")
    assert np.any(annotated[10, 10])
    assert np.any(annotated != frame)


def test_vertical_diagonal_and_stationary_speed_use_image_coordinates():
    first = pose_geometry(pose(), 100, 100, 2)[1]
    downward = pose_geometry(pose(shift_y=10), 100, 100, 4, first)[0]
    assert downward["movement_speed_pixels_per_second"] == pytest.approx(5)
    assert downward["movement_direction_degrees"] == pytest.approx(90)
    first = pose_geometry(pose(shift_x=10), 100, 100, 0)[1]
    diagonal = pose_geometry(pose(shift_x=20, shift_y=10), 100, 100, 2, first)[0]
    assert diagonal["movement_speed_pixels_per_second"] == pytest.approx(5 * 2**0.5, abs=.001)
    assert diagonal["movement_direction_degrees"] == pytest.approx(45)


def test_angular_velocity_uses_irregular_video_time_deltas():
    first_person = pose(elbow_bend=0)
    _first, state, _timings = pose_geometry(first_person, 100, 100, 1)
    second_person = pose(elbow_bend=20)
    second, _state, _timings = pose_geometry(second_person, 100, 100, 3, state)
    measure = second["joint_measurements"]["left_elbow"]
    assert second["elapsed_seconds"] == pytest.approx(2)
    assert measure["angular_velocity_degrees_per_second"] == pytest.approx(
        measure["angular_change_degrees"] / 2)
    assert measure["keypoint_confidence"] == pytest.approx(.9)


def test_missing_joint_observation_breaks_velocity_continuity_and_resets_smoothing():
    history = MotionHistory(max_observations=10)
    missing = pose()
    del missing["keypoints"]["left_wrist"]
    history.observe([pose(elbow_bend=0)], 100, 100, 0)
    assert "left_elbow" not in history.observe([missing], 100, 100, .1)[0]["joint_measurements"]
    resumed = history.observe([pose(elbow_bend=20)], 100, 100, .2)[0]
    elbow = resumed["joint_measurements"]["left_elbow"]
    assert elbow["angular_velocity_degrees_per_second"] is None
    assert elbow["smoothed_angle_degrees"] == pytest.approx(elbow["raw_angle_degrees"])


def test_motion_history_keeps_track_states_separate_and_resets_after_gap():
    history = MotionHistory(max_observations=10, max_gap_seconds=2.5)
    first = history.observe([pose(track_id=1, shift_x=0), pose(track_id=2, shift_x=40)],
                            100, 100, 1, frame_number=10, video_id="video-a", lighting_mode="DAY")
    second = history.observe([pose(track_id=1, shift_x=10), pose(track_id=2, shift_x=45)],
                             100, 100, 2, frame_number=20, video_id="video-a", lighting_mode="NIGHT_VISION")
    assert len(first) == len(second) == 2
    one = next(row for row in second if row["track_id"] == 1)
    two = next(row for row in second if row["track_id"] == 2)
    assert one["movement_speed_pixels_per_second"] == pytest.approx(10)
    assert two["movement_speed_pixels_per_second"] == pytest.approx(5)
    assert one["video_id"] == "video-a" and one["frame_number"] == 20
    assert one["lighting_mode"] == "NIGHT_VISION"
    later = history.observe([pose(track_id=1, shift_x=30)], 100, 100, 6,
                            frame_number=60, video_id="video-a")[-1]
    assert later["movement_speed_pixels_per_second"] is None
    assert later["track_segment_id"] != one["track_segment_id"]
    assert history.ordered() == sorted(history.ordered(), key=lambda row: (row["timestamp_seconds"], row["track_id"]))


def test_motion_history_caps_stored_measurements_without_fabricating_data():
    history = MotionHistory(max_observations=1)
    history.observe([pose(track_id=1)], 100, 100, 1)
    extra = history.observe([pose(track_id=2)], 100, 100, 2)
    assert len(history.ordered()) == 1
    assert history.dropped_count == 1
    assert len(extra) == 1  # current frame can still be annotated, but is not persisted


def test_video_processor_attempts_pose_and_measures_every_decoded_frame(tmp_path):
    from app.services.video_processor import analyze_video

    source = tmp_path / "motion-analytics-fixture.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 2, (100, 100))
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG test encoder is unavailable")
    for _ in range(6):
        writer.write(np.full((100, 100, 3), 50, dtype=np.uint8))
    writer.release()

    class PoseFixture:
        METHOD = "deterministic test fixture"
        model_name = "mock-pose"
        model_version = "test"
        def __init__(self): self.index = 0
        def inspect_with_evidence(self, frame, timestamp):
            item = pose(track_id=7, shift_x=self.index * 5)
            self.index += 1
            return frame, None, {"timestamp_seconds": timestamp, "person_count": 1,
                "people": [item], "transition_status": "insufficient_temporal_evidence"}

    result = analyze_video(str(source), 1.0, 10, PoseFixture(),
                           requested_lighting_mode="day", video_id="video-fixture")
    stored = result["motion_analytics"]["measurements"]
    assert result["motion_analytics"]["sampling_basis"].startswith("every successfully decoded frame")
    assert result["motion_analytics"]["decoded_frames"] == 6
    assert result["motion_analytics"]["pose_inference_attempted_frames"] == 6
    assert result["motion_analytics"]["pose_processed_frames"] == 6
    assert result["pose_analysis"]["frames_sampled"] == 3  # legacy fall/zone sample cadence
    assert stored and all(row["video_id"] == "video-fixture" for row in stored)
    assert all(row["track_id"] == 7 for row in stored)
    assert all(row["frame_number"] is not None for row in stored)
    assert [row["timestamp_seconds"] for row in stored] == sorted(row["timestamp_seconds"] for row in stored)
    assert any(row["joint_measurements"]["left_knee"]["angular_velocity_degrees_per_second"] is not None
               for row in stored[1:])


def test_every_frame_motion_rows_flush_in_bounded_batches_without_embedding(tmp_path):
    from app.services.video_processor import analyze_video

    source = tmp_path / "motion-batch-fixture.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 3, (100, 100))
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG test encoder is unavailable")
    for _ in range(7):
        writer.write(np.full((100, 100, 3), 50, dtype=np.uint8))
    writer.release()

    class PoseFixture:
        METHOD = "deterministic test fixture"
        def __init__(self): self.index = 0
        def inspect_with_evidence(self, frame, timestamp, update_transition=True):
            item = pose(track_id=3, shift_x=self.index)
            self.index += 1
            return frame, None, {"person_count": 1, "people": [item]}

    batches = []
    result = analyze_video(str(source), 1, 10, PoseFixture(), requested_lighting_mode="day",
        video_id="video-fixture", motion_batch_callback=lambda rows: batches.append(list(rows)),
        motion_run_id="0123456789abcdef01234567", motion_batch_size=3)
    saved = [row for batch in batches for row in batch]
    assert [row["frame_number"] for row in saved] == list(range(7))
    assert result["motion_analytics"]["measurement_count"] == 7
    assert result["motion_analytics"]["measurements"] == []
    assert all(len(batch) <= 3 for batch in batches)
