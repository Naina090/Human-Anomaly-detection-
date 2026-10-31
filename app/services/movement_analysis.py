"""Conservative temporal observations from tracked COCO human skeletons.

These labels summarize observed motion; they are not anomaly decisions, intent,
identity, or clinically validated actions.
"""
from __future__ import annotations

import math
from app.services.motion_geometry import MotionHistory


COCO_KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)

# Standard 17-keypoint COCO pose topology used by Ultralytics COCO pose models.
COCO_SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def _point(keypoints, name, threshold):
    item = keypoints.get(name) if isinstance(keypoints, dict) else None
    if not isinstance(item, dict):
        return None
    try:
        x, y = float(item["x"]), float(item["y"])
        confidence = float(item.get("confidence"))
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(confidence) and confidence >= threshold):
        return None
    return x, y, confidence


def _angle_at(first, center, last, width, height):
    """Joint angle in degrees after normalizing each axis by frame dimensions."""
    if not first or not center or not last:
        return None
    a = ((first[0] - center[0]) / width, (first[1] - center[1]) / height)
    b = ((last[0] - center[0]) / width, (last[1] - center[1]) / height)
    denominator = math.hypot(*a) * math.hypot(*b)
    if denominator <= 1e-9:
        return None
    cosine = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / denominator))
    return math.degrees(math.acos(cosine))


def classify_pose_observation(person, width, height, timestamp, prior=None,
                              confidence_threshold=0.3):
    """Return a provisional, explainable motion label and normalized measures."""
    try:
        timestamp = float(timestamp)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(timestamp) or timestamp < 0:
        return None
    track_id = person.get("track_id")
    if track_id is None or width <= 0 or height <= 0:
        return None
    try:
        track_id = int(track_id)
    except (TypeError, ValueError, OverflowError):
        return None
    points = person.get("keypoints") or {}
    shoulders = [_point(points, key, confidence_threshold) for key in ("left_shoulder", "right_shoulder")]
    hips = [_point(points, key, confidence_threshold) for key in ("left_hip", "right_hip")]
    knees = [_point(points, key, confidence_threshold) for key in ("left_knee", "right_knee")]
    ankles = [_point(points, key, confidence_threshold) for key in ("left_ankle", "right_ankle")]
    valid = [p for p in shoulders + hips + knees + ankles if p is not None]
    if len(valid) < 4 or len([p for p in shoulders if p]) < 1 or len([p for p in hips if p]) < 1:
        return None
    center_x = sum(p[0] for p in shoulders + hips if p) / len([p for p in shoulders + hips if p]) / width
    center_y = sum(p[1] for p in shoulders + hips if p) / len([p for p in shoulders + hips if p]) / height
    shoulder = (sum(p[0] for p in shoulders if p) / len([p for p in shoulders if p]),
                sum(p[1] for p in shoulders if p) / len([p for p in shoulders if p]))
    hip = (sum(p[0] for p in hips if p) / len([p for p in hips if p]),
           sum(p[1] for p in hips if p) / len([p for p in hips if p]))
    torso_dx = (hip[0] - shoulder[0]) / width
    torso_dy = (hip[1] - shoulder[1]) / height
    torso_angle = math.degrees(math.atan2(abs(torso_dx), max(abs(torso_dy), 1e-5)))
    ankle_y = sum(p[1] for p in ankles if p) / len([p for p in ankles if p]) / height if any(ankles) else None
    knee_y = sum(p[1] for p in knees if p) / len([p for p in knees if p]) / height if any(knees) else None
    keypoint_values = {name: _point(points, name, confidence_threshold) for name in
        ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
         "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")}
    knee_angles = [_angle_at(keypoint_values[f"{side}_hip"], keypoint_values[f"{side}_knee"],
                             keypoint_values[f"{side}_ankle"], width, height)
                   for side in ("left", "right")]
    knee_angles = [angle for angle in knee_angles if angle is not None]
    knee_flexion = (sum(180.0 - angle for angle in knee_angles) / len(knee_angles)
                    if knee_angles else None)
    hip_angles = [_angle_at(keypoint_values[f"{side}_shoulder"], keypoint_values[f"{side}_hip"],
                            keypoint_values[f"{side}_knee"], width, height)
                  for side in ("left", "right")]
    hip_angles = [angle for angle in hip_angles if angle is not None]
    # Hip-to-ankle proportion is resolution-independent and only interpreted
    # as a crouch when leg evidence is present and the torso is broadly upright.
    leg_ratio = ((ankle_y - center_y) if ankle_y is not None else None)
    speed = None
    angle_change = None
    limb_speed = None
    dt = None
    if prior:
        dt = float(timestamp) - prior["timestamp_seconds"]
        if dt > 0:
            speed = math.hypot(center_x - prior["center_x_normalized"],
                               center_y - prior["center_y_normalized"]) / dt
            angle_change = abs(torso_angle - prior["torso_angle_degrees"])
            prior_joints = prior.get("_joint_positions") or {}
            speeds = []
            for name, point in keypoint_values.items():
                old = prior_joints.get(name)
                if point is not None and old is not None:
                    speeds.append(math.hypot((point[0] - old[0]) / width,
                                             (point[1] - old[1]) / height) / dt)
            if speeds:
                limb_speed = sum(speeds) / len(speeds)
    if speed is None:
        category = "insufficient_temporal_evidence"
    elif angle_change is not None and angle_change >= 38 and speed >= 0.12:
        category = "sudden_posture_change"
    elif torso_angle > 56 and speed >= 0.12:
        category = "possible_fall_related_movement"
    elif (leg_ratio is not None and knee_y is not None and knee_flexion is not None
          and knee_flexion >= 35 and leg_ratio < 0.28
          and knee_y > center_y and (ankle_y is None or ankle_y > knee_y) and torso_angle < 48):
        category = "crouching"
    elif speed >= 0.65:
        category = "running_like_movement"
    elif speed >= 0.18:
        category = "walking_like_movement"
    elif speed < 0.025:
        category = "low_movement"
    elif torso_angle < 38:
        category = "standing"
    else:
        category = "movement_observed"
    confidence_values = [p[2] for p in valid]
    return {
        "track_id": int(track_id), "timestamp_seconds": round(float(timestamp), 4),
        "category": category, "lighting_mode": None,
        "keypoint_confidence_mean": round(sum(confidence_values) / len(confidence_values), 4),
        "center_x_normalized": round(center_x, 5), "center_y_normalized": round(center_y, 5),
        "torso_angle_degrees": round(torso_angle, 2),
        "hip_to_ankle_height_normalized": round(leg_ratio, 5) if leg_ratio is not None else None,
        "knee_height_normalized": round(knee_y, 5) if knee_y is not None else None,
        "movement_speed_normalized_per_second": round(speed, 5) if speed is not None else None,
        "posture_angle_change_degrees": round(angle_change, 2) if angle_change is not None else None,
        "mean_limb_motion_normalized_per_second": round(limb_speed, 5) if limb_speed is not None else None,
        "mean_knee_flexion_degrees": round(knee_flexion, 2) if knee_flexion is not None else None,
        "mean_hip_angle_degrees": round(sum(hip_angles) / len(hip_angles), 2) if hip_angles else None,
        "hip_to_knee_height_normalized": round((knee_y - center_y), 5) if knee_y is not None else None,
        "evidence_observations": 1,
        "_joint_positions": {name: (point[0], point[1]) for name, point in keypoint_values.items()
                              if point is not None},
    }


class MovementAnalyzer:
    """Build brief category episodes per temporary track, never anomaly events."""
    def __init__(self, min_observations=2, confidence_threshold=0.3,
                 max_observation_gap_seconds=2.5, max_motion_measurements=2000,
                 retain_motion_measurements=True):
        self.min_observations = max(2, int(min_observations))
        self.confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))
        self.max_observation_gap_seconds = max(0.5, float(max_observation_gap_seconds))
        self.previous = {}
        self.active = {}
        self.observations = []
        self.labels = {}
        self.segment_numbers = {}
        self.motion_history = MotionHistory(confidence_threshold, max_motion_measurements,
                                            max_observation_gap_seconds,
                                            retain_measurements=retain_motion_measurements)

    def observe(self, people, width, height, timestamp, lighting_mode=None,
                frame_number=None, video_id=None):
        try:
            timestamp = float(timestamp)
        except (TypeError, ValueError, OverflowError):
            return {}
        if not math.isfinite(timestamp) or timestamp < 0:
            return {}
        self.labels = {}
        self.motion_history.observe(people, width, height, timestamp,
                                   frame_number=frame_number, video_id=video_id,
                                   lighting_mode=lighting_mode)
        seen = set()
        for person in people or []:
            track_id = person.get("track_id")
            if track_id is None:
                continue
            try:
                track_id = int(track_id)
            except (TypeError, ValueError, OverflowError):
                continue
            seen.add(track_id)
            prior = self.previous.get(track_id)
            gap = float(timestamp) - prior["timestamp_seconds"] if prior else None
            if prior and (gap <= 0 or gap > self.max_observation_gap_seconds):
                active = self.active.pop(track_id, None)
                if active and active["evidence_observations"] >= self.min_observations:
                    self.observations.append(active)
                self.previous.pop(track_id, None)
                prior = None
            if prior is None:
                self.segment_numbers[track_id] = self.segment_numbers.get(track_id, 0) + 1
            row = classify_pose_observation(person, width, height, timestamp,
                                            prior, self.confidence_threshold)
            if row is None:
                self.previous.pop(track_id, None)
                active = self.active.pop(track_id, None)
                if active and active["evidence_observations"] >= self.min_observations:
                    self.observations.append(active)
                continue
            row["lighting_mode"] = lighting_mode
            row["track_segment_id"] = f"{track_id}:movement{self.segment_numbers[track_id]}"
            row["keypoint_names"] = list((person.get("keypoints") or {}).keys())
            internal_row = dict(row)
            self.previous[track_id] = internal_row
            row = {key: value for key, value in row.items() if key != "_joint_positions"}
            self.labels[track_id] = row["category"]
            active = self.active.get(track_id)
            category = row["category"]
            if active and active["category"] == category:
                active["end_timestamp_seconds"] = row["timestamp_seconds"]
                active["evidence_observations"] += 1
                active["last_measurements"] = row
            else:
                if active and active["evidence_observations"] >= self.min_observations:
                    self.observations.append(active)
                # A transition category is itself supported by two distinct
                # timestamped poses (prior and current), even if the category
                # changes again at the next sample.
                transition_support = (2 if category in {"sudden_posture_change", "possible_fall_related_movement"}
                                      and row.get("movement_speed_normalized_per_second") is not None else 1)
                self.active[track_id] = {
                    "track_id": track_id, "category": category,
                    "track_segment_id": row["track_segment_id"],
                    "start_timestamp_seconds": row["timestamp_seconds"],
                    "end_timestamp_seconds": row["timestamp_seconds"],
                    "lighting_mode": lighting_mode, "evidence_observations": transition_support,
                    "first_measurements": row, "last_measurements": row,
                    "interpretation": "Observed movement summary from tracked pose samples; not an anomaly, identity, or intent assessment.",
                }
        # Missing tracker observations do not imply movement or identity continuity.
        for track_id in list(self.previous):
            if track_id not in seen:
                self.previous.pop(track_id, None)
                active = self.active.pop(track_id, None)
                if active and active["evidence_observations"] >= self.min_observations:
                    self.observations.append(active)
        return dict(self.labels)

    def finish(self):
        for active in self.active.values():
            if active["evidence_observations"] >= self.min_observations:
                self.observations.append(active)
        self.active.clear()
        return sorted(self.observations, key=lambda row: (row["start_timestamp_seconds"], row["track_id"]))
