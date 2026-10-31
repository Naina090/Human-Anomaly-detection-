"""2D pose geometry for existing COCO-keypoint observations.

Angles are image-plane measurements. Segment orientation uses x-right/y-down
coordinates in degrees from +x; torso inclination is degrees from vertical.
These are not 3D anatomical rotations or calibrated physical measurements.
COCO-17 has no finger, hand, heel, or toe landmarks, so wrist-flexion and
ankle-flexion angles cannot be measured; forearm and lower-leg orientations
are reported as segment directions instead.
"""
from __future__ import annotations

import math
import time


JOINT_DEFINITIONS = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder": ("left_hip", "left_shoulder", "left_elbow"),
    "right_shoulder": ("right_hip", "right_shoulder", "right_elbow"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
}

SEGMENT_DEFINITIONS = {
    "shoulder_line": ("left_shoulder", "right_shoulder"),
    "hip_line": ("left_hip", "right_hip"),
    "left_upper_arm": ("left_shoulder", "left_elbow"),
    "right_upper_arm": ("right_shoulder", "right_elbow"),
    "left_forearm": ("left_elbow", "left_wrist"),
    "right_forearm": ("right_elbow", "right_wrist"),
    "left_thigh": ("left_hip", "left_knee"),
    "right_thigh": ("right_hip", "right_knee"),
    "left_lower_leg": ("left_knee", "left_ankle"),
    "right_lower_leg": ("right_knee", "right_ankle"),
}


def angle_at(a, b, c):
    """Interior angle in degrees at B; invalid/missing/degenerate data => None."""
    try:
        ax, ay = float(a[0]) - float(b[0]), float(a[1]) - float(b[1])
        cx, cy = float(c[0]) - float(b[0]), float(c[1]) - float(b[1])
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    values = (ax, ay, cx, cy)
    if not all(math.isfinite(value) for value in values):
        return None
    first_len, second_len = math.hypot(ax, ay), math.hypot(cx, cy)
    if first_len <= 1e-9 or second_len <= 1e-9:
        return None
    cosine = max(-1.0, min(1.0, (ax * cx + ay * cy) / (first_len * second_len)))
    return math.degrees(math.acos(cosine))


def segment_orientation(a, b):
    """Signed orientation in degrees from +x, with positive y directed down."""
    try:
        dx, dy = float(b[0]) - float(a[0]), float(b[1]) - float(a[1])
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if not math.isfinite(dx) or not math.isfinite(dy) or math.hypot(dx, dy) <= 1e-9:
        return None
    return math.degrees(math.atan2(dy, dx))


def signed_angular_difference(current, previous):
    """Shortest signed degree difference for wrapped segment orientations."""
    if current is None or previous is None:
        return None
    try:
        current, previous = float(current), float(previous)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(current) or not math.isfinite(previous):
        return None
    return (current - previous + 180.0) % 360.0 - 180.0


def _valid_point(keypoints, name, threshold):
    item = (keypoints or {}).get(name)
    if not isinstance(item, dict):
        return None
    try:
        x, y, confidence = float(item["x"]), float(item["y"]), float(item["confidence"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (not all(math.isfinite(value) for value in (x, y, confidence)) or confidence < threshold or
            x < 0 or y < 0):
        return None
    return x, y, confidence


def pose_geometry(person, width, height, timestamp, previous=None, confidence_threshold=0.3):
    """Compute available angles, center displacement, speed and direction."""
    try:
        timestamp, width, height = float(timestamp), float(width), float(height)
    except (TypeError, ValueError, OverflowError):
        return None
    track_id = person.get("track_id")
    try:
        track_id = int(track_id)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(v) and v > 0 for v in (width, height)) or not math.isfinite(timestamp) or timestamp < 0:
        return None
    points = person.get("keypoints") or {}
    joint_started = time.perf_counter()
    valid = {name: _valid_point(points, name, confidence_threshold) for name in {
        name for definition in JOINT_DEFINITIONS.values() for name in definition
    } | {name for pair in SEGMENT_DEFINITIONS.values() for name in pair}}
    angles = {}
    confidence_by_joint = {}
    for joint, names in JOINT_DEFINITIONS.items():
        p = [valid.get(name) for name in names]
        raw = angle_at(p[0][:2], p[1][:2], p[2][:2]) if all(p) else None
        if raw is not None:
            angles[joint] = raw
            confidence_by_joint[joint] = min(point[2] for point in p)
    for name, (first, last) in SEGMENT_DEFINITIONS.items():
        a, b = valid.get(first), valid.get(last)
        orientation = segment_orientation(a[:2], b[:2]) if a and b else None
        if orientation is not None:
            angles[name] = orientation
            confidence_by_joint[name] = min(a[2], b[2])
    joint_seconds = time.perf_counter() - joint_started
    left_shoulder, right_shoulder = valid.get("left_shoulder"), valid.get("right_shoulder")
    left_hip, right_hip = valid.get("left_hip"), valid.get("right_hip")
    shoulder_mid = tuple((left_shoulder[i] + right_shoulder[i]) / 2 for i in (0, 1)) if left_shoulder and right_shoulder else (left_shoulder or right_shoulder)
    hip_mid = tuple((left_hip[i] + right_hip[i]) / 2 for i in (0, 1)) if left_hip and right_hip else (left_hip or right_hip)
    if hip_mid is not None:
        center = hip_mid
        reference = "midpoint_between_visible_hips" if left_hip and right_hip else "single_visible_hip"
    elif shoulder_mid is not None:
        center = shoulder_mid
        reference = "midpoint_between_visible_shoulders" if left_shoulder and right_shoulder else "single_visible_shoulder"
    else:
        return None
    if shoulder_mid is not None and hip_mid is not None:
        dx, dy = hip_mid[0] - shoulder_mid[0], hip_mid[1] - shoulder_mid[1]
        torso = math.degrees(math.atan2(abs(dx), max(abs(dy), 1e-9)))
        angles["torso_inclination"] = torso
        confidence_by_joint["torso_inclination"] = min(
            p[2] for p in (left_shoulder, right_shoulder, left_hip, right_hip) if p)
    else:
        torso = None
    center_x, center_y = center[0] / width, center[1] / height
    previous = previous or {}
    elapsed = timestamp - previous.get("timestamp_seconds", timestamp)
    valid_elapsed = math.isfinite(elapsed) and elapsed > 1e-9
    old_angles = previous.get("angles") or {}
    angle_history = previous.get("angle_history") or {}
    velocity_started = time.perf_counter()
    joint_values = {}
    current_angle_history = {}
    for joint, raw in angles.items():
        history = list(angle_history.get(joint, []))
        history.append(raw)
        recent = history[-3:]  # trailing mean over the last three valid poses for this joint
        current_angle_history[joint] = recent
        if joint in SEGMENT_DEFINITIONS and recent:
            unwrapped = [recent[0]]
            for value in recent[1:]:
                unwrapped.append(unwrapped[-1] + signed_angular_difference(value, recent[len(unwrapped) - 1]))
            smoothed = ((sum(unwrapped) / len(unwrapped) + 180.0) % 360.0) - 180.0
        else:
            smoothed = sum(recent) / len(recent)
        old_raw = old_angles.get(joint)
        delta = (signed_angular_difference(raw, old_raw) if joint in SEGMENT_DEFINITIONS
                 else raw - old_raw) if old_raw is not None and valid_elapsed else None
        joint_values[joint] = {
            "valid": True,
            "raw_angle_degrees": round(raw, 3),
            "smoothed_angle_degrees": round(smoothed, 3),
            "angular_change_degrees": round(delta, 3) if delta is not None else None,
            "angular_velocity_degrees_per_second": round(delta / elapsed, 3) if delta is not None else None,
            "angular_change_direction": "increasing" if delta is not None and delta > 0 else "decreasing" if delta is not None and delta < 0 else "unchanged" if delta is not None else None,
            "keypoint_confidence": round(confidence_by_joint[joint], 4),
        }
    velocity_seconds = time.perf_counter() - velocity_started
    speed_started = time.perf_counter()
    dx_px = center[0] - previous.get("center_x_pixels", center[0]) if valid_elapsed else None
    dy_px = center[1] - previous.get("center_y_pixels", center[1]) if valid_elapsed else None
    speed_px = math.hypot(dx_px, dy_px) / elapsed if dx_px is not None else None
    direction = math.degrees(math.atan2(dy_px, dx_px)) if dx_px is not None and speed_px > 1e-9 else None
    speed_seconds = time.perf_counter() - speed_started
    result = {
        "track_id": track_id, "timestamp_seconds": round(timestamp, 6),
        "frame_number": person.get("frame_number"), "lighting_mode": person.get("lighting_mode"),
        "source_fps": person.get("source_fps"), "measurement_valid": bool(joint_values),
        "unavailable_joint_angles": {
            "left_wrist": "COCO-17 does not include hand/finger landmarks required for wrist flexion.",
            "right_wrist": "COCO-17 does not include hand/finger landmarks required for wrist flexion.",
            "left_ankle": "COCO-17 does not include foot landmarks required for ankle flexion.",
            "right_ankle": "COCO-17 does not include foot landmarks required for ankle flexion.",
        },
        "joint_measurements": joint_values,
        "body_center_x_pixels": round(center[0], 3), "body_center_y_pixels": round(center[1], 3),
        "body_center_x_normalized": round(center_x, 6), "body_center_y_normalized": round(center_y, 6),
        "body_center_reference": reference,
        "horizontal_displacement_pixels": round(dx_px, 3) if dx_px is not None else None,
        "vertical_displacement_pixels": round(dy_px, 3) if dy_px is not None else None,
        "movement_speed_pixels_per_second": round(speed_px, 3) if speed_px is not None else None,
        "movement_speed_normalized_per_second": round(speed_px / math.hypot(width, height), 6) if speed_px is not None else None,
        "movement_direction_degrees": round(direction, 3) if direction is not None else None,
        "direction_convention": "degrees from +x; positive y points down",
        "elapsed_seconds": round(elapsed, 6) if valid_elapsed else None,
        "lighting_mode": person.get("lighting_mode"),
    }
    return result, {"timestamp_seconds": timestamp, "center_x_pixels": center[0],
                    "center_y_pixels": center[1], "angles": angles,
                    "angle_history": current_angle_history}, {
                        "joint_angle_calculation_seconds": joint_seconds,
                        "angular_velocity_calculation_seconds": velocity_seconds,
                        "movement_speed_calculation_seconds": speed_seconds,
                    }


class MotionHistory:
    """Independent rolling geometry state per temporary tracker ID/video job."""
    def __init__(self, confidence_threshold=0.3, max_observations=2000, max_gap_seconds=2.5,
                 retain_measurements=True):
        self.confidence_threshold = float(confidence_threshold)
        self.max_observations = max(1, int(max_observations))
        self.retain_measurements = bool(retain_measurements)
        self.max_gap_seconds = max(0.1, float(max_gap_seconds))
        self.previous = {}
        self.segment_numbers = {}
        self.measurements = []
        self.dropped_count = 0
        self.measurement_count = 0
        self.joint_measurement_count = 0
        self.angular_velocity_measurement_count = 0
        self.last_batch = []
        self.stage_timings = {"joint_angle_calculation_seconds": 0.0,
                              "angular_velocity_calculation_seconds": 0.0,
                              "movement_speed_calculation_seconds": 0.0}

    def observe(self, people, width, height, timestamp, frame_number=None,
                video_id=None, lighting_mode=None):
        new_rows = []
        for person in people or []:
            track_id = person.get("track_id")
            if track_id is None:
                continue
            try:
                track_id = int(track_id)
            except (TypeError, ValueError, OverflowError):
                continue
            prepared = dict(person)
            prepared["frame_number"] = frame_number
            prepared["lighting_mode"] = lighting_mode
            prior = self.previous.get(track_id)
            try:
                gap = float(timestamp) - prior["timestamp_seconds"] if prior else None
            except (TypeError, ValueError, OverflowError):
                gap = None
            reset_segment = prior is None
            if prior and (gap is None or gap <= 0 or gap > self.max_gap_seconds):
                prior = None
                reset_segment = True
            value = pose_geometry(prepared, width, height, timestamp, prior, self.confidence_threshold)
            if value is None:
                # A low-confidence/invalid pose breaks temporal continuity; do
                # not calculate the next velocity across this missing sample.
                self.previous.pop(track_id, None)
                continue
            if reset_segment:
                self.segment_numbers[track_id] = self.segment_numbers.get(track_id, 0) + 1
            measurement, state, timings = value
            for key, elapsed_seconds in timings.items():
                self.stage_timings[key] += elapsed_seconds
            measurement["video_id"] = str(video_id) if video_id is not None else None
            measurement["track_segment_id"] = f"{track_id}:pose{self.segment_numbers[track_id]}"
            segment_start = (float(timestamp) if reset_segment else
                             float(prior.get("segment_start_timestamp", timestamp)))
            state["segment_start_timestamp"] = segment_start
            measurement["movement_duration_seconds"] = round(max(0.0, float(timestamp) - segment_start), 6)
            measurement["keypoint_confidence_mean"] = round(
                sum(point["keypoint_confidence"] for point in measurement["joint_measurements"].values()) /
                len(measurement["joint_measurements"]), 4) if measurement["joint_measurements"] else None
            self.joint_measurement_count += len(measurement["joint_measurements"])
            self.angular_velocity_measurement_count += sum(
                value.get("angular_velocity_degrees_per_second") is not None
                for value in measurement["joint_measurements"].values())
            self.previous[track_id] = state
            self.measurement_count += 1
            if self.retain_measurements:
                if len(self.measurements) < self.max_observations:
                    self.measurements.append(measurement)
                else:
                    self.dropped_count += 1
            new_rows.append(measurement)
        self.last_batch = new_rows
        return new_rows

    def ordered(self):
        return sorted(self.measurements, key=lambda row: (row["timestamp_seconds"], row["track_id"]))
