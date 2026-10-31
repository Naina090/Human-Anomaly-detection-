"""Optional local Ultralytics pose inference and explainable fall heuristic."""
import importlib.util
import os
import time
from importlib.metadata import PackageNotFoundError, version
from threading import local

import numpy as np
import cv2

from app.services.movement_analysis import COCO_KEYPOINT_NAMES, COCO_SKELETON


class DetectorUnavailable(RuntimeError):
    pass


COCO_JOINTS = {
    5: "left_shoulder", 6: "right_shoulder",
    11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee",
    15: "left_ankle", 16: "right_ankle",
}
COCO_INDEX_TO_NAME = dict(enumerate(COCO_KEYPOINT_NAMES))
METHOD = "YOLO pose + temporal fall-transition heuristic (research prototype)"
_WORKER_MODELS = local()


def _worker_pose_model(model_path):
    """Cache YOLO weights once per background worker thread."""
    from ultralytics import YOLO
    path = os.path.realpath(model_path)
    key = (path, os.path.getmtime(path))
    models = getattr(_WORKER_MODELS, "models", None)
    if models is None:
        models = _WORKER_MODELS.models = {}
    model = models.get(key)
    if model is None:
        model = YOLO(path)
        models.clear()
        models[key] = model
    # Each video gets a fresh tracking sequence while reusing the loaded pose
    # weights. ByteTrack IDs therefore cannot continue from the previous clip.
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", ()) or ():
        reset = getattr(tracker, "reset", None)
        if callable(reset):
            reset()
    return model


def _pose_overlay(frame, people_points, confidences, boxes, box_confidences, track_ids,
                  keypoint_min_confidence=0.15, overlay_mode="skeleton_tracking"):
    """Draw COCO-17 skeletons and temporary IDs with confidence-aware joints."""
    output = frame.copy()
    height, width = output.shape[:2]
    for index, points in enumerate(people_points):
        row_conf = confidences[index] if confidences is not None and index < len(confidences) else None
        valid = {}
        for joint_index, point in enumerate(points[:len(COCO_KEYPOINT_NAMES)]):
            try:
                x, y = float(point[0]), float(point[1])
                score = float(row_conf[joint_index]) if row_conf is not None and joint_index < len(row_conf) else 0.0
            except (TypeError, ValueError, IndexError):
                continue
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(score)):
                continue
            if score >= keypoint_min_confidence and -width <= x < 2 * width and -height <= y < 2 * height:
                valid[joint_index] = (int(round(x)), int(round(y)), score)
        for first, second in COCO_SKELETON:
            if first in valid and second in valid:
                cv2.line(output, valid[first][:2], valid[second][:2], (75, 220, 225), 2, cv2.LINE_AA)
        for joint_index, (x, y, score) in valid.items():
            color = (70, 225, 115) if score >= 0.5 else (55, 180, 255)
            cv2.circle(output, (x, y), 4 if score >= 0.5 else 3, color, -1, cv2.LINE_AA)
            cv2.circle(output, (x, y), 5, (8, 22, 34), 1, cv2.LINE_AA)
        # The analysis video is visual evidence: retain available person boxes
        # even when the legacy control requested a skeleton-only presentation.
        box = boxes[index] if boxes is not None and index < len(boxes) else None
        track = track_ids[index] if track_ids is not None and index < len(track_ids) else None
        if box is not None:
            try:
                x1, y1, x2, y2 = [int(round(float(value))) for value in box]
                cv2.rectangle(output, (x1, y1), (x2, y2), (73, 208, 230), 2, cv2.LINE_AA)
                confidence = (float(box_confidences[index]) if box_confidences is not None
                              and index < len(box_confidences) else None)
                identity_label = f"Person ID {int(float(track))}" if track is not None and float(track) >= 0 else "Person · ID unavailable"
                if confidence is not None and np.isfinite(confidence):
                    identity_label += f" · {confidence:.2f}"
                cv2.putText(output, identity_label, (max(0, x1), max(18, y1 - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (73, 208, 230), 1, cv2.LINE_AA)
            except (TypeError, ValueError, OverflowError):
                pass
    return output


def detector_status(model_path):
    """Report availability without loading weights or downloading anything."""
    model_path = os.path.expanduser(str(model_path or "").strip())
    if not model_path:
        return {"configured": False, "ready": False, "status": "not_configured",
                "method": METHOD, "message": "Validation only · no pose model configured"}
    if importlib.util.find_spec("ultralytics") is None:
        return {"configured": True, "ready": False, "status": "unavailable",
                "method": METHOD, "message": "Pose package unavailable · install requirements-ai.txt"}
    if not os.path.isfile(model_path):
        return {"configured": True, "ready": False, "status": "unavailable",
                "method": METHOD, "message": "Local pose weights not found · set POSE_MODEL_PATH to an existing file"}
    return {"configured": True, "ready": True, "status": "ready", "method": METHOD,
            "message": "Local pose model ready for analysis"}


class YoloPoseFallDetector:
    """COCO pose frontend; a single low posture can never emit a fall candidate."""

    METHOD = METHOD

    def __init__(self, model_path, settings=None):
        model_path = os.path.abspath(os.path.expanduser(str(model_path or "")))
        if not model_path or not os.path.isfile(model_path):
            raise DetectorUnavailable("Pose model weights were not found at the configured local path.")
        if importlib.util.find_spec("ultralytics") is None:
            raise DetectorUnavailable("Install the optional requirements-ai.txt dependencies to use pose analysis.")
        try:
            self.model = _worker_pose_model(model_path)
            if getattr(self.model, "task", None) not in (None, "pose"):
                raise DetectorUnavailable(
                    "The configured local model is not a pose model. Set POSE_MODEL_PATH to COCO pose weights."
                )
        except Exception as exc:
            if isinstance(exc, DetectorUnavailable):
                raise
            raise DetectorUnavailable("Pose model could not be loaded. Check the local weights and AI dependencies.") from exc
        try:
            self.model_version = version("ultralytics")
        except PackageNotFoundError:
            self.model_version = None
        self.model_name = os.path.basename(model_path)
        settings = settings or {}
        # A new detector/model is constructed for every video job, so Ultralytics'
        # persistent tracker state cannot leak from one uploaded video to another.
        self.tracker = str(settings.get("tracker", "bytetrack.yaml") or "bytetrack.yaml")
        self.overlay_mode = str(settings.get("overlay_mode", "skeleton_tracking"))
        self.transition_window_seconds = max(.2, float(settings.get("transition_window_seconds", 1.5)))
        self.minimum_vertical_drop = max(.01, min(.8, float(settings.get("minimum_vertical_drop", .12))))
        self.minimum_horizontal_persistence = max(.2, float(settings.get("minimum_horizontal_persistence", .6)))
        self.candidate_score_threshold = max(0, min(1, float(settings.get("candidate_score_threshold", .55))))
        self.previous = None
        self.track_center = None
        self.track_id = None
        self.upright_seen = False
        self.upright_timestamp = None
        self.horizontal_since = None
        self.drop_seen_at = None
        self.drop_fraction = None
        self.drop_start_at = None
        self.emitted = False
        self.last_model_track_seconds = 0.0

    def inspect(self, frame, timestamp):
        """Compatibility interface for earlier callers: return overlay and candidate."""
        overlay, event, _evidence = self.inspect_with_evidence(frame, timestamp)
        return overlay, event

    def inspect_with_evidence(self, frame, timestamp, update_transition=True):
        """Return annotated frame, optional candidate, and compact sampled pose evidence."""
        # Pose estimation provides person boxes and keypoints in the same result.
        # Ultralytics tracking attaches temporary IDs; retain a predict fallback
        # for older compatible integrations and lightweight test doubles.
        track = getattr(self.model, "track", None)
        model_started = time.perf_counter()
        self.last_model_track_seconds = 0.0
        try:
            if callable(track):
                results = track(frame, persist=True, tracker=getattr(self, "tracker", "bytetrack.yaml"), verbose=False)
            else:
                results = self.model.predict(frame, verbose=False)
        finally:
            self.last_model_track_seconds = time.perf_counter() - model_started
        if not results:
            if update_transition:
                self._reset_transition()
            return frame, None, self._empty_evidence(timestamp)
        result = results[0]
        keypoints = getattr(result, "keypoints", None)
        xy = getattr(keypoints, "xy", None) if keypoints is not None else None
        if xy is None:
            if update_transition:
                self._reset_transition()
            return result.plot(), None, self._empty_evidence(timestamp)
        pts = xy.cpu().numpy() if hasattr(xy, "cpu") else xy.numpy() if hasattr(xy, "numpy") else xy
        conf_tensor = getattr(keypoints, "conf", None)
        confs = conf_tensor.cpu().numpy() if conf_tensor is not None and hasattr(conf_tensor, "cpu") else (
            conf_tensor.numpy() if conf_tensor is not None and hasattr(conf_tensor, "numpy") else conf_tensor)
        if len(pts) == 0:
            if update_transition:
                self._reset_transition()
            return result.plot(), None, self._empty_evidence(timestamp)
        boxes = getattr(result, "boxes", None)
        box_tensor = getattr(boxes, "xyxy", None) if boxes is not None else None
        box_conf_tensor = getattr(boxes, "conf", None) if boxes is not None else None
        id_tensor = getattr(boxes, "id", None) if boxes is not None else None
        def as_numpy(value):
            if value is None:
                return None
            if hasattr(value, "cpu"):
                value = value.cpu()
            if hasattr(value, "numpy"):
                value = value.numpy()
            return value
        boxes_array = as_numpy(box_tensor)
        box_confidences = as_numpy(box_conf_tensor)
        track_ids = as_numpy(id_tensor)
        evidence_people = []
        for person_idx, person_points in enumerate(pts):
            confidence_row = confs[person_idx] if confs is not None and person_idx < len(confs) else None
            joints = {}
            for point_idx, joint_name in COCO_INDEX_TO_NAME.items():
                if point_idx >= len(person_points):
                    continue
                point = person_points[point_idx]
                confidence = (float(confidence_row[point_idx]) if confidence_row is not None
                              and point_idx < len(confidence_row) else None)
                joints[joint_name] = {
                    "x": round(float(point[0]), 2), "y": round(float(point[1]), 2),
                    "confidence": round(confidence, 5) if confidence is not None else None,
                }
            bbox = None
            if boxes_array is not None and person_idx < len(boxes_array):
                try:
                    values = [round(float(value), 2) for value in boxes_array[person_idx]]
                    if len(values) == 4 and values[2] > values[0] and values[3] > values[1]:
                        bbox = values
                except (TypeError, ValueError):
                    bbox = None
            box_confidence = None
            if box_confidences is not None and person_idx < len(box_confidences):
                try:
                    box_confidence = round(float(box_confidences[person_idx]), 5)
                except (TypeError, ValueError):
                    pass
            track_id = None
            if track_ids is not None and person_idx < len(track_ids):
                try:
                    value = float(track_ids[person_idx])
                    if value >= 0 and value.is_integer():
                        track_id = int(value)
                except (TypeError, ValueError):
                    pass
            evidence_people.append({"keypoints": joints, "bbox_xyxy": bbox,
                                    "detection_confidence": box_confidence, "track_id": track_id})

        annotated_frame = _pose_overlay(frame, pts, confs, boxes_array,
                                        box_confidences, track_ids,
                                        overlay_mode=getattr(self, "overlay_mode", "skeleton_tracking"))

        # Tracking and pose extraction run for every decoded frame. The legacy
        # fall-transition heuristic intentionally advances only at its prior
        # sampled cadence so this motion upgrade does not change fall behavior.
        if not update_transition:
            return annotated_frame, None, {
                "timestamp_seconds": float(timestamp), "person_count": int(len(pts)),
                "model_name": getattr(self, "model_name", None),
                "model_version": getattr(self, "model_version", None),
                "keypoint_names": list(COCO_KEYPOINT_NAMES), "people": evidence_people,
                "tracker": getattr(self, "tracker", "predict_fallback"),
                "tracking_status": "tracked" if any(p.get("track_id") is not None for p in evidence_people) else "ids_unavailable",
                "transition_status": "not_evaluated_this_frame",
            }

        # Continue the fall heuristic with the pose having the strongest shoulder/hip evidence.
        person_ids = [person.get("track_id") for person in evidence_people]
        eligible = []
        if confs is not None:
            for person_idx, person_points in enumerate(pts):
                if person_idx >= len(confs):
                    continue
                torso_confidences = confs[person_idx, [5, 6, 11, 12]]
                torso_points = person_points[[5, 6, 11, 12], :]
                if (np.isfinite(torso_confidences).all() and np.isfinite(torso_points).all()
                        and bool((torso_confidences >= .25).all())):
                    eligible.append(person_idx)
        if not eligible:
            self._reset_transition()
            evidence = {"timestamp_seconds": float(timestamp), "person_count": int(len(pts)),
                        "model_name": getattr(self, "model_name", None),
                        "model_version": getattr(self, "model_version", None),
                        "keypoint_names": list(COCO_KEYPOINT_NAMES), "people": evidence_people,
                        "transition_status": "inconclusive_low_confidence",
                        "tracking_status": "tracked" if any(value is not None for value in person_ids) else "ids_unavailable"}
            return annotated_frame, None, evidence
        prior_track_id = getattr(self, "track_id", None)
        if (prior_track_id is not None and prior_track_id in person_ids
                and person_ids.index(prior_track_id) in eligible):
            idx = person_ids.index(prior_track_id)
        else:
            idx = max(eligible, key=lambda person_idx: float(confs[person_idx, [5, 6, 11, 12]].mean()))
        selected_track_id = person_ids[idx]
        if prior_track_id is not None and selected_track_id != prior_track_id:
            # Do not join two people into one temporal fall sequence when a
            # tracker loses or changes its association.
            self._reset_transition()
        self.track_id = selected_track_id
        k = pts[idx]
        confidence_mean = float(confs[idx, [5, 6, 11, 12]].mean()) if confs is not None else None
        evidence = {"timestamp_seconds": float(timestamp), "person_count": int(len(pts)),
                    "model_name": getattr(self, "model_name", None),
                    "model_version": getattr(self, "model_version", None),
                    "keypoint_names": list(COCO_KEYPOINT_NAMES), "people": evidence_people,
                    "selected_person_index": idx,
                    "selected_torso_confidence": round(confidence_mean, 5) if confidence_mean is not None else None}
        person_bbox = None
        if idx < len(evidence_people):
            person_bbox = evidence_people[idx].get("bbox_xyxy")
            evidence["selected_track_id"] = evidence_people[idx].get("track_id")
            evidence["selected_person_detection_confidence"] = evidence_people[idx].get("detection_confidence")
        evidence["tracker"] = getattr(self, "tracker", "predict_fallback")
        evidence["tracking_status"] = "tracked" if any(person.get("track_id") is not None for person in evidence_people) else "ids_unavailable"
        if confidence_mean is not None and confidence_mean < .25:
            self._reset_transition()
            evidence["transition_status"] = "inconclusive_low_confidence"
            return annotated_frame, None, evidence
        if not all(float(k[i][1]) >= 0 for i in (5, 6, 11, 12)):
            self._reset_transition()
            evidence["transition_status"] = "inconclusive_missing_torso"
            return annotated_frame, None, evidence

        shoulder = k[[5, 6]].mean(axis=0)
        hip = k[[11, 12]].mean(axis=0)
        torso = hip - shoulder
        torso_ratio = abs(float(torso[0])) / max(abs(float(torso[1])), 1.0)
        center_y = float(k[[5, 6, 11, 12]].mean(axis=0)[1])
        self.track_center = k[[5, 6, 11, 12]].mean(axis=0)
        frame_height = frame.shape[0]
        upright = torso_ratio < 0.65
        horizontal = torso_ratio > 1.15
        prior = self.previous
        if upright and center_y < frame_height * 0.72:
            self.upright_seen = True
            self.upright_timestamp = float(timestamp)
        dropped = bool(prior and 0 <= timestamp - prior[0] <= self.transition_window_seconds and
                       center_y - prior[1] > frame_height * self.minimum_vertical_drop)
        if dropped and self.upright_seen:
            self.drop_seen_at = float(timestamp)
            self.drop_start_at = float(prior[0])
            self.drop_fraction = float((center_y - prior[1]) / frame_height)
        if horizontal:
            if self.horizontal_since is None:
                self.horizontal_since = float(timestamp)
        else:
            self.horizontal_since = None
        persistent = (self.horizontal_since is not None and
                      timestamp - self.horizontal_since >= self.minimum_horizontal_persistence)
        event = None
        if (self.upright_seen and self.drop_seen_at is not None and
                timestamp - self.drop_seen_at <= self.transition_window_seconds and persistent and not self.emitted):
            drop_score = min(1.0, max(0.0, (self.drop_fraction or 0) / .3))
            orient_score = min(1.0, torso_ratio / 2.0)
            persistence_score = min(1.0, (timestamp - self.horizontal_since) / 1.2)
            score = .5 * drop_score + .3 * orient_score + .2 * persistence_score
            if score < self.candidate_score_threshold:
                self.previous = (float(timestamp), center_y)
                evidence["transition_status"] = "candidate_below_score_threshold"
                evidence["anomaly_score"] = round(float(score), 4)
                return annotated_frame, None, evidence
            event = {
                "event_type": "potential_fall", "label": "Potential fall candidate",
                "timestamp_seconds": float(timestamp),
                "start_timestamp_seconds": float(self.drop_start_at if self.drop_start_at is not None else self.drop_seen_at),
                "end_timestamp_seconds": float(timestamp),
                "confidence": None, "anomaly_score": round(float(score), 4),
                "detection_method": self.METHOD,
                "track_id": evidence_people[idx].get("track_id") if idx < len(evidence_people) else None,
                "details": {
                    "previously_upright": True,
                    "upright_timestamp_seconds": self.upright_timestamp,
                    "rapid_drop_timestamp_seconds": self.drop_seen_at,
                    "vertical_drop_fraction": round(float(self.drop_fraction or 0), 4),
                    "torso_horizontal_ratio": round(torso_ratio, 4),
                    "horizontal_persistence_seconds": round(timestamp - self.horizontal_since, 3),
                    "score_components": "0.5*drop + 0.3*orientation + 0.2*persistence",
                    "model_name": getattr(self, "model_name", None),
                    "model_version": getattr(self, "model_version", None),
                    "candidate_score_threshold": self.candidate_score_threshold,
                    "person_detection_confidence": (evidence_people[idx].get("detection_confidence")
                                                     if idx < len(evidence_people) else None),
                    "limitations": "Unvalidated heuristic; sitting, bending, intentional lying, occlusion, and tracking switches can affect results.",
                },
            }
            if person_bbox is not None:
                event["details"]["person_bbox_xyxy"] = person_bbox
            self.emitted = True
        self.previous = (float(timestamp), center_y)
        evidence["transition_status"] = "candidate_emitted" if event else (
            "upright_seen" if self.upright_seen else "observed_no_candidate")
        evidence["torso_horizontal_ratio"] = round(torso_ratio, 4)
        return annotated_frame, event, evidence

    @staticmethod
    def _empty_evidence(timestamp):
        return {"timestamp_seconds": float(timestamp), "person_count": 0, "people": [],
                "transition_status": "no_person_observed"}

    def _reset_transition(self):
        self.previous = None
        self.track_center = None
        self.track_id = None
        self.upright_seen = False
        self.upright_timestamp = None
        self.horizontal_since = None
        self.drop_seen_at = None
        self.drop_fraction = None
        self.drop_start_at = None
