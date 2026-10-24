"""Optional Ultralytics COCO object detection with temporary ByteTrack IDs.

No weights are downloaded by this module. The configured model must be a local
file and expose supported COCO class names.
"""
from collections import defaultdict, deque
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
    "backpack", "handbag", "suitcase",
}
METHOD = "Ultralytics YOLO COCO detector + ByteTrack"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_model_path(model_path):
    """Resolve configured relative model paths from the project, not CWD."""
    if not model_path:
        return ""
    path = Path(os.path.expanduser(os.fspath(model_path)))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


class ObjectDetectorUnavailable(RuntimeError):
    """Raised when an optional local object model cannot be used."""


def detector_status(model_path="", tracker="bytetrack.yaml"):
    if not model_path:
        return {"configured": False, "ready": False, "status": "not_configured",
                "method": METHOD,
                "message": "Object detection is not configured. Set OBJECT_MODEL_PATH to local COCO-trained YOLO weights."}
    if importlib.util.find_spec("ultralytics") is None:
        return {"configured": True, "ready": False, "status": "unavailable",
                "method": METHOD, "message": "Object detector package unavailable. Install requirements-ai.txt."}
    model_path = resolve_model_path(model_path)
    if not os.path.isfile(model_path):
        return {"configured": True, "ready": False, "status": "unavailable",
                "method": METHOD, "message": "Configured object model file was not found. The app does not download weights automatically."}
    return {"configured": True, "ready": True, "status": "ready", "method": METHOD,
            "message": f"Local object model configured with {tracker}; supported COCO classes are checked when a job starts."}


def _as_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _class_name(names, class_id):
    try:
        value = names.get(class_id, names.get(str(class_id))) if isinstance(names, dict) else names[class_id]
    except (IndexError, KeyError, TypeError):
        return None
    return str(value).strip().casefold() if value is not None else None


class YoloObjectTracker:
    """Run a locally configured Ultralytics detector and preserve temporary tracks."""

    METHOD = METHOD

    def __init__(self, model_path, confidence=0.25, tracker="bytetrack.yaml"):
        model_path = resolve_model_path(model_path)
        if not model_path or not os.path.isfile(model_path):
            raise ObjectDetectorUnavailable("Set OBJECT_MODEL_PATH to an existing local object-detection weights file.")
        if importlib.util.find_spec("ultralytics") is None:
            raise ObjectDetectorUnavailable("Install the optional requirements-ai.txt dependencies to use object detection.")
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        except Exception as exc:
            raise ObjectDetectorUnavailable("The configured object model could not be loaded.") from exc
        try:
            self.package_version = version("ultralytics")
        except PackageNotFoundError:
            self.package_version = None
        self.names = getattr(self.model, "names", {})
        self.class_ids = [int(class_id) for class_id, name in
                          ((self.names.items()) if isinstance(self.names, dict) else enumerate(self.names))
                          if str(name).strip().casefold() in SUPPORTED_CLASSES]
        if not self.class_ids:
            raise ObjectDetectorUnavailable(
                "The configured model does not contain supported COCO classes (person, bicycle, vehicles, bags, or suitcase)."
            )
        self.model_path = model_path
        self.confidence = max(0.01, min(0.99, float(confidence)))
        self.tracker = tracker or "bytetrack.yaml"
        self.track_paths = defaultdict(lambda: deque(maxlen=30))

    def detect(self, frame, timestamp_seconds, frame_number):
        """Return normalized per-frame observations; no unique identity is implied."""
        results = self.model.track(frame, persist=True, tracker=self.tracker,
                                   conf=self.confidence, classes=self.class_ids, verbose=False)
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        coordinates = _as_numpy(getattr(boxes, "xyxy", None))
        class_ids = _as_numpy(getattr(boxes, "cls", None))
        confidences = _as_numpy(getattr(boxes, "conf", None))
        track_ids = _as_numpy(getattr(boxes, "id", None))
        if coordinates is None or class_ids is None or confidences is None:
            return []
        coordinates = coordinates.reshape((-1, 4))
        class_ids = class_ids.reshape(-1)
        confidences = confidences.reshape(-1)
        track_ids = track_ids.reshape(-1) if track_ids is not None else None
        names = getattr(result, "names", self.names)
        observations = []
        for index, coordinates_row in enumerate(coordinates):
            class_id = int(class_ids[index])
            label = _class_name(names, class_id)
            if label not in SUPPORTED_CLASSES:
                continue
            x1, y1, x2, y2 = (round(float(item), 1) for item in coordinates_row)
            track_id = None
            if track_ids is not None and index < len(track_ids):
                candidate = float(track_ids[index])
                if np.isfinite(candidate) and candidate >= 0:
                    track_id = int(candidate)
            confidence = round(float(confidences[index]), 5)
            observation = {
                "event_type": "object_detection",
                "label": label,
                "timestamp_seconds": float(timestamp_seconds),
                "frame_number": int(frame_number),
                "confidence": confidence,
                "anomaly_score": None,
                "track_id": track_id,
                "detection_method": self.METHOD,
                "details": {"bbox_xyxy": [x1, y1, x2, y2],
                            "track_id_is_temporary": track_id is not None},
            }
            observations.append(observation)
            if label == "person" and track_id is not None:
                self.track_paths[track_id].append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
        return observations

    def annotate(self, frame, observations):
        output = frame.copy()
        colors = {"person": (60, 210, 230), "bicycle": (240, 180, 70), "car": (240, 180, 70),
                  "motorcycle": (240, 180, 70), "bus": (240, 180, 70),
                  "train": (240, 180, 70), "truck": (240, 180, 70), "backpack": (120, 210, 120),
                  "handbag": (120, 210, 120), "suitcase": (120, 210, 120)}
        for observation in observations:
            x1, y1, x2, y2 = (int(value) for value in observation["details"]["bbox_xyxy"])
            label = observation["label"]
            color = colors.get(label, (60, 210, 230))
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            track = f" ID {observation['track_id']}" if observation.get("track_id") is not None else ""
            caption = f"{label} {observation['confidence']:.2f}{track}"
            cv2.putText(output, caption, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, color, 1, cv2.LINE_AA)
        for points in self.track_paths.values():
            if len(points) > 1:
                cv2.polylines(output, [np.asarray(points, dtype=np.int32)], False,
                              (60, 210, 230), 2, cv2.LINE_AA)
        return output
