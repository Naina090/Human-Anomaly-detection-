"""Optional temporal violence-activity review detector.

The X3D-M checkpoint classifies a short clip as Violence or Non-Violence. Its
score is an uncalibrated model output and is not a finding about intent or an
individual. It does not localize people, so no person or polygon attribution
is made here.
"""
from collections import deque
from importlib.metadata import PackageNotFoundError, version
import logging
import math
import os
import threading
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)
METHOD = "X3D-M temporal video classification (RWF-2000 Violence / Non-Violence)"
MODEL_NAME = "X3D-M RWF-2000 fine-tuned violence-activity classifier"
MODEL_REVISION = "visionlab-ai/school-violence-detection-models@a744b6a"
FRAME_COUNT = 16
FRAME_SIZE = (224, 224)
MEAN = np.asarray([0.45, 0.45, 0.45], dtype=np.float32).reshape(1, 1, 1, 3)
STD = np.asarray([0.225, 0.225, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)

_CACHE_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()
_MODEL_CACHE = {}


class FightDetectorUnavailable(RuntimeError):
    """Raised when the optional temporal action model cannot be initialized."""


def detector_status(model_path, enabled=True):
    path = os.path.expanduser(str(model_path or "").strip())
    if not enabled:
        return {"configured": bool(path), "ready": False, "status": "disabled",
                "method": METHOD, "message": "Fighting detection is disabled."}
    if not path:
        return {"configured": False, "ready": False, "status": "not_configured",
                "method": METHOD,
                "message": "No temporal fighting model is configured; this run will make no fighting prediction."}
    if not os.path.isfile(path):
        return {"configured": True, "ready": False, "status": "unavailable",
                "method": METHOD, "message": "Configured fighting-model weights were not found."}
    try:
        import torch  # noqa: F401
        from pytorchvideo.models.hub import x3d_m  # noqa: F401
    except Exception:
        return {"configured": True, "ready": False, "status": "unavailable",
                "method": METHOD, "message": "PyTorchVideo is unavailable; install requirements-ai.txt."}
    return {"configured": True, "ready": True, "status": "ready",
            "method": METHOD, "message": "Local temporal model is available."}


def _load_model(path, requested_device="auto"):
    """Load only tensor data from the locally configured checkpoint.

    Safe PyTorch deserialization is retained. The checkpoint contains a small
    NumPy scalar/dtype metadata entry, so only those specific globals are
    allow-listed; arbitrary pickle execution is never enabled.
    """
    import torch
    from torch import nn
    from pytorchvideo.models.hub import x3d_m

    path = os.path.realpath(path)
    device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        device = "cpu"
    cache_key = (path, device, os.path.getmtime(path))
    with _CACHE_LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key], torch, device
        used_device = device
        try:
            numpy_globals = []
            numpy_core = getattr(np, "_core", None) or getattr(np, "core", None)
            scalar = getattr(numpy_core, "multiarray", None)
            if scalar is not None and hasattr(scalar, "scalar"):
                numpy_globals.append(scalar.scalar)
            numpy_globals.append(np.dtype)
            float64_dtype = getattr(getattr(np, "dtypes", None), "Float64DType", None)
            if float64_dtype is not None:
                numpy_globals.append(float64_dtype)
            with torch.serialization.safe_globals(numpy_globals):
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise FightDetectorUnavailable(
                "The fighting checkpoint could not be loaded with safe tensor-only deserialization."
            ) from exc
        if not isinstance(checkpoint, dict):
            raise FightDetectorUnavailable("The fighting checkpoint has an unsupported structure.")
        config = checkpoint.get("chosen_config") or {}
        if config.get("model") not in {None, "x3d_m"} or int(config.get("frames", FRAME_COUNT)) != FRAME_COUNT:
            raise FightDetectorUnavailable("Checkpoint architecture or frame configuration is not X3D-M/16-frame compatible.")
        state = checkpoint.get("model") or checkpoint.get("model_state_dict")
        if not isinstance(state, dict):
            raise FightDetectorUnavailable("The checkpoint does not contain model weights.")
        model = x3d_m(pretrained=False)
        model.blocks[5].proj = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, 2))
        state = {key.removeprefix("backbone."): value for key, value in state.items()}
        try:
            model.load_state_dict(state, strict=True)
            model.eval()
            model.to(device)
        except Exception as exc:
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                    model.to("cpu")
                    device = "cpu"
                except Exception:
                    raise FightDetectorUnavailable("X3D-M could not initialize on the available device.") from exc
            else:
                raise FightDetectorUnavailable("X3D-M weights do not match the installed PyTorchVideo model.") from exc
        if device != used_device:
            cache_key = (path, device, os.path.getmtime(path))
        _MODEL_CACHE[cache_key] = model
        return model, torch, device


class FightActionDetector:
    """Streaming 16-frame X3D-M inference with conservative event merging."""

    METHOD = METHOD

    def __init__(self, model_path, *, enabled=True, threshold=0.4,
                 window_seconds=4.0, stride_seconds=1.0,
                 confirmation_windows=2, device="auto", model_loader=None):
        self.model_path = os.path.expanduser(str(model_path or "").strip())
        self.enabled = bool(enabled)
        self.threshold = float(threshold)
        self.window_seconds = max(2.0, min(4.0, float(window_seconds)))
        self.stride_seconds = max(0.25, float(stride_seconds))
        self.confirmation_windows = max(2, int(confirmation_windows))
        self.sample_seconds = self.window_seconds / FRAME_COUNT
        self.status = "not_configured"
        self.error_message = None
        self.windows_analyzed = 0
        self.frames_sampled = 0
        self.last_sample_time = None
        self.last_window_time = None
        self.pending = []
        self.active = None
        self.events = []
        self.max_model_score = None
        self.inference_seconds = 0.0
        self.device = "cpu"
        self.model = None
        self.torch = None
        self.buffer = deque(maxlen=FRAME_COUNT)
        self.model_loader = model_loader or _load_model
        status = detector_status(self.model_path, self.enabled)
        if not self.enabled:
            self.status = "disabled"
            return
        if not self.model_path:
            self.status = "not_configured"
            return
        if not os.path.isfile(self.model_path):
            self.status = "unavailable"
            self.error_message = status["message"]
            return
        try:
            self.model, self.torch, self.device = self.model_loader(self.model_path, device)
            self.status = "ready"
        except FightDetectorUnavailable as exc:
            self.status = "unavailable"
            self.error_message = str(exc)
        except Exception as exc:
            self.status = "unavailable"
            self.error_message = "The configured temporal action model could not be initialized."
            log.exception("Fighting detector initialization failed (%s).", type(exc).__name__)

    def _infer(self):
        clip = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                         for frame, _time, _index in self.buffer], axis=0)
        clip = (clip.astype(np.float32) / 255.0 - MEAN) / STD
        tensor = self.torch.from_numpy(clip.copy()).permute(3, 0, 1, 2).unsqueeze(0).to(self.device)
        started = time.perf_counter()
        try:
            with _INFERENCE_LOCK, self.torch.inference_mode():
                logits = self.model(tensor)
                scores = self.torch.softmax(logits.reshape(1, -1), dim=1)[0]
                violence_score = float(scores[1].detach().cpu().item())
        finally:
            self.inference_seconds += time.perf_counter() - started
        if not math.isfinite(violence_score):
            raise ValueError("The model returned a non-finite violence score.")
        return violence_score

    def _record_window(self, score):
        start = float(self.buffer[0][1])
        end = float(self.buffer[-1][1])
        self.windows_analyzed += 1
        self.max_model_score = score if self.max_model_score is None else max(self.max_model_score, score)
        item = {"start": start, "end": end, "score": score,
                "frame_number": int(self.buffer[0][2])}
        positive = score >= self.threshold
        max_gap = self.stride_seconds * 1.75
        if positive:
            if self.active:
                if start - self.active["end"] <= max_gap:
                    self.active["end"] = end
                    self.active["scores"].append(score)
                    self.active["windows"] += 1
                else:
                    self.events.append(self._event(self.active))
                    self.active = None
            if self.active is None:
                if self.pending and start - self.pending[-1]["end"] <= max_gap:
                    self.pending.append(item)
                else:
                    self.pending = [item]
                if len(self.pending) >= self.confirmation_windows:
                    self.active = {"start": self.pending[0]["start"], "end": end,
                                   "start_frame": self.pending[0]["frame_number"],
                                   "scores": [entry["score"] for entry in self.pending],
                                   "windows": len(self.pending)}
                    self.pending = []
        else:
            self.pending = []
            if self.active:
                self.events.append(self._event(self.active))
                self.active = None
        self.last_window_time = end

    def _event(self, event):
        maximum = max(event["scores"])
        return {"event_type": "SUSPECTED_FIGHTING",
                "label": "Suspected violence-related activity — requires review",
                "timestamp_seconds": float(event["start"]),
                "start_timestamp_seconds": float(event["start"]),
                "end_timestamp_seconds": float(event["end"]),
                "frame_number": int(event["start_frame"]),
                "confidence": None, "anomaly_score": None, "model_score": float(maximum),
                "track_id": None,
                "detection_method": self.METHOD,
                "details": {"model_name": MODEL_NAME, "model_revision": MODEL_REVISION,
                    "model_class": "Violence", "score_kind": "uncalibrated softmax class score",
                    "model_score": float(maximum), "decision_threshold": self.threshold,
                    "window_seconds": self.window_seconds,
                    "window_stride_seconds": self.stride_seconds,
                    "consecutive_positive_windows": int(event["windows"]),
                    "window_scores": [float(value) for value in event["scores"]],
                    "person_attribution": "unavailable; classifier operates on the whole frame clip",
                    "zone_association_status": "unavailable",
                    "zone_association_reason": "The temporal classifier does not spatially localize the activity.",
                    "interpretation": "Candidate violence-related clip classification; requires human review and is not proof of fighting or wrongdoing."}}

    def observe_frame(self, frame, timestamp_seconds, frame_number):
        """Feed source frames in order; retain only a small resized temporal buffer."""
        if self.status != "ready":
            return
        timestamp = float(timestamp_seconds)
        if not math.isfinite(timestamp) or timestamp < 0:
            return
        tolerance = self.sample_seconds * 0.2
        if self.last_sample_time is not None and timestamp - self.last_sample_time < self.sample_seconds - tolerance:
            return
        resized = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
        self.buffer.append((resized, timestamp, int(frame_number)))
        self.last_sample_time = timestamp
        self.frames_sampled += 1
        if len(self.buffer) < FRAME_COUNT:
            return
        if self.last_window_time is not None and timestamp - self.last_window_time < self.stride_seconds - tolerance:
            return
        score = self._infer()
        self._record_window(score)

    def finish(self):
        """Close any confirmed interval and report actual model execution state."""
        if self.active:
            self.events.append(self._event(self.active))
            self.active = None
        analysis_status = self.status
        if self.status == "ready":
            analysis_status = "complete" if self.windows_analyzed else "insufficient_video"
        return list(self.events), {"status": analysis_status, "method": self.METHOD,
            "model_name": MODEL_NAME if self.model is not None else None,
            "model_revision": MODEL_REVISION if self.model is not None else None,
            "device": self.device if self.model is not None else None,
            "frames_sampled": self.frames_sampled, "windows_analyzed": self.windows_analyzed,
            "candidate_event_count": len(self.events), "maximum_model_score": self.max_model_score,
            "decision_threshold": self.threshold if self.model is not None else None,
            "window_seconds": self.window_seconds, "window_stride_seconds": self.stride_seconds,
            "confirmation_windows": self.confirmation_windows,
            "score_interpretation": "Uncalibrated softmax class score; review signal only.",
            "error_message": self.error_message,
            "limitation": "Clip-level Violence / Non-Violence classification cannot identify a specific person or associate the activity with a polygon zone."}


def package_version():
    try:
        return version("pytorchvideo")
    except PackageNotFoundError:
        return None
