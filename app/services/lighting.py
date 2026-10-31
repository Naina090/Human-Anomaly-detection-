"""Low-light assessment and conservative frame enhancement for video analysis."""
from __future__ import annotations

import math

import cv2
import numpy as np


LIGHTING_MODES = {"auto", "day", "night_vision"}


def assess_capture(capture, frame_count, sample_count=7, threshold=70.0,
                   low_light_fraction=0.6, requested_mode="auto"):
    """Measure a bounded set of representative frames and rewind the capture.

    Brightness is mean grayscale luminance in OpenCV's 0..255 range. Auto mode
    requires a majority of sampled frames below the configured threshold and a
    low median, preventing one dark frame from switching the entire video.
    """
    mode = str(requested_mode or "auto").strip().lower()
    if mode not in LIGHTING_MODES:
        raise ValueError("Lighting mode must be auto, day, or night_vision.")
    count = max(1, int(sample_count or 1))
    total_frames = max(0, int(frame_count or 0))
    indices = np.linspace(0, max(0, total_frames - 1), min(count, total_frames), dtype=int) if total_frames else [0]
    brightness, contrast = [], []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok or frame is None or getattr(frame, "size", 0) == 0:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness.append(float(np.mean(gray)))
        contrast.append(float(np.std(gray)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not brightness:
        raise ValueError("No representative video frames could be decoded for lighting assessment.")
    average = float(np.mean(brightness))
    median = float(np.median(brightness))
    dark_fraction = sum(value < float(threshold) for value in brightness) / len(brightness)
    detected = "LOW_LIGHT" if dark_fraction >= float(low_light_fraction) and median < float(threshold) else "DAY"
    if mode == "day":
        applied = "DAY"
        enhancement = False
    elif mode == "night_vision":
        applied = "NIGHT_VISION"
        enhancement = True
    else:
        applied = "NIGHT_VISION" if detected == "LOW_LIGHT" else "DAY"
        enhancement = applied == "NIGHT_VISION"
    warnings = []
    if median < 18:
        warnings.append("Very low luminance: enhancement cannot recover details absent from the source footage.")
    if float(np.mean(contrast)) < 5:
        warnings.append("Sampled frames have very low contrast; blur, camera movement, and visibility may limit pose reliability.")
    if len(brightness) < min(count, total_frames or count):
        warnings.append("Some representative frames could not be decoded; lighting assessment has limited coverage.")
    return {
        "requested_mode": mode.upper(), "detected_mode": detected,
        "lighting_mode": applied, "night_enhancement_applied": enhancement,
        "average_brightness": round(average, 3), "median_brightness": round(median, 3),
        "average_contrast": round(float(np.mean(contrast)), 3),
        "dark_sample_fraction": round(float(dark_fraction), 3),
        "sampled_frame_count": len(brightness), "brightness_threshold": float(threshold),
        "warnings": warnings,
    }


def enhance_low_light_frame(frame, clip_limit=2.2, tile_grid=8, gamma=0.78,
                            brightness_offset=8.0, denoise=False):
    """Apply bounded CLAHE and gamma adjustment while preserving frame size.

    Enhancement changes visibility and pixel values; it does not reconstruct
    missing scene detail. Daylight callers should bypass this function.
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        raise ValueError("A non-empty decoded frame is required for enhancement.")
    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0.35 <= gamma <= 1.0:
        raise ValueError("Night-vision gamma must be between 0.35 and 1.0.")
    grid = max(2, min(16, int(tile_grid)))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=max(1.0, float(clip_limit)), tileGridSize=(grid, grid))
    lightness = clahe.apply(lightness)
    lut = np.clip(((np.arange(256, dtype=np.float32) / 255.0) ** gamma) * 255.0 +
                  float(brightness_offset), 0, 255).astype(np.uint8)
    lightness = cv2.LUT(lightness, lut)
    if denoise:
        lightness = cv2.GaussianBlur(lightness, (3, 3), 0.55)
    enhanced = cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    return enhanced
