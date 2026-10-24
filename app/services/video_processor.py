import time
import logging
import math
import os
import uuid
from collections import Counter, defaultdict
import cv2
import numpy as np
from flask import current_app, has_app_context
from app.logging_utils import sanitized_traceback
from app.services.timestamps import format_video_timestamp

log = logging.getLogger(__name__)

class VideoError(Exception):
    pass


def _zone_transition_events(pose_frames, zones, width, height, entry_confirmation_frames=2,
                            exit_confirmation_frames=2, max_track_gap_seconds=1.5,
                            boundary_margin_px=8.0):
    """Emit entry incidents only after a stable outside-to-inside track transition.

    A first observation inside is informational only. Signed polygon distance
    provides a deadband around edges, and transitions need consecutive sampled
    observations to resist boundary jitter and brief detector/track noise.
    """
    if not zones or width <= 0 or height <= 0:
        return []
    active_zones = [zone for zone in zones if zone.get("enabled") is True]
    states = {}
    emitted = []
    entry_confirmation_frames = max(2, int(entry_confirmation_frames))
    exit_confirmation_frames = max(2, int(exit_confirmation_frames))
    max_track_gap_seconds = max(.1, float(max_track_gap_seconds))
    boundary_margin_px = max(0.0, float(boundary_margin_px))
    for sample in pose_frames:
        timestamp = sample.get("timestamp_seconds")
        frame_number = sample.get("frame_number")
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0:
            continue
        for person in sample.get("people") or []:
            track_id = person.get("track_id")
            box = person.get("bbox_xyxy")
            if track_id is None or not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            try:
                x1, _y1, x2, y2 = (float(value) for value in box)
                normalized_track_id = int(track_id)
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (x1, x2, y2)) or x2 <= x1 or y2 < 0:
                continue
            point = ((x1 + x2) / 2.0, y2)
            for zone in active_zones:
                polygon = np.asarray(
                    [[round(x * width), round(y * height)] for x, y in zone["points"]],
                    dtype=np.int32).reshape((-1, 1, 2))
                signed_distance = float(cv2.pointPolygonTest(polygon, point, True))
                location = "inside" if signed_distance > boundary_margin_px else (
                    "outside" if signed_distance < -boundary_margin_px else "boundary")
                key = (str(zone["zone_id"]), normalized_track_id)
                state = states.setdefault(key, {"phase": "unknown", "last_seen": None,
                    "outside_count": 0, "inside_count": 0, "outside_first": None,
                    "inside_first": None, "presence_emitted": False,
                    "segment_number": 1, "segment_id": f"{normalized_track_id}:s1"})
                last_seen = state.get("last_seen")
                if last_seen is not None and timestamp <= last_seen:
                    # Duplicate/out-of-order sample times cannot establish a
                    # physical transition and must not reset state as a new ID.
                    continue
                if last_seen is not None and timestamp - last_seen > max_track_gap_seconds:
                    # A stale/reused tracker ID starts a new segment. A missing
                    # observation is not evidence that the person exited.
                    previous_segment = state["segment_id"]
                    previous_phase = state["phase"]
                    previous_seen = float(last_seen)
                    state["segment_number"] += 1
                    state["segment_id"] = f"{normalized_track_id}:s{state['segment_number']}"
                    state.update({"phase": "unknown", "outside_count": 0, "inside_count": 0,
                                  "outside_first": None, "inside_first": None,
                                  "presence_emitted": False})
                    emitted.append({"event_type": "zone_tracking_gap", "label":
                        f"Tracking continuity for person track {normalized_track_id} was interrupted in {zone.get('name') or zone['zone_id']}",
                        "zone_id": str(zone["zone_id"]), "zone_name": zone.get("name") or zone["zone_id"],
                        "timestamp_seconds": float(timestamp),
                        "video_timestamp": format_video_timestamp(timestamp),
                        "frame_number": int(frame_number) if frame_number is not None else None,
                        "confidence": person.get("detection_confidence"), "anomaly_score": None,
                        "track_id": normalized_track_id, "track_segment_id": state["segment_id"],
                        "detection_method": "YOLO pose person box + ByteTrack observation-gap safeguard",
                        "details": {"previous_track_segment_id": previous_segment,
                            "previous_zone_phase": previous_phase,
                            "last_observed_timestamp_seconds": previous_seen,
                            "resumed_observation_timestamp_seconds": float(timestamp),
                            "observation_gap_seconds": round(float(timestamp) - previous_seen, 3),
                            "exit_inferred": False,
                            "interpretation": "Tracking continuity was unavailable across this observation gap. No exit or re-entry is inferred."}})
                state["last_seen"] = float(timestamp)

                if location == "boundary":
                    state["outside_count"] = state["inside_count"] = 0
                    state["outside_first"] = state["inside_first"] = None
                    continue

                def make_event(event_type, event_timestamp, event_frame, confidence, *,
                               transition_details=None, event_box=None, event_point=None):
                    zone_name = zone.get("name") or zone["zone_id"]
                    evidence_box = event_box if event_box is not None else box
                    evidence_point = event_point if event_point is not None else point
                    if event_type == "zone_presence_observed":
                        label = f"Person track {normalized_track_id} first observed inside {zone_name}; entry time unknown"
                        interpretation = "Initial occupancy observation only; not a confirmed entry, anomaly, or identity."
                    elif event_type == "RESTRICTED_ZONE_ENTRY":
                        label = f"Person track {normalized_track_id} entered restricted zone {zone_name}"
                        interpretation = "Confirmed tracked entry into an enabled polygon; requires human review and does not establish intent or identity."
                    elif event_type == "zone_tracking_gap":
                        label = f"Tracking continuity for person track {normalized_track_id} was interrupted in {zone_name}"
                        interpretation = "Observation gap; no exit, re-entry, identity continuity, or anomaly is inferred."
                    else:
                        label = f"Person track {normalized_track_id} exited zone {zone_name}"
                        interpretation = "Confirmed tracked exit from an enabled polygon; not an anomaly or identity."
                    details = {"zone_id": zone["zone_id"], "zone_name": zone_name,
                        "bottom_center_xy": [round(evidence_point[0], 2), round(evidence_point[1], 2)],
                        "person_bbox_xyxy": list(evidence_box), "polygon_signed_distance_px": signed_distance,
                        "boundary_margin_px": boundary_margin_px, "interpretation": interpretation}
                    if transition_details:
                        details.update(transition_details)
                    details["track_segment_id"] = state["segment_id"]
                    return {"event_type": event_type, "label": label,
                        "zone_id": str(zone["zone_id"]), "zone_name": zone_name,
                        "timestamp_seconds": float(event_timestamp),
                        "video_timestamp": format_video_timestamp(event_timestamp),
                        "frame_number": int(event_frame) if event_frame is not None else None,
                        "confidence": confidence, "anomaly_score": None,
                        "track_id": normalized_track_id,
                        "track_segment_id": state["segment_id"],
                        "detection_method": "YOLO pose person box + ByteTrack + confirmed polygon crossing",
                        "details": details}

                if state["phase"] == "unknown":
                    if location == "inside":
                        state["phase"] = "inside_unconfirmed"
                        if not state["presence_emitted"]:
                            state["presence_emitted"] = True
                            emitted.append(make_event("zone_presence_observed", timestamp, frame_number,
                                                      person.get("detection_confidence")))
                    else:
                        state["outside_count"] += 1
                        if state["outside_count"] >= exit_confirmation_frames:
                            state["phase"] = "outside"
                            state["outside_count"] = 0
                            state["outside_last_timestamp"] = float(timestamp)
                    continue

                if state["phase"] == "inside_unconfirmed":
                    if location == "outside":
                        state["outside_count"] += 1
                        if state["outside_count"] >= exit_confirmation_frames:
                            state["phase"] = "outside"
                            state["outside_count"] = 0
                            state["outside_last_timestamp"] = float(timestamp)
                            state["presence_emitted"] = False
                    else:
                        state["outside_count"] = 0
                    continue

                if state["phase"] == "outside":
                    if location == "inside":
                        if state["inside_count"] == 0:
                            state["inside_first"] = {"timestamp": float(timestamp),
                                "frame_number": frame_number, "point": [round(point[0], 2), round(point[1], 2)],
                                "bbox": list(box), "confidence": person.get("detection_confidence")}
                        state["inside_count"] += 1
                        if state["inside_count"] >= entry_confirmation_frames:
                            first = state["inside_first"]
                            emitted.append(make_event("RESTRICTED_ZONE_ENTRY", first["timestamp"],
                                first["frame_number"], first["confidence"], transition_details={
                                    "previously_confirmed_outside": True,
                                    "outside_confirmation_frames": exit_confirmation_frames,
                                    "inside_confirmation_frames": state["inside_count"],
                                    "last_outside_timestamp_seconds": float(state.get("outside_last_timestamp", timestamp)),
                                    "first_inside_timestamp_seconds": first["timestamp"],
                                    "crossing_bbox_xyxy": first["bbox"],
                                    "crossing_bottom_center_xy": first["point"],
                                }, event_box=first["bbox"], event_point=first["point"]))
                            state["phase"] = "inside"
                            state["inside_count"] = 0
                            state["inside_first"] = None
                    else:
                        state["inside_count"] = 0
                        state["inside_first"] = None
                        state["outside_last_timestamp"] = float(timestamp)
                    continue

                # A confirmed-inside track needs consecutive, clearly outside
                # samples before the state flips and a later re-entry is allowed.
                if state["phase"] == "inside":
                    if location == "outside":
                        if state["outside_count"] == 0:
                            state["outside_first"] = {"timestamp": float(timestamp),
                                "frame_number": frame_number}
                        state["outside_count"] += 1
                        if state["outside_count"] >= exit_confirmation_frames:
                            first = state["outside_first"]
                            emitted.append(make_event("zone_exit", first["timestamp"],
                                first["frame_number"], person.get("detection_confidence"),
                                transition_details={"outside_confirmation_frames": state["outside_count"]}))
                            state["phase"] = "outside"
                            state["outside_count"] = 0
                            state["outside_last_timestamp"] = float(timestamp)
                            state["outside_first"] = None
                    else:
                        state["outside_count"] = 0
                        state["outside_first"] = None
    return emitted


def make_browser_playable_webm(source_path, output_path):
    """Convert a legacy annotated MP4V file to VP8/WebM for HTML video players.

    Returns True only when the generated file can be reopened and a frame decoded.
    The source is left untouched; output is atomically published after validation.
    """
    if os.path.isfile(output_path):
        existing = cv2.VideoCapture(output_path)
        try:
            valid, _metadata = _validate_annotated_video(output_path)
            if valid:
                return True
        finally:
            existing.release()
        # Serving a request must never delete or replace a historical output.
        return False

    capture = cv2.VideoCapture(source_path)
    if not capture.isOpened():
        capture.release()
        return False
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    if fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    temporary_path = output_path[:-5] + ".partial-" + uuid.uuid4().hex + ".webm"
    writer = None
    frames = 0
    try:
        writer = cv2.VideoWriter(temporary_path, cv2.VideoWriter_fourcc(*"VP80"), fps, (width, height))
        if not writer.isOpened():
            writer.release()
            writer = None
            try:
                os.remove(temporary_path)
            except OSError:
                pass
            return False
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            frames += 1
    except cv2.error:
        try:
            os.remove(temporary_path)
        except OSError:
            pass
        return False
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if frames == 0:
        try:
            os.remove(temporary_path)
        except OSError:
            pass
        return False

    valid, _metadata = _validate_annotated_video(temporary_path)
    if not valid:
        try:
            os.remove(temporary_path)
        except OSError:
            pass
        return False
    try:
        published, _reason = _publish_without_overwrite(temporary_path, output_path)
        return published
    except OSError:
        try:
            os.remove(temporary_path)
        except OSError:
            pass
        return False


def make_browser_playable_mp4(source_path, output_path):
    """Create a finalized H.264/MP4 browser copy without changing the source.

    PyAV's FFmpeg libraries provide the H.264 encoder when available. The
    conversion is streamed frame-by-frame, uses a fast CPU preset, validates
    container metadata and decoded frames, and publishes without overwriting
    an existing artifact. Returns validated metadata, or None on failure.
    """
    try:
        import av
        from fractions import Fraction
    except ImportError:
        return None

    def validate_mp4(path):
        valid, metadata = _validate_annotated_video(path)
        if not valid:
            return None
        try:
            container = av.open(path)
            try:
                stream = next((item for item in container.streams if item.type == "video"), None)
                if (stream is None or stream.codec_context.name != "h264"
                        or container.format.name.split(",")[0] != "mov"):
                    return None
                duration = (float(stream.duration * stream.time_base)
                            if stream.duration is not None and stream.time_base else None)
                if duration is None or duration <= 0:
                    return None
                metadata.update({"codec": "H.264 (libx264)", "container": ".mp4",
                                 "duration_seconds": duration,
                                 "frame_count": int(stream.frames or metadata["frame_count"]),
                                 "fps": float(stream.average_rate or metadata["fps"])})
                return metadata
            finally:
                container.close()
        except Exception:
            return None

    if os.path.isfile(output_path):
        return validate_mp4(output_path)
    temporary_path = os.path.splitext(output_path)[0] + ".partial-" + uuid.uuid4().hex + ".mp4"
    source = destination = None
    try:
        source = av.open(source_path)
        input_stream = next((item for item in source.streams if item.type == "video"), None)
        if input_stream is None:
            return None
        fps = input_stream.average_rate or input_stream.base_rate
        if not fps or float(fps) <= 0:
            return None
        rate = Fraction(fps.numerator, fps.denominator)
        width, height = int(input_stream.width), int(input_stream.height)
        if width <= 0 or height <= 0:
            return None
        destination = av.open(temporary_path, mode="w", format="mp4")
        try:
            output_stream = destination.add_stream("libx264", rate=rate)
        except Exception:
            return None
        output_stream.width = width
        output_stream.height = height
        output_stream.pix_fmt = "yuv420p"
        output_stream.options = {"preset": "ultrafast", "crf": "23"}
        output_stream.codec_context.max_b_frames = 0
        frame_number = 0
        for frame in source.decode(input_stream):
            converted = frame.reformat(width=width, height=height, format="yuv420p")
            converted.pts = frame_number
            converted.time_base = Fraction(rate.denominator, rate.numerator)
            for packet in output_stream.encode(converted):
                destination.mux(packet)
            frame_number += 1
        for packet in output_stream.encode():
            destination.mux(packet)
        destination.close()
        destination = None
        source.close()
        source = None
        if frame_number < 1:
            return None
        metadata = validate_mp4(temporary_path)
        if not metadata or metadata.get("frame_count", 0) < 1:
            return None
        published, _reason = _publish_without_overwrite(temporary_path, output_path)
        if not published:
            return validate_mp4(output_path)
        return validate_mp4(output_path)
    except Exception:
        return None
    finally:
        for container in (source, destination):
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass
        if os.path.isfile(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass

def _publish_without_overwrite(temporary_path, output_path):
    """Atomically publish on the same filesystem without replacing history."""
    try:
        # A hard-link create is atomic and fails if another request already
        # published this name. Both paths are deliberately in PROCESSED_DIR.
        os.link(temporary_path, output_path)
    except FileExistsError:
        return False, "exists"
    except (OSError, NotImplementedError):
        return False, "publish_failed"
    try:
        os.remove(temporary_path)
    except OSError:
        # The final artifact is valid; leaving a private .partial file is safer
        # than marking a successfully published video unavailable.
        pass
    return True, None


def _validate_annotated_video(path, expected=None):
    """Validate a finalized media file and selected frames before publication."""
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        return False, None
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            return False, None
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = count / fps if fps > 0 and count > 0 else 0.0
        if fps <= 0 or count <= 0 or width <= 0 or height <= 0:
            return False, None
        if expected:
            expected_width, expected_height, expected_frames, expected_fps = expected
            if ((width, height) != (expected_width, expected_height)
                    or abs(count - expected_frames) > max(2, int(expected_frames * .02))
                    or abs(fps - expected_fps) > max(.1, expected_fps * .01)):
                return False, None
        decoded = []
        for frame_index in (0, count // 2, count - 1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            decoded.append(bool(ok and frame is not None and getattr(frame, "size", 0)))
        if not all(decoded):
            return False, None
        extension = os.path.splitext(path)[1].lower()
        return True, {"container": extension,
                      "codec": "VP8" if extension == ".webm" else "not identified by OpenCV",
                      "fps": fps,
                      "frame_count": count, "width": width, "height": height,
                      "duration_seconds": duration, "size_bytes": os.path.getsize(path),
                      "sampled_frames_decoded": decoded}
    except (OSError, cv2.error, ValueError):
        return False, None
    finally:
        capture.release()


def analyze_video(path, sample_seconds, max_duration, detector=None, progress=None, output_path=None,
                 face_detector=None, object_detector=None, zones=None,
                 zone_sample_seconds=0.5, zone_entry_confirmation_frames=2,
                 zone_exit_confirmation_frames=2, zone_track_max_gap_seconds=1.5,
                 zone_boundary_margin_px=8.0, fight_detector=None, output_max_width=640):
    start = time.monotonic()
    profile = defaultdict(float)
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise VideoError("OpenCV could not read this video. Try an MP4 encoded with H.264.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 else 0
    if duration <= 0:
        capture.release()
        raise VideoError("The file has no readable video frames.")
    if duration > max_duration:
        capture.release()
        raise VideoError(f"Video is {duration:.0f}s; maximum allowed is {max_duration}s.")
    # Read at a bounded interval without retaining a complete video in memory.
    active_zones = [zone for zone in (zones or []) if zone.get("enabled") is True]
    try:
        zone_sample_seconds = max(0.1, float(zone_sample_seconds))
    except (TypeError, ValueError):
        zone_sample_seconds = 0.5
    effective_sample_seconds = min(float(sample_seconds), zone_sample_seconds) if active_zones else float(sample_seconds)
    interval = max(1, int(fps * effective_sample_seconds))
    read = 0
    events = []
    pose_frames = []
    pose_error = None
    pose_frames_sampled = 0
    pose_frames_with_people = 0
    face_frames = []
    face_error = None
    object_error = None
    fight_error = None
    object_frames_sampled = 0
    object_counts = Counter()
    person_track_ids = set()
    pose_track_ids = set()
    object_observations = 0
    frame_index = 0
    last_video_timestamp = None
    writer = None
    annotated = False
    annotated_output_error = None
    annotated_output_metadata = None
    temporary_output_path = None
    try:
        output_max_width = max(0, int(output_max_width or 0))
    except (TypeError, ValueError):
        output_max_width = 640
    output_scale = (output_max_width / width if output_max_width and width > output_max_width else 1.0)
    output_width = max(2, int(round(width * output_scale / 2) * 2))
    output_height = max(2, int(round(height * output_scale / 2) * 2))
    if (detector is not None or face_detector is not None or object_detector is not None or
            fight_detector is not None or zones) and output_path:
        try:
            # VP8 belongs in WebM. Write to a unique temporary WebM and publish
            # only after final metadata and key-frame validation.
            temporary_output_path = os.path.splitext(output_path)[0] + ".partial-" + uuid.uuid4().hex + ".webm"
            writer = cv2.VideoWriter(temporary_output_path, cv2.VideoWriter_fourcc(*"VP80"), fps,
                                     (output_width, output_height))
            if not writer.isOpened():
                writer.release()
                writer = None
                annotated_output_error = "OpenCV could not initialize the VP8/WebM encoder."
        except cv2.error:
            writer = None
            annotated_output_error = "OpenCV could not initialize the VP8/WebM encoder."
    try:
        while frame_index < frame_count:
            stage_started = time.perf_counter()
            ok, frame = capture.read()
            profile["video_decode_seconds"] += time.perf_counter() - stage_started
            if not ok:
                break
            # Prefer advancing OpenCV presentation timestamps (useful for
            # variable-frame-rate files); fall back to frame index / FPS when
            # the backend returns a stale, zero, or invalid value.
            fallback_timestamp = frame_index / fps if fps > 0 else 0.0
            try:
                presentation_timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            except (TypeError, ValueError, cv2.error):
                presentation_timestamp = None
            if (presentation_timestamp is not None and math.isfinite(presentation_timestamp)
                    and presentation_timestamp >= 0
                    and (last_video_timestamp is None or presentation_timestamp > last_video_timestamp)
                    and (duration <= 0 or presentation_timestamp <= duration + 1.0)):
                timestamp = round(presentation_timestamp, 6)
            else:
                timestamp = round(fallback_timestamp, 6)
            last_video_timestamp = timestamp
            if fight_detector is not None:
                try:
                    stage_started = time.perf_counter()
                    fight_detector.observe_frame(frame, timestamp, frame_index)
                    profile["fight_pipeline_seconds"] += time.perf_counter() - stage_started
                except Exception as exc:
                    config = current_app.config if has_app_context() else {}
                    log.error("Optional fighting inference stopped (%s); traceback follows:\n%s",
                              type(exc).__name__, sanitized_traceback(exc, config))
                    fight_error = "Temporal fighting inference stopped after a model error. See server logs for the exception type."
                    fight_detector = None
            if frame_index % interval == 0:
                read += 1
                source_frame = frame
                if detector is not None:
                    try:
                        if hasattr(detector, "inspect_with_evidence"):
                            stage_started = time.perf_counter()
                            frame, event, pose_evidence = detector.inspect_with_evidence(frame, timestamp)
                            profile["pose_and_tracking_seconds"] += time.perf_counter() - stage_started
                            pose_evidence["frame_number"] = int(frame_index)
                            pose_evidence["detection_method"] = getattr(detector, "METHOD", "pose estimator")
                            pose_frames.append(pose_evidence)
                            pose_frames_sampled += 1
                            pose_frames_with_people += int(pose_evidence.get("person_count", 0) > 0)
                            for person in pose_evidence.get("people", []):
                                if person.get("track_id") is not None:
                                    pose_track_ids.add(person["track_id"])
                        else:
                            stage_started = time.perf_counter()
                            frame, event = detector.inspect(frame, timestamp)
                            profile["pose_and_tracking_seconds"] += time.perf_counter() - stage_started
                            pose_frames_sampled += 1
                        annotated = writer is not None
                        if event:
                            event["frame_number"] = int(frame_index)
                            events.append(event)
                    except Exception as exc:
                        config = current_app.config if has_app_context() else {}
                        log.error("Optional pose inference stopped (%s); traceback follows:\n%s",
                                  type(exc).__name__, sanitized_traceback(exc, config))
                        pose_error = "Pose inference stopped after a model error. See server logs for the exception type."
                        detector = None
                if object_detector is not None:
                    try:
                        stage_started = time.perf_counter()
                        observations = object_detector.detect(source_frame, timestamp, frame_index)
                        profile["object_detection_and_tracking_seconds"] += time.perf_counter() - stage_started
                        object_frames_sampled += 1
                        object_observations += len(observations)
                        for observation in observations:
                            events.append(observation)
                            object_counts[observation["label"]] += 1
                            if observation.get("label") == "person" and observation.get("track_id") is not None:
                                person_track_ids.add(observation["track_id"])
                        stage_started = time.perf_counter()
                        frame = object_detector.annotate(frame, observations)
                        profile["object_annotation_seconds"] += time.perf_counter() - stage_started
                        annotated = annotated or writer is not None
                    except Exception as exc:
                        # Optional object inference must not invalidate decoded video,
                        # face analysis, or the existing fall heuristic.
                        config = current_app.config if has_app_context() else {}
                        log.error("Optional object inference stopped (%s); traceback follows:\n%s",
                                  type(exc).__name__, sanitized_traceback(exc, config))
                        object_error = "Object detection stopped after a model inference error."
                        object_detector = None
                if face_detector is not None:
                    try:
                        stage_started = time.perf_counter()
                        boxes = face_detector.detect(source_frame)
                        profile["face_detection_seconds"] += time.perf_counter() - stage_started
                        face_frames.append({"frame_index": int(frame_index),
                                            "timestamp_seconds": float(timestamp),
                                            "face_count": len(boxes), "boxes": boxes})
                        stage_started = time.perf_counter()
                        frame = face_detector.annotate(frame, boxes)
                        profile["face_annotation_seconds"] += time.perf_counter() - stage_started
                        annotated = writer is not None
                    except Exception as exc:
                        # A face-detector error must not stop the existing anomaly
                        # pipeline or invalidate successfully decoded video frames.
                        config = current_app.config if has_app_context() else {}
                        log.error("Optional face inference stopped (%s); traceback follows:\n%s",
                                  type(exc).__name__, sanitized_traceback(exc, config))
                        face_error = "Face detection stopped after an OpenCV processing error."
                        face_detector = None
                if progress:
                    progress(min(95, int(95 * frame_index / max(frame_count, 1))))
            if writer is not None:
                try:
                    if zones:
                        from app.services.roi import draw_zones
                        frame = draw_zones(frame, zones)
                        annotated = True
                    encode_frame = frame
                    if output_scale < 1.0:
                        stage_started = time.perf_counter()
                        encode_frame = cv2.resize(frame, (output_width, output_height),
                                                   interpolation=cv2.INTER_AREA)
                        profile["annotation_resize_seconds"] += time.perf_counter() - stage_started
                    stage_started = time.perf_counter()
                    writer.write(encode_frame)
                    profile["annotated_encode_seconds"] += time.perf_counter() - stage_started
                except cv2.error:
                    writer.release()
                    writer = None
                    annotated = False
                    annotated_output_error = "The annotated video encoder failed while writing frames."
            frame_index += 1
    except Exception:
        if temporary_output_path and os.path.isfile(temporary_output_path):
            try:
                os.remove(temporary_output_path)
            except OSError:
                pass
        raise
    finally:
        capture.release()
        if writer is not None:
            stage_started = time.perf_counter()
            writer.release()
            profile["annotated_encode_seconds"] += time.perf_counter() - stage_started
    # A writer can open successfully but still produce an incomplete/corrupt
    # file. Validate dimensions, FPS, frame count, and first/middle/final frames.
    if annotated and output_path:
        stage_started = time.perf_counter()
        annotated, annotated_output_metadata = _validate_annotated_video(
            temporary_output_path, expected=(output_width, output_height, frame_index, fps))
        profile["annotated_output_validation_seconds"] += time.perf_counter() - stage_started
        if annotated:
            published, reason = _publish_without_overwrite(temporary_output_path, output_path)
            if not published:
                annotated = False
                annotated_output_metadata = None
                annotated_output_error = ("An annotated file already exists at this path; the historical output was preserved."
                                          if reason == "exists" else
                                          "The validated annotated video could not be published safely.")
        else:
            annotated = False
            annotated_output_metadata = None
            annotated_output_error = "The generated annotated video failed final metadata or frame validation."
        if not annotated and temporary_output_path and os.path.isfile(temporary_output_path):
            try:
                os.remove(temporary_output_path)
            except OSError:
                pass
    elif temporary_output_path and os.path.isfile(temporary_output_path):
        try:
            os.remove(temporary_output_path)
        except OSError:
            pass
    if read == 0:
        raise VideoError("No readable frames were found in this file.")
    fight_events = []
    fight_analysis = {"status": "not_configured", "method": "X3D-M temporal video classification",
                      "candidate_event_count": 0, "error_message": None}
    if fight_detector is not None:
        try:
            stage_started = time.perf_counter()
            fight_events, fight_analysis = fight_detector.finish()
            profile["fight_finalize_seconds"] += time.perf_counter() - stage_started
        except Exception as exc:
            config = current_app.config if has_app_context() else {}
            log.error("Could not finalize fighting events (%s); traceback follows:\n%s",
                      type(exc).__name__, sanitized_traceback(exc, config))
            fight_analysis = {"status": "failed", "candidate_event_count": 0,
                              "error_message": "Temporal fighting analysis could not be finalized."}
    elif fight_error:
        fight_analysis.update({"status": "failed", "error_message": fight_error})
    events.extend(fight_events)
    if zones:
        stage_started = time.perf_counter()
        from app.services.roi import associate_event_with_zones
        for event in events:
            if event.get("event_type") == "SUSPECTED_FIGHTING":
                # Clip classification has no spatial localization; don't infer
                # that a candidate happened inside a person or ROI polygon.
                event["zone_association_status"] = "unavailable"
                event.setdefault("details", {})["zone_association_status"] = "unavailable"
            elif event.get("event_type") not in {"object_detection", "zone_entry", "zone_exit"}:
                associate_event_with_zones(event, zones, width, height)
        events.extend(_zone_transition_events(
            pose_frames, zones, width, height,
            entry_confirmation_frames=zone_entry_confirmation_frames,
            exit_confirmation_frames=zone_exit_confirmation_frames,
            max_track_gap_seconds=zone_track_max_gap_seconds,
            boundary_margin_px=zone_boundary_margin_px))
        profile["restricted_zone_logic_seconds"] += time.perf_counter() - stage_started
    frames_with_faces = sum(1 for item in face_frames if item["face_count"] > 0)
    face_counts = [item["face_count"] for item in face_frames]
    if fight_detector is not None:
        fight_inference = float(getattr(fight_detector, "inference_seconds", 0.0) or 0.0)
        profile["x3d_model_inference_seconds"] = fight_inference
        profile["fight_sampling_and_preprocess_seconds"] = max(
            0.0, profile.get("fight_pipeline_seconds", 0.0) - fight_inference)
    return {"duration": duration, "fps": fps, "width": width, "height": height,
            "processing_seconds": time.monotonic() - start, "sampled_frames": read,
            "annotated_resolution": [output_width, output_height] if annotated else None,
            "stage_timings": {name: round(seconds, 6) for name, seconds in sorted(profile.items())},
            "sample_interval_seconds": interval / fps if fps > 0 else effective_sample_seconds,
            "events": events, "method": getattr(detector, "METHOD", "YOLO pose COCO keypoints")
            if (detector or pose_frames_sampled or pose_error) else
            getattr(fight_detector, "METHOD", "X3D-M temporal video classification")
            if fight_analysis.get("status") == "complete" else "validation_only",
            "analysis_mode": "pose_heuristic" if (detector or pose_frames_sampled or pose_error) else
            "temporal_action" if fight_analysis.get("status") == "complete" else "validation_only",
            "annotated_output": bool(annotated),
            "annotated_output_error": annotated_output_error,
            "annotated_output_metadata": annotated_output_metadata,
            "pose_frames": pose_frames,
            "pose_analysis": {"status": "failed" if pose_error else ("complete" if pose_frames_sampled else "not_configured"),
                              "method": getattr(detector, "METHOD", "YOLO pose COCO keypoints") if detector else "YOLO pose COCO keypoints",
                              "model_name": getattr(detector, "model_name", None),
                              "model_version": getattr(detector, "model_version", None),
                              "frames_sampled": pose_frames_sampled,
                              "frames_with_people": pose_frames_with_people,
                              "person_track_count_estimate": len(pose_track_ids),
                              "person_track_count_note": "Approximate temporary pose-tracker IDs; not unique people.",
                              "tracking_status": ("complete" if pose_track_ids else
                                  "ids_unavailable" if pose_frames_with_people else
                                  "no_people_observed" if pose_frames_sampled else "not_run"),
                              "zone_presence_observed_count": sum(event["event_type"] == "zone_presence_observed" for event in events),
                              "zone_entry_count": sum(event["event_type"] == "RESTRICTED_ZONE_ENTRY" for event in events),
                              "zone_exit_count": sum(event["event_type"] == "zone_exit" for event in events),
                              "error_message": pose_error},
            "fight_analysis": fight_analysis,
            "face_frames": face_frames,
            "face_analysis": {"status": "failed" if face_error else ("complete" if face_counts else "unavailable"),
                              "method": face_detector.METHOD if face_detector else "OpenCV Haar cascade (haarcascade_frontalface_default)",
                              "frames_sampled": len(face_frames),
                              "frames_with_faces": frames_with_faces,
                              "face_observations": sum(face_counts),
                              "max_faces_per_frame": max(face_counts, default=0),
                              "error_message": face_error},
            "object_analysis": {"status": "failed" if object_error else ("complete" if object_frames_sampled else "not_run"),
                                "method": getattr(object_detector, "METHOD", "Ultralytics YOLO COCO detector + ByteTrack"),
                                "frames_sampled": object_frames_sampled,
                                "observations": object_observations,
                                "counts_by_class": dict(sorted(object_counts.items())),
                                "person_track_count_estimate": len(person_track_ids),
                                "person_track_count_note": "Approximate temporary tracker IDs, not unique identities.",
                                "confidence_threshold": getattr(object_detector, "confidence", None),
                                "error_message": object_error}}
