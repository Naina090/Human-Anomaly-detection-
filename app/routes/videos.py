import os
import uuid
import logging
import time
import os.path
import math
import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from bson import ObjectId
from pymongo.errors import DuplicateKeyError
from flask import Blueprint, current_app, render_template, request, redirect, url_for, flash, send_file, abort, jsonify, Response
from werkzeug.utils import secure_filename
from app.services.repository import (create_video, find_video_by_upload_request, get_video, get_events, update_video,
    create_job, update_job, get_job_for_video, get_face_frames, save_face_frames,
    get_pose_frames, save_pose_frames, claim_job)
from app.db import DatabaseUnavailable
from app.services.storage import safe_video_path, safe_processed_path
from app.services.video_processor import (analyze_video, VideoError, make_browser_playable_webm,
                                          make_browser_playable_mp4)
from app.services.timestamps import event_video_seconds, format_video_timestamp
from app.services.fall_detector import YoloPoseFallDetector, DetectorUnavailable, detector_status, METHOD as POSE_METHOD
from app.services.face_detector import OpenCVFaceDetector, FaceDetectorUnavailable, detector_status as face_detector_status
from app.services.object_detector import (ObjectDetectorUnavailable, YoloObjectTracker,
    detector_status as object_detector_status)
from app.services.fight_detector import FightActionDetector, FightDetectorUnavailable
from app.services.report_service import build_analysis_summary, video_report_csv, explain_event
from app.services.pdf_report import build_pdf
from app.services.video_validator import validate_video_file, VideoValidationError
from app.services.roi import ZoneValidationError, parse_zones
from app.logging_utils import sanitized_traceback

bp = Blueprint("videos", __name__)
log = logging.getLogger(__name__)
ALLOWED = {"mp4", "avi", "mov", "mkv"}


def group_object_observations(events):
    """Summarize sampled object observations by temporary tracker ID.

    Records without a tracker ID remain separate single observations so they
    are never presented as one inferred object or person.
    """
    groups = {}
    for index, event in enumerate(events):
        if event.get("event_type") != "object_detection":
            continue
        track_id = event.get("track_id", event.get("track"))
        key = ("track", str(track_id)) if track_id is not None else (
            "untracked", str(event.get("_id", index)), index)
        summary = groups.get(key)
        if summary is None:
            summary = {"track_id": track_id, "labels": set(), "observation_count": 0,
                       "timestamps": []}
            groups[key] = summary
        label = str(event.get("label") or "Unknown object")
        summary["labels"].add(label)
        summary["observation_count"] += 1
        timestamp = event_video_seconds(event)
        if timestamp is not None:
            summary["timestamps"].append(timestamp)

    output = []
    for summary in groups.values():
        labels = sorted(summary["labels"])
        timestamps = summary.pop("timestamps")
        first = min(timestamps) if timestamps else None
        last = max(timestamps) if timestamps else None
        summary["object_class"] = labels[0] if len(labels) == 1 else "Class varied: " + ", ".join(labels)
        summary["first_seen_seconds"] = first
        summary["last_seen_seconds"] = last
        summary["first_seen_label"] = format_video_timestamp(first) if first is not None else "Time unavailable"
        summary["last_seen_label"] = format_video_timestamp(last) if last is not None else "Time unavailable"
        summary["track_label"] = str(summary["track_id"]) if summary["track_id"] is not None else "Unavailable"
        output.append(summary)
    return sorted(output, key=lambda item: (
        item["first_seen_seconds"] is None,
        item["first_seen_seconds"] if item["first_seen_seconds"] is not None else 0,
        item["track_id"] is None,
        str(item["track_id"] or ""), item["object_class"]))


@bp.get("/upload")
def upload_page():
    return render_template("upload.html", max_mb=current_app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
                           upload_request_id=uuid.uuid4().hex,
                           detector=detector_status(current_app.config.get("POSE_MODEL_PATH", "")),
                           face_detector=face_detector_status(current_app.config.get("FACE_CASCADE_PATH", "")),
                           object_detector=object_detector_status(
                               current_app.config.get("OBJECT_MODEL_PATH", ""),
                               current_app.config.get("OBJECT_TRACKER", "bytetrack.yaml")))


@bp.post("/upload")
def upload():
    file = request.files.get("video")
    if not file or not file.filename:
        flash("Choose a video file first.", "error")
        return redirect(url_for("videos.upload_page"))
    clean = secure_filename(file.filename)
    ext = clean.rsplit(".", 1)[-1].lower() if "." in clean else ""
    if ext not in ALLOWED:
        flash("Supported extensions: MP4, AVI, MOV and MKV (decoder support varies by system).", "error")
        return redirect(url_for("videos.upload_page"))
    try:
        zones = parse_zones(request.form.get("roi_zones"))
    except ZoneValidationError as exc:
        flash(str(exc), "error")
        return redirect(url_for("videos.upload_page"))
    upload_request_id = request.form.get("upload_request_id", "")
    try:
        upload_request_id = uuid.UUID(hex=upload_request_id).hex
    except (ValueError, AttributeError):
        # Older/API clients without the field remain compatible; browser forms
        # receive one fresh key per rendered upload page.
        upload_request_id = uuid.uuid4().hex
    existing = find_video_by_upload_request(upload_request_id)
    if existing:
        return redirect(url_for("videos.results", video_id=str(existing["_id"])))
    video_id = uuid.uuid4().hex
    stored = video_id + "." + ext
    path = os.path.join(current_app.config["UPLOAD_DIR"], stored)
    try:
        file.save(path)
        file_size = os.path.getsize(path) if os.path.isfile(path) else 0
        if file_size <= 0:
            raise VideoValidationError("The selected video file is empty. Choose a non-empty video.")
        # Verify actual decodable video content before creating MongoDB records
        # or occupying a background-processing slot. This requires no AI model.
        video_metadata = validate_video_file(path)
        maximum_duration = current_app.config["MAX_DURATION_SECONDS"]
        if video_metadata["duration"] is not None and video_metadata["duration"] > maximum_duration:
            raise VideoValidationError(
                f"This video is longer than the configured {maximum_duration}-second limit. Choose a shorter video."
            )
    except VideoValidationError as exc:
        if os.path.isfile(path):
            os.remove(path)
        flash(str(exc), "error")
        return redirect(url_for("videos.upload_page"))
    except OSError as exc:
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                log.warning("Could not remove incomplete upload after a filesystem error.")
        log.error("Video upload filesystem failure (%s); traceback follows:\n%s",
                  type(exc).__name__, sanitized_traceback(exc, current_app.config))
        raise
    document = {
        "upload_request_id": upload_request_id,
        "original_filename": clean,
        "stored_filename": stored,
        "file_path": stored,
        "upload_date": datetime.now(timezone.utc),
        "processing_status": "queued",
        **video_metadata,
        "validation_status": "valid",
        "validation_message": "OpenCV opened the file and decoded a video frame.",
        "processing_duration": None, "error_message": None,
        "file_size": file_size,
        "zones": zones,
        "zone_analysis": {"status": "queued" if any(zone["enabled"] for zone in zones) else
                          ("disabled" if zones else "not_configured"),
                          "zone_count": len(zones),
                          "active_zone_count": sum(zone["enabled"] for zone in zones)},
        "face_analysis": {"status": "queued", "method": OpenCVFaceDetector.METHOD,
                          "frames_sampled": 0, "frames_with_faces": 0,
                          "face_observations": 0, "max_faces_per_frame": 0},
        "pose_analysis": {"status": "queued" if current_app.config.get("POSE_MODEL_PATH") else "not_configured",
                          "method": POSE_METHOD, "frames_sampled": 0,
                          "frames_with_people": 0},
        "object_analysis": {"status": "queued" if current_app.config.get("OBJECT_MODEL_PATH") else "not_configured",
                            "method": YoloObjectTracker.METHOD, "frames_sampled": 0,
                            "observations": 0, "counts_by_class": {},
                            "person_track_count_estimate": 0},
        "fight_analysis": {"status": "queued" if current_app.config.get("FIGHT_DETECTION_ENABLED") and current_app.config.get("FIGHT_MODEL_PATH") else
                           ("disabled" if not current_app.config.get("FIGHT_DETECTION_ENABLED") else "not_configured"),
                           "candidate_event_count": 0},
    }
    try:
        record_id = create_video(document)
    except DuplicateKeyError:
        existing = find_video_by_upload_request(upload_request_id)
        if os.path.isfile(path):
            os.remove(path)
        if existing:
            return redirect(url_for("videos.results", video_id=str(existing["_id"])))
        raise
    except DatabaseUnavailable as exc:
        try:
            from app.services.local_fallback import save_pending_upload
            record_id, job_id = save_pending_upload(document, str(exc))
            from app.services.local_fallback import get_pending_video
            saved_local = get_pending_video(record_id) or {}
            if saved_local.get("stored_filename") != stored:
                if os.path.isfile(path):
                    os.remove(path)
                return redirect(url_for("videos.results", video_id=record_id))
            app = current_app._get_current_object()
            if not app.extensions["processing_slots"].acquire(blocking=False):
                from app.services.local_fallback import fail_pending_video
                fail_pending_video(record_id, "The local processing queue is full. Retry this video shortly.")
            else:
                try:
                    app.extensions["processing_executor"].submit(_run_analysis, app, record_id, job_id, path)
                    log.warning("Video %s and job %s are running in local fallback; MongoDB remains unavailable.",
                                record_id[:8], job_id[:8])
                except RuntimeError:
                    app.extensions["processing_slots"].release()
                    from app.services.local_fallback import fail_pending_video
                    fail_pending_video(record_id, "The local worker is shutting down. Restart Flask and retry this video.")
            return redirect(url_for("videos.results", video_id=record_id))
        except Exception as fallback_error:
            log.error("Local upload fallback failed (%s); validated upload could not be indexed.",
                      type(fallback_error).__name__)
            if os.path.isfile(path):
                os.remove(path)
            return render_template("error.html", error_label="UPLOAD NOT QUEUED",
                error_title="The video could not be saved to history.",
                message="MongoDB is unavailable and the local fallback could not be written. The upload was removed safely."), 503
    except Exception:
        if os.path.isfile(path):
            os.remove(path)
        raise
    video_id = str(record_id)
    app = current_app._get_current_object()
    if not app.extensions["processing_slots"].acquire(blocking=False):
        message = "The processing queue is full. Try again shortly."
        update_video(video_id, {"processing_status": "failed", "error_message": message,
                                "face_analysis": {"status": "not_run", "method": OpenCVFaceDetector.METHOD,
                                                  "error_message": message}})
        flash("The processing queue is full. Try again shortly.", "error")
        return redirect(url_for("videos.results", video_id=video_id))
    now = datetime.now(timezone.utc)
    try:
        job_id = create_job({"video_id": ObjectId(video_id), "status": "queued", "progress_percent": 0,
                             "started_at": None, "created_at": now, "completed_at": None, "error_message": None})
        update_video(video_id, {"active_job_id": job_id})
    except Exception:
        app.extensions["processing_slots"].release()
        update_video(video_id, {"processing_status": "failed",
                                "error_message": "The processing job could not be saved to MongoDB.",
                                "face_analysis": {"status": "not_run", "method": OpenCVFaceDetector.METHOD}})
        raise
    try:
        app.extensions["processing_executor"].submit(_run_analysis, app, video_id, str(job_id), path)
    except RuntimeError:
        app.extensions["processing_slots"].release()
        message = "The local processing worker is shutting down. Retry this video after restarting the website."
        update_video(video_id, {"processing_status": "failed", "error_message": message})
        update_job(str(job_id), {"status": "failed", "error_message": message,
                                 "completed_at": datetime.now(timezone.utc)})
    return redirect(url_for("videos.results", video_id=video_id))


def _run_analysis(app, video_id, job_id, path):
    started = time.monotonic()
    worker_profile = defaultdict(float)
    def persist_call(name, function, *args, **kwargs):
        persist_started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            worker_profile["database_local_persistence_seconds"] += time.perf_counter() - persist_started
    output_name = video_id + "_annotated.webm"
    output_path = os.path.join(app.config["PROCESSED_DIR"], output_name)
    compatibility_output_path = None
    if os.path.exists(output_path):
        # Retries must not replace an older annotated artifact for the same
        # record. The selected internal name is persisted with final results.
        output_name = video_id + "_annotated_" + uuid.uuid4().hex[:8] + ".webm"
        output_path = os.path.join(app.config["PROCESSED_DIR"], output_name)
    try:
        with app.app_context():
            claimed_job = claim_job(job_id)
            if not claimed_job:
                log.info("Skipped duplicate or non-queued worker for job %s.", str(job_id)[:8])
                return
            persist_call("persistence", update_video, video_id, {"processing_status": "processing"})
            saved_video = get_video(video_id) or {}
            zones = saved_video.get("zones") or []
            detector = None
            face_detector = None
            object_detector = None
            fight_detector = None
            face_init_error = None
            object_init_error = None
            pose_init_error = None
            fight_init_error = None
            try:
                stage_started = time.perf_counter()
                face_detector = OpenCVFaceDetector(app.config.get("FACE_CASCADE_PATH", ""))
                worker_profile["face_model_initialization_seconds"] = time.perf_counter() - stage_started
            except FaceDetectorUnavailable as exc:
                face_init_error = str(exc)
            except Exception as exc:
                log.error("Face detector initialization failed (%s); traceback follows:\n%s",
                          type(exc).__name__, sanitized_traceback(exc, app.config))
                face_init_error = "The OpenCV face detector failed to initialize."
            persist_call("persistence", update_video, video_id, {"face_analysis": {
                "status": "processing" if face_detector else "unavailable",
                "method": OpenCVFaceDetector.METHOD,
                "error_message": face_init_error,
            }})
            object_model_path = app.config.get("OBJECT_MODEL_PATH", "")
            if object_model_path:
                try:
                    stage_started = time.perf_counter()
                    object_detector = YoloObjectTracker(
                        object_model_path,
                        app.config.get("OBJECT_MODEL_CONFIDENCE", 0.25),
                        app.config.get("OBJECT_TRACKER", "bytetrack.yaml"),
                    )
                    worker_profile["object_model_initialization_seconds"] = time.perf_counter() - stage_started
                except ObjectDetectorUnavailable as exc:
                    object_init_error = str(exc)
                except Exception as exc:
                    log.error("Object detector initialization failed (%s); traceback follows:\n%s",
                              type(exc).__name__, sanitized_traceback(exc, app.config))
                    object_init_error = "The configured object detector could not be initialized."
            object_status = "processing" if object_detector else (
                "unavailable" if object_model_path else "not_configured")
            persist_call("persistence", update_video, video_id, {"object_analysis": {
                "status": object_status, "method": YoloObjectTracker.METHOD,
                "model_name": os.path.basename(object_model_path) if object_model_path else None,
                "error_message": object_init_error,
            }})
            model_path = app.config.get("POSE_MODEL_PATH", "")
            if model_path:
                persist_call("persistence", update_video, video_id, {"pose_analysis": {"status": "processing",
                    "method": POSE_METHOD,
                    "model_name": os.path.basename(model_path)}})
                try:
                    stage_started = time.perf_counter()
                    detector = YoloPoseFallDetector(model_path, {
                        "transition_window_seconds": app.config.get("FALL_TRANSITION_WINDOW_SECONDS", 1.5),
                        "minimum_vertical_drop": app.config.get("FALL_MIN_VERTICAL_DROP", .12),
                        "minimum_horizontal_persistence": app.config.get("FALL_MIN_HORIZONTAL_PERSISTENCE", .6),
                        "candidate_score_threshold": app.config.get("FALL_CANDIDATE_SCORE_THRESHOLD", .55),
                        "tracker": app.config.get("POSE_TRACKER", "bytetrack.yaml"),
                    })
                    worker_profile["pose_model_initialization_seconds"] = time.perf_counter() - stage_started
                except DetectorUnavailable as exc:
                    pose_init_error = str(exc)
                except Exception as exc:
                    log.error("Pose detector initialization failed (%s); traceback follows:\n%s",
                              type(exc).__name__, sanitized_traceback(exc, app.config))
                    pose_init_error = "The configured pose detector could not be initialized."
            else:
                persist_call("persistence", update_video, video_id, {"pose_analysis": {"status": "not_configured",
                    "method": POSE_METHOD, "model_name": None}})
            fight_model_path = app.config.get("FIGHT_MODEL_PATH", "")
            fight_enabled = app.config.get("FIGHT_DETECTION_ENABLED", True)
            fight_initial_status = "disabled" if not fight_enabled else "not_configured"
            if fight_enabled and fight_model_path:
                try:
                    stage_started = time.perf_counter()
                    fight_detector = FightActionDetector(
                        fight_model_path,
                        enabled=True,
                        threshold=app.config.get("FIGHT_SCORE_THRESHOLD", 0.4),
                        window_seconds=app.config.get("FIGHT_WINDOW_SECONDS", 4.0),
                        stride_seconds=app.config.get("FIGHT_WINDOW_STRIDE", 1.0),
                        confirmation_windows=app.config.get("FIGHT_CONFIRMATION_WINDOWS", 2),
                        device=app.config.get("FIGHT_DEVICE", "auto"),
                    )
                    worker_profile["fight_model_initialization_seconds"] = time.perf_counter() - stage_started
                    if fight_detector.status == "ready":
                        fight_initial_status = "processing"
                    else:
                        fight_init_error = fight_detector.error_message or "The configured fighting model is unavailable."
                        fight_initial_status = "unavailable"
                        fight_detector = None
                except Exception as exc:
                    log.error("Fighting detector initialization failed (%s); traceback follows:\n%s",
                              type(exc).__name__, sanitized_traceback(exc, app.config))
                    fight_init_error = "The configured temporal action detector could not be initialized."
                    fight_initial_status = "unavailable"
                    fight_detector = None
            persist_call("persistence", update_video, video_id, {"fight_analysis": {
                "status": fight_initial_status,
                "method": "X3D-M temporal video classification (RWF-2000 Violence / Non-Violence)",
                "model_name": os.path.basename(fight_model_path) if fight_model_path else None,
                "candidate_event_count": 0,
                "error_message": fight_init_error,
            }})
            output_path = output_path if (detector or face_detector or object_detector or fight_detector or zones) else None
            last_progress = [-1]
            def record_progress(percent):
                # Avoid rewriting the local JSON fallback for every source frame.
                if percent != last_progress[0]:
                    last_progress[0] = percent
                    persist_call("persistence", update_job, job_id, {"progress_percent": percent})
            result = analyze_video(path, app.config["SAMPLE_EVERY_SECONDS"], app.config["MAX_DURATION_SECONDS"], detector,
                                   record_progress, output_path,
                                   face_detector=face_detector, object_detector=object_detector, zones=zones,
                                   zone_sample_seconds=app.config.get("ZONE_TRACK_SAMPLE_SECONDS", 0.5),
                                   zone_entry_confirmation_frames=app.config.get("ZONE_ENTRY_CONFIRMATION_FRAMES", 2),
                                   zone_exit_confirmation_frames=app.config.get("ZONE_EXIT_CONFIRMATION_FRAMES", 2),
                                   zone_track_max_gap_seconds=app.config.get("ZONE_TRACK_MAX_GAP_SECONDS", 1.5),
                                   zone_boundary_margin_px=app.config.get("ZONE_BOUNDARY_MARGIN_PX", 8),
                                   fight_detector=fight_detector,
                                   output_max_width=app.config.get("ANNOTATED_OUTPUT_MAX_WIDTH", 640))
            if result.get("annotated_output") and output_path:
                compatibility_name = os.path.splitext(output_name)[0] + ".mp4"
                compatibility_output_path = os.path.join(app.config["PROCESSED_DIR"], compatibility_name)
                if os.path.exists(compatibility_output_path):
                    compatibility_name = video_id + "_annotated_" + uuid.uuid4().hex[:8] + ".mp4"
                    compatibility_output_path = os.path.join(app.config["PROCESSED_DIR"], compatibility_name)
                encode_started = time.perf_counter()
                mp4_metadata = make_browser_playable_mp4(output_path, compatibility_output_path)
                result.setdefault("stage_timings", {})["browser_h264_transcode_seconds"] = time.perf_counter() - encode_started
                if mp4_metadata:
                    output_name = compatibility_name
                    result["annotated_output_metadata"] = mp4_metadata
                    result["annotated_playback_note"] = "Validated H.264/MP4 playback copy created; the original annotated WebM was preserved."
                else:
                    compatibility_output_path = None
                    result["annotated_playback_note"] = "H.264/MP4 playback copy could not be created; the original annotated WebM remains available."
            if not result.get("annotated_output") and output_path and os.path.isfile(output_path):
                os.remove(output_path)
            if result["events"]:
                from app.services.repository import save_events, initialize_alert_delivery
                stage_started = time.perf_counter()
                saved_count = persist_call("persistence", save_events, video_id, job_id, result["events"])
                worker_profile["event_persistence_seconds"] = time.perf_counter() - stage_started
                if saved_count != len(result["events"]):
                    raise RuntimeError("Not all analysis events were confirmed persisted; alert delivery was not queued.")
                try:
                    persist_call("persistence", initialize_alert_delivery, video_id, result["events"])
                except Exception as alert_error:
                    # The event and an initial pending/disabled alert record were
                    # saved atomically. A later outage must not turn completed
                    # video analysis into a failed job; startup can resume it.
                    log.error("Could not finalize alert queue for video %s (%s); traceback follows:\n%s",
                              video_id[:8], type(alert_error).__name__,
                              sanitized_traceback(alert_error, app.config))
            if result.get("face_frames"):
                stage_started = time.perf_counter()
                persist_call("persistence", save_face_frames, video_id, result["face_frames"])
                worker_profile["face_frame_persistence_seconds"] = time.perf_counter() - stage_started
            if result.get("pose_frames"):
                stage_started = time.perf_counter()
                persist_call("persistence", save_pose_frames, video_id, result["pose_frames"])
                worker_profile["pose_frame_persistence_seconds"] = time.perf_counter() - stage_started
            face_result = result.get("face_analysis", {})
            if face_init_error:
                face_result.update({"status": "unavailable", "error_message": face_init_error})
            object_result = result.get("object_analysis", {})
            if object_init_error:
                object_result.update({"status": "unavailable", "error_message": object_init_error})
            elif not object_model_path:
                object_result.update({"status": "not_configured", "error_message": None})
            object_result["model_name"] = os.path.basename(object_model_path) if object_model_path else None
            object_result["method"] = YoloObjectTracker.METHOD
            object_result["package_version"] = getattr(object_detector, "package_version", None)
            object_result["confidence_threshold"] = (object_result.get("confidence_threshold")
                                                       if object_result.get("confidence_threshold") is not None
                                                       else (app.config.get("OBJECT_MODEL_CONFIDENCE", 0.25)
                                                             if object_detector else None))
            result["object_analysis"] = object_result
            pose_result = result.get("pose_analysis", {})
            if pose_init_error:
                pose_result.update({"status": "unavailable", "error_message": pose_init_error,
                                    "model_name": os.path.basename(model_path) if model_path else None})
            elif not model_path:
                pose_result.update({"status": "not_configured", "error_message": None})
            result["pose_analysis"] = pose_result
            fight_result = result.get("fight_analysis", {})
            if fight_init_error:
                fight_result.update({"status": "unavailable", "error_message": fight_init_error,
                                     "candidate_event_count": 0})
            elif not fight_enabled:
                fight_result.update({"status": "disabled", "candidate_event_count": 0})
            elif not fight_model_path:
                fight_result.update({"status": "not_configured", "candidate_event_count": 0,
                    "error_message": "No temporal fighting model is configured; no fighting prediction was produced."})
            result["fight_analysis"] = fight_result
            active_zones = [zone for zone in zones if zone.get("enabled") is True]
            zone_events_inside = sum(event.get("zone_association_status") == "inside_active_zone"
                                     for event in result["events"] if event.get("event_type") == "potential_fall")
            restricted_entries = sum(event.get("event_type") == "RESTRICTED_ZONE_ENTRY"
                                     for event in result["events"])
            if not zones:
                zone_result = {"status": "not_configured", "zone_count": 0, "active_zone_count": 0,
                               "events_inside_active_zones": 0}
            elif not active_zones:
                zone_result = {"status": "disabled", "zone_count": len(zones), "active_zone_count": 0,
                               "events_inside_active_zones": 0}
            elif pose_result.get("status") == "complete":
                zone_result = {"status": "complete", "zone_count": len(zones),
                               "active_zone_count": len(active_zones),
                               "events_inside_active_zones": zone_events_inside,
                               "confirmed_entry_events": restricted_entries,
                               "entry_detection_status": "complete" if pose_result.get("status") == "complete" else "unavailable",
                               "entry_confirmation_frames": app.config.get("ZONE_ENTRY_CONFIRMATION_FRAMES", 2),
                               "tracking_sample_seconds": result.get("sample_interval_seconds"),
                               "spatial_rule": "bottom-center of the selected pose person's bounding box"}
            else:
                zone_result = {"status": "unavailable", "zone_count": len(zones),
                               "active_zone_count": len(active_zones), "events_inside_active_zones": 0,
                               "error_message": "Zone polygons were saved, but no configured pose model produced localized anomaly events for this run."}
            anomaly_events = [event for event in result["events"]
                              if event.get("event_type") not in {"object_detection", "zone_presence_observed", "zone_entry", "zone_exit", "zone_tracking_gap", "RESTRICTED_ZONE_ENTRY"}]
            object_events = [event for event in result["events"]
                             if event.get("event_type") == "object_detection"]
            result_profile = dict(result.get("stage_timings") or {})
            result_profile.update(worker_profile)
            stage_started = time.perf_counter()
            video_saved = persist_call("persistence", update_video, video_id, {"processing_status": "complete", "duration": result.get("duration"),
                "frame_rate": result.get("fps"), "width": result.get("width"), "height": result.get("height"),
                "processing_duration": time.monotonic()-started, "error_message": None,
                "analysis_mode": result.get("analysis_mode"), "detection_method": result.get("method"),
                "event_count": len(anomaly_events), "object_detection_count": len(object_events),
                "face_analysis": face_result,
                "face_analysis_status": face_result.get("status", "unavailable"),
                "pose_analysis": pose_result,
                "object_analysis": object_result,
                "zone_analysis": zone_result,
                "analysis_settings": {"sample_every_seconds": result.get("sample_interval_seconds", app.config["SAMPLE_EVERY_SECONDS"]),
                                      "zone_track_sample_seconds": app.config.get("ZONE_TRACK_SAMPLE_SECONDS", 0.5),
                                      "zone_entry_confirmation_frames": app.config.get("ZONE_ENTRY_CONFIRMATION_FRAMES", 2),
                                      "zone_exit_confirmation_frames": app.config.get("ZONE_EXIT_CONFIRMATION_FRAMES", 2),
                                      "zone_track_max_gap_seconds": app.config.get("ZONE_TRACK_MAX_GAP_SECONDS", 1.5),
                                      "zone_boundary_margin_px": app.config.get("ZONE_BOUNDARY_MARGIN_PX", 8),
                                      "max_duration_seconds": app.config["MAX_DURATION_SECONDS"],
                                      "fall_transition_window_seconds": app.config.get("FALL_TRANSITION_WINDOW_SECONDS", 1.5),
                                      "fall_min_vertical_drop": app.config.get("FALL_MIN_VERTICAL_DROP", .12),
                                      "fall_min_horizontal_persistence": app.config.get("FALL_MIN_HORIZONTAL_PERSISTENCE", .6),
                                      "fall_candidate_score_threshold": app.config.get("FALL_CANDIDATE_SCORE_THRESHOLD", .55),
                                      "pose_tracker": app.config.get("POSE_TRACKER", "bytetrack.yaml"),
                                      "object_confidence": app.config.get("OBJECT_MODEL_CONFIDENCE", 0.25),
                                      "object_tracker": app.config.get("OBJECT_TRACKER", "bytetrack.yaml"),
                                      "fight_score_threshold": app.config.get("FIGHT_SCORE_THRESHOLD", 0.4),
                                      "fight_window_seconds": app.config.get("FIGHT_WINDOW_SECONDS", 4.0),
                                      "fight_window_stride": app.config.get("FIGHT_WINDOW_STRIDE", 1.0),
                                      "fight_confirmation_windows": app.config.get("FIGHT_CONFIRMATION_WINDOWS", 2)},
                "fight_analysis": fight_result,
                "analysis_summary": build_analysis_summary(result, result.get("sample_interval_seconds", app.config["SAMPLE_EVERY_SECONDS"])),
                "analysis_stage_timings": result_profile,
                "processed_filename": output_name if result.get("annotated_output") else None,
                "annotated_output_status": ("ready" if result.get("annotated_output") else
                                             "unavailable" if result.get("annotated_output_error") else "not_requested"),
                "annotated_output_metadata": result.get("annotated_output_metadata"),
                "annotated_playback_note": result.get("annotated_playback_note"),
                "annotated_output_error": result.get("annotated_output_error")})
            worker_profile["final_video_persistence_seconds"] = time.perf_counter() - stage_started
            if not video_saved:
                raise RuntimeError("Final analysis results were not confirmed persisted to the video record.")
            # Publishing the video-level complete marker follows all results,
            # frames and metadata writes. The video record is the status
            # authority used by Results, History and Dashboard.
            stage_started = time.perf_counter()
            job_saved = persist_call("persistence", update_job, job_id, {"status": "completed", "progress_percent": 100,
                                "completed_at": datetime.now(timezone.utc)})
            worker_profile["final_job_persistence_seconds"] = time.perf_counter() - stage_started
            if not job_saved:
                raise RuntimeError("The completed analysis job state was not confirmed persisted.")
            profile_text = ", ".join(f"{key}={value:.3f}s" for key, value in sorted(worker_profile.items()))
            log.info("Video analysis profile video=%s total=%.3fs stages={%s}",
                     video_id[:8], time.monotonic() - started, profile_text)
            # Dispatch only after incident and video metadata writes have
            # succeeded. SMTP work runs on a separate executor and cannot block
            # this video-processing worker.
            from app.services.notifier import schedule_pending_alerts
            try:
                stage_started = time.perf_counter()
                schedule_pending_alerts(app, video_id)
                worker_profile["alert_queue_seconds"] = time.perf_counter() - stage_started
                log.info("Email alert queue profile video=%s queue_seconds=%.6f; GIF/SMTP work stays on alert worker.",
                         video_id[:8], worker_profile["alert_queue_seconds"])
            except Exception as alert_error:
                log.error("Could not start email delivery for video %s (%s); saved alert remains available for retry:\n%s",
                          video_id[:8], type(alert_error).__name__,
                          sanitized_traceback(alert_error, app.config))
    except (VideoError, DetectorUnavailable) as exc:
        _fail_analysis(app, video_id, job_id, started, str(exc))
    except Exception as exc:
        log.error("Video worker failed for record %s (%s); traceback follows:\n%s",
                  video_id[:8], type(exc).__name__, sanitized_traceback(exc, app.config))
        _fail_analysis(app, video_id, job_id, started, "The video could not be processed. Check the video codec and model configuration.")
    finally:
        if ((output_path and os.path.isfile(output_path)) or
                (compatibility_output_path and os.path.isfile(compatibility_output_path))):
            with app.app_context():
                row = get_video(video_id)
                if not row or row.get("processing_status") != "complete":
                    for generated_path in (output_path, compatibility_output_path):
                        if (generated_path and os.path.isfile(generated_path) and
                                (not row or os.path.basename(generated_path) != row.get("processed_filename"))):
                            try:
                                os.remove(generated_path)
                            except OSError:
                                log.warning("Could not remove incomplete processed video for record %s.", video_id[:8])
        app.extensions["processing_slots"].release()


def _fail_analysis(app, video_id, job_id, started, message):
    with app.app_context():
        now = datetime.now(timezone.utc)
        current = get_video(video_id) or {}
        fighting = current.get("fight_analysis") or {}
        changes = {"processing_status": "failed", "processing_duration": time.monotonic()-started,
                   "error_message": message,
                   "face_analysis": {"status": "failed", "method": OpenCVFaceDetector.METHOD,
                                     "error_message": "Face analysis did not complete because the video job failed."}}
        if fighting.get("status") in {"queued", "processing", "ready"}:
            fighting = {**fighting, "status": "failed",
                        "error_message": "Fighting analysis did not complete because the video job failed."}
            changes["fight_analysis"] = fighting
        update_video(video_id, changes)
        update_job(job_id, {"status": "failed", "error_message": message, "completed_at": now})


@bp.get("/results/<video_id>")
def results(video_id):
    if ObjectId.is_valid(video_id):
        from app.services.repository import recover_video_alerts
        try:
            recovered = recover_video_alerts(
                current_app.config.get("ALERT_SENDING_STALE_SECONDS", 900), video_id)
            if recovered:
                # Recovery may return only previously reserved, definitely
                # pre-SMTP attempts to pending. The atomic worker claim keeps
                # page refreshes from creating or duplicating a digest.
                from app.services.notifier import schedule_pending_alerts
                schedule_pending_alerts(current_app._get_current_object(), video_id)
        except Exception as exc:
            log.warning("Could not reconcile stale email status for video %s (%s).",
                        video_id[:8], type(exc).__name__)
    video = get_video(video_id)
    if not video:
        abort(404)
    events = get_events(video_id, limit=2000)
    video["id"] = str(video["_id"])
    video["original_name"] = video.get("original_filename")
    # Older MongoDB records may predate upload_date, or contain a serialized
    # timestamp. Keep their results page usable instead of calling strftime
    # on a missing/non-datetime value.
    uploaded_at = video.get("upload_date")
    if isinstance(uploaded_at, datetime):
        if uploaded_at.tzinfo is None:
            uploaded_at = uploaded_at.replace(tzinfo=timezone.utc)
        video["uploaded_at"] = uploaded_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    elif isinstance(uploaded_at, str):
        try:
            parsed_upload_date = datetime.fromisoformat(uploaded_at.replace("Z", "+00:00"))
            if parsed_upload_date.tzinfo is None:
                parsed_upload_date = parsed_upload_date.replace(tzinfo=timezone.utc)
            video["uploaded_at"] = parsed_upload_date.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        except ValueError:
            video["uploaded_at"] = "Unknown"
    else:
        video["uploaded_at"] = "Unknown"
    video["status"] = video.get("processing_status") or "unknown"
    video["local_only"] = bool(video.get("local_only"))
    video["database_message"] = video.get("database_message")
    video["fps"] = video.get("frame_rate")
    video["processing_seconds"] = video.get("processing_duration")
    video["error"] = video.get("error_message")
    video["analysis_mode"] = video.get("analysis_mode", "pending")
    for notification in video.get("email_alert_notifications", []):
        notification["episode_time_labels"] = []
        for episode in notification.get("episodes", []):
            intervals = episode.get("intervals") or []
            if intervals:
                label = ", ".join(
                    f"{format_video_timestamp(bounds[0])}–{format_video_timestamp(bounds[1])}"
                    for bounds in intervals if isinstance(bounds, (list, tuple)) and len(bounds) == 2)
            else:
                label = format_video_timestamp(episode.get("timestamp_seconds")) or "Time unavailable"
            notification["episode_time_labels"].append({**episode, "time_label": label})
    video["has_processed_video"] = bool(safe_processed_path(video, current_app.config["PROCESSED_DIR"]))
    video["face_analysis"] = video.get("face_analysis") or {"status": "not_run", "face_observations": 0}
    video["pose_analysis"] = video.get("pose_analysis") or {"status": "not_configured", "frames_sampled": 0}
    video["object_analysis"] = video.get("object_analysis") or {"status": "not_configured", "counts_by_class": {}}
    video["fight_analysis"] = video.get("fight_analysis") or {
        "status": "not_configured", "candidate_event_count": 0,
        "error_message": "No temporal fighting model was configured for this run."}
    video["zones"] = video.get("zones") or []
    video["zone_analysis"] = video.get("zone_analysis") or {
        "status": "not_configured", "zone_count": len(video["zones"]),
        "active_zone_count": sum(zone.get("enabled") is True for zone in video["zones"])}
    zone_names = {zone.get("zone_id"): zone.get("name", zone.get("zone_id")) for zone in video["zones"]}
    face_frames = get_face_frames(video_id)
    pose_frames = get_pose_frames(video_id)
    for face_frame in face_frames:
        face_frame["timestamp"] = face_frame.get("timestamp_seconds", 0)
    for event in events:
        event["category"] = ("Suspected Violence-Related Activity"
                             if event.get("event_type") == "SUSPECTED_FIGHTING"
                             else event.get("event_type", "Uncategorized"))
        timestamp = event_video_seconds(event)
        event["timestamp_seconds"] = timestamp
        event["timestamp"] = timestamp
        event["video_timestamp"] = (format_video_timestamp(timestamp) if timestamp is not None
                                     else event.get("video_timestamp"))
        event["timestamp_label"] = (f"{event['video_timestamp']} · {timestamp:.2f}s"
                                     if timestamp is not None and event.get("video_timestamp")
                                     else "Time unavailable")
        event["initial_occupancy"] = event.get("event_type") == "zone_presence_observed"
        event["track_segment"] = event.get("track_segment_id") or (event.get("details") or {}).get("track_segment_id")
        if event["initial_occupancy"]:
            event["zone_time_label"] = (f"Initial occupancy — entry time not observed (observed at {event['video_timestamp']})"
                                         if event.get("video_timestamp") else
                                         "Initial occupancy — entry time not observed")
        elif event.get("event_type") == "RESTRICTED_ZONE_ENTRY":
            event["zone_time_label"] = (f"Entry time {event['video_timestamp']}" if event.get("video_timestamp")
                                         else "Entry time unavailable in this legacy record")
        elif event.get("event_type") == "zone_tracking_gap":
            details = event.get("details") or {}
            prior = details.get("last_observed_timestamp_seconds")
            event["zone_time_label"] = (f"Tracking gap observed at {event['video_timestamp']} (last seen at {format_video_timestamp(prior)}; no exit inferred)"
                                         if event.get("video_timestamp") and isinstance(prior, (int, float))
                                         else "Tracking continuity gap; no exit inferred")
        for bound in ("start_timestamp_seconds", "end_timestamp_seconds"):
            try:
                point = float(event.get(bound))
                formatted = format_video_timestamp(point) if math.isfinite(point) and point >= 0 else None
                event[bound + "_label"] = (f"{formatted} ({point:.2f}s)" if formatted else "time unavailable")
            except (TypeError, ValueError):
                event[bound + "_label"] = "time unavailable"
        start_seconds = event.get("start_timestamp_seconds")
        end_seconds = event.get("end_timestamp_seconds")
        try:
            start_seconds, end_seconds = float(start_seconds), float(end_seconds)
            event["time_range_label"] = (f"{format_video_timestamp(start_seconds)}–{format_video_timestamp(end_seconds)}"
                                          if math.isfinite(start_seconds) and math.isfinite(end_seconds)
                                          and 0 <= start_seconds <= end_seconds else None)
        except (TypeError, ValueError):
            event["time_range_label"] = None
        event["score"] = event.get("model_score", event.get("anomaly_score", event.get("confidence")))
        try:
            numeric_score = float(event["score"])
            event["score_label"] = f"{numeric_score:.3f}" if math.isfinite(numeric_score) else None
        except (TypeError, ValueError):
            event["score_label"] = str(event["score"]) if event["score"] is not None else None
        event["method"] = event.get("detection_method", "Unknown")
        event["label"] = event.get("label", "")
        event["track"] = event.get("track_id")
        event["frame"] = event.get("frame_number")
        event["details"] = event.get("details") or event.get("metadata") or {}
        event["zone_status_label"] = {
            "inside_active_zone": "Inside active zone",
            "outside_active_zones": "Outside active zones",
            "unavailable_no_person_location": "Zone check unavailable: no person location",
            "unavailable": "Zone association unavailable",
        }.get(event.get("zone_association_status"), "No zone-specific check")
        event["zone_status_class"] = {
            "inside_active_zone": "zone-event-inside",
            "outside_active_zones": "zone-event-outside",
            "unavailable_no_person_location": "zone-event-unavailable",
        }.get(event.get("zone_association_status"), "")
        event["zone_names"] = [zone_names.get(zone_id, zone_id) for zone_id in event.get("zone_matches", [])]
        event["id"] = str(event.get("_id", ""))
        event["explanation"] = explain_event(event)
    object_observations = [event for event in events if event.get("event_type") == "object_detection"]
    object_tracks = group_object_observations(object_observations)
    event_types = sorted({event.get("event_type", "unknown") for event in events})
    event_labels = sorted({event["label"] for event in events if event.get("label")})
    event_tracks = sorted({str(event["track_id"]) for event in events if event.get("track_id") is not None})
    return render_template("results.html", video=video, events=events, face_frames=face_frames,
                           pose_frames=pose_frames,
                           event_types=event_types, event_labels=event_labels, event_tracks=event_tracks,
                           object_observations=object_observations, object_tracks=object_tracks,
                           event_limit_reached=len(events) >= 2000)


@bp.get("/api/videos/<video_id>/alerts")
def alert_status(video_id):
    """Return saved alert status; only stale, already-reserved safe retries may resume."""
    if not ObjectId.is_valid(video_id):
        abort(404)
    from app.services.notifier import smtp_configuration
    from app.services.repository import recover_video_alerts
    try:
        recovered = recover_video_alerts(
            current_app.config.get("ALERT_SENDING_STALE_SECONDS", 900), video_id)
        if recovered:
            from app.services.notifier import schedule_pending_alerts
            schedule_pending_alerts(current_app._get_current_object(), video_id)
    except Exception as exc:
        log.warning("Could not reconcile email status for video %s (%s).", video_id[:8], type(exc).__name__)
    try:
        video = get_video(video_id)
    except DatabaseUnavailable as exc:
        return jsonify({"error": str(exc), "delivery_status": "unavailable",
                        "notifications": []}), 503
    if not video:
        abort(404)
    notifications = []
    for item in video.get("email_alert_notifications", []):
        episodes = item.get("episodes") or []
        episode_summaries = []
        for episode in episodes:
            intervals = episode.get("intervals") or []
            interval_labels = []
            for bounds in intervals:
                if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                    try:
                        interval_labels.append(f"{format_video_timestamp(float(bounds[0]))}–{format_video_timestamp(float(bounds[1]))}")
                    except (TypeError, ValueError):
                        continue
            timestamp = episode.get("timestamp_seconds")
            label = ", ".join(interval_labels) if interval_labels else format_video_timestamp(timestamp)
            episode_summaries.append({"event_type": str(episode.get("event_type", "candidate")),
                                      "time_label": label or "Time unavailable"})
        notifications.append({
            # Keep the historical database value compatible while presenting
            # the user-facing state as Skipped.
            "status": "skipped" if item.get("status") == "disabled" else item.get("status", "pending"),
            "sent_at": item.get("sent_at").isoformat() if hasattr(item.get("sent_at"), "isoformat") else item.get("sent_at"),
            "last_attempt_at": item.get("last_attempt_at").isoformat() if hasattr(item.get("last_attempt_at"), "isoformat") else item.get("last_attempt_at"),
            "failure_reason": item.get("failure_reason"),
            "gif_available": bool(item.get("gif_filename")),
            "categories": sorted({str(ep.get("event_type", "candidate")) for ep in episodes}),
            "episode_summaries": episode_summaries,
            "episode_count": len(episodes),
        })
    sent_count = sum(item.get("status") == "sent" for item in notifications)
    ready, readiness_message = smtp_configuration(current_app.config)
    delivery_status = (notifications[-1]["status"] if notifications else
                       ("pending" if ready else "skipped"))
    return jsonify({"notifications": notifications,
                    "reserved_count": len(notifications),
                    "sent_count": sent_count,
                    "maximum": 2,
                    "limit_reached": bool(video.get("alert_email_limit_reached")),
                    "delivery_status": delivery_status,
                    "status_message": readiness_message if not ready else None})


@bp.get("/results/<video_id>/report.csv")
def video_report(video_id):
    video = get_video(video_id)
    if not video:
        abort(404)
    events = get_events(video_id, limit=50000)
    content = video_report_csv(video, events)
    return Response(content, mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=analysis-{video_id[:8]}.csv"})


@bp.get("/results/<video_id>/report.pdf")
def video_report_pdf(video_id):
    video = get_video(video_id)
    if not video:
        abort(404)
    events = get_events(video_id, limit=2000)
    job = get_job_for_video(video_id) or {}
    content = build_pdf(video, job, events)
    return Response(content, mimetype="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename=analysis-{video_id[:8]}.pdf"})


@bp.get("/api/videos/<video_id>/status")
def status(video_id):
    try:
        video = get_video(video_id)
    except DatabaseUnavailable as exc:
        # Keep transport/database outages distinct from an invalid or missing
        # video ID. The browser can show this sanitized diagnostic and retry a
        # bounded number of times without displaying a false 404.
        return jsonify({"status": "unavailable", "progress": 0, "error": str(exc)}), 503
    if not video:
        abort(404)
    job = get_job_for_video(video_id) or {}
    # The video document is the user-facing source of truth. It is committed
    # together with all final detector summaries before the job is finalized.
    # Older video documents without a status use their job record as fallback.
    status = video.get("processing_status") or job.get("status") or "queued"
    if status in {"running", "processing"}:
        status = "processing"
    elif status in {"complete", "completed"}:
        status = "complete"
    elif status not in {"queued", "failed", "interrupted", "waiting_for_mongodb"}:
        status = "queued"
    if status == "complete" and job.get("status") in {"queued", "processing", "running"}:
        # A browser may observe the video commit in the few milliseconds before
        # the worker's job commit. Safely roll the same persisted job forward;
        # never move a terminal job backwards.
        job_id = job.get("_id")
        if job_id:
            try:
                update_job(str(job_id), {"status": "completed", "progress_percent": 100,
                                         "completed_at": datetime.now(timezone.utc)})
            except Exception as exc:
                log.error("Could not reconcile completed job state for video %s (%s); traceback follows:\n%s",
                          str(video_id)[:8], type(exc).__name__,
                          sanitized_traceback(exc, current_app.config))
    progress = 100 if status == "complete" else job.get("progress_percent", 0)
    return jsonify({"status": status, "progress": progress,
                    "database_status": "local_only" if video.get("local_only") else "primary",
                    "error": video.get("error_message") or job.get("error_message"),
                    "analysis_mode": video.get("analysis_mode"),
                    "face_status": (video.get("face_analysis") or {}).get("status", "not_run"),
                    "face_error": (video.get("face_analysis") or {}).get("error_message"),
                    "pose_status": (video.get("pose_analysis") or {}).get("status", "not_configured"),
                    "pose_error": (video.get("pose_analysis") or {}).get("error_message"),
                    "object_status": (video.get("object_analysis") or {}).get("status", "not_configured"),
                    "object_error": (video.get("object_analysis") or {}).get("error_message")})


@bp.get("/media/<video_id>")
def media(video_id):
    video = get_video(video_id)
    if not video:
        abort(404)
    target = safe_video_path(video, current_app.config["UPLOAD_DIR"])
    if not target:
        abort(404)
    extension = os.path.splitext(target)[1].lower()
    mime = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
            ".avi": "video/x-msvideo"}.get(extension, "application/octet-stream")
    return send_file(target, mimetype=mime, as_attachment=False, conditional=True, etag=True)


@bp.get("/media/<video_id>/download")
def download_media(video_id):
    video = get_video(video_id)
    if not video:
        abort(404)
    target = safe_video_path(video, current_app.config["UPLOAD_DIR"])
    if not target:
        abort(404)
    return send_file(target, mimetype="application/octet-stream", as_attachment=True,
                     download_name=video.get("original_filename", "video"), conditional=True, etag=True)


def _browser_compatible_processed_path(video, target):
    """Prefer explicit H.264/MP4 metadata; safely cache a playable copy for legacy WebM."""
    extension = os.path.splitext(target)[1].lower()
    metadata = video.get("annotated_output_metadata") or {}
    if extension == ".mp4" and "h264" in str(metadata.get("codec", "")).lower().replace(".", ""):
        return target
    if extension == ".webm":
        try:
            stat = os.stat(target)
            signature = f"{os.path.basename(target)}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
            token = hashlib.sha256(signature).hexdigest()[:8]
            sidecar_name = str(video["_id"]) + "_annotated_" + token + ".mp4"
            sidecar = os.path.join(current_app.config["PROCESSED_DIR"], sidecar_name)
            if make_browser_playable_mp4(target, sidecar):
                safe_sidecar = safe_processed_path({"processed_filename": sidecar_name}, current_app.config["PROCESSED_DIR"])
                if safe_sidecar:
                    return safe_sidecar
        except (OSError, KeyError, ValueError):
            pass
        return target
    if extension == ".mp4":
        browser_path = os.path.join(current_app.config["PROCESSED_DIR"], str(video["_id"]) + "_annotated.webm")
        if make_browser_playable_webm(target, browser_path):
            return browser_path
    return target


@bp.get("/processed/<video_id>")
def processed_media(video_id):
    video = get_video(video_id)
    if not video:
        abort(404)
    target = safe_processed_path(video, current_app.config["PROCESSED_DIR"])
    if not target:
        if video.get("processing_status") in {"queued", "processing"}:
            return jsonify({"error_code": "processing", "message": "Annotated video is still processing."}), 409
        return jsonify({"error_code": "missing", "message": video.get("annotated_output_error") or
                        "The saved annotated video is unavailable."}), 404
    target = _browser_compatible_processed_path(video, target)
    extension = os.path.splitext(target)[1].lower()
    mime = "video/webm" if extension == ".webm" else "video/mp4"
    return send_file(target, mimetype=mime, as_attachment=False, conditional=True, etag=True)



@bp.get("/processed/<video_id>/download")
def download_processed_media(video_id):
    video = get_video(video_id)
    if not video:
        abort(404)
    target = safe_processed_path(video, current_app.config["PROCESSED_DIR"])
    if not target:
        abort(404)
    target = _browser_compatible_processed_path(video, target)
    extension = os.path.splitext(target)[1].lower()
    mime = "video/webm" if extension == ".webm" else "video/mp4"
    return send_file(target, mimetype=mime, as_attachment=True,
                     download_name="annotated-analysis" + extension, conditional=True, etag=True)
