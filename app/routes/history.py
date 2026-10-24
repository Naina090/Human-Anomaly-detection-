import os
import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from bson import ObjectId
from flask import Blueprint, current_app, render_template, request, redirect, url_for, flash, abort
from app.services.repository import (list_videos, delete_video, object_id, video_is_processing,
    get_video, get_events, queue_video_retry, create_job, update_video, update_job, get_job_for_video)
from app.services.storage import safe_video_path, safe_processed_path
from app.services.timestamps import event_video_seconds, format_video_timestamp

bp = Blueprint("history", __name__)


def _format_upload_date(value):
    """Render UTC-written and legacy upload dates in India Standard Time."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return "Unknown"
    else:
        return "Unknown"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(ZoneInfo("Asia/Kolkata")).strftime("%d %b %Y, %I:%M %p IST")


@bp.get("/history")
def history():
    q = request.args.get("q", "").strip()[:100]
    status = request.args.get("status", "")
    try:
        page = max(1, min(1000, int(request.args.get("page", "1"))))
    except ValueError:
        page = 1
    videos, total = list_videos(q, status, page)
    for video in videos:
        video["id"] = str(video["_id"])
        video["original_name"] = video.get("original_filename")
        video["uploaded_at"] = _format_upload_date(video.get("upload_date"))
        video["status"] = video.get("processing_status")
        video["status_label"] = {
            "queued": "Pending", "waiting_for_mongodb": "Pending · Local fallback",
            "processing": "Processing", "running": "Processing",
            "complete": "Complete", "completed": "Complete",
            "failed": "Failed", "interrupted": "Interrupted",
        }.get(video["status"], "Pending" if not video["status"] else str(video["status"]).replace("_", " ").title())
        video["file_size"] = video.get("file_size", 0)
        video["face_analysis"] = video.get("face_analysis") or {"status": "not_configured"}
        video["pose_analysis"] = video.get("pose_analysis") or {"status": "not_configured"}
        video["object_analysis"] = video.get("object_analysis") or {"status": "not_configured"}
        video["fight_analysis"] = video.get("fight_analysis") or {"status": "not_configured"}
        video["suspected_fighting_count"] = video.get("suspected_fighting_count", 0)
        entry = video.get("latest_restricted_zone_entry")
        seconds = event_video_seconds(entry)
        video["latest_zone_entry_time"] = format_video_timestamp(seconds) if seconds is not None else None
        video["latest_zone_entry_name"] = ((entry or {}).get("zone_name") or
            ((entry or {}).get("details") or {}).get("zone_name"))
        history_events = get_events(video["id"], limit=100)
        for event in history_events:
            event["history_time"] = format_video_timestamp(event_video_seconds(event))
            event["history_end_time"] = format_video_timestamp(
                event.get("end_timestamp_seconds", event.get("end_video_position_seconds")))
            details = event.get("details") if isinstance(event.get("details"), dict) else {}
            event["history_zone_name"] = (event.get("zone_name") or details.get("zone_name") or
                                           details.get("zone_id"))
            score = event.get("anomaly_score")
            event["history_score_kind"] = "Anomaly score (uncalibrated)"
            if score is None:
                score = event.get("confidence")
                event["history_score_kind"] = "Detector confidence"
            try:
                score = float(score) if score is not None else None
                event["history_score_label"] = f"{score:.3f}" if score is not None and math.isfinite(score) else None
            except (TypeError, ValueError, OverflowError):
                event["history_score_label"] = None
        video["history_events"] = history_events
    return render_template("history.html", videos=videos, q=q, status=status, page=page, total=total, per_page=25)


@bp.post("/history/<video_id>/retry")
def retry(video_id):
    if object_id(video_id) is None:
        abort(404)
    video = get_video(video_id)
    if not video:
        abort(404)
    if video.get("processing_status") not in {"failed", "interrupted"}:
        flash("Only failed or interrupted videos can be retried.", "error")
        return redirect(url_for("videos.results", video_id=video_id))
    target = safe_video_path(video, current_app.config["UPLOAD_DIR"])
    if not target:
        flash("The original video file is missing or unavailable. This job cannot be retried.", "error")
        return redirect(url_for("videos.results", video_id=video_id))
    app = current_app._get_current_object()
    if not app.extensions["processing_slots"].acquire(blocking=False):
        flash("The processing queue is full. Try again shortly.", "error")
        return redirect(url_for("videos.results", video_id=video_id))
    try:
        claimed = queue_video_retry(video_id)
    except Exception:
        app.extensions["processing_slots"].release()
        raise
    if not claimed:
        app.extensions["processing_slots"].release()
        flash("This video was already claimed for retry or is no longer retryable.", "error")
        return redirect(url_for("videos.results", video_id=video_id))
    try:
        if video.get("local_only"):
            job_id = (get_job_for_video(video_id) or {}).get("_id")
            if not job_id:
                raise RuntimeError("Local retry job was not created.")
        else:
            now = datetime.now(timezone.utc)
            job_id = create_job({"video_id": ObjectId(video_id), "status": "queued", "progress_percent": 0,
                                 "started_at": None, "created_at": now, "completed_at": None,
                                 "error_message": None, "retry": True})
            update_video(video_id, {"active_job_id": job_id})
    except Exception:
        try:
            update_video(video_id, {"processing_status": "failed",
                                    "error_message": "The retry job could not be queued. Try again after checking MongoDB.",
                                    "face_analysis": {"status": "not_run", "error_message": "The retry job was not started."}})
        finally:
            app.extensions["processing_slots"].release()
        raise
    from app.routes.videos import _run_analysis
    try:
        app.extensions["processing_executor"].submit(_run_analysis, app, video_id, str(job_id), target)
    except RuntimeError:
        app.extensions["processing_slots"].release()
        message = "The local worker is shutting down. Restart the website and retry again."
        update_video(video_id, {"processing_status": "failed", "error_message": message})
        update_job(str(job_id), {"status": "failed", "error_message": message,
                                 "completed_at": datetime.now(timezone.utc)})
        flash(message, "error")
        return redirect(url_for("videos.results", video_id=video_id))
    flash("Video analysis was queued again.", "success")
    return redirect(url_for("videos.results", video_id=video_id))


@bp.post("/history/<video_id>/delete")
def delete(video_id):
    if object_id(video_id) is None:
        abort(404)
    if video_is_processing(video_id):
        flash("This video is still being processed. Delete it after the job finishes.", "error")
        return redirect(url_for("history.history"))
    row = delete_video(video_id)
    if not row:
        abort(404)
    target = safe_video_path(row, current_app.config["UPLOAD_DIR"])
    if target:
        os.remove(target)
    processed = safe_processed_path(row, current_app.config["PROCESSED_DIR"])
    if processed:
        os.remove(processed)
    flash("Video and related events deleted.", "success")
    return redirect(url_for("history.history"))
