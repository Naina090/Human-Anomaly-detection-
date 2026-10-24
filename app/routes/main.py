from flask import Blueprint, render_template, current_app
from datetime import datetime, timezone
from app.services.repository import dashboard_stats
from app.services.fall_detector import detector_status
from app.services.object_detector import detector_status as object_detector_status
from app.services.fight_detector import detector_status as fight_detector_status

bp = Blueprint("main", __name__)


@bp.get("/")
def home():
    return render_template("index.html")


@bp.get("/dashboard")
def dashboard():
    stats = dashboard_stats()
    for video in stats["recent_videos"]:
        video["id"] = str(video["_id"])
        uploaded_at = video.get("upload_date")
        if isinstance(uploaded_at, datetime):
            if uploaded_at.tzinfo is None:
                uploaded_at = uploaded_at.replace(tzinfo=timezone.utc)
            video["uploaded_display"] = uploaded_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        else:
            video["uploaded_display"] = "Unknown"
    total_events = sum(item["count"] for item in stats["categories"])
    for item in stats["categories"]:
        item["percent"] = round(100 * item["count"] / total_events) if total_events else 0
    stats["mongo_connected"] = bool(current_app.extensions.get("mongo_available"))
    stats["detector"] = detector_status(current_app.config.get("POSE_MODEL_PATH", ""))
    stats["object_detector"] = object_detector_status(current_app.config.get("OBJECT_MODEL_PATH", ""),
                                                        current_app.config.get("OBJECT_TRACKER", "bytetrack.yaml"))
    stats["fight_detector"] = fight_detector_status(current_app.config.get("FIGHT_MODEL_PATH", ""),
        current_app.config.get("FIGHT_DETECTION_ENABLED", True))
    for event in stats["recent"]:
        event["id"] = str(event["_id"])
        event["timestamp"] = event.get("timestamp_seconds", 0)
        event["category"] = event.get("event_type", "Uncategorized")
        event["method"] = event.get("detection_method", "Unknown")
    return render_template("dashboard.html", **stats)


@bp.get("/about")
def about():
    return render_template("about.html")
