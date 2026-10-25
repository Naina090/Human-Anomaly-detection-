from datetime import datetime, time as day_time, timedelta, timezone

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for

from app.db import DatabaseUnavailable
from app.services.repository import get_video, list_incidents, object_id, update_incident
from app.services.timestamps import event_video_seconds, format_video_timestamp

bp = Blueprint("incidents", __name__)
INCIDENT_STATUSES = {"new", "reviewing", "resolved", "false_positive"}


def _local_request_only():
    if request.remote_addr not in {"127.0.0.1", "::1", "::ffff:127.0.0.1"}:
        abort(403)


def _day(value, end=False):
    if not value:
        return None
    try:
        date_value = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        abort(400)
    if end:
        date_value += timedelta(days=1)
    return datetime.combine(date_value, day_time.min, tzinfo=timezone.utc)


@bp.get("/incidents")
def incident_list():
    _local_request_only()
    status = request.args.get("status", "")
    if status and status not in INCIDENT_STATUSES:
        abort(400)
    start, end = _day(request.args.get("start")), _day(request.args.get("end"), end=True)
    if start and end and start >= end:
        abort(400)
    rows = list_incidents(status, start, end)
    for event in rows:
        event.setdefault("incident_status", "new")
        event.setdefault("review_notes", "")
        event["id"] = str(event["_id"])
        event["video_id_text"] = str(event.get("video_id", ""))
        video = get_video(event["video_id_text"])
        event["filename"] = (video or {}).get("original_filename", "Video record unavailable")
        notification = next((item for item in (video or {}).get("email_alert_notifications", [])
                             if event["id"] in {str(value) for value in item.get("event_ids", [])}), None)
        event["email_delivery_status"] = ((notification or {}).get("status") or
            (event.get("alert_delivery") or {}).get("status", "not queued"))
        event["email_delivery_failure"] = ((notification or {}).get("failure_reason") or
            (event.get("alert_delivery") or {}).get("failure_reason"))
        seconds = event_video_seconds(event)
        event["video_timestamp_label"] = format_video_timestamp(seconds) if seconds is not None else None
    return render_template("incidents.html", incidents=rows, status=status,
                           start=request.args.get("start", ""), end=request.args.get("end", ""))


@bp.post("/incidents/<event_id>/review")
def review(event_id):
    _local_request_only()
    if object_id(event_id) is None:
        abort(404)
    status = request.form.get("status", "")
    if status not in INCIDENT_STATUSES:
        abort(400)
    next_url = request.form.get("next", "")
    if not next_url.startswith("/") or next_url.startswith("//"):
        next_url = url_for("incidents.incident_list")
    try:
        saved = update_incident(event_id, status, request.form.get("review_notes", ""))
    except DatabaseUnavailable:
        flash("Review was not saved because the database and local fallback are unavailable. Please retry.", "error")
        return redirect(next_url)
    if not saved:
        abort(404)
    flash("Incident review saved.", "success")
    return redirect(next_url)
