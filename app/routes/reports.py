import csv
import io
from datetime import datetime
from flask import Blueprint, render_template, request, Response, abort
from app.services.repository import report_data
from app.services.timestamps import event_video_seconds, format_video_timestamp
from app.services.report_service import explain_event, group_event_episodes

bp = Blueprint("reports", __name__)


def validated_date(value):
    if not value:
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return value
    except ValueError:
        abort(400)


@bp.get("/reports")
def reports():
    start, end = validated_date(request.args.get("start", "")), validated_date(request.args.get("end", ""))
    if start and end and start > end:
        abort(400)
    videos, events = report_data(start, end)
    for video in videos:
        video["id"] = str(video["_id"])
        video["status"] = video.get("processing_status")
    for event in events:
        event["video_id"] = str(event["video_id"])
        event["category"] = event.get("event_type", "Uncategorized")
        event["timestamp"] = event_video_seconds(event)
        event["video_timestamp"] = format_video_timestamp(event["timestamp"])
        event["score"] = event.get("model_score", event.get("anomaly_score", event.get("confidence")))
        event["method"] = event.get("detection_method", "Unknown")
        event["explanation"] = explain_event(event)
        event["zone_name"] = (event.get("zone_name") or (event.get("details") or {}).get("zone_name") or
                              (event.get("details") or {}).get("zone_id"))
        try:
            event_start = float(event.get("start_timestamp_seconds"))
            event_end = float(event.get("end_timestamp_seconds"))
            event["time_range_label"] = (f"{format_video_timestamp(event_start)}-{format_video_timestamp(event_end)}"
                                          if event_start >= 0 and event_end > event_start else None)
        except (TypeError, ValueError):
            event["time_range_label"] = None
    episodes = group_event_episodes(events)
    event_groups = {str(event.get("_id")): episode["group_id"] for episode in episodes
                    for event in episode.get("events", [])}
    for event in events:
        event["episode_group_id"] = event_groups.get(str(event.get("_id")))
    categories = {}
    for episode in episodes:
        categories[episode["event_type"]] = categories.get(episode["event_type"], 0) + 1
    return render_template("reports.html", videos=videos, events=events, start=start, end=end,
                           anomaly_episode_count=len(episodes), episode_categories=categories)


@bp.get("/reports.csv")
def csv_export():
    start, end = validated_date(request.args.get("start", "")), validated_date(request.args.get("end", ""))
    if start and end and start > end:
        abort(400)
    videos, events = report_data(start, end)
    out = io.StringIO(); writer = csv.writer(out)
    writer.writerow(["video_id", "filename", "uploaded_at", "status", "event_category", "label",
                     "timestamp_seconds", "video_timestamp", "start_timestamp_seconds", "end_timestamp_seconds",
                     "event_interval", "episode_group_id", "zone_name", "frame_number", "confidence",
                     "anomaly_score", "model_score", "track_id", "incident_status", "review_notes",
                     "method", "explanation", "details"])
    episodes = group_event_episodes(events)
    event_groups = {str(event.get("_id")): episode["group_id"] for episode in episodes
                    for event in episode.get("events", [])}
    names = {str(v["_id"]): (v.get("original_filename", ""), v.get("upload_date", ""), v.get("processing_status", "")) for v in videos}
    for e in events:
        event_video_id = str(e["video_id"])
        filename, uploaded, status = names.get(event_video_id, (e.get("original_filename", ""), "", ""))
        start_seconds, end_seconds = e.get("start_timestamp_seconds"), e.get("end_timestamp_seconds")
        try:
            interval = (f"{format_video_timestamp(float(start_seconds))}-{format_video_timestamp(float(end_seconds))}"
                       if float(start_seconds) >= 0 and float(end_seconds) > float(start_seconds) else "")
        except (TypeError, ValueError):
            interval = ""
        writer.writerow([event_video_id, filename, uploaded, status, e.get("event_type", ""),
                         e.get("label", ""), event_video_seconds(e) if event_video_seconds(e) is not None else "",
                         format_video_timestamp(event_video_seconds(e)) or "",
                         start_seconds if start_seconds is not None else "",
                         end_seconds if end_seconds is not None else "", interval, event_groups.get(str(e.get("_id")), ""),
                         (e.get("zone_name") or (e.get("details") or {}).get("zone_name") or (e.get("details") or {}).get("zone_id") or ""),
                         e.get("frame_number", ""),
                         e.get("confidence", ""), e.get("anomaly_score", ""), e.get("model_score", (e.get("details") or {}).get("model_score", "")), e.get("track_id", ""),
                         e.get("incident_status", "not reviewed"), e.get("review_notes", ""),
                         e.get("detection_method", ""), explain_event(e), repr(e.get("details", e.get("metadata", "")))])
    return Response(out.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=anomaly-report.csv"})
