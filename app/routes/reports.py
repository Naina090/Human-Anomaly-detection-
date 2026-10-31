"""Reports & Analytics views backed by the existing video/event repositories."""
import csv
import io
import math
from datetime import datetime, timezone
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from flask import Blueprint, abort, current_app, render_template, request, Response

from app.services.event_groups import group_event_episodes
from app.services.repository import report_data_with_source
from app.services.storage import safe_processed_path, safe_video_path
from app.services.timestamps import event_video_seconds, format_video_timestamp, utc_datetime_sort_key
from app.services.report_service import explain_event, motion_analytics_csv

bp = Blueprint("reports", __name__)
IST = ZoneInfo("Asia/Kolkata")
VIDEO_PAGE_SIZE = 15
EVENT_PAGE_SIZE = 15
EVIDENCE_PAGE_SIZE = 50
MOTION_PAGE_SIZE = 50
TABS = {"overview", "videos", "timeline", "evidence", "export"}
REVIEW_STATUSES = {"new", "not_reviewed", "reviewing", "resolved", "false_positive"}
EVENT_LABELS = {
    "potential_fall": "Potential Fall",
    "RESTRICTED_ZONE_ENTRY": "Restricted Zone Entry",
    "SUSPECTED_FIGHTING": "Suspected Violence-Related Activity",
    "object_detection": "Object Detection",
    "zone_presence_observed": "Zone Presence Observed",
    "zone_entry": "Zone Entry Observation",
    "zone_exit": "Zone Exit Observation",
    "zone_tracking_gap": "Zone Tracking Gap",
}


def validated_date(value):
    if not value:
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return value
    except ValueError:
        abort(400)


def _page(value, maximum=100000):
    try:
        return max(1, min(maximum, int(value or 1)))
    except (TypeError, ValueError):
        abort(400)


def _display_date(value):
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return "Date unavailable"
    if not isinstance(value, datetime):
        return "Date unavailable"
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(IST).strftime("%d %b %Y, %I:%M %p IST")


def _number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _label(event):
    event_type = str(event.get("event_type") or "unknown")
    return EVENT_LABELS.get(event_type, event_type.replace("_", " ").title())


def _event_status(event):
    status = str(event.get("incident_status") or "new").lower().replace(" ", "_")
    return "new" if status == "not_reviewed" else status


def _episode_status(episode):
    statuses = {_event_status(event) for event in episode.get("events", [])}
    if "reviewing" in statuses:
        return "reviewing"
    if statuses and statuses <= {"resolved"}:
        return "resolved"
    if statuses and statuses <= {"false_positive"}:
        return "false_positive"
    return "new"


def _safe_csv(value):
    """Prevent spreadsheet formula execution in untrusted exported text."""
    value = "" if value is None else str(value)
    if value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _prepare_data(start, end):
    videos, events, source = report_data_with_source(start, end)
    videos_by_id = {str(video.get("_id")): video for video in videos}
    events_by_video = {video_id: [] for video_id in videos_by_id}
    for event in events:
        video_id = str(event.get("video_id", ""))
        video = videos_by_id.get(video_id)
        if video:
            event["video_id_text"] = video_id
            event["original_filename"] = video.get("original_filename", "")
            event["uploaded_display"] = _display_date(video.get("upload_date"))
            event["video_timestamp_seconds"] = event_video_seconds(event)
            event["video_timestamp_label"] = format_video_timestamp(event["video_timestamp_seconds"])
            event["category_label"] = _label(event)
            event["score_value"] = _number(event.get("model_score", event.get("anomaly_score", event.get("confidence"))))
            event["score_type"] = ("Uncalibrated model score" if event.get("event_type") == "SUSPECTED_FIGHTING"
                                    and event.get("model_score") is not None else
                                    "Heuristic review score" if event.get("anomaly_score") is not None else
                                    "Detector confidence" if event.get("confidence") is not None else "Not recorded")
            event["explanation"] = explain_event(event)
            event["zone_name_display"] = (event.get("zone_name") or (event.get("details") or {}).get("zone_name") or
                                           (event.get("details") or {}).get("zone_id"))
            event["score_display"] = (f"{event['score_value']:.3f}" if event["score_value"] is not None else "Not recorded")
            events_by_video.setdefault(video_id, []).append(event)
    for video in videos:
        video_id = str(video.get("_id"))
        video["id"] = video_id
        video["status"] = str(video.get("processing_status") or "unknown").lower()
        video["uploaded_display"] = _display_date(video.get("upload_date"))
        video["event_rows"] = events_by_video.get(video_id, [])
        video["episodes"] = group_event_episodes(video["event_rows"])
        video["event_count"] = len(video["event_rows"])
        video["episode_count"] = len(video["episodes"])
        video["anomaly_counts"] = {}
        for episode in video["episodes"]:
            category = _label({"event_type": episode["event_type"]})
            video["anomaly_counts"][category] = video["anomaly_counts"].get(category, 0) + 1
        statuses = [_episode_status(episode) for episode in video["episodes"]]
        video["review_summary"] = {
            "new": sum(status == "new" for status in statuses),
            "reviewing": sum(status == "reviewing" for status in statuses),
            "resolved": sum(status == "resolved" for status in statuses),
            "false_positive": sum(status == "false_positive" for status in statuses),
        }
        video["original_available"] = bool(safe_video_path(video, current_app.config["UPLOAD_DIR"]))
        video["annotated_available"] = bool(safe_processed_path(video, current_app.config["PROCESSED_DIR"]))
        video["sample_interval"] = (video.get("analysis_settings") or {}).get("sample_every_seconds")
        lighting = video.get("lighting_analysis") if isinstance(video.get("lighting_analysis"), dict) else {}
        video["lighting_mode_display"] = video.get("lighting_mode") or lighting.get("lighting_mode") or "Not recorded"
        video["detected_lighting_mode"] = lighting.get("detected_mode") or "Not recorded"
        video["night_enhancement_display"] = ("Applied" if video.get("night_enhancement_applied") is True else
            "Not applied" if lighting.get("status") == "complete" else "Not recorded")
        video["lighting_warning_count"] = len(lighting.get("warnings") or [])
        video["movement_observation_count"] = len(video.get("movement_observations") or [])
    return videos, events, source


def _filtered_episodes(videos, events, video_filter="", category="", review="", search=""):
    by_video = {video["id"]: video for video in videos}
    episodes = []
    for video in videos:
        for raw in video["episodes"]:
            event_type = raw["event_type"]
            status = _episode_status(raw)
            if video_filter and video["id"] != video_filter:
                continue
            if category and event_type != category:
                continue
            if review and status != review:
                continue
            event = (raw.get("events") or [{}])[0]
            searchable = " ".join((video.get("original_filename", ""), _label(event),
                                   str(event.get("label", "")), str(raw.get("zone_id", "")),
                                   str(raw.get("track_id", "")))).casefold()
            if search and search.casefold() not in searchable:
                continue
            entry = dict(raw)
            entry.update({
                "video_id_text": video["id"], "original_filename": video.get("original_filename", ""),
                "uploaded_display": video.get("uploaded_display"), "category_label": _label(event),
                "review_status": status, "event_id": str(event.get("_id", "")),
                "score_value": _number(event.get("model_score", event.get("anomaly_score", event.get("confidence")))),
                "score_type": ("Uncalibrated model score" if event.get("event_type") == "SUSPECTED_FIGHTING"
                               and event.get("model_score") is not None else
                               "Heuristic review score" if event.get("anomaly_score") is not None else
                               "Detector confidence" if event.get("confidence") is not None else "Not recorded"),
                "explanation": explain_event(event), "zone_name_display": event.get("zone_name_display"),
                "track_id": event.get("track_id"), "event": event,
            })
            entry["start_label"] = format_video_timestamp(entry.get("start_timestamp_seconds"))
            entry["end_label"] = format_video_timestamp(entry.get("end_timestamp_seconds"))
            episodes.append(entry)
    episodes.sort(key=lambda item: (-by_video[item["video_id_text"]].get("_sort_upload", 0),
                                    item.get("start_timestamp_seconds", 0), item["event_type"], item.get("group_id", "")))
    return episodes


def _paginate(rows, page, page_size):
    pages = max(1, (len(rows) + page_size - 1) // page_size)
    page = min(page, pages)
    return rows[(page - 1) * page_size:page * page_size], page, pages


def _filtered_events(videos, events, video_filter="", category="", review="", search=""):
    video_by_id = {video["id"]: video for video in videos}
    output = []
    for event in events:
        video_id = str(event.get("video_id", ""))
        video = video_by_id.get(video_id)
        if not video:
            continue
        if video_filter and video_id != video_filter:
            continue
        event_type = str(event.get("event_type") or "unknown")
        if category and event_type != category:
            continue
        status = _event_status(event)
        if review and status != review:
            continue
        searchable = " ".join((video.get("original_filename", ""), _label(event),
                               str(event.get("label", "")), str(event.get("track_id", "")),
                               str(event.get("zone_name_display", "")), str(event.get("detection_method", "")))).casefold()
        if search and search.casefold() not in searchable:
            continue
        event["event_id"] = str(event.get("_id", ""))
        event["status_label"] = status.replace("_", " ").title()
        event["upload_sort"] = utc_datetime_sort_key(video.get("upload_date"))
        output.append(event)
    output.sort(key=lambda item: (-item.get("upload_sort", float("-inf")),
                                  event_video_seconds(item) if event_video_seconds(item) is not None else -1,
                                  item["event_id"]))
    return output


@bp.get("/reports")
def reports():
    start, end = validated_date(request.args.get("start", "")), validated_date(request.args.get("end", ""))
    if start and end and start > end:
        abort(400)
    tab = request.args.get("tab", "overview")
    if tab not in TABS:
        abort(400)
    videos, events, data_source = _prepare_data(start, end)
    for video in videos:
        video["_sort_upload"] = utc_datetime_sort_key(video.get("upload_date"))
    upload_search = request.args.get("video_q", "")[:100].strip()
    processing_status = request.args.get("processing_status", "")[:40].strip().lower()
    lighting_filter = request.args.get("lighting_mode", "").strip().upper()
    if lighting_filter and lighting_filter not in {"DAY", "LOW_LIGHT", "NIGHT_VISION", "NOT_RECORDED"}:
        abort(400)
    video_sort = request.args.get("video_sort", "newest")
    if video_sort not in {"newest", "oldest"}:
        abort(400)
    filtered_videos = [v for v in videos if (not upload_search or upload_search.casefold() in str(v.get("original_filename", "")).casefold())
                       and (not processing_status or v["status"] == processing_status)
                       and (not lighting_filter or
                            ((v.get("detected_lighting_mode") if lighting_filter == "LOW_LIGHT" else
                              v.get("lighting_mode_display")) ==
                             ("Not recorded" if lighting_filter == "NOT_RECORDED" else lighting_filter)))]
    filtered_videos.sort(key=lambda v: (v["_sort_upload"], v["id"]), reverse=video_sort == "newest")
    video_page_rows, video_page, video_pages = _paginate(filtered_videos, _page(request.args.get("video_page")), VIDEO_PAGE_SIZE)

    motion_video_filter = request.args.get("motion_video", "")[:80].strip()
    motion_joint_filter = request.args.get("motion_joint", "")[:60].strip()
    motion_observation_filter = request.args.get("motion_observation", "")[:80].strip()
    motion_track_text = request.args.get("motion_track", "")[:20].strip()
    try:
        motion_track_filter = int(motion_track_text) if motion_track_text else None
    except ValueError:
        abort(400)
    allowed_motion_joints = {"left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
        "left_knee", "right_knee", "left_hip", "right_hip", "torso_inclination", "shoulder_line",
        "hip_line", "left_upper_arm", "right_upper_arm", "left_forearm", "right_forearm",
        "left_thigh", "right_thigh", "left_lower_leg", "right_lower_leg"}
    if motion_joint_filter and motion_joint_filter not in allowed_motion_joints:
        abort(400)
    motion_rows, motion_samples, motion_measurements = [], [], []
    for video in videos:
        if motion_video_filter and video["id"] != motion_video_filter:
            continue
        if upload_search and upload_search.casefold() not in str(video.get("original_filename", "")).casefold():
            continue
        if processing_status and video["status"] != processing_status:
            continue
        if lighting_filter and ((video.get("detected_lighting_mode") if lighting_filter == "LOW_LIGHT"
                                 else video.get("lighting_mode_display")) !=
                                ("Not recorded" if lighting_filter == "NOT_RECORDED" else lighting_filter)):
            continue
        for sample in (video.get("motion_analytics") or {}).get("measurements", []) or []:
            if motion_track_filter is not None and sample.get("track_id") != motion_track_filter:
                continue
            observation = str(sample.get("movement_observation") or "")
            if motion_observation_filter and observation != motion_observation_filter:
                continue
            timestamp = _number(sample.get("timestamp_seconds"))
            if timestamp is None:
                continue
            joints = sample.get("joint_measurements") or {}
            if motion_joint_filter and motion_joint_filter not in joints:
                continue
            motion_samples.append((video, sample))
            for joint_name, value in joints.items():
                if motion_joint_filter and joint_name != motion_joint_filter:
                    continue
                motion_rows.append({"video_id": video["id"], "filename": video.get("original_filename", ""),
                    "track_id": sample.get("track_id"), "timestamp_seconds": timestamp,
                    "timestamp_label": format_video_timestamp(timestamp) or "Time unavailable",
                    "frame_number": sample.get("frame_number"), "lighting_mode": sample.get("lighting_mode"),
                    "joint": joint_name, "raw_angle": value.get("raw_angle_degrees"),
                    "smoothed_angle": value.get("smoothed_angle_degrees"),
                    "angular_velocity": value.get("angular_velocity_degrees_per_second"),
                    "speed": sample.get("movement_speed_pixels_per_second"),
                    "observation": observation or "Not classified"})
            motion_measurements.append(sample)
    motion_rows.sort(key=lambda row: (utc_datetime_sort_key(next((v.get("upload_date") for v in videos
        if v["id"] == row["video_id"]), None)), row["timestamp_seconds"], row["track_id"] or 0))
    motion_page_rows, motion_page, motion_pages = _paginate(
        motion_rows, _page(request.args.get("motion_page")), MOTION_PAGE_SIZE)
    motion_track_options = sorted({sample.get("track_id") for video in videos
        for sample in ((video.get("motion_analytics") or {}).get("measurements") or [])
        if sample.get("track_id") is not None})
    motion_joint_options = sorted({joint_name for video in videos
        for sample in ((video.get("motion_analytics") or {}).get("measurements") or [])
        for joint_name in (sample.get("joint_measurements") or {})})
    motion_observation_options = sorted({str(sample.get("movement_observation")) for video in videos
        for sample in ((video.get("motion_analytics") or {}).get("measurements") or [])
        if sample.get("movement_observation")})

    event_video = request.args.get("video_id", "")[:80]
    category_filter = request.args.get("category", "")[:80]
    review_filter = request.args.get("review", "")[:40].lower().replace(" ", "_")
    event_search = request.args.get("event_q", "")[:100].strip()
    event_sort = request.args.get("event_sort", "newest")
    if event_sort not in {"newest", "oldest"}:
        abort(400)
    if review_filter and review_filter not in REVIEW_STATUSES:
        abort(400)

    anomaly_events = [event for event in events if event.get("event_type") in {
        "potential_fall", "RESTRICTED_ZONE_ENTRY", "SUSPECTED_FIGHTING"}]
    upload_sort_by_video = {video["id"]: video["_sort_upload"] for video in videos}
    episodes = _filtered_episodes(videos, anomaly_events, event_video, category_filter, review_filter, event_search)
    if event_sort == "oldest":
        episodes.sort(key=lambda item: (upload_sort_by_video.get(item["video_id_text"], 0),
                                        item.get("start_timestamp_seconds", 0), item["event_type"], item.get("group_id", "")))
    timeline_rows, event_page, event_pages = _paginate(episodes, _page(request.args.get("event_page")), EVENT_PAGE_SIZE)
    evidence = _filtered_events(videos, events, event_video, category_filter, review_filter, event_search)
    if event_sort == "oldest":
        evidence.sort(key=lambda item: (item.get("upload_sort", float("-inf")),
                                        event_video_seconds(item) if event_video_seconds(item) is not None else -1,
                                        item["event_id"]))
    evidence_rows, evidence_page, evidence_pages = _paginate(evidence, _page(request.args.get("evidence_page")), EVIDENCE_PAGE_SIZE)

    categories = {}
    review_counts = {"new": 0, "reviewing": 0, "resolved": 0, "false_positive": 0}
    for video in videos:
        for episode in video["episodes"]:
            label = _label({"event_type": episode["event_type"]})
            categories[label] = categories.get(label, 0) + 1
            review_counts[_episode_status(episode)] += 1
    statuses = [v["status"] for v in videos]
    completed_count = sum(value in {"complete", "completed"} for value in statuses)
    failed_count = sum(value == "failed" for value in statuses)
    other_count = len(statuses) - completed_count - failed_count
    night_analysis_count = sum(video.get("lighting_mode_display") == "NIGHT_VISION" for video in videos)
    low_visibility_warning_count = sum(video.get("lighting_warning_count", 0) > 0 for video in videos)
    motion_measurement_count = sum((video.get("motion_analytics") or {}).get("measurement_count",
                                   len((video.get("motion_analytics") or {}).get("measurements") or []))
                                   for video in videos)
    raw_observation_count = len(events)
    run_statuses = sorted({value for value in statuses if value})
    for video in video_page_rows:
        video["summary_text"] = (str(video.get("analysis_summary") or "No saved analysis summary.")[:900])
        video["pose"] = video.get("pose_analysis") or {}
        video["objects"] = video.get("object_analysis") or {}
        video["fighting"] = video.get("fight_analysis") or {}
        video["zones"] = video.get("roi_zones") or video.get("zones") or []

    truncated = len(videos) >= 5000
    anomaly_episode_count = sum(categories.values())
    export_videos = [video for video in videos
                     if (not event_video or video["id"] == event_video)
                     and (not upload_search or upload_search.casefold() in str(video.get("original_filename", "")).casefold())
                     and (not processing_status or video["status"] == processing_status)
                     and (not lighting_filter or
                          ((video.get("detected_lighting_mode") if lighting_filter == "LOW_LIGHT" else
                            video.get("lighting_mode_display")) ==
                           ("Not recorded" if lighting_filter == "NOT_RECORDED" else lighting_filter)))]
    export_categories = {}
    for episode in episodes:
        export_categories[episode["category_label"]] = export_categories.get(episode["category_label"], 0) + 1
    export_events = _filtered_events(videos, events, event_video, category_filter, review_filter, event_search)
    export_completed_count = sum(video["status"] in {"complete", "completed"} for video in export_videos)
    export_failed_count = sum(video["status"] == "failed" for video in export_videos)
    args = request.args.to_dict(flat=True)
    global_preserved = [(key, value) for key, value in args.items() if key not in {"start", "end"}]
    video_keys = {"tab", "start", "end", "video_q", "processing_status", "lighting_mode", "video_sort", "video_page"}
    timeline_keys = {"tab", "start", "end", "video_id", "category", "review", "event_q", "event_sort", "event_page"}
    evidence_keys = timeline_keys | {"evidence_page", "motion_video", "motion_track", "motion_joint",
                                    "motion_observation", "motion_page"}
    video_preserved = [(key, value) for key, value in args.items() if key not in video_keys]
    timeline_preserved = [(key, value) for key, value in args.items() if key not in timeline_keys]
    evidence_preserved = [(key, value) for key, value in args.items() if key not in evidence_keys]
    cleared_args = {key: value for key, value in args.items() if key not in {"start", "end"}}
    clear_dates_url = "/reports" + ("?" + urlencode(cleared_args) if cleared_args else "")
    return render_template("reports.html", tab=tab, start=start or "", end=end or "",
        videos=videos, video_rows=video_page_rows, video_page=video_page, video_pages=video_pages,
        video_total=len(filtered_videos), video_q=upload_search, processing_status=processing_status,
        video_sort=video_sort, events=events, episodes=episodes, timeline_rows=timeline_rows,
        event_page=event_page, event_pages=event_pages, evidence_rows=evidence_rows,
        evidence_page=evidence_page, evidence_pages=evidence_pages, event_video=event_video,
        category_filter=category_filter, review_filter=review_filter, event_search=event_search,
        event_sort=event_sort, categories=categories, review_counts=review_counts,
        completed_count=completed_count, failed_count=failed_count, other_count=other_count,
        raw_observation_count=raw_observation_count, data_source=data_source,
        anomaly_episode_count=anomaly_episode_count,
        export_videos=export_videos, export_categories=export_categories,
        export_event_count=len(export_events), export_episode_count=len(episodes),
        export_completed_count=export_completed_count, export_failed_count=export_failed_count,
        video_statuses=run_statuses, page_size=VIDEO_PAGE_SIZE,
        global_preserved=global_preserved, video_preserved=video_preserved,
        timeline_preserved=timeline_preserved, evidence_preserved=evidence_preserved,
        clear_dates_url=clear_dates_url,
        truncated=truncated, night_analysis_count=night_analysis_count,
        low_visibility_warning_count=low_visibility_warning_count,
        motion_measurement_count=motion_measurement_count,
        lighting_filter=lighting_filter,
        motion_rows=motion_page_rows, motion_page=motion_page, motion_pages=motion_pages,
        motion_total=len(motion_rows), motion_sample_count=len(motion_samples),
        motion_video_filter=motion_video_filter, motion_track_filter=motion_track_text,
        motion_joint_filter=motion_joint_filter, motion_observation_filter=motion_observation_filter,
        motion_video_options=videos,
        motion_track_options=motion_track_options, motion_joint_options=motion_joint_options,
        motion_observation_options=motion_observation_options,
        report_generated=_display_date(datetime.now(timezone.utc)))


@bp.get("/reports/motion.csv")
def motion_csv_export():
    start, end = validated_date(request.args.get("start", "")), validated_date(request.args.get("end", ""))
    if start and end and start > end:
        abort(400)
    videos, _events, _source = _prepare_data(start, end)
    video_filter = request.args.get("motion_video", "")[:80].strip()
    track_text = request.args.get("motion_track", "")[:20].strip()
    joint = request.args.get("motion_joint", "")[:60].strip()
    observation = request.args.get("motion_observation", "")[:80].strip()
    video_search = request.args.get("video_q", "")[:100].strip()
    lighting_filter = request.args.get("lighting_mode", "").strip().upper()
    try:
        track = int(track_text) if track_text else None
    except ValueError:
        abort(400)
    if lighting_filter and lighting_filter not in {"DAY", "LOW_LIGHT", "NIGHT_VISION", "NOT_RECORDED"}:
        abort(400)
    selected = []
    for video in videos:
        if video_filter and video["id"] != video_filter:
            continue
        if video_search and video_search.casefold() not in str(video.get("original_filename", "")).casefold():
            continue
        if lighting_filter and ((video.get("detected_lighting_mode") if lighting_filter == "LOW_LIGHT"
                                 else video.get("lighting_mode_display")) !=
                                ("Not recorded" if lighting_filter == "NOT_RECORDED" else lighting_filter)):
            continue
        for sample in (video.get("motion_analytics") or {}).get("measurements", []) or []:
            if track is not None and sample.get("track_id") != track:
                continue
            if observation and str(sample.get("movement_observation") or "") != observation:
                continue
            row = dict(sample)
            if joint:
                if joint not in (sample.get("joint_measurements") or {}):
                    continue
                row["joint_measurements"] = {joint: sample["joint_measurements"][joint]}
            selected.append(row)
    content = motion_analytics_csv("filtered-video-set", selected)
    return Response(content, mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=anomaX-motion-analytics.csv"})


@bp.get("/reports.csv")
def csv_export():
    start, end = validated_date(request.args.get("start", "")), validated_date(request.args.get("end", ""))
    if start and end and start > end:
        abort(400)
    videos, events, _source = _prepare_data(start, end)
    video_filter = request.args.get("video_id", "")[:80]
    category = request.args.get("category", "")[:80]
    review = request.args.get("review", "")[:40].lower().replace(" ", "_")
    search = request.args.get("event_q", "")[:100].strip()
    video_search = request.args.get("video_q", "")[:100].strip()
    processing_status = request.args.get("processing_status", "")[:40].strip().lower()
    if review and review not in REVIEW_STATUSES:
        abort(400)
    selected_videos = [v for v in videos if (not video_filter or v["id"] == video_filter)
                       and (not video_search or video_search.casefold() in str(v.get("original_filename", "")).casefold())
                       and (not processing_status or v["status"] == processing_status)]
    selected_ids = {v["id"] for v in selected_videos}
    selected_events = _filtered_events(videos, events, video_filter, category, review, search)
    detailed = request.args.get("include_details", "") in {"1", "true", "on"}
    out = io.StringIO(newline="")
    columns = ["record_type", "video_id", "filename", "uploaded_ist", "processing_status",
               "duration_seconds", "processing_duration_seconds", "sample_interval_seconds",
               "lighting_mode", "detected_lighting_mode", "average_brightness", "night_enhancement_applied",
               "movement_observation_count", "low_visibility_warning_count",
               "pose_status", "pose_model", "object_status", "object_model", "fight_status", "fight_model",
               "analysis_summary", "annotated_output_status", "event_id", "event_type", "event_category", "label",
               "timestamp_seconds", "video_timestamp", "start_timestamp_seconds", "end_timestamp_seconds", "event_interval", "track_id",
               "zone_name", "detector", "score", "score_type", "review_status", "review_notes", "evidence"]
    writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for video in selected_videos:
        pose = video.get("pose_analysis") or {}
        objects = video.get("object_analysis") or {}
        fighting = video.get("fight_analysis") or {}
        writer.writerow({"record_type": "video", "video_id": _safe_csv(video["id"]),
            "filename": _safe_csv(video.get("original_filename")),
            "uploaded_ist": video.get("uploaded_display"), "processing_status": video.get("status"),
            "duration_seconds": _safe_csv(video.get("duration", "")),
            "processing_duration_seconds": _safe_csv(video.get("processing_duration", "")),
            "sample_interval_seconds": _safe_csv((video.get("analysis_settings") or {}).get("sample_every_seconds", "")),
            "lighting_mode": _safe_csv(video.get("lighting_mode_display")),
            "detected_lighting_mode": _safe_csv(video.get("detected_lighting_mode")),
            "average_brightness": _safe_csv(video.get("average_brightness", "")),
            "night_enhancement_applied": _safe_csv(video.get("night_enhancement_applied", "")),
            "movement_observation_count": video.get("movement_observation_count", 0),
            "low_visibility_warning_count": video.get("lighting_warning_count", 0),
            "pose_status": _safe_csv(pose.get("status", "not recorded")), "pose_model": _safe_csv(pose.get("model_name")),
            "object_status": _safe_csv(objects.get("status", "not recorded")), "object_model": _safe_csv(objects.get("model_name")),
            "fight_status": _safe_csv(fighting.get("status", "not recorded")), "fight_model": _safe_csv(fighting.get("model_name")),
            "analysis_summary": _safe_csv(video.get("analysis_summary")),
            "annotated_output_status": _safe_csv(video.get("annotated_output_status"))})
    for event in selected_events:
        if str(event.get("video_id")) not in selected_ids:
            continue
        details = event.get("details") or event.get("metadata") or {}
        writer.writerow({"record_type": "event", "video_id": _safe_csv(event.get("video_id_text")),
            "filename": _safe_csv(event.get("original_filename")), "event_id": _safe_csv(event.get("event_id")),
            "event_type": _safe_csv(event.get("event_type")),
            "event_category": _safe_csv(event.get("category_label")),
            "label": _safe_csv(event.get("label")),
            "timestamp_seconds": _safe_csv(event_video_seconds(event) if event_video_seconds(event) is not None else ""),
            "video_timestamp": event.get("video_timestamp_label") or "",
            "start_timestamp_seconds": _safe_csv(event.get("start_timestamp_seconds", "")),
            "end_timestamp_seconds": _safe_csv(event.get("end_timestamp_seconds", "")),
            "event_interval": (f"{format_video_timestamp(event.get('start_timestamp_seconds'))}-{format_video_timestamp(event.get('end_timestamp_seconds'))}"
                               if _number(event.get("start_timestamp_seconds")) is not None and
                               _number(event.get("end_timestamp_seconds")) is not None and
                               _number(event.get("end_timestamp_seconds")) > _number(event.get("start_timestamp_seconds")) else ""),
            "track_id": _safe_csv(event.get("track_id")),
            "zone_name": _safe_csv(event.get("zone_name_display")),
            "detector": _safe_csv(event.get("detection_method")),
            "score": _safe_csv(event.get("model_score", event.get("anomaly_score", event.get("confidence", "")))),
            "score_type": event.get("score_type"), "review_status": event.get("status_label"),
            "review_notes": _safe_csv(event.get("review_notes")),
            "evidence": _safe_csv(explain_event(event) + (" " + repr(details)[:2000] if detailed else ""))})
    filename = "anomaX-ai-report-detailed.csv" if detailed else "anomaX-ai-report-summary.csv"
    return Response(out.getvalue(), mimetype="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f"attachment; filename={filename}"})
