"""Dependency-free, text-only PDF reports built from persisted records."""
import textwrap
from datetime import date, datetime

from app.services.report_service import explain_event
from app.services.timestamps import event_video_seconds, format_video_timestamp
from app.services.event_groups import group_event_episodes


def _display(value):
    if value is None:
        return "Not available"
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def report_lines(video, job, events):
    """Return a bounded, evidence-grounded list of report lines."""
    video_id = str(video.get("_id", ""))
    safe_name = str(video.get("original_filename", "video"))[:180]
    settings = video.get("analysis_settings") or {}
    pose = video.get("pose_analysis") or {}
    objects = video.get("object_analysis") or {}
    fighting = video.get("fight_analysis") or {}
    episodes = group_event_episodes(events)
    categories = {}
    for episode in episodes:
        categories[episode["event_type"]] = categories.get(episode["event_type"], 0) + 1
    lines = [
        "Sentinel Activity Lab - Video Analysis Report",
        "Educational research output. Candidate signals require human review.",
        "",
        f"Video ID: {video_id}",
        f"Filename: {safe_name}",
        f"Processing status: {_display(video.get('processing_status'))}",
        f"Uploaded: {_display(video.get('upload_date'))}",
        f"Duration (seconds): {_display(video.get('duration'))}",
        f"Job status: {_display((job or {}).get('status'))}",
        f"Job started: {_display((job or {}).get('started_at'))}",
        f"Job completed: {_display((job or {}).get('completed_at'))}",
        f"Frame sample interval (seconds): {_display(settings.get('sample_every_seconds'))}",
        f"Pose detector: {_display(pose.get('model_name'))} / {_display(pose.get('status'))}",
        f"Object detector: {_display(objects.get('model_name'))} / {_display(objects.get('status'))}",
        f"Object observations by class: {_display(objects.get('counts_by_class') or 'No saved class counts')}",
        f"Fighting detector: {_display(fighting.get('model_name'))} / {_display(fighting.get('status'))}",
        f"Fighting candidate intervals: {_display(fighting.get('candidate_event_count', 0))}",
        f"Detector summary: {str(video.get('analysis_summary') or 'No saved analysis summary.')[:1200]}",
        "",
        f"Persisted event records: {len(events)}",
        f"Anomaly review episodes: {len(episodes)}",
        "Episodes by category: " + (", ".join(f"{name}: {count}" for name, count in sorted(categories.items()))
                                     if categories else "No saved anomaly-candidate episodes"),
        "",
        "Chronological event timeline (all individual stored events retained):",
    ]
    for index, event in enumerate(events[:2000], 1):
        timestamp_text = format_video_timestamp(event_video_seconds(event)) or "time unavailable"
        score = event.get("model_score", event.get("anomaly_score"))
        confidence = event.get("confidence")
        score_text = (f"uncalibrated model score {score}" if event.get("event_type") == "SUSPECTED_FIGHTING" and score is not None else
                      f"anomaly score {score}" if score is not None else
                      f"confidence {confidence}" if confidence is not None else "score unavailable")
        start = event.get("start_timestamp_seconds")
        end = event.get("end_timestamp_seconds")
        interval_text = (f"Interval: {format_video_timestamp(start)} to {format_video_timestamp(end)}"
                         if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start
                         else f"Point timestamp: {timestamp_text}")
        details = event.get("details") or event.get("metadata") or {}
        zone = event.get("zone_name", details.get("zone_name", details.get("zone_id")))
        score_evidence = (f"model score {event.get('model_score')} (uncalibrated)"
                          if event.get("model_score") is not None else
                          f"anomaly score {event.get('anomaly_score')}" if event.get("anomaly_score") is not None else
                          f"detector confidence {confidence}" if confidence is not None else "no score provided")
        lines.extend([
            "",
            f"Event {index}: {event.get('event_type', 'event')} - {event.get('label', '')}",
            f"{interval_text}; {score_text}; detector: {event.get('detection_method') or 'unknown'}",
            f"Zone: {_display(zone)}; temporary track ID: {_display(event.get('track_id'))}; {score_evidence}",
            f"Review status: {event.get('incident_status') or 'not reviewed'}",
            f"Review notes: {str(event.get('review_notes') or 'None')[:1000]}",
            f"Explanation: {explain_event(event)[:1200]}",
        ])
    if not events:
        lines.append("No event records were saved for this video.")
    lines.extend(["", "Limitations", "This report contains saved detector output only. Scores are not calibrated probabilities or proof of an incident.",
                  "Violence-class video predictions are clip-level review signals. They do not identify people or provide polygon-zone localization.",
                  "Validation-only processing does not produce anomaly predictions. Model accuracy has not been established by this report."])
    wrapped = []
    for line in lines:
        if not line:
            wrapped.append("")
        else:
            wrapped.extend(textwrap.wrap(line, width=92, break_long_words=True, break_on_hyphens=False) or [""])
    return wrapped


def _pdf_literal(value):
    # Core PDF Helvetica uses WinAnsi; use an ASCII-safe representation so
    # arbitrary user notes cannot break the PDF content stream.
    text = "".join(char if ord(char) >= 32 else " " for char in str(value))
    text = text.encode("ascii", "replace").decode("ascii")
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(video, job, events):
    """Build a valid multi-page PDF with only the Python standard library."""
    lines = report_lines(video, job, events)
    page_chunks = [lines[i:i + 48] for i in range(0, len(lines), 48)] or [[]]
    objects = {}
    font_id = 3
    objects[font_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    page_ids = []
    next_id = 4
    content_ids = []
    for chunk in page_chunks:
        content_id, page_id = next_id, next_id + 1
        next_id += 2
        page_ids.append(page_id)
        content_ids.append(content_id)
        commands = ["BT", "/F1 9 Tf", "50 790 Td", "12 TL"]
        commands.extend(f"({_pdf_literal(line)}) Tj T*" for line in chunk)
        commands.append("ET")
        objects[content_id] = ("<< /Length %d >>\nstream\n%s\nendstream" % (
            len("\n".join(commands).encode("ascii")), "\n".join(commands))).encode("ascii")
        objects[page_id] = f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>".encode("ascii")
    objects[2] = ("<< /Type /Pages /Kids [%s] /Count %d >>" % (
        " ".join(f"{page_id} 0 R" for page_id in page_ids), len(page_ids))).encode("ascii")
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    max_id = max(objects)
    result = bytearray(b"%PDF-1.4\n%SentinelReport\n")
    offsets = [0] * (max_id + 1)
    for object_id in range(1, max_id + 1):
        offsets[object_id] = len(result)
        result.extend(f"{object_id} 0 obj\n".encode("ascii"))
        result.extend(objects[object_id])
        result.extend(b"\nendobj\n")
    xref_offset = len(result)
    result.extend(f"xref\n0 {max_id + 1}\n".encode("ascii"))
    result.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        result.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    result.extend(f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii"))
    return bytes(result)
