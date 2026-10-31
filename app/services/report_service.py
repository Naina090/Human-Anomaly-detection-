"""Formatting helpers for evidence-based reports built from saved MongoDB data."""
import csv
import io
from app.services.timestamps import event_video_seconds, format_video_timestamp
from app.services.event_groups import group_event_episodes, event_signature


def explain_event(event):
    """Describe only fields that the configured detector actually recorded."""
    kind = event.get("event_type", "event").replace("_", " ")
    method = event.get("detection_method") or "recorded detector"
    if event.get("event_type") == "SUSPECTED_FIGHTING":
        details = event.get("details") or {}
        score = event.get("model_score", details.get("model_score"))
        score_text = f" The uncalibrated Violence class score was {score:.3f}." if isinstance(score, (int, float)) else " No model score was recorded."
        return (f"A temporal video classifier emitted a candidate Violence-class interval using {method}."
                f"{score_text} The classifier does not localize a person or establish polygon-zone membership; "
                "sports, rough play, and other intense interactions can be confused with violence. Human review is required.")
    if event.get("event_type") == "zone_tracking_gap":
        details = event.get("details") or {}
        gap = details.get("observation_gap_seconds")
        gap_text = f"{float(gap):.2f}" if isinstance(gap, (int, float)) else "an unknown"
        return (f"Temporary track {event.get('track_id')} resumed observations after a {gap_text} second gap "
                f"in {details.get('zone_name') or details.get('zone_id') or 'the zone'}. "
                "The missing observations do not prove an exit, re-entry, or identity continuity.")
    if kind == "object detection":
        confidence = event.get("confidence")
        score = f" Model confidence: {confidence}." if confidence is not None else " No confidence value was supplied."
        return f"Observed {event.get('label') or 'a supported object'} using {method}; this is an object observation, not an anomaly.{score}"
    if event.get("event_type") == "RESTRICTED_ZONE_ENTRY":
        details = event.get("details") or {}
        zone = details.get("zone_name") or details.get("zone_id") or "the active restricted zone"
        confidence = event.get("confidence")
        confidence_text = f" The person-detector confidence was {confidence}." if confidence is not None else " The detector did not provide a confidence value."
        return (f"Temporary track {event.get('track_id')} was confirmed outside and then inside {zone}, "
                f"using the bottom-center of its detected person box and consecutive sampled observations.{confidence_text} "
                "This is a restricted-area entry for human review, not proof of intent, identity, or wrongdoing.")
    if event.get("event_type") in {"zone_presence_observed", "zone_entry", "zone_exit"}:
        details = event.get("details") or {}
        zone = details.get("zone_name") or details.get("zone_id") or "an active zone"
        if event.get("event_type") == "zone_presence_observed":
            return (f"Temporary person track {event.get('track_id')} was first observed inside {zone} at this sampled time. "
                    "Its entry time is unknown; this is a zone occupancy observation, not an anomaly or identity.")
        action = "entered" if event.get("event_type") == "zone_entry" else "exited"
        return (f"Temporary person track {event.get('track_id')} {action} {zone}, based on the bottom-center "
                "of a detected person box. This is a zone occupancy observation, not an anomaly or identity.")
    if event.get("event_type") == "potential_fall":
        details = event.get("details") or event.get("metadata") or {}
        evidence = []
        if details.get("previously_upright") is True:
            evidence.append("prior upright pose was observed")
        if details.get("vertical_drop_fraction") is not None:
            evidence.append(f"measured vertical drop fraction {details['vertical_drop_fraction']}")
        if details.get("torso_horizontal_ratio") is not None:
            evidence.append(f"measured torso orientation ratio {details['torso_horizontal_ratio']}")
        if details.get("horizontal_persistence_seconds") is not None:
            evidence.append(f"horizontal posture persisted {details['horizontal_persistence_seconds']} seconds")
        score = event.get("anomaly_score")
        score_text = f" Heuristic review score: {score}." if score is not None else " No score was recorded."
        evidence_text = "; ".join(evidence) if evidence else "the stored event contains no detailed transition evidence"
        return f"Candidate flagged by {method} because {evidence_text}. This is a review signal, not confirmation.{score_text}"
    return f"Stored {kind} from {method}. Review the recorded evidence before drawing conclusions."

from app.services.repository import report_data


def build_analysis_summary(result, sample_seconds):
    """Summarize only the detector outputs and metadata present in ``result``."""
    duration = result.get("duration")
    sampled = int(result.get("sampled_frames") or 0)
    parts = []
    if duration is not None:
        parts.append(f"Video duration: {float(duration):.2f} seconds; {sampled} frames sampled at approximately {float(sample_seconds):g}-second intervals.")

    objects = result.get("object_analysis") or {}
    if objects.get("status") == "complete":
        counts = objects.get("counts_by_class") or {}
        if counts:
            count_text = ", ".join(f"{label}: {count} frame observations" for label, count in sorted(counts.items()))
            parts.append(f"Supported object observations: {count_text}.")
        else:
            parts.append("No supported objects were observed in the sampled frames; this does not rule out objects between samples.")
        tracks = int(objects.get("person_track_count_estimate") or 0)
        parts.append(f"Approximate person tracks: {tracks}; tracker IDs are temporary associations, not unique identities.")
    elif objects.get("status") in {"unavailable", "failed"}:
        parts.append("Object detection was unavailable: " + (objects.get("error_message") or "the configured detector could not run."))
    else:
        parts.append("No object detector was configured; no object predictions were produced.")

    pose = result.get("pose_analysis") or {}
    if pose.get("status") == "complete":
        parts.append(f"Pose estimation sampled {int(pose.get('frames_sampled') or 0)} frames; "
                     f"people were observed in {int(pose.get('frames_with_people') or 0)} sampled frames. "
                     "Keypoints are model estimates, not identity data.")
        if pose.get("person_track_count_estimate") is not None:
            parts.append(f"Approximate pose-tracker IDs observed: {int(pose.get('person_track_count_estimate') or 0)}; "
                         "temporary IDs are not unique identities.")
        entries = int(pose.get("zone_entry_count") or 0)
        exits = int(pose.get("zone_exit_count") or 0)
        initial = int(pose.get("zone_presence_observed_count") or 0)
        if entries or exits or initial:
            parts.append(f"Polygon tracking recorded {entries} confirmed restricted-zone entries, {exits} confirmed exits, and "
                         f"{initial} tracks first observed inside (entry time unknown). Entries are review events, not proof of intent or identity.")
    elif pose.get("status") in {"unavailable", "failed"}:
        parts.append("Pose estimation was unavailable: " +
                     (pose.get("error_message") or "the configured model could not run."))
    else:
        parts.append("No pose model was configured; no pose or fall estimate was produced.")

    fighting = result.get("fight_analysis") or {}
    if fighting.get("status") == "complete":
        count = int(fighting.get("candidate_event_count") or 0)
        parts.append(f"Temporal fighting review model analyzed {int(fighting.get('windows_analyzed') or 0)} clips and emitted {count} candidate interval(s). "
                     "This Violence / Non-Violence model score is uncalibrated, cannot identify a person or polygon location, and requires human review.")
    elif fighting.get("status") == "insufficient_video":
        parts.append("The configured temporal fighting model did not run because the video was shorter than its required clip window; no fighting prediction was produced.")
    elif fighting.get("status") in {"unavailable", "failed"}:
        parts.append("Temporal fighting analysis was unavailable: " + (fighting.get("error_message") or "the configured model could not run."))
    elif fighting.get("status") == "disabled":
        parts.append("Temporal fighting analysis was disabled for this run; no fighting prediction was produced.")
    else:
        parts.append("No temporal fighting model was configured; no fighting prediction was produced.")

    events = result.get("events") or []
    fall_events = [event for event in events if event.get("event_type") == "potential_fall"]
    if pose.get("status") in {"unavailable", "failed"}:
        pass
    elif pose.get("status") == "complete" or result.get("analysis_mode") == "pose_heuristic":
        if fall_events:
            times = ", ".join(f"{float(event.get('timestamp_seconds', 0)):.2f}s (score {event.get('anomaly_score')})" for event in fall_events)
            parts.append("The unvalidated pose transition heuristic produced candidate review signals at " + times + ". Scores are review signals, not proof of an incident.")
        else:
            parts.append("The unvalidated pose transition heuristic produced no candidate event in the sampled frames; this is not evidence that no fall occurred.")
    elif pose.get("status") not in {"not_configured", None}:
        parts.append("No pose-based anomaly model was configured, so no fall/anomaly prediction was produced.")
    return " ".join(parts)


def video_report_csv(video, events):
    """Return a flat report with one metadata row and one row per stored event."""
    output = io.StringIO(newline="")
    columns = ["record_type", "video_id", "filename", "status", "duration_seconds",
               "processing_duration_seconds", "annotated_output_status", "annotated_output_format",
               "sample_interval_seconds", "pose_status", "pose_model", "pose_model_version",
               "lighting_mode", "detected_lighting_mode", "average_brightness",
               "night_enhancement_applied", "movement_observation_count", "low_visibility_warning_count",
               "pose_frames_sampled", "object_status", "object_model", "object_counts_by_class",
               "person_track_count_estimate", "fight_status", "fight_model", "summary", "event_type", "label",
               "timestamp_seconds", "start_timestamp_seconds", "end_timestamp_seconds",
               "video_timestamp", "event_interval", "event_group_id", "incident_status", "review_notes",
               "frame_number", "confidence", "anomaly_score", "model_score", "track_id", "detection_method", "details"]
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    video_id = str(video.get("_id", ""))
    writer.writerow({"record_type": "video", "video_id": video_id,
        "filename": video.get("original_filename", ""),
        "status": video.get("processing_status", ""),
        "duration_seconds": video.get("duration", ""),
        "processing_duration_seconds": video.get("processing_duration", ""),
        "annotated_output_status": video.get("annotated_output_status", ""),
        "annotated_output_format": " / ".join(str(value) for value in (
            (video.get("annotated_output_metadata") or {}).get("codec"),
            (video.get("annotated_output_metadata") or {}).get("container")) if value),
        "sample_interval_seconds": (video.get("analysis_settings") or {}).get("sample_every_seconds", ""),
        "lighting_mode": video.get("lighting_mode", "Not recorded"),
        "detected_lighting_mode": (video.get("lighting_analysis") or {}).get("detected_mode", "Not recorded"),
        "average_brightness": video.get("average_brightness", ""),
        "night_enhancement_applied": video.get("night_enhancement_applied", "Not recorded"),
        "movement_observation_count": len(video.get("movement_observations") or []),
        "low_visibility_warning_count": len((video.get("lighting_analysis") or {}).get("warnings") or []),
        "pose_status": (video.get("pose_analysis") or {}).get("status", ""),
        "pose_model": (video.get("pose_analysis") or {}).get("model_name", ""),
        "pose_model_version": (video.get("pose_analysis") or {}).get("model_version", ""),
        "pose_frames_sampled": (video.get("pose_analysis") or {}).get("frames_sampled", ""),
        "object_status": (video.get("object_analysis") or {}).get("status", ""),
        "object_model": (video.get("object_analysis") or {}).get("model_name", ""),
        "object_counts_by_class": repr((video.get("object_analysis") or {}).get("counts_by_class", {})),
        "person_track_count_estimate": (video.get("object_analysis") or {}).get("person_track_count_estimate", ""),
        "fight_status": (video.get("fight_analysis") or {}).get("status", ""),
        "fight_model": (video.get("fight_analysis") or {}).get("model_name", ""),
        "summary": video.get("analysis_summary", "")})
    # Ordinary movement summaries stay separate from anomaly event records.
    for movement in video.get("movement_observations") or []:
        start = movement.get("start_timestamp_seconds")
        end = movement.get("end_timestamp_seconds")
        writer.writerow({
            "record_type": "movement_observation", "video_id": video_id,
            "filename": video.get("original_filename", ""),
            "lighting_mode": movement.get("lighting_mode", video.get("lighting_mode", "Not recorded")),
            "timestamp_seconds": start if isinstance(start, (int, float)) else "",
            "video_timestamp": (format_video_timestamp(start) or "") if isinstance(start, (int, float)) else "",
            "start_timestamp_seconds": start if isinstance(start, (int, float)) else "",
            "end_timestamp_seconds": end if isinstance(end, (int, float)) else "",
            "event_interval": (f"{format_video_timestamp(start)}-{format_video_timestamp(end)}"
                               if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start else ""),
            "event_type": "movement_observation", "label": movement.get("category", ""),
            "track_id": movement.get("track_id", ""),
            "details": repr({"track_segment_id": movement.get("track_segment_id"),
                             "evidence_observations": movement.get("evidence_observations"),
                             "interpretation": movement.get("interpretation", "")}),
        })
    groups = group_event_episodes(events)
    event_groups = {str(event.get("_id")): group["group_id"] for group in groups
                    for event in group.get("events", [])}
    for event in events:
        start = event.get("start_timestamp_seconds")
        end = event.get("end_timestamp_seconds")
        interval = (f"{format_video_timestamp(start)}-{format_video_timestamp(end)}"
                    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start else "")
        writer.writerow({"record_type": "event", "video_id": video_id,
            "filename": video.get("original_filename", ""),
            "status": video.get("processing_status", ""),
            "event_type": event.get("event_type", ""), "label": event.get("label", ""),
            "timestamp_seconds": event_video_seconds(event) if event_video_seconds(event) is not None else "",
            "video_timestamp": format_video_timestamp(event_video_seconds(event)) or "",
            "event_interval": interval,
            "event_group_id": event_groups.get(str(event.get("_id")), ""),
            "incident_status": event.get("incident_status", ""),
            "review_notes": event.get("review_notes", ""),
            "start_timestamp_seconds": event.get("start_timestamp_seconds", ""),
            "end_timestamp_seconds": event.get("end_timestamp_seconds", ""),
            "frame_number": event.get("frame_number", ""),
            "confidence": event.get("confidence", ""),
            "anomaly_score": event.get("anomaly_score", ""),
            "model_score": event.get("model_score", (event.get("details") or {}).get("model_score", "")),
            "track_id": event.get("track_id", ""),
            "detection_method": event.get("detection_method", ""),
            "details": repr(event.get("details", event.get("metadata", "")))})
    return output.getvalue()


def motion_analytics_csv(video_id, measurements):
    """Export stored sampled pose geometry; no inference is run during export."""
    output = io.StringIO(newline="")
    fields = ["video_id", "tracking_id", "track_segment_id", "frame_number", "source_fps",
              "video_timestamp_seconds", "movement_duration_seconds", "measurement_valid",
              "lighting_mode", "joint_name", "raw_angle_degrees", "smoothed_angle_degrees",
              "angular_change_degrees", "angular_velocity_degrees_per_second",
              "body_center_x_pixels", "body_center_y_pixels", "body_center_x_normalized",
              "body_center_y_normalized", "horizontal_displacement_pixels",
              "vertical_displacement_pixels", "elapsed_seconds", "movement_speed_pixels_per_second",
              "movement_speed_normalized_per_second", "movement_speed_unit",
              "movement_direction_degrees", "movement_observation", "keypoint_confidence"]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for sample in measurements:
        common = {
            "video_id": str(sample.get("video_id") or video_id), "tracking_id": sample.get("track_id"),
            "track_segment_id": _csv_text(sample.get("track_segment_id")),
            "frame_number": sample.get("frame_number"),
            "source_fps": _csv_number(sample.get("source_fps"), 4),
            "video_timestamp_seconds": _csv_number(sample.get("timestamp_seconds"), 4),
            "movement_duration_seconds": _csv_number(sample.get("movement_duration_seconds"), 4),
            "measurement_valid": bool(sample.get("measurement_valid", bool(sample.get("joint_measurements")))),
            "lighting_mode": _csv_text(sample.get("lighting_mode")),
            "body_center_x_pixels": _csv_number(sample.get("body_center_x_pixels"), 3),
            "body_center_y_pixels": _csv_number(sample.get("body_center_y_pixels"), 3),
            "body_center_x_normalized": _csv_number(sample.get("body_center_x_normalized"), 6),
            "body_center_y_normalized": _csv_number(sample.get("body_center_y_normalized"), 6),
            "horizontal_displacement_pixels": _csv_number(sample.get("horizontal_displacement_pixels"), 3),
            "vertical_displacement_pixels": _csv_number(sample.get("vertical_displacement_pixels"), 3),
            "elapsed_seconds": _csv_number(sample.get("elapsed_seconds"), 6),
            "movement_speed_pixels_per_second": _csv_number(sample.get("movement_speed_pixels_per_second"), 3),
            "movement_speed_normalized_per_second": _csv_number(sample.get("movement_speed_normalized_per_second"), 6),
            "movement_speed_unit": "pixels/second; not calibrated real-world speed",
            "movement_direction_degrees": _csv_number(sample.get("movement_direction_degrees"), 3),
            "movement_observation": _csv_text(sample.get("movement_observation")),
        }
        joints = sample.get("joint_measurements") or {}
        if not joints:
            writer.writerow({**common, "joint_name": "", "raw_angle_degrees": "",
                             "smoothed_angle_degrees": "", "angular_change_degrees": "",
                             "angular_velocity_degrees_per_second": "", "keypoint_confidence": ""})
        for joint, value in joints.items():
            if not isinstance(value, dict):
                continue
            writer.writerow({**common, "joint_name": _csv_text(joint),
                "raw_angle_degrees": _csv_number(value.get("raw_angle_degrees"), 3),
                "smoothed_angle_degrees": _csv_number(value.get("smoothed_angle_degrees"), 3),
                "angular_change_degrees": _csv_number(value.get("angular_change_degrees"), 3),
                "angular_velocity_degrees_per_second": _csv_number(value.get("angular_velocity_degrees_per_second"), 3),
                "keypoint_confidence": _csv_number(value.get("keypoint_confidence"), 4)})
    return output.getvalue()


def _csv_number(value, precision):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ""
    import math
    return round(float(value), precision) if math.isfinite(float(value)) else ""


def _csv_text(value):
    value = "" if value is None else str(value)
    if value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


__all__ = ["report_data", "video_report_csv", "motion_analytics_csv", "group_event_episodes", "event_signature"]
