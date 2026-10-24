"""Grouping helpers for notification/report episodes built from saved events."""
import hashlib
import json
import math

from app.services.timestamps import event_video_seconds

ALERTABLE_EVENTS = {"RESTRICTED_ZONE_ENTRY", "potential_fall", "SUSPECTED_FIGHTING"}


def _valid_number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    except (TypeError, ValueError):
        return None


def event_signature(event):
    """Stable event identity that survives ObjectId changes on a retry."""
    details = event.get("details") or {}
    timestamp = event_video_seconds(event)
    start = _valid_number(event.get("start_timestamp_seconds"))
    end = _valid_number(event.get("end_timestamp_seconds"))
    identity = {
        "type": event.get("event_type"),
        "time": round(timestamp, 1) if timestamp is not None else None,
        "start": round(start, 1) if start is not None else None,
        "end": round(end, 1) if end is not None else None,
        "track": event.get("track_id"),
        "zone": event.get("zone_id", details.get("zone_id")),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def group_event_episodes(events, merge_gap_seconds=1.0):
    """Group related detector intervals without filling unobserved time gaps.

    Point events (including restricted-zone entries) remain independent. For
    merged temporal episodes, ``intervals`` retains each observed interval so
    callers never imply continuous activity across a gap.
    """
    candidates = []
    for event in events or []:
        event_type = event.get("event_type")
        if event_type not in ALERTABLE_EVENTS:
            continue
        timestamp = event_video_seconds(event)
        if timestamp is None:
            continue
        start = _valid_number(event.get("start_timestamp_seconds"))
        end = _valid_number(event.get("end_timestamp_seconds"))
        has_interval = start is not None and end is not None and end > start
        if not has_interval:
            start = end = timestamp
        details = event.get("details") or {}
        candidates.append({"event": dict(event), "event_type": event_type,
            "start": start, "end": end, "has_interval": has_interval,
            "track_id": event.get("track_id"),
            "zone_id": event.get("zone_id", details.get("zone_id")),
            "signature": event_signature(event)})
    candidates.sort(key=lambda item: (item["start"], item["end"], item["event_type"]))
    groups = []
    gap = max(0.0, float(merge_gap_seconds))
    interval_buckets = {}
    for item in candidates:
        # Zone entries and point observations are never combined into a
        # fabricated interval. Unknown-track falls are also kept independent.
        if item["event_type"] == "RESTRICTED_ZONE_ENTRY" or not item["has_interval"] or (
                item["event_type"] == "potential_fall" and item["track_id"] is None):
            groups.append({"event_type": item["event_type"], "events": [item["event"]],
                "signatures": [item["signature"]], "track_id": item["track_id"],
                "zone_id": item["zone_id"], "start": item["start"], "end": item["end"],
                "intervals": [[item["start"], item["end"]]] if item["has_interval"] else []})
            continue
        key = (item["event_type"], item["track_id"], item["zone_id"])
        interval_buckets.setdefault(key, []).append(item)

    for (event_type, track_id, zone_id), bucket in interval_buckets.items():
        active = None
        for item in sorted(bucket, key=lambda row: (row["start"], row["end"])):
            if active is None or item["start"] > active["last_end"] + gap:
                if active is not None:
                    groups.append(active)
                active = {"event_type": event_type, "events": [item["event"]],
                    "signatures": [item["signature"]], "track_id": track_id,
                    "zone_id": zone_id, "start": item["start"], "end": item["end"],
                    "last_end": item["end"], "intervals": [[item["start"], item["end"]]]}
                continue
            active["events"].append(item["event"])
            active["signatures"].append(item["signature"])
            active["last_end"] = max(active["last_end"], item["end"])
            active["end"] = max(active["end"], item["end"])
            if item["start"] <= active["intervals"][-1][1]:
                active["intervals"][-1][1] = max(active["intervals"][-1][1], item["end"])
            else:
                active["intervals"].append([item["start"], item["end"]])
        if active is not None:
            groups.append(active)

    for group in groups:
        group.pop("last_end", None)
        group["start_timestamp_seconds"] = min(group["start"], group["end"])
        group["end_timestamp_seconds"] = max(group["start"], group["end"]) if group["intervals"] else None
        group["group_id"] = hashlib.sha256("|".join(sorted(group["signatures"])).encode("ascii")).hexdigest()
        group["event_count"] = len(group["events"])
    return sorted(groups, key=lambda group: (group["start_timestamp_seconds"], group["event_type"]))
