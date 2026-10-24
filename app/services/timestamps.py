"""Safe conversion and formatting for positions inside a video."""
import math
from datetime import datetime, timezone


def utc_datetime_sort_key(value):
    """Normalize stored BSON datetimes and legacy ISO strings for chronology.

    Naive legacy values are treated as UTC, matching the application's prior
    timestamp writes. Invalid or absent values sort behind dated uploads.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return float("-inf")
    else:
        return float("-inf")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    try:
        return parsed.astimezone(timezone.utc).timestamp()
    except (OverflowError, OSError, ValueError):
        return float("-inf")


def event_video_seconds(event):
    """Return a persisted video position, including known legacy field names.

    Wall-clock fields such as ``created_at`` are deliberately never used as a
    substitute for an absent position in the source video.
    """
    if not isinstance(event, dict):
        return None
    for key in ("timestamp_seconds", "video_position_seconds", "timestamp"):
        value = event.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            seconds = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(seconds) and seconds >= 0:
            return seconds
    return None


def format_video_timestamp(seconds):
    """Format a video position as MM:SS.ss or HH:MM:SS.ss; None if invalid."""
    if isinstance(seconds, bool) or seconds is None:
        return None
    try:
        seconds = float(seconds)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    total_centiseconds = round(seconds * 100)
    hours, remaining = divmod(total_centiseconds, 360000)
    minutes, remaining = divmod(remaining, 6000)
    whole_seconds, centiseconds = divmod(remaining, 100)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"
    total_minutes = int(seconds // 60)
    return f"{total_minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"
