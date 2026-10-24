"""Atomic local JSON persistence for uploads/jobs/results while Atlas is down.

The in-process lock also takes an OS file lock so separate Flask worker
processes sharing this fallback file cannot reserve the same alert slot.
It never fabricates events or reports MongoDB persistence as successful.
"""
import json
import os
import tempfile
from datetime import datetime, timezone
from threading import RLock
from bson import ObjectId

class _FallbackLock:
    """Serialize fallback reads/writes both between threads and processes."""
    def __init__(self):
        self._thread_lock = RLock()
        self._stream = None

    def __enter__(self):
        self._thread_lock.acquire()
        try:
            lock_path = os.path.abspath(_path()) + ".lock"
            os.makedirs(os.path.dirname(lock_path), exist_ok=True)
            self._stream = open(lock_path, "a+b")
            self._stream.seek(0, os.SEEK_END)
            if self._stream.tell() == 0:
                self._stream.write(b"\0")
                self._stream.flush()
            self._stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._stream.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX)
            return self
        except Exception:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            self._thread_lock.release()
            raise

    def __exit__(self, _exc_type, _exc, _traceback):
        try:
            if self._stream is not None:
                if os.name == "nt":
                    import msvcrt
                    self._stream.seek(0)
                    msvcrt.locking(self._stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
                self._stream.close()
                self._stream = None
        finally:
            self._thread_lock.release()


_lock = _FallbackLock()


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, ObjectId):
        return str(value)
    raise TypeError(type(value).__name__)


def _path():
    from flask import current_app
    return current_app.config["LOCAL_FALLBACK_PATH"]


def _read():
    try:
        with open(_path(), "r", encoding="utf-8") as stream:
            rows = json.load(stream)
            return rows if isinstance(rows, list) else []
    except FileNotFoundError:
        return []


def _write(rows):
    path = _path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="local-history-", suffix=".tmp", dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(rows, stream, default=_json_default, ensure_ascii=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def save_pending_upload(video, error):
    """Persist a validated local upload and a real local background job."""
    with _lock:
        rows = _read()
        identifier = str(ObjectId())
        job_id = str(ObjectId())
        now = datetime.now(timezone.utc)
        row = {**video, "_id": identifier, "processing_status": "queued",
               "error_message": None, "database_status": "local_only",
               "database_message": "MongoDB is unavailable; this analysis is being stored locally until Atlas reconnects.",
               "local_only": True, "upload_date": now, "updated_at": now,
               "active_job_id": job_id,
               "local_job": {"_id": job_id, "video_id": identifier, "status": "queued",
                             "progress_percent": 0, "created_at": now, "started_at": None,
                             "completed_at": None, "error_message": None,
                             "database_error": str(error)}}
        rows.append(row)
        _write(rows)
        return identifier, job_id


def get_pending_video(identifier):
    with _lock:
        return next((row for row in _read() if row.get("_id") == str(identifier)), None)


def update_pending_video(identifier, changes):
    """Apply Mongo-style dotted updates to one local-only video document."""
    with _lock:
        rows = _read()
        row = next((item for item in rows if item.get("_id") == str(identifier)), None)
        if row is None:
            return False
        target_status = changes.get("processing_status")
        if target_status is not None:
            current = row.get("processing_status")
            allowed = {
                "queued": {None, "queued", "failed", "interrupted", "waiting_for_mongodb"},
                "processing": {None, "queued", "processing", "waiting_for_mongodb"},
                "complete": {None, "queued", "processing", "complete"},
                "failed": {None, "queued", "processing", "waiting_for_mongodb", "failed"},
                "interrupted": {None, "queued", "processing", "interrupted"},
                "waiting_for_mongodb": {None, "queued", "waiting_for_mongodb"},
            }
            if current not in allowed.get(target_status, set()):
                return False
        for key, value in changes.items():
            parts = key.split(".")
            target = row
            for part in parts[:-1]:
                if not isinstance(target.get(part), dict):
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value
        row["updated_at"] = datetime.now(timezone.utc)
        _write(rows)
        return True


def update_pending_job(job_id, changes):
    with _lock:
        rows = _read()
        row = next((item for item in rows
                    if str((item.get("local_job") or {}).get("_id")) == str(job_id)), None)
        if row is None:
            return None
        job = row.setdefault("local_job", {})
        target_status = changes.get("status")
        if target_status is not None:
            current = job.get("status")
            allowed = {
                "queued": {None, "queued"},
                "processing": {"queued", "processing", "running"},
                "completed": {"queued", "processing", "running", "completed"},
                "failed": {"queued", "processing", "running", "failed"},
                "interrupted": {"queued", "processing", "running", "interrupted"},
            }
            if current not in allowed.get(target_status, set()):
                return False
        job.update(changes)
        row["updated_at"] = datetime.now(timezone.utc)
        _write(rows)
        return True


def claim_pending_job(job_id):
    """Atomically transition one local job from queued to processing."""
    with _lock:
        rows = _read()
        row = next((item for item in rows
                    if str((item.get("local_job") or {}).get("_id")) == str(job_id)), None)
        job = (row or {}).get("local_job") or {}
        if row is None or job.get("status") != "queued":
            return None
        now = datetime.now(timezone.utc)
        job.update({"status": "processing", "started_at": now})
        row.update({"processing_status": "processing", "updated_at": now})
        _write(rows)
        return dict(job)


def find_pending_video_by_job(job_id):
    with _lock:
        return next((row for row in _read()
                     if str((row.get("local_job") or {}).get("_id")) == str(job_id)), None)


def save_pending_events(identifier, events):
    """Replace this run's events to keep retries idempotent."""
    prepared = []
    for event in events or []:
        item = dict(event)
        item.setdefault("_id", ObjectId())
        item["video_id"] = ObjectId(str(identifier))
        prepared.append(item)
    saved = update_pending_video(identifier, {"local_events": prepared}) if get_pending_video(identifier) else False
    return len(prepared) if saved else 0


def get_pending_events(identifier, limit=2000):
    row = get_pending_video(identifier) or {}
    events = [event for event in row.get("local_events", []) if event.get("superseded") is not True]
    events.sort(key=lambda event: float(event.get("timestamp_seconds") or 0))
    return events[:max(1, min(50000, int(limit)))]


def update_pending_alert(identifier, delivery):
    """Persist delivery state on exactly one local event."""
    with _lock:
        rows = _read()
        for row in rows:
            for event in row.get("local_events", []):
                if str(event.get("_id")) == str(identifier):
                    event["alert_delivery"] = dict(delivery)
                    alerts = row.setdefault("local_alert_deliveries", [])
                    alert = next((item for item in alerts if str(item.get("_id")) == str(identifier)), None)
                    if alert is None:
                        alert = {key: value for key, value in event.items() if key != "alert_delivery"}
                        alerts.append(alert)
                    alert["alert_delivery"] = dict(delivery)
                    row["updated_at"] = datetime.now(timezone.utc)
                    _write(rows)
                    return True
            for alert in row.get("local_alert_deliveries", []):
                if str(alert.get("_id")) == str(identifier):
                    alert["alert_delivery"] = dict(delivery)
                    row["updated_at"] = datetime.now(timezone.utc)
                    _write(rows)
                    return True
        return False


def claim_pending_alert(identifier, now):
    """Atomically claim a locally persisted alert to prevent concurrent sends."""
    with _lock:
        rows = _read()
        for row in rows:
            records = row.get("local_alert_deliveries", []) + row.get("local_events", [])
            target = next((event for event in records if str(event.get("_id")) == str(identifier)), None)
            if target and (target.get("alert_delivery") or {}).get("status") == "pending":
                delivery = dict(target.get("alert_delivery") or {})
                delivery.update({"status": "sending", "attempts": int(delivery.get("attempts", 0)) + 1,
                                 "last_attempt_at": now})
                for event in records:
                    if str(event.get("_id")) == str(identifier):
                        event["alert_delivery"] = dict(delivery)
                row["updated_at"] = now
                _write(rows)
                return dict(target)
        return None


def list_pending_alerts(limit=100):
    with _lock:
        rows = _read()
    result = {}
    for row in rows:
        for event in row.get("local_alert_deliveries", []) + row.get("local_events", []):
            if (event.get("superseded") is not True and
                    (event.get("alert_delivery") or {}).get("status") == "pending"):
                result[str(event.get("_id"))] = dict(event)
    return list(result.values())[:max(1, min(1000, int(limit)))]


def list_local_alert_records(video_id=None, limit=1000):
    with _lock:
        rows = _read()
    result = []
    for row in rows:
        if video_id and str(row.get("_id")) != str(video_id):
            continue
        result.extend(dict(event) for event in row.get("local_alert_deliveries", []))
    return result[:max(1, min(50000, int(limit)))]


def recover_local_alerts():
    """Put alerts interrupted by a process restart back into the pending queue."""
    with _lock:
        rows = _read()
        changed = False
        for row in rows:
            for event in row.get("local_alert_deliveries", []) + row.get("local_events", []):
                delivery = event.get("alert_delivery") or {}
                if delivery.get("status") == "sending":
                    delivery.update({"status": "pending", "failure_reason": "Delivery worker restarted; retrying."})
                    changed = True
        if changed:
            _write(rows)


def reserve_local_video_alert(identifier, notification, *, enabled, legacy_sent_count=0):
    """Atomically append one video-level notification and consume at most two slots."""
    with _lock:
        rows = _read()
        row = next((item for item in rows if item.get("_id") == str(identifier)), None)
        if row is None:
            return None, "missing_video"
        existing = row.setdefault("email_alert_notifications", [])
        if any(item.get("notification_id") == notification.get("notification_id") for item in existing):
            return None, "duplicate"
        reported = set(row.get("alert_reported_group_ids") or [])
        group_ids = list(notification.get("group_ids") or [])
        if any(group_id in reported for group_id in group_ids):
            return None, "duplicate"
        used = max(int(row.get("alert_email_slots_used", 0) or 0),
                   min(2, max(0, int(legacy_sent_count or 0))))
        if enabled and used >= 2:
            row["alert_email_slots_used"] = used
            row["alert_email_limit_reached"] = True
            row["alert_reported_group_ids"] = sorted(reported.union(group_ids))
            _write(rows)
            return None, "limit_reached"
        now = datetime.now(timezone.utc)
        record = {**notification, "video_id": str(identifier),
                  "status": "pending" if enabled else "disabled", "attempts": 0,
                  "queued_at": now, "last_attempt_at": None, "sent_at": None,
                  "failure_reason": None if enabled else "Email alerts are disabled in configuration."}
        existing.append(record)
        row["alert_reported_group_ids"] = sorted(reported.union(group_ids))
        row["alert_email_slots_used"] = used + (1 if enabled else 0)
        row["alert_email_limit_reached"] = False
        row["updated_at"] = now
        _write(rows)
        return record["notification_id"], record["status"]


def list_pending_local_video_alerts(video_id=None, limit=100):
    with _lock:
        rows = _read()
    output = []
    for row in rows:
        if video_id and str(row.get("_id")) != str(video_id):
            continue
        for notification in row.get("email_alert_notifications", []):
            if notification.get("status") == "pending":
                output.append(dict(notification))
    return output[:max(1, min(1000, int(limit)))]


def claim_local_video_alert(notification_id, now):
    with _lock:
        rows = _read()
        for row in rows:
            for notification in row.get("email_alert_notifications", []):
                if (notification.get("notification_id") == str(notification_id)
                        and notification.get("status") == "pending"):
                    notification.update({"status": "sending",
                        "attempts": int(notification.get("attempts", 0)) + 1,
                        "last_attempt_at": now,
                        "delivery_protocol_version": 2})
                    notification.pop("smtp_transmission_started_at", None)
                    row["updated_at"] = now
                    _write(rows)
                    return dict(notification)
        return None


def update_local_video_alert(notification_id, changes):
    with _lock:
        rows = _read()
        for row in rows:
            for notification in row.get("email_alert_notifications", []):
                if notification.get("notification_id") == str(notification_id):
                    notification.update(changes)
                    row["updated_at"] = datetime.now(timezone.utc)
                    _write(rows)
                    return True
        return False


def recover_local_video_alerts(stale_before=None, video_id=None):
    """Resolve stale sends; retry only v2 attempts known not to have reached SMTP."""
    with _lock:
        rows = _read()
        changed = False
        recovered_count = 0
        now = datetime.now(timezone.utc)
        for row in rows:
            if video_id and str(row.get("_id")) != str(video_id):
                continue
            for notification in row.get("email_alert_notifications", []):
                if notification.get("status") != "sending":
                    continue
                attempted_at = notification.get("last_attempt_at")
                if isinstance(attempted_at, str):
                    try:
                        attempted_at = datetime.fromisoformat(attempted_at.replace("Z", "+00:00"))
                    except ValueError:
                        attempted_at = None
                if isinstance(attempted_at, datetime) and attempted_at.tzinfo is None:
                    attempted_at = attempted_at.replace(tzinfo=timezone.utc)
                if stale_before and isinstance(attempted_at, datetime) and attempted_at > stale_before:
                    continue
                can_retry = (notification.get("delivery_protocol_version") == 2
                             and not notification.get("smtp_transmission_started_at"))
                status = "pending" if can_retry else "uncertain"
                reason = ("Delivery worker stopped before SMTP transmission; safe retry queued."
                          if can_retry else
                          "Delivery was interrupted after SMTP transmission may have begun; automatic resend was suppressed.")
                notification.update({"status": status, "recovered_at": now,
                                     "failure_reason": reason})
                changed = True
                recovered_count += 1
        if changed:
            _write(rows)
        return recovered_count


def list_pending_incidents(status="", start=None, end=None, limit=1000):
    """Return real reviewable events from local-only records."""
    excluded = {"object_detection", "zone_presence_observed", "zone_entry", "zone_exit", "zone_tracking_gap"}
    with _lock:
        rows = _read()
    incidents = []
    for row in rows:
        for event in row.get("local_events", []):
            if event.get("superseded") is True or event.get("event_type") in excluded:
                continue
            item = dict(event)
            item.setdefault("incident_status", "new")
            item.setdefault("review_notes", "")
            created = item.get("created_at")
            if isinstance(created, str):
                try:
                    created = datetime.fromisoformat(created.replace("Z", "+00:00"))
                except ValueError:
                    created = None
            if status and item.get("incident_status") != status:
                continue
            if start and (not isinstance(created, datetime) or created < start):
                continue
            if end and (not isinstance(created, datetime) or created >= end):
                continue
            incidents.append(item)
    incidents.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return incidents[:max(1, min(5000, int(limit)))]


def update_pending_incident(event_id, status, notes):
    """Update review fields on exactly one locally persisted incident event."""
    if status not in {"new", "reviewing", "resolved", "false_positive"}:
        return False
    now = datetime.now(timezone.utc)
    with _lock:
        rows = _read()
        for row in rows:
            for event in row.get("local_events", []):
                if str(event.get("_id")) == str(event_id) and event.get("superseded") is not True:
                    if event.get("event_type") in {"object_detection", "zone_presence_observed", "zone_entry", "zone_exit", "zone_tracking_gap"}:
                        return False
                    event.update({"incident_status": status, "review_notes": str(notes or "")[:4000],
                                  "review_updated_at": now})
                    row["updated_at"] = now
                    _write(rows)
                    return True
        return False


def save_pending_frames(identifier, kind, frames):
    if kind not in {"pose", "face"}:
        return 0
    # Match the bounded per-video frame query limits used by the Mongo backend.
    bounded = [dict(frame) for frame in (frames or [])[-1200:]]
    return update_pending_video(identifier, {f"local_{kind}_frames": bounded}) if get_pending_video(identifier) else 0


def get_pending_frames(identifier, kind, limit=1200):
    row = get_pending_video(identifier) or {}
    key = "local_pose_frames" if kind == "pose" else "local_face_frames"
    return (row.get(key) or [])[:max(1, min(2000, int(limit)))]


def retry_pending_video(identifier):
    """Create a fresh local job and clear prior attempt evidence atomically."""
    with _lock:
        rows = _read()
        row = next((item for item in rows if item.get("_id") == str(identifier)), None)
        if row is None or row.get("processing_status") not in {"failed", "interrupted"}:
            return None
        now, job_id = datetime.now(timezone.utc), str(ObjectId())
        row.update({"processing_status": "queued", "error_message": None,
                    "active_job_id": job_id, "local_events": [], "local_face_frames": [],
                    "local_pose_frames": [], "processed_filename": None, "event_count": 0,
                    "analysis_mode": "pending", "processing_duration": None,
                    "face_analysis": {"status": "queued"}, "pose_analysis": {"status": "queued"},
                    "object_analysis": {"status": "queued"}, "updated_at": now,
                    "local_job": {"_id": job_id, "video_id": str(identifier), "status": "queued",
                                  "progress_percent": 0, "created_at": now, "started_at": None,
                                  "completed_at": None, "error_message": None, "retry": True}})
        _write(rows)
        return job_id


def prepare_local_jobs_on_startup():
    """Recover legacy waiting records and identify queued jobs to submit.

    A thread cannot survive Flask restart. Jobs that were processing are marked
    interrupted for an explicit retry; never claim they completed.
    """
    ready = []
    with _lock:
        rows = _read()
        changed = False
        now = datetime.now(timezone.utc)
        for row in rows:
            job = row.get("local_job") or {}
            status = row.get("processing_status")
            if status == "processing":
                message = "The Flask process stopped while this local job was running. Retry it from Results."
                row.update({"processing_status": "interrupted", "error_message": message,
                            "updated_at": now})
                job.update({"status": "interrupted", "error_message": message, "completed_at": now})
                row["local_job"] = job
                changed = True
            elif status in {"waiting_for_mongodb", "queued"}:
                if not job.get("_id") or job.get("status") == "waiting_for_mongodb":
                    job = {"_id": str(ObjectId()), "video_id": row.get("_id"), "status": "queued",
                           "progress_percent": 0, "created_at": now, "started_at": None,
                           "completed_at": None, "error_message": None}
                    row["active_job_id"] = job["_id"]
                    row["local_job"] = job
                    changed = True
                if status == "waiting_for_mongodb":
                    row.update({"processing_status": "queued", "error_message": None,
                                "database_status": "local_only", "updated_at": now})
                    changed = True
                ready.append({"video_id": row.get("_id"), "job_id": job["_id"]})
        if changed:
            _write(rows)
    return ready


def fail_pending_video(identifier, message):
    update_pending_video(identifier, {"processing_status": "failed", "error_message": message,
                                      "local_job.status": "failed", "local_job.error_message": message,
                                      "local_job.completed_at": datetime.now(timezone.utc)})


def list_pending_videos(search="", status="", page=1, per_page=25):
    with _lock:
        rows = _read()
    if search:
        rows = [row for row in rows if search.lower() in row.get("original_filename", "").lower()]
    if status:
        rows = [row for row in rows if row.get("processing_status") == status]
    rows.sort(key=lambda row: row.get("upload_date", ""), reverse=True)
    return rows[(page - 1) * per_page:page * per_page], len(rows)


def delete_pending_video(identifier):
    with _lock:
        rows = _read()
        found = next((row for row in rows if row.get("_id") == str(identifier)), None)
        if found:
            _write([row for row in rows if row.get("_id") != str(identifier)])
        return found
