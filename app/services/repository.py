"""MongoDB persistence operations kept separate from Flask route handlers."""
import logging
import re
import time
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from bson import ObjectId
from bson.errors import InvalidId
from pymongo import DESCENDING
from pymongo import ReturnDocument
from pymongo.errors import ConfigurationError, ConnectionFailure, PyMongoError
from flask import current_app
from app.db import DatabaseUnavailable, _friendly_connection_error, get_database
from app.services.timestamps import event_video_seconds, utc_datetime_sort_key

log = logging.getLogger(__name__)


def object_id(value):
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        return None


def _run(fn):
    try:
        return fn(get_database())
    except DatabaseUnavailable:
        raise
    except PyMongoError as exc:
        # A server-side write rejection (for example MongoDB code 40 for
        # conflicting update paths) proves that MongoDB was reached. Do not
        # turn an application/schema error into a connectivity failure or
        # switch this record to local fallback.
        if isinstance(exc, (ConnectionFailure, ConfigurationError)):
            current_app.extensions["mongo_available"] = False
            message = _friendly_connection_error(exc)
            current_app.extensions["mongo_error"] = message
            current_app.extensions["mongo_retry_at"] = time.monotonic() + max(
                0.0, float(current_app.config.get("MONGO_RETRY_INTERVAL_SECONDS", 5)))
            log.warning("MongoDB persistence connection failed (%s, code=%s): %s",
                        type(exc).__name__, getattr(exc, "code", None), message)
            raise DatabaseUnavailable(message) from exc
        raise


def create_video(document):
    return _run(lambda db: db.videos.insert_one(document).inserted_id)


def find_video_by_upload_request(request_id):
    """Find a prior submission by its per-form idempotency key."""
    request_id = str(request_id or "")[:80]
    if not request_id:
        return None
    from app.services.local_fallback import find_pending_video_by_upload_request
    local = find_pending_video_by_upload_request(request_id)
    if local and local.get("local_only"):
        return local
    try:
        row = _run(lambda db: db.videos.find_one({"upload_request_id": request_id}))
    except DatabaseUnavailable:
        return local
    return row or local


def create_job(document):
    return _run(lambda db: db.processing_jobs.insert_one(document).inserted_id)


def update_job(job_id, changes):
    oid = object_id(job_id)
    if oid is None:
        return False
    from app.services.local_fallback import update_pending_job
    local_result = update_pending_job(str(job_id), changes)
    if local_result is not None:
        return local_result
    return _run(lambda db: current_job_update(db, oid, changes))


def claim_job(job_id):
    """Allow exactly one in-process worker to claim a queued Mongo job."""
    oid = object_id(job_id)
    if oid is None:
        return None
    from app.services.local_fallback import claim_pending_job, find_pending_video_by_job
    local_video = find_pending_video_by_job(str(job_id))
    if local_video and local_video.get("local_only"):
        return claim_pending_job(str(job_id))
    def claim(db):
        return db.processing_jobs.find_one_and_update(
            {"_id": oid, "status": "queued"},
            {"$set": {"status": "processing", "started_at": datetime.now(timezone.utc)}},
            return_document=True,
        )
    return _run(claim)


def current_job_update(db, oid, changes):
    query = {"_id": oid}
    if "status" in changes:
        target = changes["status"]
        transitions = {
            "queued": {"queued"},
            "processing": {"queued", "processing", "running"},
            "completed": {"queued", "processing", "running", "completed"},
            "failed": {"queued", "processing", "running", "failed"},
            "interrupted": {"queued", "processing", "running", "interrupted"},
        }
        allowed = transitions.get(target)
        if allowed is None:
            return False
        query["status"] = {"$in": [None, *sorted(allowed)]}
    result = db.processing_jobs.update_one(query, {"$set": changes})
    return result.matched_count == 1


def _analysis_settings_set_fields(changes):
    """Expand parent and dotted analysis-settings updates into conflict-free leaves.

    MongoDB rejects a single $set that contains both ``analysis_settings`` and
    ``analysis_settings.child``. Writing leaf paths also preserves pre-existing
    sibling settings in the document.
    """
    result = {}
    settings = changes.get("analysis_settings")
    nested_updates = [(key, value) for key, value in changes.items()
                      if key.startswith("analysis_settings.")]

    def add_leaf(path, value):
        # Resolve any scalar/dotted overlap deterministically. The later,
        # explicitly supplied path wins; no ancestor/descendant pair reaches
        # MongoDB in the same update.
        for existing in list(result):
            if existing == path or existing.startswith(path + ".") or path.startswith(existing + "."):
                result.pop(existing)
        result[path] = value

    def flatten(prefix, value):
        if isinstance(value, dict) and value:
            for key, child in value.items():
                flatten(f"{prefix}.{key}", child)
        else:
            add_leaf(prefix, value)

    if isinstance(settings, dict):
        if settings:
            for key, value in settings.items():
                flatten(f"analysis_settings.{key}", value)
        elif not nested_updates:
            # An empty mapping has no leaves to merge and is an explicit value.
            result["analysis_settings"] = {}

    for key, value in nested_updates:
        add_leaf(key, value)

    normalized = {key: value for key, value in changes.items()
                  if key != "analysis_settings" and not key.startswith("analysis_settings.")}
    normalized.update(result)
    return normalized


def get_job_for_video(video_id):
    oid = object_id(video_id)
    if oid is None:
        return None
    from app.services.local_fallback import get_pending_video
    local = get_pending_video(video_id)
    if local and local.get("local_only"):
        return local.get("local_job")
    try:
        return _run(lambda db: db.processing_jobs.find_one({"video_id": oid}, sort=[("created_at", -1)]))
    except DatabaseUnavailable:
        from app.services.local_fallback import get_pending_video
        row = get_pending_video(video_id)
        return (row or {}).get("local_job")


def video_is_processing(video_id):
    oid = object_id(video_id)
    if oid is None:
        return False
    from app.services.local_fallback import get_pending_video
    local = get_pending_video(video_id)
    if local and local.get("local_only"):
        return local.get("processing_status") in {"queued", "processing"}
    try:
        return _run(lambda db: db.processing_jobs.count_documents(
            {"video_id": oid, "status": {"$in": ["queued", "processing", "running"]}}) > 0)
    except DatabaseUnavailable:
        from app.services.local_fallback import get_pending_video
        row = get_pending_video(video_id)
        return bool(row and row.get("processing_status") in {"queued", "processing"})


def update_video(video_id, changes):
    oid = object_id(video_id)
    if oid is None:
        return False
    changes = {**_analysis_settings_set_fields(changes), "updated_at": datetime.now(timezone.utc)}
    from app.services.local_fallback import get_pending_video, update_pending_video
    local = get_pending_video(video_id)
    if local and local.get("local_only"):
        return update_pending_video(video_id, changes)
    def update(db):
        query = {"_id": oid}
        if "processing_status" in changes:
            target = changes["processing_status"]
            transitions = {
                "queued": {"queued", "failed", "interrupted", "waiting_for_mongodb"},
                "processing": {"queued", "processing", "waiting_for_mongodb"},
                "complete": {"queued", "processing", "complete"},
                "failed": {"queued", "processing", "waiting_for_mongodb", "failed"},
                "interrupted": {"queued", "processing", "interrupted"},
                "waiting_for_mongodb": {"queued", "waiting_for_mongodb"},
            }
            allowed = transitions.get(target)
            if allowed is None:
                return False
            query["processing_status"] = {"$in": [None, *sorted(allowed)]}
        result = db.videos.update_one(query, {"$set": changes})
        return result.matched_count == 1
    return _run(update)


def get_video(video_id):
    oid = object_id(video_id)
    if oid is None:
        return None
    from app.services.local_fallback import get_pending_video
    local = get_pending_video(video_id)
    if local and local.get("local_only"):
        return local
    try:
        row = _run(lambda db: db.videos.find_one({"_id": oid}))
    except DatabaseUnavailable:
        # A known ID that cannot be checked during a database outage is not a
        # missing record. Let callers return an unavailable response rather
        # than alternating between 200 and a misleading 404.
        fallback = get_pending_video(video_id)
        if fallback is not None:
            return fallback
        raise
    if row is None:
        row = get_pending_video(video_id)
    return row


def get_events(video_id, limit=2000):
    oid = object_id(video_id)
    if oid is None:
        return []
    limit = max(1, min(50000, int(limit)))
    from app.services.local_fallback import get_pending_events, get_pending_video
    local = get_pending_video(video_id)
    if local and local.get("local_only"):
        return get_pending_events(video_id, limit)
    try:
        return _run(lambda db: list(db.detection_events.find(
            {"video_id": oid, "superseded": {"$ne": True}}).sort("timestamp_seconds", 1).limit(limit)))
    except DatabaseUnavailable:
        return []


def save_events(video_id, job_id, events):
    """Persist this job's events in its owning backend; replace local retries."""
    oid = object_id(video_id)
    if oid is None or not events:
        return 0
    from app.services.local_fallback import get_pending_video, save_pending_events
    local = get_pending_video(video_id)
    if local and local.get("local_only"):
        now = datetime.now(timezone.utc)
        for event in events:
            event.setdefault("_id", ObjectId())
            event.setdefault("job_id", ObjectId(str(job_id)))
            event.setdefault("created_at", now)
            if event.get("event_type") in _ALERTABLE_EVENTS:
                event.setdefault("alert_delivery", _new_alert_delivery(video_id, event))
            if event.get("event_type") not in {"object_detection", "zone_presence_observed", "zone_entry", "zone_exit", "zone_tracking_gap"}:
                event.setdefault("incident_status", "new")
                event.setdefault("review_notes", "")
        return save_pending_events(video_id, events)
    now = datetime.now(timezone.utc)
    documents = []
    for event in events:
        item = dict(event)
        item.setdefault("_id", ObjectId())
        item.update({"video_id": oid, "job_id": object_id(job_id), "created_at": now})
        event.setdefault("_id", item["_id"])
        if item.get("event_type") in _ALERTABLE_EVENTS:
            item.setdefault("alert_delivery", _new_alert_delivery(video_id, item))
            event.setdefault("alert_delivery", item["alert_delivery"])
        if item.get("event_type") not in {"object_detection", "zone_presence_observed", "zone_entry", "zone_exit", "zone_tracking_gap"}:
            item.setdefault("incident_status", "new")
            item.setdefault("review_notes", "")
        documents.append(item)
    return _run(lambda db: len(db.detection_events.insert_many(documents).inserted_ids))


_ALERTABLE_EVENTS = {"RESTRICTED_ZONE_ENTRY", "potential_fall", "SUSPECTED_FIGHTING"}


def _new_alert_delivery(video_id, event):
    enabled = bool(current_app.config.get("EMAIL_ALERTS_ENABLED", False))
    # Events stay individually auditable, but SMTP work is reserved and sent at
    # the video level by the consolidated outbox below.
    return {"status": "included", "attempts": 0,
            "queued_at": datetime.now(timezone.utc), "last_attempt_at": None,
            "sent_at": None,
            "failure_reason": ("Email alerts are disabled in configuration." if not enabled else
                               "Individual event messages are suppressed; delivery is consolidated per video."),
            "dedupe_key": _alert_dedupe_key(video_id, event),
            "gif_filename": None, "gif_size_bytes": None}


def _alert_dedupe_key(video_id, event):
    identity = {
        "video": str(video_id), "type": event.get("event_type"),
        "start": round(float(event.get("start_timestamp_seconds", event.get("timestamp_seconds", 0)) or 0), 1),
        "end": round(float(event.get("end_timestamp_seconds", event.get("timestamp_seconds", 0)) or 0), 1),
        "zone": (event.get("details") or {}).get("zone_id"),
        "track": event.get("track_id"),
    }
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def initialize_alert_delivery(video_id, events):
    """Reserve one consolidated, durable video-level alert after event persistence."""
    oid = object_id(video_id)
    if oid is None:
        return []
    from app.services.local_fallback import (get_pending_video, get_pending_events,
        list_local_alert_records)
    local = get_pending_video(video_id)
    local_only = bool(local and local.get("local_only"))
    eligible = [dict(event) for event in events or []
                if event.get("event_type") in _ALERTABLE_EVENTS and event.get("_id") is not None]
    if not eligible:
        return []

    keys = {_alert_dedupe_key(video_id, event) for event in eligible}
    legacy_rows = list_local_alert_records(video_id, 50000) + get_pending_events(video_id, 50000) if local_only else []
    if local_only:
        sent_keys = {(row.get("alert_delivery") or {}).get("dedupe_key") for row in legacy_rows
                     if (row.get("alert_delivery") or {}).get("status") == "sent"}
        legacy_sent_count = len(sent_keys)
    else:
        def existing_sent(db):
            return list(db.detection_events.find({"video_id": oid,
                "alert_delivery.dedupe_key": {"$in": list(keys)},
                "alert_delivery.status": "sent"}, {"alert_delivery.dedupe_key": 1}))
        sent_rows = _run(existing_sent)
        sent_keys = {(row.get("alert_delivery") or {}).get("dedupe_key") for row in sent_rows}
        legacy_sent_count = min(2, _run(lambda db: db.detection_events.count_documents(
            {"video_id": oid, "alert_delivery.status": "sent"})))
    eligible = [event for event in eligible if _alert_dedupe_key(video_id, event) not in sent_keys]
    if not eligible:
        return []

    from app.services.event_groups import group_event_episodes
    from app.services.notifier import build_video_notification
    video = local if local_only else get_video(video_id)
    if not video:
        return []
    groups = group_event_episodes(eligible,
        current_app.config.get("ALERT_EPISODE_MERGE_GAP_SECONDS", 1.0))
    reported = set((video or {}).get("alert_reported_group_ids") or [])
    groups = [group for group in groups if group["group_id"] not in reported]
    if not groups:
        return []
    notification = build_video_notification(video_id, video, groups)
    enabled = bool(current_app.config.get("EMAIL_ALERTS_ENABLED", False))
    from app.services.local_fallback import reserve_local_video_alert
    if local_only:
        notification_id, _status = reserve_local_video_alert(video_id, notification,
            enabled=enabled, legacy_sent_count=legacy_sent_count)
    else:
        notification_id, _status = reserve_video_alert(video_id, notification,
            enabled=enabled, legacy_sent_count=legacy_sent_count)
    return [notification_id] if notification_id and enabled else []


def reserve_video_alert(video_id, notification, *, enabled, legacy_sent_count=0):
    """Atomically reserve an alert group and enforce the two-email video cap."""
    oid = object_id(video_id)
    if oid is None:
        return None, "missing_video"
    group_ids = list(notification.get("group_ids") or [])
    if not group_ids:
        return None, "empty"
    notification_id = str(notification["notification_id"])
    now = datetime.now(timezone.utc)
    def reserve(db):
        db.videos.update_one({"_id": oid}, {"$max": {
            "alert_email_slots_used": min(2, max(0, int(legacy_sent_count or 0)))}})
        record = {**notification, "video_id": oid,
                  "status": "pending" if enabled else "skipped", "attempts": 0,
                  "queued_at": now, "last_attempt_at": None, "sent_at": None,
                  "failure_reason": None if enabled else "Email alerts are disabled in configuration."}
        query = {"_id": oid, "alert_reported_group_ids": {"$nin": group_ids},
                 "email_alert_notifications.notification_id": {"$ne": notification_id}}
        update = {"$push": {"email_alert_notifications": record},
                  "$addToSet": {"alert_reported_group_ids": {"$each": group_ids}},
                  "$set": {"alert_email_limit_reached": False}}
        if enabled:
            query["$or"] = [{"alert_email_slots_used": {"$exists": False}},
                            {"alert_email_slots_used": {"$lt": 2}}]
            update["$inc"] = {"alert_email_slots_used": 1}
        updated = db.videos.find_one_and_update(query, update, return_document=ReturnDocument.AFTER)
        if updated:
            return notification_id, record["status"]
        video = db.videos.find_one({"_id": oid}, {"alert_reported_group_ids": 1,
            "alert_email_slots_used": 1})
        if video and int(video.get("alert_email_slots_used", 0) or 0) >= 2 and enabled:
            db.videos.update_one({"_id": oid}, {"$set": {"alert_email_limit_reached": True}})
            return None, "limit_reached"
        return None, "duplicate"
    return _run(reserve)


def list_pending_video_alerts(video_id=None, limit=100):
    oid = object_id(video_id) if video_id else None
    from app.services.local_fallback import list_pending_local_video_alerts
    local_rows = list_pending_local_video_alerts(str(video_id) if video_id else None, limit)
    query = {"email_alert_notifications.status": "pending"}
    if oid:
        query["_id"] = oid
    try:
        docs = _run(lambda db: list(db.videos.find(query, {"email_alert_notifications": 1}).limit(
            max(1, min(1000, int(limit))))))
    except DatabaseUnavailable:
        docs = []
    rows = []
    for doc in docs:
        for item in doc.get("email_alert_notifications", []):
            if item.get("status") == "pending":
                rows.append({**item, "video_id": doc["_id"]})
    return (rows + local_rows)[:max(1, min(1000, int(limit)))]


def claim_video_alert(notification_id):
    from app.services.local_fallback import claim_local_video_alert
    now = datetime.now(timezone.utc)
    local = claim_local_video_alert(str(notification_id), now)
    if local:
        return local
    def claim(db):
        doc = db.videos.find_one_and_update(
            {"email_alert_notifications.notification_id": str(notification_id),
             "email_alert_notifications.status": "pending"},
            {"$set": {"email_alert_notifications.$.status": "sending",
                      "email_alert_notifications.$.last_attempt_at": now,
                      "email_alert_notifications.$.delivery_protocol_version": 2},
             "$unset": {"email_alert_notifications.$.smtp_transmission_started_at": ""},
             "$inc": {"email_alert_notifications.$.attempts": 1}},
            return_document=ReturnDocument.AFTER)
        if not doc:
            return None
        item = next((entry for entry in doc.get("email_alert_notifications", [])
                     if entry.get("notification_id") == str(notification_id)), None)
        return {**(item or {}), "video_id": doc["_id"]} if item else None
    try:
        return _run(claim)
    except DatabaseUnavailable:
        return None


def update_video_alert(notification_id, changes):
    from app.services.local_fallback import update_local_video_alert
    if update_local_video_alert(str(notification_id), changes):
        return True
    flattened = {f"email_alert_notifications.$.{key}": value for key, value in changes.items()}
    return _run(lambda db: db.videos.update_one(
        {"email_alert_notifications.notification_id": str(notification_id)},
        {"$set": flattened}).matched_count == 1)


def recover_video_alerts(stale_after_seconds=900, video_id=None):
    """Resolve only stale sends; never auto-resend an ambiguous SMTP attempt.

    A v2 reservation with no durable transmission marker was interrupted while
    preparing evidence and is safe to return to pending. Any legacy or marked
    attempt may have reached SMTP and is moved to uncertain instead.
    """
    from app.services.local_fallback import recover_local_video_alerts
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=max(60, int(stale_after_seconds)))
    local_recovered = recover_local_video_alerts(cutoff, str(video_id) if video_id else None)
    oid = object_id(video_id) if video_id else None
    def recover(db):
        count = 0
        query = {"email_alert_notifications.status": "sending"}
        if oid:
            query["_id"] = oid
        docs = list(db.videos.find(query, {"email_alert_notifications": 1}))
        for doc in docs:
            for item in doc.get("email_alert_notifications", []):
                if item.get("status") != "sending":
                    continue
                attempted_at = item.get("last_attempt_at")
                if isinstance(attempted_at, datetime):
                    if attempted_at.tzinfo is None:
                        attempted_at = attempted_at.replace(tzinfo=timezone.utc)
                    if attempted_at > cutoff:
                        continue
                elif attempted_at:
                    try:
                        parsed = datetime.fromisoformat(str(attempted_at).replace("Z", "+00:00"))
                        if parsed.tzinfo is None:
                            parsed = parsed.replace(tzinfo=timezone.utc)
                        if parsed > cutoff:
                            continue
                    except ValueError:
                        pass
                can_retry = (item.get("delivery_protocol_version") == 2
                             and not item.get("smtp_transmission_started_at"))
                new_status = "pending" if can_retry else "uncertain"
                reason = ("Delivery worker stopped before SMTP transmission; safe retry queued."
                          if can_retry else
                          "Delivery was interrupted after SMTP transmission may have begun; automatic resend was suppressed.")
                attempt_match = ({"$in": [None]} if item.get("last_attempt_at") is None
                                 else item.get("last_attempt_at"))
                result = db.videos.update_one({"_id": doc["_id"],
                    "email_alert_notifications": {"$elemMatch": {
                        "notification_id": item.get("notification_id"), "status": "sending",
                        "last_attempt_at": attempt_match}}}, {"$set": {
                            "email_alert_notifications.$.status": new_status,
                            "email_alert_notifications.$.recovered_at": now,
                            "email_alert_notifications.$.failure_reason": reason}})
                count += result.modified_count
        return count
    try:
        return local_recovered + _run(recover)
    except DatabaseUnavailable:
        return local_recovered


def list_pending_alerts(video_id=None, limit=100):
    oid = object_id(video_id) if video_id else None
    from app.services.local_fallback import list_pending_alerts as list_local
    local_rows = list_local(limit)
    query = {"alert_delivery.status": "pending", "superseded": {"$ne": True}}
    if oid:
        query["video_id"] = oid
    try:
        rows = _run(lambda db: list(db.detection_events.find(query).limit(max(1, min(1000, int(limit))))))
    except DatabaseUnavailable:
        rows = []
    if video_id:
        local_rows = [row for row in local_rows if str(row.get("video_id")) == str(oid)]
    return (rows + local_rows)[:max(1, min(1000, int(limit)))]


def claim_alert_delivery(event_id):
    oid = object_id(event_id)
    if oid is None:
        return None
    now = datetime.now(timezone.utc)
    from app.services.local_fallback import claim_pending_alert
    # Local IDs are strings in JSON but BSON ObjectIds in event content.
    local_event = claim_pending_alert(str(event_id), now)
    if local_event:
        return local_event
    try:
        return _run(lambda db: db.detection_events.find_one_and_update(
            {"_id": oid, "alert_delivery.status": "pending", "superseded": {"$ne": True}},
            {"$set": {"alert_delivery.status": "sending",
                      "alert_delivery.last_attempt_at": now},
             "$inc": {"alert_delivery.attempts": 1}}, return_document=ReturnDocument.AFTER))
    except DatabaseUnavailable:
        return None


def update_alert_delivery(event_id, changes):
    oid = object_id(event_id)
    if oid is None:
        return False
    from app.services.local_fallback import update_pending_alert
    if update_pending_alert(str(event_id), changes):
        return True
    flattened = {f"alert_delivery.{key}": value for key, value in changes.items()}
    return _run(lambda db: db.detection_events.update_one({"_id": oid}, {"$set": flattened}).matched_count == 1)


def recover_alert_deliveries():
    now = datetime.now(timezone.utc)
    from app.services.local_fallback import recover_local_alerts
    recover_local_alerts()
    try:
        return _run(lambda db: db.detection_events.update_many(
            {"alert_delivery.status": "sending"},
            {"$set": {"alert_delivery.status": "pending",
                      "alert_delivery.failure_reason": "Delivery worker restarted; retrying.",
                      "alert_delivery.recovered_at": now}}).modified_count)
    except DatabaseUnavailable:
        return 0




def list_incidents(status="", start=None, end=None, limit=1000):
    allowed = {"new", "reviewing", "resolved", "false_positive"}
    query = {"event_type": {"$nin": ["object_detection", "zone_presence_observed", "zone_entry", "zone_exit", "zone_tracking_gap"]}, "superseded": {"$ne": True}}
    if status in allowed:
        query["incident_status"] = status
    if start or end:
        query["created_at"] = {}
        if start:
            query["created_at"]["$gte"] = start
        if end:
            query["created_at"]["$lt"] = end
    try:
        mongo_rows = _run(lambda db: list(db.detection_events.find(query).sort(
            [("created_at", DESCENDING), ("timestamp_seconds", 1)]).limit(max(1, min(5000, int(limit))))))
    except DatabaseUnavailable:
        mongo_rows = []
    from app.services.local_fallback import list_pending_incidents
    local_rows = list_pending_incidents(status, start, end, limit)
    rows = mongo_rows + local_rows
    rows.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return rows[:max(1, min(5000, int(limit)))]


def update_incident(event_id, status, notes):
    """Update review metadata on the existing candidate event (one-to-one)."""
    oid = object_id(event_id)
    if oid is None or status not in {"new", "reviewing", "resolved", "false_positive"}:
        return False
    notes = str(notes or "")[:4000]
    from app.services.local_fallback import update_pending_incident
    if update_pending_incident(str(event_id), status, notes):
        return True
    def update(db):
        event = db.detection_events.find_one({"_id": oid, "event_type": {"$nin": ["object_detection", "zone_presence_observed", "zone_entry", "zone_exit", "zone_tracking_gap"]},
                                              "superseded": {"$ne": True}},
                                             {"video_id": 1})
        if not event or not isinstance(event.get("video_id"), ObjectId):
            return False
        if db.videos.find_one({"_id": event["video_id"]}, {"_id": 1}) is None:
            return False
        result = db.detection_events.update_one(
            {"_id": oid, "event_type": {"$nin": ["object_detection", "zone_presence_observed", "zone_entry", "zone_exit", "zone_tracking_gap"]}, "superseded": {"$ne": True}},
            {"$set": {"incident_status": status, "review_notes": notes,
                       "review_updated_at": datetime.now(timezone.utc)}})
        return result.modified_count == 1 or result.matched_count == 1
    return _run(update)


def get_face_frames(video_id, limit=1200):
    oid = object_id(video_id)
    if oid is None:
        return []
    from app.services.local_fallback import get_pending_frames, get_pending_video
    local = get_pending_video(video_id)
    if local and local.get("local_only"):
        return get_pending_frames(video_id, "face", limit)
    try:
        return _run(lambda db: list(db.face_detection_frames.find({"video_id": oid, "superseded": {"$ne": True}},
            {"_id": 0}).sort("timestamp_seconds", 1).limit(max(1, min(2000, int(limit))))))
    except DatabaseUnavailable:
        return []


def save_face_frames(video_id, frames):
    oid = object_id(video_id)
    if oid is None or not frames:
        return 0
    from app.services.local_fallback import get_pending_video, save_pending_frames
    local = get_pending_video(video_id)
    if local and local.get("local_only"):
        return save_pending_frames(video_id, "face", frames)
    now = datetime.now(timezone.utc)
    documents = [{**frame, "video_id": oid, "created_at": now} for frame in frames]
    return _run(lambda db: len(db.face_detection_frames.insert_many(documents).inserted_ids))


def get_pose_frames(video_id, limit=1200):
    oid = object_id(video_id)
    if oid is None:
        return []
    from app.services.local_fallback import get_pending_frames, get_pending_video
    local = get_pending_video(video_id)
    if local and local.get("local_only"):
        return get_pending_frames(video_id, "pose", limit)
    try:
        return _run(lambda db: list(db.pose_detection_frames.find(
            {"video_id": oid, "superseded": {"$ne": True}}, {"_id": 0}
        ).sort("timestamp_seconds", 1).limit(max(1, min(2000, int(limit))))))
    except DatabaseUnavailable:
        return []


def save_pose_frames(video_id, frames):
    oid = object_id(video_id)
    if oid is None or not frames:
        return 0
    from app.services.local_fallback import get_pending_video, save_pending_frames
    local = get_pending_video(video_id)
    if local and local.get("local_only"):
        return save_pending_frames(video_id, "pose", frames)
    now = datetime.now(timezone.utc)
    documents = [{**frame, "video_id": oid, "created_at": now} for frame in frames]
    return _run(lambda db: len(db.pose_detection_frames.insert_many(documents).inserted_ids))


def queue_video_retry(video_id):
    """Atomically claim a failed/interrupted video for one retry worker."""
    oid = object_id(video_id)
    if oid is None:
        return False
    from app.services.local_fallback import get_pending_video, retry_pending_video
    local = get_pending_video(video_id)
    if local and local.get("local_only"):
        return retry_pending_video(video_id) is not None
    def claim(db):
        result = db.videos.update_one(
            {"_id": oid, "processing_status": {"$in": ["failed", "interrupted"]}},
            {"$set": {"processing_status": "queued", "error_message": None,
                       "face_analysis": {"status": "queued", "method": "OpenCV Haar cascade (haarcascade_frontalface_default)"},
                       "pose_analysis": {"status": "queued"}, "object_analysis": {"status": "queued"},
                       "event_count": 0, "analysis_mode": "pending", "detection_method": None,
                       "processing_duration": None, "processed_filename": None,
                       "updated_at": datetime.now(timezone.utc)}})
        if result.modified_count != 1:
            return False
        # Preserve prior actual results but mark them as superseded so a retry's
        # result page and analytics do not combine multiple attempts.
        db.detection_events.update_many({"video_id": oid, "superseded": {"$ne": True}},
            {"$set": {"superseded": True, "superseded_at": datetime.now(timezone.utc)}})
        db.face_detection_frames.update_many({"video_id": oid, "superseded": {"$ne": True}},
            {"$set": {"superseded": True, "superseded_at": datetime.now(timezone.utc)}})
        db.pose_detection_frames.update_many({"video_id": oid, "superseded": {"$ne": True}},
            {"$set": {"superseded": True, "superseded_at": datetime.now(timezone.utc)}})
        return True
    return _run(claim)


def mark_incomplete_jobs_interrupted():
    """Mark work left by a prior local Flask process; no worker survives restart."""
    now = datetime.now(timezone.utc)
    def recover(db):
        active_ids = [row["_id"] for row in db.videos.find(
            {"processing_status": {"$in": ["queued", "processing"]}}, {"_id": 1})]
        if not active_ids:
            return 0, 0
        jobs = db.processing_jobs.update_many(
            {"video_id": {"$in": active_ids}, "status": {"$in": ["queued", "processing", "running"]}},
            {"$set": {"status": "interrupted", "error_message": "The Flask process stopped before this job completed.",
                       "completed_at": now}})
        videos = db.videos.update_many(
            {"_id": {"$in": active_ids}, "processing_status": {"$in": ["queued", "processing"]}},
            {"$set": {"processing_status": "interrupted",
                       "error_message": "The Flask process stopped before this job completed.",
                       "pose_analysis.status": "interrupted",
                       "face_analysis.status": "interrupted",
                       "object_analysis.status": "interrupted",
                       "updated_at": now}})
        return jobs.modified_count, videos.modified_count
    return _run(recover)


def dashboard_stats():
    def query(db):
        videos = db.videos
        events = db.detection_events
        total = videos.count_documents({})
        completed = videos.count_documents({"processing_status": "complete"})
        failed = videos.count_documents({"processing_status": "failed"})
        active = videos.count_documents({"processing_status": {"$in": ["queued", "processing"]}})
        # Dashboard anomaly metrics intentionally exclude ordinary object
        # observations. Those are shown separately on per-video results.
        active_events = {"superseded": {"$ne": True}, "event_type": {"$nin": ["object_detection", "zone_presence_observed", "zone_entry", "zone_exit", "zone_tracking_gap", "RESTRICTED_ZONE_ENTRY"]}}
        category_events = {"superseded": {"$ne": True}, "event_type": {"$nin": ["object_detection", "zone_presence_observed", "zone_entry", "zone_exit", "zone_tracking_gap"]}}
        event_feed = {"superseded": {"$ne": True}, "event_type": {"$nin": ["object_detection", "zone_presence_observed", "zone_entry", "zone_exit", "zone_tracking_gap"]}}
        event_count = events.count_documents(active_events)
        object_query = {"superseded": {"$ne": True}, "event_type": "object_detection"}
        object_observations = events.count_documents(object_query)
        object_categories = list(events.aggregate([
            {"$match": object_query},
            {"$group": {"_id": "$label", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]))
        track_groups = list(events.aggregate([
            {"$match": {**object_query, "label": "person", "track_id": {"$ne": None}}},
            {"$group": {"_id": {"video_id": "$video_id", "track_id": "$track_id"}}},
            {"$limit": 10001},
        ]))
        face_summary = list(videos.aggregate([
            {"$group": {"_id": None,
                        "completed": {"$sum": {"$cond": [{"$eq": ["$face_analysis.status", "complete"]}, 1, 0]}},
                        "observations": {"$sum": {"$ifNull": ["$face_analysis.face_observations", 0]}}}}
        ]))
        face_summary = face_summary[0] if face_summary else {"completed": 0, "observations": 0}
        categories = list(events.aggregate([{"$match": category_events},
            {"$group": {"_id": "$event_type", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}]))
        zone_entry_count = events.count_documents({"superseded": {"$ne": True}, "event_type": "RESTRICTED_ZONE_ENTRY"})
        suspected_fighting_count = events.count_documents({"superseded": {"$ne": True}, "event_type": "SUSPECTED_FIGHTING"})
        if zone_entry_count:
            categories.append({"_id": "RESTRICTED_ZONE_ENTRY", "count": zone_entry_count})
        recent = list(events.aggregate([
            {"$match": event_feed}, {"$sort": {"created_at": -1}}, {"$limit": 8},
            {"$lookup": {"from": "videos", "localField": "video_id", "foreignField": "_id", "as": "video"}},
            {"$unwind": {"path": "$video", "preserveNullAndEmptyArrays": True}},
        ]))
        recent_videos = list(videos.find({}, {"original_filename": 1, "upload_date": 1,
            "processing_status": 1, "duration": 1, "processing_duration": 1}).sort("upload_date", DESCENDING).limit(6))
        return {"videos": total, "done": completed, "failed": failed, "active": active, "events": event_count,
                "restricted_zone_entries": zone_entry_count,
                "suspected_fighting_events": suspected_fighting_count,
                "queued": videos.count_documents({"processing_status": "queued"}),
                "processing": videos.count_documents({"processing_status": "processing"}),
                "object_observations": object_observations,
                "object_categories": [{"label": row["_id"] or "Unknown", "count": row["count"]}
                                      for row in object_categories],
                "person_track_estimate": len(track_groups),
                "person_track_estimate_capped": len(track_groups) > 10000,
                "face_videos": face_summary.get("completed", 0),
                "face_observations": face_summary.get("observations", 0),
                "categories": [{"category": row["_id"] or "Uncategorized", "count": row["count"]} for row in categories],
                "recent_videos": recent_videos,
                "recent": [{**row, "original_filename": (row.get("video") or {}).get("original_filename", "Deleted video")} for row in recent]}
    try:
        result = _run(query)
    except DatabaseUnavailable:
        result = {"videos": 0, "done": 0, "failed": 0, "active": 0, "events": 0,
                  "restricted_zone_entries": 0, "suspected_fighting_events": 0,
                  "queued": 0, "processing": 0, "object_observations": 0, "object_categories": [],
                  "person_track_estimate": 0, "person_track_estimate_capped": False,
                  "face_videos": 0, "face_observations": 0, "categories": [],
                  "recent_videos": [], "recent": []}
    from app.services.local_fallback import list_pending_videos
    local, _ = list_pending_videos(per_page=5000)
    result["videos"] += len(local)
    local_events = [event for row in local for event in row.get("local_events", [])
                    if event.get("superseded") is not True]
    result["done"] += sum(row.get("processing_status") == "complete" for row in local)
    result["failed"] += sum(row.get("processing_status") in {"failed", "interrupted"} for row in local)
    result["queued"] += sum(row.get("processing_status") == "queued" for row in local)
    result["processing"] += sum(row.get("processing_status") == "processing" for row in local)
    result["active"] += sum(row.get("processing_status") in {"waiting_for_mongodb", "queued", "processing"}
                             for row in local)
    excluded = {"object_detection", "zone_presence_observed", "zone_entry", "zone_exit", "zone_tracking_gap", "RESTRICTED_ZONE_ENTRY"}
    anomaly_events = [event for event in local_events if event.get("event_type") not in excluded]
    result["events"] += len(anomaly_events)
    result["restricted_zone_entries"] += sum(event.get("event_type") == "RESTRICTED_ZONE_ENTRY" for event in local_events)
    result["suspected_fighting_events"] += sum(event.get("event_type") == "SUSPECTED_FIGHTING" for event in local_events)
    local_categories = Counter(event.get("event_type", "Uncategorized") for event in anomaly_events)
    local_entry_count = sum(event.get("event_type") == "RESTRICTED_ZONE_ENTRY" for event in local_events)
    if local_entry_count:
        local_categories["RESTRICTED_ZONE_ENTRY"] += local_entry_count
    result["categories"].extend({"category": category, "count": count}
                                for category, count in local_categories.items())
    object_events = [event for event in local_events if event.get("event_type") == "object_detection"]
    result["object_observations"] += len(object_events)
    object_counts = Counter(event.get("label", "Unknown") for event in object_events)
    result["object_categories"].extend({"label": label, "count": count}
                                       for label, count in object_counts.items())
    result["face_videos"] += sum((row.get("face_analysis") or {}).get("status") == "complete" for row in local)
    result["face_observations"] += sum((row.get("face_analysis") or {}).get("face_observations", 0) for row in local)
    result["person_track_estimate"] += sum((row.get("pose_analysis") or {}).get("person_track_count_estimate", 0) for row in local)
    for row in local:
        candidate_count = sum(event.get("event_type") not in excluded for event in row.get("local_events", []))
        row["event_count"] = candidate_count
    result["recent"] = list(result["recent"])
    result["recent"].extend({**event, "original_filename": row.get("original_filename", "Local video")}
                            for row in local for event in row.get("local_events", [])
                        if event.get("superseded") is not True and event.get("event_type") not in {"object_detection", "zone_presence_observed", "zone_entry", "zone_exit", "zone_tracking_gap"})
    result["recent"].sort(key=lambda event: str(event.get("created_at", "")), reverse=True)
    result["recent"] = result["recent"][:8]
    result["recent_videos"] = (result["recent_videos"] + local)[:6]
    return result


def list_videos(search="", status="", page=1, per_page=25):
    query = {}
    if search:
        query["original_filename"] = {"$regex": re.escape(search[:100]), "$options": "i"}
    if status in {"complete", "failed", "processing", "queued", "interrupted", "waiting_for_mongodb"}:
        query["processing_status"] = status
    page = max(1, min(1000, int(page)))
    def find(db):
        coll = db.videos
        total = coll.count_documents(query)
        # Fetch only the prefix needed for this merged Mongo/local page. The
        # records are sorted together below before applying the page offset.
        docs = list(coll.find(query).sort("upload_date", DESCENDING).limit(page * per_page))
        event_counts = {row["_id"]: row["count"] for row in db.detection_events.aggregate([
            {"$match": {"video_id": {"$in": [d["_id"] for d in docs]},
                        "superseded": {"$ne": True}, "event_type": {"$nin": ["object_detection", "zone_presence_observed", "zone_entry", "zone_exit", "zone_tracking_gap", "RESTRICTED_ZONE_ENTRY"]}}},
            {"$group": {"_id": "$video_id", "count": {"$sum": 1}}},
        ])} if docs else {}
        entry_counts = {row["_id"]: row["count"] for row in db.detection_events.aggregate([
            {"$match": {"video_id": {"$in": [d["_id"] for d in docs]},
                        "superseded": {"$ne": True}, "event_type": "RESTRICTED_ZONE_ENTRY"}},
            {"$group": {"_id": "$video_id", "count": {"$sum": 1}}},
        ])} if docs else {}
        latest_entries = {}
        if docs:
            for event in db.detection_events.find(
                    {"video_id": {"$in": [d["_id"] for d in docs]},
                     "superseded": {"$ne": True}, "event_type": "RESTRICTED_ZONE_ENTRY"},
                    {"video_id": 1, "timestamp_seconds": 1, "video_position_seconds": 1,
                     "timestamp": 1, "video_timestamp": 1, "zone_id": 1, "zone_name": 1,
                     "details": 1, "track_id": 1}).sort("timestamp_seconds", -1):
                latest_entries.setdefault(event["video_id"], event)
        fighting_counts = {row["_id"]: row["count"] for row in db.detection_events.aggregate([
            {"$match": {"video_id": {"$in": [d["_id"] for d in docs]},
                        "superseded": {"$ne": True}, "event_type": "SUSPECTED_FIGHTING"}},
            {"$group": {"_id": "$video_id", "count": {"$sum": 1}}},
        ])} if docs else {}
        event_type_summaries = defaultdict(list)
        if docs:
            for row in db.detection_events.aggregate([
                {"$match": {"video_id": {"$in": [doc["_id"] for doc in docs]},
                            "superseded": {"$ne": True}, "event_type": {"$ne": "object_detection"}}},
                {"$group": {"_id": {"video_id": "$video_id",
                                      "event_type": {"$ifNull": ["$event_type", "unknown"]},
                                      "review_status": {"$ifNull": ["$incident_status", "not reviewed"]}},
                            "count": {"$sum": 1}}},
                {"$sort": {"_id.event_type": 1, "_id.review_status": 1}},
            ]):
                event_type_summaries[row["_id"]["video_id"]].append({
                    "event_type": row["_id"]["event_type"],
                    "review_status": row["_id"]["review_status"], "count": row["count"]})
        for doc in docs:
            doc["event_count"] = event_counts.get(doc["_id"], 0)
            doc["restricted_zone_entry_count"] = entry_counts.get(doc["_id"], 0)
            doc["latest_restricted_zone_entry"] = latest_entries.get(doc["_id"])
            doc["suspected_fighting_count"] = fighting_counts.get(doc["_id"], 0)
            doc["event_type_summary"] = event_type_summaries.get(doc["_id"], [])
        return docs, total
    try:
        docs, total = _run(find)
    except DatabaseUnavailable:
        docs, total = [], 0
    from app.services.local_fallback import list_pending_videos
    local, local_total = list_pending_videos(search, status, 1, page * per_page)
    for row in local:
        excluded = {"object_detection", "zone_presence_observed", "zone_entry", "zone_exit", "zone_tracking_gap", "RESTRICTED_ZONE_ENTRY"}
        row["event_count"] = sum(event.get("event_type") not in excluded
                                  for event in row.get("local_events", []))
        local_summary = Counter(
            (str(event.get("event_type") or "unknown"),
             str(event.get("incident_status") or "not reviewed"))
            for event in row.get("local_events", [])
            if event.get("superseded") is not True and event.get("event_type") != "object_detection")
        row["event_type_summary"] = [{"event_type": event_type,
                                      "review_status": review_status, "count": count}
                                     for (event_type, review_status), count in sorted(local_summary.items())]
        row["restricted_zone_entry_count"] = sum(event.get("event_type") == "RESTRICTED_ZONE_ENTRY"
                                                  and event.get("superseded") is not True
                                                  for event in row.get("local_events", []))
        entries = [event for event in row.get("local_events", [])
                   if event.get("event_type") == "RESTRICTED_ZONE_ENTRY" and event.get("superseded") is not True]
        entries.sort(key=lambda event: (event_video_seconds(event)
                                        if event_video_seconds(event) is not None else -1), reverse=True)
        row["latest_restricted_zone_entry"] = entries[0] if entries else None
        row["suspected_fighting_count"] = sum(event.get("event_type") == "SUSPECTED_FIGHTING"
                                               and event.get("superseded") is not True
                                               for event in row.get("local_events", []))
    merged = docs + local
    merged.sort(key=lambda row: (utc_datetime_sort_key(row.get("upload_date")),
                                 str(row.get("_id", ""))), reverse=True)
    offset = (page - 1) * per_page
    return merged[offset:offset + per_page], total + local_total


def delete_video(video_id):
    oid = object_id(video_id)
    if oid is None:
        return None
    def delete(db):
        video = db.videos.find_one({"_id": oid})
        if video is None:
            return None
        db.detection_events.delete_many({"video_id": oid})
        db.face_detection_frames.delete_many({"video_id": oid})
        db.pose_detection_frames.delete_many({"video_id": oid})
        db.processing_jobs.delete_many({"video_id": oid})
        db.videos.delete_one({"_id": oid})
        return video
    try:
        row = _run(delete)
    except DatabaseUnavailable:
        row = None
    if row is None:
        from app.services.local_fallback import delete_pending_video
        row = delete_pending_video(video_id)
    return row


def report_data(start=None, end=None):
    query = {}
    date_filter = {}
    if start:
        date_filter["$gte"] = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end:
        date_filter["$lt"] = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).replace(tzinfo=timezone.utc)
    if date_filter:
        query["upload_date"] = date_filter
    def find(db):
        videos = list(db.videos.find(query).sort("upload_date", DESCENDING).limit(5000))
        ids = [v["_id"] for v in videos]
        events = list(db.detection_events.find({"video_id": {"$in": ids}, "superseded": {"$ne": True}}).sort("timestamp_seconds", 1)) if ids else []
        names = {v["_id"]: v.get("original_filename", "") for v in videos}
        for event in events:
            event["original_filename"] = names.get(event["video_id"], "")
        return videos, events
    try:
        videos, events = _run(find)
    except DatabaseUnavailable:
        videos, events = [], []
    from app.services.local_fallback import list_pending_videos
    local, _ = list_pending_videos(page=1, per_page=5000)
    def in_range(row):
        value = row.get("upload_date")
        if isinstance(value, datetime):
            value = value.date().isoformat()
        elif isinstance(value, str):
            value = value[:10]
        else:
            return not (start or end)
        return (not start or value >= start) and (not end or value <= end)
    local = [row for row in local if in_range(row)]
    videos.extend(local)
    events.extend(event for row in local for event in row.get("local_events", [])
                  if event.get("superseded") is not True)
    local_names = {str(row.get("_id")): row.get("original_filename", "") for row in local}
    for event in events:
        event["original_filename"] = event.get("original_filename") or local_names.get(str(event.get("video_id")), "")
    return videos, events
