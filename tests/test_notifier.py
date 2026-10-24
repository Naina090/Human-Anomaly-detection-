import io
import multiprocessing
import os
import socketserver
import threading
import time
from datetime import datetime, timedelta, timezone

import cv2
import mongomock
import pytest
from PIL import Image

from app import create_app
from app.services import notifier
from app.services.local_fallback import (get_pending_events, recover_local_alerts,
    save_pending_upload, update_pending_alert, get_pending_video, recover_local_video_alerts)
from app.services.repository import (initialize_alert_delivery, save_events,
    list_pending_alerts, update_alert_delivery, claim_alert_delivery)
from app.services.repository import (list_pending_video_alerts, claim_video_alert,
    update_video_alert, reserve_video_alert)
from app.services.event_groups import group_event_episodes


def _claim_local_alert_in_process(fallback_path, notification_id, result_queue):
    from datetime import datetime, timezone
    from flask import Flask
    from app.services.local_fallback import claim_local_video_alert
    app = Flask("sentinel-alert-lock-test")
    app.config["LOCAL_FALLBACK_PATH"] = fallback_path
    with app.app_context():
        result_queue.put(claim_local_video_alert(notification_id, datetime.now(timezone.utc)) is not None)


class _SMTPHandler(socketserver.StreamRequestHandler):
    def handle(self):
        self.wfile.write(b"220 localhost test smtp\r\n")
        while True:
            line = self.rfile.readline()
            if not line:
                break
            command = line.decode("ascii", "ignore").strip().upper()
            if command.startswith("EHLO") or command.startswith("HELO"):
                self.wfile.write(b"250-localhost\r\n250 SIZE 10000000\r\n")
            elif command.startswith(("MAIL FROM", "RCPT TO", "RSET", "NOOP")):
                self.wfile.write(b"250 ok\r\n")
            elif command == "DATA":
                self.wfile.write(b"354 send data\r\n")
                chunks = []
                while True:
                    part = self.rfile.readline()
                    if part in (b".\r\n", b".\n", b""):
                        break
                    chunks.append(part)
                self.server.messages.append(b"".join(chunks))
                self.wfile.write(b"250 accepted\r\n")
            elif command == "QUIT":
                self.wfile.write(b"221 bye\r\n")
                break
            else:
                self.wfile.write(b"250 ok\r\n")


class _SMTPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


@pytest.fixture()
def smtp_sink():
    server = _SMTPServer(("127.0.0.1", 0), _SMTPHandler)
    server.messages = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], server.messages
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture()
def alert_app(tmp_path):
    mongo = mongomock.MongoClient()
    app = create_app({"TESTING": True, "SECRET_KEY": "test", "MONGO_CLIENT": mongo,
        "MONGO_DB_NAME": "alert_tests", "MONGO_PING_ON_STARTUP": False,
        "UPLOAD_DIR": str(tmp_path / "uploads"), "PROCESSED_DIR": str(tmp_path / "processed"),
        "LOCAL_FALLBACK_PATH": str(tmp_path / "local.json"),
        "ALERT_EVIDENCE_DIR": str(tmp_path / "evidence"), "EMAIL_ALERTS_ENABLED": False,
        "SMTP_HOST": "", "SMTP_PORT": 587, "SMTP_USERNAME": "", "SMTP_PASSWORD": "",
        "ALERT_SENDER_EMAIL": "", "ALERT_RECIPIENT_EMAILS": []})
    yield app
    app.extensions["alert_executor"].shutdown(wait=True, cancel_futures=True)
    app.extensions["processing_executor"].shutdown(wait=True, cancel_futures=True)
    mongo.close()


def _make_clip(path, seconds=2.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (96, 64))
    assert writer.isOpened()
    for index in range(int(seconds * 20)):
        frame = (index * 5 % 180) * __import__("numpy").ones((64, 96, 3), dtype="uint8")
        cv2.circle(frame, (10 + index % 70, 30), 8, (255, 20, 20), -1)
        writer.write(frame)
    writer.release()
    return path


def _local_video(app, tmp_path, *, seconds=2.0):
    from bson import ObjectId
    clip = _make_clip(tmp_path / "mail-clip.avi", seconds)
    os.makedirs(app.config["UPLOAD_DIR"], exist_ok=True)
    stored = "a" * 32 + ".avi"
    target = os.path.join(app.config["UPLOAD_DIR"], stored)
    os.replace(clip, target)
    with app.app_context():
        video_id, job_id = save_pending_upload({"original_filename": "mail-clip.avi",
            "stored_filename": stored, "file_path": stored, "file_size": os.path.getsize(target),
            "zones": []}, "test fallback")
        event = {"event_type": "RESTRICTED_ZONE_ENTRY", "label": "Entry candidate",
            "timestamp_seconds": 0.1, "track_id": 7,
            "details": {"zone_id": "zone-1", "zone_name": "North gate"},
            "detection_method": "Test detector"}
        assert save_events(video_id, job_id, [event]) == 1
        initialize_alert_delivery(video_id, [event])
    return video_id, event["_id"], target


def _configure_smtp(app, port):
    app.config.update({"EMAIL_ALERTS_ENABLED": True, "SMTP_HOST": "127.0.0.1", "SMTP_PORT": port,
        "SMTP_USERNAME": "", "SMTP_PASSWORD": "", "SMTP_USE_TLS": False,
        "ALERT_SENDER_EMAIL": "sentinel@example.test",
        "ALERT_RECIPIENT_EMAILS": ["reviewer@example.test"],
        "ALERT_SMTP_TIMEOUT_SECONDS": 3, "ALERT_MAX_GIF_BYTES": 1024 * 1024,
        "ALERT_GIF_WIDTH": 160})


def _local_notification(app, video_id, notification_id=None):
    with app.app_context():
        row = get_pending_video(video_id) or {}
        return next((item for item in row.get("email_alert_notifications", [])
                     if notification_id is None or item.get("notification_id") == notification_id), None)


def test_alerts_disabled_and_missing_smtp_credentials_are_clear(alert_app, tmp_path):
    assert notifier.smtp_configuration(alert_app.config) == (False, "Email alerts are disabled in configuration.")
    video_id, _event_id, _source = _local_video(alert_app, tmp_path)
    assert _local_notification(alert_app, video_id)["status"] == "skipped"
    alert_app.config.update({"EMAIL_ALERTS_ENABLED": True})
    ready, reason = notifier.smtp_configuration(alert_app.config)
    assert not ready and "SMTP host" in reason
    with pytest.raises(notifier.AlertConfigurationError):
        notifier.send_test_email(alert_app)


def test_gif_is_short_aspect_preserving_and_clips_at_video_boundaries(alert_app, tmp_path):
    clip = _make_clip(tmp_path / "short.avi", 2.0)
    path = tmp_path / "evidence.gif"
    result = notifier.generate_event_gif(str(clip), {"timestamp_seconds": 1.95}, str(path), max_bytes=500_000)
    assert result["duration_seconds"] <= 3 and result["size_bytes"] <= 500_000
    with Image.open(path) as gif:
        assert gif.format == "GIF" and gif.n_frames > 1
        assert gif.width <= 320 and gif.height < gif.width


def test_gif_uses_approximately_three_seconds_when_video_has_enough_context(tmp_path):
    clip = _make_clip(tmp_path / "longer.avi", 5.0)
    path = tmp_path / "three-seconds.gif"
    result = notifier.generate_event_gif(str(clip), {"timestamp_seconds": 2.5}, str(path), max_bytes=500_000)
    with Image.open(path) as gif:
        durations = []
        for index in range(gif.n_frames):
            gif.seek(index)
            durations.append(gif.info.get("duration", 0))
    assert 2.9 <= sum(durations) / 1000 <= 3.2
    assert result["size_bytes"] <= 500_000


def test_gif_fails_safely_for_missing_timestamp_and_oversize_budget(tmp_path):
    clip = _make_clip(tmp_path / "clip.avi", 1.0)
    with pytest.raises(ValueError, match="timestamp"):
        notifier.generate_event_gif(str(clip), {}, str(tmp_path / "no-time.gif"))
    with pytest.raises(ValueError, match="size limit"):
        notifier.generate_event_gif(str(clip), {"timestamp_seconds": 0.2},
                                    str(tmp_path / "tiny-budget.gif"), max_bytes=1)


def test_local_fallback_consolidated_delivery_with_one_gif(alert_app, tmp_path, smtp_sink):
    port, messages = smtp_sink
    _configure_smtp(alert_app, port)
    video_id, _event_id, _ = _local_video(alert_app, tmp_path)
    notification_id = _local_notification(alert_app, video_id)["notification_id"]
    with alert_app.app_context():
        notifier.schedule_pending_alerts(alert_app, video_id)
    deadline = time.time() + 8
    delivery = None
    while time.time() < deadline:
        delivery = _local_notification(alert_app, video_id, notification_id)
        if delivery and delivery.get("status") in {"sent", "failed"}:
            break
        time.sleep(.05)
    assert delivery["status"] == "sent"
    assert delivery["gif_size_bytes"] <= alert_app.config["ALERT_MAX_GIF_BYTES"]
    assert delivery["gif_duration_seconds"] <= 3
    assert len(messages) == 1
    assert b"sentinel-evidence.gif" in messages[0]
    assert b"Automated signals require human review" in messages[0]
    assert b"Analysis identifier" in messages[0]
    assert b"North gate" in messages[0]


def test_smtp_acceptance_retries_only_status_persistence_without_resending(
        alert_app, tmp_path, smtp_sink, monkeypatch):
    from app.services import repository
    port, messages = smtp_sink
    _configure_smtp(alert_app, port)
    video_id, _event_id, _source = _local_video(alert_app, tmp_path)
    notification_id = _local_notification(alert_app, video_id)["notification_id"]
    real_update = repository.update_video_alert
    transient = {"remaining": 1}

    def flaky_status_write(identifier, changes):
        if changes.get("status") == "sent" and transient["remaining"]:
            transient["remaining"] -= 1
            raise RuntimeError("temporary persistence failure")
        return real_update(identifier, changes)

    monkeypatch.setattr(repository, "update_video_alert", flaky_status_write)
    assert notifier._deliver_one(alert_app, notification_id)
    saved = _local_notification(alert_app, video_id, notification_id)
    assert saved["status"] == "sent"
    assert transient["remaining"] == 0
    assert len(messages) == 1  # retrying the database write never resends SMTP


def test_test_email_uses_loopback_smtp_without_incident(alert_app, smtp_sink):
    port, messages = smtp_sink
    _configure_smtp(alert_app, port)
    assert notifier.send_test_email(alert_app)
    assert len(messages) == 1 and b"configuration test" in messages[0].lower()


def test_smtp_auth_and_timeout_failures_are_sanitized(monkeypatch, alert_app):
    import smtplib
    assert "authentication was rejected" in notifier._failure_reason(smtplib.SMTPAuthenticationError(535, b"bad"))
    assert "timeout" in notifier._failure_reason(TimeoutError("timeout"))
    assert "password" not in notifier._failure_reason(TimeoutError("sensitive-password"))


def test_smtp_auth_failure_persists_failed_video_notification(alert_app, tmp_path, monkeypatch):
    import smtplib
    _configure_smtp(alert_app, 2525)
    alert_app.config.update({"SMTP_USERNAME": "unit-user", "SMTP_PASSWORD": "unit-password"})
    video_id, _event_id, _ = _local_video(alert_app, tmp_path)
    notification_id = _local_notification(alert_app, video_id)["notification_id"]

    class RejectAuth:
        def __init__(self, *_args, **_kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def ehlo(self): pass
        def login(self, *_args): raise smtplib.SMTPAuthenticationError(535, b"rejected")
        def send_message(self, *_args): raise AssertionError("must not send after rejected login")

    monkeypatch.setattr(notifier.smtplib, "SMTP", RejectAuth)
    monkeypatch.setattr(notifier.time, "sleep", lambda _seconds: None)
    assert not notifier._deliver_one(alert_app, notification_id)
    delivery = _local_notification(alert_app, video_id, notification_id)
    assert delivery["status"] == "failed" and delivery["attempts"] == 1
    assert "authentication" in delivery["failure_reason"].lower()
    assert "unit-password" not in delivery["failure_reason"]


def test_smtp_connect_timeout_retries_only_before_transmission(alert_app, tmp_path, monkeypatch):
    _configure_smtp(alert_app, 2525)
    video_id, _event_id, _ = _local_video(alert_app, tmp_path)
    notification_id = _local_notification(alert_app, video_id)["notification_id"]
    attempts = {"count": 0}
    class TimedOutSMTP:
        def __init__(self, *_args, **_kwargs):
            attempts["count"] += 1
            raise TimeoutError("server timeout")
    monkeypatch.setattr(notifier.smtplib, "SMTP", TimedOutSMTP)
    monkeypatch.setattr(notifier.time, "sleep", lambda _seconds: None)
    assert not notifier._deliver_one(alert_app, notification_id)
    event = _local_notification(alert_app, video_id, notification_id)
    assert attempts["count"] == 3
    assert event["status"] == "failed"
    assert event["attempts"] == 3


def test_missing_video_evidence_is_a_failed_delivery_not_an_empty_email(alert_app, tmp_path):
    _configure_smtp(alert_app, 2525)
    video_id, _event_id, source = _local_video(alert_app, tmp_path)
    notification_id = _local_notification(alert_app, video_id)["notification_id"]
    os.remove(source)
    assert not notifier._deliver_one(alert_app, notification_id)
    event = _local_notification(alert_app, video_id, notification_id)
    assert event["status"] == "failed"
    assert "source video" in event["failure_reason"].lower()


def test_grouping_preserves_gaps_and_distinct_restricted_entries():
    events = [
        {"_id": "fall-a", "event_type": "potential_fall", "track_id": 3,
         "timestamp_seconds": 4.0, "start_timestamp_seconds": 4.0, "end_timestamp_seconds": 5.0},
        {"_id": "fall-b", "event_type": "potential_fall", "track_id": 3,
         "timestamp_seconds": 5.5, "start_timestamp_seconds": 5.5, "end_timestamp_seconds": 6.0},
        {"_id": "entry-a", "event_type": "RESTRICTED_ZONE_ENTRY", "track_id": 3,
         "timestamp_seconds": 5.0, "details": {"zone_id": "north"}},
        {"_id": "entry-b", "event_type": "RESTRICTED_ZONE_ENTRY", "track_id": 3,
         "timestamp_seconds": 6.0, "details": {"zone_id": "north"}},
    ]
    groups = group_event_episodes(events, merge_gap_seconds=1.0)
    fall = next(group for group in groups if group["event_type"] == "potential_fall")
    entries = [group for group in groups if group["event_type"] == "RESTRICTED_ZONE_ENTRY"]
    assert len(fall["events"]) == 2
    assert fall["intervals"] == [[4.0, 5.0], [5.5, 6.0]]  # The unobserved gap remains visible.
    assert len(entries) == 2


def test_local_video_uses_one_consolidated_email_slot_for_many_categories(alert_app, tmp_path, smtp_sink):
    port, messages = smtp_sink
    _configure_smtp(alert_app, port)
    clip = _make_clip(tmp_path / "combined-clip.avi", 2.0)
    os.makedirs(alert_app.config["UPLOAD_DIR"], exist_ok=True)
    stored = "b" * 32 + ".avi"
    target = os.path.join(alert_app.config["UPLOAD_DIR"], stored)
    os.replace(clip, target)
    events = [
        {"event_type": "RESTRICTED_ZONE_ENTRY", "timestamp_seconds": 1.0,
         "track_id": 7, "details": {"zone_id": "zone-1", "zone_name": "North gate"}},
        {"event_type": "potential_fall", "timestamp_seconds": 2.0,
         "start_timestamp_seconds": 1.0, "end_timestamp_seconds": 2.0,
         "track_id": 1, "anomaly_score": .63, "details": {"previously_upright": True}},
        {"event_type": "SUSPECTED_FIGHTING", "timestamp_seconds": 3.0,
         "start_timestamp_seconds": 3.0, "end_timestamp_seconds": 5.0,
         "model_score": .71, "detection_method": "X3D-M"},
        {"event_type": "zone_presence_observed", "timestamp_seconds": 1.5},
    ]
    with alert_app.app_context():
        video_id, job_id = save_pending_upload({"original_filename": "combined-clip.avi",
            "stored_filename": stored, "file_path": stored, "file_size": os.path.getsize(target),
            "zones": []}, "test fallback")
        assert save_events(video_id, job_id, events) == 4
        queued_ids = initialize_alert_delivery(video_id, events)
        assert len(queued_ids) == 1
        row = get_pending_video(video_id)
        notifications = row["email_alert_notifications"]
        assert row["alert_email_slots_used"] == 1
        assert len(notifications) == 1
        payload = notifications[0]
        assert payload["category_count"] == 3
        assert {episode["event_type"] for episode in payload["episodes"]} == {
            "RESTRICTED_ZONE_ENTRY", "potential_fall", "SUSPECTED_FIGHTING"}
        assert payload["notification_id"] == queued_ids[0]
        assert len(payload["events"]) == 3
    assert notifier._deliver_one(alert_app, queued_ids[0])
    assert len(messages) == 1
    assert b"Restricted Zone Entry Detected" in messages[0]
    assert b"Potential Fall Candidate" in messages[0]
    assert b"Suspected Violence Candidate" in messages[0]
    assert messages[0].count(b"Content-Disposition: attachment") == 1


def test_video_never_reserves_more_than_two_emails_and_suppresses_replays(alert_app, tmp_path):
    from bson import ObjectId
    _configure_smtp(alert_app, 2525)
    video_id, _event_id, _ = _local_video(alert_app, tmp_path)
    first = _local_notification(alert_app, video_id)
    with alert_app.app_context():
        update_video_alert(first["notification_id"], {"status": "sent", "sent_at": "accepted"})
        row = get_pending_video(video_id)
        job_id = row["local_job"]["_id"]
        second_event = {"event_type": "potential_fall", "timestamp_seconds": 2.0,
            "start_timestamp_seconds": 1.0, "end_timestamp_seconds": 2.0,
            "track_id": 2, "_id": ObjectId(), "details": {"previously_upright": True}}
        save_events(video_id, job_id, [second_event])
        second_ids = initialize_alert_delivery(video_id, [second_event])
        assert len(second_ids) == 1
        third_event = {"event_type": "SUSPECTED_FIGHTING", "timestamp_seconds": 8.0,
            "start_timestamp_seconds": 7.0, "end_timestamp_seconds": 9.0,
            "_id": ObjectId(), "model_score": .8}
        save_events(video_id, job_id, [third_event])
        assert initialize_alert_delivery(video_id, [third_event]) == []
        saved = get_pending_video(video_id)
        assert saved["alert_email_slots_used"] == 2
        assert len(saved["email_alert_notifications"]) == 2
        assert saved["alert_email_limit_reached"] is True
        # Reprocessing the already reported fall group cannot reserve another send.
        assert initialize_alert_delivery(video_id, [second_event]) == []


def test_concurrent_workers_atomically_claim_one_video_email(alert_app, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    _configure_smtp(alert_app, 2525)
    video_id, _event_id, _ = _local_video(alert_app, tmp_path)
    notification_id = _local_notification(alert_app, video_id)["notification_id"]
    def claim(_index):
        with alert_app.app_context():
            return claim_video_alert(notification_id)
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(claim, range(8)))
    assert sum(claim is not None for claim in claims) == 1
    assert _local_notification(alert_app, video_id, notification_id)["status"] == "sending"


def test_local_fallback_claim_is_atomic_across_processes(alert_app, tmp_path):
    _configure_smtp(alert_app, 2525)
    video_id, _event_id, _ = _local_video(alert_app, tmp_path)
    notification_id = _local_notification(alert_app, video_id)["notification_id"]
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [context.Process(target=_claim_local_alert_in_process,
        args=(alert_app.config["LOCAL_FALLBACK_PATH"], notification_id, result_queue))
        for _ in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(15)
    assert all(process.exitcode == 0 for process in processes)
    claims = [result_queue.get(timeout=2) for _ in processes]
    assert sum(claims) == 1
    assert _local_notification(alert_app, video_id, notification_id)["status"] == "sending"


def test_uncertain_smtp_acceptance_is_not_retried(alert_app, tmp_path, monkeypatch):
    _configure_smtp(alert_app, 2525)
    video_id, _event_id, _ = _local_video(alert_app, tmp_path)
    notification_id = _local_notification(alert_app, video_id)["notification_id"]
    state = {"send_calls": 0}
    class AcceptedThenTimedOut:
        def __init__(self, *_args, **_kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def ehlo(self): pass
        def login(self, *_args): pass
        def send_message(self, _message):
            state["send_calls"] += 1
            raise TimeoutError("final SMTP response timed out")
    monkeypatch.setattr(notifier.smtplib, "SMTP", AcceptedThenTimedOut)
    monkeypatch.setattr(notifier.time, "sleep", lambda _seconds: None)
    assert not notifier._deliver_one(alert_app, notification_id)
    saved = _local_notification(alert_app, video_id, notification_id)
    assert state["send_calls"] == 1
    assert saved["status"] == "uncertain"
    with alert_app.app_context():
        assert not list_pending_video_alerts(video_id)


def test_restart_marks_in_flight_local_alert_uncertain_instead_of_resending(alert_app, tmp_path):
    _configure_smtp(alert_app, 2525)
    video_id, _event_id, _ = _local_video(alert_app, tmp_path)
    notification_id = _local_notification(alert_app, video_id)["notification_id"]
    with alert_app.app_context():
        assert claim_video_alert(notification_id)
        assert update_video_alert(notification_id, {
            "smtp_transmission_started_at": datetime.now(timezone.utc) - timedelta(hours=1),
            "last_attempt_at": datetime.now(timezone.utc) - timedelta(hours=1),
        })
        recover_local_video_alerts()
    saved = _local_notification(alert_app, video_id, notification_id)
    assert saved["status"] == "uncertain"
    with alert_app.app_context():
        assert not list_pending_video_alerts(video_id)


def test_restart_safely_requeues_stale_pre_smtp_local_alert(alert_app, tmp_path):
    _configure_smtp(alert_app, 2525)
    video_id, _event_id, _ = _local_video(alert_app, tmp_path)
    notification_id = _local_notification(alert_app, video_id)["notification_id"]
    with alert_app.app_context():
        assert claim_video_alert(notification_id)
        assert update_video_alert(notification_id, {
            "last_attempt_at": datetime.now(timezone.utc) - timedelta(hours=1),
        })
        recover_local_video_alerts(datetime.now(timezone.utc) - timedelta(minutes=10))
    saved = _local_notification(alert_app, video_id, notification_id)
    assert saved["status"] == "pending"
    with alert_app.app_context():
        assert list_pending_video_alerts(video_id)


def test_mongo_alert_recovery_distinguishes_pre_smtp_and_ambiguous_states(alert_app):
    from datetime import datetime, timedelta, timezone
    from app.services.repository import recover_video_alerts
    db = alert_app.extensions["mongo_db"]
    stale = datetime.now(timezone.utc) - timedelta(hours=1)
    video_id = db.videos.insert_one({"email_alert_notifications": [
        {"notification_id": "safe-before-smtp", "status": "sending",
         "delivery_protocol_version": 2, "last_attempt_at": stale},
        {"notification_id": "maybe-accepted", "status": "sending",
         "delivery_protocol_version": 2, "last_attempt_at": stale,
         "smtp_transmission_started_at": stale},
        {"notification_id": "legacy-send", "status": "sending", "last_attempt_at": stale},
    ]}).inserted_id
    with alert_app.app_context():
        assert recover_video_alerts(stale_after_seconds=120) == 3
    alerts = db.videos.find_one({"_id": video_id})["email_alert_notifications"]
    statuses = {item["notification_id"]: item["status"] for item in alerts}
    assert statuses == {"safe-before-smtp": "pending", "maybe-accepted": "uncertain", "legacy-send": "uncertain"}


def test_mongodb_video_notification_is_atomic_and_excludes_occupancy(alert_app):
    from bson import ObjectId
    _configure_smtp(alert_app, 2525)
    db = alert_app.extensions["mongo_db"]
    video_id = db.videos.insert_one({"original_filename": "saved.mp4"}).inserted_id
    job_id = ObjectId()
    events = [
        {"event_type": "potential_fall", "timestamp_seconds": 2.0},
        {"event_type": "RESTRICTED_ZONE_ENTRY", "timestamp_seconds": 3.0,
         "details": {"zone_id": "z1"}},
        {"event_type": "SUSPECTED_FIGHTING", "timestamp_seconds": 4.0,
         "start_timestamp_seconds": 4.0, "end_timestamp_seconds": 7.0},
        {"event_type": "zone_presence_observed", "timestamp_seconds": 1.0},
    ]
    with alert_app.app_context():
        assert save_events(str(video_id), str(job_id), events) == 4
        queued_ids = initialize_alert_delivery(str(video_id), events)
        assert len(queued_ids) == 1
        claimed = claim_video_alert(queued_ids[0])
        assert claimed["status"] == "sending"
        assert claim_video_alert(queued_ids[0]) is None
        assert update_video_alert(queued_ids[0], {"status": "sent", "sent_at": "test"})
        saved_video = db.videos.find_one({"_id": video_id})
        assert saved_video["alert_email_slots_used"] == 1
        assert saved_video["email_alert_notifications"][0]["status"] == "sent"
        assert len(saved_video["email_alert_notifications"][0]["episodes"]) == 3
        assert db.detection_events.find_one({"video_id": video_id, "event_type": "zone_presence_observed"}).get("alert_delivery") is None


def test_results_and_incidents_show_delivery_status(alert_app, tmp_path):
    video_id, _event_id, _ = _local_video(alert_app, tmp_path)
    client = alert_app.test_client()
    result = client.get(f"/results/{video_id}")
    incidents = client.get("/incidents")
    assert result.status_code == incidents.status_code == 200
    assert b"Email alert delivery" in result.data
    assert b"Disabled" in result.data
    assert b"Email alert" in incidents.data
