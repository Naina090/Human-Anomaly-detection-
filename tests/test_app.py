import io
import json
import os
import sys
import time
import types
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import mongomock
import numpy as np
import pytest
from bson import ObjectId
from pymongo.errors import ConfigurationError, OperationFailure

from app import create_app
from app.db import DatabaseUnavailable, _friendly_connection_error, get_database
from app.services.repository import (
    create_video, dashboard_stats, delete_video, get_video, list_videos,
    object_id, report_data, report_data_with_source, update_video, create_job, claim_job, update_incident,
)
from app.services.fall_detector import YoloPoseFallDetector
from app.services.storage import safe_video_path, safe_processed_path
from app.services.face_detector import OpenCVFaceDetector, FaceDetectorUnavailable, detector_status as face_detector_status
from app.services.object_detector import (ObjectDetectorUnavailable, YoloObjectTracker,
    detector_status as object_detector_status)
from app.services.video_processor import analyze_video
from app.services.video_processor import make_browser_playable_webm
from app.services.roi import ZoneValidationError, associate_event_with_zones, parse_zones
from app.services.video_validator import validate_video_file, VideoValidationError
from app.services.report_service import build_analysis_summary, video_report_csv, explain_event
from app.services.pdf_report import build_pdf


@pytest.fixture()
def app(tmp_path):
    client = mongomock.MongoClient()
    application = create_app({
        "TESTING": True, "SECRET_KEY": "test-secret", "MONGO_CLIENT": client,
        "MONGO_DB_NAME": "test_human_anomaly", "MONGO_PING_ON_STARTUP": False,
        "UPLOAD_DIR": str(tmp_path / "uploads"), "PROCESSED_DIR": str(tmp_path / "processed"),
        "LOCAL_FALLBACK_PATH": str(tmp_path / "local_history.json"),
        "REQUEST_ERROR_LOG_PATH": str(tmp_path / "sentinel-errors.log"),
        "MAX_CONTENT_LENGTH": 2_000_000, "MAX_DURATION_SECONDS": 30,
        "SAMPLE_EVERY_SECONDS": 0.1, "POSE_MODEL_PATH": "", "OBJECT_MODEL_PATH": "",
        "FIGHT_DETECTION_ENABLED": False, "FIGHT_MODEL_PATH": "",
    })
    yield application
    application.extensions["processing_executor"].shutdown(wait=True, cancel_futures=True)
    client.close()
    error_logger = application.extensions.get("request_error_logger")
    if error_logger:
        for handler in list(error_logger.handlers):
            if getattr(handler, "_sentinel_log_path", None) == os.path.abspath(application.config["REQUEST_ERROR_LOG_PATH"]):
                error_logger.removeHandler(handler)
                handler.close()


@pytest.fixture()
def client(app):
    return app.test_client()


def test_anomax_branding_renders_across_shared_website_pages(client, app):
    assert client.get("/static/css/branding.css").status_code == 200
    paths = ["/", "/dashboard", "/upload", "/history", "/incidents", "/reports", "/about"]
    with app.app_context():
        video_id = str(create_video(video_doc("brand-check.mp4", status="completed")))
    paths.append(f"/results/{video_id}")
    for path in paths:
        response = client.get(path)
        assert response.status_code == 200, path
        page = response.get_data(as_text=True)
        assert "AnomaX AI" in page, path
        assert "Detect Patterns. Discover Anomalies." in page, path
        assert "INTELLIGENT HUMAN ANOMALY DETECTION SYSTEM" in page, path
        assert "© 2026 AnomaX AI. All rights reserved." in page, path
        assert "Sentinel" not in page, path
        assert 'href="/upload">Upload</a>' in page, path
        assert 'href="/history">History</a>' in page, path
        assert 'href="/reports">Reports</a>' in page, path


def csrf(client):
    client.get("/upload")
    with client.session_transaction() as session:
        return session["csrf_token"]


def video_doc(name="clip.mp4", status="processing"):
    return {
        "original_filename": name, "stored_filename": "internal.mp4", "file_path": "internal.mp4",
        "upload_date": datetime.now(timezone.utc), "processing_status": status,
        "duration": None, "frame_rate": None, "width": None, "height": None,
        "processing_duration": None, "error_message": None,
    }


def make_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV video writer is unavailable")
    for i in range(10):
        writer.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def wait_for_status(app, video_id, statuses=("complete", "failed")):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        row = app.extensions["mongo_db"].videos.find_one({"_id": ObjectId(video_id)})
        if row and row["processing_status"] in statuses:
            job = app.extensions["mongo_db"].processing_jobs.find_one(
                {"video_id": ObjectId(video_id)}, sort=[("created_at", -1)])
            job_status = (job or {}).get("status")
            if ((row["processing_status"] == "complete" and job_status == "completed") or
                    (row["processing_status"] == "failed" and job_status == "failed") or
                    row["processing_status"] not in {"complete", "failed"}):
                return row
        time.sleep(.02)
    raise AssertionError("Background video job did not finish in time")


def wait_for_local_status(app, video_id, statuses=("complete", "failed"), timeout=10):
    from app.services.local_fallback import get_pending_video
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with app.app_context():
            row = get_pending_video(video_id)
        job_status = (row or {}).get("local_job", {}).get("status")
        video_status = (row or {}).get("processing_status")
        job_is_terminal_for_video = (
            (video_status == "complete" and job_status == "completed") or
            (video_status == "failed" and job_status == "failed") or
            video_status not in {"complete", "failed"})
        if row and video_status in statuses and job_is_terminal_for_video:
            return row
        time.sleep(.02)
    raise AssertionError("Local background job did not finish in time")


def test_app_startup_and_pages_render(client):
    for url in ("/", "/dashboard", "/about", "/history", "/incidents", "/reports", "/upload"):
        assert client.get(url).status_code == 200
    upload = client.get("/upload")
    assert b"Smart Surveillance Zone Configuration" in upload.data
    assert b"roi-canvas" in upload.data and client.get("/static/js/roi.js").status_code == 200
    assert b"Select a video frame and click Draw Zone" in upload.data
    assert b"zone-frame-slider" in upload.data and b"Preparing video frame" in upload.data
    assert b"Finish Polygon" in upload.data and b"Undo Last Point" in upload.data


def test_empty_history_shows_real_empty_state(client):
    response = client.get("/history")
    assert response.status_code == 200
    assert b"No matching videos" in response.data
    assert b"will appear here" in response.data


def test_results_alert_status_is_read_only_and_renders_latest_state(client, app, monkeypatch):
    import app.services.notifier as notifier
    refresh_calls = []
    monkeypatch.setattr(notifier, "schedule_pending_alerts",
                        lambda *_args: refresh_calls.append("scheduled"))
    now = datetime.now(timezone.utc)
    video_id = app.extensions["mongo_db"].videos.insert_one({
        "original_filename": "alert-status.mp4", "processing_status": "complete",
        "upload_date": now, "duration": 10, "email_alert_slots_used": 1,
        "email_alert_notifications": [{"notification_id": "digest-1", "status": "sent",
            "sent_at": now, "last_attempt_at": now, "episodes": [
                {"event_type": "potential_fall", "timestamp_seconds": 4.0}]}],
    }).inserted_id
    response = client.get(f"/api/videos/{video_id}/alerts")
    assert response.status_code == 200
    assert response.json["sent_count"] == 1 and response.json["maximum"] == 2
    assert response.json["notifications"][0]["status"] == "sent"
    page = client.get(f"/results/{video_id}")
    assert page.status_code == 200
    assert b"Delivery status:" in page.data
    assert b"Sent" in page.data
    assert b"alert-live-status" in page.data
    assert b"alert-live-summary" in page.data
    assert page.data.count(b"class=\"method-card alert-delivery alert-live-status\"") == 1
    assert refresh_calls == []
    assert app.extensions["mongo_db"].videos.find_one({"_id": video_id})["email_alert_notifications"][0]["status"] == "sent"


def test_alert_status_maps_legacy_disabled_to_skipped(client, app):
    video_id = app.extensions["mongo_db"].videos.insert_one({
        "original_filename": "skipped-alert.mp4", "processing_status": "complete",
        "email_alert_notifications": [{"notification_id": "disabled-legacy", "status": "disabled",
                                        "failure_reason": "Email alerts disabled", "episodes": []}],
    }).inserted_id
    response = client.get(f"/api/videos/{video_id}/alerts")
    assert response.status_code == 200
    assert response.json["delivery_status"] == "skipped"
    assert response.json["notifications"][0]["status"] == "skipped"


def test_alert_status_poll_moves_from_sending_to_sent_and_stops_at_terminal(client, app):
    video_id = app.extensions["mongo_db"].videos.insert_one({
        "original_filename": "live-alert.mp4", "processing_status": "complete",
        "email_alert_notifications": [{"notification_id": "live-digest", "status": "sending",
                                        "last_attempt_at": datetime.now(timezone.utc),
                                        "delivery_protocol_version": 2,
                                        "episodes": [{"event_type": "potential_fall", "timestamp_seconds": 2.0}] }],
    }).inserted_id
    url = f"/api/videos/{video_id}/alerts"
    sending = client.get(url)
    assert sending.status_code == 200 and sending.json["delivery_status"] == "sending"
    app.extensions["mongo_db"].videos.update_one(
        {"_id": video_id}, {"$set": {"email_alert_notifications.0.status": "sent"}})
    sent = client.get(url)
    assert sent.status_code == 200 and sent.json["delivery_status"] == "sent"
    script = client.get("/static/js/results.js")
    assert script.status_code == 200
    assert b"terminal.has(state)" in script.data
    assert b"cache:'no-store'" in script.data


def test_alert_status_database_failure_returns_safe_retryable_response(client, app, monkeypatch):
    import app.routes.videos as video_routes
    from app.db import DatabaseUnavailable
    video_id = app.extensions["mongo_db"].videos.insert_one({
        "original_filename": "unavailable-alert.mp4", "processing_status": "complete",
    }).inserted_id
    monkeypatch.setattr(video_routes, "get_video",
                        lambda _identifier: (_ for _ in ()).throw(DatabaseUnavailable("database temporarily unavailable")))
    response = client.get(f"/api/videos/{video_id}/alerts")
    assert response.status_code == 503
    assert response.json["delivery_status"] == "unavailable"
    assert "database temporarily unavailable" in response.json["error"]


def test_alert_status_resumes_only_previously_reserved_stale_pre_smtp_delivery(client, app, monkeypatch):
    import app.services.notifier as notifier
    calls = []
    monkeypatch.setattr(notifier, "schedule_pending_alerts",
                        lambda _application, identifier: calls.append(str(identifier)))
    stale = datetime.now(timezone.utc).replace(year=datetime.now(timezone.utc).year - 1)
    video_id = app.extensions["mongo_db"].videos.insert_one({
        "original_filename": "already-reserved.mp4", "processing_status": "complete",
        "email_alert_notifications": [{"notification_id": "reserved-once", "status": "sending",
            "delivery_protocol_version": 2, "last_attempt_at": stale, "episodes": []}],
    }).inserted_id

    response = client.get(f"/api/videos/{video_id}/alerts")
    saved = app.extensions["mongo_db"].videos.find_one({"_id": video_id})

    assert response.status_code == 200
    assert response.json["notifications"][0]["status"] == "pending"
    assert calls == [str(video_id)]
    assert len(saved["email_alert_notifications"]) == 1


@pytest.mark.parametrize("status", ["processing", "failed", "complete"])
def test_results_page_handles_legacy_video_without_optional_fields(client, app, status):
    # This reproduces the prior 500: results() used to call strftime() on the
    # absent upload_date field in older MongoDB documents.
    video_id = app.extensions["mongo_db"].videos.insert_one({
        "original_filename": "legacy-video.mp4",
        "processing_status": status,
    }).inserted_id

    response = client.get(f"/results/{video_id}")

    assert response.status_code == 200
    assert b"legacy-video.mp4" in response.data
    assert b"Uploaded Unknown" in response.data
    assert b"No face results are stored" in response.data
    assert app.extensions["mongo_db"].detection_events.count_documents({"video_id": video_id}) == 0


def test_results_page_returns_404_for_invalid_or_missing_video(client):
    assert client.get("/results/not-an-object-id").status_code == 404
    assert client.get("/results/507f1f77bcf86cd799439011").status_code == 404


def test_history_page_handles_legacy_video_without_upload_date(client, app):
    app.extensions["mongo_db"].videos.insert_one({
        "original_filename": "legacy-history-video.avi",
        "stored_filename": "legacy-internal.avi",
        "processing_status": "complete",
    })

    response = client.get("/history")

    assert response.status_code == 200
    assert b"legacy-history-video.avi" in response.data
    assert b"Unknown" in response.data


def test_history_table_shows_ist_and_expandable_saved_ai_details(client, app):
    from app.routes.history import _format_upload_date
    assert _format_upload_date(datetime(2026, 10, 24, 20, 34, tzinfo=timezone.utc)) == "25 Oct 2026, 02:04 AM IST"
    with app.app_context():
        video_id = create_video({**video_doc("very-long-surveillance-video-filename.mp4", "complete"),
            "upload_date": datetime(2026, 10, 24, 20, 34, tzinfo=timezone.utc),
            "file_size": 2_400_000, "duration": 10.0, "processing_duration": 28.39,
            "face_analysis": {"status": "complete", "frames_with_faces": 4, "face_observations": 7},
            "pose_analysis": {"status": "complete", "model_name": "yolo26n-pose.pt",
                              "frames_with_people": 6},
            "object_analysis": {"status": "complete", "model_name": "yolo26n.pt",
                                "observations": 12, "counts_by_class": {"person": 10, "car": 2},
                                "person_track_count_estimate": 3},
            "fight_analysis": {"status": "complete", "model_name": "X3D-M", "candidate_event_count": 1}})
        db = app.extensions["mongo_db"]
        db.detection_events.insert_many([
            {"video_id": video_id, "event_type": "potential_fall", "incident_status": "reviewing",
             "timestamp_seconds": 3.0},
            {"video_id": video_id, "event_type": "RESTRICTED_ZONE_ENTRY", "incident_status": "new",
             "timestamp_seconds": 5.0, "track_id": 9, "details": {"zone_name": "North Gate"}},
            {"video_id": video_id, "event_type": "zone_presence_observed", "incident_status": "not reviewed",
             "timestamp_seconds": 1.0},
            {"video_id": video_id, "event_type": "object_detection", "label": "person"},
        ])
    response = client.get("/history")
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    expected_columns = ["Video", "Uploaded (IST)", "Status", "More Details", "Actions"]
    header = page[page.index("<thead>"):page.index("</thead>") + len("</thead>")]
    assert header.count('<th scope="col">') == 5
    assert [header.index(f">{label}</th>") for label in expected_columns] == sorted(
        header.index(f">{label}</th>") for label in expected_columns)
    assert "25 Oct 2026, 02:04 AM IST" in page
    assert "2.29 MB" in page and "28.39 s" in page and "File size:" in page
    assert "Face Analysis" in page and "7" in page and "not unique individuals" in page
    assert "yolo26n-pose.pt" in page and "yolo26n.pt" in page
    assert "12" in page and "3" in page and "requiring human review" in page
    assert "Fighting: 1 candidate" in page and "Anomalies: 1 event" in page
    assert "Zone Entries:" in page and "1 entry" in page
    assert "Confirmed restricted-zone entry events" in page
    assert "Initial zone occupancy observations (entry not observed)" in page
    assert "Reviewing" in page and "Restricted Zone Entry" in page
    assert "00:05.00" in page and "Temporary track: 9" in page and "Zone: North Gate" in page
    assert "Anomalies:" in page and "Zone Entries:" in page
    assert "history-details-toggle" in page and "history-mobile-details" in page
    assert "history-mobile-analysis" in page and "Anomaly Events" in page and "Restricted-Zone Entries" in page
    assert "history-details-button" in page and "More Details" in page
    assert "Saved zones:" in page and "Temporary track: 9" in page
    assert "@media(max-width:820px)" in page and "history-mobile" in page
    assert "history-page{overflow-x:clip}" in page and "min-width:900px" in page
    assert 'colspan="5"' in page
    assert f"/results/{video_id}" in page and f"/media/{video_id}/download" in page
    assert f"/history/{video_id}/delete" in page and "aria-expanded=\"false\"" in page
    history_script = client.get("/static/js/history.js").get_data(as_text=True)
    assert "closeOtherDetails(row)" in history_script and "closeOtherDetails(details)" in history_script


def test_history_fallback_statuses_and_legacy_ai_fields_are_truthful(client, app):
    from app.services.local_fallback import save_pending_upload
    with app.app_context():
        video_id, _job_id = save_pending_upload({"original_filename": "local-history.avi",
            "stored_filename": "local-history-internal.avi", "file_path": "local-history-internal.avi",
            "file_size": 2048, "duration": 2.5}, "test fallback")
    page = client.get("/history").get_data(as_text=True)
    assert "local-history.avi" in page and "Pending" in page
    assert "Not Configured" in page
    assert "Total face observations:" in page
    assert "0.00 seconds" not in page


def test_dashboard_shows_database_and_detector_readiness_without_fake_counts(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert b"MongoDB" in response.data and b"Connected" in response.data
    assert b"Pose analysis" in response.data and b"Validation only" in response.data
    assert b">0</strong>" in response.data
    assert b"No videos yet" in response.data and b"No events recorded" in response.data


def test_mongodb_connection_failure_is_friendly(tmp_path):
    app = create_app({"TESTING": True, "SECRET_KEY": "test", "MONGO_URI": "mongodb://127.0.0.1:1/",
                      "MONGO_TIMEOUT_MS": 20, "MONGO_DB_NAME": "connection_test",
                      "UPLOAD_DIR": str(tmp_path / "u"), "PROCESSED_DIR": str(tmp_path / "p")})
    client = app.test_client()
    for path in ("/dashboard", "/history", "/reports"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.data
        assert b"internal error" not in response.data.lower()
    assert client.get("/").status_code == 200
    app.extensions["mongo_client"].close()


def test_flask_retries_client_creation_after_temporary_startup_dns_failure(tmp_path, monkeypatch):
    import app.db as database_module
    attempts = {"count": 0}
    recovered_client = mongomock.MongoClient()

    def client_factory(*_args, **_kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ConfigurationError("DNS SRV lookup temporarily timed out")
        return recovered_client

    monkeypatch.setattr(database_module, "MongoClient", client_factory)
    application = create_app({"TESTING": True, "SECRET_KEY": "test", "MONGO_URI": "mongodb+srv://user:pass@cluster.example.test/db",
        "MONGO_DB_NAME": "recovery_test", "MONGO_TIMEOUT_MS": 20, "MONGO_RETRY_INTERVAL_SECONDS": 0,
        "UPLOAD_DIR": str(tmp_path / "u"), "PROCESSED_DIR": str(tmp_path / "p")})
    assert application.extensions["mongo_client"] is None
    assert application.extensions["mongo_available"] is False

    with application.app_context():
        db = get_database()
        saved_id = db.videos.insert_one({"original_filename": "recovered.avi"}).inserted_id
        assert db.videos.find_one({"_id": saved_id})["original_filename"] == "recovered.avi"

    assert attempts["count"] == 2
    assert application.extensions["mongo_available"] is True
    assert application.extensions["mongo_ping_ok"] is True
    application.extensions["processing_executor"].shutdown(wait=True, cancel_futures=True)
    recovered_client.close()


def test_replica_set_no_primary_is_not_misreported_as_dns():
    from pymongo.errors import ServerSelectionTimeoutError
    error = ServerSelectionTimeoutError("No replica set members match selector; ReplicaSetNoPrimary")
    message = _friendly_connection_error(error)
    assert "replica-set primary" in message
    assert "DNS" not in message


def test_analyze_video_route_logs_unexpected_upload_failure_without_exposing_traceback(
        client, app, tmp_path, monkeypatch, caplog):
    import app.routes.videos as video_routes

    # Exercise the same /upload page and multipart POST used by Analyze/New
    # Analysis, while forcing an unexpected persistence-layer exception.
    page = client.get("/upload")
    assert page.status_code == 200 and b'action="/upload"' in page.data
    payload = make_video(tmp_path / "route-diagnostic.avi")
    app.config["MONGO_URI"] = "mongodb+srv://diagnostic-user:diagnostic-secret@invalid.example/db"
    monkeypatch.setattr(video_routes, "create_video", lambda _document: (_ for _ in ()).throw(
        RuntimeError("injected upload persistence failure: " + app.config["MONGO_URI"])))

    response = client.post("/upload", data={
        "csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "route-diagnostic.avi"),
    }, content_type="multipart/form-data")

    assert response.status_code == 500
    assert b"injected upload persistence failure" not in response.data
    assert b"Traceback" not in response.data
    request_id = response.headers.get("X-Request-ID")
    assert request_id and f"Reference: {request_id}".encode() in response.data
    assert f"Unhandled request exception request_id={request_id} (RuntimeError) at endpoint videos.upload" in caplog.text
    assert "injected upload persistence failure" in caplog.text
    assert "Traceback (most recent call last)" in caplog.text
    assert "diagnostic-secret" not in caplog.text
    assert "<redacted MongoDB URI>" in caplog.text
    persisted_log = (tmp_path / "sentinel-errors.log").read_text(encoding="utf-8")
    assert request_id in persisted_log
    assert "Traceback (most recent call last)" in persisted_log
    assert "diagnostic-secret" not in persisted_log
    assert "<redacted MongoDB URI>" in persisted_log
    assert list((tmp_path / "uploads").iterdir()) == []


def test_startup_identifies_ping_authentication_failure_without_credentials(tmp_path, caplog):
    class Admin:
        def command(self, _name):
            raise OperationFailure("Authentication failed", 18)
    class FakeClient:
        admin = Admin()
        def __getitem__(self, _name):
            return object()
    application = create_app({"TESTING": True, "SECRET_KEY": "test", "MONGO_CLIENT": FakeClient(),
                              "MONGO_DB_NAME": "isolated_test", "MONGO_PING_ON_STARTUP": True,
                              "UPLOAD_DIR": str(tmp_path / "u"), "PROCESSED_DIR": str(tmp_path / "p")})
    assert application.extensions["mongo_ping_ok"] is False
    assert "startup ping failed (OperationFailure, code=18)" in caplog.text
    application.extensions["processing_executor"].shutdown(wait=True, cancel_futures=True)


def test_atlas_dns_failure_is_explained_and_traceback_redacts_connection_uri(tmp_path, monkeypatch, caplog):
    import app.db as database_module
    from pymongo.errors import ConfigurationError

    fake_uri = "mongodb+srv://diagnostic-user:diagnostic-secret@cluster.invalid/db"
    def fail_client(*_args, **_kwargs):
        raise ConfigurationError("DNS query timed out for " + fake_uri)
    monkeypatch.setattr(database_module, "MongoClient", fail_client)

    application = create_app({"TESTING": True, "SECRET_KEY": "test", "MONGO_URI": fake_uri,
                              "MONGO_DB_NAME": "isolated_dns_test", "MONGO_TIMEOUT_MS": 20,
                              "UPLOAD_DIR": str(tmp_path / "u"), "PROCESSED_DIR": str(tmp_path / "p")})
    assert application.extensions["mongo_client"] is None
    assert "PyMongo could not resolve" in application.extensions["mongo_error"]
    assert "Traceback (most recent call last)" in caplog.text
    assert "diagnostic-secret" not in caplog.text
    assert "<redacted MongoDB URI>" in caplog.text
    application.extensions["processing_executor"].shutdown(wait=True, cancel_futures=True)


def test_startup_identifies_index_permission_failure_after_successful_ping(tmp_path, caplog):
    class Admin:
        def command(self, _name):
            return {"ok": 1}
    class Collection:
        def create_index(self, *_args, **_kwargs):
            raise OperationFailure("not authorized", 13)
    class FakeDatabase:
        videos = Collection()
    class FakeClient:
        admin = Admin()
        def __getitem__(self, _name):
            return FakeDatabase()
    application = create_app({"TESTING": True, "SECRET_KEY": "test", "MONGO_CLIENT": FakeClient(),
                              "MONGO_DB_NAME": "isolated_test", "MONGO_PING_ON_STARTUP": True,
                              "UPLOAD_DIR": str(tmp_path / "u"), "PROCESSED_DIR": str(tmp_path / "p")})
    assert application.extensions["mongo_ping_ok"] is True
    assert application.extensions["mongo_available"] is False
    assert "startup index creation failed (OperationFailure, code=13)" in caplog.text
    application.extensions["processing_executor"].shutdown(wait=True, cancel_futures=True)


def test_missing_or_example_mongodb_uri_keeps_public_pages_available(tmp_path):
    for uri in ("", "paste_your_atlas_connection_string_here",
                "mongodb+srv://<db_username>:<db_password>@cluster0.xxxxx.mongodb.net/"):
        app = create_app({"TESTING": True, "SECRET_KEY": "test", "MONGO_URI": uri,
                          "MONGO_DB_NAME": "unconfigured_test",
                          "UPLOAD_DIR": str(tmp_path / "u"), "PROCESSED_DIR": str(tmp_path / "p")})
        client = app.test_client()
        assert app.extensions["mongo_client"] is None
        assert client.get("/").status_code == 200
        response = client.get("/dashboard")
        assert response.status_code == 200
        assert b"Unavailable" in response.data
        app.extensions["processing_executor"].shutdown(wait=True, cancel_futures=True)


def test_repository_create_read_update_and_dashboard(app):
    with app.app_context():
        record_id = create_video(video_doc())
        assert isinstance(record_id, ObjectId)
        assert get_video(str(record_id))["original_filename"] == "clip.mp4"
        assert update_video(str(record_id), {"processing_status": "complete", "duration": 12.5,
            "face_analysis": {"status": "complete", "face_observations": 4}})
        stats = dashboard_stats()
        assert (stats["videos"], stats["done"], stats["failed"], stats["events"]) == (1, 1, 0, 0)
        assert (stats["face_videos"], stats["face_observations"]) == (1, 4)


def test_safe_objectid_validation():
    assert object_id("not-an-object-id") is None
    assert object_id("507f1f77bcf86cd799439011") == ObjectId("507f1f77bcf86cd799439011")


def test_operation_failure_messages_distinguish_auth_and_permissions():
    auth_message = _friendly_connection_error(OperationFailure("Authentication failed", 18))
    permission_message = _friendly_connection_error(OperationFailure("not authorized", 13))
    assert "authentication failed" in auth_message.lower()
    assert "permission" in permission_message.lower()


def test_dns_failure_message_names_timed_out_resolver_without_leaking_uri():
    uri = "mongodb+srv://safe-user:secret-pass@cluster.example.mongodb.net/db"
    message = _friendly_connection_error(
        ConfigurationError("DNS SRV resolution timed out for " + uri +
                            "; Server Do53:192.168.1.1@53 answered timeout")
    )
    assert "192.168.1.1" in message
    assert "UDP/TCP port 53" in message
    assert "secret-pass" not in message and uri not in message


def test_fall_heuristic_requires_a_transition_and_persistence():
    class Tensor:
        def __init__(self, values): self.values = np.asarray(values)
        def cpu(self): return self
        def numpy(self): return self.values
        def __len__(self): return len(self.values)
    class Keypoints:
        def __init__(self, points): self.xy = Tensor([points]); self.conf = Tensor([[.9] * 17])
    class Result:
        def __init__(self, points): self.keypoints = Keypoints(points)
        def plot(self): return np.zeros((100, 100, 3), dtype=np.uint8)
    class PoseModel:
        def __init__(self, frames): self.frames = iter(frames)
        def predict(self, frame, verbose=False): return [Result(next(self.frames))]
    def pose(center, horizontal=False):
        pts = np.zeros((17, 2), dtype=float)
        if horizontal:
            pts[5], pts[6], pts[11], pts[12] = [35, center], [65, center], [42, center+3], [72, center+3]
        else:
            pts[5], pts[6], pts[11], pts[12] = [45, center-20], [55, center-20], [45, center+20], [55, center+20]
        return pts.tolist()
    detector = YoloPoseFallDetector.__new__(YoloPoseFallDetector)
    detector.model = PoseModel([pose(40), pose(78, True), pose(78, True)])
    detector.previous = None; detector.track_center = None; detector.upright_seen = False; detector.upright_timestamp = None; detector.horizontal_since = None
    detector.drop_seen_at = None; detector.drop_fraction = None; detector.drop_start_at = None; detector.emitted = False
    detector.transition_window_seconds = 1.5; detector.minimum_vertical_drop = .12
    detector.minimum_horizontal_persistence = .6; detector.candidate_score_threshold = .55
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detector.inspect(frame, 0)[1] is None
    assert detector.inspect(frame, 1)[1] is None
    event = detector.inspect(frame, 2)[1]
    assert event["event_type"] == "potential_fall"
    assert event["confidence"] is None and 0 <= event["anomaly_score"] <= 1
    assert event["start_timestamp_seconds"] == 0 and event["end_timestamp_seconds"] == 2
    assert event["details"]["previously_upright"] is True

    static = YoloPoseFallDetector.__new__(YoloPoseFallDetector)
    static.model = PoseModel([pose(80, True), pose(80, True), pose(80, True)])
    static.previous = None; static.track_center = None; static.upright_seen = False; static.upright_timestamp = None; static.horizontal_since = None
    static.drop_seen_at = None; static.drop_fraction = None; static.drop_start_at = None; static.emitted = False
    static.transition_window_seconds = 1.5; static.minimum_vertical_drop = .12
    static.minimum_horizontal_persistence = .6; static.candidate_score_threshold = .55
    assert all(static.inspect(frame, t)[1] is None for t in (0, 1, 2))


def test_object_detector_normalizes_supported_classes_and_temporary_tracks(tmp_path, monkeypatch):
    import app.services.object_detector as object_module
    model_path = tmp_path / "local-coco.pt"
    model_path.write_bytes(b"test model marker")
    names = {0: "person", 1: "bicycle", 2: "car", 24: "backpack"}
    boxes = types.SimpleNamespace(
        xyxy=np.asarray([[1, 2, 30, 50], [10, 10, 40, 40], [20, 20, 30, 30], [0, 0, 1, 1]], dtype=float),
        cls=np.asarray([0, 2, 24, 1]), conf=np.asarray([.91, .82, .77, .99]),
        id=np.asarray([12, 4, -1, 2]),
    )
    class Result:
        pass
    Result.names = names
    Result.boxes = boxes
    class FakeYOLO:
        def __init__(self, _path): self.calls = []; self.names = names
        def track(self, frame, **kwargs):
            self.calls.append(kwargs)
            return [Result()]
    monkeypatch.setattr(object_module.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))

    detector = YoloObjectTracker(str(model_path), confidence=.3)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    observations = detector.detect(frame, 1.25, 25)
    assert [row["label"] for row in observations] == ["person", "car", "backpack", "bicycle"]
    assert observations[0]["timestamp_seconds"] == 1.25 and observations[0]["frame_number"] == 25
    assert observations[0]["confidence"] == .91 and observations[0]["track_id"] == 12
    assert observations[1]["track_id"] == 4 and observations[2]["track_id"] is None
    assert observations[3]["track_id"] == 2
    assert detector.model.calls[0]["persist"] is True
    assert detector.model.calls[0]["tracker"] == "bytetrack.yaml"
    assert detector.model.calls[0]["classes"] == [0, 1, 2, 24]
    annotated = detector.annotate(frame, observations)
    assert np.any(annotated != frame) and np.all(frame == 0)
    captions = []
    monkeypatch.setattr(object_module.cv2, "putText",
                        lambda image, text, *args, **kwargs: (captions.append(text), image)[1])
    detector.annotate(frame, observations)
    assert "person 0.91 ID 12" in captions
    assert "car 0.82 ID 4" in captions
    later_observations = detector.detect(frame, 1.75, 35)
    assert later_observations[0]["track_id"] == 12
    assert all(call["persist"] is True for call in detector.model.calls)


def test_object_detector_returns_empty_for_frames_without_supported_objects(tmp_path, monkeypatch):
    import app.services.object_detector as object_module
    model_path = tmp_path / "local-coco.pt"
    model_path.write_bytes(b"test model marker")
    boxes = types.SimpleNamespace(xyxy=np.empty((0, 4)), cls=np.empty((0,)),
                                  conf=np.empty((0,)), id=None)
    result = types.SimpleNamespace(boxes=boxes, names={0: "person"})

    class FakeYOLO:
        names = {0: "person"}
        def __init__(self, _path): pass
        def track(self, _frame, **_kwargs): return [result]

    monkeypatch.setattr(object_module.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    detector = YoloObjectTracker(str(model_path))
    assert detector.detect(np.zeros((32, 32, 3), dtype=np.uint8), 0, 0) == []


def test_object_detector_reports_missing_configuration_without_downloading(tmp_path, monkeypatch):
    import app.services.object_detector as object_module
    assert object_detector_status()["status"] == "not_configured"
    missing = tmp_path / "missing.pt"
    with pytest.raises(ObjectDetectorUnavailable, match="existing local"):
        YoloObjectTracker(str(missing))
    model_path = tmp_path / "available.pt"
    model_path.write_bytes(b"test marker")
    monkeypatch.setattr(object_module.importlib.util, "find_spec", lambda _name: None)
    assert object_detector_status(str(model_path))["status"] == "unavailable"
    with pytest.raises(ObjectDetectorUnavailable, match="requirements-ai"):
        YoloObjectTracker(str(model_path))


def test_object_model_relative_path_is_resolved_from_project_root_not_cwd(tmp_path, monkeypatch):
    from app.services.object_detector import resolve_model_path
    project_relative = resolve_model_path("models/not-installed.pt")
    monkeypatch.chdir(tmp_path)
    assert resolve_model_path("models/not-installed.pt") == project_relative
    status = object_detector_status("models/not-installed.pt")
    assert status["configured"] is True
    assert status["status"] == "unavailable"
    assert "not found" in status["message"]


def test_results_group_object_observations_without_counting_unique_people(client, app):
    from app.routes.videos import group_object_observations
    observations = [
        {"event_type": "object_detection", "label": "person", "track_id": 7,
         "timestamp_seconds": 1.0, "confidence": .8, "frame_number": 30,
         "details": {"bbox_xyxy": [1, 2, 30, 50]}},
        {"event_type": "object_detection", "label": "person", "track_id": 7,
         "timestamp_seconds": 3.0, "confidence": .9, "frame_number": 90,
         "details": {"bbox_xyxy": [2, 3, 31, 51]}},
        {"event_type": "object_detection", "label": "car", "track_id": 9,
         "timestamp_seconds": 2.0, "confidence": .7, "frame_number": 60,
         "details": {"bbox_xyxy": [4, 5, 40, 60]}},
        {"event_type": "object_detection", "label": "person", "track_id": None,
         "timestamp_seconds": 4.0, "details": {"bbox_xyxy": [0, 0, 10, 10]}},
        {"event_type": "object_detection", "label": "person", "track_id": None,
         "timestamp_seconds": 5.0, "details": {"bbox_xyxy": [0, 0, 10, 10]}},
        {"event_type": "potential_fall", "timestamp_seconds": 6.0},
    ]
    grouped = group_object_observations(observations)
    person_track = next(item for item in grouped if item["track_id"] == 7)
    assert person_track["object_class"] == "person"
    assert person_track["first_seen_seconds"] == 1.0
    assert person_track["last_seen_seconds"] == 3.0
    assert person_track["observation_count"] == 2
    assert len([item for item in grouped if item["track_id"] is None]) == 2

    with app.app_context():
        video_id = create_video({**video_doc("object-summary.mp4", status="complete"),
                                 "object_analysis": {"status": "complete", "observations": 5,
                                                     "frames_sampled": 3, "counts_by_class": {"person": 4, "car": 1},
                                                     "person_track_count_estimate": 1}})
        db = app.extensions["mongo_client"][app.config["MONGO_DB_NAME"]]
        for item in observations[:5]:
            db.detection_events.insert_one({**item, "video_id": video_id})
    response = client.get(f"/results/{video_id}")
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "First observed" in page and "Last observed" in page
    assert "Grouped summaries below count sampled observations" in page
    assert "Full object-detection evidence (5 sampled observations)" in page
    assert 'class="timeline-event object-observation"' in page
    assert "No anomaly or restricted-zone events were recorded" in page


def test_pose_model_alias_does_not_trigger_automatic_download(monkeypatch):
    from app.services.fall_detector import DetectorUnavailable, detector_status as pose_status
    import app.services.fall_detector as pose_module
    monkeypatch.setattr(pose_module.importlib.util, "find_spec", lambda _name: object())
    assert pose_status("yolo26n-pose.pt")["status"] == "unavailable"
    with pytest.raises(DetectorUnavailable, match="not found"):
        YoloPoseFallDetector("yolo26n-pose.pt")


def test_pose_detector_uses_persistent_local_tracker_and_preserves_real_person_outputs(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import app.services.fall_detector as pose_module
    marker = tmp_path / "local-pose.pt"
    marker.write_bytes(b"test fixture only")
    monkeypatch.setattr(pose_module.importlib.util, "find_spec", lambda _name: object())
    calls = []
    points = np.zeros((1, 17, 2), dtype=np.float32)
    points[0, 5] = [20, 10]; points[0, 6] = [30, 10]
    points[0, 11] = [22, 30]; points[0, 12] = [28, 30]
    confidences = np.full((1, 17), .9, dtype=np.float32)
    class FixtureResult:
        keypoints = SimpleNamespace(xy=points, conf=confidences)
        boxes = SimpleNamespace(xyxy=np.array([[10, 5, 40, 60]], dtype=np.float32),
                                conf=np.array([.87], dtype=np.float32), id=np.array([41], dtype=np.float32))
        def plot(self): return np.zeros((64, 64, 3), dtype=np.uint8)
    class FixtureYOLO:
        task = "pose"
        def __init__(self, _path): pass
        def track(self, _frame, **kwargs):
            calls.append(kwargs)
            return [FixtureResult()]
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FixtureYOLO))
    detector = YoloPoseFallDetector(str(marker), {"tracker": "bytetrack.yaml"})
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    _overlay, _event, evidence = detector.inspect_with_evidence(frame, 0.0)
    _overlay, _event, evidence = detector.inspect_with_evidence(frame, 1.0)
    assert len(calls) == 2 and all(call["persist"] is True for call in calls)
    person = evidence["people"][0]
    assert person["track_id"] == 41
    assert person["bbox_xyxy"] == [10.0, 5.0, 40.0, 60.0]
    assert person["detection_confidence"] == .87
    assert evidence["tracking_status"] == "tracked"
    assert evidence["people"][0]["keypoints"]["left_shoulder"]["confidence"] == .9


def test_pose_detector_rejects_local_non_pose_weights(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import app.services.fall_detector as pose_module
    marker = tmp_path / "detect-only.pt"
    marker.write_bytes(b"test fixture only")
    monkeypatch.setattr(pose_module.importlib.util, "find_spec", lambda _name: object())
    class DetectOnlyModel:
        task = "detect"
        def __init__(self, _path): pass
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=DetectOnlyModel))
    with pytest.raises(pose_module.DetectorUnavailable, match="not a pose model"):
        pose_module.YoloPoseFallDetector(str(marker))


def test_zone_tracker_requires_confirmed_transition_and_allows_exit_reentry():
    from app.services.video_processor import _zone_transition_events
    zones = [{"zone_id": "restricted", "name": "Restricted", "enabled": True,
              "points": [[.3, .3], [.7, .3], [.7, .7], [.3, .7]]}]
    def sample(t, x1, x2, track=7, confidence=.8):
        return {"timestamp_seconds": t, "frame_number": int(t * 10), "people": [
            {"track_id": track, "bbox_xyxy": [x1, 10, x2, 50], "detection_confidence": confidence}]}
    records = [sample(0, 10, 20), sample(.5, 10, 20),
               sample(1, 35, 45, confidence=.91), sample(1.5, 35, 45),
               sample(2, 10, 20), sample(2.5, 10, 20),
               sample(3, 35, 45), sample(3.5, 35, 45)]
    events = _zone_transition_events(records, zones, 100, 100)
    entries = [event for event in events if event["event_type"] == "RESTRICTED_ZONE_ENTRY"]
    assert len(entries) == 2
    assert [event["timestamp_seconds"] for event in entries] == [1, 3]
    assert entries[0]["confidence"] == .91
    assert entries[0]["details"]["zone_name"] == "Restricted"
    assert "confirmed outside and then inside" in explain_event(entries[0])
    assert len([event for event in events if event["event_type"] == "zone_exit"]) == 1
    assert _zone_transition_events(records, [], 100, 100) == []


def test_zone_tracker_does_not_infer_entry_from_first_observation_or_dwell():
    from app.services.video_processor import _zone_transition_events
    zones = [{"zone_id": "z", "name": "Zone", "enabled": True,
              "points": [[.3, .3], [.7, .3], [.7, .7], [.3, .7]]}]
    records = [{"timestamp_seconds": t, "frame_number": t,
                "people": [{"track_id": 1, "bbox_xyxy": [35, 10, 45, 50], "detection_confidence": .7}]}
               for t in range(6)]
    events = _zone_transition_events(records, zones, 100, 100)
    assert [event["event_type"] for event in events] == ["zone_presence_observed"]
    assert "entry time is unknown" in explain_event(events[0])


def test_zone_tracker_supports_multiple_people_zones_and_disabled_zones():
    from app.services.video_processor import _zone_transition_events
    zones = [
        {"zone_id": "left", "name": "Left", "enabled": True,
         "points": [[.1, .3], [.35, .3], [.35, .7], [.1, .7]]},
        {"zone_id": "right", "name": "Right", "enabled": True,
         "points": [[.65, .3], [.9, .3], [.9, .7], [.65, .7]]},
        {"zone_id": "off", "name": "Disabled", "enabled": False,
         "points": [[.4, .3], [.6, .3], [.6, .7], [.4, .7]]},
    ]
    def sample(t, people):
        return {"timestamp_seconds": t, "frame_number": int(t * 10), "people": people}
    def person(track, x1, x2):
        return {"track_id": track, "bbox_xyxy": [x1, 10, x2, 50], "detection_confidence": .8}
    records = [sample(0, [person(1, 40, 50), person(2, 40, 50)]),
               sample(.5, [person(1, 40, 50), person(2, 40, 50)]),
               sample(1, [person(1, 20, 30), person(2, 70, 80)]),
               sample(1.5, [person(1, 20, 30), person(2, 70, 80)])]
    entries = [event for event in _zone_transition_events(records, zones, 100, 100)
               if event["event_type"] == "RESTRICTED_ZONE_ENTRY"]
    assert {(event["track_id"], event["details"]["zone_id"]) for event in entries} == {(1, "left"), (2, "right")}


def test_zone_tracker_ignores_missing_ids_interruptions_and_boundary_jitter():
    from app.services.video_processor import _zone_transition_events
    zones = [{"zone_id": "z", "name": "Zone", "enabled": True,
              "points": [[.3, .3], [.7, .3], [.7, .7], [.3, .7]]}]
    def record(t, x1, x2, track=1):
        people = [] if track is False else [{"track_id": None if track is None else track,
            "bbox_xyxy": [x1, 10, x2, 50], "detection_confidence": .8}]
        return {"timestamp_seconds": t, "frame_number": t, "people": people}
    missing = [record(0, 10, 20, None), record(.5, 35, 45, None), record(1, 35, 45, None)]
    assert not [e for e in _zone_transition_events(missing, zones, 100, 100) if e["event_type"] == "RESTRICTED_ZONE_ENTRY"]
    interrupted = [record(0, 10, 20), record(.5, 10, 20), record(3, 35, 45), record(3.5, 35, 45)]
    assert not [e for e in _zone_transition_events(interrupted, zones, 100, 100) if e["event_type"] == "RESTRICTED_ZONE_ENTRY"]
    assert any(e["event_type"] == "zone_tracking_gap" for e in _zone_transition_events(interrupted, zones, 100, 100))
    jitter = [record(0, 10, 20), record(.5, 10, 20), record(1, 29, 35),
              record(1.5, 35, 45), record(2, 29, 35), record(2.5, 35, 45)]
    assert not [e for e in _zone_transition_events(jitter, zones, 100, 100) if e["event_type"] == "RESTRICTED_ZONE_ENTRY"]


def test_zone_tracker_starts_new_segment_after_gap_without_inventing_exit_or_entry():
    from app.services.video_processor import _zone_transition_events
    zones = [{"zone_id": "z", "name": "Zone", "enabled": True,
              "points": [[.3, .3], [.7, .3], [.7, .7], [.3, .7]]}]
    samples = [{"timestamp_seconds": 0, "frame_number": 0,
                "people": [{"track_id": 4, "bbox_xyxy": [35, 10, 45, 50], "detection_confidence": .7}]},
               {"timestamp_seconds": 3, "frame_number": 30,
                "people": [{"track_id": 4, "bbox_xyxy": [35, 10, 45, 50], "detection_confidence": .8}]},
               {"timestamp_seconds": 3.5, "frame_number": 35,
                "people": [{"track_id": 4, "bbox_xyxy": [35, 10, 45, 50], "detection_confidence": .8}]}]
    events = _zone_transition_events(samples, zones, 100, 100)
    assert [event["event_type"] for event in events] == [
        "zone_presence_observed", "zone_tracking_gap", "zone_presence_observed"]
    assert events[0]["track_segment_id"] == "4:s1"
    assert events[1]["details"]["previous_track_segment_id"] == "4:s1"
    assert events[1]["track_segment_id"] == "4:s2"
    assert events[1]["details"]["exit_inferred"] is False
    assert not [event for event in events if event["event_type"] in {"zone_exit", "RESTRICTED_ZONE_ENTRY"}]


def test_sampled_pose_pipeline_saves_one_neutral_zone_entry_for_occupied_track(tmp_path):
    from app.services.video_processor import analyze_video
    source = tmp_path / "tracked-zone.avi"
    make_video(source)
    zones = [{"zone_id": "door", "name": "Door", "enabled": True,
              "points": [[.1, .1], [.9, .1], [.9, .9], [.1, .9]]}]
    class TrackedPoseFixture:
        METHOD = "fixture pose tracker"
        def inspect_with_evidence(self, frame, timestamp):
            evidence = {"timestamp_seconds": timestamp, "person_count": 1,
                            "people": [{"track_id": 3, "bbox_xyxy": [20, 5, 40, 30],
                                    "detection_confidence": .91, "keypoints": {}}],
                        "transition_status": "observed_no_candidate"}
            return frame, None, evidence
    result = analyze_video(str(source), .2, 30, TrackedPoseFixture(), zones=zones)
    assert [event["event_type"] for event in result["events"]] == ["zone_presence_observed"]
    assert result["events"][0]["timestamp_seconds"] == 0
    assert result["events"][0]["anomaly_score"] is None
    assert result["pose_analysis"]["person_track_count_estimate"] == 1
    assert result["pose_analysis"]["zone_presence_observed_count"] == 1
    assert result["pose_analysis"]["zone_entry_count"] == 0
    assert result["pose_analysis"]["zone_exit_count"] == 0


def test_enabled_zone_uses_configured_tracking_cadence_only_when_active(tmp_path):
    from app.services.video_processor import analyze_video
    source = tmp_path / "zone-cadence.avi"
    make_video(source)
    class CountingDetector:
        METHOD = "fixture"
        def __init__(self): self.calls = 0
        def inspect_with_evidence(self, frame, timestamp):
            self.calls += 1
            return frame, None, {"timestamp_seconds": timestamp, "person_count": 0, "people": []}
    zones = [{"zone_id": "test", "name": "Test", "enabled": True,
              "points": [[.2, .2], [.8, .2], [.8, .8], [.2, .8]]}]
    detector = CountingDetector()
    zone_result = analyze_video(str(source), 1.0, 30, detector, zones=zones,
                                zone_sample_seconds=.5)
    assert detector.calls == 2
    assert zone_result["sample_interval_seconds"] == .5
    detector = CountingDetector()
    normal_result = analyze_video(str(source), 1.0, 30, detector,
                                  zones=[{**zones[0], "enabled": False}],
                                  zone_sample_seconds=.5)
    assert detector.calls == 1
    assert normal_result["sample_interval_seconds"] == 1.0


def test_object_sampling_summary_and_csv_use_actual_structured_results(tmp_path):
    source = tmp_path / "sampled.avi"
    make_video(source)
    class FixtureObjectDetector:
        METHOD = "fixture detector"
        confidence = .25
        def detect(self, _frame, timestamp, frame_number):
            return [{"event_type": "object_detection", "label": "person",
                     "timestamp_seconds": timestamp, "frame_number": frame_number,
                     "confidence": .9, "anomaly_score": None, "track_id": 3,
                     "detection_method": self.METHOD,
                     "details": {"bbox_xyxy": [1, 2, 20, 40]}}]
        def annotate(self, frame, _observations): return frame
    result = analyze_video(str(source), .2, 30, object_detector=FixtureObjectDetector())
    assert result["object_analysis"]["frames_sampled"] == 5
    assert result["object_analysis"]["counts_by_class"] == {"person": 5}
    assert result["object_analysis"]["person_track_count_estimate"] == 1
    assert [event["timestamp_seconds"] for event in result["events"]] == [0, .2, .4, .6, .8]
    summary = build_analysis_summary(result, .2)
    assert "5 frame observations" in summary and "not unique identities" in summary
    video = {"_id": ObjectId(), "original_filename": "sampled.avi", "processing_status": "complete",
             "analysis_summary": summary, "analysis_settings": {"sample_every_seconds": .2}}
    csv_text = video_report_csv(video, result["events"])
    assert "object_detection" in csv_text and "person" in csv_text and "0.9" in csv_text


def test_background_worker_persists_only_detector_emitted_events(client, app, tmp_path, monkeypatch):
    import app.routes.videos as video_routes
    model_marker = tmp_path / "configured-model.pt"
    model_marker.write_bytes(b"test marker")
    app.config["POSE_MODEL_PATH"] = str(model_marker)
    class FixtureDetector:
        METHOD = "synthetic detector fixture"
        def inspect(self, frame, timestamp):
            return frame, {"event_type": "potential_fall", "timestamp_seconds": timestamp,
                           "confidence": None, "anomaly_score": .8, "detection_method": self.METHOD,
                           "metadata": {"source": "synthetic test fixture"}}
        def inspect_with_evidence(self, frame, timestamp):
            overlay, event = self.inspect(frame, timestamp)
            evidence = {"timestamp_seconds": timestamp, "person_count": 1,
                        "people": [{"keypoints": {"left_shoulder": {"x": 12.0, "y": 10.0,
                                                                        "confidence": .9}}}],
                        "transition_status": "candidate_emitted", "model_name": "fixture-pose",
                        "model_version": "test"}
            return overlay, event, evidence
    monkeypatch.setattr(video_routes, "YoloPoseFallDetector", lambda _path, _settings=None: FixtureDetector())
    payload = make_video(tmp_path / "pose-test.avi")
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "pose-test.avi")}, content_type="multipart/form-data")
    video_id = response.headers["Location"].rsplit("/", 1)[-1]
    row = wait_for_status(app, video_id)
    assert row["analysis_mode"] == "pose_heuristic"
    event = app.extensions["mongo_db"].detection_events.find_one({"video_id": ObjectId(video_id)})
    assert event["event_type"] == "potential_fall" and event["confidence"] is None
    assert event["anomaly_score"] == .8 and event["metadata"]["source"] == "synthetic test fixture"
    assert row["pose_analysis"]["status"] == "complete"
    pose_evidence = app.extensions["mongo_db"].pose_detection_frames.find_one({"video_id": ObjectId(video_id)})
    assert pose_evidence["timestamp_seconds"] == 0
    assert pose_evidence["people"][0]["keypoints"]["left_shoulder"]["confidence"] == .9
    results = client.get(f"/results/{video_id}")
    assert b"synthetic detector fixture" in results.data
    assert b"Inspect sampled pose evidence" in results.data
    if row.get("processed_filename"):
        assert b"Annotated analysis output" in results.data


def test_processed_path_is_restricted_to_generated_name(tmp_path):
    root = tmp_path / "processed"
    root.mkdir()
    name = "507f1f77bcf86cd799439011_annotated.mp4"
    (root / name).write_bytes(b"video")
    assert safe_processed_path({"processed_filename": name}, str(root)) == str((root / name).resolve())
    webm_name = "507f1f77bcf86cd799439011_annotated.webm"
    (root / webm_name).write_bytes(b"webm")
    assert safe_processed_path({"processed_filename": webm_name}, str(root)) == str((root / webm_name).resolve())
    versioned = "507f1f77bcf86cd799439011_annotated_a1b2c3d4.webm"
    (root / versioned).write_bytes(b"webm")
    assert safe_processed_path({"processed_filename": versioned}, str(root)) == str((root / versioned).resolve())
    outside = tmp_path / "private.txt"
    outside.write_text("keep")
    assert safe_processed_path({"processed_filename": str(outside)}, str(root)) is None


def test_original_mp4_route_has_range_mime_and_missing_file_behavior(client, app, tmp_path):
    video_id = ObjectId()
    internal_name = "a" * 32 + ".mp4"
    upload_path = tmp_path / "uploads" / internal_name
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(upload_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV MP4V test writer is unavailable")
    for index in range(10):
        writer.write(np.full((48, 64, 3), index * 18, dtype=np.uint8))
    writer.release()
    app.config["UPLOAD_DIR"] = str(upload_path.parent)
    app.extensions["mongo_db"].videos.insert_one({
        "_id": video_id, "original_filename": "source-check.mp4",
        "stored_filename": internal_name, "file_path": internal_name,
        "processing_status": "complete",
    })

    response = client.get(f"/media/{video_id}", headers={"Range": "bytes=0-63"})
    assert response.status_code == 206
    assert response.mimetype == "video/mp4"
    assert response.headers["Content-Length"] == "64"
    assert response.headers["Content-Range"].startswith("bytes 0-63/")
    assert len(response.data) == 64
    assert client.get(f"/media/{ObjectId()}").status_code == 404


def test_historical_webm_route_keeps_range_playback_and_reports_missing_output(client, app, tmp_path):
    video_id = ObjectId()
    source = tmp_path / "legacy-source.mp4"
    webm = tmp_path / "processed" / f"{video_id}_annotated.webm"
    webm.parent.mkdir(parents=True, exist_ok=True)
    make_video(source)
    assert make_browser_playable_webm(str(source), str(webm))
    app.config["PROCESSED_DIR"] = str(webm.parent)
    app.extensions["mongo_db"].videos.insert_one({
        "_id": video_id, "original_filename": "legacy-source.mp4",
        "stored_filename": "not-used.mp4", "file_path": "not-used.mp4",
        "processing_status": "complete", "processed_filename": webm.name,
    })

    response = client.get(f"/processed/{video_id}", headers={"Range": "bytes=0-63"})
    assert response.status_code == 206
    assert response.mimetype in {"video/mp4", "video/webm"}
    assert response.headers["Content-Length"] == "64"
    assert response.headers["Content-Range"].startswith("bytes 0-63/")
    assert webm.exists()  # Historical WebM is preserved; the route serves a compatibility copy.
    is_h264_copy = response.mimetype == "video/mp4"
    if is_h264_copy:
        assert response.data[4:8] == b"ftyp"
    browser_copy = tmp_path / ("browser-copy.mp4" if is_h264_copy else "browser-copy.webm")
    full_response = client.get(f"/processed/{video_id}")
    browser_copy.write_bytes(full_response.data)
    if is_h264_copy:
        import av
        container = av.open(str(browser_copy))
        stream = container.streams.video[0]
        frames = list(container.decode(stream))
        assert stream.codec_context.name == "h264"
        assert stream.duration * stream.time_base == pytest.approx(1.0, abs=.1)
        assert len(frames) == 10
        container.close()
    assert client.get(f"/processed/{ObjectId()}").status_code == 404


def test_results_layout_contains_balanced_media_and_scrollable_evidence(client, app):
    video_id = app.extensions["mongo_db"].videos.insert_one(
        video_doc("layout-check.mp4", "complete")).inserted_id
    app.extensions["mongo_db"].detection_events.insert_one({
        "video_id": video_id, "event_type": "potential_fall", "label": "review candidate",
        "timestamp_seconds": 1.5, "details": {},
    })
    page = client.get(f"/results/{video_id}")
    css = client.get("/static/css/style.css").data
    js = client.get("/static/js/results.js").data
    assert page.status_code == 200
    assert b'class="result-media-grid"' in page.data
    assert b'class="panel analysis-summary-panel"' in page.data
    assert b'class="timeline-table"' in page.data
    assert b"grid-template-columns:repeat(2,minmax(0,1fr))" in css
    assert b"@media(max-width:820px)" in css
    assert b".player-retry[hidden]{display:none!important}" in css
    assert b"overflow-x:auto" in css and b"min-width:1120px" in css
    assert b"loadeddata" in js
    assert b"video.error" in js and b"video.load()" in js
    assert b"addEventListener('stalled'" in js and b"false);" in js
    assert b"stateVersion" in js and b"browserMessage" in js
    assert b"data-seconds" in page.data


def test_annotated_publish_never_overwrites_an_existing_artifact(tmp_path):
    from app.services.video_processor import _publish_without_overwrite
    temporary = tmp_path / "partial.webm"
    target = tmp_path / "saved.webm"
    temporary.write_bytes(b"new validated output")
    target.write_bytes(b"historical output")

    published, reason = _publish_without_overwrite(str(temporary), str(target))

    assert not published and reason == "exists"
    assert target.read_bytes() == b"historical output"
    assert temporary.exists()


def test_legacy_mp4v_result_is_converted_to_browser_playable_webm(client, app, tmp_path):
    video_id = ObjectId()
    filename = f"{video_id}_annotated.mp4"
    source = tmp_path / "processed" / filename
    source.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV MP4V test writer is unavailable")
    for index in range(5):
        writer.write(np.full((48, 64, 3), index * 25, dtype=np.uint8))
    writer.release()
    app.config["PROCESSED_DIR"] = str(source.parent)
    app.extensions["mongo_db"].videos.insert_one({
        "_id": video_id, "original_filename": "legacy.avi", "stored_filename": "not-used.avi",
        "file_path": "not-used.avi", "processing_status": "complete", "processed_filename": filename,
    })

    response = client.get(f"/processed/{video_id}")

    assert response.status_code == 200
    assert response.mimetype == "video/webm"
    converted = tmp_path / "round-trip.webm"
    converted.write_bytes(response.data)
    capture = cv2.VideoCapture(str(converted))
    ok, frame = capture.read()
    capture.release()
    assert ok and frame is not None and frame.size > 0
    assert source.exists()  # Conversion preserves the original stored result.


def test_missing_configured_pose_model_is_reported_as_partial_success(client, app, tmp_path):
    app.config["POSE_MODEL_PATH"] = str(tmp_path / "missing-weights.pt")
    payload = make_video(tmp_path / "missing-model.avi")
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "missing-model.avi")}, content_type="multipart/form-data")
    video_id = response.headers["Location"].rsplit("/", 1)[-1]
    row = wait_for_status(app, video_id)
    assert row["processing_status"] == "complete"
    assert row["pose_analysis"]["status"] == "unavailable"
    assert "weights" in row["pose_analysis"]["error_message"].lower()
    assert row["analysis_mode"] == "validation_only"


def test_safe_video_path_requires_generated_internal_name(tmp_path):
    folder = tmp_path / "uploads"
    folder.mkdir()
    good_name = "a" * 32 + ".mp4"
    good = folder / good_name
    good.write_bytes(b"video")
    assert safe_video_path({"stored_filename": good_name, "file_path": good_name}, str(folder)) == str(good.resolve())
    outside = tmp_path / "important.txt"
    outside.write_text("keep")
    assert safe_video_path({"stored_filename": str(outside), "file_path": str(outside)}, str(folder)) is None
    assert outside.read_text() == "keep"


def test_history_search_filter_event_and_csv(client, app):
    with app.app_context():
        one = create_video(video_doc("fall-study.mp4", "complete"))
        create_video(video_doc("scene-two.avi", "failed"))
        db = app.extensions["mongo_db"]
        db.detection_events.insert_one({"video_id": one, "event_type": "prototype_test_event",
            "timestamp_seconds": 3.0, "confidence": None, "anomaly_score": None,
            "detection_method": "test-fixture", "created_at": datetime.now(timezone.utc)})
        stats = dashboard_stats()
        assert stats["events"] == 1 and stats["categories"][0]["category"] == "prototype_test_event"
        videos, events = report_data()
        assert len(videos) == 2 and len(events) == 1
        assert report_data("2099-01-01", "2099-01-02") == ([], [])
        assert list_videos("scene", "failed")[1] == 1
    response = client.get("/history?q=fall&status=complete")
    assert response.status_code == 200 and b"fall-study.mp4" in response.data and b"scene-two.avi" not in response.data
    csv_response = client.get("/reports.csv")
    assert csv_response.status_code == 200 and b"prototype_test_event" in csv_response.data


def test_delete_removes_only_video_and_its_events(app):
    with app.app_context():
        target = create_video(video_doc("remove.mp4"))
        keep = create_video(video_doc("keep.mp4"))
        db = app.extensions["mongo_db"]
        db.detection_events.insert_one({"video_id": target, "event_type": "x"})
        db.detection_events.insert_one({"video_id": keep, "event_type": "y"})
        deleted = delete_video(str(target))
        assert deleted["_id"] == target
        assert get_video(str(target)) is None
        assert db.detection_events.count_documents({"video_id": target}) == 0
        assert get_video(str(keep)) is not None
        assert db.detection_events.count_documents({"video_id": keep}) == 1


def test_delete_route_removes_owned_file(client, app, tmp_path):
    payload = make_video(tmp_path / "delete-me.avi")
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "delete-me.avi")}, content_type="multipart/form-data")
    video_id = response.headers["Location"].rsplit("/", 1)[-1]
    wait_for_status(app, video_id)
    record = app.extensions["mongo_db"].videos.find_one({})
    stored_path = os.path.join(app.config["UPLOAD_DIR"], record["stored_filename"])
    assert os.path.isfile(stored_path)
    deleted = client.post(f"/history/{video_id}/delete", data={"csrf_token": csrf(client)})
    assert deleted.status_code == 302
    assert not os.path.exists(stored_path)
    assert app.extensions["mongo_db"].videos.count_documents({}) == 0


def test_csrf_and_unsupported_upload(client):
    assert client.post("/history/no/delete").status_code == 400
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(b"not a video"), "bad.exe")}, content_type="multipart/form-data", follow_redirects=True)
    assert b"Supported extensions" in response.data


def test_upload_saves_processing_result_and_database_error_cleanup(client, app, tmp_path, monkeypatch):
    payload = make_video(tmp_path / "small.avi")
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "sample.avi")}, content_type="multipart/form-data")
    assert response.status_code == 302
    video_id = response.headers["Location"].rsplit("/", 1)[-1]
    saved = wait_for_status(app, video_id)
    job = app.extensions["mongo_db"].processing_jobs.find_one({"video_id": ObjectId(video_id)})
    assert job["status"] == "completed" and job["progress_percent"] == 100
    assert saved["active_job_id"] == job["_id"]
    result_page = client.get(response.headers["Location"])
    assert result_page.status_code == 200 and b"Video validation complete" in result_page.data
    assert b"OpenCV validation passed" in result_page.data
    assert b"Face detection" in result_page.data
    assert saved["file_size"] == len(payload)
    assert saved["validation_status"] == "valid"
    assert (saved["width"], saved["height"], saved["frame_count"]) == (64, 48, 10)
    assert saved["face_analysis"]["status"] == "complete"
    assert saved["face_analysis"]["frames_sampled"] > 0
    original = client.get(f"/media/{video_id}")
    download = client.get(f"/media/{video_id}/download")
    assert original.status_code == 200 and original.data == payload
    assert download.status_code == 200 and download.data == payload
    assert "attachment" in download.headers["Content-Disposition"]
    assert saved.get("processed_filename")
    assert saved["processed_filename"].endswith(("_annotated.mp4", "_annotated.webm"))
    annotated = client.get(f"/processed/{video_id}")
    annotated_download = client.get(f"/processed/{video_id}/download")
    assert annotated.status_code == 200 and annotated.data
    assert annotated.data != original.data
    assert annotated.mimetype in {"video/mp4", "video/webm"}
    assert saved["annotated_output_status"] == "ready"
    output_metadata = saved["annotated_output_metadata"]
    assert output_metadata["container"] in {".mp4", ".webm"}
    if output_metadata["container"] == ".mp4":
        assert "H.264" in output_metadata["codec"]
    assert output_metadata["frame_count"] == 10
    assert output_metadata["duration_seconds"] == pytest.approx(1.0, abs=.1)
    assert output_metadata["width"] == 64 and output_metadata["height"] == 48
    assert output_metadata["sampled_frames_decoded"] == [True, True, True]
    byte_range = client.get(f"/processed/{video_id}", headers={"Range": "bytes=0-31"})
    assert byte_range.status_code == 206
    assert byte_range.headers["Content-Range"].startswith("bytes 0-31/")
    assert len(byte_range.data) == 32
    assert byte_range.mimetype == annotated.mimetype
    # Verify that the exact streamed result is a decodable video, not an empty
    # container that would render as the browser's black video player.
    annotated_path = tmp_path / "served-annotated.mp4"
    annotated_path.write_bytes(annotated.data)
    decoded = cv2.VideoCapture(str(annotated_path))
    count = int(decoded.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_ok = []
    for position in (0, count // 2, max(0, count - 1)):
        decoded.set(cv2.CAP_PROP_POS_FRAMES, position)
        ok, frame = decoded.read()
        frames_ok.append(bool(ok and frame is not None and frame.size > 0))
    decoded.release()
    assert frames_ok == [True, True, True]
    assert annotated_download.status_code == 200
    assert "attachment" in annotated_download.headers["Content-Disposition"]
    assert app.extensions["mongo_db"].face_detection_frames.count_documents({"video_id": ObjectId(video_id)}) > 0
    restarted = create_app({"TESTING": True, "SECRET_KEY": "test-secret",
        "MONGO_CLIENT": app.extensions["mongo_client"], "MONGO_DB_NAME": "test_human_anomaly",
        "MONGO_PING_ON_STARTUP": False, "UPLOAD_DIR": app.config["UPLOAD_DIR"],
        "PROCESSED_DIR": app.config["PROCESSED_DIR"]})
    assert restarted.test_client().get(f"/media/{video_id}").data == payload
    restarted.extensions["processing_executor"].shutdown(wait=True, cancel_futures=True)
    assert app.extensions["mongo_db"].videos.count_documents({"processing_status": "complete"}) == 1
    status = client.get(f"/api/videos/{video_id}/status")
    assert status.status_code == 200 and status.json["status"] == "complete"
    job = app.extensions["mongo_db"].processing_jobs.find_one({"video_id": ObjectId(video_id)})
    assert job["progress_percent"] == 100 and job["completed_at"] is not None

    import app.routes.videos as video_routes
    monkeypatch.setattr(video_routes, "create_video", lambda _doc: (_ for _ in ()).throw(DatabaseUnavailable("offline")))
    before = set(os.listdir(app.config["UPLOAD_DIR"]))
    failed = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "orphan.avi")}, content_type="multipart/form-data")
    assert failed.status_code == 302
    assert len(list((tmp_path / "uploads").iterdir())) == len(before) + 1
    assert b"orphan.avi" in client.get("/history").data
    app.extensions["processing_executor"].shutdown(wait=True, cancel_futures=True)
    assert before.issubset(set(os.listdir(app.config["UPLOAD_DIR"])))
    assert len(set(os.listdir(app.config["UPLOAD_DIR"])) - before) == 1


def test_upload_request_id_prevents_duplicate_jobs_but_allows_intentional_repeat_uploads(
        client, app, tmp_path):
    payload = make_video(tmp_path / "repeatable-name.avi")
    token = "a" * 32
    first = client.post("/upload", data={"csrf_token": csrf(client),
        "upload_request_id": token,
        "video": (io.BytesIO(payload), "same-name.avi")}, content_type="multipart/form-data")
    duplicate = client.post("/upload", data={"csrf_token": csrf(client),
        "upload_request_id": token,
        "video": (io.BytesIO(payload), "same-name.avi")}, content_type="multipart/form-data")
    assert first.status_code == duplicate.status_code == 302
    assert first.headers["Location"] == duplicate.headers["Location"]
    video_id = first.headers["Location"].rsplit("/", 1)[-1]
    wait_for_status(app, video_id)
    db = app.extensions["mongo_db"]
    assert db.videos.count_documents({"upload_request_id": token}) == 1
    assert db.processing_jobs.count_documents({"video_id": ObjectId(video_id)}) == 1

    # A fresh form token represents a deliberate new analysis, even when the
    # user selects the same local filename and bytes again.
    repeated = client.post("/upload", data={"csrf_token": csrf(client),
        "upload_request_id": "b" * 32,
        "video": (io.BytesIO(payload), "same-name.avi")}, content_type="multipart/form-data")
    assert repeated.status_code == 302
    second_id = repeated.headers["Location"].rsplit("/", 1)[-1]
    assert second_id != video_id
    wait_for_status(app, second_id)
    assert db.videos.count_documents({"original_filename": "same-name.avi"}) == 2
    assert db.processing_jobs.count_documents({"video_id": {"$in": [ObjectId(video_id), ObjectId(second_id)]}}) == 2


def test_new_upload_is_immediately_visible_and_updates_same_history_record(
        client, app, tmp_path, monkeypatch):
    db = app.extensions["mongo_db"]
    with app.app_context():
        older_id = create_video({**video_doc("older-failed.avi", "failed"),
            "upload_date": datetime.now(timezone.utc) - timedelta(days=1)})
    queued_calls = []
    monkeypatch.setattr(app.extensions["processing_executor"], "submit",
                        lambda *args, **kwargs: queued_calls.append((args, kwargs)))
    payload = make_video(tmp_path / "newest-upload.avi")
    upload_page = client.get("/upload").get_data(as_text=True)
    request_key = upload_page.split('name="upload_request_id" value="', 1)[1].split('"', 1)[0]
    assert request_key
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "upload_request_id": request_key,
        "video": (io.BytesIO(payload), "newest-upload.avi")}, content_type="multipart/form-data")
    assert response.status_code == 302 and len(queued_calls) == 1
    video_id = response.headers["Location"].rsplit("/", 1)[-1]
    video = db.videos.find_one({"_id": ObjectId(video_id)})
    job = db.processing_jobs.find_one({"video_id": ObjectId(video_id)})
    assert video and video["processing_status"] == "queued"
    assert job and job["status"] == "queued"

    history = client.get("/history")
    assert history.status_code == 200
    page = history.get_data(as_text=True)
    table_body = page[page.index("<tbody>"):page.index("</tbody>")]
    assert table_body.index("newest-upload.avi") < table_body.index("older-failed.avi")
    assert client.get(f"/history?status=queued").data.find(b"newest-upload.avi") >= 0
    assert client.get(f"/history?status=failed").data.find(b"older-failed.avi") >= 0

    with app.app_context():
        for status, label in (("processing", "Processing"), ("complete", "Complete")):
            assert update_video(video_id, {"processing_status": status})
            refreshed = client.get("/history").data
            assert b"newest-upload.avi" in refreshed and label.encode() in refreshed
            assert db.videos.count_documents({"_id": ObjectId(video_id)}) == 1
            assert db.processing_jobs.count_documents({"video_id": ObjectId(video_id)}) == 1


def test_new_upload_form_gets_fresh_key_and_file_change_rotates_existing_key(client):
    first = client.get("/upload").get_data(as_text=True)
    second = client.get("/upload").get_data(as_text=True)
    first_key = first.split('name="upload_request_id" value="', 1)[1].split('"', 1)[0]
    second_key = second.split('name="upload_request_id" value="', 1)[1].split('"', 1)[0]
    assert first_key != second_key
    script = client.get("/static/js/main.js").get_data(as_text=True)
    assert "if(hasSelectedFile&&uploadRequestId)" in script
    assert "uploadRequestId.value=window.crypto?.randomUUID" in script


def test_local_fallback_upload_is_visible_before_worker_finishes(client, app, tmp_path, monkeypatch):
    from app.db import DatabaseUnavailable
    import app.routes.videos as video_routes
    monkeypatch.setattr(video_routes, "create_video",
                        lambda _document: (_ for _ in ()).throw(DatabaseUnavailable("offline")))
    monkeypatch.setattr(app.extensions["processing_executor"], "submit", lambda *args, **kwargs: None)
    payload = make_video(tmp_path / "local-pending.avi")
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "upload_request_id": "f" * 32,
        "video": (io.BytesIO(payload), "local-pending.avi")}, content_type="multipart/form-data")
    assert response.status_code == 302
    video_id = response.headers["Location"].rsplit("/", 1)[-1]
    page = client.get("/history").get_data(as_text=True)
    assert "local-pending.avi" in page and "Pending" in page
    from app.services.local_fallback import get_pending_video
    with app.app_context():
        local = get_pending_video(video_id)
    assert local and local["processing_status"] == "queued"
    assert (local.get("local_job") or {}).get("status") == "queued"


def test_history_merges_mongodb_and_local_rows_by_actual_utc_upload_time(app):
    from app.services.local_fallback import save_pending_upload, update_pending_video
    with app.app_context():
        current = datetime.now(timezone.utc)
        local_id, _job = save_pending_upload({"original_filename": "local-newest.avi",
            "stored_filename": "local-newest-internal.avi", "file_path": "local-newest-internal.avi"}, "offline")
        update_pending_video(local_id, {"upload_date": current + timedelta(minutes=1)})
        mongo_id = create_video({**video_doc("mongo-older.avi", "processing"), "upload_date": current})
        videos, total = list_videos()
        assert total == 2
        assert str(videos[0]["_id"]) == str(local_id)
        assert videos[1]["_id"] == mongo_id


def test_local_fallback_upload_request_id_is_idempotent(client, app):
    from app.services.local_fallback import save_pending_upload, list_pending_videos
    document = {"upload_request_id": "c" * 32, "original_filename": "offline.avi",
                "stored_filename": "first-copy.avi", "file_path": "first-copy.avi"}
    with app.app_context():
        video_id, job_id = save_pending_upload(document, "offline")
        duplicate_id, duplicate_job_id = save_pending_upload(
            {**document, "stored_filename": "duplicate-copy.avi", "file_path": "duplicate-copy.avi"}, "offline")
        assert duplicate_id == video_id and duplicate_job_id == job_id
        matching, _total = list_pending_videos(search="offline.avi", per_page=100)
        matching = [row for row in matching if row.get("upload_request_id") == "c" * 32]
        assert len(matching) == 1
        assert matching[0]["stored_filename"] == "first-copy.avi"


def test_invalid_video_is_rejected_before_database_or_job_creation(client, app, tmp_path):
    before_files = set(os.listdir(app.config["UPLOAD_DIR"]))
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(b"not a video"), "broken.mp4")},
        content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200
    assert b"could not open this video" in response.data.lower() or b"could not decode" in response.data.lower()
    assert app.extensions["mongo_db"].videos.count_documents({}) == 0
    assert app.extensions["mongo_db"].processing_jobs.count_documents({}) == 0
    assert set(os.listdir(app.config["UPLOAD_DIR"])) == before_files


def test_video_validator_checks_decodable_frame_and_extracts_metadata(tmp_path):
    video_path = tmp_path / "readable.avi"
    make_video(video_path)
    details = validate_video_file(str(video_path))
    assert details["frame_rate"] == 10
    assert details["frame_count"] == 10
    assert details["duration"] == 1
    assert (details["width"], details["height"]) == (64, 48)

    corrupt_path = tmp_path / "corrupt.mp4"
    corrupt_path.write_bytes(b"not a video")
    with pytest.raises(VideoValidationError):
        validate_video_file(str(corrupt_path))

    with pytest.raises(VideoValidationError):
        validate_video_file(str(tmp_path / "missing.mp4"))


@pytest.mark.parametrize("zones", [[], [{"zone_id": "front_gate", "name": "Front gate", "enabled": True,
    "points": [[.1, .1], [.9, .1], [.9, .9], [.1, .9]]}]])
def test_valid_upload_completes_in_local_fallback_when_mongodb_write_fails(
        client, app, tmp_path, monkeypatch, zones):
    from app.db import DatabaseUnavailable
    import app.routes.videos as video_routes
    from app.services.local_fallback import get_pending_video
    monkeypatch.setattr(video_routes, "create_video",
                        lambda _document: (_ for _ in ()).throw(DatabaseUnavailable("temporary network failure")))
    app.extensions["mongo_available"] = False
    payload = make_video(tmp_path / "offline-upload.avi")
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "roi_zones": json.dumps(zones), "video": (io.BytesIO(payload), "offline-upload.avi")},
        content_type="multipart/form-data")
    assert response.status_code == 302
    video_id = response.location.rsplit("/", 1)[-1]
    assert app.extensions["mongo_db"].videos.count_documents({}) == 0
    local_files = list((tmp_path / "uploads").iterdir())
    assert len(local_files) == 1 and local_files[0].read_bytes() == payload
    row = wait_for_local_status(app, video_id)
    assert row["processing_status"] == "complete"
    assert row["local_job"]["status"] == "completed"
    assert row["local_job"]["progress_percent"] == 100
    assert row["pose_analysis"]["status"] == "not_configured"
    assert row["face_analysis"]["status"] == "complete"
    assert row["zones"] == zones
    history = client.get("/history")
    assert history.status_code == 200 and b"offline-upload.avi" in history.data
    assert b"complete" in history.data
    assert b"offline-upload.avi" in client.get("/dashboard").data
    assert client.get("/reports").status_code == 200
    result = client.get(response.location)
    assert result.status_code == 200 and b"Primary database unavailable" in result.data
    assert b"local fallback" in result.data.lower()
    playback = client.get(f"/media/{video_id}")
    assert playback.status_code == 200 and playback.data == payload
    assert app.extensions["mongo_db"].processing_jobs.count_documents({}) == 0
    with app.app_context():
        assert get_pending_video(video_id)["local_only"] is True


def test_local_background_failure_is_persisted_and_reported(client, app, tmp_path, monkeypatch):
    from app.db import DatabaseUnavailable
    from app.services.video_processor import VideoError
    import app.routes.videos as video_routes
    monkeypatch.setattr(video_routes, "create_video",
        lambda _doc: (_ for _ in ()).throw(DatabaseUnavailable("offline")))
    monkeypatch.setattr(video_routes, "analyze_video",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(VideoError("fixture decode failure")))
    payload = make_video(tmp_path / "local-failure.avi")
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "local-failure.avi")}, content_type="multipart/form-data")
    video_id = response.location.rsplit("/", 1)[-1]
    row = wait_for_local_status(app, video_id)
    assert row["processing_status"] == "failed"
    assert row["local_job"]["status"] == "failed"
    assert "fixture decode failure" in row["error_message"]
    assert client.get(f"/api/videos/{video_id}/status").json["status"] == "failed"
    results = client.get(response.location)
    assert results.status_code == 200 and b"fixture decode failure" in results.data


def test_mocked_fighting_candidate_persists_to_results_history_dashboard_and_reports(
        client, app, tmp_path, monkeypatch):
    import app.routes.videos as video_routes

    class FixtureFightDetector:
        METHOD = "fixture temporal classifier"

        def __init__(self, *_args, **_kwargs):
            self.status = "ready"
            self.error_message = None

        def observe_frame(self, *_args):
            pass

        def finish(self):
            event = {"event_type": "SUSPECTED_FIGHTING", "label": "Mock candidate for test only",
                     "timestamp_seconds": .2, "start_timestamp_seconds": .2,
                     "end_timestamp_seconds": .8, "frame_number": 2,
                     "model_score": .73, "anomaly_score": None, "confidence": None,
                     "track_id": None, "detection_method": self.METHOD,
                     "details": {"model_score": .73, "score_kind": "test fixture",
                                 "zone_association_status": "unavailable"}}
            summary = {"status": "complete", "method": self.METHOD,
                       "model_name": "fixture-only", "model_revision": "test",
                       "frames_sampled": 16, "windows_analyzed": 2,
                       "candidate_event_count": 1, "maximum_model_score": .73,
                       "decision_threshold": .4, "window_seconds": 4,
                       "window_stride_seconds": 1, "confirmation_windows": 2,
                       "error_message": None,
                       "limitation": "Fixture output, not a real prediction."}
            return [event], summary

    monkeypatch.setattr(video_routes, "FightActionDetector", FixtureFightDetector)
    app.config.update({"FIGHT_DETECTION_ENABLED": True,
                       "FIGHT_MODEL_PATH": str(tmp_path / "fixture-weights.pt")})
    payload = make_video(tmp_path / "fight-workflow.avi")
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "fight-workflow.avi")},
        content_type="multipart/form-data")
    assert response.status_code == 302
    video_id = response.location.rsplit("/", 1)[-1]
    row = wait_for_status(app, video_id)
    assert row["processing_status"] == "complete"
    assert row["fight_analysis"]["status"] == "complete"
    saved = app.extensions["mongo_db"].detection_events.find_one(
        {"video_id": ObjectId(video_id), "event_type": "SUSPECTED_FIGHTING"})
    assert saved and saved["model_score"] == .73
    result_page = client.get(response.location)
    assert b"Suspected Violence-Related Activity" in result_page.data
    assert b"0.730 uncalibrated model score" in result_page.data
    assert b"Mock candidate for test only" in result_page.data
    assert b"00:00.20" in result_page.data and b"00:00.80" in result_page.data
    video_csv = client.get(f"/results/{video_id}/report.csv")
    assert b"SUSPECTED_FIGHTING" in video_csv.data
    assert b"00:00.20-00:00.80" in video_csv.data
    assert b"00:00.20" in client.get("/reports").data
    assert b"00:00.20-00:00.80" in client.get("/reports.csv").data
    assert b"SUSPECTED_FIGHTING" in client.get(f"/results/{video_id}/report.pdf").data
    assert b"1" in client.get("/dashboard").data
    assert b"Mock candidate for test only" in client.get("/incidents").data


def test_suspected_fighting_event_is_reviewable_in_local_fallback(app):
    from app.services.local_fallback import save_pending_upload, list_pending_incidents
    from app.services.repository import save_events

    with app.app_context():
        video_id, job_id = save_pending_upload({
            "original_filename": "offline-review.mp4", "stored_filename": "offline.mp4",
            "file_path": "offline.mp4", "processing_status": "complete",
            "fight_analysis": {"status": "complete", "candidate_event_count": 1},
            "local_only": True,
        }, "Atlas unavailable")
        assert save_events(video_id, job_id, [{
            "event_type": "SUSPECTED_FIGHTING", "timestamp_seconds": 1.0,
            "start_timestamp_seconds": 1.0, "end_timestamp_seconds": 2.0,
            "model_score": .68, "detection_method": "fixture temporal model",
            "details": {"zone_association_status": "unavailable"},
        }]) == 1
        rows = list_pending_incidents()
        assert len(rows) == 1 and rows[0]["event_type"] == "SUSPECTED_FIGHTING"
        assert rows[0]["model_score"] == .68
        stats = dashboard_stats()
        assert stats["suspected_fighting_events"] == 1


def test_missing_fighting_weights_do_not_break_video_upload(client, app, tmp_path):
    app.config.update({"FIGHT_DETECTION_ENABLED": True,
                       "FIGHT_MODEL_PATH": str(tmp_path / "weights-not-installed.pt")})
    payload = make_video(tmp_path / "without-fight-weights.avi")
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "without-fight-weights.avi")},
        content_type="multipart/form-data")
    assert response.status_code == 302
    video_id = response.location.rsplit("/", 1)[-1]
    row = wait_for_status(app, video_id)
    assert row["processing_status"] == "complete"
    assert row["fight_analysis"]["status"] == "unavailable"
    assert row["fight_analysis"]["candidate_event_count"] == 0
    assert app.extensions["mongo_db"].detection_events.count_documents(
        {"video_id": ObjectId(video_id), "event_type": "SUSPECTED_FIGHTING"}) == 0
    assert b"not found" in client.get(response.location).data.lower()


def test_startup_resumes_legacy_waiting_local_job(app, tmp_path):
    from app.services.local_fallback import save_pending_upload, update_pending_video
    payload_path = tmp_path / "startup-local.avi"
    payload = make_video(payload_path)
    stored = "a" * 32 + ".avi"
    (Path(app.config["UPLOAD_DIR"]) / stored).write_bytes(payload)
    document = {"original_filename": "startup-local.avi", "stored_filename": stored,
                "file_path": stored, "file_size": len(payload), "processing_status": "queued",
                "zones": [], "pose_analysis": {"status": "queued"},
                "face_analysis": {"status": "queued"}, "object_analysis": {"status": "not_configured"}}
    with app.app_context():
        video_id, _job_id = save_pending_upload(document, "Atlas temporarily unavailable")
        update_pending_video(video_id, {"processing_status": "waiting_for_mongodb",
            "local_job.status": "waiting_for_mongodb"})

    restarted = create_app({"TESTING": False, "SECRET_KEY": "startup-test", "MONGO_URI": "",
        "MONGO_DB_NAME": "offline_startup", "MONGO_PING_ON_STARTUP": False,
        "UPLOAD_DIR": app.config["UPLOAD_DIR"], "PROCESSED_DIR": str(tmp_path / "restart-processed"),
        "LOCAL_FALLBACK_PATH": app.config["LOCAL_FALLBACK_PATH"],
        "POSE_MODEL_PATH": "", "PROCESSING_WORKERS": 1, "PROCESSING_QUEUE_SIZE": 1,
        "MAX_DURATION_SECONDS": 30, "SAMPLE_EVERY_SECONDS": .1})
    try:
        row = wait_for_local_status(restarted, video_id)
        assert row["processing_status"] == "complete"
        assert row["local_job"]["status"] == "completed"
        assert restarted.test_client().get(f"/results/{video_id}").status_code == 200
    finally:
        restarted.extensions["processing_executor"].shutdown(wait=True, cancel_futures=True)


def test_upload_persists_object_observations_tracks_summary_and_report(client, app, tmp_path, monkeypatch):
    import app.routes.videos as video_routes
    model_path = tmp_path / "local-object-model.pt"
    model_path.write_bytes(b"test marker")
    app.config["OBJECT_MODEL_PATH"] = str(model_path)
    app.config["OBJECT_MODEL_CONFIDENCE"] = .3
    class FixtureObjectDetector:
        METHOD = "fixture object detector"
        confidence = .3
        def __init__(self, *_args): pass
        def detect(self, _frame, timestamp, frame_number):
            return [
                {"event_type": "object_detection", "label": "person", "timestamp_seconds": timestamp,
                 "frame_number": frame_number, "confidence": .91, "anomaly_score": None, "track_id": 8,
                 "detection_method": self.METHOD, "details": {"bbox_xyxy": [1, 2, 20, 40]}},
                {"event_type": "object_detection", "label": "car", "timestamp_seconds": timestamp,
                 "frame_number": frame_number, "confidence": .82, "anomaly_score": None, "track_id": None,
                 "detection_method": self.METHOD, "details": {"bbox_xyxy": [22, 8, 48, 30]}},
            ]
        def annotate(self, frame, _observations): return frame.copy()
    monkeypatch.setattr(video_routes, "YoloObjectTracker", FixtureObjectDetector)
    payload = make_video(tmp_path / "object-upload.avi")
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "object-upload.avi")}, content_type="multipart/form-data")
    assert response.status_code == 302
    video_id = response.headers["Location"].rsplit("/", 1)[-1]
    saved = wait_for_status(app, video_id)
    assert saved["object_analysis"]["status"] == "complete"
    assert saved["object_analysis"]["counts_by_class"] == {"car": 10, "person": 10}
    assert saved["object_analysis"]["person_track_count_estimate"] == 1
    assert saved["event_count"] == 0 and saved["object_detection_count"] == 20
    object_event = app.extensions["mongo_db"].detection_events.find_one(
        {"video_id": ObjectId(video_id), "label": "person"})
    assert object_event["confidence"] == .91 and object_event["anomaly_score"] is None
    assert object_event["timestamp_seconds"] == 0 and object_event["frame_number"] == 0
    assert object_event["track_id"] == 8 and object_event["details"]["bbox_xyxy"] == [1, 2, 20, 40]
    assert "car: 10 frame observations" in saved["analysis_summary"]
    assert "person: 10 frame observations" in saved["analysis_summary"]
    page = client.get(response.headers["Location"])
    assert page.status_code == 200 and b"person tracking" in page.data
    assert b"event-type-filter" in page.data and b"event-track-filter" in page.data
    assert b"event-time-from" in page.data and b"timeline-event" in page.data
    report = client.get(f"/results/{video_id}/report.csv")
    assert report.status_code == 200 and b"object_detection" in report.data and b"person" in report.data
    dashboard = client.get("/dashboard")
    assert b"<strong>1</strong>" in dashboard.data
    with app.app_context():
        stats = dashboard_stats()
    assert stats["object_observations"] == 20 and stats["person_track_estimate"] == 1
    assert {item["label"]: item["count"] for item in stats["object_categories"]} == {"car": 10, "person": 10}
    assert app.extensions["mongo_db"].detection_events.count_documents(
        {"video_id": ObjectId(video_id), "event_type": {"$ne": "object_detection"}}) == 0


def test_opencv_face_detector_loads_bundled_cascade_and_handles_empty_frame():
    assert face_detector_status()["ready"] is True
    detector = OpenCVFaceDetector()
    assert detector.detect(np.zeros((120, 160, 3), dtype=np.uint8)) == []
    assert detector.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []


def test_face_detector_missing_config_is_reported_without_download(tmp_path):
    missing = str(tmp_path / "not-present.xml")
    assert face_detector_status(missing)["ready"] is False
    with pytest.raises(FaceDetectorUnavailable, match="FACE_CASCADE_PATH"):
        OpenCVFaceDetector(missing)


def test_missing_face_cascade_does_not_block_other_video_processing(client, app, tmp_path):
    app.config["FACE_CASCADE_PATH"] = str(tmp_path / "missing-cascade.xml")
    payload = make_video(tmp_path / "missing-cascade.avi")
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "missing-cascade.avi")}, content_type="multipart/form-data")
    video_id = response.headers["Location"].rsplit("/", 1)[-1]
    row = wait_for_status(app, video_id)
    assert row["processing_status"] == "complete"
    assert row["face_analysis"]["status"] == "unavailable"
    assert row["analysis_mode"] == "validation_only"


def test_face_boxes_support_multiple_results_and_are_annotated():
    class TwoFaces:
        def detectMultiScale(self, *_args, **_kwargs):
            return np.array([[4, 5, 20, 21], [40, 10, 22, 22]])
    detector = OpenCVFaceDetector.__new__(OpenCVFaceDetector)
    detector.cascade = TwoFaces()
    frame = np.zeros((80, 90, 3), dtype=np.uint8)
    boxes = detector.detect(frame)
    assert len(boxes) == 2 and boxes[0]["width"] == 20
    marked = detector.annotate(frame, boxes)
    assert np.any(marked != frame) and np.all(frame == 0)


def test_face_detector_initialization_error_does_not_fail_video_processing(client, app, tmp_path, monkeypatch):
    import app.routes.videos as video_routes
    class BrokenFaceDetector:
        METHOD = OpenCVFaceDetector.METHOD
        def __init__(self, _path=""):
            raise RuntimeError("private OpenCV detail")
    monkeypatch.setattr(video_routes, "OpenCVFaceDetector", BrokenFaceDetector)
    payload = make_video(tmp_path / "face-init-failure.avi")
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "face-init-failure.avi")}, content_type="multipart/form-data")
    video_id = response.headers["Location"].rsplit("/", 1)[-1]
    row = wait_for_status(app, video_id)
    assert row["processing_status"] == "complete"
    assert row["face_analysis"]["status"] == "unavailable"
    assert "private OpenCV detail" not in str(row["face_analysis"])


def test_annotated_writer_failure_does_not_break_original_video_processing(client, app, tmp_path, monkeypatch):
    payload = make_video(tmp_path / "writer-failure.avi")
    monkeypatch.setattr(cv2, "VideoWriter", lambda *_args, **_kwargs:
                        (_ for _ in ()).throw(cv2.error("writer unavailable")))
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "writer-failure.avi")}, content_type="multipart/form-data")
    video_id = response.headers["Location"].rsplit("/", 1)[-1]
    row = wait_for_status(app, video_id)
    assert row["processing_status"] == "complete"
    assert not row.get("processed_filename")
    assert client.get(f"/media/{video_id}").data == payload


def test_failed_video_retry_claims_one_worker_and_persists_new_job(client, app, tmp_path):
    payload = make_video(tmp_path / "retry-source.avi")
    stored = "a" * 32 + ".avi"
    path = os.path.join(app.config["UPLOAD_DIR"], stored)
    with open(path, "wb") as output:
        output.write(payload)
    doc = video_doc("retry-source.avi", "failed")
    doc.update({"stored_filename": stored, "file_path": stored, "file_size": len(payload)})
    with app.app_context():
        video_id = create_video(doc)
        db = app.extensions["mongo_db"]
        db.detection_events.insert_one({"video_id": video_id, "event_type": "partial_attempt"})
        db.face_detection_frames.insert_one({"video_id": video_id, "timestamp_seconds": 99, "face_count": 1})
    response = client.post(f"/history/{video_id}/retry", data={"csrf_token": csrf(client)})
    assert response.status_code == 302
    row = wait_for_status(app, str(video_id))
    assert row["processing_status"] == "complete"
    assert app.extensions["mongo_db"].processing_jobs.count_documents({"video_id": video_id}) == 1
    old_event = app.extensions["mongo_db"].detection_events.find_one(
        {"video_id": video_id, "event_type": "partial_attempt"})
    old_frame = app.extensions["mongo_db"].face_detection_frames.find_one(
        {"video_id": video_id, "timestamp_seconds": 99})
    assert old_event["superseded"] is True and old_frame["superseded"] is True
    assert app.extensions["mongo_db"].detection_events.count_documents(
        {"video_id": video_id, "superseded": {"$ne": True}}) == 0
    with app.app_context():
        assert dashboard_stats()["events"] == 0
        assert all(event.get("event_type") != "partial_attempt" for event in report_data()[1])
    assert b"partial_attempt" not in client.get("/reports.csv").data
    client.post(f"/history/{video_id}/retry", data={"csrf_token": csrf(client)})
    assert app.extensions["mongo_db"].processing_jobs.count_documents({"video_id": video_id}) == 1


def test_startup_marks_incomplete_jobs_interrupted_in_isolated_database(tmp_path):
    mongo = mongomock.MongoClient()
    db = mongo["interruption_test"]
    video_id = ObjectId()
    db.videos.insert_one({"_id": video_id, "processing_status": "processing"})
    db.processing_jobs.insert_one({"video_id": video_id, "status": "running"})
    application = create_app({"TESTING": True, "SECRET_KEY": "test", "MONGO_CLIENT": mongo,
        "MONGO_DB_NAME": "interruption_test", "MONGO_PING_ON_STARTUP": False,
        "UPLOAD_DIR": str(tmp_path / "u"), "PROCESSED_DIR": str(tmp_path / "p")})
    interrupted = db.videos.find_one({"_id": video_id})
    assert interrupted["processing_status"] == "interrupted"
    assert interrupted["pose_analysis"]["status"] == "interrupted"
    assert interrupted["object_analysis"]["status"] == "interrupted"
    job = db.processing_jobs.find_one({"video_id": video_id})
    assert job["status"] == "interrupted" and job["completed_at"] is not None
    application.extensions["processing_executor"].shutdown(wait=True, cancel_futures=True)
    mongo.close()


def test_face_frames_are_removed_with_only_their_video(app):
    with app.app_context():
        target = create_video(video_doc("face-target.mp4"))
        other = create_video(video_doc("face-other.mp4"))
        db = app.extensions["mongo_db"]
        db.face_detection_frames.insert_one({"video_id": target, "timestamp_seconds": 1, "face_count": 1})
        db.face_detection_frames.insert_one({"video_id": other, "timestamp_seconds": 2, "face_count": 0})
        delete_video(str(target))
        assert db.face_detection_frames.count_documents({"video_id": target}) == 0
        assert db.face_detection_frames.count_documents({"video_id": other}) == 1


def test_empty_video_upload_is_rejected_before_mongodb_insert(client, app):
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(b""), "empty.mp4")}, content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200 and b"file is empty" in response.data
    assert app.extensions["mongo_db"].videos.count_documents({}) == 0


def test_roi_is_optional_and_validates_normalized_coordinates():
    assert parse_zones(None) == []
    zone = {"zone_id": "zone_1", "name": "Restricted Area", "enabled": True,
            "points": [[.1, .1], [.8, .1], [.8, .8], [.1, .8]]}
    assert parse_zones(json.dumps([zone]))[0]["name"] == "Restricted Area"
    for invalid in (
        [{"points": [[.1, .1], [.2, .2]]}],
        [{"points": [[-.1, .1], [.8, .1], [.8, .8]]}],
        [{"points": [[.1, .1], [.1, .1], [.1, .1]]}],
    ):
        with pytest.raises(ZoneValidationError):
            parse_zones(json.dumps(invalid))


def test_roi_associates_localized_pose_candidate_with_inside_or_outside_zone():
    zones = parse_zones([{"zone_id": "restricted", "name": "Restricted Area", "enabled": True,
                          "points": [[.1, .1], [.6, .1], [.6, .9], [.1, .9]]}])
    inside = {"event_type": "potential_fall", "details": {"person_bbox_xyxy": [20, 10, 40, 60]}}
    outside = {"event_type": "potential_fall", "details": {"person_bbox_xyxy": [80, 10, 95, 60]}}
    unavailable = {"event_type": "potential_fall", "details": {}}
    assert associate_event_with_zones(inside, zones, 100, 100)["zone_association_status"] == "inside_active_zone"
    assert associate_event_with_zones(inside, zones, 100, 100)["zone_matches"] == ["restricted"]
    assert associate_event_with_zones(outside, zones, 100, 100)["zone_association_status"] == "outside_active_zones"
    assert associate_event_with_zones(unavailable, zones, 100, 100)["zone_association_status"] == "unavailable_no_person_location"


def test_roi_upload_persists_zone_and_reports_model_limitation(client, app, tmp_path):
    payload = make_video(tmp_path / "roi-sample.avi")
    zone = {"zone_id": "zone_1", "name": "Entry", "enabled": True,
            "points": [[.1, .1], [.9, .1], [.9, .9], [.1, .9]]}
    response = client.post("/upload", data={"csrf_token": csrf(client), "roi_zones": json.dumps([zone]),
        "video": (io.BytesIO(payload), "roi-sample.avi")}, content_type="multipart/form-data")
    assert response.status_code == 302
    video_id = response.headers["Location"].rsplit("/", 1)[-1]
    saved = wait_for_status(app, video_id)
    assert saved["zones"] == [zone]
    assert saved["zone_analysis"]["status"] == "unavailable"
    result = client.get(response.location)
    assert result.status_code == 200 and b"Smart surveillance zones" in result.data
    assert b"pose model produced localized anomaly events" in result.data


def test_roi_pipeline_checks_pose_candidate_and_annotates_derived_video(tmp_path):
    class LocalizedFixture:
        METHOD = "test pose localization"
        def inspect(self, frame, timestamp):
            return frame, {"event_type": "potential_fall", "timestamp_seconds": timestamp,
                           "details": {"person_bbox_xyxy": [20, 10, 40, 35]}}

    source = tmp_path / "roi-pipeline.avi"
    payload = make_video(source)
    zones = parse_zones([{"zone_id": "entry", "name": "Entry", "enabled": True,
                          "points": [[.1, .1], [.6, .1], [.6, .9], [.1, .9]]}])
    output = tmp_path / "roi-derived.webm"
    result = analyze_video(str(source), .2, 30, LocalizedFixture(), output_path=str(output), zones=zones)
    assert result["events"][0]["zone_association_status"] == "inside_active_zone"
    assert result["events"][0]["zone_matches"] == ["entry"]
    assert result["annotated_output"] is True
    cap = cv2.VideoCapture(str(output))
    try:
        ok, frame = cap.read()
        assert ok and frame is not None and frame.size
    finally:
        cap.release()


def test_roi_candidate_is_persisted_as_zone_specific_when_pose_localizes_it(client, app, tmp_path, monkeypatch):
    import app.routes.videos as video_routes
    marker = tmp_path / "pose-weights-marker.pt"
    marker.write_bytes(b"test only")
    app.config["POSE_MODEL_PATH"] = str(marker)

    class LocalizedFixture:
        METHOD = "test pose localization"
        def inspect_with_evidence(self, frame, timestamp):
            event = {"event_type": "potential_fall", "timestamp_seconds": timestamp,
                     "anomaly_score": .7, "confidence": None,
                     "details": {"person_bbox_xyxy": [20, 10, 40, 35]}}
            evidence = {"timestamp_seconds": timestamp, "person_count": 1, "people": [],
                        "transition_status": "candidate_emitted"}
            return frame, event, evidence

    monkeypatch.setattr(video_routes, "YoloPoseFallDetector", lambda *_args, **_kwargs: LocalizedFixture())
    payload = make_video(tmp_path / "localized.avi")
    zone = {"zone_id": "entry", "name": "Entry", "enabled": True,
            "points": [[.1, .1], [.6, .1], [.6, .9], [.1, .9]]}
    response = client.post("/upload", data={"csrf_token": csrf(client), "roi_zones": json.dumps([zone]),
        "video": (io.BytesIO(payload), "localized.avi")}, content_type="multipart/form-data")
    video_id = response.headers["Location"].rsplit("/", 1)[-1]
    row = wait_for_status(app, video_id)
    event = app.extensions["mongo_db"].detection_events.find_one({"video_id": ObjectId(video_id)})
    assert row["zone_analysis"]["status"] == "complete"
    assert row["zone_analysis"]["events_inside_active_zones"] > 0
    assert event["zone_association_status"] == "inside_active_zone" and event["zone_matches"] == ["entry"]
    result = client.get(response.location)
    assert b"Inside active zone: Entry" in result.data


def test_invalid_roi_is_rejected_before_file_or_database_write(client, app, tmp_path):
    payload = make_video(tmp_path / "bad-roi.avi")
    before_files = set(os.listdir(app.config["UPLOAD_DIR"]))
    bad = {"zone_id": "zone_1", "name": "Too small", "enabled": True,
           "points": [[.1, .1], [.8, .8]]}
    response = client.post("/upload", data={"csrf_token": csrf(client), "roi_zones": json.dumps([bad]),
        "video": (io.BytesIO(payload), "bad-roi.avi")}, content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200 and b"needs between 3 and 100 points" in response.data
    assert set(os.listdir(app.config["UPLOAD_DIR"])) == before_files
    assert app.extensions["mongo_db"].videos.count_documents({}) == 0


def test_queued_job_is_claimed_once_and_moves_through_explicit_states(app):
    from app.services.repository import update_job
    with app.app_context():
        job_id = create_job({"video_id": ObjectId(), "status": "queued", "created_at": datetime.now(timezone.utc)})
        claimed = claim_job(str(job_id))
        assert claimed and claimed["status"] == "processing"
        assert claim_job(str(job_id)) is None
        assert update_job(str(job_id), {"status": "completed", "progress_percent": 100})
        assert app.extensions["mongo_db"].processing_jobs.find_one({"_id": job_id})["status"] == "completed"


def test_completed_job_and_video_status_cannot_regress(app, client):
    from app.services.repository import update_job
    from app.services.local_fallback import save_pending_upload
    with app.app_context():
        video_id, job_id = save_pending_upload(video_doc("stable-status.mp4", "queued"), "test fallback")
        assert claim_job(job_id)["status"] == "processing"
        assert update_video(video_id, {"processing_status": "complete",
                                       "analysis_summary": "Persisted results"})
        assert update_job(job_id, {"status": "completed", "progress_percent": 100})
        assert update_job(job_id, {"status": "processing"}) is False
        assert update_job(job_id, {"status": "queued"}) is False
        assert update_video(video_id, {"processing_status": "processing"}) is False
        assert update_video(video_id, {"processing_status": "queued"}) is False
    responses = [client.get(f"/api/videos/{video_id}/status").json for _ in range(4)]
    assert [row["status"] for row in responses] == ["complete"] * 4
    assert responses[-1]["progress"] == 100


def test_status_api_does_not_trust_stale_completed_job_before_video_results(app, client):
    with app.app_context():
        video_id = create_video(video_doc("late-results.mp4", "processing"))
        job_id = create_job({"video_id": video_id, "status": "completed", "progress_percent": 100})
    result = client.get(f"/api/videos/{video_id}/status")
    assert result.status_code == 200
    assert result.json["status"] == "processing"


def test_video_timestamp_format_and_legacy_event_fields():
    from app.services.timestamps import event_video_seconds, format_video_timestamp
    assert event_video_seconds({"timestamp_seconds": "2.4"}) == 2.4
    assert event_video_seconds({"timestamp": 62.35}) == 62.35
    assert event_video_seconds({"created_at": datetime.now(timezone.utc)}) is None
    assert event_video_seconds({"timestamp_seconds": "nan", "timestamp": -1}) is None
    assert format_video_timestamp(2.4) == "00:02.40"
    assert format_video_timestamp(62.35) == "01:02.35"
    assert format_video_timestamp(3661.2) == "01:01:01.20"
    assert format_video_timestamp(float("inf")) is None


def test_legacy_zone_time_renders_from_stored_numeric_timestamp(client, app):
    with app.app_context():
        video_id = create_video(video_doc("legacy-zone.avi", "complete"))
        app.extensions["mongo_db"].detection_events.insert_one({
            "video_id": video_id, "event_type": "RESTRICTED_ZONE_ENTRY",
            "zone_name": "North gate", "track_id": 5, "timestamp": 2.4,
            "details": {"zone_name": "North gate", "person_bbox_xyxy": [4, 5, 10, 30]},
        })
    response = client.get(f"/results/{video_id}")
    assert response.status_code == 200
    assert b"00:02.40" in response.data
    history = client.get("/history")
    assert b"Latest confirmed entry:" in history.data and b"00:02.40" in history.data
    incidents = client.get("/incidents")
    assert b"00:02.40" in incidents.data


def test_zone_entry_records_formatted_video_timestamp_and_crossing_box():
    from app.services.video_processor import _zone_transition_events
    zone = [{"zone_id": "gate", "name": "North Gate", "enabled": True,
             "points": [[.3, .3], [.7, .3], [.7, .7], [.3, .7]]}]
    def sample(t, frame, x1, x2):
        return {"timestamp_seconds": t, "frame_number": frame, "people": [
            {"track_id": 7, "bbox_xyxy": [x1, 10, x2, 50], "detection_confidence": .88}]}
    rows = [sample(0, 0, 10, 20), sample(.5, 5, 10, 20),
            sample(1.0, 10, 35, 45), sample(1.5, 15, 35, 45)]
    entries = [event for event in _zone_transition_events(rows, zone, 100, 100)
               if event["event_type"] == "RESTRICTED_ZONE_ENTRY"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["timestamp_seconds"] == 1.0 and entry["video_timestamp"] == "00:01.00"
    assert entry["zone_id"] == "gate" and entry["zone_name"] == "North Gate"
    assert entry["details"]["person_bbox_xyxy"] == [35, 10, 45, 50]
    assert entry["details"]["crossing_bbox_xyxy"] == [35, 10, 45, 50]


def test_initial_zone_occupancy_does_not_show_a_fabricated_entry_time(client, app):
    with app.app_context():
        video_id = create_video({**video_doc("occupancy.avi", "complete"), "zones": [
            {"zone_id": "z", "name": "Loading bay", "enabled": True, "points": []}]})
        app.extensions["mongo_db"].detection_events.insert_one({
            "video_id": video_id, "event_type": "zone_presence_observed",
            "timestamp_seconds": 1.25, "track_id": 2,
            "label": "first observed inside", "details": {"zone_name": "Loading bay"}})
    response = client.get(f"/results/{video_id}")
    assert b"Initial occupancy \xe2\x80\x94 entry time not observed" in response.data
    assert b"observed at 00:01.25" in response.data


def test_incident_review_validates_event_and_status_and_is_local_only(client, app):
    video_id = ObjectId()
    app.extensions["mongo_db"].videos.insert_one({"_id": video_id, "original_filename": "review.avi"})
    event_id = app.extensions["mongo_db"].detection_events.insert_one({
        "video_id": video_id, "event_type": "potential_fall", "label": "Potential fall candidate",
        "timestamp_seconds": 1.25, "anomaly_score": .61, "incident_status": "new",
        "review_notes": "", "created_at": datetime.now(timezone.utc),
    }).inserted_id
    token = csrf(client)
    response = client.post(f"/incidents/{event_id}/review", data={
        "csrf_token": token, "status": "reviewing", "review_notes": "Check frame 12",
        "next": "//attacker.example"})
    assert response.status_code == 302 and response.headers["Location"].endswith("/incidents")
    saved = app.extensions["mongo_db"].detection_events.find_one({"_id": event_id})
    assert saved["incident_status"] == "reviewing" and saved["review_notes"] == "Check frame 12"
    assert b"Potential fall candidate" in client.get("/incidents?status=reviewing").data
    assert client.post(f"/incidents/{event_id}/review", data={
        "csrf_token": token, "status": "confirmed", "review_notes": "invalid"}).status_code == 400
    assert client.post(f"/incidents/{event_id}/review", data={
        "csrf_token": token, "status": "resolved"}, environ_overrides={"REMOTE_ADDR": "10.0.0.8"}).status_code == 403
    object_id = app.extensions["mongo_db"].detection_events.insert_one({
        "video_id": video_id, "event_type": "object_detection", "label": "person"}).inserted_id
    with app.app_context():
        assert not update_incident(str(object_id), "resolved", "must not become incident")


def test_restricted_zone_entry_persists_in_local_fallback_and_is_reviewable(app, client):
    from app.services.local_fallback import save_pending_upload
    from app.services.repository import save_events, list_incidents, get_events, list_videos
    with app.app_context():
        video_id, job_id = save_pending_upload(video_doc("local-zone.mp4", "queued"), "offline fixture")
        entry = {"event_type": "RESTRICTED_ZONE_ENTRY", "label": "Entered north gate",
                 "timestamp_seconds": 2.5, "confidence": .83, "anomaly_score": None,
                 "track_id": 12, "detection_method": "YOLO pose + ByteTrack + polygon transition",
                 "details": {"zone_id": "north", "zone_name": "North gate",
                             "previously_confirmed_outside": True}}
        assert save_events(video_id, job_id, [entry]) == 1
        saved = get_events(video_id)
        assert len(saved) == 1 and saved[0]["event_type"] == "RESTRICTED_ZONE_ENTRY"
        assert saved[0]["incident_status"] == "new" and saved[0]["track_id"] == 12
        incidents = list_incidents()
        stored = next(row for row in incidents if str(row.get("_id")) == str(saved[0]["_id"]))
        assert stored["details"]["zone_name"] == "North gate"
        assert update_incident(str(stored["_id"]), "reviewing", "Check the doorway clip") is True
        reviewed = next(row for row in list_incidents("reviewing") if str(row.get("_id")) == str(stored["_id"]))
        assert reviewed["review_notes"] == "Check the doorway clip"
        videos, _ = list_videos(search="local-zone")
        assert videos[0]["event_count"] == 0
        assert videos[0]["restricted_zone_entry_count"] == 1
    assert b"RESTRICTED_ZONE_ENTRY" in client.get(f"/results/{video_id}").data
    history_page = client.get("/history").data
    assert b"local-zone.mp4" in history_page and b"Zone Entries:" in history_page and b"1 entr" in history_page
    assert b"local-zone.mp4" in client.get("/incidents?status=reviewing").data
    assert b"RESTRICTED_ZONE_ENTRY" in client.get("/dashboard").data


def test_restricted_zone_entry_persists_in_mongo_repository_and_dashboard(app):
    from app.services.repository import save_events, list_incidents, list_videos
    with app.app_context():
        video_id = create_video(video_doc("mongo-zone.mp4", "complete"))
        event = {"event_type": "RESTRICTED_ZONE_ENTRY", "label": "Entered loading bay",
                 "timestamp_seconds": 4.0, "confidence": .76, "track_id": 9,
                 "details": {"zone_id": "bay", "zone_name": "Loading bay"},
                 "detection_method": "YOLO pose + ByteTrack + confirmed polygon transition"}
        assert save_events(str(video_id), str(ObjectId()), [event]) == 1
        stored = app.extensions["mongo_db"].detection_events.find_one({"video_id": video_id})
        assert stored["event_type"] == "RESTRICTED_ZONE_ENTRY"
        assert stored["incident_status"] == "new"
        assert any(item["_id"] == stored["_id"] for item in list_incidents())
        videos, _ = list_videos(search="mongo-zone")
        assert videos[0]["event_count"] == 0
        assert videos[0]["restricted_zone_entry_count"] == 1
        stats = dashboard_stats()
        assert stats["restricted_zone_entries"] == 1
        assert stats["events"] == 0  # A zone entry is a review event, not an anomaly prediction.


def test_result_timeline_handles_missing_or_invalid_timestamps(client, app):
    with app.app_context():
        video_id = create_video(video_doc("invalid-time.avi", "complete"))
        app.extensions["mongo_db"].detection_events.insert_one({
            "video_id": video_id, "event_type": "potential_fall", "label": "Potential fall candidate",
            "timestamp_seconds": "not-a-number", "anomaly_score": .6,
            "details": {}, "created_at": datetime.now(timezone.utc), "incident_status": "new"})
    response = client.get(f"/results/{video_id}")
    assert response.status_code == 200
    assert b"Time unavailable" in response.data
    assert b"disabled aria-disabled=\"true\"" in response.data


def test_pdf_report_generation_with_and_without_events(client, app):
    with app.app_context():
        video_id = create_video({**video_doc("report (draft).avi", "complete"), "analysis_summary": "Validation only."})
        app.extensions["mongo_db"].detection_events.insert_one({
            "video_id": video_id, "event_type": "potential_fall", "label": "Candidate (review)",
            "timestamp_seconds": 2.5, "anomaly_score": .72, "detection_method": "pose heuristic",
            "details": {"previously_upright": True, "vertical_drop_fraction": .3,
                        "torso_horizontal_ratio": 1.7, "horizontal_persistence_seconds": .8},
            "incident_status": "reviewing", "review_notes": "Needs (manual) review",
            "created_at": datetime.now(timezone.utc)})
    empty_pdf = client.get(f"/results/{video_id}/report.pdf")
    assert empty_pdf.status_code == 200 and empty_pdf.mimetype == "application/pdf"
    assert empty_pdf.data.startswith(b"%PDF-1.4") and empty_pdf.data.endswith(b"%%EOF\n")
    xref_offset = int(empty_pdf.data.rsplit(b"startxref\n", 1)[1].splitlines()[0])
    assert empty_pdf.data[xref_offset:xref_offset + 4] == b"xref"
    assert b"Candidate \\(review\\)" in empty_pdf.data
    event = {"event_type": "potential_fall", "label": "Candidate (review)",
             "timestamp_seconds": 2.5, "anomaly_score": .72, "detection_method": "pose heuristic",
             "details": {"previously_upright": True, "vertical_drop_fraction": .3,
                         "torso_horizontal_ratio": 1.7, "horizontal_persistence_seconds": .8},
             "incident_status": "reviewing", "review_notes": "Needs (manual) review"}
    pdf = build_pdf({"_id": video_id, "original_filename": "clip (one).avi",
                     "processing_status": "complete"}, {"status": "completed"}, [event])
    assert b"Candidate \\(review\\)" in pdf
    assert b"score: 0.72" in pdf.lower()
    assert b"Needs \\(manual\\) review" in pdf
    assert b"connection string" not in pdf.lower()


def test_pdf_report_empty_record_has_clear_zero_event_message(client, app):
    with app.app_context():
        video_id = create_video(video_doc("no-events.avi", "complete"))
    response = client.get(f"/results/{video_id}/report.pdf")
    assert response.status_code == 200
    assert b"No event records were saved" in response.data


def test_reports_tabs_filters_and_real_overview_metrics(client, app):
    with app.app_context():
        first = create_video({**video_doc("analytics-alpha.mp4", "complete"),
                              "upload_date": datetime(2026, 1, 2, 1, 0, tzinfo=timezone.utc),
                              "duration": 12.5, "processing_duration": 3.2,
                              "pose_analysis": {"status": "complete", "model_name": "pose-fixture"},
                              "analysis_summary": "Saved evidence summary."})
        second = create_video({**video_doc("analytics-beta.mp4", "failed"),
                               "upload_date": datetime(2026, 1, 3, 1, 0, tzinfo=timezone.utc)})
        create_video({**video_doc("legacy-iso-date.mp4", "queued"),
                      "upload_date": "2026-01-02T01:00:00Z"})
        db = app.extensions["mongo_db"]
        db.detection_events.insert_many([
            {"video_id": first, "event_type": "potential_fall", "timestamp_seconds": 2.0,
             "start_timestamp_seconds": 2.0, "end_timestamp_seconds": 3.0,
             "anomaly_score": .61, "detection_method": "pose heuristic", "incident_status": "new"},
            {"video_id": first, "event_type": "object_detection", "timestamp_seconds": 2.5,
             "label": "person", "confidence": .91, "detection_method": "YOLO", "track_id": 4},
            {"video_id": second, "event_type": "RESTRICTED_ZONE_ENTRY", "timestamp_seconds": 4.0,
             "zone_name": "Door", "track_id": 2, "incident_status": "reviewing"},
        ])
    page = client.get("/reports")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    for tab in ("Overview", "Video Reports", "Event Timeline", "Technical Evidence", "Export &amp; Print"):
        assert tab in html
    assert "VIDEOS IN RANGE" in html and "SAVED DETECTOR OBSERVATIONS" in html
    assert "Candidate episodes by category" in html and "Review queue" in html
    assert "Data source: MongoDB" in html
    assert "<span>VIDEOS IN RANGE</span><strong>3</strong>" in html
    assert "<span>COMPLETED RUNS</span><strong>1</strong>" in html
    assert "<span>ANOMALY REVIEW EPISODES</span><strong>2</strong>" in html
    assert "<span>SAVED DETECTOR OBSERVATIONS</span><strong>3</strong>" in html
    assert "<span>FAILED RUNS</span><strong>1</strong>" in html
    assert "<span>OTHER RUN STATES</span><strong>1</strong>" in html
    video_date_page = client.get("/reports?tab=videos")
    assert b"02 Jan 2026, 06:30 AM IST" in video_date_page.data
    ist_day = client.get("/reports?tab=videos&start=2026-01-02&end=2026-01-02")
    assert ist_day.status_code == 200
    ist_table = ist_day.get_data(as_text=True).split('<table class="report-table">', 1)[1].split("</table>", 1)[0]
    assert "analytics-alpha.mp4" in ist_table and "legacy-iso-date.mp4" in ist_table
    assert "analytics-beta.mp4" not in ist_table
    videos_tab = client.get("/reports?tab=videos&video_q=alpha")
    video_table = videos_tab.get_data(as_text=True).split('<table class="report-table">', 1)[1].split("</table>", 1)[0]
    assert "analytics-alpha.mp4" in video_table and "analytics-beta.mp4" not in video_table
    clear_link = client.get("/reports?tab=videos&video_q=alpha&start=2026-01-01").get_data(as_text=True)
    assert 'href="/reports?tab=videos&amp;video_q=alpha"' in clear_link
    assert b"View Report" in videos_tab.data and b"video-report-" in videos_tab.data
    assert b"Saved evidence summary." in videos_tab.data
    timeline = client.get("/reports?tab=timeline&category=potential_fall")
    assert b"Potential Fall" in timeline.data and b"00:02.00" in timeline.data
    assert b"Save review" in timeline.data
    evidence = client.get("/reports?tab=evidence&category=object_detection")
    assert b"Object Detection" in evidence.data and b"0.910" in evidence.data
    assert client.get("/reports?tab=invalid").status_code == 400
    assert client.get("/reports?start=2026-02-02&end=2026-01-01").status_code == 400


def test_reports_group_episodes_per_video_and_review_updates_persist(client, app):
    with app.app_context():
        one = create_video(video_doc("same-name.mp4", "complete"))
        two = create_video(video_doc("same-name.mp4", "complete"))
        db = app.extensions["mongo_db"]
        a = db.detection_events.insert_one({"video_id": one, "event_type": "potential_fall",
            "timestamp_seconds": 4, "start_timestamp_seconds": 4, "end_timestamp_seconds": 6,
            "track_id": 7, "anomaly_score": .55, "incident_status": "new"}).inserted_id
        db.detection_events.insert_one({"video_id": two, "event_type": "potential_fall",
            "timestamp_seconds": 4.5, "start_timestamp_seconds": 4.5, "end_timestamp_seconds": 6.5,
            "track_id": 7, "anomaly_score": .6, "incident_status": "new"})
    response = client.get("/reports?tab=timeline")
    assert response.status_code == 200
    assert response.get_data(as_text=True).count("same-name.mp4") >= 2
    # Identical event times and track IDs across uploads must remain two episodes.
    assert b"ANOMALY REVIEW EPISODES" in client.get("/reports").data
    token = csrf(client)
    review = client.post(f"/incidents/{a}/review", data={"csrf_token": token,
        "status": "reviewing", "review_notes": "Inspect the stored pose transition.",
        "next": "/reports?tab=timeline"})
    assert review.status_code == 302
    timeline = client.get("/reports?tab=timeline&review=reviewing")
    assert b"Inspect the stored pose transition." in timeline.data
    assert b"Reviewing" in timeline.data


def test_reports_csv_protects_formula_text_and_supports_detail_option(client, app):
    with app.app_context():
        video_id = create_video(video_doc("=HYPERLINK(\"x\").mp4", "complete"))
        app.extensions["mongo_db"].detection_events.insert_one({
            "video_id": video_id, "event_type": "potential_fall", "label": "=IMPORTXML(\"x\")",
            "timestamp_seconds": 1, "review_notes": "+cmd", "details": {"evidence": "@formula"}})
    summary = client.get("/reports.csv")
    detailed = client.get("/reports.csv?include_details=1")
    assert summary.status_code == detailed.status_code == 200
    assert b"'=HYPERLINK" in summary.data and b"'=IMPORTXML" in summary.data
    assert b"'+cmd" in summary.data
    assert b"@formula" not in summary.data
    assert b"@formula" in detailed.data


def test_reports_use_one_authoritative_store_and_fallback_when_atlas_unavailable(client, app):
    from app.services.local_fallback import save_pending_upload
    with app.app_context():
        mongo_id = create_video(video_doc("atlas-only.mp4", "complete"))
        save_pending_upload({**video_doc("unsynced-local.mp4", "queued"),
                             "upload_request_id": "reports-local-probe"}, "isolated test")
        videos, _events, source = report_data_with_source()
        assert source == "mongodb"
        assert any(str(row["_id"]) == str(mongo_id) for row in videos)
        assert not any(row.get("original_filename") == "unsynced-local.mp4" for row in videos)
        app.extensions["mongo_available"] = False
        app.extensions["mongo_retry_at"] = float("inf")
        videos, _events, source = report_data_with_source()
        assert source == "local_fallback"
        assert any(row.get("original_filename") == "unsynced-local.mp4" for row in videos)
    page = client.get("/reports")
    assert page.status_code == 200 and b"Data source: Local fallback" in page.data


def test_reports_server_pagination_and_expandable_details(client, app):
    with app.app_context():
        for index in range(18):
            create_video({**video_doc(f"paged-{index:02d}.mp4", "complete"),
                          "upload_date": datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index)})
    first = client.get("/reports?tab=videos")
    second = client.get("/reports?tab=videos&video_page=2")
    assert first.status_code == second.status_code == 200
    first_table = first.get_data(as_text=True).split('<table class="report-table">', 1)[1].split("</table>", 1)[0]
    second_table = second.get_data(as_text=True).split('<table class="report-table">', 1)[1].split("</table>", 1)[0]
    assert "paged-17.mp4" in first_table and "paged-02.mp4" not in first_table
    assert "paged-02.mp4" in second_table and "paged-17.mp4" not in second_table
    assert b"aria-expanded=\"false\"" in first.data and b"data-expand-report" in first.data
    assert client.get("/static/js/reports.js").status_code == 200
    assert client.get("/static/css/reports.css").status_code == 200


def test_reports_review_save_failure_is_visible_without_false_success(client, app, monkeypatch):
    import app.routes.incidents as incident_routes
    event_id = str(ObjectId())

    def unavailable(*_args, **_kwargs):
        raise DatabaseUnavailable("isolated test outage")

    monkeypatch.setattr(incident_routes, "update_incident", unavailable)
    response = client.post(f"/incidents/{event_id}/review", data={
        "csrf_token": csrf(client), "status": "reviewing", "review_notes": "No write confirmed.",
        "next": "/reports?tab=timeline"})
    assert response.status_code == 302
    page = client.get(response.location)
    assert b"Review was not saved" in page.data
    assert b"Incident review saved" not in page.data


def test_reports_empty_store_renders_every_tab_without_missing_review_context(client, app):
    response = client.get("/reports")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'id="panel-overview"' in html and 'id="panel-videos"' in html
    assert 'id="panel-timeline"' in html and 'id="panel-evidence"' in html
    assert 'id="panel-export"' in html
    assert "New / not reviewed" in html and "No candidate episodes in this range" in html


def test_shared_navbar_assets_and_links_render_correctly_on_all_pages(client, app):
    paths = ["/dashboard", "/upload", "/history", "/incidents", "/reports", "/about"]
    for path in paths:
        response = client.get(path)
        assert response.status_code == 200, path
        html = response.get_data(as_text=True)
        assert 'href="/static/css/style.css"' in html
        assert 'href="/static/css/branding.css"' in html
        assert "branding.css<link" not in html
        for route in ("/dashboard", "/upload", "/history", "/incidents", "/reports", "/about"):
            assert f'href="{route}"' in html
        assert 'href="/upload">New analysis' in html
        assert html.index('class="brand"') < html.index('class="nav-links"') < html.index('class="nav-cta"')
    assert 'href="/static/css/reports.css"' in client.get("/reports").get_data(as_text=True)
    assert 'href="/static/css/reports.css"' not in client.get("/dashboard").get_data(as_text=True)


def test_analysis_settings_parent_and_fight_child_update_without_mongo_path_conflict(app):
    with app.app_context():
        video_id = create_video({**video_doc("settings-merge.mp4", "processing"),
                                 "analysis_settings": {"sample_every_seconds": .5,
                                                       "pose_tracker": "bytetrack.yaml",
                                                       "nested": {"keep": True}}})
        assert update_video(str(video_id), {
            "analysis_settings": {"zone_track_sample_seconds": .25,
                                  "fight_score_threshold": .42},
            "analysis_settings.fight_score_threshold": .47,
        }) is True
        saved = get_video(str(video_id))
        assert saved["analysis_settings"] == {
            "sample_every_seconds": .5,
            "pose_tracker": "bytetrack.yaml",
            "nested": {"keep": True},
            "zone_track_sample_seconds": .25,
            "fight_score_threshold": .47,
        }


def test_nested_analysis_setting_update_preserves_sibling_values(app):
    with app.app_context():
        video_id = create_video({**video_doc("settings-sibling.mp4", "processing"),
                                 "analysis_settings": {"sampling": {"pose": .5, "fight": 1.0},
                                                       "model": "x3d"}})
        assert update_video(str(video_id), {"analysis_settings.sampling.pose": .25}) is True
        saved = get_video(str(video_id))
        assert saved["analysis_settings"] == {
            "sampling": {"pose": .25, "fight": 1.0}, "model": "x3d"}


def test_write_error_code_40_is_not_misreported_as_database_connectivity(app, monkeypatch):
    from pymongo.errors import WriteError
    import app.services.repository as repository

    class BrokenCollection:
        def update_one(self, *_args, **_kwargs):
            raise WriteError("conflicting update paths", 40)

    class BrokenDatabase:
        videos = BrokenCollection()

    with app.app_context():
        app.extensions["mongo_available"] = True
        monkeypatch.setattr(repository, "get_database", lambda: BrokenDatabase())
        with pytest.raises(WriteError) as error:
            update_video(str(ObjectId()), {"analysis_settings": {"fight_score_threshold": .5},
                                            "analysis_settings.fight_score_threshold": .6})
        assert error.value.code == 40
        assert app.extensions["mongo_available"] is True
        message = _friendly_connection_error(error.value)
        assert "application update conflict" in message
        assert "network" in message


def test_status_api_distinguishes_database_outage_from_missing_video(app, client, monkeypatch):
    import app.services.repository as repository

    with app.app_context():
        video_id = create_video(video_doc("status-outage.mp4", "complete"))
    monkeypatch.setattr(repository, "get_database",
                        lambda: (_ for _ in ()).throw(DatabaseUnavailable("temporary database outage")))
    unavailable = client.get(f"/api/videos/{video_id}/status")
    assert unavailable.status_code == 503
    assert unavailable.json["status"] == "unavailable"
    assert "temporary database outage" in unavailable.json["error"]
    invalid = client.get("/api/videos/not-an-object-id/status")
    assert invalid.status_code == 404
