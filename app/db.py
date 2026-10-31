"""Application-scoped MongoDB connection and index setup."""
import logging
import re
import time
from urllib.parse import unquote
from flask import current_app
from pymongo import MongoClient
from pymongo.errors import ConfigurationError, OperationFailure, PyMongoError, ServerSelectionTimeoutError
from app.logging_utils import sanitized_traceback

log = logging.getLogger(__name__)
_URI_PLACEHOLDERS = (
    "paste_your", "connection_string_here", "your_cluster", "yourcluster",
    "your_username", "your_password", "<db_user>", "<db_username>",
    "<db_password>", "<username>", "<password>", "xxxxx",
)


class DatabaseUnavailable(RuntimeError):
    """Raised when a database operation cannot be completed."""


def _friendly_connection_error(error):
    """Return safe setup guidance without including a URI or credential."""
    if isinstance(error, ConfigurationError):
        detail = str(error).lower()
        if "dns" in detail or "resolution" in detail:
            resolver_match = re.search(r"Server Do53:([^@\s]+)@53", str(error))
            resolver_hint = (f" Python's DNS query to resolver {resolver_match.group(1)} timed out."
                             if resolver_match else " Python's DNS query timed out.")
            return ("PyMongo could not resolve the MongoDB Atlas DNS address. If PowerShell resolves SRV records but Python times out, "
                    + resolver_hint + " Ask your network administrator to allow Python DNS SRV/TXT lookups (UDP/TCP port 53) "
                    "or configure the adapter to use an approved DNS resolver. Keep the Atlas URI and TLS settings unchanged.")
        return "MongoDB connection settings could not be parsed. Check the URI copied from Atlas and URL-encode special password characters."
    if isinstance(error, OperationFailure):
        code = getattr(error, "code", None)
        detail = str(error).lower()
        if code == 40:
            return ("MongoDB rejected an update because the same document path was included more than once. "
                    "This is an application update conflict, not a network or permission failure.")
        if code == 18 or "authentication failed" in detail or "bad auth" in detail:
            return "MongoDB authentication failed (code 18). Check the Atlas Database Access username/password and URL-encode password special characters."
        if code == 13 or "not authorized" in detail or "unauthorized" in detail:
            return "MongoDB authenticated the user but denied permission (code 13). Grant the Atlas database user readWrite on the configured database; startup also needs permission to create indexes."
        return "MongoDB returned an operation failure (code {}). Check the Atlas database user's roles and connection settings.".format(code or "unknown")
    if isinstance(error, ServerSelectionTimeoutError):
        detail = str(error).lower()
        if "replicasetnoprimary" in detail or "no primary" in detail:
            return ("MongoDB server selection timed out because no Atlas replica-set primary responded. "
                    "Check that the Atlas cluster is running, its Network Access list allows this computer, and outbound MongoDB traffic is permitted.")
        return "MongoDB is unavailable. Check Atlas Network Access, internet connectivity, and that the cluster is running."
    return "MongoDB is unavailable. Check the Atlas URI, network access, and database credentials."


def init_database(app):
    app.extensions["mongo_error"] = None
    app.extensions["mongo_ping_ok"] = False
    app.extensions["mongo_retry_at"] = 0.0
    try:
        client = app.config.get("MONGO_CLIENT")
        uri = (app.config.get("MONGO_URI") or "").strip()
        normalized_uri = unquote(uri).lower()
        if client is None and (not uri or any(marker in normalized_uri for marker in _URI_PLACEHOLDERS)):
            app.extensions["mongo_client"] = None
            app.extensions["mongo_db"] = None
            app.extensions["mongo_available"] = False
            app.extensions["mongo_retry_at"] = time.monotonic() + max(
                0.0, float(app.config.get("MONGO_RETRY_INTERVAL_SECONDS", 5)))
            app.extensions["mongo_error"] = (
                "MongoDB is not configured. Open the .env file beside app.py and replace the value after "
                "MONGO_URI= with the connection string copied from MongoDB Atlas. Keep it private."
            )
            log.warning("MongoDB is not configured. Replace the MONGO_URI placeholder in the project .env file with the Atlas connection string.")
            return
        if client is None:
            client = MongoClient(
                app.config["MONGO_URI"],
                serverSelectionTimeoutMS=app.config["MONGO_TIMEOUT_MS"],
                connectTimeoutMS=app.config["MONGO_TIMEOUT_MS"],
                appname="human-anomaly-web",
            )
        db = client[app.config["MONGO_DB_NAME"]]
    except (PyMongoError, ValueError) as exc:
        app.extensions["mongo_client"] = None
        app.extensions["mongo_db"] = None
        app.extensions["mongo_available"] = False
        app.extensions["mongo_error"] = _friendly_connection_error(exc)
        app.extensions["mongo_retry_at"] = time.monotonic() + max(
            0.0, float(app.config.get("MONGO_RETRY_INTERVAL_SECONDS", 5)))
        log.warning("MongoDB configuration is unavailable (%s, code=%s): %s",
                    type(exc).__name__, getattr(exc, "code", None), app.extensions["mongo_error"])
        log.error("MongoDB client initialization traceback (credentials redacted):\n%s",
                  sanitized_traceback(exc, app.config))
        return
    app.extensions["mongo_client"] = client
    app.extensions["mongo_db"] = db
    app.extensions["mongo_available"] = False
    stage = "ping"
    try:
        if app.config.get("MONGO_PING_ON_STARTUP", True):
            client.admin.command("ping")
        app.extensions["mongo_ping_ok"] = True
        stage = "index creation"
        db.videos.create_index("upload_date", name="videos_upload_date")
        db.videos.create_index("processing_status", name="videos_status")
        db.videos.create_index([("original_filename", 1), ("upload_date", -1)], name="videos_filename_date")
        db.videos.create_index("upload_request_id", unique=True, sparse=True,
                               name="videos_upload_request_unique")
        db.videos.create_index("email_alert_notifications.status", name="videos_email_alert_status")
        db.detection_events.create_index("video_id", name="events_video_id")
        db.detection_events.create_index("created_at", name="events_created_at")
        db.detection_events.create_index("event_type", name="events_event_type")
        db.detection_events.create_index([("incident_status", 1), ("created_at", -1)],
                                         name="events_incident_status_date")
        db.detection_events.create_index("alert_delivery.status", name="events_alert_delivery_status")
        db.detection_events.create_index([("video_id", 1), ("timestamp_seconds", 1), ("event_type", 1)],
                                         name="events_video_timeline")
        db.processing_jobs.create_index("video_id", name="jobs_video_id")
        db.processing_jobs.create_index("status", name="jobs_status")
        db.face_detection_frames.create_index([("video_id", 1), ("timestamp_seconds", 1)],
                                              name="face_frames_video_time")
        db.pose_detection_frames.create_index([("video_id", 1), ("timestamp_seconds", 1)],
                                              name="pose_frames_video_time")
        db.motion_analytics_frames.create_index(
            [("video_id", 1), ("analysis_run_id", 1), ("timestamp_seconds", 1), ("track_id", 1)],
            name="motion_frames_video_run_time")
        db.motion_analytics_frames.create_index(
            [("video_id", 1), ("analysis_run_id", 1), ("frame_number", 1),
             ("track_id", 1), ("track_segment_id", 1)],
            unique=True, name="motion_frames_run_frame_track")
        app.extensions["mongo_available"] = True
        app.extensions["mongo_retry_at"] = 0.0
    except PyMongoError as exc:
        # The exception text can include the configured host; never log a URI.
        app.extensions["mongo_error"] = _friendly_connection_error(exc)
        app.extensions["mongo_retry_at"] = time.monotonic() + max(
            0.0, float(app.config.get("MONGO_RETRY_INTERVAL_SECONDS", 5)))
        log.warning("MongoDB startup %s failed (%s, code=%s): %s",
                    stage, type(exc).__name__, getattr(exc, "code", None), app.extensions["mongo_error"])
        log.error("MongoDB startup %s traceback (credentials redacted):\n%s",
                  stage, sanitized_traceback(exc, app.config))


def get_database():
    app = current_app._get_current_object()
    client = app.extensions.get("mongo_client")
    if client is None:
        # A startup SRV/DNS failure leaves no MongoClient to ping. Retry client
        # construction after a short backoff so a later successful standalone
        # ping/network recovery is also observed by this already-running Flask app.
        if time.monotonic() >= app.extensions.get("mongo_retry_at", 0.0):
            init_database(app)
            client = app.extensions.get("mongo_client")
        if client is None:
            raise DatabaseUnavailable(app.extensions.get("mongo_error") or (
                "MongoDB is not configured. Open the .env file beside app.py and replace the value after "
                "MONGO_URI= with the connection string copied from MongoDB Atlas (Connect → Drivers → Python → PyMongo). "
                "Keep that connection string private."
            ))
    if not app.extensions.get("mongo_available"):
        if time.monotonic() < app.extensions.get("mongo_retry_at", 0.0):
            raise DatabaseUnavailable(app.extensions.get("mongo_error") or "MongoDB is temporarily unavailable.")
        try:
            client.admin.command("ping")
            app.extensions["mongo_available"] = True
            app.extensions["mongo_error"] = None
            app.extensions["mongo_retry_at"] = 0.0
        except PyMongoError as exc:
            app.extensions["mongo_available"] = False
            app.extensions["mongo_error"] = _friendly_connection_error(exc)
            app.extensions["mongo_retry_at"] = time.monotonic() + max(
                0.0, float(app.config.get("MONGO_RETRY_INTERVAL_SECONDS", 5)))
            log.error("MongoDB request ping failed (%s); traceback follows:\n%s",
                      type(exc).__name__, sanitized_traceback(exc, app.config))
            raise DatabaseUnavailable(app.extensions["mongo_error"]) from exc
    return app.extensions["mongo_db"]
