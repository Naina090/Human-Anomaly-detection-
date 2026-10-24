import os
import secrets
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore
from flask import Flask, render_template, session, request, abort, g
from werkzeug.exceptions import HTTPException
from config import Config
from .db import init_database, DatabaseUnavailable
from .logging_utils import sanitized_traceback
from .routes import register_routes


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_object(Config)
    # This is a locally edited college project. Reload changed Jinja templates
    # and avoid showing a browser-cached pre-update dashboard.
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    if test_config:
        app.config.update(test_config)
    # A test app must never inherit a developer's live SMTP settings from .env.
    # Individual notifier tests can explicitly enable delivery to a loopback sink.
    if app.config.get("TESTING") and not (test_config and "EMAIL_ALERTS_ENABLED" in test_config):
        app.config["EMAIL_ALERTS_ENABLED"] = False
    os.makedirs(app.config["UPLOAD_DIR"], exist_ok=True)
    os.makedirs(app.config["PROCESSED_DIR"], exist_ok=True)
    init_database(app)
    if app.extensions.get("mongo_available"):
        try:
            with app.app_context():
                from app.services.repository import mark_incomplete_jobs_interrupted
                mark_incomplete_jobs_interrupted()
        except DatabaseUnavailable:
            logging.getLogger(__name__).warning("Could not recover interrupted jobs because MongoDB is unavailable.")
    workers = max(1, int(app.config.get("PROCESSING_WORKERS", 1)))
    app.extensions["processing_executor"] = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="video-analysis")
    app.extensions["processing_slots"] = BoundedSemaphore(workers + max(0, int(app.config.get("PROCESSING_QUEUE_SIZE", 3))))
    app.extensions["alert_executor"] = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sentinel-email-alert")
    app.extensions["alert_futures"] = {}
    register_routes(app)

    # The local JSON fallback survives a Flask restart, but an executor thread
    # does not. Resume queued/legacy waiting records after routes and the worker
    # pool exist; mark jobs interrupted by the previous process explicitly.
    # Flask's debug reloader imports the app once in its watcher process and
    # again in the serving process. Only the serving child may recover jobs or
    # schedule durable alerts, otherwise the watcher can mark live jobs
    # interrupted or start a competing worker.
    reloader_parent = (app.config.get("DEBUG") and
                       os.environ.get("WERKZEUG_RUN_MAIN", "").lower() != "true")
    if not app.config.get("TESTING") and not reloader_parent:
        try:
            from app.services.local_fallback import prepare_local_jobs_on_startup, fail_pending_video
            from app.services.storage import safe_video_path
            from app.services.repository import get_video
            from app.routes.videos import _run_analysis
            with app.app_context():
                local_jobs = prepare_local_jobs_on_startup()
                for item in local_jobs:
                    video_id, job_id = item["video_id"], item["job_id"]
                    video = get_video(video_id)
                    path = safe_video_path(video, app.config["UPLOAD_DIR"]) if video else None
                    if not path:
                        fail_pending_video(video_id, "The local upload file is missing or outside the upload folder; this job cannot run.")
                        continue
                    if not app.extensions["processing_slots"].acquire(blocking=False):
                        fail_pending_video(video_id, "The local processing queue is full. Retry this video after another job finishes.")
                        continue
                    try:
                        app.extensions["processing_executor"].submit(_run_analysis, app, video_id, job_id, path)
                    except RuntimeError:
                        app.extensions["processing_slots"].release()
                        fail_pending_video(video_id, "The local worker could not start. Restart Flask and retry this video.")
        except Exception as exc:
            logging.getLogger(__name__).exception(
                "Local fallback worker recovery failed (%s). Existing JSON records were preserved.",
                type(exc).__name__,
            )
        # Alert delivery has its own single-thread worker and durable status on
        # the saved event. A restart requeues interrupted SMTP attempts.
        from app.services.notifier import recover_and_schedule_alerts
        recover_and_schedule_alerts(app)

    @app.before_request
    def assign_request_id():
        # A short per-request token connects a browser error page to the full
        # server-side traceback without logging the URL query or request body.
        g.request_id = secrets.token_hex(8)

    @app.before_request
    def check_csrf():
        if "csrf_token" not in session:
            session["csrf_token"] = secrets.token_urlsafe(24)
        if request.method == "POST" and request.form.get("csrf_token") != session["csrf_token"]:
            abort(400)

    @app.after_request
    def avoid_stale_local_pages(response):
        if response.mimetype == "text/html" or request.endpoint == "static":
            response.headers["Cache-Control"] = "no-store"
        request_id = getattr(g, "request_id", None)
        if request_id:
            response.headers["X-Request-ID"] = request_id
        return response

    @app.context_processor
    def csrf_context():
        model_path = app.config.get("POSE_MODEL_PATH", "")
        from app.services.notifier import smtp_configuration
        alert_ready, alert_reason = smtp_configuration(app.config)
        return {"csrf_token": session["csrf_token"],
                "detector_configured": bool(model_path and os.path.isfile(model_path)),
                "email_alerts_ready": alert_ready,
                "email_alerts_message": alert_reason}

    @app.errorhandler(404)
    def not_found(_error):
        return render_template("404.html"), 404

    @app.errorhandler(413)
    def too_large(_error):
        return render_template("error.html", message="The video exceeds the configured upload limit."), 413

    @app.errorhandler(DatabaseUnavailable)
    def database_unavailable(error):
        return render_template(
            "error.html",
            error_label="MONGODB CONNECTION",
            error_title="MongoDB is temporarily unavailable.",
            message=str(error),
        ), 503

    @app.errorhandler(Exception)
    def unexpected_error(error):
        # Werkzeug HTTP errors (404, 405, etc.) should keep their normal status.
        if isinstance(error, HTTPException):
            return error
        # Log the full server-side traceback for diagnosis, but never include the
        # request body, uploaded content, query string, or MongoDB credentials.
        formatted = sanitized_traceback(error, app.config)
        app.logger.error(
            "Unhandled request exception request_id=%s (%s) at endpoint %s; traceback follows:\n%s",
            getattr(g, "request_id", "unknown"), type(error).__name__, request.endpoint or "unknown", formatted,
        )
        return render_template(
            "error.html",
            message="The server could not complete this request. Share the reference below with your project support person.",
            request_id=getattr(g, "request_id", "unknown"),
        ), 500

    return app
