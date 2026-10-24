"""Durable, opt-in SMTP alerts for already-persisted candidate incidents."""
import html
import logging
import os
import smtplib
import ssl
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import make_msgid
from threading import Lock

import cv2

from app.services.storage import safe_processed_path, safe_video_path

log = logging.getLogger(__name__)
ALERTABLE = {"RESTRICTED_ZONE_ENTRY", "potential_fall", "SUSPECTED_FIGHTING"}
_EVENT_LABELS = {
    "RESTRICTED_ZONE_ENTRY": "Restricted Zone Entry Detected",
    "potential_fall": "Potential Fall Candidate",
    "SUSPECTED_FIGHTING": "Suspected Violence Candidate",
}


class AlertConfigurationError(ValueError):
    pass


def smtp_configuration(config):
    if not config.get("EMAIL_ALERTS_ENABLED"):
        return False, "Email alerts are disabled in configuration."
    required = (config.get("SMTP_HOST"), config.get("SMTP_PORT"),
                config.get("ALERT_SENDER_EMAIL"), config.get("ALERT_RECIPIENT_EMAILS"))
    if not all(required):
        return False, "Email alert delivery is enabled, but SMTP host, port, sender, or recipients are missing."
    if bool(config.get("SMTP_USERNAME")) != bool(config.get("SMTP_PASSWORD")):
        return False, "SMTP username and password must both be configured, or both left empty for a trusted relay."
    return True, None


def _message(config, subject, html_body, text_body, attachment=None, event_id=None):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = config["ALERT_SENDER_EMAIL"]
    msg["To"] = ", ".join(config["ALERT_RECIPIENT_EMAILS"])
    msg["Message-ID"] = (f"<sentinel-{event_id}@sentinel.local>" if event_id
                         else make_msgid(idstring="test", domain="sentinel.local"))
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")
    if attachment:
        with open(attachment, "rb") as stream:
            msg.add_attachment(stream.read(), maintype="image", subtype="gif", filename="sentinel-evidence.gif")
    return msg


def _utc_label(value):
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            value = None
    if not isinstance(value, datetime):
        return "unavailable"
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _send(config, message, before_transmission=None):
    timeout = max(2.0, min(60.0, float(config.get("ALERT_SMTP_TIMEOUT_SECONDS", 10))))
    host, port = config["SMTP_HOST"], int(config["SMTP_PORT"])
    context = ssl.create_default_context()
    smtp_class = smtplib.SMTP_SSL if config.get("SMTP_USE_TLS") and port == 465 else smtplib.SMTP
    with smtp_class(host, port, timeout=timeout, context=context) if smtp_class is smtplib.SMTP_SSL else smtp_class(host, port, timeout=timeout) as server:
        server.ehlo()
        if config.get("SMTP_USE_TLS") and port != 465:
            server.starttls(context=context)
            server.ehlo()
        if config.get("SMTP_USERNAME"):
            server.login(config["SMTP_USERNAME"], config["SMTP_PASSWORD"])
        if before_transmission:
            before_transmission()
        server.send_message(message)


def send_test_email(app):
    """Send a configuration test message; never requires or fabricates an event."""
    config = app.config
    ready, reason = smtp_configuration(config)
    if not ready:
        raise AlertConfigurationError(reason)
    msg = _message(config, "[Sentinel] SMTP configuration test",
                   "<p>Sentinel successfully sent this test message using the configured SMTP relay.</p>",
                   "Sentinel SMTP configuration test.")
    _send(config, msg)
    return True


def _frame_as_image(frame_bgr, width):
    from PIL import Image
    height, source_width = frame_bgr.shape[:2]
    if width > 0 and source_width > width:
        target_height = max(1, round(height * width / source_width))
        frame_bgr = cv2.resize(frame_bgr, (width, target_height), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def generate_event_gif(video_path, event, output_path, *, max_bytes=3 * 1024 * 1024,
                       width=320, preferred_path=None):
    """Write a bounded ~3-second GIF around an event, preserving source aspect ratio."""
    from PIL import Image
    timestamp = event.get("timestamp_seconds")
    if not isinstance(timestamp, (int, float)) or timestamp < 0:
        raise ValueError("The event has no valid video timestamp, so GIF evidence cannot be created.")
    start_event, end_event = event.get("start_timestamp_seconds"), event.get("end_timestamp_seconds")
    if (isinstance(start_event, (int, float)) and isinstance(end_event, (int, float))
            and 0 <= start_event <= end_event):
        timestamp = (float(start_event) + float(end_event)) / 2
    selected = preferred_path if preferred_path and os.path.isfile(preferred_path) else video_path
    cap = cv2.VideoCapture(selected)
    try:
        if not cap.isOpened():
            if selected != video_path:
                cap.release()
                cap = cv2.VideoCapture(video_path)
                selected = video_path
            if not cap.isOpened():
                raise ValueError("The source video could not be opened to create evidence.")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        duration = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps if fps > 0 else 0
        start = max(0.0, float(timestamp) - 1.5)
        if duration > 0:
            start = min(start, max(0.0, duration - 3.0))
            seconds = min(3.0, max(0.05, duration - start))
        else:
            seconds = 3.0
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        # Retry lower frame rates/resolutions if the configured size budget is exceeded.
        for target_width, sample_fps in [(max(96, int(width)), 6), (240, 4), (160, 2)]:
            frames = []
            count = max(1, min(18, int(seconds * sample_fps)))
            for index in range(count):
                cap.set(cv2.CAP_PROP_POS_MSEC, (start + index / sample_fps) * 1000)
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames.append(_frame_as_image(frame, target_width))
            if not frames and selected != video_path:
                cap.release()
                cap = cv2.VideoCapture(video_path)
                selected = video_path
                if cap.isOpened():
                    for index in range(count):
                        cap.set(cv2.CAP_PROP_POS_MSEC, (start + index / sample_fps) * 1000)
                        ok, frame = cap.read()
                        if ok and frame is not None:
                            frames.append(_frame_as_image(frame, target_width))
            if not frames:
                raise ValueError("No video frames could be decoded for the event preview.")
            temporary = output_path + ".tmp.gif"
            frame_duration_ms = max(10, round((1000 / sample_fps) / 10) * 10)
            frames[0].save(temporary, format="GIF", save_all=True, append_images=frames[1:],
                           duration=frame_duration_ms, loop=0, optimize=True, disposal=2)
            for image in frames:
                image.close()
            if os.path.getsize(temporary) <= max(1, int(max_bytes)):
                os.replace(temporary, output_path)
                return {"path": output_path, "size_bytes": os.path.getsize(output_path),
                        "duration_seconds": frame_duration_ms * len(frames) / 1000,
                        "source": "annotated" if selected != video_path else "original"}
            os.remove(temporary)
        raise ValueError("The event preview exceeded the configured GIF attachment size limit.")
    finally:
        cap.release()


def _failure_reason(exc):
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        return "SMTP authentication was rejected. Check the configured SMTP account and app password."
    if isinstance(exc, (TimeoutError, smtplib.SMTPServerDisconnected, ConnectionError, OSError)):
        return "The SMTP server could not be reached before the configured timeout."
    if isinstance(exc, smtplib.SMTPRecipientsRefused):
        return "The SMTP server rejected all configured alert recipients."
    if isinstance(exc, smtplib.SMTPException):
        return "The SMTP server rejected the alert message. Check relay settings and recipient permissions."
    if isinstance(exc, ValueError):
        detail = str(exc).lower()
        if "source video" in detail:
            return "The saved source video is missing or outside the configured uploads folder."
        if "timestamp" in detail:
            return "The saved event has no valid timestamp for GIF evidence."
        if "size limit" in detail:
            return "The event GIF exceeded the configured attachment size limit."
        if "frames" in detail or "opened" in detail:
            return "Frames could not be decoded from the saved video for GIF evidence."
    return "Alert evidence could not be prepared from the saved video."


def build_video_notification(video_id, video, episodes):
    """Create a bounded digest snapshot based only on persisted detector events."""
    import hashlib
    from app.services.report_service import explain_event
    from app.services.timestamps import event_video_seconds

    notification_episodes = []
    all_events = []
    for episode in episodes:
        events = episode.get("events") or []
        if not events:
            continue
        all_events.extend(events)
        details = events[0].get("details") or {}
        score_values = [event.get("model_score", event.get("anomaly_score")) for event in events]
        score_values = [float(value) for value in score_values
                        if isinstance(value, (int, float))]
        notification_episodes.append({
            "group_id": episode["group_id"],
            "event_type": episode["event_type"],
            "start_timestamp_seconds": episode.get("start_timestamp_seconds"),
            "end_timestamp_seconds": episode.get("end_timestamp_seconds"),
            "intervals": [list(interval) for interval in episode.get("intervals", [])],
            "timestamp_seconds": event_video_seconds(events[0]),
            "event_count": len(events),
            "event_ids": [str(event.get("_id", "")) for event in events],
            "track_id": episode.get("track_id"),
            "zone_id": episode.get("zone_id"),
            "zone_name": details.get("zone_name", episode.get("zone_name")),
            "score": max(score_values) if score_values else None,
            "score_kind": (events[0].get("details") or {}).get("score_kind"),
            "detector": events[0].get("detection_method") or "Configured detector",
            "explanation": explain_event(events[0])[:500],
        })
    group_ids = sorted(episode["group_id"] for episode in notification_episodes)
    digest = hashlib.sha256((str(video_id) + "|" + "|".join(group_ids)).encode("utf-8")).hexdigest()
    unique_events = {str(event.get("_id")): event for event in all_events}
    events_snapshot = list(unique_events.values())[:100]
    return {
        "notification_id": digest,
        "analysis_id": str(video_id),
        "group_ids": group_ids,
        "event_ids": [str(event.get("_id", "")) for event in events_snapshot],
        "original_filename": os.path.basename(video.get("original_filename") or "uploaded-video"),
        "episodes": notification_episodes[:50],
        "events": events_snapshot,
        "category_count": len({episode["event_type"] for episode in notification_episodes}),
        "episode_count": len(notification_episodes),
        "created_at": datetime.now(timezone.utc),
    }


def _episode_times(episode):
    from app.services.timestamps import format_video_timestamp
    intervals = episode.get("intervals") or []
    if intervals:
        return ", ".join(f"{format_video_timestamp(start)}–{format_video_timestamp(end)}"
                          for start, end in intervals)
    timestamp = episode.get("timestamp_seconds")
    return format_video_timestamp(timestamp) if isinstance(timestamp, (int, float)) else "Video time unavailable"


def _notification_bodies(video, notification):
    episodes = notification.get("episodes") or []
    filename = os.path.basename(video.get("original_filename") or notification.get("original_filename") or "uploaded-video")
    categories = sorted({_EVENT_LABELS.get(item.get("event_type"), item.get("event_type", "Candidate event"))
                         for item in episodes})
    subject = "[Sentinel Alert] Consolidated anomaly review"
    lines = ["Sentinel anomaly alert", f"Video: {filename}",
             f"Analysis identifier: {notification.get('analysis_id', 'unavailable')}",
             "Categories: " + (", ".join(categories) if categories else "No category label available"),
             "", "Candidate episodes:"]
    html_rows = []
    for episode in episodes[:12]:
        category = _EVENT_LABELS.get(episode.get("event_type"), episode.get("event_type", "Candidate event"))
        time_text = _episode_times(episode)
        score = episode.get("score")
        score_text = f"Highest saved detector score: {score:.3f} (uncalibrated/review score)." if isinstance(score, (int, float)) else "No detector score was provided."
        zone = episode.get("zone_name") or episode.get("zone_id")
        zone_text = f" Zone: {zone}." if zone else ""
        explanation = str(episode.get("explanation") or "Review the saved event evidence.")
        lines.append(f"- {category}: {time_text}. {score_text}{zone_text} {explanation}")
        html_rows.append("<li><strong>" + html.escape(str(category)) + "</strong>: " +
                         html.escape(time_text) + ". " + html.escape(score_text + zone_text + " " + explanation) + "</li>")
    if len(episodes) > 12:
        lines.append(f"- {len(episodes) - 12} additional saved episodes are available in Sentinel Results.")
        html_rows.append(f"<li>{len(episodes) - 12} additional saved episodes are available in Sentinel Results.</li>")
    analysis_date = _utc_label(notification.get("created_at"))
    lines.extend(["", f"Analysis date: {analysis_date}",
        "Automated signals require human review. They may be inaccurate and are not proof of wrongdoing or an emergency.",
        "The attached GIF is one representative preview; the detailed, chronological evidence remains in Sentinel Results."])
    body = ("<h2>Sentinel anomaly alert</h2><dl>"
        f"<dt>Video</dt><dd>{html.escape(filename)}</dd>"
        f"<dt>Analysis identifier</dt><dd>{html.escape(str(notification.get('analysis_id', 'unavailable')))}</dd>"
        f"<dt>Categories</dt><dd>{html.escape(', '.join(categories) if categories else 'Unavailable')}</dd>"
        f"<dt>Analysis date</dt><dd>{html.escape(analysis_date)}</dd></dl>"
        "<h3>Candidate episodes</h3><ul>" + "".join(html_rows) + "</ul>"
        "<p><strong>Automated signals require human review.</strong> They may be inaccurate and are not proof of wrongdoing or an emergency.</p>"
        "<p>The attached GIF is one representative preview. Detailed saved evidence remains in Sentinel Results.</p>")
    return subject, body, "\n".join(lines)


def _deliver_one(app, notification_id):
    """Deliver one consolidated video digest; never send per-event messages."""
    from app.services.repository import claim_video_alert, update_video_alert, get_video as fetch_video
    from app.services.repository import object_id
    with app.app_context():
        ready, reason = smtp_configuration(app.config)
        if not ready:
            try:
                update_video_alert(notification_id, {"status": "skipped", "failure_reason": reason})
            except Exception:
                log.exception("Could not persist disabled consolidated email alert status.")
            return False
        claimed = claim_video_alert(notification_id)
        if not claimed:
            return False
        video_id = str(claimed.get("video_id", ""))
        video = fetch_video(video_id)
        if not video:
            update_video_alert(notification_id, {"status": "failed",
                "failure_reason": "The saved video record is unavailable."})
            return False
        source = safe_video_path(video, app.config["UPLOAD_DIR"])
        annotated = safe_processed_path(video, app.config["PROCESSED_DIR"])
        evidence_event = next(iter(claimed.get("events") or []), None)
        if evidence_event is None:
            update_video_alert(notification_id, {"status": "failed",
                "failure_reason": "No saved event is available for the consolidated notification."})
            return False
        gif_path = os.path.join(app.config["ALERT_EVIDENCE_DIR"], f"{notification_id}.gif")
        transmission_started = False
        last_reason = None
        attempt = int(claimed.get("attempts", 1))
        for attempt in range(attempt, 4):
            try:
                if not source:
                    raise ValueError("The saved source video is missing or unsafe.")
                gif_started = time.perf_counter()
                gif = generate_event_gif(source, evidence_event, gif_path,
                    max_bytes=app.config.get("ALERT_MAX_GIF_BYTES", 3 * 1024 * 1024),
                    width=app.config.get("ALERT_GIF_WIDTH", 320), preferred_path=annotated)
                gif_seconds = time.perf_counter() - gif_started
                subject, html_body, text_body = _notification_bodies(video, claimed)
                msg = _message(app.config, subject, html_body, text_body, gif_path,
                               notification_id.replace("-", "")[:32])
                transmission_marked = update_video_alert(notification_id, {
                    "smtp_transmission_started_at": datetime.now(timezone.utc)})
                if not transmission_marked:
                    raise RuntimeError("The durable SMTP transmission marker could not be saved; message was not sent.")
                smtp_started = time.perf_counter()
                def mark_transmission():
                    nonlocal transmission_started
                    transmission_started = True
                _send(app.config, msg, before_transmission=mark_transmission)
                smtp_seconds = time.perf_counter() - smtp_started
                # send_message returned, so SMTP accepted the message. Retry
                # only the durable status write here; never resend the email.
                sent_changes = {"status": "sent", "sent_at": datetime.now(timezone.utc),
                    "last_attempt_at": datetime.now(timezone.utc), "failure_reason": None,
                    "gif_filename": os.path.basename(gif_path), "gif_size_bytes": gif["size_bytes"],
                    "gif_duration_seconds": gif["duration_seconds"], "gif_source": gif["source"],
                    "gif_generation_seconds": gif_seconds, "smtp_delivery_seconds": smtp_seconds}
                status_saved = False
                for persist_attempt in range(1, 4):
                    try:
                        status_saved = bool(update_video_alert(notification_id, sent_changes))
                    except Exception as persist_exc:
                        log.error("SMTP accepted alert %s but final status write attempt %s failed (%s).",
                                  str(notification_id)[:8], persist_attempt, type(persist_exc).__name__)
                    if status_saved:
                        break
                    if persist_attempt < 3:
                        time.sleep(0.1 * persist_attempt)
                if not status_saved:
                    log.error("SMTP accepted alert %s, but Sent could not be confirmed in persistent storage; no resend was attempted.",
                              str(notification_id)[:8])
                    return False
                return True
            except Exception as exc:
                last_reason = ("SMTP delivery outcome is uncertain; automatic resend was suppressed."
                               if transmission_started else _failure_reason(exc))
                log.warning("Sentinel consolidated alert failed (notification=%s, attempt=%s, type=%s): %s",
                            str(notification_id)[:8], attempt, type(exc).__name__, last_reason)
                if transmission_started:
                    # Once send_message may have begun, SMTP could have
                    # accepted the message even if its final response timed out.
                    # Mark uncertain and consume the slot; never send a duplicate.
                    try:
                        update_video_alert(notification_id, {"status": "uncertain",
                            "attempts": attempt, "last_attempt_at": datetime.now(timezone.utc),
                            "failure_reason": "SMTP delivery outcome is uncertain; automatic resend was suppressed."})
                    except Exception:
                        log.exception("Could not persist uncertain SMTP delivery state.")
                    return False
                permanent = isinstance(exc, (smtplib.SMTPAuthenticationError,
                    smtplib.SMTPRecipientsRefused, ValueError))
                if permanent or attempt >= 3:
                    try:
                        update_video_alert(notification_id, {"status": "failed", "attempts": attempt,
                            "last_attempt_at": datetime.now(timezone.utc), "failure_reason": last_reason})
                    except Exception:
                        log.exception("Could not persist failed consolidated delivery state.")
                    return False
                # Retry is safe only because SMTP transmission has not begun.
                try:
                    update_video_alert(notification_id, {"status": "pending", "attempts": attempt,
                        "last_attempt_at": datetime.now(timezone.utc), "failure_reason": last_reason})
                    time.sleep((1, 2)[min(attempt - 1, 1)])
                    claimed = claim_video_alert(notification_id)
                    if not claimed:
                        return False
                except Exception:
                    log.exception("Could not safely reserve a pre-transmission retry.")
                    return False
        return False


def _configured_alerts(app):
    ready, reason = smtp_configuration(app.config)
    if not ready:
        from app.services.repository import update_video_alert
        from app.services.repository import list_pending_video_alerts
        for notification in list_pending_video_alerts(limit=1000):
            try:
                update_video_alert(notification["notification_id"], {"status": "skipped", "failure_reason": reason})
            except Exception:
                log.exception("Could not persist disabled video alert state.")
        return []
    from app.services.repository import list_pending_video_alerts
    return list_pending_video_alerts(limit=1000)


def schedule_pending_alerts(app, video_id=None):
    ready, reason = smtp_configuration(app.config)
    from app.services.repository import list_pending_video_alerts, update_video_alert
    if not ready:
        pending = list_pending_video_alerts(video_id, limit=1000)
        for notification in pending:
            try:
                update_video_alert(notification["notification_id"], {"status": "skipped", "failure_reason": reason})
            except Exception:
                log.exception("Could not update disabled alert status.")
        return
    pending = list_pending_video_alerts(video_id, limit=1000)
    futures = app.extensions.setdefault("alert_futures", {})
    lock = app.extensions.setdefault("alert_futures_lock", Lock())
    executor = app.extensions["alert_executor"]
    for notification in pending:
        notification_id = str(notification.get("notification_id", ""))
        if not notification_id:
            continue
        with lock:
            if notification_id in futures:
                continue
            future = executor.submit(_deliver_one, app, notification_id)
            futures[notification_id] = future
            future.add_done_callback(lambda _future, key=notification_id, rows=futures, mutex=lock: _remove_future(key, rows, mutex))


def _remove_future(event_id, futures, lock):
    with lock:
        futures.pop(event_id, None)


def recover_and_schedule_alerts(app):
    from app.services.repository import recover_video_alerts
    try:
        with app.app_context():
            recover_video_alerts()
            schedule_pending_alerts(app)
    except Exception as exc:
        log.error("Could not recover alert outbox (%s); pending records remain persisted.", type(exc).__name__)
