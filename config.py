import os
import secrets
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
from dotenv import load_dotenv

# Resolve .env next to this file so VS Code's current working directory does
# not affect which configuration the website reads.
load_dotenv(BASE_DIR / ".env")


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_hex(32)
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_MB", "250")) * 1024 * 1024
    MONGO_URI = os.getenv("MONGO_URI", "").strip()
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "human_anomaly_db")
    MONGO_TIMEOUT_MS = int(os.getenv("MONGO_TIMEOUT_MS", "1500"))
    MONGO_RETRY_INTERVAL_SECONDS = float(os.getenv("MONGO_RETRY_INTERVAL_SECONDS", "5"))
    MONGO_PING_ON_STARTUP = True
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads"))
    PROCESSED_DIR = os.getenv("PROCESSED_DIR", str(BASE_DIR / "processed"))
    LOCAL_FALLBACK_PATH = os.getenv("LOCAL_FALLBACK_PATH", str(BASE_DIR / "instance" / "local_history.json"))
    SAMPLE_EVERY_SECONDS = float(os.getenv("SAMPLE_EVERY_SECONDS", "1.0"))
    # Active polygon zones need denser observations than the default general
    # analysis cadence to establish a tracked boundary transition.
    ZONE_TRACK_SAMPLE_SECONDS = float(os.getenv("ZONE_TRACK_SAMPLE_SECONDS", "0.5"))
    ZONE_ENTRY_CONFIRMATION_FRAMES = int(os.getenv("ZONE_ENTRY_CONFIRMATION_FRAMES", "2"))
    ZONE_EXIT_CONFIRMATION_FRAMES = int(os.getenv("ZONE_EXIT_CONFIRMATION_FRAMES", "2"))
    ZONE_TRACK_MAX_GAP_SECONDS = float(os.getenv("ZONE_TRACK_MAX_GAP_SECONDS", "1.5"))
    ZONE_BOUNDARY_MARGIN_PX = float(os.getenv("ZONE_BOUNDARY_MARGIN_PX", "8"))
    MAX_DURATION_SECONDS = int(os.getenv("MAX_DURATION_SECONDS", "900"))
    # Annotation preview can be downscaled without changing detector input,
    # timestamps, sampling cadence, or the untouched uploaded source video.
    ANNOTATED_OUTPUT_MAX_WIDTH = int(os.getenv("ANNOTATED_OUTPUT_MAX_WIDTH", "640"))
    POSE_MODEL_PATH = os.getenv("POSE_MODEL_PATH", "")
    POSE_TRACKER = os.getenv("POSE_TRACKER", "bytetrack.yaml").strip() or "bytetrack.yaml"
    FALL_TRANSITION_WINDOW_SECONDS = float(os.getenv("FALL_TRANSITION_WINDOW_SECONDS", "1.5"))
    FALL_MIN_VERTICAL_DROP = float(os.getenv("FALL_MIN_VERTICAL_DROP", "0.12"))
    FALL_MIN_HORIZONTAL_PERSISTENCE = float(os.getenv("FALL_MIN_HORIZONTAL_PERSISTENCE", "0.6"))
    FALL_CANDIDATE_SCORE_THRESHOLD = float(os.getenv("FALL_CANDIDATE_SCORE_THRESHOLD", "0.55"))
    # Object detection is opt-in and only loads an existing local weight file.
    OBJECT_MODEL_PATH = os.getenv("OBJECT_MODEL_PATH", "").strip()
    OBJECT_MODEL_CONFIDENCE = float(os.getenv("OBJECT_MODEL_CONFIDENCE", "0.25"))
    OBJECT_TRACKER = os.getenv("OBJECT_TRACKER", "bytetrack.yaml").strip() or "bytetrack.yaml"
    # Optional temporal violence-activity model. Empty keeps fighting analysis
    # explicitly unavailable while all existing detectors continue to run.
    FIGHT_DETECTION_ENABLED = os.getenv("FIGHT_DETECTION_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
    FIGHT_MODEL_PATH = os.getenv("FIGHT_MODEL_PATH", "").strip()
    if FIGHT_MODEL_PATH and not Path(FIGHT_MODEL_PATH).is_absolute():
        FIGHT_MODEL_PATH = str((BASE_DIR / FIGHT_MODEL_PATH).resolve())
    FIGHT_SCORE_THRESHOLD = float(os.getenv("FIGHT_SCORE_THRESHOLD", "0.4"))
    FIGHT_WINDOW_SECONDS = float(os.getenv("FIGHT_WINDOW_SECONDS", "4.0"))
    FIGHT_WINDOW_STRIDE = float(os.getenv("FIGHT_WINDOW_STRIDE", "1.0"))
    FIGHT_CONFIRMATION_WINDOWS = int(os.getenv("FIGHT_CONFIRMATION_WINDOWS", "2"))
    FIGHT_DEVICE = os.getenv("FIGHT_DEVICE", "auto").strip().lower() or "auto"
    # SMTP delivery is opt-in. Values remain empty in the checked-in example.
    EMAIL_ALERTS_ENABLED = os.getenv("EMAIL_ALERTS_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
    SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").strip().lower() in {"1", "true", "yes", "on"}
    ALERT_SENDER_EMAIL = os.getenv("ALERT_SENDER_EMAIL", "").strip()
    ALERT_RECIPIENT_EMAILS = [value.strip() for value in os.getenv("ALERT_RECIPIENT_EMAILS", "").split(",") if value.strip()]
    ALERT_SMTP_TIMEOUT_SECONDS = float(os.getenv("ALERT_SMTP_TIMEOUT_SECONDS", "10"))
    ALERT_MAX_GIF_BYTES = int(os.getenv("ALERT_MAX_GIF_BYTES", str(3 * 1024 * 1024)))
    ALERT_GIF_WIDTH = int(os.getenv("ALERT_GIF_WIDTH", "320"))
    ALERT_EPISODE_MERGE_GAP_SECONDS = float(os.getenv("ALERT_EPISODE_MERGE_GAP_SECONDS", "1.0"))
    ALERT_EVIDENCE_DIR = os.getenv("ALERT_EVIDENCE_DIR", str(BASE_DIR / "instance" / "alert_evidence"))
    if ALERT_EVIDENCE_DIR and not Path(ALERT_EVIDENCE_DIR).is_absolute():
        ALERT_EVIDENCE_DIR = str((BASE_DIR / ALERT_EVIDENCE_DIR).resolve())
    # Empty means use the frontal-face cascade bundled with opencv-python.
    FACE_CASCADE_PATH = os.getenv("FACE_CASCADE_PATH", "").strip()
    PROCESSING_WORKERS = int(os.getenv("PROCESSING_WORKERS", "1"))
    PROCESSING_QUEUE_SIZE = int(os.getenv("PROCESSING_QUEUE_SIZE", "3"))
    DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"
