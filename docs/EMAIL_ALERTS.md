# Sentinel email alerts

Email alerts are optional and **disabled by default**. The video worker only queues alerts for already-persisted `RESTRICTED_ZONE_ENTRY`, `potential_fall`, and `SUSPECTED_FIGHTING` events. Face/object observations, zone occupancy notes, and unavailable detectors never trigger mail.

## Configure

Copy the following settings into the existing local `.env` without replacing its other entries. Use an SMTP provider's app password where applicable; do not put credentials in source files or commit `.env`.

```dotenv
EMAIL_ALERTS_ENABLED=true
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USERNAME=your_smtp_username
SMTP_PASSWORD=your_smtp_app_password
SMTP_USE_TLS=true
ALERT_SENDER_EMAIL=sentinel@example.com
ALERT_RECIPIENT_EMAILS=reviewer@example.com,backup@example.com
ALERT_SMTP_TIMEOUT_SECONDS=10
ALERT_MAX_GIF_BYTES=3145728
ALERT_GIF_WIDTH=320
ALERT_EVIDENCE_DIR=instance/alert_evidence
ALERT_EPISODE_MERGE_GAP_SECONDS=1.0
```

Restart Flask after editing `.env`. Port 587 uses STARTTLS; port 465 uses SMTP over TLS. Certificate verification remains enabled. `SMTP_USERNAME` and `SMTP_PASSWORD` must either both be supplied or both be empty for a trusted relay. Email content escapes event and filename text. Sentinel does not include public video links because this project has no user accounts and local media URLs are private.

The callable `app.services.notifier.send_test_email(app)` sends a configuration test message without needing an incident. Automated tests use a local SMTP sink or mocks and never send to real recipients.

## Delivery and evidence

Once the analysis events have been persisted and the video job has completed, Sentinel groups eligible events into a concise video-level digest. A digest can include suspected falls, suspected violence, and confirmed restricted-zone entries together. The grouping gap defaults to one second and is configurable with `ALERT_EPISODE_MERGE_GAP_SECONDS`. Event records remain individually stored for review. Grouped interval episodes retain separate observed intervals when a gap exists; restricted-zone entries remain separate point events, including exits and re-entries in the detailed timeline.

Delivery state is stored durably on the existing video record (and in the existing local fallback video row). Each analysis reserves an alert slot atomically, and the per-video slot count is capped at two across retries. Normally one message contains all incidents from the completed run; a later message can only cover newly observed groups, and the cap still applies. A stable digest identifier and atomic claim prevent two workers from sending the same queued message. The alert executor keeps SMTP work outside video processing.

Before SMTP transmission starts, transient setup failures may retry up to three times with short backoff. Once message transmission may have started, a timeout or process restart is recorded as `uncertain` and is not automatically resent: the SMTP server might already have accepted it. `sent` means SMTP accepted the message; `pending`, `sending`, `failed`, `disabled`, and `uncertain` are also visible delivery states. SMTP does not offer exactly-once delivery, so suppressing an uncertain retry favors avoiding duplicate alerts over guaranteeing delivery. No existing alert records are reset or resent by this digest path.

GIF evidence is decoded from the annotated output when available and falls back to the original upload. Up to three seconds around the event is sampled at a maximum of six frames per second and resized without changing aspect ratio. Clips at video boundaries are shorter when less source footage exists. The output remains under `instance/alert_evidence/`, is not served as a public static file, and is attached only when it fits the configured byte limit. If video evidence cannot be opened or the GIF cannot fit, delivery is marked failed with a sanitized reason instead of sending a misleading empty attachment.

Messages say that alerts are automated review signals. X3D-M candidates are described as suspected violence, not confirmed fighting; scores are uncalibrated. Recipients must be authorized to review the footage. This demonstration has no login system and should remain local or behind appropriate access controls.

## Verify without real recipients

The test suite starts a temporary local SMTP sink bound to loopback, creates a small synthetic video, verifies the GIF MIME attachment, and checks persisted delivery state. It does not contact a public SMTP service. Real provider delivery still requires valid provider settings and an authorized test recipient.
