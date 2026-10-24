# Web-Based Human Anomaly Detection System

Sentinel Activity Lab is a browser-based Python Flask research prototype. Flask and Jinja serve the landing page, dashboard, upload, results, history, reports and methodology pages. Uploaded video files stay in local storage; MongoDB stores their metadata, processing jobs and only genuine detector event records. There is no mobile app, React frontend, or Node build step.

## What works

- MongoDB persistence through a shared PyMongo client: video records, job status, event history, dashboard queries, deletion and CSV reporting.
- Local upload with internal random filenames, extension/size constraints, browser preview, and a server-side OpenCV check that opens the file and decodes a frame before creating MongoDB records or a job. A readable video can pass validation without an AI model; metadata and queueing still require MongoDB to be reachable. Invalid/corrupt videos are rejected before entering History.
- Bounded in-process background queue with progress/status polling. This is appropriate for a single-user local demonstration; a process restart loses queued work, and it is not a multi-server worker system.
- Results-page in-app completion/failure notices and candidate-fall review callouts. They are browser-local notices, not email or emergency alerts.
- OpenCV streaming decode/metadata checks. Without model weights, a completed run is explicitly `validation_only` and creates no anomaly events.
- Optional pose-based `potential_fall` transition prototype (see below). Annotated MP4 output is generated when OpenCV's video writer supports the selected codec.
- Optional COCO object detection and person tracking through Ultralytics. It runs only with a configured local model file and optional AI dependencies; the application never downloads weights on startup. Object observations are stored separately by event type and are excluded from anomaly-event totals.
- Results include an evidence-grounded summary generated from stored detections and a per-video CSV download. The summary is deterministic structured reporting, not an LLM interpretation.
- Searchable history, status filter, safe record deletion, actual database-backed counts and date-filtered CSV reports.
- Per-video and date-filtered CSV downloads; the Reports and Results pages can be printed using the browser's **Save as PDF** option (the app does not add a server-side PDF package).
- CSRF-protected state-changing forms. There is no user account/authentication layer; keep this demo on a trusted local machine.

## Defined AI activity and limitations

The optional detector uses an Ultralytics COCO-17 pose model to estimate keypoints. It does **not** use generic person detection as a fall classifier. The rule requires (a) prior upright torso evidence, (b) a rapid body-center vertical drop within 1.5 seconds, and (c) a horizontal torso persisting for at least 0.6 seconds. Its heuristic score is `0.5 × min(1, drop_fraction / 0.3) + 0.3 × min(1, torso_horizontal_ratio / 2) + 0.2 × min(1, persistence_seconds / 1.2)`; it is an anomaly score, not model confidence. Confidence remains null. The current prototype emits at most one qualifying event per video. It is not clinically or scientifically validated, can miss events or flag other movements, and cannot establish intent, danger, injury, or identity. Fighting and general unusual behavior are not implemented. An empty event list does not mean that a video is normal.

Pose model dependency and weights are optional. Ultralytics documents COCO pose keypoints and pretrained pose weights in its [Pose task guide](https://docs.ultralytics.com/tasks/pose); its licensing page explains the default AGPL-3.0 terms and Enterprise option for qualifying proprietary/commercial use: [Ultralytics licensing](https://www.ultralytics.com/license). Review license terms before distributing a project that bundles the package or weights. The app does not train a model. Hardware speed depends on the clip, CPU/GPU and PyTorch build; no performance or accuracy claim is made.

Object detection/tracking is a separate optional pipeline. It uses Ultralytics detection models with COCO classes supported by the selected weights (person, bicycles, cars, motorcycles, buses, trains, backpacks, handbags and suitcases where present). Ultralytics documents [detection](https://docs.ultralytics.com/tasks/detect), [tracking with ByteTrack](https://docs.ultralytics.com/modes/track), and its [license terms](https://www.ultralytics.com/license). Detector confidence is the model's per-observation output, not a calibrated probability. Counts are sampled-frame observations; distinct person track IDs are temporary tracker estimates and do not identify real-world people. Review the model and package license before use or redistribution.

## Requirements and setup (Windows / VS Code)

Install Python 3.10+ and have a MongoDB Atlas cluster. Open this folder in VS Code, then run these PowerShell commands from the project folder:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

### Configure MongoDB Atlas

1. Create/sign in to MongoDB Atlas and create a cluster.
2. Under **Database Access**, create a database user with a strong password. This is distinct from your Atlas login.
3. Under **Network Access**, allow your current IP address. Avoid allowing all IPs for routine use.
4. On the cluster, choose **Connect → Drivers → Python** and copy the connection URI. Put that URI only in `.env`; URL-encode reserved password characters.
5. Open `.env` and set:

```dotenv
SECRET_KEY=use-a-long-private-random-value
MONGO_URI=mongodb+srv://DB_USER:ENCODED_PASSWORD@YOUR_CLUSTER.mongodb.net/?retryWrites=true&w=majority
MONGO_DB_NAME=human_anomaly_db
```

`.env` is ignored by Git. The first successful connection creates indexes; MongoDB shows collections after their first records are inserted. Upload a short video, then in Atlas open **Browse Collections → human_anomaly_db → videos**. Event rows appear only when the configured detector actually emits an event.

Open `.env` in VS Code and replace the sample text after `MONGO_URI=` with the private URI you copied. Keep it on one line and do not share or commit it. Replace Atlas sample username/password/cluster placeholders in the copied URI; URL-encode reserved password characters. Leave `MONGO_DB_NAME=human_anomaly_db`. Restart Flask after saving.

If `.env` already exists, keep it and edit it; do not run `Copy-Item` over it because that would replace your private settings. The application reads `.env` beside `app.py` even if VS Code starts it from another working directory. If `MONGO_URI` is missing or remains the example placeholder, the landing page stays available and database pages show setup instructions. The app does not guess a local MongoDB address.

### Optional pose-model setup

Install PyTorch using the current command recommended by the [official PyTorch install selector](https://pytorch.org/get-started/locally/) for your Windows CPU/CUDA setup, then install the optional integration:

```powershell
python -m pip install -r requirements-ai.txt
```

Obtain the documented pretrained `yolo26n-pose.pt` pose weights from the Ultralytics Pose task using a source and license you have reviewed. Set `POSE_MODEL_PATH` in `.env` to the local weights file. No model is automatically downloaded. If pose weights or dependencies are unavailable, that stage is marked unavailable while video validation and independently available analyses continue. When configured, the pose model runs Ultralytics tracking with `persist=True` and `POSE_TRACKER` (default `bytetrack.yaml`) on the sampled frames. COCO shoulder, hip, knee, and ankle keypoints, detected person boxes, actual box confidence, and temporary track IDs are stored by frame in `pose_detection_frames`; Results links sampled pose times to the source player and displays approximate track counts. A fresh model/tracker is created per video job. Polygon occupancy emits separate `zone_entry`/`zone_exit` records only when a real track ID and person box are available; these are not anomaly events. Fall candidates remain a separate temporal heuristic event type.

The fall transition rule requires previously upright torso evidence, a configurable rapid vertical drop, and configurable sustained horizontal torso evidence. It records candidate start/end timestamps, component evidence, model name/version, and review score. Tune `FALL_TRANSITION_WINDOW_SECONDS`, `FALL_MIN_VERTICAL_DROP`, `FALL_MIN_HORIZONTAL_PERSISTENCE`, and `FALL_CANDIDATE_SCORE_THRESHOLD` in `.env`; these settings are experimental and are not accuracy guarantees.

### Optional object detection and person tracking

First install a PyTorch build appropriate for your Windows CPU/CUDA setup from the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install the optional package:

```powershell
python -m pip install -r requirements-ai.txt
```

Acquire a compatible local Ultralytics COCO detection weight file (the small `yolo26n.pt` model is one documented option) from an authorized source and review [Ultralytics license terms](https://www.ultralytics.com/license). No weight file is included or fetched by this project. Set `OBJECT_MODEL_PATH=C:\Models\yolo26n.pt` in `.env`; `OBJECT_MODEL_CONFIDENCE` and `OBJECT_TRACKER` configure the threshold and tracker. Restart Flask. If the optional package or local weights are unavailable, face/video processing continues and object analysis is explicitly shown unavailable. The test suite uses a fake detector for integration tests; it does not claim real model inference was tested.

### Start the browser website

```powershell
python app.py
```

Open http://127.0.0.1:5000. If PowerShell blocks virtual-environment activation, use:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `MONGO_URI`, `MONGO_DB_NAME` | no URI by default, `human_anomaly_db` | Private Atlas connection URI/database |
| `SECRET_KEY` | generated per process if omitted | Flask session signature; set a stable private value |
| `MAX_UPLOAD_MB` | 250 | Maximum request size |
| `MAX_DURATION_SECONDS` | 900 | Maximum clip duration |
| `ANNOTATED_OUTPUT_MAX_WIDTH` | 640 | Maximum width of the generated preview; detector inference still uses original-resolution source frames |
| `SAMPLE_EVERY_SECONDS` | 1.0 | Frame inference sampling interval |
| `PROCESSING_WORKERS` | 1 | Concurrent jobs in this Flask process |
| `PROCESSING_QUEUE_SIZE` | 3 | Additional queued jobs |
| `POSE_MODEL_PATH` | unset | Optional local pose weight path |
| `POSE_TRACKER` | `bytetrack.yaml` | Ultralytics tracker for person IDs on sampled pose frames |
| `FALL_TRANSITION_WINDOW_SECONDS` | 1.5 | Maximum time gap for rapid-drop evidence |
| `FALL_MIN_VERTICAL_DROP` | 0.12 | Minimum drop as a fraction of frame height |
| `FALL_MIN_HORIZONTAL_PERSISTENCE` | 0.6 | Required horizontal torso duration in seconds |
| `FALL_CANDIDATE_SCORE_THRESHOLD` | 0.55 | Minimum heuristic score to emit a candidate |
| `OBJECT_MODEL_PATH` | unset | Optional local Ultralytics detection weights; no auto-download |
| `OBJECT_MODEL_CONFIDENCE` | 0.25 | Detector confidence threshold |
| `OBJECT_TRACKER` | `bytetrack.yaml` | Ultralytics tracker configuration |
| `UPLOAD_DIR`, `PROCESSED_DIR` | project folders | Local originals and annotated output |

## Collections and architecture

`videos` stores original filename, random internal file reference, upload date, duration, FPS, resolution, processing state/time, per-stage status, analysis settings and summaries. `processing_jobs` stores video reference, queued/running/complete/failed state, progress and timestamps. `detection_events` stores video ObjectId, event type, label, video timestamp or interval, nullable model confidence, optional heuristic score, method, frame/track references and evidence metadata. `object_detection` records are actual sampled detections and are not counted as anomaly events. `pose_detection_frames` stores bounded per-sampled-frame keypoint evidence. Large files are not stored in MongoDB.

```mermaid
flowchart LR
 Browser --> Flask[Flask routes and Jinja]
 Flask --> Mongo[(MongoDB: videos, processing_jobs, detection_events)]
 Flask --> Queue[Bounded in-process thread pool]
 Queue --> CV[OpenCV streaming decoder and sampler]
 CV --> Obj[Optional YOLO object detector and ByteTrack]
 Obj --> Mongo
 CV --> Pose[Optional YOLO pose model]
 Pose --> PoseFrames[(MongoDB pose_detection_frames)]
 Pose --> Rule[Temporal fall-transition heuristic]
 CV --> Local[(Local uploads and processed files)]
 Rule --> Mongo
 Mongo --> Summary[Evidence-grounded summary and CSV]
 Mongo --> Pages[Results timeline, dashboard, history, reports]
```

## Tests

Tests use an isolated in-memory `mongomock` database; they never connect to or clear the configured database. Run:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

The suite covers Flask pages, Mongo connection failure, repository persistence, event/report queries, upload and validation errors, original and annotated media retrieval/download, OpenCV cascade checks, face-result persistence, retry/interruption handling, deletion boundaries, ObjectId checks, synthetic pose heuristic inputs, and object event normalization/persistence through a fake detector. It does not download model files or verify real Ultralytics inference, real-world model performance, positive face-video quality, or a live Atlas server; perform those checks separately with consented footage and configured services.

## Troubleshooting

- **Setup page says MongoDB is not configured:** open the project-root `.env` file and replace the entire example value after `MONGO_URI=` with the URI from Atlas Connect -> Drivers -> Python -> PyMongo. Save the file and restart Flask. Never add the real URI to `.env.example` or share it.
- **MongoDB unavailable:** confirm `.env` is beside `app.py`, the URI/user/password are valid, Atlas allows your current IP, and the cluster is running. Restart Flask after edits. Error pages do not reveal credentials.
- **Invalid password with special characters:** URL-encode reserved characters in the URI password component.
- **Unreadable video:** use a short MP4 (H.264 is commonly supported) or a codec available in your OpenCV build. Extension alone does not prove the content is a video.
- **Model cannot load:** confirm `requirements-ai.txt`, PyTorch hardware build, model path and weights. Do not commit large weights or secrets.
- **No events:** expected in validation-only mode, or when the heuristic did not observe its transition conditions. It is not evidence that no fall occurred.
- **No annotated preview:** produced only when a configured detector and a working OpenCV MP4 writer are available. Original playback and download remain available even when output encoding is unsupported.

## Privacy and project submission

Use consented, non-sensitive footage. The local demo has no authentication, account-based authorization, HTTPS deployment, or distributed job recovery; do not expose it to the public internet. Media is served through record IDs and resolved within configured folders. Delete records from history to remove their local files and associated events. The system does not recognize identity or label a person as dangerous.

See [college project report](docs/PROJECT_REPORT.md) and [viva preparation](docs/VIVA_PREPARATION.md). They distinguish implemented behavior from future work; add only sources and experiment results you actually verify.

## Face detection and saved video library

The existing History page is also the video library. It lists stored uploads, file sizes, upload dates, processing/face status, anomaly-event counts, results links, and safe download links. The Results page streams the original from the local `uploads` folder, offers a separate download, and shows sampled-frame face counts and timestamps. When a video writer is available, the worker creates a separate annotated MP4 under `processed`; it never edits the original. The download routes validate the stored internal filename and resolve it within the configured storage directory before streaming.

Face detection uses OpenCV's `CascadeClassifier` with the pretrained `haarcascade_frontalface_default.xml` file shipped in the installed `opencv-python` package. This requires no extra Python package or separate model download in the current environment. The cascade is the OpenCV frontal-face model from the [OpenCV source tree](https://github.com/opencv/opencv/tree/4.x/data/haarcascades); its XML includes the Intel Open Source Computer Vision Library license notice and redistribution conditions. Review that notice and the OpenCV package license before redistributing the package. OpenCV documents the cascade detector and its `detectMultiScale` behavior in its [CascadeClassifier reference](https://docs.opencv.org/4.12.0/d1/de5/classcv_1_1CascadeClassifier.html).

The cascade detects frontal face locations; it is not face recognition and returns no calibrated confidence score. Results are sampled at `SAMPLE_EVERY_SECONDS`, so counts represent face observations on sampled frames, not unique people or a count of everyone who appeared in the video. Small, turned-away, obscured, poorly lit, or distant faces can be missed, and false detections are possible. The app stores a separate `face_analysis` summary on each video and sampled-frame records in `face_detection_frames`; they are not inserted into the anomaly `detection_events` collection. Existing anomaly processing remains independent.

By default, `FACE_CASCADE_PATH` is empty and the app uses the cascade path provided by `cv2.data.haarcascades`. Set it in `.env` only to use another compatible OpenCV cascade XML file. The app does not download model files automatically. If the bundled cascade is absent or an override cannot be loaded, face status is shown as unavailable and the other video/anomaly processing continues.

On application startup, persisted jobs left queued or running by a previous local Flask process are marked `interrupted`, with the same status applied to their video records. History and Results provide a retry action for failed/interrupted videos if the original file still exists. Retry claims the video with a conditional MongoDB update so repeated requests cannot start two workers for the same record. Partial results from earlier attempts are retained in MongoDB with `superseded: true` and excluded from current result pages and analytics. This recovery assumes a single local Flask worker process; this in-process queue is not a distributed multi-server job system.

Existing video documents need no destructive migration. Older records without `file_size` or `face_analysis` remain viewable and show a not-run/unknown face state. The app creates an index for `face_detection_frames` at startup. Face frame records are removed when their owning video is deleted.
#   H u m a n - A n o m a l y - d e t e c t i o n - 
 
 #   H u m a n - A n o m a l y - d e t e c t i o n - 
 
 

## Optional polygon surveillance zones

On **Analyze video**, select a clip, wait for the first-frame preview, then choose **Draw Zone** and click at least three points. **Finish Polygon** or click near the first point to close it. You can add multiple zones, rename them before drawing, disable all zones for the upload, undo, or clear them. The browser sends normalized frame coordinates (each x/y value is between 0 and 1); the server validates and saves the polygons with the video record. Generated annotated output may show translucent zone overlays; the uploaded source is never modified.

Zone-specific fall-candidate association runs only when a configured pose model emits a candidate and its result includes the selected person's bounding box. The documented test point is that box's bottom center, checked against each enabled polygon using OpenCV. A point inside a polygon is labelled **Inside active zone**; a localized candidate outside is labelled **Outside active zones**. With no pose model, or if a candidate has no usable person location, the application reports that zone association is unavailable and does not infer a location from face or generic object detections. Polygon setup does not add anomaly predictions by itself.

## Optional suspected-fighting review analysis

The optional X3D-M temporal Violence / Non-Violence classifier setup, model limitations, source, licensing notes, frame sampling, configuration, and troubleshooting are documented in [docs/FIGHT_DETECTION.md](docs/FIGHT_DETECTION.md). Install its optional dependencies with `.venv\Scripts\python.exe -m pip install -r requirements-ai.txt`. The detector does not identify individuals or associate clip-level results with polygon zones.

