# Object detection and tracking setup

Sentinel uses the existing Ultralytics integration in `app/services/object_detector.py`. It runs one locally configured detector instance for each video job, calls Ultralytics tracking with `bytetrack.yaml`, annotates sampled frames, and persists supported class observations. Track IDs are temporary associations within one video analysis, not real-world identities.

## Model

Use the official COCO-pretrained Ultralytics YOLO26 nano detection weights, `yolo26n.pt`. This is the detection model; `yolo26n-pose.pt` is a separate pose model and is not substituted for object detection. This application reports the supported subset it filters for: person, bicycle, car, motorcycle, bus, train, truck, backpack, handbag, and suitcase. Other COCO labels are ignored by this project's object summary.

Ultralytics documents `yolo26n.pt` as the COCO-pretrained YOLO26 detection model and demonstrates loading it through `YOLO("yolo26n.pt")`: [YOLO26 model documentation](https://docs.ultralytics.com/models/yolo26/) and [object detection task documentation](https://docs.ultralytics.com/tasks/detect/). The package can fetch official weights when that command is run. Review Ultralytics' [license terms](https://www.ultralytics.com/license) before redistributing or deploying the model; the default AGPL-3.0 option has source and distribution obligations, while other licensing may be required for some uses. COCO dataset licensing and attribution should also be reviewed for the intended use.

## Install the local weight file on Windows

Run this once from the project folder. It explicitly asks the installed Ultralytics package to load the named official model while the current directory is `models`, so the weight file is placed in the project's `models` folder. This is a manual setup command; the Flask app does not download model weights at startup or when a user uploads a video.

```powershell
cd "C:\Users\hp\Desktop\Human Anomaly"
New-Item -ItemType Directory -Force models | Out-Null
Push-Location models
try {
    ..\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
} finally {
    Pop-Location
}
```

Then ensure the local `.env` contains:

```dotenv
OBJECT_MODEL_PATH=models/yolo26n.pt
OBJECT_MODEL_CONFIDENCE=0.25
OBJECT_TRACKER=bytetrack.yaml
```

`OBJECT_MODEL_PATH` is resolved relative to the project root even if Flask is launched from another working directory. The existing pose configuration remains independent (`POSE_MODEL_PATH` and `POSE_TRACKER`). Do not replace the pose file with the detection file or vice versa.

Restart Flask after the file is present. The Results page should report the object detector unavailable if the configured file or Ultralytics dependency is missing, and should report actual sampled detections only after inference. A completed run with zero detections is different from an unavailable model.

## Dependencies and limitations

Install the project's existing optional AI requirements with `..\.venv\Scripts\python.exe -m pip install -r requirements-ai.txt` from the project directory. Ultralytics `track` uses ByteTrack; `lap` is already declared by the optional requirements file. No new runtime dependency is introduced by this configuration change.

COCO labels and detector confidence values are model outputs, not calibrated risk assessments. Video sampling can miss objects between sampled frames. Temporary track IDs can split or change after occlusion. Object detections are not anomaly events.
