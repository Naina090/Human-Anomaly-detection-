# Suspected fighting review model

Sentinel's optional fighting stage is a temporal X3D-M video classifier from the author-published `visionlab-ai/school-violence-detection-models` collection. The selected file is `final/final_x3d_realtime.pt` (about 12.3 MB), published on Hugging Face at <https://huggingface.co/visionlab-ai/school-violence-detection-models>. The model card labels the two output classes **Non-Violence** and **Violence**. This is clip-level classification, not person-level action localization.

The model card states MIT licensing and reports validation results on 400 RWF-2000 validation clips. Those numbers are the publisher's claims and have not been independently reproduced in Sentinel. The RWF-2000 dataset has separate restrictions: its authors prohibit redistribution and commercial use without SMIIP Lab approval. Sentinel does not fetch or redistribute that dataset. The model card says this model cannot distinguish fights from other forms of violence and may flag sports or intense interactions. A candidate is only a review signal; it is not proof of fighting, identity, intent, or wrongdoing.

## Setup

1. Keep the approved checkpoint at `models/final_x3d_realtime.pt`. It is ignored by Git (`*.pt`).
2. Install the optional inference package in the same virtual environment as Flask:

   ```powershell
   .venv\Scripts\python.exe -m pip install -r requirements-ai.txt
   ```

   `requirements-ai.txt` pins PyTorchVideo to 0.1.5 and leaves the existing Torch/CUDA build alone. PyTorchVideo also installs `fvcore`, `iopath`, `av`, and small support packages. Install or change Torch only by using the official PyTorch selector for the target hardware.
3. Set these optional entries in `.env` (keep other settings unchanged):

   ```dotenv
   FIGHT_DETECTION_ENABLED=1
   FIGHT_MODEL_PATH=models/final_x3d_realtime.pt
   FIGHT_SCORE_THRESHOLD=0.4
   FIGHT_WINDOW_SECONDS=4.0
   FIGHT_WINDOW_STRIDE=1.0
   FIGHT_CONFIRMATION_WINDOWS=2
   FIGHT_DEVICE=auto
   ```

   Relative checkpoint paths resolve from the project directory. `FIGHT_DEVICE=auto` selects CUDA when available and otherwise uses CPU. Set `FIGHT_DETECTION_ENABLED=0` to disable this stage without affecting the existing detectors.

## Inference behavior

The worker already decodes each upload once. While it reads those frames, the fighting detector retains only a resized temporal buffer: 16 RGB frames at 224×224 over a four-second window. It evaluates overlapping windows one second apart. A candidate interval requires two consecutive windows at or above the configured 0.4 decision threshold, then overlapping candidates are merged into one continuous incident. Model softmax scores are displayed as **uncalibrated model scores**, not probabilities. The threshold and preprocessing match the selected model card's X3D-M configuration.

The classifier has no person boxes, track IDs, or spatial attention output. Therefore person attribution and polygon-zone association remain unavailable for its events. Existing restricted-zone entry and pose-based fall events continue to use their own localized YOLO evidence and are not changed by fighting scores. The original upload is not modified; existing annotated output remains separate and only includes overlays produced by the existing detectors.

If the checkpoint, PyTorchVideo package, or model initialization is unavailable, the result records `unavailable` and explains why; upload, pose/fall, and zone analysis continue. Videos shorter than the four-second clip window are reported as `insufficient_video`, not as a negative fighting classification. Inference exceptions are logged server-side with traceback and mark this optional stage failed.

## Limitations and validation

RWF-2000 is a small, dataset-specific surveillance benchmark. Its publisher's metrics do not establish performance on Sentinel footage, your camera placement, sports, rough play, or other scenes. Sentinel has no local sensitivity/specificity calibration dataset by default. Validate with consented, representative clips and document false positives and false negatives before drawing conclusions. The system does not automatically notify anyone or make decisions about people.
