# Frame-level motion analytics

New analyses with a configured pose detector call the existing YOLO pose model
and ByteTrack once for every successfully decoded source frame. The selected
lighting branch is applied before pose inference; enhancement changes pixel
values but not frame dimensions, so normalized polygon coordinates and pose
coordinates remain aligned. The original source is never modified. X3D-M still
receives the original decoded frame, and face/object and fall/zone decisions
retain their existing sampling cadence.

For every returned pose result, the corresponding frame is annotated before
encoding with the COCO-17 skeleton, valid joints, available person box and
temporary track ID. Current-frame smoothed joint angles are placed beside their
joint centers, and a compact per-track panel shows available image-plane speed,
direction, torso orientation, and supported movement observation. Frames with
no people receive a lighting/time banner only. Failed inference frames do not
reuse a preceding frame's pose or measurements. The output metadata records the
number of pose-result frames annotated and the number of angle labels and
per-person panels drawn.

Every-frame rows are written in bounded batches to MongoDB's additive
`motion_analytics_frames` collection. A local-only video writes append-only
JSON Lines under `<LOCAL_FALLBACK_PATH>.motion/`. A video record points to the
active analysis run ID; older embedded sampled measurements remain readable.
The API pages evidence rows and streams CSV output in batches. Run-specific
identifiers keep retries from mixing measurements from separate analysis runs.

Angles are 2D image-plane estimates from COCO-17 keypoints, not 3D anatomical
rotations. Interior joint angles use the configured keypoint triplets;
shoulder/hip lines, limb segments, and torso inclination are reported as
orientations. Raw values are retained. Smoothed values are a trailing arithmetic
mean across the last three valid observations for that joint; segment
orientations are unwrapped across the 0/360-degree boundary before averaging.
Missing/low-confidence keypoints break that joint's smoothing history and do not
produce a fabricated angle. COCO-17 contains no finger/hand or heel/toe
landmarks, so wrist-flexion and ankle-flexion angles are not measurable; forearm
and lower-leg segment directions are available instead.

Angular velocity uses the difference between valid video presentation
timestamps. Track gaps beyond the configured motion continuity window, invalid
poses, and tracker ID changes start a new motion segment. Body-center speed and
direction use image-plane pixels and video time; they are not physical speed.
Track IDs are temporary within one video run and do not represent identities.

Profiling reports decode, `model.track()` (pose inference plus ByteTrack
association), pose-evidence/overlay, geometry, fight inference, zone logic,
encoding, persistence, total time, pose throughput, and current process RSS
where `psutil` is already available. Ultralytics exposes inference and tracker
association through one `track()` call here, so their sub-times are not
independently measurable without replacing or instrumenting the framework's
internal predictor.

Older records remain explicitly marked as sampled when their saved metadata
does not identify the frame collection. No migration rewrites those records.
