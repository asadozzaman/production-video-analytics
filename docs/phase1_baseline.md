# Phase 1 — Working Video Analytics Baseline

## Objective

Establish a reproducible end-to-end video analytics baseline before custom
dataset preparation or model training.

Target pipeline:

Video input
→ YOLO detection
→ ByteTrack tracking
→ persistent track IDs
→ annotated video
→ JSON/CSV results
→ runtime metrics

## Baseline assets

### Input video

A public real-world traffic/pedestrian video is used for the initial pipeline
validation.

Local path:

`data/input/test_traffic.mp4`

The source video is intentionally not committed to Git.

### Detection model

Pretrained Ultralytics YOLO26n.

Local path:

`models/yolo26n.pt`

Model weights are intentionally excluded from Git and can be downloaded
separately.

## Scope

This phase validates the engineering pipeline only.

It does not include:

- custom annotation
- custom model training
- dataset development
- FastAPI
- Docker
- cloud deployment

These will be introduced after the baseline pipeline is reproducible.

## Success criteria

The baseline is complete when one command can process the test video and
produce:

- annotated MP4
- frame-level JSON
- track-level CSV
- run summary
- detection/tracking timing metrics

## Verified YOLO Detection Baseline

Status: **PASS**

The baseline was executed locally on CPU and the complete test video was
processed successfully.

### Runtime environment

| Component | Version / Value |
|---|---|
| Python | 3.10.10 |
| Ultralytics | 8.4.171 |
| PyTorch | 2.14.1+cpu |
| CUDA available | False |
| Execution device | CPU |

### Input video

| Property | Value |
|---|---|
| File | `data/input/test_traffic.mp4` |
| Codec | H.264 |
| Resolution | 1920 × 1080 |
| Frame rate | 25 FPS |
| Frame count | 393 |
| Duration | 15.72 seconds |

### Detection configuration

- Model: YOLO26n
- Weights: `models/yolo26n.pt`
- Confidence threshold: 0.25
- Inference image size: 640
- Device: CPU

### Baseline command

```bash
yolo predict \
  model=models/yolo26n.pt \
  source=data/input/test_traffic.mp4 \
  conf=0.25 \
  imgsz=640 \
  device=cpu \
  save=True \
  project=outputs \
  name=step2_yolo_baseline
```

## Step 3 — Repository YOLO detection pipeline

Status: **PASS** (detection only, validated locally on CPU)

The repository runner decodes frames with OpenCV, passes them through
`UltralyticsYOLODetector` behind the `Detector` interface, converts model
results to `Detection` objects, and calls `VideoAnalyticsPipeline.process_frame()`.
The runner draws those detections and writes the annotated video. `NullTracker`
returns no tracks; tracking and persistent IDs are not enabled yet.

Command used from the repository root (PowerShell):

```powershell
& 'env/Scripts/python.exe' -m src.run_video --source data/input/test_traffic.mp4 --model models/yolo26n.pt --output outputs/step3_repo_yolo.mp4 --device cpu --conf 0.25 --imgsz 640
```

The unspecified IoU threshold came from `configs/default.yaml` (0.45).
The output is `outputs/step3_repo_yolo.mp4` and is excluded from Git.

The source video is 1920 × 1080 at 25.00 FPS. Source video FPS describes
the media playback rate. Detection throughput divides processed frames by
time spent in the detection stage. End-to-end processing throughput divides
processed frames by total wall time, including model load and MP4 finalization.

| Runtime result | Codex validation run | Independent verification (user-reported) |
|---|---:|---:|
| Processed frames | 393 | 393 |
| Total detections | 2,288 | 2,288 |
| Total wall time | 32.74 s | 38.99 s |
| Average detection latency | 58.76 ms/frame | 71.31 ms/frame |
| Detection throughput (inference stage) | 17.02 frames/s | 14.02 frames/s |
| End-to-end processing throughput | 12.00 frames/s | ≈10.08 frames/s |

The output was reopened with OpenCV and all 393 frames decoded. A sampled
output frame visibly contains bounding boxes, class names, and confidence
scores. This confirms the same basic detection operation as the verified raw
Ultralytics CLI baseline, now routed through the repository's abstractions.
The user reported that the independent verification also completed the full
pipeline and passed `python -m compileall src` and `git diff --check`. These
are separate local runs, not a controlled speed comparison with each other
or the CLI baseline.

## Step 4 — Repository ByteTrack integration

Status: **PASS** (functional integration on CPU; no formal tracking-accuracy
evaluation)

The independently verified reference baseline used Ultralytics `yolo track`
with `tracker=bytetrack.yaml`, `conf=0.25`, `imgsz=640`, and `device=cpu` on
the same 393-frame video. The user visually observed persistent IDs in that
reference output: car ID 101, truck ID 112, and fire hydrant ID 30 across
consecutive frames. These reference IDs are separate from the repository run.

The repository command used from its root (PowerShell) was:

```powershell
& 'env/Scripts/python.exe' -m src.run_video --source data/input/test_traffic.mp4 --model models/yolo26n.pt --output outputs/step4_repo_bytetrack.mp4 --device cpu --conf 0.25 --imgsz 640 --tracker bytetrack
```

The runner decodes each frame with OpenCV. `UltralyticsYOLODetector` emits
repository `Detection` objects; `ByteTrackAdapter` converts those to the
NumPy-backed box input expected by Ultralytics `BYTETracker.update()` and
converts its active outputs to repository `Track` objects. The pipeline owns
stage timing and the runner draws IDs, class names, and scores from `Track`
objects. The detector does not call `model.track()`.

The tracker settings in `configs/default.yaml` map as follows:

| Repository setting | Ultralytics ByteTrack setting | Value |
|---|---|---:|
| `track_high_threshold` | `track_high_thresh` | 0.5 |
| `track_low_threshold` | `track_low_thresh` | 0.1 |
| `new_track_threshold` | `new_track_thresh` | 0.6 |
| `grace_frames` | `track_buffer` (maximum lost frames retained) | 30 |
| `match_threshold` | `match_thresh` | 0.8 |
| `fuse_score` | `fuse_score` | true |

These repository high and new-track thresholds differ from Ultralytics
`bytetrack.yaml` defaults (both 0.25). The detector's 0.25 confidence cutoff
also means ByteTrack never receives boxes below 0.25, even though its
low threshold is 0.1. Results from the two paths are therefore not a
controlled tracker comparison.

`Track.age` is elapsed frames since ByteTrack's `start_frame`, including any
lost interval. `Track.missed_frames` is the difference between the current
tracker frame and the track's last update; active tracks returned by this
adapter normally have zero missed frames. No values are fabricated for tracks
that ByteTrack does not return.

| Observed repository result | Value |
|---|---:|
| Processed frames | 393 |
| Total detections | 2,288 |
| Total track observations | 1,761 |
| Unique track IDs observed | 42 |
| Source video FPS | 25.00 |
| Output resolution | 1920 × 1080 |
| Total wall time | 30.04 s |
| Average detection latency | 51.52 ms/frame |
| Detection throughput (inference stage) | 19.41 frames/s |
| Average tracking latency | 1.42 ms/frame |
| Tracking throughput (association stage) | 701.88 frames/s |
| End-to-end processing throughput | 13.08 frames/s |

Output: `outputs/step4_repo_bytetrack.mp4` (excluded from Git). It was reopened
with OpenCV and all 393 frames decoded at 1920 × 1080 and 25 FPS. In output
frames 100 and 101, the same truck retained ID 11, the same car retained ID
14, and the same fire hydrant retained ID 5. This confirms visible ID
continuity in these examples, not tracking accuracy. Formal validation will
require ground truth and metrics such as HOTA, IDF1, ID switches, and
fragmentation.

The focused adapter tests, syntax/import checks, and `git diff --check`
passed. The full detection-only Step 3 command also completed all 393 frames
with 2,288 detections after the tracking addition.

## Step 5 — Machine-readable run results

Status: **PASS** (functional artifact and consistency validation on CPU)

The repository runner now writes one frame entry and zero or more track CSV
rows from each `FrameResult` while keeping the annotated video workflow from
Step 4. Serialization lives in `src/results.py`; the detector and tracker do
not write output files.

Command used from the repository root (PowerShell):

```powershell
& 'env/Scripts/python.exe' -m src.run_video --source data/input/test_traffic.mp4 --model models/yolo26n.pt --output-dir outputs/step5_machine_results --device cpu --conf 0.25 --imgsz 640 --tracker bytetrack
```

The completed directory contains:

```text
outputs/step5_machine_results/
├── annotated.mp4
├── tracks.csv
├── frames.json
└── summary.json
```

`tracks.csv` is UTF-8 with one row per active track observation. Its columns,
in order, are `frame_index`, `timestamp_seconds`, `track_id`, `class_id`,
`class_name`, `confidence`, `x1`, `y1`, `x2`, and `y2`. Detection-only runs
write the header with no fabricated track rows.

`frames.json` has top-level `schema_version: 1` and a `frames` array. Each
entry has `frame_index`, `timestamp_seconds`, `detections`, and `tracks`.
Detections contain `xyxy`, `confidence`, `class_id`, and `class_name`. Tracks
contain `track_id`, `xyxy`, `confidence`, `class_id`, `class_name`, `age`, and
`missed_frames`. Coordinates and timestamps are JSON numbers; missing class
names are `null`. Empty detection or track lists are `[]`.

`summary.json` has top-level `schema_version: 1` and the groups `run`,
`input`, `model`, `tracker`, `results`, `performance`, and `environment`. It
records the effective CLI/YAML settings used to construct the detector and
tracker. For this run it records model `models/yolo26n.pt`, confidence 0.25,
IoU 0.45, image size 640, device `cpu`, and the ByteTrack settings from
`configs/default.yaml`. It also records Python 3.10.10, Ultralytics 8.4.171,
PyTorch 2.14.1+cpu, and CUDA unavailable. No unmeasured resource or accuracy
claims are included.

Frame and CSV records stream to a temporary directory, so decoded image
arrays are not accumulated for output. Once the video writer closes, the
runner writes a completed `summary.json` and renames the temporary directory
to the requested output directory. An existing `--output-dir` is rejected to
avoid replacing a completed run. The legacy `--output path.mp4` still writes
the MP4 at that path and places the three structured files in a sibling
`path/` directory; its summary is promoted last as the completion marker.
Total wall time covers model loading through video and frame-result writing,
before summary creation and publication.

| Observed repository result | Value |
|---|---:|
| Processed frames | 393 |
| Total detections | 2,288 |
| Total track observations | 1,761 |
| Unique track IDs observed | 42 |
| Source video FPS | 25.00 |
| Output resolution | 1920 × 1080 |
| Total wall time | 26.39 s |
| Average detection latency | 44.28 ms/frame |
| Detection throughput (inference stage) | 22.58 frames/s |
| Average tracking latency | 1.24 ms/frame |
| Tracking throughput (association stage) | 806.88 frames/s |
| End-to-end processing throughput | 14.89 frames/s |

The MP4 reopened and decoded all 393 frames at 1920 × 1080 and 25 FPS; a
sampled frame visibly retained boxes, IDs, class names, and confidence scores.
The JSON reopened and parsed with 393 sequential frame indices (0–392) and
monotonic timestamps. Its detections summed to 2,288; its tracks and CSV rows
both summed to 1,761. The CSV contained 42 distinct track IDs, matching the
summary. Every CSV frame index and timestamp matched its frame entry.

Five focused unit tests passed, including the existing ByteTrack tests.
`python -m compileall src` and `git diff --check` passed. A separate full
detection-only run also processed 393 frames and 2,288 detections, produced
parseable JSON and a header-only CSV, and decoded a complete MP4.

These artifacts demonstrate pipeline and schema integration. They do not
measure detection or tracking accuracy, ID switches, memory use, or cost.
