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
