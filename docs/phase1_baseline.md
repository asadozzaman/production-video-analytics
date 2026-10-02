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