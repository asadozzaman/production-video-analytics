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