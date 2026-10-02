"""Run the detection-only pipeline on a local video."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from time import perf_counter
from typing import Sequence

import cv2
import numpy as np
import yaml

from .detection import Detection
from .detectors import UltralyticsYOLODetector
from .tracking import NullTracker
from .video_pipeline import VideoAnalyticsPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Input video path")
    parser.add_argument("--model", type=Path, help="YOLO weights path")
    parser.add_argument("--output", type=Path, default=Path("outputs/step3_repo_yolo.mp4"))
    parser.add_argument("--device", help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--conf", type=float, help="Confidence threshold")
    parser.add_argument("--iou", type=float, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, help="Inference image size")
    return parser.parse_args()


def make_pipeline(args: argparse.Namespace) -> tuple[VideoAnalyticsPipeline, Path]:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    detection_config = config["detection"]
    source = args.source or config["input"]["source"]
    model = args.model or detection_config["model_path"]
    if not source:
        raise ValueError("Input video is required; pass --source.")
    if not model:
        raise ValueError("Model weights are required; pass --model.")

    detector = UltralyticsYOLODetector(
        model_path=model,
        confidence_threshold=args.conf if args.conf is not None else detection_config["confidence_threshold"],
        iou_threshold=args.iou if args.iou is not None else detection_config["iou_threshold"],
        image_size=args.imgsz if args.imgsz is not None else detection_config["image_size"],
        device=args.device if args.device is not None else detection_config["device"],
    )
    return VideoAnalyticsPipeline(detector, NullTracker()), Path(source)


def draw_detections(frame: np.ndarray, detections: Sequence[Detection]) -> None:
    for detection in detections:
        x1, y1, x2, y2 = (round(value) for value in detection.xyxy)
        label = f"{detection.class_name or detection.class_id} {detection.confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)


def run(args: argparse.Namespace) -> None:
    started = perf_counter()
    pipeline, source = make_pipeline(args)
    output = args.output
    if not source.is_file():
        raise FileNotFoundError(f"Input video not found: {source}")
    if source.resolve() == output.resolve():
        raise ValueError("Input and output paths must differ.")

    capture = cv2.VideoCapture(str(source))
    writer: cv2.VideoWriter | None = None
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open input video: {source}")

    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
            raise RuntimeError(f"Invalid video metadata: {source}")
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot decode first frame: {source}")

        try:
            output.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(f"Cannot create output directory: {output.parent}") from exc
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create output VideoWriter: {output}")

        pipeline.reset()
        processed = 0
        total_detections = 0
        while ok:
            if frame.shape[:2] != (height, width):
                raise RuntimeError(f"Unexpected frame size at index {processed}: {frame.shape[:2]}")
            result = pipeline.process_frame(frame, processed, processed / fps)
            draw_detections(frame, result.detections)
            writer.write(frame)
            processed += 1
            total_detections += len(result.detections)
            ok, frame = capture.read()

        if expected_frames > 0 and processed != expected_frames:
            raise RuntimeError(f"Video ended after {processed} frames; metadata reports {expected_frames}.")
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    wall_seconds = perf_counter() - started
    detection = pipeline.metrics.detection
    print(f"Source: {source}")
    print(f"Output: {output}")
    print(f"Processed frames: {processed}")
    print(f"Total detections: {total_detections}")
    print(f"Source video FPS: {fps:.2f}")
    print(f"Resolution: {width}x{height}")
    print(f"Total wall time: {wall_seconds:.2f} s")
    print(f"Average detection latency: {detection.mean_ms:.2f} ms/frame")
    print(f"Detection throughput (inference stage): {detection.fps:.2f} frames/s")
    print(f"End-to-end processing throughput: {processed / wall_seconds:.2f} frames/s")


def main() -> None:
    try:
        run(parse_args())
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
