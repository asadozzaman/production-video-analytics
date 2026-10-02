"""Run the repository detection or tracking pipeline on a local video."""

from __future__ import annotations

import argparse
import math
import os
import platform
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Sequence

import cv2
import numpy as np
import torch
import ultralytics
import yaml

from .detection import Detection
from .detectors import UltralyticsYOLODetector
from .results import result_streams, write_summary
from .tracking import NullTracker, Track, Tracker
from .trackers import ByteTrackAdapter
from .video_pipeline import VideoAnalyticsPipeline


@dataclass(frozen=True)
class ByteTrackSettings:
    track_high_threshold: float
    track_low_threshold: float
    new_track_threshold: float
    grace_frames: int
    match_threshold: float
    fuse_score: bool


@dataclass(frozen=True)
class RunSettings:
    source: Path
    model: Path
    detector_backend: str
    confidence_threshold: float
    iou_threshold: float
    image_size: int
    device: str
    tracker_backend: str
    bytetrack: ByteTrackSettings | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Input video path")
    parser.add_argument("--model", type=Path, help="YOLO weights path")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output", type=Path, default=Path("outputs/step3_repo_yolo.mp4"))
    output_group.add_argument("--output-dir", type=Path, help="Directory for annotated.mp4 and result files")
    parser.add_argument("--device", help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--conf", type=float, help="Confidence threshold")
    parser.add_argument("--iou", type=float, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, help="Inference image size")
    parser.add_argument("--tracker", choices=("none", "bytetrack"), default="none")
    return parser.parse_args()


def make_pipeline(args: argparse.Namespace) -> tuple[VideoAnalyticsPipeline, RunSettings]:
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

    tracking_config = config["tracking"]
    bytetrack = None
    if args.tracker == "bytetrack":
        bytetrack = ByteTrackSettings(
            track_high_threshold=float(tracking_config["track_high_threshold"]),
            track_low_threshold=float(tracking_config["track_low_threshold"]),
            new_track_threshold=float(tracking_config["new_track_threshold"]),
            grace_frames=int(tracking_config["grace_frames"]),
            match_threshold=float(tracking_config["match_threshold"]),
            fuse_score=bool(tracking_config["fuse_score"]),
        )
    settings = RunSettings(
        source=Path(source),
        model=Path(model),
        detector_backend=str(detection_config["backend"]),
        confidence_threshold=float(args.conf if args.conf is not None else detection_config["confidence_threshold"]),
        iou_threshold=float(args.iou if args.iou is not None else detection_config["iou_threshold"]),
        image_size=int(args.imgsz if args.imgsz is not None else detection_config["image_size"]),
        device=str(args.device if args.device is not None else detection_config["device"]),
        tracker_backend=args.tracker,
        bytetrack=bytetrack,
    )
    detector = UltralyticsYOLODetector(
        model_path=settings.model,
        confidence_threshold=settings.confidence_threshold,
        iou_threshold=settings.iou_threshold,
        image_size=settings.image_size,
        device=settings.device,
    )
    tracker: Tracker = NullTracker()
    if bytetrack is not None:
        tracker = ByteTrackAdapter(
            track_high_threshold=bytetrack.track_high_threshold,
            track_low_threshold=bytetrack.track_low_threshold,
            new_track_threshold=bytetrack.new_track_threshold,
            grace_frames=bytetrack.grace_frames,
            match_threshold=bytetrack.match_threshold,
            fuse_score=bytetrack.fuse_score,
        )
    return VideoAnalyticsPipeline(detector, tracker), settings


def draw_detections(frame: np.ndarray, detections: Sequence[Detection]) -> None:
    for detection in detections:
        x1, y1, x2, y2 = (round(value) for value in detection.xyxy)
        label = f"{detection.class_name or detection.class_id} {detection.confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)


def draw_tracks(frame: np.ndarray, tracks: Sequence[Track], detections: Sequence[Detection]) -> None:
    class_names = {detection.class_id: detection.class_name for detection in detections}
    for track in tracks:
        x1, y1, x2, y2 = (round(value) for value in track.xyxy)
        class_name = class_names.get(track.class_id) or str(track.class_id)
        label = f"ID {track.track_id} {class_name} {track.confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (max(0, x1), max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)


def build_summary(
    *,
    settings: RunSettings,
    run_name: str,
    started_at_utc: str,
    output_video: Path,
    results_directory: Path,
    width: int,
    height: int,
    source_fps: float,
    reported_frames: int,
    processed_frames: int,
    total_detections: int,
    total_track_observations: int,
    unique_track_ids: set[int],
    pipeline: VideoAnalyticsPipeline,
    wall_seconds: float,
) -> dict[str, object]:
    performance: dict[str, float] = {
        "total_wall_seconds": wall_seconds,
        "average_detection_latency_ms": pipeline.metrics.detection.mean_ms,
        "detection_throughput_fps": pipeline.metrics.detection.fps,
        "end_to_end_processing_throughput_fps": processed_frames / wall_seconds,
    }
    if settings.bytetrack is not None:
        performance["average_tracking_latency_ms"] = pipeline.metrics.tracking.mean_ms
        performance["tracking_throughput_fps"] = pipeline.metrics.tracking.fps

    return {
        "schema_version": 1,
        "run": {
            "name": run_name,
            "started_at_utc": started_at_utc,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "completed",
            "annotated_video": str(output_video),
            "results_directory": str(results_directory),
        },
        "input": {
            "source": str(settings.source),
            "width": width,
            "height": height,
            "source_fps": source_fps,
            "reported_frame_count": reported_frames if reported_frames > 0 else None,
            "processed_frame_count": processed_frames,
            "duration_seconds": reported_frames / source_fps if reported_frames > 0 else None,
        },
        "model": {
            "path": str(settings.model),
            "backend": settings.detector_backend,
            "confidence_threshold": settings.confidence_threshold,
            "iou_threshold": settings.iou_threshold,
            "image_size": settings.image_size,
            "device": settings.device,
        },
        "tracker": {
            "enabled": settings.bytetrack is not None,
            "backend": settings.tracker_backend,
            "configuration": asdict(settings.bytetrack) if settings.bytetrack is not None else {},
        },
        "results": {
            "total_detections": total_detections,
            "total_track_observations": total_track_observations,
            "unique_track_ids": len(unique_track_ids),
        },
        "performance": performance,
        "environment": {
            "python_version": platform.python_version(),
            "ultralytics_version": ultralytics.__version__,
            "pytorch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        },
    }


def publish_results(stage: Path, output_dir: Path | None, output: Path) -> tuple[Path, Path]:
    if output_dir is not None:
        stage.rename(output_dir)
        return output_dir / "annotated.mp4", output_dir

    results_directory = output.parent / output.stem
    results_directory.mkdir(parents=True, exist_ok=True)
    summary_path = results_directory / "summary.json"
    summary_path.unlink(missing_ok=True)
    os.replace(stage / "annotated.mp4", output)
    os.replace(stage / "tracks.csv", results_directory / "tracks.csv")
    os.replace(stage / "frames.json", results_directory / "frames.json")
    os.replace(stage / "summary.json", summary_path)
    return output, results_directory


def run(args: argparse.Namespace) -> None:
    started = perf_counter()
    started_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    output_dir: Path | None = args.output_dir
    output = output_dir / "annotated.mp4" if output_dir is not None else args.output
    results_directory = output_dir if output_dir is not None else output.parent / output.stem
    if output_dir is not None and output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    pipeline, settings = make_pipeline(args)
    if not settings.source.is_file():
        raise FileNotFoundError(f"Input video not found: {settings.source}")
    if settings.source.resolve() == output.resolve():
        raise ValueError("Input and output paths must differ.")

    parent = output.parent if output_dir is None else output_dir.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Cannot create output directory: {parent}") from exc
    stage = Path(tempfile.mkdtemp(prefix=f".{results_directory.name}-", dir=parent))
    if stage.resolve().parent != parent.resolve():
        raise RuntimeError(f"Temporary results directory escaped output parent: {stage}")
    try:
        capture = cv2.VideoCapture(str(settings.source))
        writer: cv2.VideoWriter | None = None
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot open input video: {settings.source}")

        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
                raise RuntimeError(f"Invalid video metadata: {settings.source}")
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode first frame: {settings.source}")

            staged_video = stage / "annotated.mp4"
            writer = cv2.VideoWriter(str(staged_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot create output VideoWriter: {staged_video}")

            pipeline.reset()
            processed = 0
            total_detections = 0
            total_track_observations = 0
            unique_track_ids: set[int] = set()
            with result_streams(stage) as write_result:
                while ok:
                    if frame.shape[:2] != (height, width):
                        raise RuntimeError(f"Unexpected frame size at index {processed}: {frame.shape[:2]}")
                    result = pipeline.process_frame(frame, processed, processed / fps)
                    write_result(result)
                    if settings.bytetrack is not None:
                        draw_tracks(frame, result.tracks, result.detections)
                    else:
                        draw_detections(frame, result.detections)
                    writer.write(frame)
                    processed += 1
                    total_detections += len(result.detections)
                    total_track_observations += len(result.tracks)
                    unique_track_ids.update(track.track_id for track in result.tracks)
                    ok, frame = capture.read()

            if expected_frames > 0 and processed != expected_frames:
                raise RuntimeError(f"Video ended after {processed} frames; metadata reports {expected_frames}.")
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        if not staged_video.is_file() or staged_video.stat().st_size == 0:
            raise RuntimeError(f"Annotated video was not written: {staged_video}")
        wall_seconds = perf_counter() - started
        summary = build_summary(
            settings=settings,
            run_name=results_directory.name,
            started_at_utc=started_at_utc,
            output_video=output,
            results_directory=results_directory,
            width=width,
            height=height,
            source_fps=fps,
            reported_frames=expected_frames,
            processed_frames=processed,
            total_detections=total_detections,
            total_track_observations=total_track_observations,
            unique_track_ids=unique_track_ids,
            pipeline=pipeline,
            wall_seconds=wall_seconds,
        )
        write_summary(stage / "summary.json", summary)
        output, results_directory = publish_results(stage, output_dir, output)
    finally:
        if stage.exists():
            if stage.resolve().parent != parent.resolve():
                raise RuntimeError(f"Refusing to remove unexpected temporary path: {stage}")
            shutil.rmtree(stage)

    detection = pipeline.metrics.detection
    print(f"Source: {settings.source}")
    print(f"Output: {output}")
    print(f"Results directory: {results_directory}")
    print(f"Processed frames: {processed}")
    print(f"Total detections: {total_detections}")
    if args.tracker == "bytetrack":
        print(f"Total track observations: {total_track_observations}")
        print(f"Unique track IDs observed: {len(unique_track_ids)}")
    print(f"Source video FPS: {fps:.2f}")
    print(f"Resolution: {width}x{height}")
    print(f"Total wall time: {wall_seconds:.2f} s")
    print(f"Average detection latency: {detection.mean_ms:.2f} ms/frame")
    print(f"Detection throughput (inference stage): {detection.fps:.2f} frames/s")
    if args.tracker == "bytetrack":
        tracking = pipeline.metrics.tracking
        print(f"Average tracking latency: {tracking.mean_ms:.2f} ms/frame")
        print(f"Tracking throughput (association stage): {tracking.fps:.2f} frames/s")
    print(f"End-to-end processing throughput: {processed / wall_seconds:.2f} frames/s")


def main() -> None:
    try:
        run(parse_args())
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
