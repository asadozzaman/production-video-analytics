"""Generate frame-aligned ByteTrack predictions for a CVAT source range."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import cv2

from .ground_truth import import_cvat_zip
from .run_video import make_pipeline
from .video_pipeline import FrameResult


@dataclass(frozen=True, slots=True)
class PredictionObservation:
    frame_index: int
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class InvalidPredictionBox:
    frame_index: int
    track_id: int
    reason: str


@dataclass(frozen=True, slots=True)
class PredictionValidation:
    expected_start_frame: int
    expected_stop_frame: int
    first_prediction_frame: int | None
    last_prediction_frame: int | None
    evaluation_frames_processed: int
    total_prediction_observations: int
    predictions_per_class: dict[str, int]
    unique_track_ids: int
    unique_ids_per_class: dict[str, int]
    invalid_boxes: tuple[InvalidPredictionBox, ...]
    out_of_image_boxes: tuple[tuple[int, int], ...]
    predictions_outside_range: tuple[int, ...]
    unsupported_classes: tuple[str, ...]
    filtered_other_classes: dict[str, int]
    alignment_errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not (
            self.invalid_boxes
            or self.predictions_outside_range
            or self.unsupported_classes
            or self.alignment_errors
        )


def validate_predictions(
    *,
    start_frame: int,
    stop_frame: int,
    allowed_classes: tuple[str, ...],
    evaluation_frames: tuple[int, ...],
    predictions: tuple[PredictionObservation, ...],
    invalid_boxes: tuple[InvalidPredictionBox, ...] = (),
    out_of_image_boxes: tuple[tuple[int, int], ...] = (),
    filtered_other_classes: dict[str, int] | None = None,
) -> PredictionValidation:
    expected_frames = tuple(range(start_frame, stop_frame + 1))
    alignment_errors: list[str] = []
    if evaluation_frames != expected_frames:
        alignment_errors.append(
            f"Evaluation frames must be every source frame {start_frame}-{stop_frame} in order; "
            f"received {len(evaluation_frames)} frames."
        )
    frame_set = set(evaluation_frames)
    outside = tuple(sorted({prediction.frame_index for prediction in predictions if not start_frame <= prediction.frame_index <= stop_frame}))
    missing_frame_entries = sorted({prediction.frame_index for prediction in predictions if prediction.frame_index not in frame_set})
    if missing_frame_entries:
        alignment_errors.append(f"Predictions lack matching evaluation frame entries: {missing_frame_entries}")
    unsupported = tuple(sorted({prediction.class_name for prediction in predictions if prediction.class_name not in allowed_classes}))
    class_counts = Counter(prediction.class_name for prediction in predictions)
    class_ids: dict[str, set[int]] = defaultdict(set)
    for prediction in predictions:
        class_ids[prediction.class_name].add(prediction.track_id)
    return PredictionValidation(
        expected_start_frame=start_frame,
        expected_stop_frame=stop_frame,
        first_prediction_frame=evaluation_frames[0] if evaluation_frames else None,
        last_prediction_frame=evaluation_frames[-1] if evaluation_frames else None,
        evaluation_frames_processed=len(evaluation_frames),
        total_prediction_observations=len(predictions),
        predictions_per_class={name: class_counts[name] for name in allowed_classes},
        unique_track_ids=len({prediction.track_id for prediction in predictions}),
        unique_ids_per_class={name: len(class_ids[name]) for name in allowed_classes},
        invalid_boxes=invalid_boxes,
        out_of_image_boxes=out_of_image_boxes,
        predictions_outside_range=outside,
        unsupported_classes=unsupported,
        filtered_other_classes=dict(sorted((filtered_other_classes or {}).items())),
        alignment_errors=tuple(alignment_errors),
    )


class PredictionCollector:
    """Record only the evaluation window while accepting earlier tracker context."""

    def __init__(self, start_frame: int, stop_frame: int, allowed_classes: tuple[str, ...], width: int, height: int) -> None:
        if start_frame < 0 or stop_frame < start_frame or width <= 0 or height <= 0:
            raise ValueError("Invalid evaluation frame range or video dimensions.")
        self.start_frame = start_frame
        self.stop_frame = stop_frame
        self.allowed_classes = allowed_classes
        self.width = width
        self.height = height
        self.evaluation_frames: list[int] = []
        self.predictions: list[PredictionObservation] = []
        self.invalid_boxes: list[InvalidPredictionBox] = []
        self.out_of_image_boxes: list[tuple[int, int]] = []
        self.filtered_other_classes: Counter[str] = Counter()
        self._class_names: dict[int, str] = {}

    def add(self, result: FrameResult) -> None:
        frame_index = int(result.frame_index)
        for detection in result.detections:
            if detection.class_name is not None:
                self._class_names[int(detection.class_id)] = detection.class_name
        if frame_index < self.start_frame:
            return
        if frame_index > self.stop_frame:
            raise ValueError(f"Source frame {frame_index} is after evaluation stop frame {self.stop_frame}.")
        expected = self.start_frame if not self.evaluation_frames else self.evaluation_frames[-1] + 1
        if frame_index != expected:
            raise ValueError(f"Expected source frame {expected}, received {frame_index}; evaluation alignment failed.")
        self.evaluation_frames.append(frame_index)

        for track in result.tracks:
            class_id = int(track.class_id)
            class_name = self._class_names.get(class_id, f"class_id:{class_id}")
            if class_name not in self.allowed_classes:
                self.filtered_other_classes[class_name] += 1
                continue
            x1, y1, x2, y2 = (float(value) for value in track.xyxy)
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                reason = "nonfinite coordinates"
            elif x2 <= x1 or y2 <= y1:
                reason = "nonpositive box width or height"
            else:
                reason = ""
            if reason:
                self.invalid_boxes.append(InvalidPredictionBox(frame_index, int(track.track_id), reason))
                continue
            if x1 < 0 or y1 < 0 or x2 > self.width or y2 > self.height:
                self.out_of_image_boxes.append((frame_index, int(track.track_id)))
            self.predictions.append(
                PredictionObservation(
                    frame_index=frame_index,
                    track_id=int(track.track_id),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(track.confidence),
                    xyxy=(x1, y1, x2, y2),
                )
            )

    def validate(self) -> PredictionValidation:
        return validate_predictions(
            start_frame=self.start_frame,
            stop_frame=self.stop_frame,
            allowed_classes=self.allowed_classes,
            evaluation_frames=tuple(self.evaluation_frames),
            predictions=tuple(self.predictions),
            invalid_boxes=tuple(self.invalid_boxes),
            out_of_image_boxes=tuple(self.out_of_image_boxes),
            filtered_other_classes=dict(self.filtered_other_classes),
        )


def format_validation(validation: PredictionValidation) -> str:
    lines = [
        f"Expected evaluation range: {validation.expected_start_frame}-{validation.expected_stop_frame}",
        f"First prediction frame: {validation.first_prediction_frame}",
        f"Last prediction frame: {validation.last_prediction_frame}",
        f"Evaluation frames processed: {validation.evaluation_frames_processed}",
        f"Total prediction observations: {validation.total_prediction_observations}",
        "Predictions per class: " + ", ".join(f"{name}={count}" for name, count in validation.predictions_per_class.items()),
        f"Unique predicted track IDs: {validation.unique_track_ids}",
        "Unique IDs per class: " + ", ".join(f"{name}={count}" for name, count in validation.unique_ids_per_class.items()),
        f"Invalid boxes: {len(validation.invalid_boxes)}",
        f"Boxes extending beyond source image: {len(validation.out_of_image_boxes)}",
        f"Predictions outside range: {list(validation.predictions_outside_range)}",
        f"Unsupported classes in output: {list(validation.unsupported_classes)}",
        f"Filtered other-class track observations: {validation.filtered_other_classes}",
    ]
    lines.extend(f"  ID {box.track_id} frame {box.frame_index}: {box.reason}" for box in validation.invalid_boxes)
    lines.extend(f"Alignment error: {error}" for error in validation.alignment_errors)
    lines.append(f"Validation: {'PASS' if validation.valid else 'FAIL'}")
    return "\n".join(lines)


def _write_predictions(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.stem}-", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            json.dump(document, file, indent=2, allow_nan=False)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> PredictionValidation:
    ground_truth = import_cvat_zip(args.ground_truth)
    if not ground_truth.valid:
        raise ValueError("Ground-truth validation failed; fix the CVAT export before generating predictions.")
    if (ground_truth.start_frame, ground_truth.stop_frame, ground_truth.task_frame_count) != (100, 219, 120):
        raise ValueError("Expected the verified CVAT source range 100-219 (120 frames).")
    if set(ground_truth.labels) != {"person", "car", "truck"}:
        raise ValueError(f"Unexpected ground-truth classes: {ground_truth.labels}")

    pipeline, settings = make_pipeline(args)
    if not settings.source.is_file():
        raise FileNotFoundError(f"Input video not found: {settings.source}")
    capture = cv2.VideoCapture(str(settings.source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open input video: {settings.source}")
    started = perf_counter()
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
            raise RuntimeError("Invalid source video metadata.")
        if reported_frames > 0 and reported_frames <= ground_truth.stop_frame:
            raise RuntimeError(f"Source video has only {reported_frames} frames; need frame {ground_truth.stop_frame}.")
        collector = PredictionCollector(ground_truth.start_frame, ground_truth.stop_frame, ground_truth.labels, width, height)
        pipeline.reset()
        for frame_index in range(ground_truth.stop_frame + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended before source frame {frame_index}; evaluation alignment failed.")
            result = pipeline.process_frame(frame, frame_index, frame_index / fps)
            collector.add(result)
    finally:
        capture.release()

    validation = collector.validate()
    print(f"Source: {settings.source}")
    print(f"Ground truth: {args.ground_truth}")
    print(f"Tracker context: source frames 0-{ground_truth.start_frame - 1} processed before recording")
    print(f"Source frames processed: {ground_truth.stop_frame + 1}")
    print(format_validation(validation))
    if not validation.valid:
        raise RuntimeError("Prediction validation failed; no output file was written.")

    document: dict[str, object] = {
        "schema_version": 1,
        "source_video": str(settings.source),
        "ground_truth_archive": str(args.ground_truth),
        "evaluation_range": {
            "start_frame": ground_truth.start_frame,
            "stop_frame": ground_truth.stop_frame,
            "expected_frames": ground_truth.task_frame_count,
            "frame_numbering": "original source video",
        },
        "tracker_context": {
            "processed_from_frame": 0,
            "warmup_frames": ground_truth.start_frame,
            "last_processed_frame": ground_truth.stop_frame,
        },
        "model": {
            "path": str(settings.model),
            "backend": settings.detector_backend,
            "confidence_threshold": settings.confidence_threshold,
            "iou_threshold": settings.iou_threshold,
            "image_size": settings.image_size,
            "device": settings.device,
        },
        "tracker": {"backend": settings.tracker_backend, "configuration": asdict(settings.bytetrack) if settings.bytetrack else {}},
        "evaluation_frames": collector.evaluation_frames,
        "predictions": [asdict(prediction) for prediction in collector.predictions],
    }
    _write_predictions(args.output, document)
    print(f"Output: {args.output}")
    print(f"Elapsed wall time: {perf_counter() - started:.2f} s")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/input/test_traffic.mp4"))
    parser.add_argument("--model", type=Path, default=Path("models/yolo26n.pt"))
    parser.add_argument("--ground-truth", type=Path, default=Path("data/ground_truth/traffic_tracker_ground_truth_v1.zip"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tracker_evaluation/predictions_100_219.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--tracker", choices=("bytetrack",), default="bytetrack")
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
