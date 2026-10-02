"""Match source-aligned CVAT and ByteTrack observations by frame, class, and IoU."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .evaluation_predictions import PredictionObservation
from .ground_truth import GroundTruthObservation, import_cvat_zip


Box = tuple[float, float, float, float]


def validate_xyxy(box: Sequence[float]) -> Box:
    """Return finite XYXY coordinates with strictly positive width and height."""
    if len(box) != 4:
        raise ValueError("XYXY box must contain exactly four coordinates.")
    try:
        coordinates = tuple(float(value) for value in box)
    except (TypeError, ValueError) as exc:
        raise ValueError("XYXY box coordinates must be numeric.") from exc
    x1, y1, x2, y2 = coordinates
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError("XYXY box coordinates must be finite.")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("XYXY box must have positive width and height.")
    return x1, y1, x2, y2


def xyxy_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Compute IoU from raw coordinates without clipping to the image."""
    ax1, ay1, ax2, ay2 = validate_xyxy(first)
    bx1, by1, bx2, by2 = validate_xyxy(second)
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = (ax2 - ax1) * (ay2 - ay1)
    second_area = (bx2 - bx1) * (by2 - by1)
    union = first_area + second_area - intersection
    value = intersection / union
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"Invalid IoU value: {value!r}")
    return value


def assign_max_iou(matrix: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return one-to-one row/column pairs with maximum total IoU."""
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)) or np.any((matrix < 0) | (matrix > 1)):
        raise ValueError("IoU matrix must be two-dimensional with finite values in [0, 1].")
    if not matrix.size:
        return ()
    rows, columns = linear_sum_assignment(matrix, maximize=True)
    return tuple((int(row), int(column)) for row, column in zip(rows, columns))


@dataclass(frozen=True, slots=True)
class MatchRecord:
    frame_index: int
    class_name: str
    gt_track_id: int
    prediction_track_id: int
    iou: float
    gt_xyxy: Box
    prediction_xyxy: Box
    prediction_confidence: float


@dataclass(frozen=True, slots=True)
class FrameMatches:
    frame_index: int
    matches: tuple[MatchRecord, ...]
    unmatched_ground_truth: tuple[GroundTruthObservation, ...]
    unmatched_predictions: tuple[PredictionObservation, ...]


@dataclass(frozen=True, slots=True)
class MatchingSummary:
    frame_start: int
    frame_stop: int
    evaluation_frames: int
    total_gt_observations: int
    total_prediction_observations: int
    accepted_matches: int
    unmatched_gt_observations: int
    unmatched_prediction_observations: int
    matches_per_class: dict[str, int]
    mean_iou: float | None
    minimum_iou: float | None
    maximum_iou: float | None
    accounting_valid: bool
    validation_errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return self.accounting_valid and not self.validation_errors


def _check_frame_inputs(
    frame_index: int,
    ground_truth: Sequence[GroundTruthObservation],
    predictions: Sequence[PredictionObservation],
    allowed_classes: tuple[str, ...],
) -> None:
    for side, observations in (("ground truth", ground_truth), ("prediction", predictions)):
        seen: set[int] = set()
        for observation in observations:
            if observation.frame_index != frame_index:
                raise ValueError(f"{side} observation from frame {observation.frame_index} supplied to frame {frame_index}.")
            if observation.class_name not in allowed_classes:
                raise ValueError(f"Unsupported {side} class {observation.class_name!r} in frame {frame_index}.")
            if observation.track_id in seen:
                raise ValueError(f"Duplicate {side} track ID {observation.track_id} in frame {frame_index}.")
            seen.add(observation.track_id)
            validate_xyxy(observation.xyxy)
            if isinstance(observation, PredictionObservation) and not math.isfinite(observation.confidence):
                raise ValueError(f"Nonfinite prediction confidence in frame {frame_index}.")


def match_frame(
    frame_index: int,
    ground_truth: Sequence[GroundTruthObservation],
    predictions: Sequence[PredictionObservation],
    allowed_classes: tuple[str, ...],
    iou_threshold: float,
) -> FrameMatches:
    if not math.isfinite(iou_threshold) or not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be finite and within [0, 1].")
    _check_frame_inputs(frame_index, ground_truth, predictions, allowed_classes)
    matches: list[MatchRecord] = []
    matched_gt_ids: set[int] = set()
    matched_prediction_ids: set[int] = set()

    for class_name in allowed_classes:
        class_gt = [item for item in ground_truth if item.class_name == class_name]
        class_predictions = [item for item in predictions if item.class_name == class_name]
        matrix = np.asarray(
            [[xyxy_iou(gt.xyxy, prediction.xyxy) for prediction in class_predictions] for gt in class_gt],
            dtype=float,
        ).reshape(len(class_gt), len(class_predictions))
        for gt_index, prediction_index in assign_max_iou(matrix):
            iou = float(matrix[gt_index, prediction_index])
            if iou < iou_threshold:
                continue
            gt = class_gt[gt_index]
            prediction = class_predictions[prediction_index]
            matched_gt_ids.add(gt.track_id)
            matched_prediction_ids.add(prediction.track_id)
            matches.append(
                MatchRecord(
                    frame_index=frame_index,
                    class_name=class_name,
                    gt_track_id=gt.track_id,
                    prediction_track_id=prediction.track_id,
                    iou=iou,
                    gt_xyxy=gt.xyxy,
                    prediction_xyxy=prediction.xyxy,
                    prediction_confidence=prediction.confidence,
                )
            )

    return FrameMatches(
        frame_index=frame_index,
        matches=tuple(matches),
        unmatched_ground_truth=tuple(item for item in ground_truth if item.track_id not in matched_gt_ids),
        unmatched_predictions=tuple(item for item in predictions if item.track_id not in matched_prediction_ids),
    )


def validate_matches(
    frames: Sequence[FrameMatches],
    ground_truth: Sequence[GroundTruthObservation],
    predictions: Sequence[PredictionObservation],
    start_frame: int,
    stop_frame: int,
    allowed_classes: tuple[str, ...],
    iou_threshold: float,
) -> MatchingSummary:
    errors: list[str] = []
    expected_frames = tuple(range(start_frame, stop_frame + 1))
    actual_frames = tuple(frame.frame_index for frame in frames)
    if actual_frames != expected_frames:
        errors.append(
            f"Frame alignment error: expected every source frame {start_frame}-{stop_frame} in order; "
            f"received {len(frames)} frames with first={actual_frames[0] if actual_frames else None}, "
            f"last={actual_frames[-1] if actual_frames else None}."
        )
    gt_by_frame: dict[int, dict[int, GroundTruthObservation]] = defaultdict(dict)
    prediction_by_frame: dict[int, dict[int, PredictionObservation]] = defaultdict(dict)
    for observation in ground_truth:
        if observation.track_id in gt_by_frame[observation.frame_index]:
            errors.append(f"Duplicate input GT track ID {observation.track_id} in frame {observation.frame_index}.")
        gt_by_frame[observation.frame_index][observation.track_id] = observation
    for observation in predictions:
        if observation.track_id in prediction_by_frame[observation.frame_index]:
            errors.append(f"Duplicate input prediction track ID {observation.track_id} in frame {observation.frame_index}.")
        prediction_by_frame[observation.frame_index][observation.track_id] = observation

    ious: list[float] = []
    class_counts: Counter[str] = Counter()
    for frame in frames:
        gt_counts: Counter[int] = Counter()
        prediction_counts: Counter[int] = Counter()
        original_gt = gt_by_frame[frame.frame_index]
        original_predictions = prediction_by_frame[frame.frame_index]
        for match in frame.matches:
            gt_counts[match.gt_track_id] += 1
            prediction_counts[match.prediction_track_id] += 1
            gt = original_gt.get(match.gt_track_id)
            prediction = original_predictions.get(match.prediction_track_id)
            if match.frame_index != frame.frame_index or gt is None or prediction is None:
                errors.append(f"Cross-frame or unknown assignment in frame {frame.frame_index}.")
                continue
            if gt.class_name != prediction.class_name or match.class_name != gt.class_name:
                errors.append(f"Cross-class assignment in frame {frame.frame_index}: GT {gt.track_id}, prediction {prediction.track_id}.")
            if match.class_name not in allowed_classes:
                errors.append(f"Unsupported match class {match.class_name!r} in frame {frame.frame_index}.")
            if match.gt_xyxy != gt.xyxy or match.prediction_xyxy != prediction.xyxy or match.prediction_confidence != prediction.confidence:
                errors.append(f"Match payload differs from input observation in frame {frame.frame_index}.")
            if not math.isfinite(match.iou) or not 0.0 <= match.iou <= 1.0:
                errors.append(f"Invalid IoU in frame {frame.frame_index}: {match.iou!r}.")
            else:
                actual_iou = xyxy_iou(gt.xyxy, prediction.xyxy)
                if not math.isclose(match.iou, actual_iou, rel_tol=1e-12, abs_tol=1e-12):
                    errors.append(f"IoU differs from input boxes in frame {frame.frame_index}.")
                if match.iou < iou_threshold:
                    errors.append(f"Match below IoU threshold in frame {frame.frame_index}.")
                ious.append(match.iou)
                class_counts[match.class_name] += 1
        for observation in frame.unmatched_ground_truth:
            gt_counts[observation.track_id] += 1
            if observation.frame_index != frame.frame_index or original_gt.get(observation.track_id) != observation:
                errors.append(f"Invalid unmatched GT observation in frame {frame.frame_index}.")
        for observation in frame.unmatched_predictions:
            prediction_counts[observation.track_id] += 1
            if observation.frame_index != frame.frame_index or original_predictions.get(observation.track_id) != observation:
                errors.append(f"Invalid unmatched prediction in frame {frame.frame_index}.")
        if any(count > 1 for count in gt_counts.values()) or any(count > 1 for count in prediction_counts.values()):
            errors.append(f"Duplicate assignment in frame {frame.frame_index}.")
        if gt_counts != Counter(original_gt.keys()) or prediction_counts != Counter(original_predictions.keys()):
            errors.append(f"Observation accounting mismatch in frame {frame.frame_index}.")

    match_count = sum(len(frame.matches) for frame in frames)
    unmatched_gt_count = sum(len(frame.unmatched_ground_truth) for frame in frames)
    unmatched_prediction_count = sum(len(frame.unmatched_predictions) for frame in frames)
    accounting_valid = (
        match_count + unmatched_gt_count == len(ground_truth)
        and match_count + unmatched_prediction_count == len(predictions)
        and not any("accounting mismatch" in error or "Duplicate assignment" in error for error in errors)
    )
    if not accounting_valid:
        errors.append(
            "Global accounting mismatch: matched + unmatched observations do not reconstruct both inputs."
        )
    return MatchingSummary(
        frame_start=start_frame,
        frame_stop=stop_frame,
        evaluation_frames=len(frames),
        total_gt_observations=len(ground_truth),
        total_prediction_observations=len(predictions),
        accepted_matches=match_count,
        unmatched_gt_observations=unmatched_gt_count,
        unmatched_prediction_observations=unmatched_prediction_count,
        matches_per_class={class_name: class_counts[class_name] for class_name in allowed_classes},
        mean_iou=sum(ious) / len(ious) if ious else None,
        minimum_iou=min(ious) if ious else None,
        maximum_iou=max(ious) if ious else None,
        accounting_valid=accounting_valid,
        validation_errors=tuple(errors),
    )


def evaluate(
    ground_truth: Sequence[GroundTruthObservation],
    predictions: Sequence[PredictionObservation],
    start_frame: int,
    stop_frame: int,
    allowed_classes: tuple[str, ...],
    iou_threshold: float,
) -> tuple[tuple[FrameMatches, ...], MatchingSummary]:
    gt_by_frame: dict[int, list[GroundTruthObservation]] = defaultdict(list)
    prediction_by_frame: dict[int, list[PredictionObservation]] = defaultdict(list)
    for observation in ground_truth:
        if not start_frame <= observation.frame_index <= stop_frame:
            raise ValueError(f"GT observation outside evaluation range: frame {observation.frame_index}.")
        gt_by_frame[observation.frame_index].append(observation)
    for observation in predictions:
        if not start_frame <= observation.frame_index <= stop_frame:
            raise ValueError(f"Prediction outside evaluation range: frame {observation.frame_index}.")
        prediction_by_frame[observation.frame_index].append(observation)
    frames = tuple(
        match_frame(frame_index, gt_by_frame[frame_index], prediction_by_frame[frame_index], allowed_classes, iou_threshold)
        for frame_index in range(start_frame, stop_frame + 1)
    )
    return frames, validate_matches(frames, ground_truth, predictions, start_frame, stop_frame, allowed_classes, iou_threshold)


def load_predictions(
    path: Path,
    start_frame: int,
    stop_frame: int,
    allowed_classes: tuple[str, ...],
) -> tuple[PredictionObservation, ...]:
    with path.open(encoding="utf-8") as file:
        document = json.load(file, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Invalid JSON number: {value}")))
    if document.get("schema_version") != 1:
        raise ValueError("Expected prediction schema version 1.")
    metadata = document.get("evaluation_range", {})
    if (metadata.get("start_frame"), metadata.get("stop_frame"), metadata.get("expected_frames")) != (
        start_frame, stop_frame, stop_frame - start_frame + 1
    ):
        raise ValueError("Prediction metadata does not match the ground-truth source frame range.")
    expected_frames = list(range(start_frame, stop_frame + 1))
    if document.get("evaluation_frames") != expected_frames:
        raise ValueError("Prediction evaluation_frames must contain every source frame in order.")
    raw_predictions = document.get("predictions")
    if not isinstance(raw_predictions, list):
        raise ValueError("Prediction records must be a JSON list.")
    predictions: list[PredictionObservation] = []
    seen: set[tuple[int, int]] = set()
    for index, raw in enumerate(raw_predictions):
        if not isinstance(raw, dict):
            raise ValueError(f"Prediction {index} is not an object.")
        frame_index, track_id, class_id = (raw.get(field) for field in ("frame_index", "track_id", "class_id"))
        if any(type(value) is not int for value in (frame_index, track_id, class_id)):
            raise ValueError(f"Prediction {index} has a noninteger frame, track, or class ID.")
        if not start_frame <= frame_index <= stop_frame:
            raise ValueError(f"Prediction {index} is outside source range: frame {frame_index}.")
        class_name = raw.get("class_name")
        if class_name not in allowed_classes:
            raise ValueError(f"Prediction {index} has unsupported class {class_name!r}.")
        confidence = raw.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
            raise ValueError(f"Prediction {index} has invalid confidence.")
        box = raw.get("xyxy")
        if not isinstance(box, (list, tuple)):
            raise ValueError(f"Prediction {index} has no XYXY box.")
        try:
            xyxy = validate_xyxy(box)
        except ValueError as exc:
            raise ValueError(f"Prediction {index} has invalid box: {exc}") from exc
        key = (frame_index, track_id)
        if key in seen:
            raise ValueError(f"Duplicate prediction track ID {track_id} in source frame {frame_index}.")
        seen.add(key)
        predictions.append(PredictionObservation(frame_index, track_id, class_id, class_name, float(confidence), xyxy))
    return tuple(predictions)


def format_summary(summary: MatchingSummary, iou_threshold: float) -> str:
    def value(number: float | None) -> str:
        return f"{number:.6f}" if number is not None else "n/a"

    lines = [
        f"Evaluation source range: {summary.frame_start}-{summary.frame_stop} ({summary.evaluation_frames} frames)",
        f"IoU threshold: {iou_threshold:.2f}",
        "Assignment method: Hungarian, maximum total IoU per frame and class",
        f"Total GT observations: {summary.total_gt_observations}",
        f"Total prediction observations: {summary.total_prediction_observations}",
        f"Accepted matches: {summary.accepted_matches}",
        f"Unmatched GT observations: {summary.unmatched_gt_observations}",
        f"Unmatched prediction observations: {summary.unmatched_prediction_observations}",
        "Matches per class: " + ", ".join(f"{name}={count}" for name, count in summary.matches_per_class.items()),
        f"Mean accepted IoU: {value(summary.mean_iou)}",
        f"Minimum accepted IoU: {value(summary.minimum_iou)}",
        f"Maximum accepted IoU: {value(summary.maximum_iou)}",
        f"Accounting validation: {'PASS' if summary.accounting_valid else 'FAIL'}",
        f"Cross-frame/cross-class/duplicate and other validation errors: {list(summary.validation_errors)}",
        f"Validation: {'PASS' if summary.valid else 'FAIL'}",
    ]
    return "\n".join(lines)


def _write_json(path: Path, document: dict[str, object]) -> None:
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


def run(args: argparse.Namespace) -> MatchingSummary:
    ground_truth = import_cvat_zip(args.ground_truth)
    if not ground_truth.valid:
        raise ValueError("Ground-truth validation failed.")
    if (ground_truth.start_frame, ground_truth.stop_frame, ground_truth.task_frame_count) != (100, 219, 120):
        raise ValueError("Expected verified GT source frames 100-219 (120 frames).")
    if set(ground_truth.labels) != {"person", "car", "truck"}:
        raise ValueError(f"Unexpected GT labels: {ground_truth.labels}")
    predictions = load_predictions(args.predictions, ground_truth.start_frame, ground_truth.stop_frame, ground_truth.labels)
    frames, summary = evaluate(
        ground_truth.observations,
        predictions,
        ground_truth.start_frame,
        ground_truth.stop_frame,
        ground_truth.labels,
        args.iou_threshold,
    )
    print(format_summary(summary, args.iou_threshold))
    if not summary.valid:
        raise RuntimeError("Matching validation failed; no output file was written.")
    document: dict[str, object] = {
        "schema_version": 1,
        "configuration": {
            "frame_start": ground_truth.start_frame,
            "frame_stop": ground_truth.stop_frame,
            "iou_threshold": args.iou_threshold,
            "class_aware": True,
            "assignment": "hungarian",
            "coordinate_policy": "raw_xyxy_no_clipping",
        },
        "ground_truth_archive": str(args.ground_truth),
        "prediction_file": str(args.predictions),
        "frames": [asdict(frame) for frame in frames],
        "summary": asdict(summary),
    }
    _write_json(args.output, document)
    print(f"Output: {args.output}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, default=Path("data/ground_truth/traffic_tracker_ground_truth_v1.zip"))
    parser.add_argument("--predictions", type=Path, default=Path("outputs/tracker_evaluation/predictions_100_219.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tracker_evaluation/matches_100_219.json"))
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
