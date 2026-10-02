"""Evaluate the saved GT and ByteTrack records with official TrackEval metrics."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from trackeval.metrics import CLEAR, HOTA, Identity

from .evaluation_matching import load_predictions, validate_xyxy, xyxy_iou
from .evaluation_predictions import PredictionObservation
from .ground_truth import GroundTruthObservation, import_cvat_zip


CLASSES = ("person", "car", "truck")
START_FRAME = 100
STOP_FRAME = 219
TRACK_EVAL_SOURCE = "https://github.com/JonathonLuiten/TrackEval"
TRACK_EVAL_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"
SIMILARITY_THRESHOLD = 0.5
HOTA_FIELDS = ("HOTA", "DetA", "AssA", "LocA", "DetRe", "DetPr", "AssRe", "AssPr")
CLEAR_FIELDS = (
    "MOTA", "MOTP", "CLR_Re", "CLR_Pr", "IDSW", "Frag", "MT", "PT", "ML",
    "CLR_TP", "CLR_FN", "CLR_FP",
)
IDENTITY_FIELDS = ("IDF1", "IDP", "IDR", "IDTP", "IDFN", "IDFP")


@dataclass(frozen=True, slots=True)
class ClassMetricInput:
    class_name: str
    data: dict[str, object]
    source_frames: tuple[int, ...]
    gt_id_map: dict[int, int]
    tracker_id_map: dict[int, int]


def verify_trackeval_source() -> dict[str, str]:
    """Require the official installed distribution at the pinned Git commit."""
    distribution = importlib.metadata.distribution("trackeval")
    provenance_text = distribution.read_text("direct_url.json")
    if provenance_text is None:
        raise RuntimeError("TrackEval installation has no Git provenance; install the pinned requirements.txt dependency.")
    provenance = json.loads(provenance_text)
    url = str(provenance.get("url", "")).removesuffix(".git").rstrip("/").lower()
    commit = provenance.get("vcs_info", {}).get("commit_id")
    if url != TRACK_EVAL_SOURCE.lower() or commit != TRACK_EVAL_COMMIT:
        raise RuntimeError(f"TrackEval must come from official commit {TRACK_EVAL_COMMIT}; found {commit!r}.")
    return {"source": TRACK_EVAL_SOURCE, "commit": TRACK_EVAL_COMMIT, "package_version": distribution.version}


def build_class_input(
    ground_truth: Sequence[GroundTruthObservation],
    predictions: Sequence[PredictionObservation],
    class_name: str,
    start_frame: int,
    stop_frame: int,
) -> ClassMetricInput:
    """Adapt one class to TrackEval's preprocessed per-sequence metric input."""
    if class_name not in CLASSES or start_frame < 0 or stop_frame < start_frame:
        raise ValueError("Invalid evaluation class or source frame range.")
    source_frames = tuple(range(start_frame, stop_frame + 1))
    gt_by_frame: dict[int, list[GroundTruthObservation]] = defaultdict(list)
    tracker_by_frame: dict[int, list[PredictionObservation]] = defaultdict(list)
    for item in ground_truth:
        if item.class_name != class_name:
            continue
        if not start_frame <= item.frame_index <= stop_frame:
            raise ValueError(f"GT {class_name} observation outside source range: {item.frame_index}.")
        validate_xyxy(item.xyxy)
        gt_by_frame[item.frame_index].append(item)
    for item in predictions:
        if item.class_name != class_name:
            continue
        if not start_frame <= item.frame_index <= stop_frame:
            raise ValueError(f"Prediction {class_name} outside source range: {item.frame_index}.")
        validate_xyxy(item.xyxy)
        if not math.isfinite(item.confidence):
            raise ValueError(f"Nonfinite prediction confidence in frame {item.frame_index}.")
        tracker_by_frame[item.frame_index].append(item)

    gt_id_map = {identity: index for index, identity in enumerate(sorted({item.track_id for items in gt_by_frame.values() for item in items}))}
    tracker_id_map = {identity: index for index, identity in enumerate(sorted({item.track_id for items in tracker_by_frame.values() for item in items}))}
    gt_ids: list[np.ndarray] = []
    tracker_ids: list[np.ndarray] = []
    similarities: list[np.ndarray] = []
    for source_frame in source_frames:
        gt_items = gt_by_frame[source_frame]
        tracker_items = tracker_by_frame[source_frame]
        if len({item.track_id for item in gt_items}) != len(gt_items):
            raise ValueError(f"Duplicate GT {class_name} ID in source frame {source_frame}.")
        if len({item.track_id for item in tracker_items}) != len(tracker_items):
            raise ValueError(f"Duplicate predicted {class_name} ID in source frame {source_frame}.")
        gt_ids.append(np.asarray([gt_id_map[item.track_id] for item in gt_items], dtype=np.int64))
        tracker_ids.append(np.asarray([tracker_id_map[item.track_id] for item in tracker_items], dtype=np.int64))
        matrix = np.asarray(
            [[xyxy_iou(gt.xyxy, tracker.xyxy) for tracker in tracker_items] for gt in gt_items],
            dtype=np.float64,
        ).reshape(len(gt_items), len(tracker_items))
        if not np.all(np.isfinite(matrix)) or np.any((matrix < 0) | (matrix > 1)):
            raise ValueError(f"Invalid IoU matrix for {class_name} in source frame {source_frame}.")
        similarities.append(matrix)
    data: dict[str, object] = {
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": similarities,
        "num_timesteps": len(source_frames),
        "num_gt_ids": len(gt_id_map),
        "num_tracker_ids": len(tracker_id_map),
        "num_gt_dets": sum(len(items) for items in gt_ids),
        "num_tracker_dets": sum(len(items) for items in tracker_ids),
    }
    adapted = ClassMetricInput(class_name, data, source_frames, gt_id_map, tracker_id_map)
    validate_class_input(adapted)
    return adapted


def validate_class_input(adapted: ClassMetricInput) -> None:
    data = adapted.data
    frames = adapted.source_frames
    if frames != tuple(range(frames[0], frames[-1] + 1)) or data["num_timesteps"] != len(frames):
        raise ValueError(f"{adapted.class_name} has invalid timestep/source-frame alignment.")
    if len(data["gt_ids"]) != len(frames) or len(data["tracker_ids"]) != len(frames) or len(data["similarity_scores"]) != len(frames):
        raise ValueError(f"{adapted.class_name} has missing metric timesteps.")
    if set(adapted.gt_id_map.values()) != set(range(data["num_gt_ids"])):
        raise ValueError(f"{adapted.class_name} GT IDs are not compact.")
    if set(adapted.tracker_id_map.values()) != set(range(data["num_tracker_ids"])):
        raise ValueError(f"{adapted.class_name} tracker IDs are not compact.")
    for timestep, (gt_ids, tracker_ids, matrix) in enumerate(zip(data["gt_ids"], data["tracker_ids"], data["similarity_scores"])):
        if matrix.shape != (len(gt_ids), len(tracker_ids)):
            raise ValueError(f"{adapted.class_name} has wrong IoU matrix shape at timestep {timestep}.")
        if len(set(gt_ids.tolist())) != len(gt_ids) or len(set(tracker_ids.tolist())) != len(tracker_ids):
            raise ValueError(f"{adapted.class_name} has duplicate IDs at timestep {timestep}.")
        if not np.all(np.isfinite(matrix)) or np.any((matrix < 0) | (matrix > 1)):
            raise ValueError(f"{adapted.class_name} has invalid IoUs at timestep {timestep}.")
    if sum(len(items) for items in data["gt_ids"]) != data["num_gt_dets"]:
        raise ValueError(f"{adapted.class_name} GT detection count is inconsistent.")
    if sum(len(items) for items in data["tracker_ids"]) != data["num_tracker_dets"]:
        raise ValueError(f"{adapted.class_name} tracker detection count is inconsistent.")


def _metrics() -> dict[str, object]:
    return {
        "HOTA": HOTA(),
        "CLEAR": CLEAR({"THRESHOLD": SIMILARITY_THRESHOLD, "PRINT_CONFIG": False}),
        "Identity": Identity({"THRESHOLD": SIMILARITY_THRESHOLD, "PRINT_CONFIG": False}),
    }


@contextmanager
def _trackeval_numpy_compatibility():
    """Restore only the aliases used by this pinned upstream TrackEval commit."""
    added: list[str] = []
    try:
        for name, original in (("float", float), ("int", int)):
            if name not in np.__dict__:
                setattr(np, name, original)
                added.append(name)
        yield
    finally:
        for name in added:
            delattr(np, name)


def evaluate_class(adapted: ClassMetricInput, metrics: dict[str, object]) -> dict[str, dict[str, object]]:
    return {name: metric.eval_sequence(adapted.data) for name, metric in metrics.items()}


def _native(value: object) -> int | float:
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"TrackEval returned a nonfinite metric: {value!r}.")
    return number


def _validate_official_result(metric: object, result: dict[str, object]) -> None:
    for field in metric.fields:
        if field not in result:
            raise ValueError(f"TrackEval {metric.get_name()} omitted metric {field}.")
        values = np.asarray(result[field], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"TrackEval metric {field} contains a nonfinite value.")
        if field in metric.integer_fields or field in metric.integer_array_fields:
            if np.any(values < 0) or not np.allclose(values, np.rint(values), atol=1e-9):
                raise ValueError(f"TrackEval count {field} is invalid.")


def summarize_metric(metric: object, result: dict[str, object], fields: tuple[str, ...]) -> dict[str, int | float]:
    """Use TrackEval's summary-row convention: mean each HOTA alpha array."""
    summary: dict[str, int | float] = {}
    for field in fields:
        if field not in result:
            raise ValueError(f"TrackEval {metric.get_name()} omitted metric {field}.")
        value = result[field]
        if field in metric.float_array_fields:
            array = np.asarray(value)
            if array.shape != (len(metric.array_labels),) or not np.all(np.isfinite(array)):
                raise ValueError(f"TrackEval {field} alpha curve is invalid.")
            summary[field] = _native(np.mean(array))
        else:
            summary[field] = _native(value)
        if field in metric.integer_fields:
            if summary[field] < 0 or not math.isclose(summary[field], round(summary[field]), abs_tol=1e-9):
                raise ValueError(f"TrackEval count {field} is invalid.")
            summary[field] = int(round(summary[field]))
    return summary


def _present_metrics(metrics: dict[str, object], raw: dict[str, dict[str, object]]) -> dict[str, object]:
    for name, metric in metrics.items():
        _validate_official_result(metric, raw[name])
    hota = raw["HOTA"]
    hota_curves = {
        field: [_native(value) for value in hota[field]]
        for field in metrics["HOTA"].float_array_fields
    }
    return {
        "HOTA": summarize_metric(metrics["HOTA"], hota, HOTA_FIELDS),
        "CLEAR": summarize_metric(metrics["CLEAR"], raw["CLEAR"], CLEAR_FIELDS),
        "Identity": summarize_metric(metrics["Identity"], raw["Identity"], IDENTITY_FIELDS),
        "hota_alpha_thresholds": [_native(value) for value in metrics["HOTA"].array_labels],
        "hota_curves": hota_curves,
    }


def evaluate_all(
    ground_truth: Sequence[GroundTruthObservation],
    predictions: Sequence[PredictionObservation],
    start_frame: int,
    stop_frame: int,
) -> tuple[dict[str, object], dict[str, object], dict[str, ClassMetricInput]]:
    """Call official per-sequence and class-combination methods."""
    metrics = _metrics()
    inputs = {
        class_name: build_class_input(ground_truth, predictions, class_name, start_frame, stop_frame)
        for class_name in CLASSES
    }
    with _trackeval_numpy_compatibility():
        raw_by_class = {class_name: evaluate_class(adapted, metrics) for class_name, adapted in inputs.items()}
        per_class = {
            class_name: {
                "input_counts": {
                    "gt_observations": adapted.data["num_gt_dets"],
                    "prediction_observations": adapted.data["num_tracker_dets"],
                    "gt_ids": adapted.data["num_gt_ids"],
                    "tracker_ids": adapted.data["num_tracker_ids"],
                },
                **_present_metrics(metrics, raw_by_class[class_name]),
            }
            for class_name, adapted in inputs.items()
        }
        aggregate: dict[str, object] = {}
        for label, method in (
            ("class_averaged", "combine_classes_class_averaged"),
            ("detection_averaged", "combine_classes_det_averaged"),
        ):
            combined = {
                name: getattr(metric, method)({class_name: raw_by_class[class_name][name] for class_name in CLASSES})
                for name, metric in metrics.items()
            }
            aggregate[label] = _present_metrics(metrics, combined)
    return per_class, aggregate, inputs


def load_diagnostic_cross_check(path: Path, gt_count: int, prediction_count: int) -> dict[str, int]:
    with path.open(encoding="utf-8") as file:
        document = json.load(file)
    summary = document.get("summary", {})
    if (
        document.get("schema_version") != 1
        or summary.get("frame_start") != START_FRAME
        or summary.get("frame_stop") != STOP_FRAME
        or summary.get("evaluation_frames") != STOP_FRAME - START_FRAME + 1
        or summary.get("gt_visible_observations") != gt_count
        or summary.get("matched_observations", 0) + summary.get("unmatched_gt_observations", 0) != gt_count
        or summary.get("matched_observations", 0) + summary.get("unmatched_prediction_observations", 0) != prediction_count
        or summary.get("accounting_valid") is not True
        or summary.get("validation_errors") != []
    ):
        raise ValueError("Existing diagnostics artifact does not align with GT and predictions.")
    names = (
        "direct_identity_change_candidates", "association_gaps",
        "reacquisition_identity_change_candidates", "reacquisition_same_identity",
    )
    counts = {name: summary.get(name) for name in names}
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("Existing diagnostics event counts are invalid.")
    return counts


def format_report(per_class: dict[str, object], aggregate: dict[str, object], cross_check: dict[str, object]) -> str:
    lines = [f"Evaluation source frames: {START_FRAME}-{STOP_FRAME} ({STOP_FRAME - START_FRAME + 1} timesteps)"]
    for class_name in CLASSES:
        result = per_class[class_name]
        hota, clear, identity = result["HOTA"], result["CLEAR"], result["Identity"]
        lines.append(f"{class_name}:")
        lines.append("  HOTA: " + " ".join(f"{field}={hota[field]:.4f}" for field in HOTA_FIELDS))
        lines.append(
            "  CLEAR: " + " ".join(
                f"{field}={clear[field]}" if field in ("IDSW", "Frag", "MT", "PT", "ML", "CLR_TP", "CLR_FN", "CLR_FP")
                else f"{field}={clear[field]:.4f}"
                for field in CLEAR_FIELDS
            )
        )
        lines.append(
            "  Identity: " + " ".join(
                f"{field}={identity[field]}" if field in ("IDTP", "IDFN", "IDFP")
                else f"{field}={identity[field]:.4f}"
                for field in IDENTITY_FIELDS
            )
        )
    for label, result in aggregate.items():
        lines.append(
            f"TrackEval {label}: HOTA={result['HOTA']['HOTA']:.4f} "
            f"IDF1={result['Identity']['IDF1']:.4f} MOTA={result['CLEAR']['MOTA']:.4f} "
            f"IDSW={result['CLEAR']['IDSW']} Frag={result['CLEAR']['Frag']}"
        )
    lines.append(
        "Existing diagnostics: "
        + ", ".join(f"{name}={cross_check[name]}" for name in (
            "direct_identity_change_candidates", "association_gaps",
            "reacquisition_identity_change_candidates", "reacquisition_same_identity",
        ))
    )
    lines.append(
        f"Official TrackEval CLEAR: IDSW={cross_check['official_trackeval_idsw']}, "
        f"Frag={cross_check['official_trackeval_frag']}"
    )
    lines.append("Validation: PASS")
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


def run(args: argparse.Namespace) -> dict[str, object]:
    provenance = verify_trackeval_source()
    ground_truth = import_cvat_zip(args.ground_truth)
    if not ground_truth.valid or (
        ground_truth.start_frame, ground_truth.stop_frame, ground_truth.task_frame_count
    ) != (START_FRAME, STOP_FRAME, STOP_FRAME - START_FRAME + 1):
        raise ValueError("GT validation or source frame alignment failed.")
    if set(ground_truth.labels) != set(CLASSES):
        raise ValueError(f"Unexpected GT classes: {ground_truth.labels}.")
    predictions = load_predictions(args.predictions, START_FRAME, STOP_FRAME, CLASSES)
    per_class, aggregate, inputs = evaluate_all(ground_truth.observations, predictions, START_FRAME, STOP_FRAME)
    if set(per_class) != set(CLASSES) or any(adapted.data["num_timesteps"] != 120 for adapted in inputs.values()):
        raise ValueError("TrackEval result is missing a class or timestep.")
    if sum(result["input_counts"]["gt_observations"] for result in per_class.values()) != len(ground_truth.observations):
        raise ValueError("GT class counts do not reconstruct input.")
    if sum(result["input_counts"]["prediction_observations"] for result in per_class.values()) != len(predictions):
        raise ValueError("Prediction class counts do not reconstruct input.")
    cross_check = load_diagnostic_cross_check(args.diagnostics, len(ground_truth.observations), len(predictions))
    cross_check["official_trackeval_idsw"] = aggregate["detection_averaged"]["CLEAR"]["IDSW"]
    cross_check["official_trackeval_frag"] = aggregate["detection_averaged"]["CLEAR"]["Frag"]
    document: dict[str, object] = {
        "schema_version": 1,
        "trackeval": provenance,
        "configuration": {
            "frame_start": START_FRAME,
            "frame_stop": STOP_FRAME,
            "num_timesteps": STOP_FRAME - START_FRAME + 1,
            "timestep_to_source_frame": list(range(START_FRAME, STOP_FRAME + 1)),
            "classes": list(CLASSES),
            "similarity": "raw_xyxy_iou_no_clipping",
            "clear_identity_threshold": SIMILARITY_THRESHOLD,
            "hota_scalar_convention": "arithmetic mean of official alpha array; raw fraction",
        },
        "inputs": {
            "ground_truth": str(args.ground_truth),
            "predictions": str(args.predictions),
            "diagnostics": str(args.diagnostics),
        },
        "per_class": per_class,
        "aggregate": aggregate,
        "diagnostic_cross_check": cross_check,
        "validation": {"passed": True, "errors": []},
    }
    _write_json(args.output, document)
    with args.output.open(encoding="utf-8") as file:
        saved = json.load(file)
    if saved.get("schema_version") != 1 or set(saved.get("per_class", {})) != set(CLASSES):
        raise RuntimeError("TrackEval output JSON did not round-trip correctly.")
    print(format_report(per_class, aggregate, cross_check))
    print(f"Output: {args.output}")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, default=Path("data/ground_truth/traffic_tracker_ground_truth_v1.zip"))
    parser.add_argument("--predictions", type=Path, default=Path("outputs/tracker_evaluation/predictions_100_219.json"))
    parser.add_argument("--diagnostics", type=Path, default=Path("outputs/tracker_evaluation/diagnostics_100_219.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tracker_evaluation/trackeval_100_219.json"))
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
