"""Render a bounded, source-frame-aligned audit of saved tracker evaluation results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

from .evaluation_diagnostics import load_matching_artifact
from .evaluation_matching import FrameMatches, load_predictions
from .evaluation_predictions import PredictionObservation
from .ground_truth import GroundTruthObservation, import_cvat_zip


START_FRAME = 100
STOP_FRAME = 219
CLASSES = ("person", "car", "truck")
EVIDENCE_TYPES = (
    "unmatched_gt",
    "unmatched_prediction",
    "direct_identity_change_candidate",
    "reacquisition_identity_change_candidate",
    "association_gap",
)


@dataclass(frozen=True, slots=True)
class Overlay:
    status: str
    class_name: str
    xyxy: tuple[float, float, float, float]
    gt_track_id: int | None = None
    prediction_track_id: int | None = None
    confidence: float | None = None
    iou: float | None = None


def render_plan(frame: FrameMatches) -> tuple[Overlay, ...]:
    """Preserve existing match decisions while labeling every visible box."""
    overlays: list[Overlay] = []
    for match in frame.matches:
        overlays.append(Overlay("accepted_gt", match.class_name, match.gt_xyxy, gt_track_id=match.gt_track_id, iou=match.iou))
        overlays.append(
            Overlay(
                "accepted_prediction", match.class_name, match.prediction_xyxy,
                prediction_track_id=match.prediction_track_id, confidence=match.prediction_confidence, iou=match.iou,
            )
        )
    for gt in frame.unmatched_ground_truth:
        overlays.append(Overlay("unmatched_gt", gt.class_name, gt.xyxy, gt_track_id=gt.track_id))
    for prediction in frame.unmatched_predictions:
        overlays.append(
            Overlay(
                "unmatched_prediction", prediction.class_name, prediction.xyxy,
                prediction_track_id=prediction.track_id, confidence=prediction.confidence,
            )
        )
    return tuple(overlays)


def render_frame(source: np.ndarray, frame: FrameMatches) -> np.ndarray:
    image = source.copy()
    colors = {
        "accepted_gt": (40, 190, 40),
        "accepted_prediction": (225, 190, 35),
        "unmatched_gt": (35, 35, 225),
        "unmatched_prediction": (35, 145, 245),
    }
    legend = (
        f"Source frame {frame.frame_index}  |  green: matched GT  |  cyan: matched prediction  "
        "|  red: unmatched GT  |  orange: unmatched prediction"
    )
    cv2.rectangle(image, (0, 0), (image.shape[1], 46), (25, 25, 25), -1)
    cv2.putText(image, legend, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (255, 255, 255), 2, cv2.LINE_AA)
    for overlay in render_plan(frame):
        x1, y1, x2, y2 = (int(round(value)) for value in overlay.xyxy)
        color = colors[overlay.status]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3 if "gt" in overlay.status else 2)
        if overlay.gt_track_id is not None:
            label = f"GT {overlay.class_name} #{overlay.gt_track_id}"
        else:
            label = f"PRED {overlay.class_name} #{overlay.prediction_track_id} {overlay.confidence:.2f}"
        if overlay.iou is not None:
            label += f" IoU {overlay.iou:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.52
        text_size, baseline = cv2.getTextSize(label, font, font_scale, 1)
        text_x = min(max(2, x1), max(2, image.shape[1] - text_size[0] - 5))
        preferred_y = y1 + text_size[1] + 7 if overlay.status == "accepted_prediction" else y1 - 7
        text_y = min(max(60, preferred_y), image.shape[0] - baseline - 3)
        cv2.rectangle(
            image, (text_x - 2, text_y - text_size[1] - 3),
            (text_x + text_size[0] + 3, text_y + baseline + 2), (15, 15, 15), -1,
        )
        cv2.putText(image, label, (text_x, text_y), font, font_scale, color, 1, cv2.LINE_AA)
    return image


def crop_bounds(xyxy: Sequence[float], width: int, height: int) -> tuple[int, int, int, int] | None:
    """Clip only the image crop to decoded pixels; matching XYXY stays raw."""
    if width <= 0 or height <= 0 or len(xyxy) != 4 or not all(math.isfinite(value) for value in xyxy):
        raise ValueError("Invalid crop dimensions or coordinates.")
    x1, y1, x2, y2 = xyxy
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Crop box has nonpositive width or height.")
    bounds = (
        max(0, min(width, math.floor(x1))),
        max(0, min(height, math.floor(y1))),
        max(0, min(width, math.ceil(x2))),
        max(0, min(height, math.ceil(y2))),
    )
    return bounds if bounds[2] > bounds[0] and bounds[3] > bounds[1] else None


def iter_source_frames(video: Path, selected_frames: Sequence[int]) -> Iterator[tuple[int, np.ndarray]]:
    """Decode from source frame zero; never rely on imprecise compressed-video seeking."""
    wanted = set(selected_frames)
    if not wanted or any(index < 0 for index in wanted):
        raise ValueError("Source frame selection must contain nonnegative indexes.")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open source video: {video}")
    try:
        for frame_index in range(max(wanted) + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Source video ended before frame {frame_index}.")
            if frame_index in wanted:
                yield frame_index, frame
    finally:
        capture.release()


def _quantile_sample(items: Sequence[object], limit: int = 3) -> list[object]:
    if not items:
        return []
    ordered = sorted(items, key=lambda item: (item.frame_index, item.track_id))
    if len(ordered) <= limit:
        return ordered
    positions = {round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)}
    return [ordered[index] for index in sorted(positions)]


def select_unmatched_predictions(frames: Sequence[FrameMatches]) -> list[tuple[PredictionObservation, str]]:
    """Choose a small deterministic sample with person time/confidence coverage."""
    unmatched = [item for frame in frames for item in frame.unmatched_predictions]
    selected: dict[tuple[int, int, str], tuple[PredictionObservation, str]] = {}
    people = [item for item in unmatched if item.class_name == "person"]
    if people:
        confidences = sorted(item.confidence for item in people)
        cuts = (confidences[len(confidences) // 3], confidences[2 * len(confidences) // 3])
        bins: dict[tuple[int, int], list[PredictionObservation]] = defaultdict(list)
        for item in people:
            time_bin = min(2, 3 * (item.frame_index - START_FRAME) // (STOP_FRAME - START_FRAME + 1))
            confidence_bin = 0 if item.confidence < cuts[0] else 1 if item.confidence < cuts[1] else 2
            bins[(time_bin, confidence_bin)].append(item)
        for (time_bin, confidence_bin), cell in sorted(bins.items()):
            time_center = START_FRAME + (time_bin + 0.5) * (STOP_FRAME - START_FRAME + 1) / 3
            confidence_center = sorted(item.confidence for item in cell)[len(cell) // 2]
            chosen = min(
                cell,
                key=lambda item: (
                    abs(item.frame_index - time_center), abs(item.confidence - confidence_center),
                    item.frame_index, item.track_id,
                ),
            )
            selected[(chosen.frame_index, chosen.track_id, chosen.class_name)] = (
                chosen, f"person time bin {time_bin + 1}/3, confidence bin {confidence_bin + 1}/3",
            )
        per_frame = Counter(item.frame_index for item in people)
        peak_frame = min(frame for frame, count in per_frame.items() if count == max(per_frame.values()))
        peak = max((item for item in people if item.frame_index == peak_frame), key=lambda item: (item.confidence, -item.track_id))
        selected.setdefault((peak.frame_index, peak.track_id, peak.class_name), (peak, "peak unmatched-person frame"))
    for class_name in ("car", "truck"):
        for item in _quantile_sample([row for row in unmatched if row.class_name == class_name]):
            selected[(item.frame_index, item.track_id, item.class_name)] = (item, "first/middle/last by source frame")
    return [selected[key] for key in sorted(selected)]


def _frame_path(frame_index: int) -> str:
    return f"frames/source_{frame_index:04d}.jpg"


def _base_item(evidence_type: str, frame_index: int, class_name: str) -> dict[str, object]:
    return {
        "evidence_type": evidence_type,
        "source_frame": frame_index,
        "class_name": class_name,
        "gt_track_id": None,
        "prediction_track_id": None,
        "confidence": None,
        "iou": None,
        "image_path": _frame_path(frame_index),
        "crop_path": None,
        "context_frames": [],
        "context_image_paths": [],
        "selection_reason": "",
    }


def select_evidence(frames: Sequence[FrameMatches], diagnostics: dict[str, object]) -> list[dict[str, object]]:
    by_frame = {frame.frame_index: frame for frame in frames}
    if tuple(by_frame) != tuple(range(START_FRAME, STOP_FRAME + 1)):
        raise ValueError("Matching frames are not source-aligned 100-219.")
    events = diagnostics["events"]
    items: list[dict[str, object]] = []
    for class_name in CLASSES:
        gt_rows = [row for frame in frames for row in frame.unmatched_ground_truth if row.class_name == class_name]
        for row in _quantile_sample(gt_rows):
            item = _base_item("unmatched_gt", row.frame_index, class_name)
            item["gt_track_id"] = row.track_id
            item["selection_reason"] = "first/middle/last unmatched GT observation for class"
            items.append(item)
    for row, reason in select_unmatched_predictions(frames):
        item = _base_item("unmatched_prediction", row.frame_index, row.class_name)
        item["prediction_track_id"] = row.track_id
        item["confidence"] = row.confidence
        item["xyxy"] = list(row.xyxy)
        item["selection_reason"] = reason
        items.append(item)
    match_lookup = {(match.frame_index, match.gt_track_id): match for frame in frames for match in frame.matches}
    for event in events["direct_identity_change_candidates"]:
        frame_index = event["current_frame"]
        match = match_lookup.get((frame_index, event["gt_track_id"]))
        if match is None or match.prediction_track_id != event["current_prediction_id"]:
            raise ValueError(f"Direct identity event does not match frame {frame_index}.")
        item = _base_item("direct_identity_change_candidate", frame_index, event["class_name"])
        item.update(gt_track_id=event["gt_track_id"], prediction_track_id=match.prediction_track_id,
                    confidence=match.prediction_confidence, iou=match.iou,
                    context_frames=[event["previous_frame"]], selection_reason="all direct identity-change candidates",
                    event_details=event)
        items.append(item)
    for event in events["reacquisition_identity_change_candidates"]:
        frame_index = event["matched_frame_after_gap"]
        match = match_lookup.get((frame_index, event["gt_track_id"]))
        if match is None or match.prediction_track_id != event["prediction_id_after_gap"]:
            raise ValueError(f"Reacquisition event does not match frame {frame_index}.")
        gap = next(
            (
                gap for gap in events["association_gaps"]
                if gap["gt_track_id"] == event["gt_track_id"]
                and gap["start_frame"] == event["gap_start_frame"]
                and gap["end_frame"] == event["gap_end_frame"]
            ),
            None,
        )
        if gap is None or not gap["visible_frames"]:
            raise ValueError(f"Reacquisition event has no association gap before frame {frame_index}.")
        midpoint = gap["visible_frames"][len(gap["visible_frames"]) // 2]
        item = _base_item("reacquisition_identity_change_candidate", frame_index, event["class_name"])
        item.update(gt_track_id=event["gt_track_id"], prediction_track_id=match.prediction_track_id,
                    confidence=match.prediction_confidence, iou=match.iou,
                    context_frames=[event["matched_frame_before_gap"], midpoint],
                    selection_reason="all different-ID reacquisitions", event_details=event)
        items.append(item)
    longest_gaps = sorted(
        events["association_gaps"], key=lambda gap: (-gap["length"], gap["gt_track_id"], gap["start_frame"])
    )[:3]
    for gap in longest_gaps:
        visible = gap["visible_frames"]
        middle = visible[len(visible) // 2]
        before = next(
            (frame for frame in range(gap["start_frame"] - 1, START_FRAME - 1, -1)
             if (frame, gap["gt_track_id"]) in match_lookup), None
        ) if gap["prediction_id_before_gap"] is not None else None
        after = next(
            (frame for frame in range(gap["end_frame"] + 1, STOP_FRAME + 1)
             if (frame, gap["gt_track_id"]) in match_lookup), None
        ) if gap["prediction_id_after_gap"] is not None else None
        context = [frame for frame in (before, visible[0], visible[-1], after) if frame is not None and frame != middle]
        item = _base_item("association_gap", middle, gap["class_name"])
        item.update(gt_track_id=gap["gt_track_id"], context_frames=list(dict.fromkeys(context)),
                    selection_reason="three longest association gaps by visible-frame length",
                    event_details=gap)
        items.append(item)
    for item in items:
        item["context_image_paths"] = [_frame_path(frame) for frame in item["context_frames"]]
    return items


def _source_records_match(
    frames: Sequence[FrameMatches],
    ground_truth: Sequence[GroundTruthObservation],
    predictions: Sequence[PredictionObservation],
) -> None:
    gt_lookup = {(item.frame_index, item.track_id): item for item in ground_truth}
    prediction_lookup = {(item.frame_index, item.track_id): item for item in predictions}
    seen_gt: set[tuple[int, int]] = set()
    seen_prediction: set[tuple[int, int]] = set()
    for frame in frames:
        for match in frame.matches:
            gt_key = (frame.frame_index, match.gt_track_id)
            prediction_key = (frame.frame_index, match.prediction_track_id)
            gt = gt_lookup.get(gt_key)
            prediction = prediction_lookup.get(prediction_key)
            if (
                gt is None or prediction is None or gt.class_name != match.class_name
                or prediction.class_name != match.class_name or gt.xyxy != match.gt_xyxy
                or prediction.xyxy != match.prediction_xyxy
                or not math.isclose(prediction.confidence, match.prediction_confidence, abs_tol=1e-12)
            ):
                raise ValueError(f"Saved match disagrees with GT or prediction source frame {frame.frame_index}.")
            if gt_key in seen_gt or prediction_key in seen_prediction:
                raise ValueError("Duplicate assignment in saved matching artifact.")
            seen_gt.add(gt_key)
            seen_prediction.add(prediction_key)
        for item in frame.unmatched_ground_truth:
            key = (item.frame_index, item.track_id)
            if gt_lookup.get(key) != item or key in seen_gt:
                raise ValueError(f"Unmatched GT disagrees with source frame {frame.frame_index}.")
            seen_gt.add(key)
        for item in frame.unmatched_predictions:
            key = (item.frame_index, item.track_id)
            if prediction_lookup.get(key) != item or key in seen_prediction:
                raise ValueError(f"Unmatched prediction disagrees with source frame {frame.frame_index}.")
            seen_prediction.add(key)
    if seen_gt != set(gt_lookup) or seen_prediction != set(prediction_lookup):
        raise ValueError("Saved matching artifact does not reconstruct the GT and prediction inputs.")


def _validate_diagnostic_events(frames: Sequence[FrameMatches], events: dict[str, object]) -> None:
    matches = {(frame.frame_index, match.gt_track_id): match for frame in frames for match in frame.matches}
    unmatched = {
        (frame.frame_index, item.track_id): item
        for frame in frames for item in frame.unmatched_ground_truth
    }
    visible: dict[int, list[int]] = defaultdict(list)
    for frame, track_id in set(matches) | set(unmatched):
        visible[track_id].append(frame)
    for track_id in visible:
        visible[track_id].sort()
    for event in events["direct_identity_change_candidates"]:
        track_id = event["gt_track_id"]
        previous, current = event["previous_frame"], event["current_frame"]
        frames_for_track = visible.get(track_id, [])
        if (
            previous not in frames_for_track or current not in frames_for_track
            or frames_for_track.index(current) != frames_for_track.index(previous) + 1
            or (previous, track_id) not in matches or (current, track_id) not in matches
            or matches[(previous, track_id)].prediction_track_id != event["previous_prediction_id"]
            or matches[(current, track_id)].prediction_track_id != event["current_prediction_id"]
            or matches[(previous, track_id)].class_name != event["class_name"]
            or matches[(current, track_id)].class_name != event["class_name"]
            or event["previous_prediction_id"] == event["current_prediction_id"]
        ):
            raise ValueError("Direct identity event is inconsistent with visible GT matches.")
    gaps = {}
    for gap in events["association_gaps"]:
        track_id = gap["gt_track_id"]
        gap_frames = gap["visible_frames"]
        timeline = visible.get(track_id, [])
        if (
            not gap_frames or len(gap_frames) != gap["length"]
            or gap_frames[0] != gap["start_frame"] or gap_frames[-1] != gap["end_frame"]
            or any((frame, track_id) not in unmatched for frame in gap_frames)
            or any(unmatched[(frame, track_id)].class_name != gap["class_name"] for frame in gap_frames)
            or gap_frames != timeline[timeline.index(gap_frames[0]):timeline.index(gap_frames[0]) + len(gap_frames)]
        ):
            raise ValueError("Association gap is inconsistent with unmatched visible GT frames.")
        start_position = timeline.index(gap_frames[0])
        end_position = start_position + len(gap_frames)
        before = matches.get((timeline[start_position - 1], track_id)) if start_position else None
        after = matches.get((timeline[end_position], track_id)) if end_position < len(timeline) else None
        if (
            (start_position > 0 and before is None)
            or (end_position < len(timeline) and after is None)
            or (before.prediction_track_id if before else None) != gap["prediction_id_before_gap"]
            or (after.prediction_track_id if after else None) != gap["prediction_id_after_gap"]
        ):
            raise ValueError("Association gap boundary identities are inconsistent.")
        key = (track_id, gap["start_frame"], gap["end_frame"])
        if key in gaps:
            raise ValueError("Duplicate association gap event.")
        gaps[key] = gap
    for event in events["reacquisition_identity_change_candidates"]:
        track_id = event["gt_track_id"]
        gap = gaps.get((track_id, event["gap_start_frame"], event["gap_end_frame"]))
        if gap is None:
            raise ValueError("Reacquisition event has no corresponding association gap.")
        timeline = visible[track_id]
        start_position = timeline.index(gap["start_frame"])
        end_position = timeline.index(gap["end_frame"])
        before_frame = timeline[start_position - 1] if start_position else None
        after_frame = timeline[end_position + 1] if end_position + 1 < len(timeline) else None
        before = matches.get((event["matched_frame_before_gap"], track_id))
        after = matches.get((event["matched_frame_after_gap"], track_id))
        if (
            before is None or after is None
            or before.prediction_track_id != event["prediction_id_before_gap"]
            or after.prediction_track_id != event["prediction_id_after_gap"]
            or before.class_name != event["class_name"] or after.class_name != event["class_name"]
            or before.prediction_track_id == after.prediction_track_id
            or gap["prediction_id_before_gap"] != before.prediction_track_id
            or gap["prediction_id_after_gap"] != after.prediction_track_id
            or before_frame != event["matched_frame_before_gap"]
            or after_frame != event["matched_frame_after_gap"]
        ):
            raise ValueError("Reacquisition event is inconsistent with its association gap.")


def _load_audit_inputs(args: argparse.Namespace) -> tuple[tuple[FrameMatches, ...], dict[str, object]]:
    ground_truth = import_cvat_zip(args.ground_truth)
    if not ground_truth.valid or (ground_truth.start_frame, ground_truth.stop_frame, ground_truth.task_frame_count) != (100, 219, 120):
        raise ValueError("GT archive is invalid or not source frames 100-219.")
    if set(ground_truth.labels) != set(CLASSES):
        raise ValueError("GT class labels are not person/car/truck.")
    predictions = load_predictions(args.predictions, START_FRAME, STOP_FRAME, CLASSES)
    frames, threshold = load_matching_artifact(args.matches)
    if threshold != 0.5:
        raise ValueError("Expected the verified 0.5 matching IoU threshold.")
    _source_records_match(frames, ground_truth.observations, predictions)
    with args.diagnostics.open(encoding="utf-8") as file:
        diagnostics = json.load(file)
    summary = diagnostics.get("summary", {})
    events = diagnostics.get("events", {})
    if (
        diagnostics.get("schema_version") != 1 or summary.get("frame_start") != START_FRAME
        or summary.get("frame_stop") != STOP_FRAME or summary.get("evaluation_frames") != 120
        or summary.get("gt_visible_observations") != len(ground_truth.observations)
        or summary.get("matched_observations") != sum(len(frame.matches) for frame in frames)
        or summary.get("unmatched_prediction_observations") != sum(len(frame.unmatched_predictions) for frame in frames)
        or summary.get("accounting_valid") is not True or summary.get("validation_errors") != []
    ):
        raise ValueError("Diagnostics artifact does not align with saved matching records.")
    for name in ("direct_identity_change_candidates", "reacquisition_identity_change_candidates", "association_gaps"):
        if not isinstance(events.get(name), list) or len(events[name]) != summary.get(name):
            raise ValueError(f"Diagnostics {name} count is inconsistent.")
    _validate_diagnostic_events(frames, events)
    with args.trackeval.open(encoding="utf-8") as file:
        formal = json.load(file)
    configuration = formal.get("configuration", {})
    if (
        formal.get("validation", {}).get("passed") is not True
        or configuration.get("timestep_to_source_frame") != list(range(START_FRAME, STOP_FRAME + 1))
        or set(formal.get("per_class", {})) != set(CLASSES)
        or formal["per_class"]["person"]["CLEAR"].get("CLR_FP") !=
        sum(item.class_name == "person" for frame in frames for item in frame.unmatched_predictions)
    ):
        raise ValueError("TrackEval artifact does not align with the verified 120-frame matching result.")
    for class_name in CLASSES:
        counts = formal["per_class"][class_name].get("input_counts", {})
        if (
            counts.get("gt_observations") != sum(item.class_name == class_name for item in ground_truth.observations)
            or counts.get("prediction_observations") != sum(item.class_name == class_name for item in predictions)
        ):
            raise ValueError(f"TrackEval input counts do not align for {class_name}.")
    return frames, diagnostics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(path: Path, document: dict[str, object]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".manifest-", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            json.dump(document, file, indent=2, allow_nan=False)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_manifest(document: dict[str, object], output_dir: Path) -> None:
    items = document.get("items")
    summary = document.get("summary", {})
    if document.get("schema_version") != 1 or not isinstance(items, list):
        raise ValueError("Evidence manifest schema is invalid.")
    seen: set[str] = set()
    counts = Counter()
    referenced_frames: set[int] = set()
    for item in items:
        if not isinstance(item, dict) or item.get("evidence_type") not in EVIDENCE_TYPES:
            raise ValueError("Evidence item has an invalid type.")
        frame = item.get("source_frame")
        if type(frame) is not int or not START_FRAME <= frame <= STOP_FRAME or item.get("class_name") not in CLASSES:
            raise ValueError("Evidence item has invalid source frame or class.")
        identity = json.dumps(
            (item["evidence_type"], frame, item.get("class_name"), item.get("gt_track_id"),
             item.get("prediction_track_id"), item.get("event_details")),
            sort_keys=True,
        )
        if identity in seen:
            raise ValueError("Duplicate evidence item.")
        seen.add(identity)
        counts[item["evidence_type"]] += 1
        referenced_frames.add(frame)
        for context in item.get("context_frames", []):
            if type(context) is not int or not START_FRAME <= context <= STOP_FRAME:
                raise ValueError("Evidence context frame is out of range.")
            referenced_frames.add(context)
        expected_paths = [_frame_path(index) for index in item.get("context_frames", [])]
        if item.get("context_image_paths") != expected_paths:
            raise ValueError("Evidence context image paths do not match source frame indexes.")
        if item.get("image_path") != _frame_path(frame):
            raise ValueError("Evidence image path does not match source frame index.")
        for relative in [item["image_path"], *expected_paths, item.get("crop_path")]:
            if relative is None:
                continue
            path = (output_dir / relative).resolve()
            if not path.is_relative_to(output_dir.resolve()) or not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Evidence image is missing or outside output directory: {relative}.")
        if item["evidence_type"] == "unmatched_prediction":
            if item.get("prediction_track_id") is None or not math.isfinite(item.get("confidence", math.nan)):
                raise ValueError("Unmatched prediction evidence lacks ID or confidence.")
    if summary.get("evidence_frames_generated") != len(referenced_frames):
        raise ValueError("Evidence frame count does not reconstruct selected source frames.")
    if summary.get("counts_by_type") != {name: counts[name] for name in EVIDENCE_TYPES}:
        raise ValueError("Evidence type counts do not reconstruct manifest items.")
    if summary.get("person_unmatched_prediction_samples") != sum(
        item["evidence_type"] == "unmatched_prediction" and item["class_name"] == "person" for item in items
    ):
        raise ValueError("Person sample count does not reconstruct manifest items.")


def run(args: argparse.Namespace) -> dict[str, object]:
    frames, diagnostics = _load_audit_inputs(args)
    items = select_evidence(frames, diagnostics)
    selected_frames = sorted({
        frame for item in items for frame in [item["source_frame"], *item["context_frames"]]
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "frames").mkdir(exist_ok=True)
    (args.output_dir / "crops").mkdir(exist_ok=True)
    by_frame = {frame.frame_index: frame for frame in frames}
    crop_items: dict[int, list[dict[str, object]]] = defaultdict(list)
    for item in items:
        if item["evidence_type"] == "unmatched_prediction":
            crop_items[item["source_frame"]].append(item)
    written_frames = 0
    for frame_index, source in iter_source_frames(args.video, selected_frames):
        image = render_frame(source, by_frame[frame_index])
        if not cv2.imwrite(str(args.output_dir / _frame_path(frame_index)), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise OSError(f"Could not write evidence frame {frame_index}.")
        written_frames += 1
        for item in crop_items[frame_index]:
            bounds = crop_bounds(item["xyxy"], source.shape[1], source.shape[0])
            if bounds is None:
                item["crop_note"] = "prediction box has no intersection with source image"
                continue
            x1, y1, x2, y2 = bounds
            crop = source[y1:y2, x1:x2]
            relative = f"crops/source_{frame_index:04d}_{item['class_name']}_prediction_{item['prediction_track_id']:04d}.jpg"
            if not cv2.imwrite(str(args.output_dir / relative), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"Could not write prediction crop at source frame {frame_index}.")
            item["crop_path"] = relative
            item["crop_bounds_xyxy"] = [x1, y1, x2, y2]
    if written_frames != len(selected_frames):
        raise RuntimeError("Source video did not yield every selected evidence frame.")
    counts = Counter(item["evidence_type"] for item in items)
    document: dict[str, object] = {
        "schema_version": 1,
        "configuration": {
            "frame_start": START_FRAME,
            "frame_stop": STOP_FRAME,
            "frame_numbering": "original source video",
            "rendering": {
                "accepted_gt": "green", "accepted_prediction": "cyan",
                "unmatched_gt": "red", "unmatched_prediction": "orange",
            },
            "selection_policy": {
                "unmatched_gt": "first, middle, last observation per class in source-frame order",
                "unmatched_person_predictions": "one nearest cell center in each populated 3x3 source-time/confidence-tercile bin; also highest-confidence observation in earliest peak-count frame",
                "other_unmatched_predictions": "first, middle, last observation per class in source-frame order",
                "direct_changes": "all events, current frame plus previous matched frame",
                "different_id_reacquisitions": "all events, after-gap frame plus before-gap and gap-midpoint context",
                "association_gaps": "three longest gaps; before/start/middle/end/after context where available",
            },
            "crop_policy": "clip only crop pixel bounds to image; preserve raw prediction XYXY in manifest and matching",
        },
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in (
                ("video", args.video), ("ground_truth", args.ground_truth), ("predictions", args.predictions),
                ("matches", args.matches), ("diagnostics", args.diagnostics), ("trackeval", args.trackeval),
            )
        },
        "items": items,
        "summary": {
            "evidence_frames_generated": written_frames,
            "counts_by_type": {name: counts[name] for name in EVIDENCE_TYPES},
            "person_unmatched_prediction_samples": sum(
                item["evidence_type"] == "unmatched_prediction" and item["class_name"] == "person" for item in items
            ),
            "crops_generated": sum(item.get("crop_path") is not None for item in items),
            "manifest_validation": "PASS",
        },
    }
    validate_manifest(document, args.output_dir)
    manifest = args.output_dir / "evidence_manifest.json"
    _write_manifest(manifest, document)
    with manifest.open(encoding="utf-8") as file:
        saved = json.load(file)
    validate_manifest(saved, args.output_dir)
    summary = document["summary"]
    print(f"Evidence frames generated: {summary['evidence_frames_generated']}")
    for name in EVIDENCE_TYPES:
        print(f"{name}: {summary['counts_by_type'][name]}")
    print(f"Person unmatched-prediction samples: {summary['person_unmatched_prediction_samples']}")
    print(f"Crops generated: {summary['crops_generated']}")
    print("Manifest validation: PASS")
    print(f"Manifest: {manifest}")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=Path("data/input/test_traffic.mp4"))
    parser.add_argument("--ground-truth", type=Path, default=Path("data/ground_truth/traffic_tracker_ground_truth_v1.zip"))
    parser.add_argument("--predictions", type=Path, default=Path("outputs/tracker_evaluation/predictions_100_219.json"))
    parser.add_argument("--matches", type=Path, default=Path("outputs/tracker_evaluation/matches_100_219.json"))
    parser.add_argument("--diagnostics", type=Path, default=Path("outputs/tracker_evaluation/diagnostics_100_219.json"))
    parser.add_argument("--trackeval", type=Path, default=Path("outputs/tracker_evaluation/trackeval_100_219.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tracker_evaluation/evidence"))
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
