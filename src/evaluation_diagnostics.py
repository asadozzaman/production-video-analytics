"""Describe per-GT-track associations from an existing validated match artifact."""

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

from .evaluation_matching import FrameMatches, MatchRecord, validate_matches, validate_xyxy
from .evaluation_predictions import PredictionObservation
from .ground_truth import GroundTruthObservation


CLASSES = ("person", "car", "truck")
CLASS_IDS = {"person": 0, "car": 2, "truck": 7}
START_FRAME = 100
STOP_FRAME = 219


@dataclass(frozen=True, slots=True)
class TimelineEntry:
    frame_index: int
    matched_prediction_id: int | None
    matched_iou: float | None


@dataclass(frozen=True, slots=True)
class TrackDiagnostics:
    gt_track_id: int
    class_name: str
    visible_start_frame: int
    visible_end_frame: int
    visible_gt_frames: int
    matched_frames: int
    unmatched_frames: int
    match_ratio: float
    prediction_ids_seen: tuple[int, ...]
    distinct_prediction_ids: int
    first_matched_frame: int | None
    last_matched_frame: int | None
    longest_consecutive_matched_run: int
    longest_consecutive_unmatched_run: int
    mean_matched_iou: float | None
    minimum_matched_iou: float | None
    maximum_matched_iou: float | None
    timeline: tuple[TimelineEntry, ...]


@dataclass(frozen=True, slots=True)
class DirectIdentityChangeCandidate:
    gt_track_id: int
    class_name: str
    previous_frame: int
    current_frame: int
    previous_prediction_id: int
    current_prediction_id: int


@dataclass(frozen=True, slots=True)
class AssociationGap:
    gt_track_id: int
    class_name: str
    start_frame: int
    end_frame: int
    length: int
    visible_frames: tuple[int, ...]
    prediction_id_before_gap: int | None
    prediction_id_after_gap: int | None


@dataclass(frozen=True, slots=True)
class Reacquisition:
    gt_track_id: int
    class_name: str
    gap_start_frame: int
    gap_end_frame: int
    matched_frame_before_gap: int
    matched_frame_after_gap: int
    prediction_id_before_gap: int
    prediction_id_after_gap: int


@dataclass(frozen=True, slots=True)
class DiagnosticEvents:
    direct_identity_change_candidates: tuple[DirectIdentityChangeCandidate, ...]
    association_gaps: tuple[AssociationGap, ...]
    reacquisition_identity_change_candidates: tuple[Reacquisition, ...]
    reacquisition_same_identity: tuple[Reacquisition, ...]


@dataclass(frozen=True, slots=True)
class UnmatchedPredictionSummary:
    total_observations: int
    observations_per_class: dict[str, int]
    frames_with_unmatched_predictions: int
    frame_indices: tuple[int, ...]
    maximum_in_one_frame: int


@dataclass(frozen=True, slots=True)
class DiagnosticsSummary:
    frame_start: int
    frame_stop: int
    evaluation_frames: int
    gt_tracks_analyzed: int
    gt_visible_observations: int
    matched_observations: int
    unmatched_gt_observations: int
    unmatched_prediction_observations: int
    tracks_with_zero_identity_change_candidates: int
    tracks_using_multiple_prediction_ids: int
    direct_identity_change_candidates: int
    association_gaps: int
    reacquisition_identity_change_candidates: int
    reacquisition_same_identity: int
    longest_association_gap: int
    per_class: dict[str, dict[str, int]]
    accounting_valid: bool
    validation_errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return self.accounting_valid and not self.validation_errors


@dataclass(frozen=True, slots=True)
class DiagnosticsResult:
    tracks: tuple[TrackDiagnostics, ...]
    events: DiagnosticEvents
    unmatched_predictions: UnmatchedPredictionSummary
    summary: DiagnosticsSummary


def _integer(value: object, description: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{description} must be an integer.")
    return value


def _number(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{description} must be a finite number.")
    return float(value)


def _class(value: object, description: str) -> str:
    if type(value) is not str or value not in CLASSES:
        raise ValueError(f"{description} must be one of {CLASSES}.")
    return value


def _box(value: object, description: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list):
        raise ValueError(f"{description} must be an XYXY array.")
    try:
        return validate_xyxy(value)
    except ValueError as exc:
        raise ValueError(f"{description}: {exc}") from exc


def _records(value: object, description: str) -> list[dict[str, object]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{description} must be an array of objects.")
    return value


def _input_observations(frames: Sequence[FrameMatches]) -> tuple[tuple[GroundTruthObservation, ...], tuple[PredictionObservation, ...]]:
    """Reconstruct the observations carried by matches and unmatched lists."""
    ground_truth: list[GroundTruthObservation] = []
    predictions: list[PredictionObservation] = []
    for frame in frames:
        ground_truth.extend(frame.unmatched_ground_truth)
        predictions.extend(frame.unmatched_predictions)
        for match in frame.matches:
            ground_truth.append(
                GroundTruthObservation(match.frame_index, match.gt_track_id, match.class_name, match.gt_xyxy, False)
            )
            predictions.append(
                PredictionObservation(
                    match.frame_index,
                    match.prediction_track_id,
                    CLASS_IDS[match.class_name],
                    match.class_name,
                    match.prediction_confidence,
                    match.prediction_xyxy,
                )
            )
    return tuple(ground_truth), tuple(predictions)


def _validate_input(frames: Sequence[FrameMatches], start_frame: int, stop_frame: int, threshold: float) -> None:
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Matching IoU threshold must be within [0, 1].")
    if tuple(frame.frame_index for frame in frames) != tuple(range(start_frame, stop_frame + 1)):
        raise ValueError(f"Matching artifact must contain every source frame {start_frame}-{stop_frame} in order.")
    gt_classes: dict[int, str] = {}
    for frame in frames:
        for match in frame.matches:
            if match.class_name not in CLASSES:
                raise ValueError(f"Unsupported match class in frame {frame.frame_index}.")
            if match.frame_index != frame.frame_index:
                raise ValueError(f"Cross-frame match in frame {frame.frame_index}.")
            if not math.isfinite(match.iou) or not threshold <= match.iou <= 1:
                raise ValueError(f"Invalid or below-threshold IoU in frame {frame.frame_index}.")
            validate_xyxy(match.gt_xyxy)
            validate_xyxy(match.prediction_xyxy)
            if not math.isfinite(match.prediction_confidence):
                raise ValueError(f"Nonfinite prediction confidence in frame {frame.frame_index}.")
        for item in frame.unmatched_ground_truth:
            if item.class_name not in CLASSES:
                raise ValueError(f"Unsupported GT class in frame {frame.frame_index}.")
            validate_xyxy(item.xyxy)
        for item in frame.unmatched_predictions:
            if item.class_name not in CLASSES:
                raise ValueError(f"Unsupported prediction class in frame {frame.frame_index}.")
            validate_xyxy(item.xyxy)
            if not math.isfinite(item.confidence):
                raise ValueError(f"Nonfinite prediction confidence in frame {frame.frame_index}.")
        for track_id, class_name in (
            [(match.gt_track_id, match.class_name) for match in frame.matches]
            + [(item.track_id, item.class_name) for item in frame.unmatched_ground_truth]
        ):
            if track_id in gt_classes and gt_classes[track_id] != class_name:
                raise ValueError(f"GT track {track_id} changes class across its visible timeline.")
            gt_classes[track_id] = class_name
    ground_truth, predictions = _input_observations(frames)
    validation = validate_matches(frames, ground_truth, predictions, start_frame, stop_frame, CLASSES, threshold)
    if not validation.valid:
        raise ValueError("Malformed matching artifact: " + "; ".join(validation.validation_errors))


def load_matching_artifact(path: Path) -> tuple[tuple[FrameMatches, ...], float]:
    """Read and validate the saved matching result, without matching again."""
    with path.open(encoding="utf-8") as file:
        document = json.load(file, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Invalid JSON number: {value}")))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("Expected matching artifact schema version 1.")
    configuration = document.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("Matching configuration is missing.")
    if (
        _integer(configuration.get("frame_start"), "frame_start"),
        _integer(configuration.get("frame_stop"), "frame_stop"),
    ) != (START_FRAME, STOP_FRAME):
        raise ValueError("Matching artifact must use source frames 100-219.")
    if configuration.get("class_aware") is not True or configuration.get("assignment") != "hungarian":
        raise ValueError("Expected class-aware Hungarian matching configuration.")
    threshold = _number(configuration.get("iou_threshold"), "iou_threshold")
    raw_frames = _records(document.get("frames"), "frames")
    frames: list[FrameMatches] = []
    for raw_frame in raw_frames:
        frame_index = _integer(raw_frame.get("frame_index"), "frame_index")
        matches: list[MatchRecord] = []
        for row in _records(raw_frame.get("matches"), f"frame {frame_index} matches"):
            matches.append(
                MatchRecord(
                    frame_index=_integer(row.get("frame_index"), "match frame_index"),
                    class_name=_class(row.get("class_name"), "match class_name"),
                    gt_track_id=_integer(row.get("gt_track_id"), "gt_track_id"),
                    prediction_track_id=_integer(row.get("prediction_track_id"), "prediction_track_id"),
                    iou=_number(row.get("iou"), "match IoU"),
                    gt_xyxy=_box(row.get("gt_xyxy"), "matched GT box"),
                    prediction_xyxy=_box(row.get("prediction_xyxy"), "matched prediction box"),
                    prediction_confidence=_number(row.get("prediction_confidence"), "prediction confidence"),
                )
            )
        unmatched_gt: list[GroundTruthObservation] = []
        for row in _records(raw_frame.get("unmatched_ground_truth"), f"frame {frame_index} unmatched GT"):
            occluded = row.get("occluded")
            if type(occluded) is not bool:
                raise ValueError("Unmatched GT occluded must be boolean.")
            unmatched_gt.append(
                GroundTruthObservation(
                    _integer(row.get("frame_index"), "unmatched GT frame_index"),
                    _integer(row.get("track_id"), "unmatched GT track_id"),
                    _class(row.get("class_name"), "unmatched GT class_name"),
                    _box(row.get("xyxy"), "unmatched GT box"),
                    occluded,
                )
            )
        unmatched_predictions: list[PredictionObservation] = []
        for row in _records(raw_frame.get("unmatched_predictions"), f"frame {frame_index} unmatched predictions"):
            unmatched_predictions.append(
                PredictionObservation(
                    _integer(row.get("frame_index"), "unmatched prediction frame_index"),
                    _integer(row.get("track_id"), "unmatched prediction track_id"),
                    _integer(row.get("class_id"), "unmatched prediction class_id"),
                    _class(row.get("class_name"), "unmatched prediction class_name"),
                    _number(row.get("confidence"), "unmatched prediction confidence"),
                    _box(row.get("xyxy"), "unmatched prediction box"),
                )
            )
        frames.append(FrameMatches(frame_index, tuple(matches), tuple(unmatched_gt), tuple(unmatched_predictions)))
    _validate_input(frames, START_FRAME, STOP_FRAME, threshold)
    recorded = document.get("summary")
    if not isinstance(recorded, dict) or recorded.get("accounting_valid") is not True or recorded.get("validation_errors") != []:
        raise ValueError("Matching artifact summary did not pass validation.")
    gt, predictions = _input_observations(frames)
    actual = validate_matches(frames, gt, predictions, START_FRAME, STOP_FRAME, CLASSES, threshold)
    for field in (
        "frame_start", "frame_stop", "evaluation_frames", "total_gt_observations",
        "total_prediction_observations", "accepted_matches", "unmatched_gt_observations",
        "unmatched_prediction_observations", "matches_per_class",
    ):
        if recorded.get(field) != getattr(actual, field):
            raise ValueError(f"Matching artifact summary disagrees with records: {field}.")
    for field in ("mean_iou", "minimum_iou", "maximum_iou"):
        old, new = recorded.get(field), getattr(actual, field)
        if old is None and new is None:
            continue
        if isinstance(old, bool) or not isinstance(old, (int, float)) or new is None or not math.isclose(old, new, rel_tol=1e-12):
            raise ValueError(f"Matching artifact summary disagrees with records: {field}.")
    return tuple(frames), threshold


def _longest_run(timeline: Sequence[TimelineEntry], matched: bool) -> int:
    longest = current = 0
    for item in timeline:
        if (item.matched_prediction_id is not None) == matched:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _track_events(track_id: int, class_name: str, timeline: tuple[TimelineEntry, ...]) -> DiagnosticEvents:
    direct: list[DirectIdentityChangeCandidate] = []
    gaps: list[AssociationGap] = []
    changed: list[Reacquisition] = []
    same: list[Reacquisition] = []
    for previous, current in zip(timeline, timeline[1:]):
        if (
            previous.matched_prediction_id is not None
            and current.matched_prediction_id is not None
            and previous.matched_prediction_id != current.matched_prediction_id
        ):
            direct.append(
                DirectIdentityChangeCandidate(
                    track_id, class_name, previous.frame_index, current.frame_index,
                    previous.matched_prediction_id, current.matched_prediction_id,
                )
            )
    index = 0
    while index < len(timeline):
        if timeline[index].matched_prediction_id is not None:
            index += 1
            continue
        start = index
        while index < len(timeline) and timeline[index].matched_prediction_id is None:
            index += 1
        before = timeline[start - 1] if start else None
        after = timeline[index] if index < len(timeline) else None
        gap_frames = tuple(item.frame_index for item in timeline[start:index])
        gaps.append(
            AssociationGap(
                track_id, class_name, gap_frames[0], gap_frames[-1], len(gap_frames), gap_frames,
                before.matched_prediction_id if before else None,
                after.matched_prediction_id if after else None,
            )
        )
        if before is not None and after is not None:
            event = Reacquisition(
                track_id, class_name, gap_frames[0], gap_frames[-1],
                before.frame_index, after.frame_index,
                before.matched_prediction_id, after.matched_prediction_id,
            )
            (same if before.matched_prediction_id == after.matched_prediction_id else changed).append(event)
    return DiagnosticEvents(tuple(direct), tuple(gaps), tuple(changed), tuple(same))


def _validate_diagnostics(
    tracks: Sequence[TrackDiagnostics],
    events: DiagnosticEvents,
    frames: Sequence[FrameMatches],
) -> tuple[str, ...]:
    errors: list[str] = []
    expected: dict[int, list[TimelineEntry]] = defaultdict(list)
    for frame in frames:
        for match in frame.matches:
            expected[match.gt_track_id].append(TimelineEntry(frame.frame_index, match.prediction_track_id, match.iou))
        for item in frame.unmatched_ground_truth:
            expected[item.track_id].append(TimelineEntry(frame.frame_index, None, None))
    if {track.gt_track_id for track in tracks} != set(expected) or len(tracks) != len(expected):
        errors.append("GT track set does not reconstruct matching input.")
    for track in tracks:
        timeline = track.timeline
        if timeline != tuple(expected.get(track.gt_track_id, ())):
            errors.append(f"GT track {track.gt_track_id} timeline does not reconstruct matching input.")
        if any(left.frame_index >= right.frame_index for left, right in zip(timeline, timeline[1:])):
            errors.append(f"GT track {track.gt_track_id} timeline is not strictly increasing.")
        if any(item.matched_iou is not None and (not math.isfinite(item.matched_iou) or not 0 <= item.matched_iou <= 1) for item in timeline):
            errors.append(f"GT track {track.gt_track_id} has invalid IoU.")
    expected_events = [
        _track_events(track.gt_track_id, track.class_name, track.timeline)
        for track in tracks
    ]
    for field in DiagnosticEvents.__dataclass_fields__:
        actual_items = getattr(events, field)
        correct_items = tuple(item for bundle in expected_events for item in getattr(bundle, field))
        if len(set(actual_items)) != len(actual_items):
            errors.append(f"Duplicate {field} event.")
        if actual_items != correct_items:
            errors.append(f"{field} events do not match the GT visible timelines.")
    return tuple(errors)


def analyze_frames(
    frames: Sequence[FrameMatches],
    start_frame: int,
    stop_frame: int,
    iou_threshold: float,
) -> DiagnosticsResult:
    """Build diagnostic timelines from existing assignments only."""
    _validate_input(frames, start_frame, stop_frame, iou_threshold)
    grouped: dict[int, list[TimelineEntry]] = defaultdict(list)
    classes: dict[int, str] = {}
    for frame in frames:
        for match in frame.matches:
            grouped[match.gt_track_id].append(TimelineEntry(frame.frame_index, match.prediction_track_id, match.iou))
            classes[match.gt_track_id] = match.class_name
        for item in frame.unmatched_ground_truth:
            grouped[item.track_id].append(TimelineEntry(frame.frame_index, None, None))
            classes[item.track_id] = item.class_name
    tracks: list[TrackDiagnostics] = []
    event_bundles: list[DiagnosticEvents] = []
    for track_id in sorted(grouped):
        timeline = tuple(grouped[track_id])
        matched = [item for item in timeline if item.matched_prediction_id is not None]
        ious = [item.matched_iou for item in matched]
        ids = tuple(sorted({item.matched_prediction_id for item in matched}))
        tracks.append(
            TrackDiagnostics(
                gt_track_id=track_id,
                class_name=classes[track_id],
                visible_start_frame=timeline[0].frame_index,
                visible_end_frame=timeline[-1].frame_index,
                visible_gt_frames=len(timeline),
                matched_frames=len(matched),
                unmatched_frames=len(timeline) - len(matched),
                match_ratio=len(matched) / len(timeline),
                prediction_ids_seen=ids,
                distinct_prediction_ids=len(ids),
                first_matched_frame=matched[0].frame_index if matched else None,
                last_matched_frame=matched[-1].frame_index if matched else None,
                longest_consecutive_matched_run=_longest_run(timeline, True),
                longest_consecutive_unmatched_run=_longest_run(timeline, False),
                mean_matched_iou=sum(ious) / len(ious) if ious else None,
                minimum_matched_iou=min(ious) if ious else None,
                maximum_matched_iou=max(ious) if ious else None,
                timeline=timeline,
            )
        )
        event_bundles.append(_track_events(track_id, classes[track_id], timeline))
    events = DiagnosticEvents(
        **{
            field: tuple(item for bundle in event_bundles for item in getattr(bundle, field))
            for field in DiagnosticEvents.__dataclass_fields__
        }
    )
    unmatched = [item for frame in frames for item in frame.unmatched_predictions]
    counts = Counter(item.class_name for item in unmatched)
    per_frame = Counter(item.frame_index for item in unmatched)
    unmatched_summary = UnmatchedPredictionSummary(
        total_observations=len(unmatched),
        observations_per_class={name: counts[name] for name in CLASSES},
        frames_with_unmatched_predictions=len(per_frame),
        frame_indices=tuple(sorted(per_frame)),
        maximum_in_one_frame=max(per_frame.values(), default=0),
    )
    matched_count = sum(len(frame.matches) for frame in frames)
    unmatched_gt_count = sum(len(frame.unmatched_ground_truth) for frame in frames)
    visible_count = matched_count + unmatched_gt_count
    per_class: dict[str, dict[str, int]] = {}
    for class_name in CLASSES:
        class_tracks = [track for track in tracks if track.class_name == class_name]
        per_class[class_name] = {
            "gt_tracks": len(class_tracks),
            "visible_gt_observations": sum(track.visible_gt_frames for track in class_tracks),
            "matched_observations": sum(track.matched_frames for track in class_tracks),
            "unmatched_gt_observations": sum(track.unmatched_frames for track in class_tracks),
            "unmatched_prediction_observations": counts[class_name],
            "direct_identity_change_candidates": sum(item.class_name == class_name for item in events.direct_identity_change_candidates),
            "association_gaps": sum(item.class_name == class_name for item in events.association_gaps),
            "reacquisition_identity_change_candidates": sum(item.class_name == class_name for item in events.reacquisition_identity_change_candidates),
            "reacquisition_same_identity": sum(item.class_name == class_name for item in events.reacquisition_same_identity),
        }
    errors = list(_validate_diagnostics(tracks, events, frames))
    accounting_valid = (
        sum(track.visible_gt_frames for track in tracks) == visible_count
        and sum(track.matched_frames for track in tracks) == matched_count
        and sum(track.unmatched_frames for track in tracks) == unmatched_gt_count
        and unmatched_summary.total_observations == sum(len(frame.unmatched_predictions) for frame in frames)
    )
    if not accounting_valid:
        errors.append("Diagnostic observation accounting does not reconstruct matching input.")
    candidates_by_track = Counter(item.gt_track_id for item in events.direct_identity_change_candidates)
    candidates_by_track.update(item.gt_track_id for item in events.reacquisition_identity_change_candidates)
    summary = DiagnosticsSummary(
        frame_start=start_frame,
        frame_stop=stop_frame,
        evaluation_frames=len(frames),
        gt_tracks_analyzed=len(tracks),
        gt_visible_observations=visible_count,
        matched_observations=matched_count,
        unmatched_gt_observations=unmatched_gt_count,
        unmatched_prediction_observations=len(unmatched),
        tracks_with_zero_identity_change_candidates=sum(candidates_by_track[track.gt_track_id] == 0 for track in tracks),
        tracks_using_multiple_prediction_ids=sum(track.distinct_prediction_ids > 1 for track in tracks),
        direct_identity_change_candidates=len(events.direct_identity_change_candidates),
        association_gaps=len(events.association_gaps),
        reacquisition_identity_change_candidates=len(events.reacquisition_identity_change_candidates),
        reacquisition_same_identity=len(events.reacquisition_same_identity),
        longest_association_gap=max((gap.length for gap in events.association_gaps), default=0),
        per_class=per_class,
        accounting_valid=accounting_valid,
        validation_errors=tuple(errors),
    )
    if not summary.valid:
        raise ValueError("Diagnostic validation failed: " + "; ".join(summary.validation_errors))
    return DiagnosticsResult(tuple(tracks), events, unmatched_summary, summary)


def format_report(result: DiagnosticsResult) -> str:
    summary = result.summary
    lines = [
        f"Evaluation source range: {summary.frame_start}-{summary.frame_stop} ({summary.evaluation_frames} frames)",
        f"GT tracks analyzed: {summary.gt_tracks_analyzed}",
        f"GT visible observations: {summary.gt_visible_observations}",
        f"Matched observations: {summary.matched_observations}",
        f"Unmatched GT observations: {summary.unmatched_gt_observations}",
        f"Unmatched prediction observations: {summary.unmatched_prediction_observations}",
        f"GT tracks with zero identity-change candidates: {summary.tracks_with_zero_identity_change_candidates}",
        f"GT tracks using multiple prediction IDs: {summary.tracks_using_multiple_prediction_ids}",
        f"Direct identity-change candidates: {summary.direct_identity_change_candidates}",
        f"Association gaps: {summary.association_gaps}",
        f"Reacquisition identity-change candidates: {summary.reacquisition_identity_change_candidates}",
        f"Reacquisition same-identity events: {summary.reacquisition_same_identity}",
        f"Longest association gap: {summary.longest_association_gap} visible GT frames",
        f"Unmatched prediction frames: {result.unmatched_predictions.frames_with_unmatched_predictions}",
        f"Maximum unmatched predictions in one frame: {result.unmatched_predictions.maximum_in_one_frame}",
        "Per-class diagnostics:",
    ]
    for name, counts in summary.per_class.items():
        lines.append(f"  {name}: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    lines.append(f"Accounting validation: {'PASS' if summary.accounting_valid else 'FAIL'}")
    lines.append(f"Validation: {'PASS' if summary.valid else 'FAIL'}")
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


def run(args: argparse.Namespace) -> DiagnosticsResult:
    frames, threshold = load_matching_artifact(args.matches)
    result = analyze_frames(frames, START_FRAME, STOP_FRAME, threshold)
    print(format_report(result))
    document: dict[str, object] = {
        "schema_version": 1,
        "configuration": {
            "frame_start": START_FRAME,
            "frame_stop": STOP_FRAME,
            "source_matching_artifact": str(args.matches),
            "matching_iou_threshold": threshold,
            "timeline_unit": "visible_gt_observation",
        },
        "tracks": [asdict(track) for track in result.tracks],
        "events": asdict(result.events),
        "unmatched_predictions": asdict(result.unmatched_predictions),
        "summary": asdict(result.summary),
    }
    _write_json(args.output, document)
    print(f"Output: {args.output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, default=Path("outputs/tracker_evaluation/matches_100_219.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tracker_evaluation/diagnostics_100_219.json"))
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
