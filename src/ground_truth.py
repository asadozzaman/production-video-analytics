"""Import and validate CVAT for video 1.1 ground truth."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile


@dataclass(frozen=True, slots=True)
class GroundTruthObservation:
    frame_index: int
    track_id: int
    class_name: str
    xyxy: tuple[float, float, float, float]
    occluded: bool


@dataclass(frozen=True, slots=True)
class OutsideMarker:
    frame_index: int
    track_id: int
    class_name: str


@dataclass(frozen=True, slots=True)
class InvalidBox:
    track_id: int
    frame_index: int | None
    reason: str


@dataclass(frozen=True, slots=True)
class TrackSpan:
    track_id: int
    class_name: str
    start_frame: int | None
    end_frame: int | None
    visible_start_frame: int | None
    visible_end_frame: int | None


@dataclass(frozen=True, slots=True)
class CVATGroundTruth:
    task_frame_count: int
    start_frame: int
    stop_frame: int
    labels: tuple[str, ...]
    tracks: tuple[TrackSpan, ...]
    observations: tuple[GroundTruthObservation, ...]
    outside_markers: tuple[OutsideMarker, ...]
    invalid_boxes: tuple[InvalidBox, ...]
    duplicate_track_ids: tuple[int, ...]
    out_of_range_frames: tuple[tuple[int, int], ...]
    unknown_track_labels: tuple[str, ...]

    @property
    def frame_count_consistent(self) -> bool:
        return self.task_frame_count == self.stop_frame - self.start_frame + 1

    @property
    def valid(self) -> bool:
        return (
            self.frame_count_consistent
            and not self.invalid_boxes
            and not self.duplicate_track_ids
            and not self.out_of_range_frames
            and not self.unknown_track_labels
        )

    @property
    def tracks_per_class(self) -> dict[str, int]:
        counts = Counter(track.class_name for track in self.tracks)
        return {label: counts[label] for label in self.labels}

    @property
    def visible_observations_per_class(self) -> dict[str, int]:
        counts = Counter(observation.class_name for observation in self.observations)
        return {label: counts[label] for label in self.labels}


def _required_int(element: ET.Element, path: str) -> int:
    value = element.findtext(path)
    if value is None:
        raise ValueError(f"Missing CVAT task {path}.")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Missing or invalid CVAT task {path}: {value!r}") from exc


def import_cvat_xml(xml_bytes: bytes) -> CVATGroundTruth:
    """Read source frame indices from box attributes, without job-local offsets."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid CVAT XML: {exc}") from exc
    if root.tag != "annotations" or root.findtext("version") != "1.1":
        raise ValueError("Expected a CVAT for video 1.1 annotations export.")
    task = root.find("meta/task")
    if task is None:
        raise ValueError("CVAT task metadata is missing.")

    task_frame_count = _required_int(task, "size")
    start_frame = _required_int(task, "start_frame")
    stop_frame = _required_int(task, "stop_frame")
    if task_frame_count <= 0 or stop_frame < start_frame:
        raise ValueError("CVAT task frame count or source frame range is invalid.")
    labels = tuple(name for label in task.findall("labels/label") if (name := label.findtext("name")))
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("CVAT task labels are missing or duplicated.")
    width_text = task.findtext("original_size/width")
    height_text = task.findtext("original_size/height")
    width = int(width_text) if width_text else None
    height = int(height_text) if height_text else None

    observations: list[GroundTruthObservation] = []
    outside_markers: list[OutsideMarker] = []
    invalid_boxes: list[InvalidBox] = []
    out_of_range_frames: list[tuple[int, int]] = []
    spans: list[TrackSpan] = []
    track_ids: list[int] = []
    unknown_labels: set[str] = set()
    visible_keys: set[tuple[int, int]] = set()

    for track in root.findall("track"):
        try:
            track_id = int(track.attrib["id"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid CVAT track ID: {track.get('id')!r}") from exc
        class_name = track.get("label") or ""
        if class_name not in labels:
            unknown_labels.add(class_name)
        track_ids.append(track_id)
        track_frames: list[int] = []
        visible_frames: list[int] = []

        for box in track.findall("box"):
            try:
                frame_index = int(box.attrib["frame"])
            except (KeyError, ValueError):
                invalid_boxes.append(InvalidBox(track_id, None, "missing or invalid frame index"))
                continue
            track_frames.append(frame_index)
            if not start_frame <= frame_index <= stop_frame:
                out_of_range_frames.append((track_id, frame_index))
                continue

            outside = box.get("outside")
            if outside == "1":
                outside_markers.append(OutsideMarker(frame_index, track_id, class_name))
                continue
            if outside != "0":
                invalid_boxes.append(InvalidBox(track_id, frame_index, "outside must be 0 or 1"))
                continue
            occluded = box.get("occluded")
            if occluded not in {"0", "1"}:
                invalid_boxes.append(InvalidBox(track_id, frame_index, "occluded must be 0 or 1"))
                continue
            try:
                x1, y1, x2, y2 = (float(box.attrib[key]) for key in ("xtl", "ytl", "xbr", "ybr"))
            except (KeyError, ValueError):
                invalid_boxes.append(InvalidBox(track_id, frame_index, "missing or nonnumeric coordinates"))
                continue
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                reason = "nonfinite coordinates"
            elif x2 <= x1 or y2 <= y1:
                reason = "nonpositive box width or height"
            elif x1 < 0 or y1 < 0 or (width is not None and x2 > width) or (height is not None and y2 > height):
                reason = "coordinates outside original image"
            elif (track_id, frame_index) in visible_keys:
                reason = "duplicate visible box for track and frame"
            else:
                reason = ""
            if reason:
                invalid_boxes.append(InvalidBox(track_id, frame_index, reason))
                continue

            visible_keys.add((track_id, frame_index))
            visible_frames.append(frame_index)
            observations.append(
                GroundTruthObservation(frame_index, track_id, class_name, (x1, y1, x2, y2), occluded == "1")
            )

        spans.append(
            TrackSpan(
                track_id=track_id,
                class_name=class_name,
                start_frame=min(track_frames) if track_frames else None,
                end_frame=max(track_frames) if track_frames else None,
                visible_start_frame=min(visible_frames) if visible_frames else None,
                visible_end_frame=max(visible_frames) if visible_frames else None,
            )
        )

    id_counts = Counter(track_ids)
    return CVATGroundTruth(
        task_frame_count=task_frame_count,
        start_frame=start_frame,
        stop_frame=stop_frame,
        labels=labels,
        tracks=tuple(spans),
        observations=tuple(sorted(observations, key=lambda item: (item.frame_index, item.track_id))),
        outside_markers=tuple(sorted(outside_markers, key=lambda item: (item.frame_index, item.track_id))),
        invalid_boxes=tuple(invalid_boxes),
        duplicate_track_ids=tuple(sorted(track_id for track_id, count in id_counts.items() if count > 1)),
        out_of_range_frames=tuple(out_of_range_frames),
        unknown_track_labels=tuple(sorted(unknown_labels)),
    )


def import_cvat_zip(path: str | Path) -> CVATGroundTruth:
    archive = Path(path)
    if not archive.is_file():
        raise FileNotFoundError(f"CVAT archive not found: {archive}")
    try:
        with ZipFile(archive) as zip_file:
            return import_cvat_xml(zip_file.read("annotations.xml"))
    except (BadZipFile, KeyError) as exc:
        raise ValueError(f"CVAT archive must contain annotations.xml: {archive}") from exc


def format_validation(dataset: CVATGroundTruth) -> str:
    lines = [
        f"Task frame count: {dataset.task_frame_count}",
        f"Source frame range: {dataset.start_frame}-{dataset.stop_frame}",
        f"Frame count consistent with range: {dataset.frame_count_consistent}",
        f"Labels: {', '.join(dataset.labels)}",
        f"Tracks: {len(dataset.tracks)}",
        "Tracks per class: " + ", ".join(f"{name}={count}" for name, count in dataset.tracks_per_class.items()),
        "Visible observations per class: "
        + ", ".join(f"{name}={count}" for name, count in dataset.visible_observations_per_class.items()),
        "Track ranges (source frames; all boxes / visible boxes):",
    ]
    for track in dataset.tracks:
        lines.append(
            f"  ID {track.track_id} {track.class_name}: "
            f"{track.start_frame}-{track.end_frame} / {track.visible_start_frame}-{track.visible_end_frame}"
        )
    lines.append(f"Outside markers: {len(dataset.outside_markers)}")
    for marker in dataset.outside_markers:
        lines.append(f"  ID {marker.track_id} {marker.class_name} frame {marker.frame_index}")
    lines.append(f"Invalid boxes: {len(dataset.invalid_boxes)}")
    for box in dataset.invalid_boxes:
        lines.append(f"  ID {box.track_id} frame {box.frame_index}: {box.reason}")
    lines.append(f"Duplicate track IDs: {list(dataset.duplicate_track_ids)}")
    lines.append(f"Frames outside expected range: {list(dataset.out_of_range_frames)}")
    lines.append(f"Unknown track labels: {list(dataset.unknown_track_labels)}")
    lines.append(f"Validation: {'PASS' if dataset.valid else 'FAIL'}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="CVAT for video 1.1 ZIP containing annotations.xml")
    args = parser.parse_args()
    try:
        dataset = import_cvat_zip(args.archive)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(format_validation(dataset))
    if not dataset.valid:
        parser.exit(1)


if __name__ == "__main__":
    main()
