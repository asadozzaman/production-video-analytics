"""Serialize repository frame results without depending on CV backends."""

from __future__ import annotations

import csv
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from .detection import Detection
from .tracking import Track
from .video_pipeline import FrameResult


TRACK_COLUMNS = (
    "frame_index",
    "timestamp_seconds",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
)


def detection_record(detection: Detection) -> dict[str, Any]:
    return {
        "xyxy": [float(value) for value in detection.xyxy],
        "confidence": float(detection.confidence),
        "class_id": int(detection.class_id),
        "class_name": detection.class_name,
    }


def track_record(track: Track, class_name: str | None) -> dict[str, Any]:
    return {
        "track_id": int(track.track_id),
        "xyxy": [float(value) for value in track.xyxy],
        "confidence": float(track.confidence),
        "class_id": int(track.class_id),
        "class_name": class_name,
        "age": int(track.age),
        "missed_frames": int(track.missed_frames),
    }


def frame_record(result: FrameResult) -> dict[str, Any]:
    class_names = {int(d.class_id): d.class_name for d in result.detections}
    return {
        "frame_index": int(result.frame_index),
        "timestamp_seconds": float(result.timestamp_seconds),
        "detections": [detection_record(detection) for detection in result.detections],
        "tracks": [track_record(track, class_names.get(int(track.class_id))) for track in result.tracks],
    }


def track_csv_row(
    frame_index: int,
    timestamp_seconds: float,
    track: Track,
    class_name: str | None,
) -> dict[str, int | float | str]:
    x1, y1, x2, y2 = track.xyxy
    return {
        "frame_index": int(frame_index),
        "timestamp_seconds": float(timestamp_seconds),
        "track_id": int(track.track_id),
        "class_id": int(track.class_id),
        "class_name": class_name or "",
        "confidence": float(track.confidence),
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
    }


@contextmanager
def result_streams(directory: Path) -> Iterator[Callable[[FrameResult], None]]:
    """Write frame entries incrementally and keep CSV columns stable."""
    with (directory / "tracks.csv").open("w", encoding="utf-8", newline="") as csv_file, (
        directory / "frames.json"
    ).open("w", encoding="utf-8", newline="") as json_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=TRACK_COLUMNS)
        csv_writer.writeheader()
        json_file.write('{"schema_version": 1, "frames": [\n')
        first = True

        def write(result: FrameResult) -> None:
            nonlocal first
            if not first:
                json_file.write(",\n")
            json.dump(frame_record(result), json_file, ensure_ascii=False, allow_nan=False)
            first = False

            class_names = {int(d.class_id): d.class_name for d in result.detections}
            for track in result.tracks:
                csv_writer.writerow(
                    track_csv_row(
                        result.frame_index,
                        result.timestamp_seconds,
                        track,
                        class_names.get(int(track.class_id)),
                    )
                )

        yield write
        json_file.write("\n]}\n")


def write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")
