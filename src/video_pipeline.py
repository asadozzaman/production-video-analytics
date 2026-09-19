"""Core orchestration for detection and tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .detection import Detection, Detector
from .metrics import Timer, TimingSummary
from .tracking import Track, Tracker


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Machine-readable result for one decoded frame."""

    frame_index: int
    timestamp_seconds: float
    detections: Sequence[Detection]
    tracks: Sequence[Track]


@dataclass(slots=True)
class PipelineMetrics:
    """Stage timings collected during one video run."""

    detection: TimingSummary
    tracking: TimingSummary


class VideoAnalyticsPipeline:
    """Coordinates interchangeable detector and tracker implementations."""

    def __init__(self, detector: Detector, tracker: Tracker) -> None:
        self.detector = detector
        self.tracker = tracker
        self.metrics = PipelineMetrics(
            detection=TimingSummary(),
            tracking=TimingSummary(),
        )

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_seconds: float,
    ) -> FrameResult:
        """Process one frame while recording stage-level latency."""
        with Timer(self.metrics.detection):
            detections = tuple(self.detector.predict(frame))

        with Timer(self.metrics.tracking):
            tracks = tuple(self.tracker.update(detections, frame))

        return FrameResult(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            detections=detections,
            tracks=tracks,
        )

    def reset(self) -> None:
        """Reset state before a new video."""
        self.tracker.reset()
        self.metrics = PipelineMetrics(
            detection=TimingSummary(),
            tracking=TimingSummary(),
        )
