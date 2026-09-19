"""Tracking-domain contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .detection import Detection


@dataclass(frozen=True, slots=True)
class Track:
    """A tracked object at one point in time."""

    track_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    age: int = 1
    missed_frames: int = 0


class Tracker(Protocol):
    """Interface implemented by ByteTrack, BoT-SORT, or other backends."""

    def update(
        self,
        detections: Sequence[Detection],
        frame: np.ndarray,
    ) -> Sequence[Track]:
        """Associate current detections with persistent track identities."""
        ...

    def reset(self) -> None:
        """Clear all active tracks before processing a new video."""
        ...
