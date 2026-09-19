"""Detection-domain contracts.

Concrete model adapters will implement these interfaces without coupling the
rest of the pipeline to a specific YOLO release or inference runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class Detection:
    """One object detection in pixel coordinates."""

    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str | None = None

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Detection must have positive width and height.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")

    @property
    def area(self) -> float:
        """Return bounding-box area in square pixels."""
        x1, y1, x2, y2 = self.xyxy
        return (x2 - x1) * (y2 - y1)


class Detector(Protocol):
    """Interface implemented by detection backends."""

    def predict(self, frame: np.ndarray) -> Sequence[Detection]:
        """Run inference on one BGR frame."""
        ...
