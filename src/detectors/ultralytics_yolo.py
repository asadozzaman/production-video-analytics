"""Ultralytics YOLO adapter for the repository's Detection contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from ..detection import Detection


class UltralyticsYOLODetector:
    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float,
        iou_threshold: float,
        image_size: int,
        device: str,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model weights not found: {path}")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1.")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("IoU threshold must be between 0 and 1.")
        if image_size <= 0:
            raise ValueError("Image size must be positive.")

        self.model = YOLO(str(path))
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        # Ultralytics uses an empty device value for automatic selection.
        self.device = "" if device == "auto" else device

    def predict(self, frame: np.ndarray) -> tuple[Detection, ...]:
        result = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None:
            return ()

        names = result.names
        detections: list[Detection] = []
        for box in boxes:
            class_id = int(box.cls.item())
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    xyxy=(x1, y1, x2, y2),
                    confidence=float(box.conf.item()),
                    class_id=class_id,
                    class_name=str(names[class_id]),
                )
            )
        return tuple(detections)
