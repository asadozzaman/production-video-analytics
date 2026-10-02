"""ByteTrack adapter for the repository's tracking contract."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from ..detection import Detection
from ..tracking import Track


class ByteTrackAdapter:
    """Associate repository detections without calling ``model.track()``."""

    def __init__(
        self,
        track_high_threshold: float,
        track_low_threshold: float,
        new_track_threshold: float,
        grace_frames: int,
        match_threshold: float,
        fuse_score: bool,
    ) -> None:
        if not 0 <= track_low_threshold <= track_high_threshold <= 1:
            raise ValueError("Tracking thresholds must satisfy 0 <= low <= high <= 1.")
        if not 0 <= new_track_threshold <= 1 or not 0 <= match_threshold <= 1:
            raise ValueError("New-track and match thresholds must be between 0 and 1.")
        if grace_frames < 0:
            raise ValueError("grace_frames must be nonnegative.")

        self._tracker = BYTETracker(
            SimpleNamespace(
                track_high_thresh=track_high_threshold,
                track_low_thresh=track_low_threshold,
                new_track_thresh=new_track_threshold,
                track_buffer=grace_frames,
                match_thresh=match_threshold,
                fuse_score=fuse_score,
            )
        )

    def update(self, detections: Sequence[Detection], frame: np.ndarray) -> tuple[Track, ...]:
        rows = np.asarray(
            [(*detection.xyxy, detection.confidence, detection.class_id) for detection in detections],
            dtype=np.float32,
        ).reshape(-1, 6)
        boxes = Boxes(rows, frame.shape[:2]).numpy()
        tracked_rows = self._tracker.update(boxes, frame)
        active_by_id = {track.track_id: track for track in self._tracker.tracked_stracks}

        tracks: list[Track] = []
        for x1, y1, x2, y2, track_id, score, class_id, _ in tracked_rows:
            identity = int(track_id)
            active = active_by_id[identity]
            tracks.append(
                Track(
                    track_id=identity,
                    xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(score),
                    class_id=int(class_id),
                    age=self._tracker.frame_id - active.start_frame + 1,
                    missed_frames=self._tracker.frame_id - active.end_frame,
                )
            )
        return tuple(tracks)

    def reset(self) -> None:
        self._tracker.reset()
