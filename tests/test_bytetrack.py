"""Focused checks for the ByteTrack domain adapter."""

import unittest

import numpy as np

from src.detection import Detection
from src.trackers import ByteTrackAdapter


class ByteTrackAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = ByteTrackAdapter(0.5, 0.1, 0.6, 30, 0.8, True)
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.detection = Detection((10, 10, 40, 40), 0.9, 2, "car")

    def test_converts_detections_and_preserves_identity(self) -> None:
        first = self.tracker.update((self.detection,), self.frame)
        second = self.tracker.update((self.detection,), self.frame)

        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertEqual(first[0].xyxy, self.detection.xyxy)
        self.assertEqual(first[0].class_id, self.detection.class_id)
        self.assertAlmostEqual(first[0].confidence, self.detection.confidence)
        self.assertEqual((first[0].age, second[0].age), (1, 2))
        self.assertEqual(second[0].missed_frames, 0)

    def test_reset_clears_state(self) -> None:
        self.tracker.update((self.detection,), self.frame)
        self.tracker.update((self.detection,), self.frame)
        self.tracker.reset()

        self.assertEqual(self.tracker.update((), self.frame), ())
        self.assertEqual(self.tracker.update((self.detection,), self.frame), ())
        track = self.tracker.update((self.detection,), self.frame)[0]
        self.assertEqual(track.track_id, 1)
        self.assertEqual(track.age, 2)
        self.assertEqual(track.missed_frames, 0)


if __name__ == "__main__":
    unittest.main()
