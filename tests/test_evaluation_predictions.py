"""Focused checks for source-aligned evaluation prediction records."""

import math
import unittest

import numpy as np

from src.detection import Detection
from src.evaluation_predictions import PredictionCollector, PredictionObservation, validate_predictions
from src.tracking import Track
from src.video_pipeline import FrameResult


def result(frame_index: int, classes: tuple[tuple[int, str], ...], tracks: tuple[Track, ...]) -> FrameResult:
    detections = tuple(Detection((1.0, 2.0, 20.0, 30.0), 0.9, class_id, name) for class_id, name in classes)
    return FrameResult(frame_index, frame_index / 25.0, detections, tracks)


def track(track_id: int, class_id: int, xyxy: tuple[float, float, float, float] = (1, 2, 20, 30)) -> Track:
    return Track(np.int64(track_id), xyxy, np.float32(0.85), np.int64(class_id))


class PredictionCollectorTests(unittest.TestCase):
    def test_preserves_source_numbering_and_excludes_warmup_frames(self) -> None:
        collector = PredictionCollector(100, 101, ("person", "car", "truck"), 100, 80)
        collector.add(result(99, ((2, "car"),), (track(7, 2),)))
        self.assertEqual(collector.evaluation_frames, [])
        self.assertEqual(collector.predictions, [])

        # The class name learned during context frames remains available.
        collector.add(result(100, (), (track(7, 2),)))
        collector.add(result(101, ((2, "car"),), ()))
        validation = collector.validate()
        self.assertTrue(validation.valid)
        self.assertEqual(collector.evaluation_frames, [100, 101])
        self.assertEqual(len(collector.predictions), 1)
        prediction = collector.predictions[0]
        self.assertEqual((prediction.frame_index, prediction.track_id, prediction.class_name), (100, 7, "car"))
        self.assertIs(type(prediction.frame_index), int)
        self.assertIs(type(prediction.track_id), int)
        self.assertIs(type(prediction.class_id), int)
        self.assertIs(type(prediction.confidence), float)
        self.assertTrue(all(type(value) is float for value in prediction.xyxy))

    def test_rejects_frames_after_stop_and_missing_alignment(self) -> None:
        collector = PredictionCollector(219, 219, ("car",), 100, 80)
        collector.add(result(219, ((2, "car"),), (track(1, 2),)))
        with self.assertRaisesRegex(ValueError, "after evaluation stop frame"):
            collector.add(result(220, ((2, "car"),), (track(1, 2),)))

        missing = validate_predictions(
            start_frame=100,
            stop_frame=101,
            allowed_classes=("car",),
            evaluation_frames=(101,),
            predictions=(),
        )
        self.assertFalse(missing.valid)
        self.assertTrue(missing.alignment_errors)
        outside = validate_predictions(
            start_frame=100,
            stop_frame=100,
            allowed_classes=("car",),
            evaluation_frames=(100,),
            predictions=(PredictionObservation(220, 1, 2, "car", 0.9, (1.0, 2.0, 20.0, 30.0)),),
        )
        self.assertFalse(outside.valid)
        self.assertEqual(outside.predictions_outside_range, (220,))

    def test_filters_other_classes_only_in_evaluation_representation(self) -> None:
        collector = PredictionCollector(100, 100, ("person", "car", "truck"), 100, 80)
        classes = ((0, "person"), (2, "car"), (7, "truck"), (10, "fire hydrant"))
        tracks = (track(1, 0), track(2, 2), track(3, 7), track(4, 10))
        frame = result(100, classes, tracks)
        collector.add(frame)
        validation = collector.validate()

        self.assertEqual(len(frame.tracks), 4)
        self.assertEqual([prediction.class_name for prediction in collector.predictions], ["person", "car", "truck"])
        self.assertEqual(validation.predictions_per_class, {"person": 1, "car": 1, "truck": 1})
        self.assertEqual(validation.unique_ids_per_class, {"person": 1, "car": 1, "truck": 1})
        self.assertEqual(validation.filtered_other_classes, {"fire hydrant": 1})
        self.assertEqual(validation.unsupported_classes, ())

        unsupported = validate_predictions(
            start_frame=100,
            stop_frame=100,
            allowed_classes=("person", "car", "truck"),
            evaluation_frames=(100,),
            predictions=(PredictionObservation(100, 4, 10, "fire hydrant", 0.85, (1.0, 2.0, 20.0, 30.0)),),
        )
        self.assertFalse(unsupported.valid)
        self.assertEqual(unsupported.unsupported_classes, ("fire hydrant",))

    def test_detects_invalid_boxes(self) -> None:
        collector = PredictionCollector(100, 100, ("car",), 100, 80)
        collector.add(
            result(
                100,
                ((2, "car"),),
                (track(1, 2, (10, 10, 10, 20)), track(2, 2, (0, 0, math.nan, 20)), track(3, 2, (1, 1, 101, 20))),
            )
        )
        validation = collector.validate()
        self.assertFalse(validation.valid)
        self.assertEqual(len(validation.invalid_boxes), 2)
        self.assertEqual(validation.out_of_image_boxes, ((100, 3),))
        self.assertEqual(len(collector.predictions), 1)
        self.assertEqual(collector.predictions[0].xyxy, (1.0, 1.0, 101.0, 20.0))


if __name__ == "__main__":
    unittest.main()
