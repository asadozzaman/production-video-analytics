"""Deterministic checks for frame- and class-aware IoU assignment."""

import math
import unittest
from dataclasses import replace

import numpy as np

from src.evaluation_matching import assign_max_iou, evaluate, match_frame, validate_matches, xyxy_iou
from src.evaluation_predictions import PredictionObservation
from src.ground_truth import GroundTruthObservation


CLASSES = ("person", "car", "truck")
BOX = (0.0, 0.0, 2.0, 2.0)


def gt(frame: int, identity: int, name: str = "car", box: tuple[float, float, float, float] = BOX) -> GroundTruthObservation:
    return GroundTruthObservation(frame, identity, name, box, False)


def prediction(
    frame: int,
    identity: int,
    name: str = "car",
    box: tuple[float, float, float, float] = BOX,
) -> PredictionObservation:
    class_id = {"person": 0, "car": 2, "truck": 7}[name]
    return PredictionObservation(frame, identity, class_id, name, 0.8, box)


class IoUTests(unittest.TestCase):
    def test_identical_boxes(self) -> None:
        self.assertEqual(xyxy_iou(BOX, BOX), 1.0)

    def test_disjoint_and_touching_boxes(self) -> None:
        self.assertEqual(xyxy_iou(BOX, (3.0, 0.0, 5.0, 2.0)), 0.0)
        self.assertEqual(xyxy_iou(BOX, (2.0, 0.0, 4.0, 2.0)), 0.0)

    def test_partial_overlap(self) -> None:
        self.assertAlmostEqual(xyxy_iou(BOX, (1.0, 0.0, 3.0, 2.0)), 1.0 / 3.0)

    def test_invalid_boxes_fail_without_negative_area(self) -> None:
        for bad in ((2.0, 0.0, 1.0, 2.0), (0.0, 0.0, 0.0, 2.0), (0.0, 0.0, math.nan, 2.0)):
            with self.subTest(box=bad), self.assertRaises(ValueError):
                xyxy_iou(bad, BOX)

    def test_coordinates_are_not_clipped(self) -> None:
        self.assertEqual(xyxy_iou((-1.0, 0.0, 1.0, 2.0), BOX), 1.0 / 3.0)


class MatchingTests(unittest.TestCase):
    def test_cross_class_never_matches(self) -> None:
        frame = match_frame(100, (gt(100, 1, "car"),), (prediction(100, 2, "truck"),), CLASSES, 0.5)
        self.assertEqual(frame.matches, ())
        self.assertEqual((len(frame.unmatched_ground_truth), len(frame.unmatched_predictions)), (1, 1))

    def test_different_frames_never_match(self) -> None:
        frames, summary = evaluate((gt(100, 1),), (prediction(101, 2),), 100, 101, CLASSES, 0.5)
        self.assertEqual(len(frames), 2)
        self.assertEqual(summary.accepted_matches, 0)
        self.assertTrue(summary.valid)

    def test_one_prediction_cannot_match_two_gt_objects(self) -> None:
        frame = match_frame(100, (gt(100, 1), gt(100, 2)), (prediction(100, 3),), CLASSES, 0.5)
        self.assertEqual(len(frame.matches), 1)
        self.assertEqual(len(frame.unmatched_ground_truth), 1)
        self.assertEqual(len(frame.unmatched_predictions), 0)

    def test_one_gt_cannot_match_two_predictions(self) -> None:
        frame = match_frame(100, (gt(100, 1),), (prediction(100, 2), prediction(100, 3)), CLASSES, 0.5)
        self.assertEqual(len(frame.matches), 1)
        self.assertEqual(len(frame.unmatched_ground_truth), 0)
        self.assertEqual(len(frame.unmatched_predictions), 1)

    def test_threshold_includes_exact_boundary(self) -> None:
        other = (1.0, 0.0, 3.0, 2.0)
        boundary = xyxy_iou(BOX, other)
        accepted = match_frame(100, (gt(100, 1),), (prediction(100, 2, box=other),), CLASSES, boundary)
        rejected = match_frame(100, (gt(100, 1),), (prediction(100, 2, box=other),), CLASSES, boundary + 0.01)
        self.assertEqual(len(accepted.matches), 1)
        self.assertEqual(len(rejected.matches), 0)

    def test_empty_gt_frame(self) -> None:
        frame = match_frame(100, (), (prediction(100, 2),), CLASSES, 0.5)
        self.assertEqual(frame.matches, ())
        self.assertEqual(len(frame.unmatched_predictions), 1)

    def test_empty_prediction_frame(self) -> None:
        frame = match_frame(100, (gt(100, 1),), (), CLASSES, 0.5)
        self.assertEqual(frame.matches, ())
        self.assertEqual(len(frame.unmatched_ground_truth), 1)

    def test_hungarian_beats_naive_highest_first_greedy(self) -> None:
        matrix = np.array([[0.90, 0.80], [0.85, 0.10]])
        pairs = assign_max_iou(matrix)
        self.assertEqual(set(pairs), {(0, 1), (1, 0)})
        self.assertAlmostEqual(sum(matrix[row, col] for row, col in pairs), 1.65)

    def test_accounting_and_duplicate_assignment_validation(self) -> None:
        ground_truth = (gt(100, 1), gt(100, 2, "person"), gt(101, 3))
        predictions = (prediction(100, 10), prediction(100, 20, "truck"), prediction(101, 30))
        frames, summary = evaluate(ground_truth, predictions, 100, 101, CLASSES, 0.5)
        self.assertTrue(summary.valid)
        self.assertEqual(summary.accepted_matches, 2)
        self.assertEqual(summary.accepted_matches + summary.unmatched_gt_observations, len(ground_truth))
        self.assertEqual(summary.accepted_matches + summary.unmatched_prediction_observations, len(predictions))
        damaged = (replace(frames[0], matches=frames[0].matches + frames[0].matches), frames[1])
        invalid = validate_matches(damaged, ground_truth, predictions, 100, 101, CLASSES, 0.5)
        self.assertFalse(invalid.valid)
        self.assertTrue(any("Duplicate assignment" in error for error in invalid.validation_errors))

    def test_cross_class_and_bad_iou_output_fail_validation(self) -> None:
        ground_truth = (gt(100, 1),)
        predictions = (prediction(100, 2),)
        frames, _ = evaluate(ground_truth, predictions, 100, 100, CLASSES, 0.5)
        match = frames[0].matches[0]
        corrupted = replace(frames[0], matches=(replace(match, class_name="truck", iou=math.nan),))
        summary = validate_matches((corrupted,), ground_truth, predictions, 100, 100, CLASSES, 0.5)
        self.assertFalse(summary.valid)
        self.assertTrue(any("Cross-class" in error for error in summary.validation_errors))
        self.assertTrue(any("Invalid IoU" in error for error in summary.validation_errors))


if __name__ == "__main__":
    unittest.main()
