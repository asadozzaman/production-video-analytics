"""Deterministic adapter and official TrackEval metric checks without YOLO."""

import json
import math
import unittest

import numpy as np

from src.evaluation_predictions import PredictionObservation
from src.evaluation_trackeval import (
    TRACK_EVAL_COMMIT,
    build_class_input,
    evaluate_all,
    verify_trackeval_source,
)
from src.ground_truth import GroundTruthObservation


BOX = (0.0, 0.0, 10.0, 10.0)


def gt(frame: int, identity: int, class_name: str = "car", box=BOX) -> GroundTruthObservation:
    return GroundTruthObservation(frame, identity, class_name, box, False)


def prediction(frame: int, identity: int, class_name: str = "car", box=BOX) -> PredictionObservation:
    class_id = {"person": 0, "car": 2, "truck": 7}[class_name]
    return PredictionObservation(frame, identity, class_id, class_name, 0.9, box)


class TrackEvalAdapterTests(unittest.TestCase):
    def test_perfect_single_track_sequence(self) -> None:
        g = tuple(gt(frame, 42) for frame in range(100, 103))
        p = tuple(prediction(frame, 99) for frame in range(100, 103))
        per_class, _, _ = evaluate_all(g, p, 100, 102)
        self.assertAlmostEqual(per_class["car"]["HOTA"]["HOTA"], 1.0)
        self.assertAlmostEqual(per_class["car"]["Identity"]["IDF1"], 1.0)
        self.assertEqual(per_class["car"]["CLEAR"]["IDSW"], 0)

    def test_perfect_multi_object_sequence(self) -> None:
        g = tuple(gt(frame, identity) for frame in range(100, 102) for identity in (3, 7))
        p = tuple(prediction(frame, identity) for frame in range(100, 102) for identity in (10, 20))
        per_class, _, _ = evaluate_all(g, p, 100, 101)
        self.assertAlmostEqual(per_class["car"]["HOTA"]["HOTA"], 1.0)
        self.assertEqual(per_class["car"]["CLEAR"]["CLR_TP"], 4)
        self.assertEqual(per_class["car"]["Identity"]["IDTP"], 4)

    def test_missed_detection(self) -> None:
        per_class, _, _ = evaluate_all((gt(100, 1), gt(101, 1)), (prediction(100, 10),), 100, 101)
        self.assertEqual(per_class["car"]["CLEAR"]["CLR_FN"], 1)
        self.assertEqual(per_class["car"]["Identity"]["IDFN"], 1)
        self.assertLess(per_class["car"]["HOTA"]["HOTA"], 1)

    def test_extra_tracker_detection(self) -> None:
        per_class, _, _ = evaluate_all(
            (gt(100, 1),),
            (prediction(100, 10), prediction(100, 11, box=(20.0, 0.0, 30.0, 10.0))),
            100, 100,
        )
        self.assertEqual(per_class["car"]["CLEAR"]["CLR_FP"], 1)
        self.assertEqual(per_class["car"]["Identity"]["IDFP"], 1)

    def test_identity_change(self) -> None:
        per_class, _, _ = evaluate_all(
            (gt(100, 1), gt(101, 1)),
            (prediction(100, 10), prediction(101, 11)),
            100, 101,
        )
        self.assertEqual(per_class["car"]["CLEAR"]["IDSW"], 1)
        self.assertLess(per_class["car"]["Identity"]["IDF1"], 1)

    def test_fragmented_track(self) -> None:
        # A second object keeps the timestep nonempty while GT 1 has a gap.
        g = (gt(100, 1), gt(101, 1), gt(101, 2, box=(20.0, 0.0, 30.0, 10.0)), gt(102, 1))
        p = (
            prediction(100, 10),
            prediction(101, 20, box=(20.0, 0.0, 30.0, 10.0)),
            prediction(102, 10),
        )
        per_class, _, _ = evaluate_all(g, p, 100, 102)
        self.assertEqual(per_class["car"]["CLEAR"]["Frag"], 1)

    def test_class_separation_with_reused_numeric_ids(self) -> None:
        g = (gt(100, 1, "car"), gt(100, 1, "truck"))
        p = (prediction(100, 10, "car"), prediction(100, 10, "truck"))
        per_class, _, inputs = evaluate_all(g, p, 100, 100)
        self.assertAlmostEqual(per_class["car"]["HOTA"]["HOTA"], 1.0)
        self.assertAlmostEqual(per_class["truck"]["HOTA"]["HOTA"], 1.0)
        self.assertEqual(inputs["car"].gt_id_map, {1: 0})
        self.assertEqual(inputs["truck"].gt_id_map, {1: 0})

    def test_compact_id_remapping_preserves_original_ids(self) -> None:
        g = (gt(100, 42), gt(100, 7))
        p = (prediction(100, 99), prediction(100, 500))
        adapted = build_class_input(g, p, "car", 100, 100)
        self.assertEqual(adapted.gt_id_map, {7: 0, 42: 1})
        self.assertEqual(adapted.tracker_id_map, {99: 0, 500: 1})
        self.assertEqual([item.track_id for item in g], [42, 7])
        self.assertEqual([item.track_id for item in p], [99, 500])

    def test_source_frame_timestep_mapping(self) -> None:
        adapted = build_class_input((), (), "person", 100, 219)
        self.assertEqual(adapted.data["num_timesteps"], 120)
        self.assertEqual(adapted.source_frames[0], 100)
        self.assertEqual(adapted.source_frames[119], 219)
        self.assertEqual(adapted.source_frames[0 + 17], 117)

    def test_iou_similarity_matrix_is_unthresholded_and_unclipped(self) -> None:
        g = (gt(100, 1), gt(100, 2, box=(20.0, 0.0, 30.0, 10.0)))
        p = (prediction(100, 10, box=(5.0, 0.0, 15.0, 10.0)),)
        adapted = build_class_input(g, p, "car", 100, 100)
        matrix = adapted.data["similarity_scores"][0]
        self.assertEqual(matrix.shape, (2, 1))
        self.assertAlmostEqual(matrix[0, 0], 1 / 3)
        self.assertEqual(matrix[1, 0], 0)
        outside = build_class_input(
            (gt(100, 1, box=(-5.0, 0.0, 5.0, 10.0)),),
            (prediction(100, 10, box=BOX),),
            "car", 100, 100,
        )
        self.assertAlmostEqual(outside.data["similarity_scores"][0][0, 0], 1 / 3)

    def test_empty_gt_class(self) -> None:
        per_class, _, _ = evaluate_all((), (prediction(100, 10, "person"),), 100, 100)
        self.assertEqual(per_class["person"]["CLEAR"]["CLR_FP"], 1)
        self.assertEqual(per_class["person"]["Identity"]["IDFP"], 1)
        self.assertEqual(per_class["person"]["HOTA"]["HOTA"], 0)

    def test_empty_prediction_class(self) -> None:
        per_class, _, _ = evaluate_all((gt(100, 1, "person"),), (), 100, 100)
        self.assertEqual(per_class["person"]["CLEAR"]["CLR_FN"], 1)
        self.assertEqual(per_class["person"]["Identity"]["IDFN"], 1)
        self.assertEqual(per_class["person"]["HOTA"]["HOTA"], 0)

    def test_output_json_serialization_and_finite_scalars(self) -> None:
        per_class, aggregate, _ = evaluate_all((gt(100, 1),), (prediction(100, 10),), 100, 100)
        saved = json.loads(json.dumps({"per_class": per_class, "aggregate": aggregate}, allow_nan=False))
        self.assertEqual(set(saved["per_class"]), {"person", "car", "truck"})
        self.assertTrue(math.isfinite(saved["per_class"]["car"]["HOTA"]["HOTA"]))
        self.assertEqual(len(saved["per_class"]["car"]["hota_alpha_thresholds"]), 19)
        self.assertEqual(len(saved["per_class"]["car"]["hota_curves"]["HOTA"]), 19)

    def test_duplicate_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate GT"):
            build_class_input((gt(100, 1), gt(100, 1)), (), "car", 100, 100)
        with self.assertRaisesRegex(ValueError, "Duplicate predicted"):
            build_class_input((), (prediction(100, 10), prediction(100, 10)), "car", 100, 100)

    def test_official_git_commit_is_pinned(self) -> None:
        self.assertEqual(verify_trackeval_source()["commit"], TRACK_EVAL_COMMIT)


if __name__ == "__main__":
    unittest.main()
