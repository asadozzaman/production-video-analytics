"""Synthetic checks for GT-visible association timelines and diagnostic events."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.evaluation_diagnostics import analyze_frames, load_matching_artifact
from src.evaluation_matching import FrameMatches, MatchRecord, xyxy_iou
from src.evaluation_predictions import PredictionObservation
from src.ground_truth import GroundTruthObservation


GT_BOX = (0.0, 0.0, 10.0, 10.0)


def frames_for(
    associations: dict[int, int | None],
    *,
    start: int = 100,
    stop: int = 104,
    shifts: dict[int, float] | None = None,
    unmatched_predictions: dict[int, tuple[PredictionObservation, ...]] | None = None,
    class_name: str = "car",
) -> tuple[FrameMatches, ...]:
    """Create a frame range with one GT track visible only at supplied frames."""
    shifts = shifts or {}
    unmatched_predictions = unmatched_predictions or {}
    result: list[FrameMatches] = []
    for frame_index in range(start, stop + 1):
        matches: tuple[MatchRecord, ...] = ()
        unmatched_gt: tuple[GroundTruthObservation, ...] = ()
        if frame_index in associations:
            prediction_id = associations[frame_index]
            if prediction_id is None:
                unmatched_gt = (GroundTruthObservation(frame_index, 1, class_name, GT_BOX, False),)
            else:
                offset = shifts.get(frame_index, 0.0)
                prediction_box = (offset, 0.0, 10.0 + offset, 10.0)
                matches = (
                    MatchRecord(
                        frame_index, class_name, 1, prediction_id,
                        xyxy_iou(GT_BOX, prediction_box), GT_BOX, prediction_box, 0.9,
                    ),
                )
        result.append(
            FrameMatches(frame_index, matches, unmatched_gt, unmatched_predictions.get(frame_index, ()))
        )
    return tuple(result)


class DiagnosticsTests(unittest.TestCase):
    def test_stable_identity_has_no_change_events(self) -> None:
        result = analyze_frames(frames_for({100: 7, 101: 7, 102: 7}), 100, 104, 0.5)
        track = result.tracks[0]
        self.assertEqual(track.prediction_ids_seen, (7,))
        self.assertEqual(track.visible_gt_frames, 3)
        self.assertEqual(track.matched_frames, 3)
        self.assertEqual(track.match_ratio, 1.0)
        self.assertEqual(result.events.direct_identity_change_candidates, ())
        self.assertEqual(result.events.reacquisition_identity_change_candidates, ())
        self.assertEqual(result.summary.tracks_with_zero_identity_change_candidates, 1)

    def test_consecutive_visible_matched_frames_change_id(self) -> None:
        result = analyze_frames(frames_for({100: 7, 101: 9}), 100, 104, 0.5)
        event = result.events.direct_identity_change_candidates[0]
        self.assertEqual((event.previous_frame, event.current_frame), (100, 101))
        self.assertEqual((event.previous_prediction_id, event.current_prediction_id), (7, 9))
        self.assertEqual(len(result.events.direct_identity_change_candidates), 1)
        self.assertEqual(result.summary.tracks_using_multiple_prediction_ids, 1)

    def test_same_identity_after_gap(self) -> None:
        result = analyze_frames(frames_for({100: 7, 101: None, 102: None, 103: 7}), 100, 104, 0.5)
        gap = result.events.association_gaps[0]
        self.assertEqual((gap.start_frame, gap.end_frame, gap.length), (101, 102, 2))
        self.assertEqual((gap.prediction_id_before_gap, gap.prediction_id_after_gap), (7, 7))
        self.assertEqual(len(result.events.reacquisition_same_identity), 1)
        self.assertEqual(result.events.reacquisition_identity_change_candidates, ())
        self.assertEqual(result.events.direct_identity_change_candidates, ())

    def test_different_identity_after_gap(self) -> None:
        result = analyze_frames(frames_for({100: 7, 101: None, 102: None, 103: 9}), 100, 104, 0.5)
        self.assertEqual(len(result.events.association_gaps), 1)
        self.assertEqual(len(result.events.reacquisition_identity_change_candidates), 1)
        self.assertEqual(result.events.reacquisition_same_identity, ())
        self.assertEqual(result.events.direct_identity_change_candidates, ())

    def test_completely_unmatched_gt_track(self) -> None:
        result = analyze_frames(frames_for({100: None, 101: None, 102: None}), 100, 104, 0.5)
        track = result.tracks[0]
        self.assertEqual(track.longest_consecutive_unmatched_run, 3)
        self.assertEqual(track.longest_consecutive_matched_run, 0)
        self.assertEqual(track.prediction_ids_seen, ())
        self.assertIsNone(track.mean_matched_iou)
        self.assertEqual(len(result.events.association_gaps), 1)
        self.assertEqual(result.events.reacquisition_same_identity, ())
        self.assertEqual(result.events.reacquisition_identity_change_candidates, ())

    def test_multiple_gaps_and_run_lengths(self) -> None:
        result = analyze_frames(
            frames_for({100: 7, 101: None, 102: 7, 103: None, 104: None, 105: 7, 106: 7}, stop=106),
            100, 106, 0.5,
        )
        track = result.tracks[0]
        self.assertEqual(len(result.events.association_gaps), 2)
        self.assertEqual(track.longest_consecutive_matched_run, 2)
        self.assertEqual(track.longest_consecutive_unmatched_run, 2)
        self.assertEqual(result.summary.longest_association_gap, 2)
        self.assertEqual(len(result.events.reacquisition_same_identity), 2)

    def test_visible_timeline_skips_absent_gt_frames(self) -> None:
        result = analyze_frames(frames_for({100: 7, 102: 9, 104: None}), 100, 104, 0.5)
        self.assertEqual([item.frame_index for item in result.tracks[0].timeline], [100, 102, 104])
        self.assertEqual(len(result.events.direct_identity_change_candidates), 1)
        self.assertEqual(result.events.association_gaps[0].visible_frames, (104,))

    def test_matched_iou_statistics(self) -> None:
        result = analyze_frames(
            frames_for({100: 7, 101: 7, 102: 7}, shifts={100: 0.0, 101: 1.0, 102: 2.0}),
            100, 104, 0.5,
        )
        values = [item.matched_iou for item in result.tracks[0].timeline]
        track = result.tracks[0]
        self.assertAlmostEqual(track.mean_matched_iou, sum(values) / 3)
        self.assertEqual(track.minimum_matched_iou, min(values))
        self.assertEqual(track.maximum_matched_iou, max(values))

    def test_unmatched_prediction_summary(self) -> None:
        extras = {
            100: (
                PredictionObservation(100, 10, 0, "person", 0.8, GT_BOX),
                PredictionObservation(100, 11, 7, "truck", 0.7, GT_BOX),
            ),
            102: (PredictionObservation(102, 12, 0, "person", 0.6, GT_BOX),),
        }
        result = analyze_frames(frames_for({}, unmatched_predictions=extras), 100, 104, 0.5)
        unmatched = result.unmatched_predictions
        self.assertEqual(unmatched.total_observations, 3)
        self.assertEqual(unmatched.observations_per_class, {"person": 2, "car": 0, "truck": 1})
        self.assertEqual(unmatched.frame_indices, (100, 102))
        self.assertEqual(unmatched.frames_with_unmatched_predictions, 2)
        self.assertEqual(unmatched.maximum_in_one_frame, 2)

    def test_accounting_reconstructs_matching_input(self) -> None:
        frames = frames_for({100: 7, 101: None, 102: 7})
        result = analyze_frames(frames, 100, 104, 0.5)
        self.assertTrue(result.summary.accounting_valid)
        self.assertEqual(result.summary.gt_visible_observations, 3)
        self.assertEqual(result.summary.matched_observations + result.summary.unmatched_gt_observations, 3)
        self.assertEqual(result.summary.matched_observations, sum(len(frame.matches) for frame in frames))

    def test_duplicate_gt_assignment_rejected(self) -> None:
        frames = list(frames_for({100: 7}))
        frames[0] = replace(frames[0], matches=frames[0].matches + frames[0].matches)
        with self.assertRaisesRegex(ValueError, "Duplicate assignment"):
            analyze_frames(frames, 100, 104, 0.5)

    def test_malformed_matching_artifact_rejected(self) -> None:
        malformed = {
            "schema_version": 1,
            "configuration": {
                "frame_start": 100, "frame_stop": 219,
                "iou_threshold": 0.5, "class_aware": True, "assignment": "hungarian",
            },
            "frames": [],
            "summary": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "malformed.json"
            path.write_text(json.dumps(malformed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "every source frame"):
                load_matching_artifact(path)


if __name__ == "__main__":
    unittest.main()
