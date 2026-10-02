"""Focused evidence selection, rendering, extraction, and manifest checks."""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.evaluation_evidence import (
    EVIDENCE_TYPES,
    _base_item,
    _source_records_match,
    _validate_diagnostic_events,
    _write_manifest,
    crop_bounds,
    iter_source_frames,
    render_frame,
    render_plan,
    select_evidence,
    select_unmatched_predictions,
    validate_manifest,
)
from src.evaluation_matching import FrameMatches, MatchRecord
from src.evaluation_predictions import PredictionObservation
from src.ground_truth import GroundTruthObservation


BOX = (2.0, 2.0, 14.0, 14.0)


def gt(frame: int, identity: int = 1, class_name: str = "car") -> GroundTruthObservation:
    return GroundTruthObservation(frame, identity, class_name, BOX, False)


def pred(frame: int, identity: int = 10, class_name: str = "car", confidence: float = 0.8, box=BOX) -> PredictionObservation:
    class_id = {"person": 0, "car": 2, "truck": 7}[class_name]
    return PredictionObservation(frame, identity, class_id, class_name, confidence, box)


def match(frame: int, prediction_id: int, identity: int = 1, class_name: str = "car") -> MatchRecord:
    return MatchRecord(frame, class_name, identity, prediction_id, 1.0, BOX, BOX, 0.8)


def sequence(
    *,
    matches: dict[int, tuple[MatchRecord, ...]] | None = None,
    unmatched_gt: dict[int, tuple[GroundTruthObservation, ...]] | None = None,
    unmatched_predictions: dict[int, tuple[PredictionObservation, ...]] | None = None,
) -> tuple[FrameMatches, ...]:
    matches = matches or {}
    unmatched_gt = unmatched_gt or {}
    unmatched_predictions = unmatched_predictions or {}
    return tuple(
        FrameMatches(frame, matches.get(frame, ()), unmatched_gt.get(frame, ()), unmatched_predictions.get(frame, ()))
        for frame in range(100, 220)
    )


def empty_diagnostics() -> dict[str, object]:
    return {"events": {
        "direct_identity_change_candidates": [],
        "reacquisition_identity_change_candidates": [],
        "association_gaps": [],
    }}


class EvidenceTests(unittest.TestCase):
    def test_source_frame_extraction_uses_original_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "tiny.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
            self.assertTrue(writer.isOpened())
            try:
                for level in (20, 65, 110, 155, 200):
                    writer.write(np.full((32, 32, 3), level, dtype=np.uint8))
            finally:
                writer.release()
            extracted = list(iter_source_frames(video, (1, 3)))
            self.assertEqual([index for index, _ in extracted], [1, 3])
            self.assertAlmostEqual(float(extracted[0][1].mean()), 65, delta=15)
            self.assertAlmostEqual(float(extracted[1][1].mean()), 155, delta=15)

    def test_deterministic_person_selection_spans_time_and_confidence(self) -> None:
        frames = sequence(unmatched_predictions={
            110: (pred(110, 11, "person", 0.3),),
            150: (pred(150, 12, "person", 0.6),),
            190: (pred(190, 13, "person", 0.9),),
        })
        first = select_unmatched_predictions(frames)
        self.assertEqual(first, select_unmatched_predictions(frames))
        self.assertEqual({item.frame_index for item, _ in first}, {110, 150, 190})
        self.assertTrue(all("person time bin" in reason for _, reason in first))

    def test_unmatched_gt_rendering_input_and_metadata(self) -> None:
        frames = sequence(unmatched_gt={120: (gt(120),)})
        overlay = render_plan(frames[20])
        self.assertEqual(overlay[0].status, "unmatched_gt")
        self.assertEqual((overlay[0].class_name, overlay[0].gt_track_id), ("car", 1))
        selected = select_evidence(frames, empty_diagnostics())
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["source_frame"], 120)
        self.assertEqual(selected[0]["image_path"], "frames/source_0120.jpg")

    def test_unmatched_prediction_rendering_input_and_metadata(self) -> None:
        frames = sequence(unmatched_predictions={120: (pred(120, 22, "truck", 0.72),)})
        overlay = render_plan(frames[20])[0]
        self.assertEqual(overlay.status, "unmatched_prediction")
        self.assertEqual((overlay.prediction_track_id, overlay.confidence), (22, 0.72))
        selected = select_evidence(frames, empty_diagnostics())
        self.assertEqual(selected[0]["prediction_track_id"], 22)
        self.assertEqual(selected[0]["confidence"], 0.72)
        self.assertEqual(selected[0]["xyxy"], list(BOX))

    def test_accepted_match_rendering_input(self) -> None:
        frame = FrameMatches(100, (match(100, 10),), (), ())
        overlays = render_plan(frame)
        self.assertEqual([item.status for item in overlays], ["accepted_gt", "accepted_prediction"])
        self.assertEqual(overlays[0].iou, 1.0)
        self.assertEqual(overlays[1].prediction_track_id, 10)

    def test_identity_event_frame_selection(self) -> None:
        frames = sequence(matches={159: (match(159, 11),), 160: (match(160, 24),)})
        diagnostics = empty_diagnostics()
        diagnostics["events"]["direct_identity_change_candidates"] = [{
            "gt_track_id": 1, "class_name": "car", "previous_frame": 159, "current_frame": 160,
            "previous_prediction_id": 11, "current_prediction_id": 24,
        }]
        _validate_diagnostic_events(frames, diagnostics["events"])
        item = select_evidence(frames, diagnostics)[0]
        self.assertEqual(item["evidence_type"], "direct_identity_change_candidate")
        self.assertEqual((item["source_frame"], item["context_frames"]), (160, [159]))
        self.assertEqual((item["prediction_track_id"], item["iou"]), (24, 1.0))

    def test_reacquisition_event_and_gap_context_selection(self) -> None:
        frames = sequence(
            matches={144: (match(144, 20),), 151: (match(151, 11),)},
            unmatched_gt={frame: (gt(frame),) for frame in range(145, 151)},
        )
        diagnostics = empty_diagnostics()
        gap = {
            "gt_track_id": 1, "class_name": "car", "start_frame": 145, "end_frame": 150, "length": 6,
            "visible_frames": list(range(145, 151)), "prediction_id_before_gap": 20,
            "prediction_id_after_gap": 11,
        }
        diagnostics["events"]["association_gaps"] = [gap]
        diagnostics["events"]["reacquisition_identity_change_candidates"] = [{
            "gt_track_id": 1, "class_name": "car", "gap_start_frame": 145, "gap_end_frame": 150,
            "matched_frame_before_gap": 144, "matched_frame_after_gap": 151,
            "prediction_id_before_gap": 20, "prediction_id_after_gap": 11,
        }]
        _validate_diagnostic_events(frames, diagnostics["events"])
        items = select_evidence(frames, diagnostics)
        reacquisition = next(item for item in items if item["evidence_type"] == "reacquisition_identity_change_candidate")
        self.assertEqual(reacquisition["source_frame"], 151)
        self.assertEqual(reacquisition["context_frames"], [144, 148])
        longest = next(item for item in items if item["evidence_type"] == "association_gap")
        self.assertEqual(longest["gt_track_id"], 1)

    def test_crop_bounds_and_out_of_image_predictions(self) -> None:
        self.assertEqual(crop_bounds((-10.0, 5.0, 20.0, 30.0), 100, 80), (0, 5, 20, 30))
        self.assertEqual(crop_bounds((80.0, 50.0, 120.0, 90.0), 100, 80), (80, 50, 100, 80))
        self.assertIsNone(crop_bounds((120.0, 0.0, 140.0, 20.0), 100, 80))
        frame = FrameMatches(100, (), (), (pred(100, 10, "person", box=(-10.0, -5.0, 20.0, 30.0)),))
        image = render_frame(np.zeros((80, 100, 3), dtype=np.uint8), frame)
        self.assertEqual(image.shape, (80, 100, 3))
        self.assertGreater(int(image.sum()), 0)

    def test_saved_matches_are_checked_against_original_records(self) -> None:
        frames = (FrameMatches(100, (match(100, 10),), (), ()),)
        _source_records_match(frames, (gt(100),), (pred(100),))
        with self.assertRaisesRegex(ValueError, "disagrees"):
            _source_records_match(frames, (gt(100),), (pred(100, box=(3.0, 2.0, 15.0, 14.0)),))

    def test_manifest_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frames").mkdir()
            (root / "crops").mkdir()
            image = np.full((32, 32, 3), 90, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(root / "frames/source_0100.jpg"), image))
            self.assertTrue(cv2.imwrite(str(root / "crops/source_0100_person_prediction_0010.jpg"), image))
            item = _base_item("unmatched_prediction", 100, "person")
            item.update(
                prediction_track_id=10, confidence=0.8, xyxy=list(BOX),
                crop_path="crops/source_0100_person_prediction_0010.jpg",
            )
            document = {
                "schema_version": 1,
                "items": [item],
                "summary": {
                    "evidence_frames_generated": 1,
                    "counts_by_type": {name: int(name == "unmatched_prediction") for name in EVIDENCE_TYPES},
                    "person_unmatched_prediction_samples": 1,
                },
            }
            path = root / "evidence_manifest.json"
            _write_manifest(path, document)
            saved = json.loads(path.read_text(encoding="utf-8"))
            validate_manifest(saved, root)
            self.assertEqual(saved["items"][0]["source_frame"], 100)


if __name__ == "__main__":
    unittest.main()
