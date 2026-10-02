"""Small, deterministic checks for machine-readable run artifacts."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.detection import Detection
from src.results import TRACK_COLUMNS, frame_record, result_streams, write_summary
from src.run_video import ByteTrackSettings, RunSettings, build_summary
from src.tracking import NullTracker, Track
from src.video_pipeline import FrameResult, VideoAnalyticsPipeline


class EmptyDetector:
    def predict(self, frame: np.ndarray) -> tuple[Detection, ...]:
        return ()


class ResultsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detection = Detection(
            (np.float32(1), np.float32(2), np.float32(11), np.float32(12)),
            np.float32(0.9),
            np.int64(2),
            "car",
        )
        self.track = Track(
            track_id=np.int64(7),
            xyxy=(np.float32(1), np.float32(2), np.float32(11), np.float32(12)),
            confidence=np.float32(0.85),
            class_id=np.int64(2),
            age=np.int64(3),
            missed_frames=np.int64(0),
        )

    def test_frame_and_csv_schemas_round_trip(self) -> None:
        first = FrameResult(0, 0.0, (self.detection,), (self.track,))
        empty = FrameResult(1, 0.04, (), ())
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with result_streams(directory) as write:
                write(first)
                write(empty)

            with (directory / "frames.json").open(encoding="utf-8") as file:
                frames = json.load(file)
            with (directory / "tracks.csv").open(encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                self.assertEqual(tuple(reader.fieldnames or ()), TRACK_COLUMNS)
                rows = list(reader)

        self.assertEqual(frames["schema_version"], 1)
        self.assertEqual(len(frames["frames"]), 2)
        self.assertEqual(frames["frames"][0]["detections"][0]["xyxy"], [1.0, 2.0, 11.0, 12.0])
        self.assertEqual(frames["frames"][0]["tracks"][0]["class_name"], "car")
        self.assertEqual(frames["frames"][0]["tracks"][0]["age"], 3)
        self.assertEqual(frames["frames"][1]["detections"], [])
        self.assertEqual(frames["frames"][1]["tracks"], [])
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["frame_index"], rows[0]["track_id"], rows[0]["class_name"]), ("0", "7", "car"))
        self.assertEqual(float(rows[0]["confidence"]), float(self.track.confidence))
        self.assertEqual(float(rows[0]["x2"]), 11.0)

    def test_empty_run_writes_parseable_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with result_streams(directory):
                pass
            frames = json.loads((directory / "frames.json").read_text(encoding="utf-8"))
            with (directory / "tracks.csv").open(encoding="utf-8", newline="") as file:
                rows = list(csv.reader(file))
        self.assertEqual(frames["frames"], [])
        self.assertEqual(rows, [list(TRACK_COLUMNS)])

    def test_summary_contains_effective_configuration_and_required_groups(self) -> None:
        settings = RunSettings(
            source=Path("video.mp4"),
            model=Path("model.pt"),
            detector_backend="yolo",
            confidence_threshold=0.37,
            iou_threshold=0.42,
            image_size=640,
            device="cpu",
            tracker_backend="bytetrack",
            bytetrack=ByteTrackSettings(0.5, 0.1, 0.6, 30, 0.8, True),
        )
        pipeline = VideoAnalyticsPipeline(detector=EmptyDetector(), tracker=NullTracker())
        pipeline.metrics.detection.observe(0.1)
        pipeline.metrics.tracking.observe(0.01)
        summary = build_summary(
            settings=settings,
            run_name="sample",
            started_at_utc="2026-01-01T00:00:00+00:00",
            output_video=Path("sample/annotated.mp4"),
            results_directory=Path("sample"),
            width=100,
            height=80,
            source_fps=25.0,
            reported_frames=1,
            processed_frames=1,
            total_detections=1,
            total_track_observations=1,
            unique_track_ids={7},
            pipeline=pipeline,
            wall_seconds=0.5,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.json"
            write_summary(path, summary)
            parsed = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(
            set(parsed),
            {"schema_version", "run", "input", "model", "tracker", "results", "performance", "environment"},
        )
        self.assertEqual(parsed["run"]["status"], "completed")
        self.assertEqual(parsed["input"]["processed_frame_count"], 1)
        self.assertEqual(parsed["model"]["confidence_threshold"], 0.37)
        self.assertEqual(parsed["tracker"]["configuration"]["new_track_threshold"], 0.6)
        self.assertEqual(parsed["results"]["unique_track_ids"], 1)
        self.assertEqual(parsed["performance"]["end_to_end_processing_throughput_fps"], 2.0)
        self.assertIn("python_version", parsed["environment"])
        self.assertIn("cuda_available", parsed["environment"])


if __name__ == "__main__":
    unittest.main()
