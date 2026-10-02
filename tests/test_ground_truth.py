"""Synthetic CVAT for video 1.1 importer fixtures."""

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from src.ground_truth import import_cvat_xml, import_cvat_zip


def xml_fixture(tracks: str, *, size: int = 3, start: int = 100, stop: int = 102) -> bytes:
    return f"""<?xml version="1.0"?>
<annotations>
  <version>1.1</version>
  <meta><task>
    <size>{size}</size><start_frame>{start}</start_frame><stop_frame>{stop}</stop_frame>
    <segments><segment><start>0</start><stop>2</stop></segment></segments>
    <labels><label><name>car</name></label><label><name>person</name></label></labels>
    <original_size><width>200</width><height>100</height></original_size>
  </task></meta>
  {tracks}
</annotations>""".encode("utf-8")


class CVATGroundTruthTests(unittest.TestCase):
    def test_source_frames_occlusion_and_outside_marker(self) -> None:
        data = xml_fixture("""
<track id="7" label="car">
  <box frame="100" outside="0" occluded="0" xtl="1" ytl="2" xbr="11" ybr="12" />
  <box frame="101" outside="0" occluded="1" xtl="2" ytl="3" xbr="12" ybr="13" />
  <box frame="102" outside="1" occluded="0" xtl="2" ytl="3" xbr="12" ybr="13" />
</track>
""")
        dataset = import_cvat_xml(data)

        self.assertTrue(dataset.valid)
        self.assertEqual((dataset.task_frame_count, dataset.start_frame, dataset.stop_frame), (3, 100, 102))
        self.assertEqual([item.frame_index for item in dataset.observations], [100, 101])
        self.assertEqual(dataset.observations[0].xyxy, (1.0, 2.0, 11.0, 12.0))
        self.assertEqual([item.occluded for item in dataset.observations], [False, True])
        self.assertEqual([(item.track_id, item.frame_index) for item in dataset.outside_markers], [(7, 102)])
        self.assertEqual(dataset.visible_observations_per_class, {"car": 2, "person": 0})
        self.assertEqual(dataset.tracks_per_class, {"car": 1, "person": 0})
        self.assertEqual((dataset.tracks[0].start_frame, dataset.tracks[0].end_frame), (100, 102))
        self.assertEqual((dataset.tracks[0].visible_start_frame, dataset.tracks[0].visible_end_frame), (100, 101))

    def test_invalid_boxes_duplicate_ids_and_out_of_range_frames(self) -> None:
        data = xml_fixture("""
<track id="1" label="car">
  <box frame="100" outside="0" occluded="0" xtl="1" ytl="1" xbr="10" ybr="10" />
  <box frame="101" outside="0" occluded="0" xtl="10" ytl="1" xbr="10" ybr="10" />
  <box frame="102" outside="0" occluded="0" xtl="1" ytl="1" xbr="201" ybr="10" />
  <box frame="103" outside="0" occluded="0" xtl="1" ytl="1" xbr="10" ybr="10" />
</track>
<track id="1" label="person">
  <box frame="100" outside="0" occluded="0" xtl="1" ytl="1" xbr="10" ybr="10" />
</track>
""")
        dataset = import_cvat_xml(data)

        self.assertFalse(dataset.valid)
        self.assertEqual(dataset.duplicate_track_ids, (1,))
        self.assertEqual(dataset.out_of_range_frames, ((1, 103),))
        self.assertEqual(len(dataset.invalid_boxes), 3)
        self.assertEqual(len(dataset.observations), 1)
        self.assertEqual(dataset.observations[0].frame_index, 100)
        self.assertEqual(dataset.tracks_per_class, {"car": 1, "person": 1})

    def test_frame_count_mismatch_and_unknown_label(self) -> None:
        data = xml_fixture('<track id="4" label="truck" />', size=4)
        dataset = import_cvat_xml(data)
        self.assertFalse(dataset.frame_count_consistent)
        self.assertEqual(dataset.unknown_track_labels, ("truck",))
        self.assertFalse(dataset.valid)

    def test_zip_read_and_missing_xml(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "annotations.zip"
            with ZipFile(archive, "w") as file:
                file.writestr("annotations.xml", xml_fixture('<track id="4" label="car" />'))
            self.assertEqual(import_cvat_zip(archive).task_frame_count, 3)
            with ZipFile(archive, "w") as file:
                file.writestr("other.xml", "<annotations />")
            with self.assertRaisesRegex(ValueError, "annotations.xml"):
                import_cvat_zip(archive)

    def test_rejects_wrong_version_and_malformed_xml(self) -> None:
        with self.assertRaisesRegex(ValueError, "1.1"):
            import_cvat_xml(xml_fixture("").replace(b"<version>1.1</version>", b"<version>1.0</version>"))
        with self.assertRaisesRegex(ValueError, "Invalid CVAT XML"):
            import_cvat_xml(b"<annotations>")


if __name__ == "__main__":
    unittest.main()
