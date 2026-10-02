# Tracker evaluation predictions: first milestone

Run the repository's `VideoAnalyticsPipeline` with `UltralyticsYOLODetector` and `ByteTrackAdapter` on `data/input/test_traffic.mp4`. The command reads the verified CVAT archive to obtain the original source range, 100–219 inclusive. It processes source frames **0–219 in order**, so ByteTrack sees frames 0–99 before the evaluation window. Only frames 100–219 are recorded. Frame indexes in the output are original video indexes, not indexes relative to the CVAT job UI.

```powershell
& 'env/Scripts/python.exe' -m src.evaluation_predictions --source data/input/test_traffic.mp4 --model models/yolo26n.pt --ground-truth data/ground_truth/traffic_tracker_ground_truth_v1.zip --output outputs/tracker_evaluation/predictions_100_219.json --device cpu --conf 0.25 --imgsz 640 --tracker bytetrack
```

The command uses the effective ByteTrack settings in `configs/default.yaml`. It runs detection and tracking on all processed frames and filters **only the saved evaluation representation** to `person`, `car`, and `truck`. Each prediction has `frame_index`, `track_id`, `class_id`, `class_name`, `confidence`, and `xyxy`, converted to native Python numeric types before JSON serialization. The JSON also includes every evaluation frame index, including frames with no predictions, and records the tracker context and effective model/tracker settings.

Validation requires exactly the ordered source frames 100–219, no saved predictions outside that range, only the three supported classes, and finite boxes with positive width and height. The CLI reports counts and fails before writing output if alignment or record validation fails. ByteTrack can extend a valid box beyond the image boundary for an object at the edge of view; those boxes retain their raw coordinates and are counted separately as a diagnostic.

## CPU validation run

Using `models/yolo26n.pt`, confidence 0.25, image size 640, and the existing ByteTrack configuration:

| Measure | Result |
| --- | ---: |
| Source frames processed, including context | 220 (0–219) |
| Evaluation frames processed | 120 (100–219) |
| Prediction observations | 496 |
| Person observations | 307 |
| Car observations | 105 |
| Truck observations | 84 |
| Unique predicted track IDs | 16 |
| Unique IDs appearing as person / car / truck | 11 / 5 / 2 |
| Invalid boxes | 0 |
| Boxes extending beyond source image | 110 |
| Predictions outside range | 0 |
| Unsupported classes in output | 0 |

The per-class unique-ID counts sum to more than 16 because IDs 11 and 24 appear as both car and truck at different times. This is an observed tracker/classification behavior, not an evaluation metric. The tracker also produced 107 fire-hydrant observations in the evaluation window; these were filtered from the prediction records. No ground-truth matching or tracking accuracy metric is computed at this milestone.
