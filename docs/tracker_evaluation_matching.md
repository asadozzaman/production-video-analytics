# Frame-by-frame tracker matching

This milestone matches the verified CVAT visible observations to the saved ByteTrack predictions for original source frames 100–219. It reads the existing ground-truth importer and prediction JSON; it does not rerun detection or tracking.

```powershell
& 'env/Scripts/python.exe' -m src.evaluation_matching --ground-truth data/ground_truth/traffic_tracker_ground_truth_v1.zip --predictions outputs/tracker_evaluation/predictions_100_219.json --output outputs/tracker_evaluation/matches_100_219.json --iou-threshold 0.5
```

For each source frame and each of `person`, `car`, and `truck`, the matcher forms an XYXY IoU matrix and uses SciPy's Hungarian assignment to maximize total IoU with one-to-one pairs. It then accepts assigned pairs with IoU at or above the CLI threshold. All remaining observations are stored as `unmatched_ground_truth` or `unmatched_predictions`; they are matching diagnostics, not tracking accuracy classifications.

IoU uses the **stored coordinates directly**. There is no clipping to the source image: the prediction run reported some boxes extending past the image edge, and clipping would change their geometry. Every box must have finite coordinates and positive width and height. Touching boxes have zero intersection.

The output JSON contains a record for every source frame, including empty frames, with accepted pairs and both unmatched lists. Validation checks frame alignment, class agreement, unique assignment, the threshold, IoU values, and reconstruction of the original observation counts. Validation failure prevents writing the output artifact. The artifact is ignored by Git under `outputs/`.

## Verified matching run

| Diagnostic | Result |
| --- | ---: |
| Evaluation source frames | 100–219 (120) |
| IoU threshold | 0.50 |
| GT visible observations | 328 |
| Prediction observations | 496 |
| Accepted matches | 223 |
| Unmatched GT observations | 105 |
| Unmatched prediction observations | 273 |
| Matches: person / car / truck | 68 / 77 / 78 |
| Mean accepted IoU | 0.753202 |
| Minimum accepted IoU | 0.503474 |
| Maximum accepted IoU | 0.948754 |
| Accounting validation | PASS |
| Cross-frame, cross-class, or duplicate assignment errors | 0 |

The observed accounting is `223 + 105 = 328` GT observations and `223 + 273 = 496` prediction observations.
