# Tracker failure evidence and benchmark audit

This step reads the verified GT, prediction, matching, diagnostics, and TrackEval artifacts and decodes the original `test_traffic.mp4`. It does **not** rerun YOLO, ByteTrack, matching, or formal metrics. Video decoding starts at source frame 0 and advances sequentially; a file named `source_0120.jpg` is rendered from original video frame 120, not CVAT-local frame 20.

```powershell
& 'env/Scripts/python.exe' -m src.evaluation_evidence --video data/input/test_traffic.mp4 --ground-truth data/ground_truth/traffic_tracker_ground_truth_v1.zip --predictions outputs/tracker_evaluation/predictions_100_219.json --matches outputs/tracker_evaluation/matches_100_219.json --diagnostics outputs/tracker_evaluation/diagnostics_100_219.json --trackeval outputs/tracker_evaluation/trackeval_100_219.json --output-dir outputs/tracker_evaluation/evidence
```

The output is under the Git-ignored `outputs/tracker_evaluation/evidence/` directory. `frames/` contains annotated source frames, `crops/` contains selected unmatched prediction crops, and `evidence_manifest.json` is the audit index. Each item records its type, original source frame, class, relevant GT and prediction IDs, confidence and IoU where applicable, image and crop paths, selection reason, and any context frames. Paths inside the manifest are relative to the evidence directory. The manifest also records SHA-256 hashes of the six source inputs.

## Deterministic selection

- **Unmatched GT:** first, middle, and last observation in source-frame order for each class, up to three per class.
- **Unmatched person predictions:** split source frames 100–219 into three equal time bins and the 239 person-observation confidences into terciles. Select one observation nearest each populated bin's time center and confidence median. Also include the highest-confidence person observation from the earliest frame with the maximum unmatched-person count, unless already selected.
- **Other unmatched predictions:** first, middle, and last observation per class, up to three each.
- **Direct identity-change candidates:** every event, with its previous matched frame as context.
- **Different-ID reacquisitions:** every event, with the matched frame before the gap and a visible gap midpoint as context.
- **Association gaps:** the three longest by visible GT observation count, with available before, start, middle, end, and after frames.

Selected items share a rendered frame when they refer to the same source frame. This avoids exporting hundreds of near-duplicates. The frame overlay follows the saved matching decision: green = accepted GT, cyan = accepted prediction, red = unmatched GT, orange = unmatched prediction. Labels show class and GT/prediction ID; predictions show confidence and accepted pairs show IoU. Prediction crops use the **original decoded frame**. Crop pixel bounds are clipped to the image when a raw tracker box extends beyond it; the stored XYXY values and matching coordinate policy are unchanged. A box entirely outside the image gets no crop and a manifest note.

## Actual evidence run

| Audit item | Count |
| --- | ---: |
| Distinct annotated source frames | 37 |
| Representative unmatched GT items | 9 |
| Representative unmatched prediction items | 16 |
| Of these, unmatched person prediction items | 10 |
| Direct identity-change events covered | 3 of 3 |
| Different-ID reacquisitions covered | 2 of 2 |
| Longest association gaps covered | 3 of 15 |
| Crops saved | 16 |
| Manifest validation | PASS |

The direct changes include car GT track 1 at frames 159→160 and truck GT track 3 at 159→160 and 161→162. The different-ID reacquisitions are car GT track 1 at frame 151 after its 145–150 gap, and truck GT track 3 at frame 159 after its 150–158 gap. The selected longest gaps have lengths 15, 11, and 10 visible GT observations.

The official TrackEval result lists 239 person `CLR_FP` observations. The ten person crops are a bounded inspection sample of unmatched prediction observations, not a root-cause classification or a replacement for the formal count. Human review should inspect the evidence before drawing conclusions.
