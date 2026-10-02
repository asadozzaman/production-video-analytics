# Tracker association diagnostics

This milestone reads the validated `matches_100_219.json` artifact and builds one chronological timeline per visible CVAT GT track. Each timeline entry is an original source frame with an accepted prediction ID and IoU, or `null` when that visible GT observation was unmatched. Runs and gap lengths count **visible GT observations**. A frame where the GT object is absent is not inserted as an unmatched observation, and leaving the evaluation range creates no event.

```powershell
& 'env/Scripts/python.exe' -m src.evaluation_diagnostics --matches outputs/tracker_evaluation/matches_100_219.json --output outputs/tracker_evaluation/diagnostics_100_219.json
```

The events have deliberately narrow meanings:

- **Direct identity-change candidate:** adjacent visible GT observations are both matched, but their prediction IDs differ.
- **Association gap:** a maximal run of unmatched visible GT observations. It records the start/end source frames, the number and list of visible GT frames, and prediction IDs before and after when present.
- **Reacquisition identity-change candidate:** a matched observation, then a real association gap, then a match with a different prediction ID.
- **Reacquisition same identity:** the same sequence ends with the previous prediction ID.

These are association diagnostics from a fixed IoU matching result. They are not official ID switches, HOTA, IDF1, or other tracking benchmark metrics. Numeric prediction IDs are interpreted only within each GT track's timeline and its class-aware matches.

The CLI validates the artifact's 120 ordered source frames, matching summary, unique assignments, classes, boxes, and IoUs before analysis. It then validates timeline reconstruction, event placement and uniqueness, and observation accounting. The runtime JSON under `outputs/` is ignored by Git.

## Validated result

| Diagnostic | Result |
| --- | ---: |
| Evaluation source frames | 100–219 (120) |
| GT tracks | 8 |
| Visible GT / matched / unmatched GT observations | 328 / 223 / 105 |
| Unmatched prediction observations | 273 |
| GT tracks with zero identity-change candidates | 6 |
| GT tracks using more than one prediction ID | 2 |
| Direct identity-change candidates | 3 |
| Association gaps | 15 |
| Reacquisitions with a different ID | 2 |
| Reacquisitions with the same ID | 4 |
| Longest association gap | 15 visible GT frames |
| Validation | PASS |

| GT track | Class | Visible | Matched | Unmatched | Match ratio | Prediction IDs | Direct | Gaps | Reacquired: different / same |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 0 | car | 34 | 26 | 8 | 0.765 | 14 | 0 | 1 | 0 / 0 |
| 1 | car | 38 | 24 | 14 | 0.632 | 11, 20, 24 | 1 | 2 | 1 / 0 |
| 2 | car | 52 | 27 | 25 | 0.519 | 26 | 0 | 3 | 0 / 1 |
| 3 | truck | 120 | 78 | 42 | 0.650 | 11, 24 | 2 | 4 | 1 / 3 |
| 4 | person | 23 | 18 | 5 | 0.783 | 18 | 0 | 1 | 0 / 0 |
| 5 | person | 21 | 17 | 4 | 0.810 | 19 | 0 | 2 | 0 / 0 |
| 6 | person | 20 | 19 | 1 | 0.950 | 16 | 0 | 1 | 0 / 0 |
| 7 | person | 20 | 14 | 6 | 0.700 | 10 | 0 | 1 | 0 / 0 |

Unmatched predictions by class: person 239, car 28, truck 6. They occur in 105 source frames; the maximum in one frame is 6. The report does not classify these as formal false positives.
