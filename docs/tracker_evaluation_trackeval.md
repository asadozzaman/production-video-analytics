# Official TrackEval metrics for the 120-frame traffic window

This step calls the official [TrackEval metric implementations](https://github.com/JonathonLuiten/TrackEval/tree/12c8791b303e0a0b50f753af204249e622d0281a/trackeval/metrics) directly: `HOTA`, `CLEAR`, and `Identity`. The dependency is pinned in `requirements.txt` to official Git commit `12c8791b303e0a0b50f753af204249e622d0281a` (package version `1.0.dev1`). The code does not use TrackEval's pedestrian-oriented MOTChallenge dataset wrapper, and it does not change TrackEval's metric source or formulas.

```powershell
& 'env/Scripts/python.exe' -m pip install -r requirements.txt
& 'env/Scripts/python.exe' -m src.evaluation_trackeval --ground-truth data/ground_truth/traffic_tracker_ground_truth_v1.zip --predictions outputs/tracker_evaluation/predictions_100_219.json --output outputs/tracker_evaluation/trackeval_100_219.json
```

The adapter reads the existing CVAT and saved prediction records. For each class and each original source frame it passes compact, zero-based GT and tracker IDs, and a full same-frame, same-class XYXY IoU matrix, to TrackEval. IoUs are **not thresholded before** the metric calls. Stored coordinates are used without image clipping. Each class has 120 timesteps: TrackEval timestep 0 maps to source frame 100, and timestep 119 maps to source frame 219. The JSON saves the complete `timestep_to_source_frame` array.

CLEAR and Identity use an explicit similarity threshold of 0.5. HOTA uses its official alpha thresholds, 0.05 through 0.95. For each HOTA summary scalar below, the adapter takes the arithmetic mean of the official result array over those 19 alpha thresholds, exactly as TrackEval's `_summary_row` convention does. The JSON stores both these scalars and all alpha curves. Values below are raw fractions; TrackEval's printed table multiplies floating summaries by 100. `HOTA(0)` is a separate official field and is **not** used as the HOTA summary.

This pinned upstream code uses the removed NumPy names `np.float` and `np.int`. A scoped compatibility context supplies those aliases only during official metric calls and removes them afterward. It does not change any metric formula. TrackEval may print an unrelated optional BURST/pycocotools import warning; BURST is not used here.

## Actual per-class results

| Class | GT / predictions | HOTA | DetA | AssA | LocA | DetRe | DetPr | AssRe | AssPr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| person | 84 / 307 | 0.2339 | 0.1505 | 0.3644 | 0.7514 | 0.5821 | 0.1593 | 0.6070 | 0.3994 |
| car | 124 / 105 | 0.4330 | 0.4417 | 0.4275 | 0.7813 | 0.5267 | 0.6221 | 0.4701 | 0.6874 |
| truck | 120 / 84 | 0.3902 | 0.5309 | 0.2890 | 0.8322 | 0.5623 | 0.8033 | 0.2964 | 0.8373 |

| Class | MOTA | MOTP | CLR_Re | CLR_Pr | IDSW | Frag | MT | PT | ML | CLR_TP | CLR_FN | CLR_FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| person | -2.0357 | 0.6873 | 0.8095 | 0.2215 | 0 | 0 | 2 | 2 | 0 | 68 | 16 | 239 |
| car | 0.3790 | 0.7517 | 0.6210 | 0.7333 | 2 | 2 | 0 | 3 | 0 | 77 | 47 | 28 |
| truck | 0.5750 | 0.8121 | 0.6500 | 0.9286 | 3 | 1 | 0 | 1 | 0 | 78 | 42 | 6 |

| Class | IDF1 | IDP | IDR | IDTP | IDFN | IDFP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| person | 0.3478 | 0.2215 | 0.8095 | 68 | 16 | 239 |
| car | 0.5764 | 0.6286 | 0.5323 | 66 | 58 | 39 |
| truck | 0.4314 | 0.5238 | 0.3667 | 44 | 76 | 40 |

The negative person MOTA is the official CLEAR result for this prediction set; it is not clamped.

## Official class combinations and diagnostic cross-check

These combinations call TrackEval's own `combine_classes_class_averaged` and `combine_classes_det_averaged` methods, respectively.

| TrackEval combination | HOTA | IDF1 | MOTA | IDSW | Frag |
| --- | ---: | ---: | ---: | ---: | ---: |
| Class-averaged | 0.3524 | 0.4519 | -0.3606 | 5 | 3 |
| Detection-averaged | 0.3329 | 0.4320 | -0.1677 | 5 | 3 |

The earlier association diagnostics report 3 direct identity-change candidates, 15 association gaps, 2 reacquisitions with a different ID, and 4 with the same ID. TrackEval CLEAR reports **IDSW 5** and **Frag 3**. Their definitions and matching decisions differ; the diagnostic event counts are not official IDSW or Frag values.

Input and output validation passed for all three classes and 120 timesteps. The generated `outputs/tracker_evaluation/trackeval_100_219.json` is ignored by Git and parses back successfully.
