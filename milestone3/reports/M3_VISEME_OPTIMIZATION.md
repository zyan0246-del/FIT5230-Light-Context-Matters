# Expanded lip-context optimisation

## Decision

The final course model should use the 24-feature global lip-dynamics representation. The earlier improvement from compact local windows did not reproduce on the newly generated segments. Weak viseme context also failed to provide a stable advantage over either the numeric window or shuffled-viseme control.

This is a successful model-selection result rather than a positive context result: the expanded experiment identifies the simpler representation that is most stable under the controlled course setting.

## Data and quality control

- 12 licensed Wikitongues identities.
- Three non-overlapping 12-second segments per identity, starting at 5, 20 and 35 seconds.
- One real and one SadTalker clip per segment: 36 matched pairs and 72 clips.
- Each pair shares identity, decoded audio, duration, resolution, frame rate and final codec settings.
- All segments from one identity stay in one train/validation/test partition and one outer fold.
- Split counts are 36 train clips, 18 validation clips and 18 test clips, balanced by class.
- MediaPipe completed 72/72 clips with minimum landmark coverage 1.00 and no failed status.

The original s0 segment was the development set. The s1/s2 segments are non-overlapping confirmation segments. They reuse the same 12 identities and SadTalker generator, so they do not establish cross-identity or cross-generator generalisation.

## Representations

| Model | Inputs | Typical dimension |
|---|---|---:|
| Global | Whole-clip lip geometry and derivatives | 24 |
| Compact window | Global block plus 36 before/during/after trajectory summaries | 60 |
| Viseme context | Compact window plus coarse viseme indicators and selected transition interactions | 114 to 132 on the new segments |
| Shuffled viseme | Same model family, with viseme triples permuted between matched pairs | 114 to 132 |

Whisper tiny.en provides word timestamps and CMUdict provides pronunciations. Phones are placed uniformly within each word and mapped to coarse visible-articulation groups. These labels remain a weak proxy, not forced alignment.

## Training objective and evaluation

All models use balanced logistic regression. For probability `p` and binary label `y`, training minimises binary logistic loss with L2 regularisation:

`L = -[y log(p) + (1-y) log(1-p)] + L2 penalty`.

The evaluation uses 20 repeated outer four-fold splits grouped by `speaker_id`. An inner three-fold grouped loop selects `C` from `{0.001, 0.01, 0.1, 1.0}` by minimum validation log loss. The classification threshold remains 0.5. Macro-F1, AUROC and log loss are reported only from outer out-of-fold predictions.

The method and expanded data construction were committed as `eed6a3b` before the expanded scores were inspected.

## Results

### Development: original 12 pairs

| Model | Macro-F1 mean ± SD | AUROC mean ± SD | Log loss mean ± SD |
|---|---:|---:|---:|
| Global | 0.675 ± 0.053 | 0.763 ± 0.025 | 0.653 ± 0.043 |
| Compact window | **0.759 ± 0.051** | **0.818 ± 0.048** | **0.594 ± 0.083** |
| Viseme context | 0.677 ± 0.040 | 0.671 ± 0.042 | 0.687 ± 0.055 |
| Shuffled viseme | 0.684 ± 0.055 | 0.719 ± 0.053 | 0.680 ± 0.079 |

Compact windows exceeded global features in 17/20 repeats for Macro-F1 and 18/20 for AUROC. This was development evidence and motivated the prespecified expansion.

### All 36 pairs

| Model | Macro-F1 mean ± SD | AUROC mean ± SD | Log loss mean ± SD |
|---|---:|---:|---:|
| Global | 0.753 ± 0.051 | **0.826 ± 0.026** | 0.553 ± 0.033 |
| Compact window | **0.753 ± 0.043** | 0.809 ± 0.029 | **0.550 ± 0.041** |
| Viseme context | 0.658 ± 0.031 | 0.726 ± 0.033 | 0.639 ± 0.041 |
| Shuffled viseme | 0.664 ± 0.034 | 0.722 ± 0.038 | 0.648 ± 0.055 |

Compact window minus global was +0.000 Macro-F1, -0.018 AUROC and -0.002 log loss on average. AUROC improved in only 8/20 repeats. Viseme context was worse than compact windows in 19/20 Macro-F1 repeats, 20/20 AUROC repeats and 20/20 log-loss repeats.

### Confirmation on only the new s1/s2 segments: 24 pairs

| Model | Macro-F1 mean ± SD | AUROC mean ± SD | Log loss mean ± SD |
|---|---:|---:|---:|
| Global | **0.762 ± 0.041** | **0.843 ± 0.027** | **0.498 ± 0.045** |
| Compact window | 0.677 ± 0.056 | 0.739 ± 0.035 | 0.608 ± 0.051 |
| Viseme context | 0.600 ± 0.027 | 0.671 ± 0.027 | 0.649 ± 0.026 |
| Shuffled viseme | 0.592 ± 0.043 | 0.647 ± 0.046 | 0.678 ± 0.068 |

Compact window minus global was -0.086 Macro-F1, -0.104 AUROC and +0.110 log loss. It improved AUROC in 0/20 repeats and improved Macro-F1 in only 1/20, with two ties. Viseme context remained below compact windows and did not consistently dominate shuffled visemes across all three metrics.

## Why the weak labels remain ineffective

1. Whisper supplies word-level timing, while phone boundaries are uniformly approximated inside each word.
2. Some segments contain little confident English ASR. One example produced only “the guitar” with word probabilities near 0.10.
3. Twelve identities still provide limited independent groups, even though there are more clips.
4. Local features add correlated dimensions that can fit segment-specific motion without improving speaker-grouped generalisation.
5. The viseme interaction block increases dimensionality beyond the 60-feature numeric model, while the category evidence remains noisy.

During the run, one short-ASR segment exposed a preprocessing bug: the last recognised word time was incorrectly used as the video boundary. The corrected code uses the landmark timeline as the video duration while retaining ASR-derived phone timestamps. No clip was deleted; the repaired pipeline completed all 72 clips and 46 tests.

## Course conclusion

Use global lip dynamics as Strategy A's final classifier. Present compact windows and viseme context as controlled ablations that were promising on the original 12 pairs but failed confirmation. This is stronger course evidence than selecting the favourable development result: it demonstrates complete execution, leakage-safe model selection, a negative control, failure analysis and a justified final decision.

The result does not show that natural coarticulation is useless. It shows that the current weak alignment and limited identity set do not support adding context features in this course experiment.
