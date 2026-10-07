# M3 exploratory improvement: multi-event context residuals

## Why this follow-up was added

The original Context v1 used one approximate vowel interval per clip and a large set of direct before/during/after interactions. On the six-clip held-out split it did not improve the context-free model. This follow-up tests whether using more of the available speech and a lower-dimensional mismatch representation is more promising.

This experiment was designed **after the original test result was known**. It is therefore post-hoc exploratory evidence. The original six test clips are not a fresh confirmatory test for this new method.

## Method

For every clip, the analysis:

1. reuses Whisper `tiny.en` word timestamps and CMUdict pronunciations;
2. retains all central-vowel proxies with word confidence at least 0.80 and valid 0.20-second context on each side;
3. robustly normalises mouth aperture and width within each clip;
4. learns expected central mouth geometry from neighbouring broad phone classes and before/after geometry, using only real training clips;
5. converts each event into context-prediction residuals and transition summaries;
6. aggregates all events to one feature row per clip before classification.

Events are never treated as independent detection samples. The detector is trained and scored at clip level. In every cross-validation fold, the expected-context model is refitted using only real clips from that fold's training partition.

Fixed parameters are recorded in `artifacts/m3_improvement/multi_context_summary.json`: seed 5230, minimum word confidence 0.80, Ridge alpha 10, logistic-regression C 0.01, four folds and 20 repeats.

## Coverage

- 547 eligible vowel events across all 24 clips;
- median 20 events per clip;
- range 1 to 47 events per clip;
- all 24 clips retained.

This is a substantial coverage improvement over Context v1's one interval per clip, but the alignment is still a word-timestamp plus uniform-phone proxy rather than forced alignment.

## Results

### Original fixed split

| Model | Test clips | Macro-F1 | AUROC |
|---|---:|---:|---:|
| Context-free | 6 | 0.486 | 0.667 |
| Single weak context | 6 | 0.486 | 0.556 |
| Multi-event context residual | 6 | 0.829 | 0.778 |

The fixed split is encouraging. However, it was already visible before this follow-up was designed, so this comparison is not independent confirmation.

### Stability check: 20 repeated four-fold runs

| Model | Mean Macro-F1 | SD | Mean AUROC | SD |
|---|---:|---:|---:|---:|
| Context-free | 0.655 | 0.205 | 0.678 | 0.204 |
| Multi-event context residual | 0.645 | 0.177 | 0.689 | 0.198 |

The paired mean change was -0.010 Macro-F1 and +0.011 AUROC. The improved method beat the baseline in only 46.3% of folds for Macro-F1 and 47.5% for AUROC. Therefore, the apparent gain on the original split is not yet stable.

## Correct conclusion

The upgrade succeeds as an engineering and feasibility experiment: it uses much more speech, learns an explicit real-speech context expectation, preserves clip-level evaluation and produces a promising result on the original split. It does **not** yet establish a reliable accuracy improvement.

For M3, present it as:

> A post-hoc multi-event extension improved the original six-clip split, but repeated cross-validation did not show a stable average gain. This narrows the next step to forced alignment and a new matched hold-out dataset rather than further tuning the same 24 clips.

Do not present `0.829` as a general detector accuracy or claim that context is proven effective.

## Reproduce

Run after the main landmark and ASR cells:

```bash
python scripts/run_multi_context_improvement.py
```

Key outputs:

- `artifacts/m3_improvement/multi_context_improvement.png`
- `artifacts/m3_improvement/fixed_split_metrics.csv`
- `artifacts/m3_improvement/fixed_split_predictions.csv`
- `artifacts/m3_improvement/repeated_cv_summary.csv`
- `artifacts/m3_improvement/repeated_cv_fold_metrics.csv`
- `artifacts/m3_improvement/multi_context_summary.json`
