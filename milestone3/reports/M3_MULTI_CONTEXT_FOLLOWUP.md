# M3 exploratory improvement: multi-event context residuals

## Why this follow-up was added

The original Context v1 used one approximate vowel interval per clip and a large set of direct before/during/after interactions. On the six-clip held-out split it did not improve the context-free model. This follow-up tests whether using more of the available speech and a lower-dimensional mismatch representation is more promising.

This experiment was designed **after the original test result was known**. It is therefore post-hoc exploratory evidence. The original six test clips are not a fresh confirmatory test for this new method.

## Method

For every clip, the analysis:

1. reuses Whisper `tiny.en` word timestamps and CMUdict pronunciations;
2. retains central-vowel proxies with word confidence at least 0.80 and valid 0.20-second context on each side; if a clip has no usable event, it retries once at 0.60 and records the clip ID and threshold;
3. robustly normalises mouth aperture and width within each clip;
4. learns expected central mouth geometry from neighbouring broad phone classes and before/after geometry, using only real training clips;
5. converts each event into context-prediction residuals and transition summaries;
6. aggregates all events to one feature row per clip before classification.

Events are never treated as independent detection samples. The detector is trained and scored at clip level. In every cross-validation fold, the expected-context model is refitted using only real clips from that fold's training partition.

Fixed parameters are recorded in `artifacts/m3_improvement/multi_context_summary.json`: seed 5230, primary/fallback word confidence 0.80/0.60, Ridge alpha 10, logistic-regression C 0.01, four folds and 20 repeats. The fallback depends only on event availability, never labels or evaluation scores. A clip still fails explicitly if neither threshold yields an event. Low-confidence events remain a methodological limitation.

## Coverage

- 547 eligible vowel events across all 24 clips;
- median 20 events per clip;
- range 1 to 47 events per clip;
- all 24 clips retained.
- no fallback clips in the saved local reference run; fresh Colab runs record any fallback IDs separately.

The 15 October fresh Colab run retained 567 events (median 21.5) and used the 0.60 fallback for `fake_hallo_00023`. Thus, the event set is not identical across environments even though the selection rule is fixed.

This is a substantial coverage improvement over Context v1's one interval per clip, but the alignment is still a word-timestamp plus uniform-phone proxy rather than forced alignment.

## Results

### Original fixed split

| Model | Test clips | Macro-F1 | AUROC |
|---|---:|---:|---:|
| Context-free | 6 | 0.486 | 0.667 |
| Single weak context | 6 | 0.486 | 0.556 |
| Multi-event context residual | 6 | 0.829 | 0.778 |

These are the **saved local** values. The fixed split looked encouraging locally, but it was already visible before this follow-up was designed, so it is not independent confirmation. A fresh Colab CPU run gave context-free 0.486/0.556 and multi-event 0.486/0.444 on the same six test clips; the local gain was not reproduced.

### Stability check: 20 repeated four-fold runs

| Model | Mean Macro-F1 | SD | Mean AUROC | SD |
|---|---:|---:|---:|---:|
| Context-free | 0.655 | 0.205 | 0.678 | 0.204 |
| Multi-event context residual | 0.645 | 0.177 | 0.689 | 0.198 |

The paired mean change was -0.010 Macro-F1 and +0.011 AUROC. The improved method beat the baseline in only 46.3% of folds for Macro-F1 and 47.5% for AUROC. Therefore, the apparent gain on the original split is not yet stable.

The fresh Colab repeated-CV means were 0.643/0.688 for context-free and 0.585/0.621 for multi-event (Macro-F1/AUROC), paired mean changes -0.057/-0.067. Both metrics were worse there. See [M3_COLAB_VERIFICATION.md](M3_COLAB_VERIFICATION.md) for the runtime comparison.

This comparison changes more than phonetic context. The context-free classifier uses logistic-regression `C=1.0`, while the multi-event classifier uses `C=0.01`; the number of speech windows and feature representation also change. Consequently, the comparison does not isolate the effect of correct neighbouring phonemes. A same-regularisation, no-phone and shuffled-phone ablation is needed before attributing any gain to coarticulation.

The expected-context Ridge model predicts its own real training clips in sample, but predicts validation and test clips out of sample. That can make training-real residuals artificially small. Speaker-level cross-fitting for training residuals is a necessary next check. Neither this limitation nor the source-collection confound is repaired by repeated folds alone.

## Correct conclusion

The upgrade succeeds as an engineering and feasibility experiment: it uses much more speech, learns an explicit real-speech context expectation and preserves clip-level evaluation. It produced a local split gain that did not reproduce in Colab. It does **not** establish a reliable accuracy improvement.

For M3, present it as:

> A post-hoc multi-event extension improved the saved local six-clip split, but not the fresh Colab rerun. Repeated cross-validation did not show a consistent gain. This narrows the next step to forced alignment and a new matched hold-out dataset rather than further tuning the same 24 clips.

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
