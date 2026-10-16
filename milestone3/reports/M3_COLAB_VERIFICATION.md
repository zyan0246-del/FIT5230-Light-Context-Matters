# Fresh Colab verification

The current nine-code-cell public notebook was opened directly from GitHub commit `5b139a3` and run from a fresh Google Colab CPU runtime on **16 October 2026**. Its embedded readable-source snapshot had SHA256 `508b11a432aa6e4a8f1fb910a6bb17dcb2f29fcbf5433a15715ed38ac6b3174c`. This is the revision containing the matched no-phone ablation, 12 public codec controls and synthetic periodic-light controls.

Notebook: `milestone3/notebooks/Milestone3.ipynb`

Runtime: Python 3.13.16, Linux 6.6.122 x86_64, Google Colab CPU

Key packages: NumPy 2.1.3, pandas 2.2.3, SciPy 1.16.3, scikit-learn 1.6.1, Matplotlib 3.10.0, MediaPipe 1.1.1, OpenCV 4.14.0.94

## Execution outcome

- All nine code cells completed without an exception.
- All 38 regression tests passed. One pandas `FutureWarning` did not affect execution.
- The 24 public clips downloaded and standardised successfully: 12 train, 6 validation and 6 test, balanced by class.
- All 24 clips achieved MediaPipe landmark coverage 1.0.
- The private Dark.Mimic rerun remained disabled as designed; the SHA256-linked aggregate tables and public controls displayed successfully.
- The multi-event run retained 567 events across all 24 clips. `fake_hallo_00023` used the documented 0.60 ASR-confidence fallback because no event met 0.80 in this runtime.
- The chromatic-control cell confirmed that synthetic 1.2 Hz global light exceeded its neutral counterpart in 12/12 real-speaker clips.

## Current-runtime metrics

| Comparison | Model | Macro-F1 | AUROC |
|---|---|---:|---:|
| Original six-clip test | Context-free | 0.486 | 0.556 |
| Original six-clip test | Weak context | 0.486 | 0.222 |
| Multi-event six-clip test | Multi-event context | 0.486 | 0.444 |
| Matched six-clip ablation | Multi-event no-phone | 0.486 | 0.444 |
| 20 repeated four-fold comparisons | Context-free | 0.643 | 0.688 |
| 20 repeated four-fold comparisons | Multi-event context | 0.585 | 0.621 |
| Matched 20 repeated four-fold ablation | Multi-event context | 0.585 | 0.621 |
| Matched 20 repeated four-fold ablation | Multi-event no-phone | 0.593 | 0.638 |

The weak context proxy reduced AUROC by 0.333 without changing Macro-F1. The post-hoc multi-event method did not improve both held-out metrics: its change from context-free was +0.000 Macro-F1 and -0.111 AUROC. Across the repeated folds, its mean changes were -0.057 Macro-F1 and -0.067 AUROC. The matched no-phone ablation also slightly outperformed the phone-context version on both repeated-fold means in this runtime. Therefore, this experiment does **not** establish a reliable phonetic-context advantage.

## Cross-environment interpretation

The saved local Python 3.12 replay gave an encouraging known-split result for multi-event context (0.829/0.778) and a mixed repeated-fold comparison. The fresh Colab result above did not reproduce that split gain. FFmpeg decoding, ASR output and dependency versions can alter event selection and features; with only three real and three fake test clips, AUROC rankings are especially unstable. The defensible conclusion is a functioning, reproducible experimental pipeline with a negative/unstable context result—not an effective new detector. The next research step is forced alignment plus a new matched hold-out set, not further tuning on these 24 clips.

The GitHub notebook already contains nine saved local outputs. The fresh Colab run was verified interactively against the same embedded source snapshot; direct Colab-to-GitHub saving was unavailable because Colab requested a separate GitHub OAuth authorization. This report therefore records the fresh runtime, exact source hash and generated metrics without relabelling the saved local outputs as Colab outputs.
