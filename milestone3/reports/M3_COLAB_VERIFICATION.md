# Colab verification records

## Final eleven-cell default workflow - 22 October 2026

The supplied Colab notebook was reviewed after the user ran all cells. All eleven code cells have execution counts 1-11 and no error outputs. The test cell reports **46 passed, 1 warning in 3.35s**. The warning concerns future pandas downcasting behavior and did not fail execution. Runtime output identifies Python 3.13.15 and Linux 6.6.122, with tests running under `/content/m3_notebook_workspace/`. Code sources match the preceding final notebook exactly.

Both `RUN_ORIGINAL_PUBLIC_PIPELINE=False` and `RUN_PRIVATE_PAIR=False` were retained. Environment setup and regression tests ran; experiment tables and plots were read or recreated from committed reference artifacts. This is a successful **default-workflow check**, not a fresh end-to-end recomputation of video experiments. No new detector accuracy or chromatic-audit measurements are claimed.

The subsequent setup-heading, hidden-source and verification-note edits preserve every code cell, execution count and saved output from this Colab copy. The final notebook SHA256 is `827dc98fec8c795e2b67cf161c3f62618f11e2d8fd9697368b45ea020864e377`. Its embedded source snapshot SHA256 is `481a8a31a4487da47279d2614fabda2f9875517feca85b4e1ff60c39c1d1b304`. Machine-readable checks and environment versions are in [the default-run record](M3_COLAB_DEFAULT_RUN_20261022.json).

Some execution timestamp fields were inherited from the reference notebook and still refer to the earlier server run. They are not used to date this verification. The new environment output, `/content/` test path and test output distinguish the supplied Colab results from the server outputs.

### Runtime packages

| Package | Version |
|---|---|
| numpy | 2.1.3 |
| pandas | 2.2.3 |
| scipy | 1.16.3 |
| scikit-learn | 1.6.1 |
| matplotlib | 3.10.0 |
| mediapipe | 1.1.1 |
| opencv-contrib-python | 4.14.0.94 |
| nbformat | 5.11.1 |
| nbclient | 0.10.4 |

## Historical full public-branch execution

The record below concerns the earlier nine-cell revision. Its recomputed metrics are separate from the saved reference results displayed by the final default workflow.

# Historical Colab verification: 16 October 2026

The historical nine-code-cell public notebook revision tested on 16 October 2026 was opened directly from GitHub commit `5b139a3` and run from a fresh Google Colab CPU runtime on **16 October 2026**. Its embedded readable-source snapshot had SHA256 `508b11a432aa6e4a8f1fb910a6bb17dcb2f29fcbf5433a15715ed38ac6b3174c`. This is the revision containing the matched no-phone ablation, 12 public codec controls and synthetic periodic-light controls.

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

That historical revision contained nine saved local outputs. The fresh Colab run was verified interactively against the same embedded source snapshot; direct Colab-to-GitHub saving was unavailable because Colab requested a separate GitHub OAuth authorization. This report therefore records the fresh runtime, exact source hash and generated metrics without relabelling the saved local outputs as Colab outputs.
