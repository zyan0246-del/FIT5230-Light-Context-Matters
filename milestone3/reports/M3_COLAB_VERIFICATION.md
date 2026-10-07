# Fresh Colab verification

Verification date: 7 October 2026

Notebook: `milestone3/notebooks/Milestone3.ipynb`

Runtime: fresh Google Colab CPU, Python 3.13.16 on Linux

## Outcome

- All eight code cells completed without an exception.
- All 32 regression tests passed; one pandas `FutureWarning` did not affect execution.
- The 24 public clips downloaded and standardised successfully.
- The fixed split remained 12 train, 6 validation and 6 test clips, balanced by class.
- All 24 clips achieved MediaPipe landmark coverage 1.0.
- The private Dark.Mimic rerun remained disabled as designed; SHA256-linked aggregate results displayed successfully.

## Cross-environment metrics

| Environment | Model | Macro-F1 | AUROC |
|---|---|---:|---:|
| Saved reference run, Python 3.12/macOS | Context-free | 0.486 | 0.667 |
| Saved reference run, Python 3.12/macOS | Weak context | 0.486 | 0.556 |
| Fresh Colab run, Python 3.13/Linux | Context-free | 0.486 | 0.556 |
| Fresh Colab run, Python 3.13/Linux | Weak context | 0.486 | 0.444 |

In both environments, weak context changed macro-F1 by 0.000 and AUROC by -0.111. The substantive conclusion is therefore stable: this weak proxy implementation does not improve the six-clip pilot.

The absolute AUROC shift is expected to be treated as environment sensitivity, not as a second independent experiment. FFmpeg builds, decoded pixels and package versions differ, and AUROC ranks are unstable with only three positive and three negative test examples. The report and presentation use the saved reference-run values and disclose the fresh-Colab values.
