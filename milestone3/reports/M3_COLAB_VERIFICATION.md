# Fresh Colab verification

The 15 October run verified the **previous** nine-code-cell public notebook from a reset Colab CPU runtime. The notebook was opened from the public GitHub URL. Its output differs materially from the saved local reference; both results are retained and labelled. The 16 October revision adds the matched no-phone ablation and public benign-codec controls and has been executed locally, but **has not yet been verified in fresh Colab**. Do not present this page as verification of the revised notebook.

Verification dates: 7 and 15 October 2026

Notebook: `milestone3/notebooks/Milestone3.ipynb`

Runtime: fresh Google Colab CPU, Python 3.13.16 on Linux

## Outcome

- All nine code cells completed without an exception on 15 October.
- All 37 regression tests passed; one pandas `FutureWarning` did not affect execution.
- The 24 public clips downloaded and standardised successfully.
- The fixed split remained 12 train, 6 validation and 6 test clips, balanced by class.
- All 24 clips achieved MediaPipe landmark coverage 1.0.
- The private Dark.Mimic rerun remained disabled as designed; SHA256-linked aggregate results displayed successfully.
- The post-hoc multi-event cell finished with 567 events, retaining all 24 clips. `fake_hallo_00023` used the documented 0.60 ASR-confidence fallback because no event met 0.80 in that runtime.

## Cross-environment metrics

| Environment | Model | Macro-F1 | AUROC |
|---|---|---:|---:|
| Saved reference run, Python 3.12/macOS | Context-free | 0.486 | 0.667 |
| Saved reference run, Python 3.12/macOS | Weak context | 0.486 | 0.556 |
| Fresh Colab, 7 October, eight cells | Context-free | 0.486 | 0.556 |
| Fresh Colab, 7 October, eight cells | Weak context | 0.486 | 0.444 |
| Fresh Colab, 15 October, nine cells | Context-free | 0.486 | 0.556 |
| Fresh Colab, 15 October, nine cells | Weak context | 0.486 | 0.222 |
| Fresh Colab, 15 October, nine cells | Multi-event context | 0.486 | 0.444 |

Macro-F1 stayed at 0.486 for the original two models in all three runs. Weak context reduced AUROC by 0.111 locally, 0.111 in the 7 October Colab run, and 0.333 in the 15 October Colab run. The substantive negative conclusion is stable, not the exact ranking score.

The 15 October multi-event result did **not** reproduce the local improvement. Saved local test Macro-F1/AUROC was 0.829/0.778, whereas fresh Colab gave 0.486/0.444, compared with its context-free 0.486/0.556. Local repeated four-fold mean changes were -0.010 Macro-F1 and +0.011 AUROC; Colab changes were -0.057 and -0.067. Therefore, no reliable gain has been established.

This is a cross-environment reproducibility check of the earlier revision, not an independent dataset. FFmpeg/codec decoding, ASR output and package versions can alter event selection and features; the ASR fallback makes one such difference visible. With only three real and three fake test clips, AUROC ranks are highly unstable. The final presentation plots saved local results and labels the earlier fresh-Colab disagreement on the relevant slides. The updated source prints a runtime-specific interpretation rather than claiming a gain unconditionally; its new ablation must be rerun in Colab separately.
