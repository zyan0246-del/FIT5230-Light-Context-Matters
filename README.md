# FIT5230 Light: Context Matters

Team: **Light.Ningyizhuo_No.1**  
Members: **Yan Zhiheng and He Hongjie**  
Theme: Speech-to-Face, Light / Defense

This repository preserves the original Milestone 1 work and the Milestone 3 project. The M3 notebook is the reproducible submission entry point.

## Milestone 3

- [Open the notebook in Google Colab](https://colab.research.google.com/github/zyan0246-del/FIT5230-Light-Context-Matters/blob/main/milestone3/notebooks/Milestone3.ipynb)
- [Project guide and findings](milestone3/README.md)
- [Executed notebook](milestone3/notebooks/Milestone3.ipynb)
- [Results report](milestone3/reports/M3_RESULTS.md)
- [Colab verification records](milestone3/reports/M3_COLAB_VERIFICATION.md)

Strategy A compares mouth-dynamics representations on 36 matched real/SadTalker pairs from 12 speakers. The expanded experiment supports the simpler global representation; compact windows and weak phonetic context did not provide a stable advantage on the new segments.

Strategy B audits periodic colour differences in one Dark.Mimic control/modified pair. Transformation controls preserve the observed trace, but synthetic periodic light also triggers the metric. The branches remain separate; these pilots do not establish a general real/fake detector or calibrated fusion model.

The final notebook contains 11 code cells. On 22 October 2026, the supplied Google Colab run completed the default workflow with execution counts 1-11, 46 passing regression tests, one non-fatal pandas FutureWarning and no error outputs. Both `RUN_ORIGINAL_PUBLIC_PIPELINE` and `RUN_PRIVATE_PAIR` were `False`. This verifies environment setup, tests and the reading/display of saved result tables; it does not regenerate SadTalker videos, rerun public-video feature extraction or recompute the private peer audit. The current notebook preserves that Colab output; only documentation and display metadata were subsequently changed. [Verification details](milestone3/reports/M3_COLAB_VERIFICATION.md) distinguish this run from the historical nine-cell Colab experiment. The separate Codex-assisted Windows Strategy B rerun completed its workflows with numerical differences from the reference run.

## Earlier work and data

The original Milestone 1 files remain under [milestone1](milestone1/). Raw private peer videos are excluded. Hash-linked aggregate results are available in the M3 notebook and project records; rerunning the private-pair branch requires the original verified files.
