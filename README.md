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
- [Historical Colab verification](milestone3/reports/M3_COLAB_VERIFICATION.md)

Strategy A compares mouth-dynamics representations on 36 matched real/SadTalker pairs from 12 speakers. The expanded experiment supports the simpler global representation; compact windows and weak phonetic context did not provide a stable advantage on the new segments.

Strategy B audits periodic colour differences in one Dark.Mimic control/modified pair. Transformation controls preserve the observed trace, but synthetic periodic light also triggers the metric. The branches remain separate; these pilots do not establish a general real/fake detector or calibrated fusion model.

The final notebook contains **11 code cells with saved expanded-server outputs and 46 passing regression tests**. The recorded fresh Colab verification applies to the historical nine-cell revision tested on 16 October 2026, with 38 passing tests. A fresh full Colab run of the final eleven-cell revision is not documented. The separate Codex-assisted Windows Strategy B rerun completed its workflows with numerical differences from the reference run.

## Earlier work and data

The original Milestone 1 files remain under [milestone1](milestone1/). Raw private peer videos are excluded. Hash-linked aggregate results are available in the M3 notebook and project records; rerunning the private-pair branch requires the original verified files.
