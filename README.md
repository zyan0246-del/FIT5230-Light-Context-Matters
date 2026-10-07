# FIT5230 Light: Context Matters

Team **Light.Ningyizhuo_No.1**

Members: **Yan Zhiheng and He Hongjie**

Theme 3: Speech-to-Face, Light side

This repository preserves the original Milestone 1 work and adds the complete Milestone 3 course pilot.

## Milestone 3 outcome

We evaluate two independent defensive signals:

1. **Yan: context-conditioned lip dynamics.** A 24-clip public pilot compares a context-free mouth-dynamics baseline with a weak phonetic-context version. On six held-out clips, weak context did not improve the result.
2. **He: periodic chromatic-injection forensics.** A paired audit detects a strong colour trace near the peer-reported 1.2 Hz setting in one Dark.Mimic A=1 pair. The trace remains above a neutral re-encode control after H.264 CRF 35, resizing and 15 FPS.

These are course-pilot findings. The repository does not claim forced phoneme alignment, full PIA reproduction, physiological waveform recovery or population-level detector accuracy.

A fresh Google Colab CPU run on 7 October 2026 completed all eight code cells and all 32 tests. Small AUROC changes across macOS/Python 3.12 and Colab/Linux/Python 3.13 left the central finding unchanged; see the [Colab verification record](milestone3/reports/M3_COLAB_VERIFICATION.md).

## Start here

- [Executed Milestone 3 notebook](milestone3/notebooks/Milestone3.ipynb)
- [Milestone 3 results](milestone3/reports/M3_RESULTS.md)
- [15-minute presentation script](milestone3/reports/M3_PRESENTATION_SCRIPT.md)
- [Editable PowerPoint presentation](milestone3/deliverables/FIT5230_M3_Context_Matters_Presentation.pptx)
- [Submission checklist](milestone3/reports/M3_SUBMISSION_CHECKLIST.md)
- [Fresh Colab verification](milestone3/reports/M3_COLAB_VERIFICATION.md)
- [Milestone 3 source and reproduction guide](milestone3/README.md)

## Reproduce the public pilot

```bash
cd milestone3
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_m3_pilot.py
python -m pytest -q
```

The public notebook contains an inspectable source snapshot and saved outputs. A fresh run downloads the public pilot clips and models. Raw public videos, model weights and generated working outputs are excluded from Git.

## Private peer material

Raw Dark.Mimic MP4 files are not redistributed. Aggregate results and SHA256 provenance are available under `milestone3/artifacts/m3_results/`. Team members with the verified files can use the guarded private rerun cell in the notebook.

## Earlier work

The original Milestone 1 notebook, documentation and demonstration outputs remain under [`milestone1/`](milestone1/).

## Responsible use

This project is defensive media-forensics coursework. Use only consented or appropriately licensed recordings. Do not commit credentials, personal videos or private peer media.
