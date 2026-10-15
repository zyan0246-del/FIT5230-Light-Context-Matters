# FIT5230 Light: Context Matters

Team **Light.Ningyizhuo_No.1**

Members: **Yan Zhiheng and He Hongjie**

Theme 3: Speech-to-Face, Light side

This repository preserves the original Milestone 1 work and contains the Milestone 3 course pilot. The public notebook is the main reproducible submission; the slides summarise the two member-owned strategies.

## Milestone 3 outcome

We evaluate two independent defensive signals:

1. **Yan: context-conditioned lip dynamics.** A 24-clip public pilot compares a context-free mouth-dynamics baseline with a weak phonetic-context version. The first context version did not improve the six-clip held-out result. A later multi-event extension improved that already-viewed split, but 20 repeated four-fold comparisons showed no stable average gain. This follow-up is exploratory.
2. **He: periodic chromatic-injection forensics.** A paired audit detects a strong colour trace near the peer-reported 1.2 Hz setting in one Dark.Mimic A=1 pair. The trace remains above a neutral re-encode control after H.264 CRF 35, resizing and 15 FPS.

These are course-pilot findings. The repository does not claim forced phoneme alignment, full PIA reproduction, physiological waveform recovery or population-level detector accuracy.

The current notebook has nine code cells with locally saved outputs, and 37 regression tests pass. A fresh Colab run also completed all nine cells, but did not reproduce the local multi-event gain. See the [Colab verification record](milestone3/reports/M3_COLAB_VERIFICATION.md) for both results and their limits.

## Start here

- [Executed Milestone 3 notebook](milestone3/notebooks/Milestone3.ipynb)
- [Open the notebook in Google Colab](https://colab.research.google.com/github/zyan0246-del/FIT5230-Light-Context-Matters/blob/main/milestone3/notebooks/Milestone3.ipynb)
- [Milestone 3 results](milestone3/reports/M3_RESULTS.md)
- [15-minute presentation script](milestone3/reports/M3_PRESENTATION_SCRIPT.md)
- [Final editable PowerPoint presentation](milestone3/deliverables/FIT5230_M3_Context_Matters_Presentation_Final_v4.pptx)
- [Moodle Colab-link text file](milestone3/deliverables/Moodle_M3_Colab_Link.txt)
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
