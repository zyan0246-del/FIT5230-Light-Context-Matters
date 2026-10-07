# Context Matters — FIT5230 Milestone 3

**Light.Ningyizhuo_No.1 · Yan Zhiheng and He Hongjie · Theme 3 / Light**

This release evaluates two complementary defenses against speech-to-face deepfakes:

1. **Yan — context-conditioned lip dynamics.** A 24-clip public pilot compares a context-free mouth-dynamics baseline with a weak phonetic-context model.
2. **He — chromatic-injection forensics.** A paired audit checks whether Dark.Mimic's reported 1.2 Hz rPPG-evasion edit leaves a periodic colour trace, including a neutral re-encode control and three post-processing conditions.

The held-out public pilot contains six clips. The context-free model reached macro-F1 0.486 and AUROC 0.667; the weak-context model reached macro-F1 0.486 and AUROC 0.556. Thus, this pilot does **not** support a context improvement claim. In the single peer pair, red-channel target-band energy was 4.63 times the neutral control and remained higher after H.264 CRF 35, resizing, and 15 FPS conversion. This is descriptive evidence from one pair, not a population-level detector result.

## Evidence and deliverables

- [`notebooks/Milestone3.ipynb`](notebooks/Milestone3.ipynb): fully executed notebook with eight saved code-cell outputs.
- [`reports/M3_RESULTS.md`](reports/M3_RESULTS.md): metrics, uncertainty, failure cases, limitations and claim boundaries.
- [`deliverables/FIT5230_M3_Context_Matters_Presentation.pptx`](deliverables/FIT5230_M3_Context_Matters_Presentation.pptx): editable 11-slide presentation.
- [`reports/M3_PRESENTATION_SCRIPT.md`](reports/M3_PRESENTATION_SCRIPT.md): 15-minute speaking plan with member attribution.
- [`artifacts/m3_results/`](artifacts/m3_results/): frozen aggregate tables, reports and figures.
- [`reports/M3_SUBMISSION_CHECKLIST.md`](reports/M3_SUBMISSION_CHECKLIST.md): final manual steps.

## Reproduce the public experiment

Python 3.12 was used for the verified run. FFmpeg must be available on `PATH`.

```bash
python -m venv .venv
source .venv/bin/activate                 # macOS/Linux
# .venv\Scripts\Activate.ps1             # Windows PowerShell
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts/download_m3_pilot.py
python scripts/audit_m3_pilot.py
python -m s2f.cli extract-manifest data/m3_pilot_manifest.csv --out outputs/m3_work/pilot_features
python scripts/build_weak_phoneme_intervals.py
python scripts/summarize_lip_pilot.py
python -m pytest -q
```

The acquisition script downloads 12 licensed Wikitongues clips and 12 TalkingHeadBench generated clips, standardises them, and records source metadata, hashes and portable FFmpeg commands. Raw videos, model weights, caches and working outputs are excluded from Git. The committed notebook also embeds a source snapshot so its saved results remain inspectable.

## Data and evaluation controls

- Fixed seed: 5230.
- Fixed split: 12 train, 6 validation and 6 test clips, balanced by class.
- `source_id` and `speaker_id` cannot cross splits.
- The context-free and weak-context models use identical clips and split.
- Vocabulary and scaling are learned on training data only; threshold 0.5 is fixed.
- Phone intervals are weak proxies from Whisper `tiny.en` word timestamps plus CMUdict, not forced alignment.
- Aggregate peer results are public; raw DarkMimic MP4 files remain private and are represented by SHA256 provenance only.

## Run the notebook in Colab

Open [`notebooks/Milestone3.ipynb`](notebooks/Milestone3.ipynb) through the repository's Colab link and select **Runtime → Run all**. The public experiment can be rebuilt from downloads. The peer audit cell verifies local filenames and hashes and is skipped when the private files are absent; its frozen aggregate outputs remain available for inspection.

## Scope and claim boundary

PIA is the research reference baseline, not a reproduced result in this repository. CALS motivates the coarticulation hypothesis but is a generation paper, not a detector baseline. This course pilot does not claim forced alignment, physiological waveform recovery, successful score fusion, publication-level generalisation or that a negative weak-context result disproves coarticulation.

## References

- [PIA official implementation](https://github.com/skrantidatta/PIA) and [paper](https://arxiv.org/abs/2510.14241)
- [CALS paper](https://arxiv.org/abs/2305.19556)
- [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker/python)
- [TalkingHeadBench dataset card](https://huggingface.co/datasets/luchaoqi/TalkingHeadBench)

This is defensive media-forensics coursework. Use only consented or appropriately licensed media.
