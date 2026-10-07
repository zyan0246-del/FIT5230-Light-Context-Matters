# Context Matters — FIT5230 Milestone 3

**Light.Ningyizhuo_No.1 · Yan Zhiheng; He Hongjie · Theme 3 / Light (Defense)**

A reproducible two-channel Light-side study. Strategy A evaluates context-free and weak context-conditioned lip dynamics on a 24-clip public pilot. Strategy B audits periodic chromatic traces introduced by Dark.Mimic's rPPG evasion on one peer-supplied aligned pair and a neutral re-encode control.

**Current evidence:** 24/24 public clips passed landmark extraction. On the six-clip held-out test, the context-free pilot reached macro-F1 0.486 and AUROC 0.667; the original single weak-context model did not improve it. A post-hoc multi-event context extension reached 0.829/0.778 on that same split, but 20 repeated four-fold runs showed no stable average gain, so it remains exploratory. In the single Dark.Mimic pair, colour energy near 1.2 Hz remained above a neutral re-encode control under H.264 CRF 35, 256×224 resizing and 15 FPS. These are course-pilot findings, not publication-scale generalisation claims. See [M3 results](reports/M3_RESULTS.md), the [multi-event follow-up](reports/M3_MULTI_CONTEXT_FOLLOWUP.md) and [submission checklist](reports/M3_SUBMISSION_CHECKLIST.md).

Submission materials: [executed notebook](notebooks/Milestone3.ipynb), [improved presentation](deliverables/FIT5230_M3_Context_Matters_Presentation_Improved_v2.pptx), [15-minute script](reports/M3_PRESENTATION_SCRIPT.md) and [Moodle Colab-link file](deliverables/Moodle_M3_Colab_Link.txt). The updated notebook has nine locally verified code cells; run it once in a fresh Colab runtime after publishing.

## Quick start

Use Python 3.12 (tested); Python 3.10–3.13 subject to MediaPipe wheel availability.

```bash
python -m venv .venv
# Windows PowerShell:
.venv/Scripts/Activate.ps1
# macOS/Linux instead: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m s2f.cli smoke
python -m s2f.cli public-demo
python -m pytest -q
```

After the M3 landmark and ASR caches exist, reproduce the explicitly post-hoc
multi-event follow-up with `python scripts/run_multi_context_improvement.py`.
Its fixed-split improvement is reported together with a repeated-cross-validation
stability check; it must not be described as fresh confirmatory evidence.

`smoke` needs no private data or downloads after installation. It creates 32 explicitly artificial trajectory fixtures, trains both models on 20 fixtures, and evaluates 4 validation / 8 test fixtures. Its artificial speaker IDs only exercise split logic; scores are **not detection evidence**. Public MP4 preprocessing downloads the same real Wikitongues and Hallo files as M1, plus a MediaPipe model. It runs on CPU. See [sources and provenance](reports/M1_PROVENANCE.md).

For notebook use, open [Milestone3.ipynb](notebooks/Milestone3.ipynb) in Colab and run all cells. It contains a verified source snapshot, public acquisition scripts, the real pilot experiment and hash-linked aggregate peer results. The peer MP4 files remain private. Local saved execution is not proof of Colab sharing permissions; the team must still upload the notebook to Colab, set **Anyone with the link can view**, and test the URL in an incognito window.

## Project layout

```text
configs/                  fixed seed, coverage and context settings
data/raw/                 original videos (excluded from release bundle)
data/synthetic/           explicitly artificial feature fixtures
data/variants/            robustness videos and parent manifest
data/*_template.csv       manifests, intervals and opponent pairs
src/s2f/                  reusable Python modules + CLI
notebooks/                standalone notebook with saved executed outputs
outputs/public_demo/      actual MP4 features, QC, previews, plots
outputs/smoke/            synthetic model metrics, predictions, figures
challenge/                strict validation/scoring and dummy examples
reports/                  evidence, M1 comparison, Ed draft, remaining work
reference/m1/             unmodified original team repository snapshot
tests/                    leakage, time-series and scorer regression checks
```

## Run a real pilot

1. Put licensed/consented MP4 files under `data/raw/`. Copy `data/manifest_template.csv` to `data/pilot.csv` and fill it using [the data contract](data/README.md). Keep original and derived clips in the same source family and split. Use multiple speakers, both real/fake classes in training, and ideally both classes in test. Do not train on the unmatched two-clip public demo.
2. Add one central phoneme/word interval per clip to `data/pilot_intervals.csv`. Use audio inspection and record annotator/tool provenance. Keep phoneme spellings consistent. An interval must have at least 0.2 s of valid context on both sides. Automatic alignment is an extension interface only.
3. Run:

```bash
python -m s2f.cli validate-manifest data/pilot.csv --protocol speaker
python -m s2f.cli extract-manifest data/pilot.csv --out outputs/pilot
python -m s2f.cli evaluate data/pilot.csv --features outputs/pilot/features --intervals data/pilot_intervals.csv --out outputs/pilot --protocol speaker
```

Omit `--intervals` for a context-free-only run. For the paired comparison, **both models use exactly the same clips and split**. All feature vocabulary/scaling is fit on training clips only. Decisions use a fixed 0.5 threshold; no test tuning. `--protocol generator` holds out fake generators while checking source families; `speaker_generator` requires both kinds of disjointness. Generator-only evaluation does not imply unseen speakers. Unsupported datasets fail validation rather than silently falling back.

Outputs include `metrics/results.csv`, individual metric JSONs (AUROC is null for one-class evaluation), `predictions/`, `models/`, confusion matrices, validated manifest and feature hashes. A small real run remains a **pilot**. Inspect subgroup/class counts and failure cases before making broader claims. PIA remains a separately named reference baseline; neither lightweight model is PIA.

## Test new / opponent clips

For an unknown clip, preprocessing does not need a label. Create a minimal inference manifest with `clip_id`; add matching phoneme columns only for a context model.

```bash
python -m s2f.cli preprocess data/raw/new_clip.mp4 --clip-id new_clip --out outputs/inference
python -m s2f.cli feature-table data/inference.csv outputs/inference/features --out outputs/inference/table.csv
python -m s2f.cli predict outputs/pilot/models/context_free.joblib outputs/inference/table.csv --out outputs/inference/predictions.csv
python -m s2f.cli compare-pairs data/pairs.csv outputs/inference/predictions.csv --out outputs/inference/pairs.csv
```

Train a real pilot model first. Synthetic models exist solely for software verification. Only load trusted joblib files. Add `--intervals` for a context feature table. If rPPG fake probabilities are supplied, `--alpha 0.5` enables an explicitly exploratory weighted average. Missing rPPG remains missing. Convert real-class probability using `1-p_real`; never substitute a raw waveform or arbitrary logit. Fusion requires validation/calibration before substantive comparison. See [interaction protocol and unsent draft](reports/DARK_MIMIC.md).

## Robustness and challenge

```bash
python -m s2f.cli variants data/pilot.csv --out data/variants
python -m s2f.cli extract-manifest data/variants/manifest.csv --out outputs/variants
python -m s2f.cli feature-table data/variants/manifest.csv outputs/variants/features --out outputs/variants/table.csv
python -m s2f.cli evaluate-frozen outputs/pilot/models/context_free.joblib data/variants/manifest.csv outputs/variants/table.csv --out outputs/robustness
python challenge/validate_submission.py challenge/dummy_submission.csv challenge/dummy_catalog.csv
python challenge/score_submission.py challenge/dummy_submission.csv challenge/dummy_catalog.csv challenge/dummy_answer_key.csv --out outputs/challenge
```

Variants include H.264 CRF 32, height 256 resize, and 15 FPS; only test clips are transformed. They retain source IDs/splits and must never enter training. For a context model, copy the parent's annotations to each variant ID, preserve times, and validate coverage again. Compare frozen clean-test and transformed-test scores on identical parent clips; fewer valid frames may cause explicit rejection.

The real 24-clip Context-Swap Gauntlet remains to be collected/annotated. Dummy scoring is executable now; it is not a released real challenge. [Challenge rules](challenge/README.md) preserve the M1 70/20/10 weights.

## Method limits

Input scope is a single visible speaker without scene cuts. MediaPipe tracks one face but does not establish identity or verify lip-sync. Geometry is measured in pixels then divided by eye distance; it does not fully correct pose. Detection gaps stay missing. Derivatives are computed within continuous valid runs; short runs and boundary frames are excluded. High-order derivatives remain sensitive to FPS and smoothing. Quality rejection is never a real/fake prediction. Context v1 supports one central interval per clip and lexical interaction features; it does not automatically localize forgeries or learn a publication-scale coarticulation model.

## Research references

- [Team M1 repository](https://github.com/zyan0246-del/FIT5230-Light-Context-Matters), preserved at commit `d2b091c4bf42f130c48fa060ce777d51e7cd9b74`.
- [PIA official implementation](https://github.com/skrantidatta/PIA) and [paper](https://arxiv.org/abs/2510.14241): main reference, not reproduced here.
- [CALS paper](https://arxiv.org/abs/2305.19556): context-aware **generation** motivates the hypothesis; it does not establish this detector's effectiveness.
- [MediaPipe Python API](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python).
