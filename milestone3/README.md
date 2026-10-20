# Context Matters — FIT5230 Milestone 3

**Light.Ningyizhuo_No.1 · Yan Zhiheng; He Hongjie · Theme 3 / Light (Defense)**

A reproducible two-channel Light-side study. Strategy A evaluates lip dynamics on a 24-clip public pilot and a 36-pair matched real/SadTalker experiment. Strategy B audits periodic chromatic traces introduced by Dark.Mimic's rPPG evasion on one peer-supplied aligned pair and a neutral re-encode control.

**Current evidence:** the original 24-clip public pilot established an executable pipeline but did not show a stable weak-phone advantage. A 12-pair matched SadTalker development run suggested that compact local windows might help, so we prespecified an expansion before inspecting its scores. The completed expansion contains 36 matched pairs / 72 clips from three non-overlapping segments for each of 12 speakers; all 72 passed landmark extraction with 1.00 coverage. On only the 24 new pairs, the 24-feature global model reached mean Macro-F1 0.762 and AUROC 0.843, while compact windows reached 0.677 and 0.739. Weak viseme context reached 0.600 and 0.671 and did not consistently beat shuffled visemes. Strategy A therefore keeps the global representation and reports the older window gain plus weak visemes as failed confirmation/negative ablations. In the single Dark.Mimic pair, colour energy near 1.2 Hz remained above a neutral re-encode control under H.264 CRF 35, 256×224 resizing and 15 FPS. Synthetic 1.2 Hz global brightness variation on 12 real-speaker clips triggered the same metric in all 12, so the chromatic metric is not attack-specific. These are course-pilot findings, not publication-scale claims. See [M3 results](reports/M3_RESULTS.md), the [expanded optimisation report](reports/M3_VISEME_OPTIMIZATION.md) and [submission checklist](reports/M3_SUBMISSION_CHECKLIST.md).

Submission materials: [executed notebook](notebooks/Milestone3.ipynb), [editable presentation v14](deliverables/FIT5230_M3_Context_Matters_Presentation_Final_v14.pptx), [15-minute script](reports/M3_PRESENTATION_SCRIPT.md) and [Moodle Colab-link file](deliverables/Moodle_M3_Colab_Link.txt). The expanded GPU pipeline completed with 46 passing tests on 17 October 2026. The earlier public branch also completed in fresh Colab; see the [verification record](reports/M3_COLAB_VERIFICATION.md).

## Presentation and analysis responsibilities

- **Yan Zhiheng:** Strategy A representation and model comparisons, speaker-grouped evaluation, analysis of the failed weak-context and compact-window gains, and selection of the final global mouth-dynamics model.
- **He Hongjie:** Strategy B compression, resizing and frame-rate controls, the periodic-light counterexample, interpretation of cross-environment verification differences, and the reason the two strategies are reported separately instead of being fused.

The planned speaking time is approximately 7.5 minutes per member. These are presentation and analysis responsibilities, not implementation ownership. Each member must describe only work that can be supported by their own records.

The supplementary [Codex-assisted Windows verification archive](deliverables/B_Local_Verification_Evidence_20261018.zip) records a separate Strategy B rerun. The tests and all planned workflows completed, but the numerical magnitudes did not exactly reproduce the reference macOS/FFmpeg run. The qualitative checks remained consistent: attack-to-neutral ratios stayed above one in the tested peer conditions, and synthetic periodic light triggered the metric on all 12 public sources. The deck retains the labelled reference results; it does not pool the two environments or assign the discrepancy to one unisolated software component.

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
The script now also writes a matched `multi_event_no_phone` ablation to
`artifacts/m3_improvement/matched_context_ablation_*.csv`. This removes only
phonetic-category inputs while retaining the same event windows, numerical
features, ridge penalty, clip aggregation, classifier and splits. Its new
outputs have been generated from audited local caches and are now saved in the
16 October notebook. That revision completed locally and in a fresh
Colab CPU runtime with nine code cells and 38 passing tests. The repeated-CV Macro-F1 and AUROC effects
point in opposite directions, so no stable context benefit is claimed.
Its fixed-split improvement is reported together with a repeated-cross-validation
stability check; it must not be described as fresh confirmatory evidence.

The current notebook adds the controlled matched SadTalker development evidence
and the prespecified 36-pair expansion. It was executed on the prepared server
with saved outputs and 46 passing tests. The original public-video branch remains optional because live
Wikimedia downloads can time out; its committed result tables and prior fresh
Colab verification remain available for inspection.

For Strategy B, `scripts/run_benign_chromatic_controls.py` measures 12 public
real-speaker neutral codec pairs. `scripts/run_chromatic_robustness.py` verifies
private peer hashes and, with `M3_REGENERATE_NEUTRAL=1`, makes a second neutral
control from the original peer control using an explicit MPEG-4 recipe. Public
aggregate results live in `artifacts/m3_results/`. The additional
`scripts/run_periodic_light_control.py` puts a synthetic 1.2 Hz global
brightness oscillation on 12 real-speaker videos; all 12 trigger the frequency
metric, demonstrating a non-attack explanation for a strong 1.2 Hz trace.
These checks improve reproducibility and critical interpretation but do not
test naturally captured lighting, motion, exposure or attacks on more than
one identity.

`smoke` needs no private data or downloads after installation. It creates 32 explicitly artificial trajectory fixtures, trains both models on 20 fixtures, and evaluates 4 validation / 8 test fixtures. Its artificial speaker IDs only exercise split logic; scores are **not detection evidence**. Public MP4 preprocessing downloads the same real Wikitongues and Hallo files as M1, plus a MediaPipe model. It runs on CPU. See [sources and provenance](reports/M1_PROVENANCE.md).

For notebook use, open the public [GitHub-backed Colab notebook](https://colab.research.google.com/github/zyan0246-del/FIT5230-Light-Context-Matters/blob/main/milestone3/notebooks/Milestone3.ipynb) and run all cells. It contains a verified source snapshot, public acquisition scripts, the real pilot experiment and hash-linked aggregate peer results. The peer MP4 files remain private. The GitHub-backed link is public; a separate Drive copy and **Anyone with the link can view** setting are only needed if the lecturer specifically requires a Drive-owned notebook.

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
