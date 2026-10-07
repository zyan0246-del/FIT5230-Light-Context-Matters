# Context-Swap Gauntlet — v1 framework

**Release status: dummy software examples only. The promised real 24-clip pack is pending.**

The intended task asks whether an apparently plausible central phoneme has a natural context-dependent transition. Planned real pack: 24 clips, 12 real / 12 generated, 3–8 seconds, central-phone matched prompt families (bee/boo/bay/bar; meet/moon/map), identity/recording/encoding matched wherever feasible, two generators if available. Manually audit context-swapped negatives for trivial cuts or audio mismatch. No controlled context-swapped videos have been produced here.

## Submission

Copy `submission_template.csv`; one row per catalog clip:

| Field | Meaning |
|---|---|
| clip_id | Exact public catalog ID |
| prediction | 0 real, 1 fake |
| fake_probability | Finite number in [0,1] |
| start_s, end_s | Both blank for predicted real; `0 <= start < end <= duration` for fake |
| cue | context_transition, timing, motion_smoothness, visual_artifact, other |
| comment | At most 30 whitespace-delimited words |

Manual inspection or original code is allowed. Metadata/reverse-search/source-filename lookup is not. Publish a catalog with only clip ID/duration; hide labels, source names, generators and target intervals. The example names `dummy_001` / `dummy_002` have no real video attached and are for schema/scorer verification only. They must not be described as a live challenge pack.

```bash
python challenge/validate_submission.py challenge/dummy_submission.csv challenge/dummy_catalog.csv
python challenge/score_submission.py challenge/dummy_submission.csv challenge/dummy_catalog.csv challenge/dummy_answer_key.csv --out outputs/challenge
```

Invalid or missing values, duplicates, missing/unknown IDs and out-of-range intervals cause a nonzero exit with a reason. They are not silently converted into valid predictions.

## Scoring (M1 contract)

- Classification: `70 * macro-F1` over fixed labels 0/1. AUROC is a tie-breaker; it is null if only one true class exists.
- Localization: `20 * mean temporal-IoU` over all clips. A true-real clip receives 1 only if predicted real with blank intervals. A false-negative fake receives 0. Otherwise use interval intersection / union against the hidden fake target.
- Interpretation: `10 * mean cue credit`. Exact category agreement receives 1. Optional organizer manual review CSV `clip_id,credit,reason` can increase an otherwise mismatched cue score to a documented credit in [0,1], supplied with `--review`. Publish the same review policy for all teams; do not silently auto-grade explanation quality.

Dummy example score is analytically 98/100: perfect classification/cue agreement, and mean localization (1+0.8)/2. This is a unit fixture, not a participant result.

Win condition once the real pack exists: beat the released context-free baseline's **total score on that same hidden pack**. An actual baseline challenge submission with intervals and cue justifications still needs to be produced; the clip classifier does not automatically localize suspicious intervals. Do not equate its classification-only metric with the total challenge score.

Optional evasion track (not scored in v1): recompression, resizing, mild blur/denoising or temporal smoothing, with unchanged transcript/audio and recognizable identity; duration within 0.1 s. Quality checks and a combined score must be fixed before opening that track. Current executable scoring covers the detection track only.

## Before release

Build the real pack and private key outside the public release folder; standardize encoding, strip metadata, randomize IDs, and exclude shared source families from training. Audit overlays, filenames, source hashes and any packaged notebook output for label leakage. `dummy_answer_key.csv` is safe to release because it is explicitly fictitious; never replace it with the hidden real answer key in the public source tree.
