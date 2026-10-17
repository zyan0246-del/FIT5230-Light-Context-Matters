# Matched real/SadTalker follow-up

## Course question

The original 24-clip public pilot compared real and generated videos from different collections. That experiment showed that the pipeline worked, but collection, identity and encoding differences could influence the classifier. This follow-up asks a narrower course-level question:

> When the real and generated videos share the same speaker, audio, duration and final encoding, which lip representation is most reliable in this small pilot?

## Controlled data construction

Twelve public Wikitongues speakers supply one 12-second real clip each. For every speaker, the script extracts one portrait and the clip's audio, then generates a matched fake with SadTalker commit `cd4c0465ae0b54a6f85af57f5c65fec9fe23e7f8`. Both members of a pair use the same decoded audio and are standardised to H.264 CRF 23, 512 by 512 pixels, 25 FPS and mono 16 kHz AAC.

The final pilot contains 12 matched pairs and 24 clips:

| Split | Pairs | Real | Fake |
|---|---:|---:|---:|
| Train | 6 | 6 | 6 |
| Validation | 3 | 3 | 3 |
| Test | 3 | 3 | 3 |

The manifest keeps each real/fake pair under one `source_id`. Grouped evaluation never separates members of a pair. SHA256 values, public licence metadata, the exact generation and FFmpeg commands, model settings and decoded-audio MD5 values are recorded in `data/matched_sadtalker_provenance.json`. All 12 pairs passed the decoded-audio equality check. All 24 clips passed MediaPipe extraction with 1.00 landmark coverage.

## Fair model comparison

All four models use logistic regression with `C=1.0`, balanced class weights, seed 5230 and a fixed 0.5 threshold. The evaluation uses 20 repeated four-fold splits grouped by matched pair. Each repeat produces one out-of-fold probability for every clip.

| Representation | Input | Feature count per fold |
|---|---|---:|
| Context-free | Whole-clip lip dynamics | 24 |
| Window, no phone | Whole-clip features plus before/during/after windows | 102 |
| Phone context | The same numeric windows plus weak phone-category interactions | 726 to 960 |
| Shuffled phone | Phone labels permuted between pairs | 726 to 960 |

The shuffled-phone model is a negative control. It preserves the model family and dimensionality while breaking the link between the audio context and the video pair.

## Results

| Representation | Macro-F1, mean ± SD | AUROC, mean ± SD |
|---|---:|---:|
| Context-free | 0.707 ± 0.069 | 0.773 ± 0.054 |
| Window, no phone | **0.748 ± 0.063** | **0.806 ± 0.057** |
| Phone context | 0.522 ± 0.044 | 0.518 ± 0.054 |
| Shuffled phone | 0.551 ± 0.060 | 0.586 ± 0.067 |

Compared with the context-free model, the no-phone window representation increased mean Macro-F1 by 0.041 and mean AUROC by 0.033. It scored higher in 15 of 20 repeats for Macro-F1 and 14 of 20 for AUROC. These repeated splits reuse the same 12 pairs, so they measure stability within this pilot rather than independent replication.

Phone context did not help. It was worse than the no-phone window model in all 20 repeats for both metrics. It also failed to beat shuffled phone labels consistently. The central category was always a vowel, neighbouring categories were sparse, and the interaction representation expanded to as many as 960 features from only 18 training clips per fold. Weak word timestamps, approximate phone placement and high dimensionality provide a plausible explanation for the negative result.

## Course conclusion

The controlled experiment repairs the largest weakness of the original cross-collection pilot. At this development stage, local temporal windows improved the original 12-pair sample, while weak phone labels did not. This finding motivated a prespecified expansion to three non-overlapping segments per identity.

The expanded confirmation did **not** retain the window advantage. On only the 24 new pairs, the global model reached 0.762 Macro-F1 / 0.843 AUROC, compared with 0.677 / 0.739 for compact windows. The final course system therefore keeps the global representation. This page documents the development experiment; `M3_VISEME_OPTIMIZATION.md` contains the final 36-pair model-selection result.

This result does not establish a general detector. The pilot has 12 speakers, one generator and derived real/fake pairs. A publication claim would require new speakers, multiple generators, independent data and stronger alignment, which are outside the M3 submission scope.

## Reproduction

On the prepared server:

```bash
cd /root/gpufree-data/FIT5230-M3-Server/milestone3
source /root/gpufree-data/m3-venv/bin/activate
python scripts/build_matched_sadtalker_pilot.py
./scripts/run_matched_sadtalker_pipeline.sh
```

The second command validates the manifest, extracts landmarks, builds weak phone intervals, runs the fixed split, runs the controlled repeated evaluation and executes the test suite. The completed run finished with 40 passing tests.
