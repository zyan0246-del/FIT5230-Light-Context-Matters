# FIT5230 Milestone 3 results

Team: Light.Ningyizhuo_No.1

Members: Yan Zhiheng and He Hongjie

Theme: Speech-to-Face, Light side

## Research question

Dark.Mimic attempts to make a generated talking-face video look physiologically plausible to an rPPG detector. We test two independent defensive signals:

1. whether the mouth motion fits its local phonetic context;
2. whether the attempted physiological evasion leaves an unusually periodic colour trace.

We report the channels separately. We do not tune score fusion on one peer pair.

## Progress since Milestone 2

- Replaced synthetic-only classifier evidence with a 24-clip public pilot.
- Added fixed speaker/source-disjoint train, validation and test splits.
- Added reproducible Whisper and CMUdict weak context annotations.
- Added a paired chromatic spectral analyser and five unit tests.
- Added a neutral re-encode control and three post-processing conditions.
- Increased the regression suite to 32 passing tests.
- Built and executed a standalone Milestone 3 notebook with saved outputs.

## Public pilot

The pilot contains 12 Wikitongues real clips and 12 TalkingHeadBench fakes: six Hallo and six LivePortrait. Each split is balanced. All speakers and source IDs occur in exactly one split. All clips were converted to a 512×512 canvas at 25 FPS, H.264 CRF 23 and mono 16 kHz audio. Some source fakes are shorter than 12 seconds.

All 24 clips achieved face-landmark coverage of 1.0. This confirms that the preprocessing path runs, but it does not establish detector accuracy.

The real and fake classes come from different source collections. A classifier may exploit framing, capture or compression differences. Therefore, the pilot is a feasibility test rather than evidence of generalisation.

## Strategy A: context-conditioned lip dynamics

Owner: Yan Zhiheng.

### Method

The context-free model summarizes normalized mouth aperture, lip width, velocity, acceleration and jerk over each clip. The context model adds statistics before, during and after one central vowel interval and interactions with neighbouring phonetic classes.

Whisper tiny.en supplies word timestamps. CMUdict supplies the word pronunciation. We place phones uniformly within a word and select a 0.20-second central-vowel proxy. This is a weak automatic annotation, not forced alignment. Exact ARPAbet phones, words, confidence and annotation method are saved for audit.

Both models use the same 12 training, six validation and six held-out test clips. Logistic regression uses train-only scaling, class balancing, a fixed seed of 5230 and a fixed 0.5 threshold. No test-set tuning was performed.

### Held-out results

| Model | Test clips | Macro-F1 | Bootstrap 95% CI | AUROC | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| Context-free | 6 | 0.486 | 0.143 to 0.829 | 0.667 | 0.108 to 1.000 |
| Weak context | 6 | 0.486 | 0.143 to 0.829 | 0.556 | 0.000 to 1.000 |

The intervals use 5,000 paired stratified bootstrap resamples. The held-out set contains only three real and three fake clips, so the intervals are necessarily wide.

### Interpretation

The weak context proxy did not improve the pilot. Macro-F1 was unchanged and AUROC decreased by 0.111. This does not refute the coarticulation hypothesis. It shows that one approximate interval per clip, 12 training clips and cross-collection data are insufficient evidence for the proposed method.

The next research experiment should use forced phoneme alignment, many intervals per clip, matched real/fake source construction and substantially more speakers. The course fallback remains the context-free lip baseline.

## Peer work

Dark.Mimic generated or modified talking-face video and attempted to make its physiological signal more plausible to an rPPG-based detector. The exchanged package contained an aligned control and an A=1 modified video. Our paired check confirmed identical audio in the earlier analysis and very similar mouth geometry. Those findings motivated a forensic question: does the manipulation itself leave a periodic decoded-pixel trace?

The raw peer videos remain outside the public repository. The notebook records SHA256 hashes and includes only aggregate results.

## Strategy B: periodic chromatic-injection forensics

Owner: He Hongjie.

### Method

For every aligned frame, we subtract control RGB means from the modified video's RGB means in the full frame and fixed forehead and cheek regions. After linear detrending and a Hann window, we compute the FFT peak, peak-to-band-median ratio and fraction of 0.7 to 3.0 Hz energy within 1.2 ± 0.12 Hz.

A neutral control re-encodes the original control with the same MPEG-4 codec family and similar target bitrate. This separates periodic injection from ordinary re-encoding. We repeat the same paired audit after H.264 CRF 35, resizing to 256×224 and reducing frame rate to 15 FPS. These conditions were fixed before analysis.

### Main observation

For the original pair, the global red-channel peak occurred at 1.159 Hz. The energy fraction near 1.2 Hz was 0.946 for Dark.Mimic A=1 and 0.204 for the neutral re-encode, a ratio of 4.63. The green-channel values were 0.939 and 0.111, a ratio of 8.48.

Red-channel attack-to-neutral energy ratios remained above one after every tested transformation:

| Condition | Attack energy | Neutral energy | Ratio |
|---|---:|---:|---:|
| Original | 0.946 | 0.204 | 4.63 |
| H.264 CRF 35 | 0.367 | 0.043 | 8.45 |
| 256×224 | 0.800 | 0.127 | 6.28 |
| 15 FPS | 0.873 | 0.042 | 20.86 |

### Interpretation

The single pair supports a descriptive claim: the A=1 modification contains a strong periodic chromatic difference near the peer-reported frequency, and the difference survives the three tested post-processing operations better than neutral re-encoding.

It does not support a population-level detection claim. The analyser requires an aligned control, uses fixed rectangular regions rather than face tracking and has only one identity. Periodicity is not an rPPG waveform. A future detector would need unpaired features, multiple identities, clean lighting controls and false-positive evaluation.

## Integrated Light-side response

The two strategies answer different questions. The lip model asks whether generated mouth movement appears plausible. The chromatic audit asks whether the adversarial attempt to repair physiological plausibility leaves another trace. The first experiment currently gives a negative small-pilot result. The second gives a strong single-pair forensic result. Keeping them separate makes the evidence and limitations clear.

## Reproducibility

- Public acquisition URLs, license metadata, FFmpeg commands and SHA256 hashes are saved.
- Private peer videos are identified by SHA256 but are not redistributed.
- Fixed split, seed, threshold and preprocessing parameters are stored.
- The notebook contains an inspectable source snapshot and saved outputs.
- All eight notebook code cells execute without errors.
- All 32 regression tests pass.

## Claim boundary

This M3 completes a defensible course experiment. It does not reproduce PIA, establish a general S2F detector or validate a deployable rPPG defense. Publication work would require a matched large-scale dataset, forced alignment, additional generators, subject-disjoint evaluation and statistical testing across many seeds or folds.
