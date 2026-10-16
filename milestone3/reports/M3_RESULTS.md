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

- The submitted M2 notebook used 32 synthetic trajectory fixtures (16 artificial identity IDs) for its classifier smoke test and two public videos for preprocessing/coverage only. It reported no real-video classifier accuracy. M3 added the first 24-clip public real/fake pilot with held-out evaluation. A separate 16-video pilot mentioned in informal feedback is not verified by the supplied M2 notebook, so we do not claim it here.
- Added fixed speaker/source-disjoint train, validation and test splits.
- Added reproducible Whisper and CMUdict weak context annotations.
- Added a paired chromatic spectral analyser and five unit tests.
- Added a neutral re-encode control and three post-processing conditions.
- Increased the regression suite to 38 passing tests.
- Built and executed a standalone Milestone 3 notebook with saved outputs.

## Public pilot

The pilot contains 12 Wikitongues real clips and 12 TalkingHeadBench fakes: six Hallo and six LivePortrait. Each split is balanced. All speakers and source IDs occur in exactly one split. All clips were converted to a 512×512 canvas at 25 FPS, H.264 CRF 23 and mono 16 kHz audio. Some source fakes are shorter than 12 seconds.

All 24 clips achieved face-landmark coverage of 1.0. This confirms that the preprocessing path runs, but it does not establish detector accuracy.

The real and fake classes come from different source collections. A classifier may exploit framing, capture or compression differences. Therefore, the pilot is a feasibility test rather than evidence of generalisation.

## Strategy A: context-conditioned lip dynamics

Presentation lead: Yan Zhiheng. Individual implementation credit has not been independently verified.

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

These are the saved local reference-run values (Python 3.12). A fresh Google Colab CPU run on 15 October 2026 completed all nine code cells and produced macro-F1 0.486 for both original models, with AUROC 0.556 for context-free and 0.222 for weak context. The weak-context difference was -0.333. A prior eight-cell Colab run on 7 October had AUROC 0.556 versus 0.444. FFmpeg builds, decoded pixels, ASR output and dependency versions can differ; with only six test clips, AUROC ranks are especially unstable. The negative weak-context conclusion held in each run, but the exact scores did not. Full details are in [M3_COLAB_VERIFICATION.md](M3_COLAB_VERIFICATION.md).

### Interpretation

The weak context proxy did not improve the pilot. Macro-F1 was unchanged and AUROC decreased by 0.111. This does not refute the coarticulation hypothesis. It shows that one approximate interval per clip, 12 training clips and cross-collection data are insufficient evidence for the proposed method.

The next research experiment should use forced phoneme alignment, many intervals per clip, matched real/fake source construction and substantially more speakers. The course fallback remains the context-free lip baseline.

### Exploratory multi-event follow-up

After observing the original result, we added a post-hoc follow-up that uses all eligible vowel proxies rather than one interval per clip. It learns expected central mouth geometry from neighbouring phone classes and real training speech, then aggregates context residuals and transition summaries to one row per clip. The saved local analysis retained 547 events across all 24 clips (median 20 per clip). The primary ASR word-confidence threshold is 0.80; a clip with no eligible event retries at 0.60, with the fallback ID recorded in the output. This makes availability explicit but does not make approximate phoneme timing more accurate.

On the original six-clip split, this extension reached macro-F1 0.829 and AUROC 0.778. Because that test split was already known, we also ran 20 repeated four-fold comparisons with the context expectation refitted inside every training fold. Mean macro-F1 was 0.645 versus 0.655 for context-free, while mean AUROC was 0.689 versus 0.678. The apparent fixed-split improvement therefore did not become a stable average gain. We treat it as a feasibility result and require a fresh matched hold-out dataset for confirmation. See [M3_MULTI_CONTEXT_FOLLOWUP.md](M3_MULTI_CONTEXT_FOLLOWUP.md).

Crucially, the fresh Colab run did **not** reproduce that local split gain: the multi-event test result was macro-F1 0.486 and AUROC 0.444, versus context-free 0.486/0.556. It retained 567 events and required one documented 0.60-confidence fallback. Colab repeated-CV mean changes were -0.057 Macro-F1 and -0.067 AUROC. Therefore, the course evidence supports a functioning but environment-sensitive experiment, not an effective new detector.

This follow-up also changes regularisation (`C=0.01` versus the context-free model's `C=1.0`) and the window/feature representation. It has not isolated the contribution of correct phonetic neighbours. Its training-real residuals are computed in sample, whereas held-out residuals are out of sample. We disclose these limitations rather than interpreting the six-clip difference as a method-specific gain.

### Matched-setting phonetic ablation added after review

The source script includes a `multi_event_no_phone` comparison. It uses the same eligible events, before/after mouth measurements, ridge penalty, clip aggregation, logistic-regression settings and fixed/CV splits as `multi_event_context`; only the previous/next/central phone-category inputs to the expected-mouth model are removed. A local replay used 547 cached events from 24 clips. On the already-known six-clip split, context scored 0.829/0.778 versus no-phone 0.625/0.667 (Macro-F1/AUROC); repeated means were 0.645/0.689 versus 0.632/0.694, already a mixed result. The exact revision was then run from scratch in fresh Colab and produced 567 events. There, both versions scored 0.486/0.444 on the six test clips; repeated means were 0.585/0.621 for context versus 0.593/0.638 for no-phone. Thus the local fixed-split advantage did not reproduce, and the fresh matched ablation favors neither a stable nor causal phonetic-context claim. The CSVs are `matched_context_ablation_fixed.csv`, `matched_context_ablation_folds.csv` and `matched_context_ablation_summary.csv`. The six-clip test was already known and real training-event residuals remain in sample, so all fixed-split comparisons are exploratory.

## Peer work

Dark.Mimic generated or modified talking-face video and attempted to make its physiological signal more plausible to an rPPG-based detector. The exchanged package contained an aligned control and an A=1 modified video. Our paired check confirmed identical audio in the earlier analysis and very similar mouth geometry. Those findings motivated a forensic question: does the manipulation itself leave a periodic decoded-pixel trace?

The raw peer videos remain outside the public repository. The notebook records SHA256 hashes and includes only aggregate results.

## Strategy B: periodic chromatic-injection forensics

Presentation lead: He Hongjie. Individual implementation credit has not been independently verified.

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

No response or counter-test from Dark.Mimic to this M3 forensic finding has been documented. We therefore report what happened in the exchanged pair, not a demonstrated defeat of the target team. The synthetic periodic-lighting control below confirms a non-attack trigger; natural lighting, motion and camera auto-exposure remain plausible confounders.

The attack evidence includes one attacked identity plus its neutral re-encode. Twelve separate benign identities are now included for codec-only and synthetic-light checks, but none has a matched attack or naturally captured lighting change. These controls narrow the interpretation and expose a specificity limit, but do not measure false-positive rate. A future suite should include natural lighting, exposure changes and head motion on benign videos, plus attacks on multiple subjects. Until then, the attack result remains a paired forensic case study.

### Additional public benign-codec controls and neutral recipe

We applied the same paired colour analysis to 12 public real-speaker videos, each compared with its own MPEG-4 800 kb/s neutral re-encode. This explicit FFmpeg recipe is in `scripts/run_benign_chromatic_controls.py`. The global red 1.2 Hz target-band fraction ranged from 0.008 to 0.359, median 0.074; the peer A=1 pair was 0.946. Because these videos have different identities, lighting and capture conditions, the numbers are descriptive and **not** an estimated false-positive rate or a matched-control test. The absolute codec-difference RMS is also recorded so a high normalized fraction is not mistaken for a large injected signal. Aggregates and source hashes are in `artifacts/m3_results/benign_chromatic_controls.csv`.

We also regenerated a neutral MP4 directly from the hash-verified private peer control with the documented MPEG-4 800 kb/s command. Using this new neutral, the peer red attack/neutral ratios were 7.61, 1.88, 17.69 and 34.76 for original, CRF 35, resize and 15 FPS, respectively. They remained above one but varied markedly from the historical-neutral ratios. The original historical neutral file was not created with a fully recorded command, so we retain both analyses and do not claim bit-exact reproduction of that file.

### Synthetic periodic-lighting counterexample

To test specificity, we added the same *global* 1.2 Hz brightness modulation (normalized amplitude 0.02) to each of the 12 public real-speaker clips, using the same MPEG-4 settings as the neutral variant. The paired red target-band fraction rose above its own neutral value in **12/12** clips: median 0.981 for synthetic periodic light versus 0.074 for neutral codec re-encoding. This is a deliberate non-physiological perturbation and demonstrates that the current frequency metric responds to periodic light as well as to the peer modification. It therefore cannot by itself attribute a detected trace to Dark.Mimic, nor can the single attacked pair support a false-positive rate or deployable classifier. The reproducible recipe and per-speaker aggregate values are in `scripts/run_periodic_light_control.py` and `artifacts/m3_results/periodic_light_controls.csv`. Naturally captured lighting, motion, auto-exposure and attacks on multiple identities remain untested.

## Integrated Light-side response

The two strategies answer different questions. The lip model asks whether generated mouth movement appears plausible. The chromatic audit asks whether the adversarial attempt to repair physiological plausibility leaves another trace. The first experiment currently gives a negative small-pilot result. The second gives a strong single-pair forensic result. Keeping them separate makes the evidence and limitations clear.

## Reproducibility

- Public acquisition URLs, license metadata, FFmpeg commands and SHA256 hashes are saved.
- Private peer videos are identified by SHA256 but are not redistributed.
- On 16 October, the three private MP4 hashes were rechecked and the four-condition chromatic audit was rerun independently. Red attack/neutral ratios matched the reported rounded values: 4.63, 8.45, 6.28 and 20.86. This reproduces the one-pair analysis, not population-level validity.
- A documented MPEG-4 recipe regenerates a new neutral control. Its output SHA256 and ratios are saved separately from the historical neutral analysis.
- Twelve public real-speaker codec-only controls were run with the same metric; their aggregate table is public, not the derived videos.
- Twelve synthetic periodic-lighting variants on those real speakers exposed a specificity failure (12/12 increased target-band fractions); the controlled recipe and aggregates are public, not the derived videos.
- Fixed split, seed, threshold and preprocessing parameters are stored.
- The notebook contains an inspectable source snapshot and saved outputs.
- All nine updated notebook code cells execute locally without errors.
- All 38 regression tests pass.

## Claim boundary

This M3 completes a defensible course experiment. It does not reproduce PIA, establish a general S2F detector or validate a deployable rPPG defense. Publication work would require a matched large-scale dataset, forced alignment, additional generators, subject-disjoint evaluation and statistical testing across many seeds or folds.
