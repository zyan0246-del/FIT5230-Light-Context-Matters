# Milestone 3 presentation plan and speaker script

Total time: 15 minutes. Target 11 slides.

## Slide 1: Title and research question (0:00 to 0:30)

Speaker: Yan.

"We are Light.Ningyizhuo.No.1. Our question is whether a speech-to-face fake that passes a physiological detector still leaves evidence in mouth motion or in the manipulation used to repair physiological signals. We test those two channels separately."

## Slide 2: M2 framework to M3 evidence (0:30 to 1:45)

Speaker: Yan.

"At Milestone 2, we had reproducible code, synthetic checks and public-video preprocessing, but no real classifier experiment. For Milestone 3 we added a 24-clip public pilot, actual held-out evaluation, weak phonetic alignment, a peer-pair forensic analyser, a neutral codec control and post-processing tests."

Point to the fixed split: 12 train, six validation and six test. Mention 37 passing tests and 24 out of 24 clips with full face-landmark coverage.

## Slide 3: Experimental framework (1:45 to 3:00)

Speaker: Yan.

"The first branch extracts lip geometry and compares a context-free model with a weak context model. The second branch compares aligned RGB signals from the Dark.Mimic control and modified video. We report each branch separately because one peer pair cannot calibrate a valid fusion model."

Emphasize the data boundaries: public videos for Strategy A, hash-identified private peer videos for Strategy B.

## Slide 4: Dark.Mimic's strategy and our response (3:00 to 5:00)

Speaker: He.

"Dark.Mimic aims to inject a more realistic periodic physiological signal into a generated talking face so that an rPPG detector such as RhythmFormer is less likely to reject it. They gave us an aligned control and an A=1 modified video. Our earlier paired analysis found identical audio and almost unchanged mouth geometry. That means their attack directly motivates a second forensic question: does the injected signal create an unusually periodic colour trace?"

State that the team did not receive enough identities for population-level accuracy. The raw peer videos remain private.

## Slide 5: Strategy A design (5:00 to 6:30)

Speaker: Yan.

"My strategy tests context-conditioned lip dynamics. The context-free model summarizes aperture, width and their time derivatives over a clip. The context version adds before, during and after statistics around one central vowel, conditioned on neighbouring phonetic classes. Both models use the same clips and the same fixed threshold."

Explain that Whisper word timestamps plus CMUdict give a weak proxy. Do not call it forced alignment.

## Slide 6: Strategy A held-out result (6:30 to 8:15)

Speaker: Yan.

"On six held-out clips, the saved local reference run gave macro-F1 0.486 for both models and AUROC 0.667 versus 0.556. A fresh Colab CPU rerun on 15 October gave the same macro-F1 and AUROC 0.556 versus 0.222. The exact AUROC changed with the runtime, but weak context was worse in both. The bootstrap intervals are wide because the test set contains only three real and three fake clips."

"The correct conclusion is that this weak context implementation does not improve the pilot."

## Slide 7: Strategy A multi-event follow-up (8:15 to 10:00)

Speaker: Yan.

"We addressed one clear limitation by using every eligible vowel proxy rather than one interval per clip. This gives 547 events in the saved local run, with a median of 20 per clip. A model learned expected central mouth geometry from real training speech and aggregated residuals back to one row per clip."

"Locally, the already-known six-clip split improved from 0.486 to 0.829 Macro-F1 and from 0.667 to 0.778 AUROC. That was post-hoc, not independent confirmation. Local repeated four-fold means were mixed: F1 0.655 to 0.645 and AUROC 0.678 to 0.689. More importantly, fresh Colab did not reproduce the fixed-split gain: follow-up test F1 0.486 and AUROC 0.444, versus baseline 0.486 and 0.556. Its repeated-CV mean changes were negative on both metrics. We need forced alignment and a fresh matched hold-out set."

If asked whether phonetic context caused the six-clip improvement: the follow-up also changes the number of windows, feature representation and classifier regularisation. A matched-parameter no-phone/shuffled-phone ablation and cross-fitted training residuals are still required.

## Slide 8: Strategy B design and neutral control (10:00 to 11:30)

Speaker: He.

"My strategy subtracts control RGB values from the modified video, then measures spectral energy near the peer-reported 1.2 Hz. I use the full frame, forehead and cheek regions. A neutral re-encode of the control estimates how much periodic energy an ordinary codec change creates."

"This is a paired forensic audit. It does not estimate a pulse waveform and it does not decide whether an arbitrary video is real or fake."

## Slide 9: Strategy B main result (11:30 to 12:45)

Speaker: He.

"In the original pair, the red-channel peak is 1.159 Hz. Energy near 1.2 Hz is 0.946 for A=1 and 0.204 for neutral re-encoding. The ratio is 4.63. The green-channel ratio is 8.48. The time-domain trace also shows a clear periodic pattern."

## Slide 10: Post-processing robustness (12:45 to 14:00)

Speaker: He.

"The red-channel attack-to-neutral ratio remains above one after all three transformations: 8.45 under H.264 CRF 35, 6.28 after resizing to 256 by 224 and 20.86 at 15 FPS. This is promising forensic evidence for this pair. It is not a general accuracy result because we have only one identity and require an aligned control."

"We have not documented a response or counter-test from Dark.Mimic to this M3 finding, so we cannot claim they were defeated. Natural periodic lighting, motion and camera auto-exposure may also produce false positives; testing those controls is our next step."

## Slide 11: Integrated conclusion (14:00 to 15:00)

Speaker: He, final sentence by Yan.

"The lip branch contains a negative first result and a transparent post-hoc follow-up. A local split improved, but the fresh Colab rerun did not reproduce it, and repeated folds do not show a consistent gain. The colour branch finds a strong but narrow trace in one peer pair."

Yan closes:

"For the course project, we have an executable, reproducible two-strategy evaluation with clear claim boundaries. For a paper, the next step is a matched multi-generator dataset with forced alignment and many attacked identities."

## Questions to prepare for

### Why did the context model fail?

The first version used only a word-timestamp proxy, 12 training clips and one interval per clip. The follow-up uses many intervals but still lacks forced alignment and matched sources. Its improvement on one known split did not remain stable across repeated folds.

### Why report the 0.829 result if it is post-hoc?

It records an exploratory local finding, but it cannot confirm effectiveness because the split was already visible, and fresh Colab did not reproduce it. We pair it with repeated cross-validation and explicitly require a new matched hold-out set.

### Why is the chromatic result not a detector?

It needs an aligned control and measures modified-minus-control differences. A deployable detector would need unpaired features and false-positive tests across lighting, motion and identities.

### Why no fusion score?

One peer pair cannot calibrate fusion without severe overfitting. We keep the channels separate and define fusion as future work.

### Did you use PIA?

PIA remains the main research reference. We did not claim reproduction because the available time and public pilot do not support a fair full PIA training comparison.
