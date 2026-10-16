# Milestone 3 presentation plan and speaker script

Total time: 15 minutes. Target 11 slides. Planned speaking time: Yan 7.5 minutes, He 7.5 minutes. This is an equal speaking allocation, not a record of implementation credit.

## Slide 1: Title and research question (0:00 to 0:30)

Speaker: Yan.

"We are Light.Ningyizhuo.No.1. Our question is whether a speech-to-face fake that passes a physiological detector still leaves evidence in mouth motion or in the manipulation used to repair physiological signals. We test those two channels separately."

## Slide 2: M2 framework to M3 evidence (0:30 to 1:45)

Speaker: Yan.

"The submitted M2 notebook used 32 synthetic trajectories from 16 artificial identity IDs to check classifier logic. It also preprocessed one real and one generated public video, but did not measure real-video detection accuracy. In M3 we built a 24-clip public pilot and added held-out evaluation, weak phonetic timing, and a separate paired forensic audit of Dark.Mimic's signal modification."

Point to the fixed split: 12 train, six validation and six test. Mention 38 passing tests and 24 out of 24 clips with full face-landmark coverage.

## Slide 3: Experimental framework (1:45 to 3:00)

Speakers: Yan for the first 30 seconds, He for the next 45 seconds.

Yan: "The public branch extracts lip geometry and compares a context-free model with a weak context model."

He: "The peer branch compares aligned RGB signals from Dark.Mimic's control and modified video. We report the branches separately because one peer pair cannot calibrate a valid fusion model."

Emphasize the data boundaries: public videos for Strategy A, hash-identified private peer videos for Strategy B.

## Slide 4: Dark.Mimic's strategy and our response (3:00 to 5:00)

Speaker: He.

"Dark.Mimic aims to inject a more realistic periodic physiological signal into a generated talking face so that an rPPG detector such as RhythmFormer is less likely to reject it. They gave us an aligned control and an A=1 modified video. Our earlier paired analysis found identical audio and almost unchanged mouth geometry. That means their attack directly motivates a second forensic question: does the injected signal create an unusually periodic colour trace?"

State that the team did not receive enough identities for population-level accuracy. The raw peer videos remain private.

## Slide 5: Strategy A design (5:00 to 6:30)

Speaker: Yan.

"I will present our lip-dynamics strategy. The context-free model summarizes aperture, width and their time derivatives over a clip. The context version adds before, during and after statistics around one central vowel, conditioned on neighbouring phonetic classes. Both models use the same clips and the same fixed threshold."

Explain that Whisper word timestamps plus CMUdict give a weak proxy. Do not call it forced alignment.

## Slide 6: Strategy A held-out result (6:30 to 8:15)

Speaker: Yan.

"On six held-out clips, the saved local reference run gave macro-F1 0.486 for both models and AUROC 0.667 versus 0.556. A fresh Colab CPU rerun on 15 October gave the same macro-F1 and AUROC 0.556 versus 0.222. The exact AUROC changed with the runtime, but weak context was worse in both. The bootstrap intervals are wide because the test set contains only three real and three fake clips."

"The correct conclusion is that this weak context implementation does not improve the pilot."

## Slide 7: Strategy A multi-event follow-up (8:15 to 10:00)

Speaker: Yan.

"We addressed one clear limitation by using every eligible vowel proxy rather than one interval per clip. This gives 547 events in the saved local run, with a median of 20 per clip. A model learned expected central mouth geometry from real training speech and aggregated residuals back to one row per clip."

"We added a matched no-phone ablation using identical events, windows, ridge and classifier settings, and splits; only phone-category inputs were removed. The saved local replay looked better on the already-known six-clip split, but its repeated-fold evidence was mixed. We then ran this exact revision from scratch in Colab. There, context and no-phone both scored F1 0.486 and AUROC 0.444 on the six test clips. Across 20 repeated four-fold runs, context averaged F1 0.585 and AUROC 0.621, versus no-phone 0.593 and 0.638. The fresh run therefore does not show a phonetic-context advantage."

If asked whether phonetic context caused the six-clip improvement: this matched ablation removes a major confound but the known split, in-sample real-training residuals and mixed repeated-fold result prevent a causal or stable-effect claim. Cross-fitting, shuffled-phone controls and a new matched hold-out are still needed.

## Slide 8: Strategy B design and neutral control (10:00 to 11:30)

Speaker: He.

"I will present our chromatic audit. It subtracts control RGB values from the modified video, then measures spectral energy near the peer-reported 1.2 Hz. We use the full frame, forehead and cheek regions. A neutral re-encode of the control estimates how much periodic energy an ordinary codec change creates."

"This is a paired forensic audit. It does not estimate a pulse waveform and it does not decide whether an arbitrary video is real or fake."

## Slide 9: Strategy B main result (11:30 to 12:45)

Speaker: He.

"In the original pair, the red-channel peak is 1.159 Hz. Energy near 1.2 Hz is 0.946 for A=1 and 0.204 for neutral re-encoding. The ratio is 4.63. The green-channel ratio is 8.48. The time-domain trace also shows a clear periodic pattern."

## Slide 10: Post-processing robustness (12:45 to 14:00)

Speaker: He.

"The red-channel attack-to-neutral ratio remains above one after all three transformations: 8.45 under H.264 CRF 35, 6.28 after resizing to 256 by 224 and 20.86 at 15 FPS. This is promising forensic evidence for this pair. It is not a general accuracy result because we have only one identity and require an aligned control."

"We also ran 12 public real-speaker videos against their own neutral codec re-encodes; their red target-band fractions ranged from 0.008 to 0.359, median 0.074. A newly generated neutral control with an explicit MPEG-4 recipe kept all four peer red ratios above one, although the ratios changed. To challenge specificity, we added synthetic 1.2 Hz global brightness variation to those 12 real videos; all 12 then triggered the same frequency metric, with median 0.981. Thus the metric is not attack-specific. We have no documented Dark.Mimic counter-test and cannot claim the peer was defeated."

He should distinguish the completed 12-speaker *codec-only* and *synthetic periodic-light* controls from the still-missing multi-identity attacks and naturally captured lighting, motion and exposure experiments. Do not call the 12-speaker values a false-positive rate.

## Slide 11: Integrated conclusion (14:00 to 15:00)

Speaker: He, final sentence by Yan.

"The lip branch contains a negative first result and a transparent matched ablation with mixed evidence. A prior local split gain failed to reproduce in Colab. The colour branch finds a strong trace in one attacked peer pair, but a synthetic-light counterexample shows the frequency metric is not attack-specific."

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

### What did each member personally do?

The speaking roles are balanced: Yan presents the public pilot, lip branch and matched ablation; He presents the peer exchange, chromatic branch and controls. This is a presentation allocation, **not** a claim that either person personally implemented every part of that branch. Each member should identify only specific code, tests, data work or analysis they can demonstrate from their own records. If both worked jointly, say so. Both should explain the interface between branches: the team deliberately did not fit a fusion weight from one peer pair. The actual historical contribution split needs member confirmation before submission.
