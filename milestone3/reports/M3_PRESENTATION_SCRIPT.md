# Milestone 3 presentation plan and speaker script

Total time: 15 minutes. Target 11 slides. Planned speaking time: Yan 7.5 minutes, He 7.5 minutes. This is an equal speaking allocation, not a record of implementation credit.

## Slide 1: Title and research question (0:00 to 0:30)

Speaker: Yan.

"We are Light.Ningyizhuo.No.1. Our question is whether a speech-to-face fake that passes a physiological detector still leaves evidence in mouth motion or in the manipulation used to repair physiological signals. We test those two channels separately."

## Slide 2: M2 framework to M3 evidence (0:30 to 1:45)

Speaker: Yan.

"The submitted M2 notebook used 32 synthetic trajectories from 16 artificial identity IDs to check classifier logic. It did not measure real-video detection accuracy. M3 first added a public pilot. We then repaired its largest source confound by generating 12 matched SadTalker clips from the same speakers and audio as 12 real clips."

Point to the fixed split: 12 train, six validation and six test clips. Mention 40 passing tests and 24 out of 24 matched clips with full face-landmark coverage.

## Slide 3: Experimental framework (1:45 to 3:00)

Speakers: Yan for the first 30 seconds, He for the next 45 seconds.

Yan: "The matched branch compares global lip dynamics, local temporal windows, weak phone context and shuffled phone labels. Every evaluation fold keeps the matched real/fake pair together."

He: "The peer branch compares aligned RGB signals from Dark.Mimic's control and modified video. We report the branches separately because one peer pair cannot calibrate a valid fusion model."

Emphasize the data boundaries: 12 public speakers and 12 derived SadTalker videos for Strategy A; hash-identified private peer videos for Strategy B.

## Slide 4: Dark.Mimic's strategy and our response (3:00 to 4:30)

Speaker: He.

"Dark.Mimic aims to inject a more realistic periodic physiological signal into a generated talking face so that an rPPG detector such as RhythmFormer is less likely to reject it. They gave us an aligned control and an A=1 modified video. Our earlier paired analysis found identical audio and almost unchanged mouth geometry. That means their attack directly motivates a second forensic question: does the injected signal create an unusually periodic colour trace?"

State that the team did not receive enough identities for population-level accuracy. The raw peer videos remain private.

## Slide 5: Strategy A design (4:30 to 6:00)

Speaker: Yan.

"Each fake uses the corresponding real speaker's portrait and audio. Both members of a pair are standardised to 12 seconds, 512 by 512, 25 FPS and identical audio and codec settings. We compare whole-clip dynamics, local before/during/after windows, weak phone interactions and shuffled phone labels. All use the same logistic-regression settings and threshold."

Explain that Whisper word timestamps plus CMUdict give a weak proxy. Do not call it forced alignment.

## Slide 6: Original Strategy A result (6:00 to 7:00)

Speaker: Yan.

"On six held-out clips, the saved local reference run gave macro-F1 0.486 for both models and AUROC 0.667 versus 0.556. A fresh Colab CPU rerun on 15 October gave the same macro-F1 and AUROC 0.556 versus 0.222. The exact AUROC changed with the runtime, but weak context was worse in both. The bootstrap intervals are wide because the test set contains only three real and three fake clips."

"This negative result motivated the controlled matched experiment instead of more tuning on the known six-clip split."

## Slide 7: Matched SadTalker result (7:00 to 9:00)

Speaker: Yan.

"The matched pilot contains 12 real/fake pairs. Each pair shares the speaker, decoded audio, duration, split and final codec settings. We use 20 repeated four-fold splits grouped by pair, so the real and fake versions never enter different folds."

"The context-free model averages Macro-F1 0.707 and AUROC 0.773. Adding local windows without phone labels raises them to 0.748 and 0.806. The window model scores higher in 15 of 20 repeats for F1 and 14 of 20 for AUROC. Weak phone context averages only 0.522 and 0.518. It is worse than no-phone windows in all 20 repeats and does not beat shuffled phone labels consistently."

Explain the failure analysis: the weak aligner selects a vowel as every central category, neighbouring categories are sparse, and the interaction representation expands from 102 numeric window features to 726–960 features with only 18 training clips per fold. The course conclusion is that temporal windows help this matched SadTalker pilot; the current phone labels do not.

## Slide 8: Strategy B design and neutral control (9:00 to 10:30)

Speaker: He.

"I will present our chromatic audit. It subtracts control RGB values from the modified video, then measures spectral energy near the peer-reported 1.2 Hz. We use the full frame, forehead and cheek regions. A neutral re-encode of the control estimates how much periodic energy an ordinary codec change creates."

"This is a paired forensic audit. It does not estimate a pulse waveform and it does not decide whether an arbitrary video is real or fake."

## Slide 9: Strategy B main result (10:30 to 11:45)

Speaker: He.

"In the original pair, the red-channel peak is 1.159 Hz. Energy near 1.2 Hz is 0.946 for A=1 and 0.204 for neutral re-encoding. The ratio is 4.63. The green-channel ratio is 8.48. The time-domain trace also shows a clear periodic pattern."

## Slide 10: Post-processing robustness (11:45 to 13:30)

Speaker: He.

"The red-channel attack-to-neutral ratio remains above one after all three transformations: 8.45 under H.264 CRF 35, 6.28 after resizing to 256 by 224 and 20.86 at 15 FPS. This is promising forensic evidence for this pair. It is not a general accuracy result because we have only one identity and require an aligned control."

"We also ran 12 public real-speaker videos against their own neutral codec re-encodes; their red target-band fractions ranged from 0.008 to 0.359, median 0.074. A newly generated neutral control with an explicit MPEG-4 recipe kept all four peer red ratios above one, although the ratios changed. To challenge specificity, we added synthetic 1.2 Hz global brightness variation to those 12 real videos; all 12 then triggered the same frequency metric, with median 0.981. Thus the metric is not attack-specific. We have no documented Dark.Mimic counter-test and cannot claim the peer was defeated."

He should distinguish the completed 12-speaker *codec-only* and *synthetic periodic-light* controls from the still-missing multi-identity attacks and naturally captured lighting, motion and exposure experiments. Do not call the 12-speaker values a false-positive rate.

## Slide 11: Integrated conclusion (13:30 to 15:00)

Speaker: He, final sentence by Yan.

"The lip branch now has a controlled course result: temporal windows improve the matched SadTalker pilot, while weak phone labels fail the shuffled control. The colour branch finds a strong trace in one attacked peer pair, but a synthetic-light counterexample shows that the frequency metric is not attack-specific."

Yan closes:

"For the course project, both strategies are executable, their controls are explicit and our conclusions stay within the tested pilots."

## Questions to prepare for

### Why did the context model fail?

Whisper supplies word timestamps and CMUdict supplies pronunciations, so phone boundaries are approximate. Every central category was a vowel, neighbouring categories were sparse, and interactions expanded to 726–960 features from only 18 training clips per fold. The no-phone windows kept 102 numeric features and performed better.

### Why show the older six-clip negative result?

It explains why we built the controlled matched pilot rather than tuning the known split. We do not use the older split as evidence for the final temporal-window result.

### Why is the chromatic result not a detector?

It needs an aligned control and measures modified-minus-control differences. A deployable detector would need unpaired features and false-positive tests across lighting, motion and identities.

### Why no fusion score?

One peer pair cannot calibrate fusion without severe overfitting. We keep the channels separate and define fusion as future work.

### Did you use PIA?

PIA remains the main research reference. We did not claim reproduction because the available time and public pilot do not support a fair full PIA training comparison.

### What did each member personally do?

The speaking roles are balanced: Yan presents the public pilot, lip branch and matched ablation; He presents the peer exchange, chromatic branch and controls. This is a presentation allocation, **not** a claim that either person personally implemented every part of that branch. Each member should identify only specific code, tests, data work or analysis they can demonstrate from their own records. If both worked jointly, say so. Both should explain the interface between branches: the team deliberately did not fit a fusion weight from one peer pair. The actual historical contribution split needs member confirmation before submission.
