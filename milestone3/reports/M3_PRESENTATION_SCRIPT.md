# Milestone 3 presentation plan and speaker script

Total time: 15 minutes. Target 11 slides. Planned speaking time: Yan 7.5 minutes, He 7.5 minutes. This is an equal speaking allocation, not a record of implementation credit.

## Slide 1: Title and research question (0:00 to 0:30)

Speaker: Yan.

"We are Light.Ningyizhuo.No.1. Our question is whether a speech-to-face fake that passes a physiological detector still leaves evidence in mouth motion or in the manipulation used to repair physiological signals. We test those two channels separately."

## Slide 2: M2 framework to M3 evidence (0:30 to 1:45)

Speaker: Yan.

"The submitted M2 notebook used 32 synthetic trajectories from 16 artificial identity IDs to check classifier logic. It did not measure real-video detection accuracy. M3 first added a public pilot. We then built a matched SadTalker experiment and expanded it to three non-overlapping segments for each of 12 speakers: 36 pairs and 72 clips."

Point to the fixed split: 36 train, 18 validation and 18 test clips. Mention 46 passing tests and 72 out of 72 matched clips with full face-landmark coverage.

## Slide 3: Experimental framework (1:45 to 3:00)

Speakers: Yan for the first 30 seconds, He for the next 45 seconds.

Yan: "The matched branch compares global lip dynamics, compact local temporal windows, weak viseme context and shuffled visemes. Every evaluation fold keeps all three segments from one speaker together. Inner grouped folds select regularisation by log loss."

He: "The peer branch compares aligned RGB signals from Dark.Mimic's control and modified video. We report the branches separately because one peer pair cannot calibrate a valid fusion model."

Emphasize the data boundaries: 12 public speakers and 36 derived SadTalker videos for Strategy A; hash-identified private peer videos for Strategy B.

## Slide 4: Dark.Mimic's strategy and our response (3:00 to 4:30)

Speaker: He.

"Dark.Mimic aims to inject a more realistic periodic physiological signal into a generated talking face so that an rPPG detector such as RhythmFormer is less likely to reject it. They gave us an aligned control and an A=1 modified video. Our earlier paired analysis found identical audio and almost unchanged mouth geometry. That means their attack directly motivates a second forensic question: does the injected signal create an unusually periodic colour trace?"

State that the team did not receive enough identities for population-level accuracy. The raw peer videos remain private.

## Slide 5: Strategy A design (4:30 to 6:00)

Speaker: Yan.

"Each fake uses the corresponding real segment's portrait and audio. Both members of a pair are standardised to 12 seconds, 512 by 512, 25 FPS and identical audio and codec settings. We compare 24 global features, a 60-feature compact window, weak viseme interactions and shuffled visemes. Outer folds are grouped by speaker, and inner grouped folds select C from four values using binary log loss. The decision threshold stays at 0.5."

Explain that Whisper word timestamps plus CMUdict give a weak proxy. Do not call it forced alignment.

## Slide 6: Original Strategy A result (6:00 to 7:00)

Speaker: Yan.

"On six held-out clips, the saved local reference run gave macro-F1 0.486 for both models and AUROC 0.667 versus 0.556. A fresh Colab CPU rerun on 15 October gave the same macro-F1 and AUROC 0.556 versus 0.222. The exact AUROC changed with the runtime, but weak context was worse in both. The bootstrap intervals are wide because the test set contains only three real and three fake clips."

"This negative result motivated the controlled matched experiment instead of more tuning on the known six-clip split."

## Slide 7: Expanded SadTalker confirmation (7:00 to 9:00)

Speaker: Yan.

"The matched experiment contains 36 real/fake pairs. Each pair shares the speaker, decoded audio, duration, split and final codec settings. We use 20 repeated four-fold splits grouped by speaker, so all three real/fake segments from one identity stay together."

"The original 12-pair development run suggested a compact-window gain. On all 36 pairs, global and compact-window Macro-F1 are both about 0.753, but global AUROC is higher: 0.826 versus 0.809. The decisive check uses only the 24 new pairs. Global features reach Macro-F1 0.762 and AUROC 0.843. Compact windows fall to 0.677 and 0.739 and improve AUROC in zero of 20 repeats. Weak viseme context reaches 0.600 and 0.671 and does not consistently beat shuffled visemes."

Explain the failure analysis: word-level timestamps give coarse phone placement, some segments have very low ASR confidence, local features add correlated dimensions, and only 12 independent identities remain. The course conclusion is that the simpler global representation is most stable; compact windows and weak visemes are transparent negative ablations.

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

"We also ran 12 public real-speaker videos against their own neutral codec re-encodes; their red target-band fractions ranged from 0.008 to 0.359, median 0.074. A newly generated neutral control with an explicit MPEG-4 recipe kept all four peer red ratios above one, although the ratios changed. To challenge specificity, we added synthetic 1.2 Hz global brightness variation to those 12 real videos; all 12 then triggered the same frequency metric, with median 0.981. Thus the metric is not attack-specific. A separate Codex-assisted Windows rerun completed the same workflows, but its numerical magnitudes differed from the reference run while these qualitative findings stayed the same. We keep the environments separate because the source of the difference was not isolated. We have no documented Dark.Mimic counter-test and cannot claim the peer was defeated."

He should distinguish the completed 12-speaker *codec-only* and *synthetic periodic-light* controls from the still-missing multi-identity attacks and naturally captured lighting, motion and exposure experiments. Do not call the 12-speaker values a false-positive rate.

## Slide 11: Integrated conclusion (13:30 to 15:00)

Speaker: He, final sentence by Yan.

"The lip branch now has a controlled model-selection result: the apparent 12-pair window gain failed confirmation, and global lip dynamics are most stable on the new segments. Weak visemes also fail to add reliable information. The colour branch finds a strong trace in one attacked peer pair, but a synthetic-light counterexample shows that the frequency metric is not attack-specific."

Yan closes:

"For the course project, both strategies are executable, their controls are explicit and our conclusions stay within the tested pilots."

## Questions to prepare for

### Why did the context model fail?

Whisper supplies word timestamps and CMUdict supplies pronunciations, so phone boundaries are approximate. Some new segments have little confident English ASR. Coarse viseme interactions still add noisy dimensions, while only 12 independent identities are available. The weak labels therefore do not reliably condition the observed mouth motion.

### Why did you keep a negative confirmation result?

The data construction and evaluation were committed before the expanded scores were inspected. The new segments showed that the favourable 12-pair window result was not stable. Selecting the global model from this controlled comparison is a more defensible course outcome than reporting only the development gain.

### Why show the older six-clip negative result?

It explains why we built the controlled matched pilot rather than tuning the known split. We do not use the older split as evidence for the final temporal-window result.

### Why is the chromatic result not a detector?

It needs an aligned control and measures modified-minus-control differences. A deployable detector would need unpaired features and false-positive tests across lighting, motion and identities.

### Why no fusion score?

One peer pair cannot calibrate fusion without severe overfitting. We keep the channels separate and define fusion as future work.

### Did you use PIA?

PIA remains the main research reference. We did not claim reproduction because the available time and public pilot do not support a fair full PIA training comparison.

### What did each member personally do?

Yan leads the presentation of Strategy A: representation comparisons, speaker-grouped evaluation, failed context/window gains and final model selection. He leads Strategy B: transformation controls, the periodic-light counterexample, cross-environment differences and the rationale for reporting the channels separately. The Windows rerun used Codex assistance. These presentation responsibilities do not establish sole implementation ownership; personal contribution statements should refer to each member's actual work records.
