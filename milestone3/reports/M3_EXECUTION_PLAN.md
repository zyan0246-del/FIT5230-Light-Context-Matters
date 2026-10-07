# FIT5230 Milestone 3 execution plan

## Scope

M3 extends the M2 reproducible framework with two distinct Light-side strategies
targeting Dark.Mimic's physiological-signal manipulation.  The goal is a complete,
honest course experiment, not a publication-scale claim.

## Strategy A - context-conditioned lip-dynamics defense

Owner: Yan Zhiheng.

1. Build a small licensed real/fake pilot with source and speaker provenance.
2. Use fixed source/speaker-disjoint splits.
3. Compare the existing context-free baseline and Context v1 on identical clips.
4. Report macro-F1, AUROC, confusion matrices, counts and failure cases.
5. Apply the frozen model to the Dark.Mimic control/attacked pair if the pilot is
   sufficiently valid; otherwise explicitly abstain from classification.

## Strategy B - periodic chromatic-injection forensics

Owner: He Hongjie.

1. Compare aligned control/attacked videos using full-frame and fixed facial-region
   RGB differences.
2. Measure FFT peak frequency, peak-to-band-median ratio and energy near the
   peer-reported 1.2 Hz setting.
3. Create a neutral re-encode of the control video and run the identical audit.
4. Test compression, resizing and reduced frame rate without changing thresholds.
5. Analyse false-positive risks from illumination, motion and codec behaviour.

This is a paired manipulation audit, not a physiological estimator and not a
standalone real/fake classifier.

## Integrated story

The lip channel tests residual S2F generation inconsistencies.  The chromatic
channel tests a trace introduced by the attempted rPPG evasion itself.  Scores are
reported separately.  Fusion is optional and must not be tuned on the peer pair.

## Minimum evidence for M3

- real pilot manifest, provenance and fixed split;
- context-free versus Context v1 results on the same held-out clips;
- Dark.Mimic paired lip and chromatic results;
- neutral re-encode control;
- at least two robustness conditions;
- saved failures and limitations;
- fresh end-to-end Colab run with outputs;
- one clearly attributed strategy and contribution per member.

## Current status (7 October 2026)

- M2 source restored from the submitted notebook snapshot.
- Existing M2 tests, synthetic smoke test and public preprocessing retained.
- Dark.Mimic private videos copied to an ignored local directory.
- Existing peer results copied into `outputs/peer_dark_mimic/`.
- Strategy B spectral module and unit tests added.
- Dark.Mimic A=1 pair analysed in global/forehead/cheek regions.
- A codec/bitrate-matched neutral re-encode control produced and analysed.
- A 24-clip public pilot was downloaded, standardised and visually audited.
- All 24 clips achieved 1.0 face-landmark coverage.
- A fixed 12/6/6 speaker/source-disjoint train/validation/test split was evaluated.
- Whisper tiny.en plus CMUdict produced explicitly labelled weak phone-context proxies.
- The context-free test result was macro-F1 0.486 and AUROC 0.667 (n=6).
- Weak context did not improve the result: macro-F1 0.486 and AUROC 0.556.
- The peer chromatic trace remained above the neutral control under three fixed post-processing conditions.
- The standalone M3 notebook completed all eight code cells with saved outputs and no errors.
- Thirty-two regression tests pass.
