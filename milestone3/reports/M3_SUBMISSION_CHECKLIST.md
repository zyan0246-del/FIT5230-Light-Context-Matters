# Milestone 3 submission checklist

## Completed locally

- [x] Two clearly allocated presentation leads: Yan, lip context; He, chromatic forensics. Actual implementation contributions still require each member's confirmation.
- [x] 24-clip public pilot with balanced fixed splits.
- [x] Source/speaker leakage validation.
- [x] 24 of 24 clips passed face-landmark coverage.
- [x] Context-free and weak-context models evaluated on identical held-out clips.
- [x] Macro-F1, AUROC, confusion matrices, predictions and failure cases saved.
- [x] Paired stratified bootstrap uncertainty reported.
- [x] Dark.Mimic control and A=1 pair analysed.
- [x] Neutral re-encode control included.
- [x] H.264 CRF 35, resize and 15 FPS conditions included.
- [x] Raw peer files excluded from the public repository.
- [x] Peer file SHA256 provenance recorded.
- [x] 46 regression tests pass in the completed expanded server pipeline.
- [x] Multi-event context follow-up added with a repeated-cross-validation stability check.
- [x] Matched no-phone ablation run locally with fixed-split and 20 repeated four-fold comparisons; report mixed results.
- [x] New matched real/SadTalker pilot built from 12 public speakers and 12 same-identity, same-audio generated videos.
- [x] All 12 real/fake pairs verified to have identical decoded audio and matching final video/audio settings.
- [x] Matched pilot evaluated with 20 repeated four-fold splits grouped by pair; no pair leakage.
- [x] Context-free, no-phone temporal windows, weak phone context and shuffled-phone control compared under the same classifier settings.
- [x] All 24 matched clips passed landmark extraction with 1.00 coverage.
- [x] Prespecified expansion completed: 36 matched pairs / 72 clips, with three non-overlapping segments per speaker.
- [x] All 72 expanded clips passed landmark extraction with minimum coverage 1.00 and no failed status.
- [x] Nested speaker-grouped comparison completed with inner log-loss selection of regularisation.
- [x] Expanded all-pair and new-segment-only results saved separately.
- [x] Development window gain failed confirmation; final Strategy A model changed to the stable 24-feature global representation.
- [x] Coarse-viseme and shuffled-viseme controls completed; weak visemes did not show a stable advantage.
- [x] Twelve public real-speaker neutral codec controls run and aggregate results saved.
- [x] Synthetic 1.2 Hz periodic-light controls run on the 12 real speakers; 12/12 triggered the frequency metric, so specificity is explicitly limited.
- [x] A new neutral peer control regenerated with a documented FFmpeg recipe; results labelled separately from historical-neutral results.
- [x] Updated Milestone3.ipynb executed locally with nine saved code-cell outputs and no errors.
- [x] Server experiment completed with 46 passing tests.
- [x] Final v13 presentation updated, all 11 slides rendered, and package/layout/native-chart validation passed.
- [x] Fifteen-minute presentation script prepared.

## Still requires a team member

- [x] Run the current matched-ablation notebook in a fresh Colab CPU session. All nine cells and 38 tests completed on 16 October; the exact source hash and runtime metrics are recorded in `M3_COLAB_VERIFICATION.md`. The notebook's committed outputs remain the labelled local run because Colab requested separate GitHub OAuth for direct save.
- [ ] Add attacks on multiple identities and naturally captured lighting, motion and exposure videos if available; otherwise state clearly that Strategy B is a one-attack-pair forensic case plus 12 public codec-only and synthetic-light controls. Synthetic light is already tested and triggers the metric.
- [ ] Yan and He each verify and rehearse a factual account of their own code, analysis and collaboration; do not infer personal contribution from the owner labels alone.
- [ ] Read every slide and practise the 15-minute timing.
- [ ] If Dark.Mimic has given feedback or a counter-test after this analysis, add its actual outcome to the critical-analysis section; otherwise state that no response is documented.
- [ ] Confirm the lecturer expects both members to speak for approximately five minutes each in the strategy section.
- [x] Upload the final repository to GitHub.
- [x] Open `notebooks/Milestone3.ipynb` in Google Colab.
- [x] The previous eight-cell notebook completed in a fresh Colab CPU runtime.
- [x] Run the prior nine-cell notebook in a reset Colab CPU runtime; all nine cells and 37 tests completed on 15 October.
- [x] Run the current matched-ablation revision in a fresh Colab CPU runtime; all nine cells and 38 tests completed on 16 October.
- [x] Verify the public GitHub-backed Colab URL loads without repository authentication.
- [x] Record the cross-environment result variation without changing the conclusion.
- [x] Correct the post-hoc interpretation: the local multi-event gain did not reproduce in Colab.
- [x] Put the verified Colab URL in a TXT file.
- [ ] Optionally copy the notebook to Drive and set **Anyone with the link can view** if the lecturer requires a Drive-owned rather than GitHub-backed Colab link.
- [ ] Submit that PDF or TXT file to the Moodle Milestone 3 page before 22 October 2026, 11:55 PM MYT.

## Do not claim

- publication-level generalisation;
- forced phoneme alignment;
- full PIA reproduction;
- a deployable standalone chromatic detector;
- physiological waveform recovery;
- successful fusion calibrated from one peer pair;
- that a negative context result proves coarticulation is useless.
