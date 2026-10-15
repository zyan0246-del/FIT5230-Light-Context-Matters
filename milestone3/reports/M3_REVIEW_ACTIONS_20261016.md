# M3 review actions, 16 October 2026

This note separates implemented changes from evidence still missing. It is not a claim of an HD grade.

The revised `notebooks/Milestone3.ipynb` has been executed locally, with saved outputs for all nine code cells and 38 passing regression tests. The editable 11-slide `deliverables/FIT5230_M3_Context_Matters_Presentation_Final_v8.pptx` was regenerated and visually checked. This exact notebook revision has **not** yet been rerun in a fresh Colab runtime. The 15 October Colab record verifies the prior nine-cell version, not the new ablation.

| Reviewer concern | Action in this revision | Evidence boundary |
|---|---|---|
| Context gain not isolated | Added a same-window, same-model `multi_event_no_phone` ablation to `scripts/run_multi_context_improvement.py`, including fixed-split and repeated-CV outputs. Replayed audited local caches on 16 October. | On the known six-clip test, context scored 0.829/0.778 versus no-phone 0.625/0.667 (Macro-F1/AUROC). Repeated-CV means were mixed: 0.645/0.689 versus 0.632/0.694. The earlier local gain failed to reproduce in Colab, and this new ablation has not yet had a fresh Colab rerun. |
| One attacked pair | Added 12 distinct public real-speaker neutral re-encode controls (red target-band median 0.074), then applied a synthetic 1.2 Hz global brightness change to each. All 12 periodic-light variants raised the fraction, median 0.981. Retained the peer pair and three post-processing checks. | The frequency metric responds to non-attack periodic light; it cannot attribute the peer trace to the attack alone. Natural lighting/motion and multi-identity attacks remain untested. No false-positive or population-level detector claim. |
| Individual contribution vague | Split the *presentation responsibility* evenly: Yan explains Strategy A and its ablation; He explains Strategy B and the controls. Added prompts for each to identify demonstrable work. | The team has not supplied verified personal work logs. The slides and report do not invent individual implementation credit. Each member must confirm actual contributions before speaking. |
| Reproduction gap | Executed the revised public notebook locally. The peer rerun script verifies all three private MP4 hashes. Added an explicit FFmpeg recipe that regenerates a second neutral control from the peer control. | The independent historical-neutral rerun matched red ratios 4.63, 8.45, 6.28 and 20.86. The recipe-generated neutral produced red ratios 7.61, 1.88, 17.69 and 34.76 across the same conditions: still above one, but sensitive to control encoding. Fresh Colab rerun of the new lip ablation remains outstanding. |
| M2 history | Checked the supplied M2 notebook: 32 synthetic trajectories (16 artificial identity IDs) and two public preprocessing videos, with no real classifier accuracy. Updated the M3 script and results report. | A separate 16-video M2 pilot was not established by that notebook. |

## Before submitting an updated M3 package

1. Run the already locally executed `notebooks/Milestone3.ipynb` in a fresh Colab CPU session, save all outputs, and compare its ablation CSVs with the local cached replay. The historical 15 October Colab output must not be relabelled as verification of this revision.
2. If the no-phone comparison does not favor phonetic context consistently across folds, report it as another negative result. Do not select only the six-clip fixed split.
3. For a private rerun, set `M3_PRIVATE_DIR` to the directory holding the three verified MP4s, then run `PYTHONPATH=src python scripts/run_chromatic_robustness.py`. To generate a new neutral control, also set `M3_REGENERATE_NEUTRAL=1` and a separate `M3_CHROMATIC_OUT`. Additional attacked identities or controlled lighting/motion videos still require new data; present the one-attack-pair scope plainly.
4. Have Yan and He confirm their own actual contributions and rehearse the 15-minute talk. The equal speaking allocation is not evidence of equal historical implementation.
5. Moodle still requires a TXT or PDF containing the Colab URL, plus the team's presentation and any other item specified by the current Moodle page.
