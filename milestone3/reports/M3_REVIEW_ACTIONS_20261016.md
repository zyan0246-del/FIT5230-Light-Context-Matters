# M3 review actions, 16 October 2026

This note separates implemented changes from evidence still missing. It is not a claim of an HD grade.

The revised `notebooks/Milestone3.ipynb` has been executed locally, with saved outputs for all nine code cells and 38 passing regression tests. The editable 11-slide `deliverables/FIT5230_M3_Context_Matters_Presentation_Final_v9.pptx` was regenerated and visually checked. The exact matched-ablation revision also completed all nine cells and 38 tests in a fresh Colab CPU runtime on 16 October.

| Reviewer concern | Action in this revision | Evidence boundary |
|---|---|---|
| Context gain not isolated | Added a same-window, same-model `multi_event_no_phone` ablation to `scripts/run_multi_context_improvement.py`, including fixed-split and repeated-CV outputs. Replayed audited local caches and reran the exact revision in fresh Colab. | Local known-split values favored context but repeated means were mixed. In fresh Colab, context and no-phone tied at 0.486/0.444 on the six-clip test; repeated means were 0.585/0.621 versus 0.593/0.638. No stable context advantage was found. |
| One attacked pair | Added 12 distinct public real-speaker neutral re-encode controls (red target-band median 0.074), then applied a synthetic 1.2 Hz global brightness change to each. All 12 periodic-light variants raised the fraction, median 0.981. Retained the peer pair and three post-processing checks. | The frequency metric responds to non-attack periodic light; it cannot attribute the peer trace to the attack alone. Natural lighting/motion and multi-identity attacks remain untested. No false-positive or population-level detector claim. |
| Individual contribution vague | Split the *presentation responsibility* evenly: Yan explains Strategy A and its ablation; He explains Strategy B and the controls. Added prompts for each to identify demonstrable work. | The team has not supplied verified personal work logs. The slides and report do not invent individual implementation credit. Each member must confirm actual contributions before speaking. |
| Reproduction gap | Executed the revised public notebook locally and in fresh Colab. The peer rerun script verifies all three private MP4 hashes. Added an explicit FFmpeg recipe that regenerates a second neutral control from the peer control. | The independent historical-neutral rerun matched red ratios 4.63, 8.45, 6.28 and 20.86. The recipe-generated neutral produced red ratios 7.61, 1.88, 17.69 and 34.76 across the same conditions: still above one, but sensitive to control encoding. Fresh Colab reproduced the code path, not the local lip improvement. |
| M2 history | Checked the supplied M2 notebook: 32 synthetic trajectories (16 artificial identity IDs) and two public preprocessing videos, with no real classifier accuracy. Updated the M3 script and results report. | A separate 16-video M2 pilot was not established by that notebook. |

## Before submitting an updated M3 package

1. Present the fresh-Colab no-phone comparison as another negative result. Do not select only the encouraging local six-clip split.
2. For a private rerun, set `M3_PRIVATE_DIR` to the directory holding the three verified MP4s, then run `PYTHONPATH=src python scripts/run_chromatic_robustness.py`. To generate a new neutral control, also set `M3_REGENERATE_NEUTRAL=1` and a separate `M3_CHROMATIC_OUT`. Additional attacked identities or controlled lighting/motion videos still require new data; present the one-attack-pair scope plainly.
3. Have Yan and He confirm their own actual contributions and rehearse the 15-minute talk. The equal speaking allocation is not evidence of equal historical implementation.
4. Moodle still requires a TXT or PDF containing the Colab URL, plus the team's presentation and any other item specified by the current Moodle page.
