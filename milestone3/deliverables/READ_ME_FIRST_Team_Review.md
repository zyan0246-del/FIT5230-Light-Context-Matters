# FIT5230 Milestone 3 — team review package (17 October 2026)

Team: Light.Ningyizhuo_No.1 (Yan Zhiheng and He Hongjie)

## What to open first

1. `deliverables/FIT5230_M3_Context_Matters_Presentation_Final_v11.pptx` — the final 11-slide presentation, including the controlled 12-pair SadTalker experiment and Dark.Mimic controls.
2. `notebooks/Milestone3.ipynb` — the revised notebook with saved outputs. The earlier public branch was also run in fresh Colab; see `reports/M3_COLAB_VERIFICATION.md`.
3. `reports/M3_PRESENTATION_SCRIPT.md` — revised speaking notes for both members.
4. `reports/M3_RESULTS.md` and `reports/M3_REVIEW_ACTIONS_20261016.md` — results, limitations and remaining actions.

## Evidence boundaries

The new matched pilot is the primary Strategy A control. It uses 12 public speakers and 12 SadTalker fakes generated with the same identity and audio. Across 20 repeated pair-grouped runs, temporal windows without phone labels improved mean Macro-F1 from 0.707 to 0.748 and AUROC from 0.773 to 0.806. Weak phone labels performed worse than the no-phone version and failed the shuffled-phone control. Keep both findings together and describe the scope as one small matched-generator pilot.

The three private Dark.Mimic videos are intentionally excluded. Their hashes were rechecked and the one-pair chromatic audit was rerun on 16 October. Twelve public real-speaker codec-only controls were added, and a separate neutral MP4 was regenerated with a documented FFmpeg recipe. Synthetic 1.2 Hz global lighting changes on those 12 speakers triggered the frequency metric in all cases, so it cannot alone attribute the peer trace to an attack. None of this establishes a population-level detector; natural lighting/motion and multiple attacked identities remain untested.

## Submission status

- The public Colab and GitHub URLs are in `deliverables/Moodle_M3_Colab_Link.txt`.
- Confirm that GitHub `main` reflects the 16 October revision before sending the Colab link.
- Moodle submission and the two-person oral presentation still require action by the team.
- Before submission, verify each member's factual account of their own work and rehearse the talk. The equal speaking allocation is not a verified record of who wrote each component.
