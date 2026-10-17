# FIT5230 Milestone 3 — team review package (17 October 2026)

Team: Light.Ningyizhuo_No.1 (Yan Zhiheng and He Hongjie)

## What to open first

1. `deliverables/FIT5230_M3_Context_Matters_Presentation_Final_v13.pptx` — the final 11-slide presentation, including the 36-pair SadTalker confirmation and Dark.Mimic controls.
2. `notebooks/Milestone3.ipynb` — the revised notebook with saved outputs. The earlier public branch was also run in fresh Colab; see `reports/M3_COLAB_VERIFICATION.md`.
3. `reports/M3_PRESENTATION_SCRIPT.md` — revised speaking notes for both members.
4. `reports/M3_RESULTS.md` and `reports/M3_REVIEW_ACTIONS_20261016.md` — results, limitations and remaining actions.

## Evidence boundaries

The final Strategy A control uses 12 public speakers, three non-overlapping segments per speaker and 36 matched SadTalker pairs. The original 12-pair development run suggested a window benefit, but the prespecified confirmation did not retain it. On only the 24 new pairs, global lip dynamics reached mean Macro-F1 0.762 / AUROC 0.843, while compact windows reached 0.677 / 0.739. Weak viseme context reached 0.600 / 0.671 and did not consistently dominate shuffled labels. Present the global representation as the final course model and the other representations as failed confirmation/negative ablations. Do not quote only the favourable 12-pair development result.

The three private Dark.Mimic videos are intentionally excluded. Their hashes were rechecked and the one-pair chromatic audit was rerun on 16 October. Twelve public real-speaker codec-only controls were added, and a separate neutral MP4 was regenerated with a documented FFmpeg recipe. Synthetic 1.2 Hz global lighting changes on those 12 speakers triggered the frequency metric in all cases, so it cannot alone attribute the peer trace to an attack. None of this establishes a population-level detector; natural lighting/motion and multiple attacked identities remain untested.

## Submission status

- The public Colab and GitHub URLs are in `deliverables/Moodle_M3_Colab_Link.txt`.
- Confirm that GitHub `main` reflects the 16 October revision before sending the Colab link.
- Moodle submission and the two-person oral presentation still require action by the team.
- Before submission, verify each member's factual account of their own work and rehearse the talk. The equal speaking allocation is not a verified record of who wrote each component.
