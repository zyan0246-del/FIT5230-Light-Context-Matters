# FIT5230 Milestone 3 — team review package (16 October 2026)

Team: Light.Ningyizhuo_No.1 (Yan Zhiheng and He Hongjie)

## What to open first

1. `deliverables/FIT5230_M3_Context_Matters_Presentation_Final_v8.pptx` — the revised, rendered 11-slide presentation, including the matched no-phone ablation, 12 public benign codec controls and the synthetic periodic-lighting counterexample.
2. `notebooks/Milestone3.ipynb` — the revised notebook, with all nine code cells executed **locally** and 38 tests passing. A fresh Colab run of this exact revision is still pending; the 15 October Colab verification concerns the prior nine-cell version.
3. `reports/M3_PRESENTATION_SCRIPT.md` — revised speaking notes for both members.
4. `reports/M3_RESULTS.md` and `reports/M3_REVIEW_ACTIONS_20261016.md` — results, limitations and remaining actions.

## Evidence boundaries

The matched no-phone ablation is in the executed final notebook. Local cached-data results are in `artifacts/m3_improvement/matched_context_ablation_*.csv`. On the known six-clip split the context version scored better, while repeated cross-validation was mixed. Do not present this as a stable or Colab-reproduced improvement.

The three private Dark.Mimic videos are intentionally excluded. Their hashes were rechecked and the one-pair chromatic audit was rerun on 16 October. Twelve public real-speaker codec-only controls were added, and a separate neutral MP4 was regenerated with a documented FFmpeg recipe. Synthetic 1.2 Hz global lighting changes on those 12 speakers triggered the frequency metric in all cases, so it cannot alone attribute the peer trace to an attack. None of this establishes a population-level detector; natural lighting/motion and multiple attacked identities remain untested.

## Submission status

- The public Colab and GitHub URLs are in `deliverables/Moodle_M3_Colab_Link.txt`.
- Confirm that GitHub `main` reflects the 16 October revision before sending the Colab link.
- Moodle submission and the two-person oral presentation still require action by the team.
- Before submission, run the revised final notebook end-to-end in fresh Colab, save outputs, and verify each member's account of their own work. The equal speaking allocation is not a verified record of who wrote each component.
