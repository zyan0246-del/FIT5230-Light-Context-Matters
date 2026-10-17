# M3 two-person handoff and contribution check

This is an **equal forward work allocation**, not a retrospective record of who
implemented the existing code. Before either person makes an individual claim,
both must check it against their own commits, notebook activity, files, notes or
tests. Do not describe an AI-assisted or jointly completed task as a solo
implementation.

| Person | Presentation responsibility | Remaining hands-on verification | Evidence to keep |
|---|---|---|---|
| Yan Zhiheng | M2-to-M3 progress, expanded matched SadTalker design, global/window/viseme comparison and shuffled control | Review the 17 October server metrics against `M3_RESULTS.md`; explain why the 12-pair window gain failed confirmation and why the global model was selected | Colab URL/version, server run date, source hashes, actual edits and test log |
| He Hongjie | Dark.Mimic threat, paired RGB/FFT audit, historical and newly generated neutral controls, 12 benign codec controls, synthetic-light counterexample and limitations | Check private peer SHA256s; rerun the four robustness conditions and benign/periodic-light scripts; inspect at least slides 8–11 and verify cited ratios; report any peer counter-test if received | Rerun commands, source hashes, aggregate CSVs, slide corrections and peer exchange record |

Both members should review the full 11-slide deck, rehearse the 15-minute timing,
and submit their own concise account of (1) personally completed tasks,
(2) joint tasks, (3) verification performed, and (4) one technical limitation
they can explain. If a task in the table was not actually completed by that
person, change the contribution statement rather than copying the allocation.

## Current evidence status

- The expanded server pipeline completed 36 matched pairs / 72 clips and 46 passing tests. The final notebook embeds these result tables and the tested source snapshot.
- The prior public revision completed nine cells and 38 tests in fresh Colab on
  16 October. The expanded controlled SadTalker experiment completed on a T4
  server. The original window gain did not confirm; the final Strategy A model
  uses global lip dynamics, while compact windows and weak visemes are negative ablations.
- The chromatic attack data contain one peer-supplied attacked identity. Twelve
  public real-speaker codec-only controls, a recipe-generated neutral control
  and synthetic periodic-light variants expose a specificity limit but do not
  substitute for additional attacked identities or naturally captured lighting,
  motion and exposure tests.
