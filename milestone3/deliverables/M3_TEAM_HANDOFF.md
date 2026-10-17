# M3 two-person handoff and contribution check

This is an **equal forward work allocation**, not a retrospective record of who
implemented the existing code. Before either person makes an individual claim,
both must check it against their own commits, notebook activity, files, notes or
tests. Do not describe an AI-assisted or jointly completed task as a solo
implementation.

| Person | Presentation responsibility | Remaining hands-on verification | Evidence to keep |
|---|---|---|---|
| Yan Zhiheng | M2-to-M3 progress, matched SadTalker design, lip features, temporal-window result and phone/shuffled controls | Review the 17 October server metrics against `M3_RESULTS.md`; explain both the temporal-window gain and the phone-context failure | Colab URL/version, server run date, source hashes, actual edits and test log |
| He Hongjie | Dark.Mimic threat, paired RGB/FFT audit, historical and newly generated neutral controls, 12 benign codec controls, synthetic-light counterexample and limitations | Check private peer SHA256s; rerun the four robustness conditions and benign/periodic-light scripts; inspect at least slides 8–11 and verify cited ratios; report any peer counter-test if received | Rerun commands, source hashes, aggregate CSVs, slide corrections and peer exchange record |

Both members should review the full 11-slide deck, rehearse the 15-minute timing,
and submit their own concise account of (1) personally completed tasks,
(2) joint tasks, (3) verification performed, and (4) one technical limitation
they can explain. If a task in the table was not actually completed by that
person, change the contribution statement rather than copying the allocation.

## Current evidence status

- The final notebook contains 10 saved server-executed code-cell outputs and 40 passing tests.
- The prior public revision completed nine cells and 38 tests in fresh Colab on
  16 October. The new controlled 12-pair SadTalker experiment completed on a T4
  server. It supports temporal windows within the pilot and rejects the current
  weak-phone representation.
- The chromatic attack data contain one peer-supplied attacked identity. Twelve
  public real-speaker codec-only controls, a recipe-generated neutral control
  and synthetic periodic-light variants expose a specificity limit but do not
  substitute for additional attacked identities or naturally captured lighting,
  motion and exposure tests.
