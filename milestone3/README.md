# Context Matters: FIT5230 Milestone 3

Team: Light.Ningyizhuo_No.1  
Members: Yan Zhiheng and He Hongjie  
Theme: Speech-to-Face, Light / Defense

## Open and run

Open the [Google Colab notebook](https://colab.research.google.com/github/zyan0246-del/FIT5230-Light-Context-Matters/blob/main/milestone3/notebooks/Milestone3.ipynb). It contains the source snapshot, environment setup, execution controls and saved outputs. Run the cells in order and follow their resource and data requirements. Regenerating the 36 SadTalker videos requires a GPU. Private peer videos are excluded; their aggregate results and hashes are provided, with an optional rerun path for holders of the original files.

The [GitHub project](https://github.com/zyan0246-del/FIT5230-Light-Context-Matters/tree/main/milestone3) contains the source, tests and experiment records. The notebook is the code submission entry point. The current presentation and historical display materials are identified in the [presentation guide](deliverables/README.md). Speaker scripts are rehearsal aids, not required Moodle submissions.

## Findings

Strategy A compares mouth-dynamics representations using 36 matched real/SadTalker pairs from 12 speakers. On the 24 new pairs, global features reached mean Macro-F1 0.762 and AUROC 0.843; compact windows reached 0.677 and 0.739. Weak phonetic context did not provide a stable advantage. The final course model uses the global representation.

Strategy B measures periodic colour differences in one aligned Dark.Mimic control/modified pair. The trace remained above the neutral reference after compression, resizing and frame-rate reduction. Synthetic periodic light also increased the metric on all 12 public control sources. The metric therefore supports a paired trace analysis; it does not establish attack attribution, real/fake classification or a population false-positive rate. The two branches remain separate because one attacked pair cannot calibrate fusion.

## Presentation responsibilities

- Yan Zhiheng: Strategy A, grouped evaluation, representation comparisons, failure analysis and final model selection.
- He Hongjie: Strategy B, transformation controls, periodic-light counterexample, cross-environment differences and integration of the two branches.

The presentation allocates approximately 7.5 minutes to each member. These responsibilities describe presentation and analysis scope; they do not assign sole implementation ownership.

## Execution records and limitations

The final notebook contains 11 code cells with saved expanded-server outputs and 46 passing regression tests. The documented fresh Colab verification applies to the historical nine-code-cell revision tested on 16 October 2026, with 38 passing tests. A fresh full Colab run of the final eleven-code-cell revision is not documented.

The separate Codex-assisted Windows Strategy B rerun completed its tests and workflows, but numerical values differed from the reference environment. Qualitative observations agreed. The runs remain separately labelled; the cause of the differences was not isolated. The Windows archive records assisted execution rather than independent manual execution by a student.

Only 12 independent speakers and one generator are used in the matched experiment. The new segments reuse those identities. Strategy B has one attacked pair and requires an aligned control; natural lighting, motion, exposure and attacks on additional identities remain untested.

## Design rationale

The notebook explains why normalized lip geometry, motion summaries, compact windows and weak context were compared using a fixed regularized classifier. These cues were hypotheses tested by controlled comparisons, not assumed deepfake signatures. The expanded results led to selecting the global representation. Strategy B explains why paired subtraction, neutral re-encoding and a periodic-light counterexample are needed, and why a trace is not a real/fake decision. The 0.20-second context margin and spectral band settings are not claimed as optimal.

## Current presentation

Use the [current v15 presentation](deliverables/FIT5230_M3_Presentation_v15_Optimized.pptx). It retains the reference results and simplifies the wording for the online presentation. Earlier decks in deliverables are historical versions; see the [version guide](deliverables/README.md). The current notebook revision adds explanatory Markdown only; experiment code and saved outputs are unchanged. This wording revision is not a new Colab execution.
