"""Build the standalone, inspectable FIT5230 Milestone 3 notebook."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def text_snapshot() -> tuple[dict[str, str], str]:
    paths = list((ROOT / "src/s2f").glob("*.py"))
    paths += list((ROOT / "tests").glob("*.py"))
    paths += list((ROOT / "challenge").glob("*.csv"))
    paths += [
        ROOT / "requirements.txt",
        ROOT / "configs/default.json",
        ROOT / "data/m3_pilot_sources.json",
        ROOT / "data/m3_pilot_manifest.csv",
        ROOT / "data/m3_pilot_provenance.json",
        ROOT / "data/manifest.schema.json",
        ROOT / "scripts/download_m3_pilot.py",
        ROOT / "scripts/audit_m3_pilot.py",
        ROOT / "scripts/build_weak_phoneme_intervals.py",
        ROOT / "scripts/summarize_lip_pilot.py",
        ROOT / "scripts/run_multi_context_improvement.py",
        ROOT / "scripts/run_chromatic_robustness.py",
        ROOT / "scripts/run_benign_chromatic_controls.py",
        ROOT / "scripts/run_periodic_light_control.py",
        ROOT / "scripts/build_matched_sadtalker_pilot.py",
        ROOT / "scripts/build_expanded_sadtalker_pilot.py",
        ROOT / "scripts/run_matched_sadtalker_evaluation.py",
        ROOT / "scripts/run_matched_sadtalker_pipeline.sh",
        ROOT / "scripts/run_expanded_sadtalker_pipeline.sh",
        ROOT / "scripts/run_viseme_context_optimization.py",
        ROOT / "data/matched_sadtalker_manifest.csv",
        ROOT / "data/matched_sadtalker_aligned_manifest.csv",
        ROOT / "data/matched_sadtalker_intervals.csv",
        ROOT / "data/matched_sadtalker_provenance.json",
        ROOT / "data/expanded_sadtalker_manifest.csv",
        ROOT / "data/expanded_sadtalker_aligned_manifest.csv",
        ROOT / "data/expanded_sadtalker_intervals.csv",
        ROOT / "data/expanded_sadtalker_provenance.json",
        ROOT / "outputs/expanded_sadtalker/qc.csv",
    ]
    paths += list((ROOT / "artifacts/m3_results").glob("*.csv"))
    paths += list((ROOT / "artifacts/m3_results").glob("*.json"))
    paths += list((ROOT / "artifacts/m3_improvement").glob("*.csv"))
    paths += list((ROOT / "artifacts/m3_improvement").glob("*.json"))
    paths += list((ROOT / "artifacts/matched_sadtalker").glob("*.csv"))
    paths += list((ROOT / "artifacts/matched_sadtalker").glob("*.json"))
    paths += list((ROOT / "artifacts/viseme_optimization").glob("*.csv"))
    paths += list((ROOT / "artifacts/viseme_optimization").glob("*.json"))
    paths += list((ROOT / "artifacts/viseme_optimization_expanded").glob("*.csv"))
    paths += list((ROOT / "artifacts/viseme_optimization_expanded").glob("*.json"))
    paths += list((ROOT / "artifacts/viseme_optimization_new_segments").glob("*.csv"))
    paths += list((ROOT / "artifacts/viseme_optimization_new_segments").glob("*.json"))
    sources = {
        path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(set(paths))
    }
    serialised = json.dumps(sources, sort_keys=True)
    return sources, hashlib.sha256(serialised.encode()).hexdigest()


def build() -> None:
    sources, digest = text_snapshot()
    bootstrap = f'''from pathlib import Path
import hashlib, json, os, subprocess, sys
SOURCES = {sources!r}
EXPECTED_SHA256 = {digest!r}
assert hashlib.sha256(json.dumps(SOURCES, sort_keys=True).encode()).hexdigest() == EXPECTED_SHA256
ROOT = Path.cwd() / "m3_notebook_workspace"
ROOT.mkdir(exist_ok=True)
for relative, content in SOURCES.items():
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_text(encoding="utf-8") != content:
        raise RuntimeError(f"Refusing to overwrite a changed file: {{target}}")
    target.write_text(content, encoding="utf-8")
os.environ["MPLCONFIGDIR"] = str(ROOT / "outputs/.mplcache")
os.environ["PYTHONPATH"] = str(ROOT / "src")
sys.path.insert(0, str(ROOT / "src"))
print("Embedded source SHA256:", EXPECTED_SHA256)
print("Inspectable text files:", len(SOURCES))
'''
    install = '''import importlib.metadata as metadata
from packaging.requirements import Requirement
requirements = [Requirement(line) for line in (ROOT / "requirements.txt").read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
missing = []
for requirement in requirements:
    try:
        if metadata.version(requirement.name) not in requirement.specifier:
            missing.append(str(requirement))
    except metadata.PackageNotFoundError:
        missing.append(str(requirement))
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")], check=True)
else:
    print("Declared package versions already satisfied.")
from s2f.demo import environment
print(json.dumps(environment(), indent=2))
'''
    cells = [
        nbf.v4.new_markdown_cell(
            "# Context Matters: FIT5230 Milestone 3\n\n"
            "**Light.Ningyizhuo_No.1 | Yan Zhiheng and He Hongjie | Theme 3, Light side**\n\n"
            "This notebook evaluates two distinct defenses. Yan presents the phonetic-context "
            "mouth-dynamics analysis. He presents the periodic-colour audit of Dark.Mimic's rPPG evasion. "
            "These are speaking roles, not independently verified implementation credits. "
            "The public pilot runs end to end on CPU. A 36-pair controlled SadTalker experiment was generated on a T4 GPU and "
            "its commands, provenance and lightweight result tables are embedded below. Peer videos remain private, so the notebook includes their hashes, "
            "aggregate results and a guarded rerun path without redistributing the MP4 files."
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Reproducible source and environment\n\n"
            "The notebook contains a readable source snapshot and verifies its SHA256 before execution. "
            "A fresh run downloads approximately 180 MB of public models and videos. No paid GPU is required to rerun "
            "the original public branch or inspect the matched results; regenerating the 36 SadTalker videos requires a GPU."
        ),
        nbf.v4.new_code_cell(bootstrap),
        nbf.v4.new_code_cell(install),
        nbf.v4.new_markdown_cell(
            "## 2. Regression tests\n\n"
            "Tests cover manifest leakage, missing landmark runs, context representation, challenge input, "
            "paired score handling and chromatic spectral calculations."
        ),
        nbf.v4.new_code_cell(
            '''test_temp = ROOT / "tmp/pytest"
test_temp.parent.mkdir(parents=True, exist_ok=True)
result = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "--tb=short", "--basetemp", str(test_temp), "tests"],
    cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")), text=True, capture_output=True,
)
print(result.stdout)
print(result.stderr)
result.check_returncode()
'''
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Public 24-clip pilot\n\n"
            "The pilot contains 12 licensed Wikitongues real clips and 12 TalkingHeadBench fakes "
            "from Hallo and LivePortrait. Splits are fixed before modelling and remain speaker/source disjoint. "
            "Real and fake clips come from different collections, so this experiment can establish feasibility only."
        ),
        nbf.v4.new_code_cell(
            '''RUN_ORIGINAL_PUBLIC_PIPELINE = False
processed = ROOT / "data/processed/m3_pilot"
if RUN_ORIGINAL_PUBLIC_PIPELINE:
    if len(list(processed.glob("*.mp4"))) != 24:
        subprocess.run([sys.executable, "scripts/download_m3_pilot.py"], cwd=ROOT, check=True)
    else:
        print("Reusing 24 hash-recorded public pilot clips from the execution cache.")
    subprocess.run([sys.executable, "scripts/audit_m3_pilot.py"], cwd=ROOT, check=True)
else:
    print("Original public-video download skipped in this saved evidence run.")
    print("Set RUN_ORIGINAL_PUBLIC_PIPELINE=True to redownload and recompute that branch.")
import pandas as pd
from IPython.display import display, Image
manifest = pd.read_csv(ROOT / "data/m3_pilot_manifest.csv")
display(manifest.groupby(["split", "label"]).size().rename("clips").reset_index())
provenance = json.loads((ROOT / "data/m3_pilot_provenance.json").read_text())
print("Hash-recorded public sources:", len(provenance))
'''
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Mouth landmarks and phonetic context\n\n"
            "MediaPipe extracts normalized lip geometry and derivatives. Whisper tiny.en supplies word timestamps; "
            "CMUdict supplies English pronunciations. Uniform phone placement within each word creates a weak proxy, "
            "not forced-alignment ground truth. We compare both models on the same 12 train, 6 validation and 6 test clips."
        ),
        nbf.v4.new_code_cell(
            '''from s2f.demo import MODEL_URL, download
def run_checked(command):
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True,
                               env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
    if completed.returncode:
        print(completed.stdout)
        print(completed.stderr)
        completed.check_returncode()
    if completed.stdout.strip():
        print(completed.stdout.strip())

model = download(MODEL_URL, ROOT / "models/face_landmarker.task")
features = ROOT / "outputs/m3_work/pilot_features"
if RUN_ORIGINAL_PUBLIC_PIPELINE:
    if not (features / "qc.csv").exists():
        run_checked([sys.executable, "-m", "s2f.cli", "extract-manifest", "data/m3_pilot_manifest.csv",
                     "--root", ".", "--out", "outputs/m3_work/pilot_features",
                     "--model", "models/face_landmarker.task"])
    qc = pd.read_csv(features / "qc.csv")
    display(qc[["clip_id", "sampled_frames", "coverage", "status"]])
    assert len(qc) == 24 and qc.coverage.min() >= 0.8
    run_checked([sys.executable, "scripts/build_weak_phoneme_intervals.py"])
    run_checked([sys.executable, "-m", "s2f.cli", "evaluate", "data/m3_pilot_aligned_manifest.csv",
                 "--features", "outputs/m3_work/pilot_features/features", "--intervals", "data/m3_pilot_intervals.csv",
                 "--root", ".", "--out", "outputs/m3_work/pilot_eval_context", "--protocol", "speaker"])
    run_checked([sys.executable, "scripts/summarize_lip_pilot.py"])
else:
    print("Using the committed, hash-linked result tables from the completed public run.")
'''
        ),
        nbf.v4.new_code_cell(
            '''if RUN_ORIGINAL_PUBLIC_PIPELINE:
    results = pd.read_csv(ROOT / "outputs/m3_work/pilot_eval_context/metrics/results.csv")
else:
    results = pd.read_csv(ROOT / "artifacts/m3_results/test_metrics_with_bootstrap.csv")
display(results)
display(pd.read_csv(ROOT / "artifacts/m3_results/held_out_predictions_and_failures.csv"))
print("Result: the weak context proxy did not improve the six-clip held-out pilot.")
'''
        ),
        nbf.v4.new_markdown_cell(
            "### Exploratory multi-event context follow-up\n\n"
            "After seeing the original test result, we added a post-hoc extension that uses every eligible vowel "
            "proxy, learns expected central mouth geometry from real training speech and aggregates residuals at "
            "clip level. The original six-clip split is therefore not fresh confirmation. We report it together "
            "with 20 repeated four-fold comparisons in which the context expectation is refit inside each fold. "
            "A matched no-phone ablation uses the same events, numeric mouth inputs, ridge penalty and classifier; "
            "it removes only phonetic-category inputs. Report its fixed split and repeated folds separately. "
            "If a clip has no usable vowel at 0.80 ASR word confidence, a 0.60 fallback is recorded; "
            "the rule is applied without reference to labels or scores."
        ),
        nbf.v4.new_code_cell(
            '''if RUN_ORIGINAL_PUBLIC_PIPELINE:
    run_checked([sys.executable, "scripts/run_multi_context_improvement.py"])
else:
    print("Displaying the committed output of the completed multi-event run.")
followup = json.loads((ROOT / "artifacts/m3_improvement/multi_context_summary.json").read_text())
display(pd.DataFrame(followup["fixed_split_metrics"]))
display(pd.DataFrame(followup["repeated_cv"]))
display(pd.DataFrame(followup["matched_context_ablation"]["fixed_split"]))
display(pd.DataFrame(followup["matched_context_ablation"]["repeated_cv_mean"]))
print("Fallback ASR clips:", followup["events"]["fallback_clips"])
print(followup["interpretation"])
'''
        ),
        nbf.v4.new_markdown_cell(
            "### Controlled real/SadTalker development run\n\n"
            "The original public pilot used different real and fake collections. This development run uses 12 matched pairs: "
            "each fake is generated from the corresponding real speaker's still image and audio, and both outputs use "
            "the same 12-second duration, 512 by 512 canvas, 25 FPS and final codec settings. Pair-grouped folds prevent "
            "the matched real and fake from entering different folds. Four representations use the same classifier, "
            "regularisation, threshold and folds. The tables below motivated, but do not replace, the prespecified expanded confirmation that follows."
        ),
        nbf.v4.new_code_cell(
            '''matched = ROOT / "artifacts/matched_sadtalker"
import matplotlib.pyplot as plt
matched_summary = pd.read_csv(matched / "summary.csv")
matched_comparisons = pd.read_csv(matched / "paired_comparisons.csv")
matched_repeats = pd.read_csv(matched / "repeat_oof_metrics.csv")
matched_run = json.loads((matched / "run.json").read_text())
matched_provenance = json.loads((ROOT / "data/matched_sadtalker_provenance.json").read_text())
display(matched_summary)
display(matched_comparisons)
print("Pairs with verified decoded-audio equality:", len(matched_provenance))
print("Feature count ranges:", matched_run["feature_count_range_by_model"])

order = ["context_free", "window_no_phone", "phone_context", "shuffled_phone"]
labels = ["Global", "Window, no phone", "Phone context", "Shuffled phone"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for axis, metric, title in [(axes[0], "macro_f1", "Out-of-fold Macro-F1"),
                            (axes[1], "auroc", "Out-of-fold AUROC")]:
    axis.boxplot([matched_repeats.loc[matched_repeats.model.eq(model), metric] for model in order],
                 tick_labels=labels)
    axis.set_ylim(0, 1.02); axis.set_title(title); axis.tick_params(axis="x", rotation=18)
    axis.grid(axis="y", alpha=.25)
fig.suptitle("Matched real/SadTalker pilot: 20 repeated pair-grouped runs")
fig.tight_layout(); plt.show()
print("Development result only: temporal windows improved these 12 pairs; expanded confirmation follows.")
'''
        ),
        nbf.v4.new_markdown_cell(
            "### Prespecified expanded-segment confirmation\n\n"
            "A development run on the original 12 pairs suggested that a compact numeric window could reduce overfitting. "
            "Before inspecting any expanded results, we committed the data construction, representations and evaluation "
            "protocol at Git commit `eed6a3b`. The same 12 speakers now contribute three non-overlapping 12-second "
            "segments each, giving 36 matched real/SadTalker pairs and 72 clips. Nested evaluation groups every fold by "
            "speaker, selects logistic-regression regularisation only inside each outer training fold using binary log loss, "
            "and keeps the decision threshold at 0.5. The compact local representation retains the 24 global features and "
            "adds 36 trajectory summaries. Coarse viseme interactions and shuffled visemes test whether weak phonetic "
            "categories add information beyond the same numeric windows.\n\n"
            "We report both all 36 pairs and only the 24 newly generated s1/s2 pairs. The latter are fresh, non-overlapping "
            "segments but still reuse the same identities and generator, so they are a confirmation within this course "
            "setting rather than an independent generalisation test."
        ),
        nbf.v4.new_code_cell(
            '''development = ROOT / "artifacts/viseme_optimization"
expanded = ROOT / "artifacts/viseme_optimization_expanded"
new_segments = ROOT / "artifacts/viseme_optimization_new_segments"

def show_viseme_run(name, directory):
    summary = pd.read_csv(directory / "summary.csv")
    comparisons = pd.read_csv(directory / "paired_comparisons.csv")
    run = json.loads((directory / "run.json").read_text())
    print(name)
    print(
        "Clips / matched pairs / speakers:",
        run.get("n_clips", 24), run.get("n_pairs", 12), run.get("n_speakers", 12),
    )
    display(summary)
    display(comparisons)
    return summary, comparisons

development_summary, development_comparisons = show_viseme_run("Development: original s0 segment", development)
expanded_summary, expanded_comparisons = show_viseme_run("All three segments per speaker", expanded)
fresh_summary, fresh_comparisons = show_viseme_run("Only new s1/s2 segments", new_segments)
expanded_qc = pd.read_csv(ROOT / "outputs/expanded_sadtalker/qc.csv")
print(
    "Expanded landmark QC:", len(expanded_qc), "clips; minimum coverage",
    round(float(expanded_qc.coverage.min()), 3), "; failed clips",
    int((expanded_qc.status != "ok").sum()),
)

def comparison_row(frame, comparison, metric):
    return frame[(frame.comparison == comparison) & (frame.metric == metric)].iloc[0]

for label, frame in [("All segments", expanded_comparisons), ("New segments only", fresh_comparisons)]:
    f1 = comparison_row(frame, "lowdim_window-minus-global", "macro_f1")
    auc = comparison_row(frame, "lowdim_window-minus-global", "auroc")
    viseme = comparison_row(frame, "viseme_context-minus-shuffled_viseme", "auroc")
    print(
        f"{label}: compact-window minus global = {f1.mean_delta:+.3f} Macro-F1 "
        f"({int(f1.positive_repeats)}/20 positive repeats), {auc.mean_delta:+.3f} AUROC "
        f"({int(auc.positive_repeats)}/20 positive repeats). "
        f"Viseme minus shuffled-viseme AUROC = {viseme.mean_delta:+.3f}."
    )
'''
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Dark.Mimic paired chromatic audit\n\n"
            "Dark.Mimic supplied one aligned control and A=1 modified video. We measure modified-minus-control RGB "
            "differences and spectral energy near the peer-reported 1.2 Hz setting. A neutral re-encode tests codec effects. "
            "The raw peer files stay outside the public repository. The aggregate below identifies them by SHA256. "
            "Twelve public real-speaker codec controls and synthetic periodic-light variants probe specificity. "
            "This paired forensic measurement does not estimate physiology and does not classify real versus fake."
        ),
        nbf.v4.new_code_cell(
            '''provenance = json.loads((ROOT / "artifacts/m3_results/peer_pair_provenance.json").read_text())
print(json.dumps(provenance, indent=2))
pair = pd.read_csv(ROOT / "artifacts/m3_results/attack_vs_neutral.csv")
robust = pd.read_csv(ROOT / "artifacts/m3_results/robustness_summary.csv")
benign = pd.read_csv(ROOT / "artifacts/m3_results/benign_chromatic_controls.csv")
new_neutral = pd.read_csv(ROOT / "artifacts/m3_results/robustness_reproduced_neutral.csv")
periodic_light = pd.read_csv(ROOT / "artifacts/m3_results/periodic_light_controls.csv")
display(pair[pair.roi.eq("global")])
display(robust)
print("Twelve real-speaker codec-only controls (not a false-positive test):")
display(benign[["clip_id", "target_band_energy_fraction", "red_delta_rms"]])
print("Peer red ratios with a newly regenerated neutral control:")
display(new_neutral[new_neutral.channel.eq("red")][["condition", "attack_to_neutral_ratio"]])
print("Neutral recipe:", json.loads((ROOT / "artifacts/m3_results/neutral_recipe.json").read_text())["command_template"])
print("Synthetic 1.2 Hz global light control (a non-attack trigger):")
display(periodic_light[["clip_id", "neutral_fraction", "periodic_fraction"]])
print("Periodic light exceeds neutral in", int((periodic_light.periodic_fraction > periodic_light.neutral_fraction).sum()), "of", len(periodic_light), "real-speaker clips")

import matplotlib.pyplot as plt
red = robust[robust.channel.eq("red")].set_index("condition").loc[["original", "h264_crf35", "resize_256", "fps_15"]]
x = range(len(red))
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.bar([i - .18 for i in x], red.attack, .36, label="Dark.Mimic A=1")
ax.bar([i + .18 for i in x], red.neutral, .36, label="Neutral re-encode")
ax.set_xticks(list(x), ["Original", "H.264 CRF 35", "256×224", "15 FPS"])
ax.set_ylim(0, 1); ax.set_ylabel("Red energy fraction near 1.2 Hz")
ax.set_title("Periodic chromatic trace under post-processing"); ax.legend(); ax.grid(axis="y", alpha=.2)
plt.show()
'''
        ),
        nbf.v4.new_markdown_cell(
            "### Optional private-pair rerun\n\n"
            "Team members can place the three verified MP4 files under `data/private/dark_mimic/` and set "
            "`RUN_PRIVATE_PAIR=True`. The public notebook skips this cell rather than downloading or fabricating private data."
        ),
        nbf.v4.new_code_cell(
            '''RUN_PRIVATE_PAIR = False
private = ROOT / "data/private/dark_mimic"
required_private = [private / "person01_fake01_control.mp4",
                    private / "person01_fake01_attacked_A1.mp4",
                    private / "person01_fake01_control_neutral_reencode.mp4"]
if RUN_PRIVATE_PAIR:
    if not all(path.exists() for path in required_private):
        raise FileNotFoundError("Place the three hash-verified private MP4 files in data/private/dark_mimic")
    subprocess.run([sys.executable, "scripts/run_chromatic_robustness.py"], cwd=ROOT, check=True,
                   env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
else:
    print("Private-pair rerun skipped. Displayed aggregate results remain linked to SHA256 provenance.")
'''
        ),
        nbf.v4.new_markdown_cell(
            "## 6. Conclusions and claim boundaries\n\n"
            "1. All 24 original public clips passed landmark coverage, but their six-clip held-out result remained small and confounded by source collection.\n"
            "2. The controlled SadTalker experiment expanded the 12-pair development run to 36 pairs / 72 clips. "
            "All clips passed landmark extraction with coverage 1.00. On only the 24 new pairs, global lip dynamics reached "
            "mean Macro-F1 0.762 and AUROC 0.843. Compact windows reached 0.677 and 0.739 and improved AUROC in 0/20 repeats. "
            "Weak viseme context reached 0.600 and 0.671 and did not consistently dominate shuffled visemes across all metrics. "
            "The final Strategy A model is therefore the 24-feature global representation; the favourable 12-pair window result "
            "is retained as development evidence that failed confirmation.\n"
            "3. The original weak phonetic-context proxy did not improve the held-out result. "
            "A post-hoc multi-event extension improved the saved local reference split, but not the fresh Colab "
            "rerun. In both environments its repeated four-fold comparisons showed no consistent gain across both "
            "metrics. ASR, decoding and package versions can change six-clip rankings; use the generated metrics "
            "above for this runtime. The expanded run also exposed one short-ASR boundary bug, which was repaired by using "
            "the landmark timeline as the video duration without changing ASR phone timestamps. No clip was removed. "
            "The course evidence therefore supports simpler global features rather than further tuning weak context.\n"
            "4. In the single attacked peer pair, periodic colour energy near 1.2 Hz remained higher than both the historical "
            "and newly regenerated neutral controls after H.264 CRF 35, resizing and 15 FPS. Twelve public real-speaker "
            "codec-only controls were also run. Synthetic 1.2 Hz global light raised the same metric on all 12 real "
            "speakers, showing that the metric is not attack-specific. These checks support a paired manipulation audit, "
            "not a general detector or measured false-positive rate; natural lighting, motion and exposure are not controlled.\n\n"
            "The integrated Light defense keeps both channels separate: lip dynamics test S2F generation artefacts, while "
            "chromatic forensics test traces introduced by the attempted rPPG evasion. We do not tune fusion on one peer pair."
        ),
    ]
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "source_snapshot_sha256": digest,
        },
    )
    import os
    destination = ROOT / "notebooks" / os.environ.get("M3_NOTEBOOK_NAME", "Milestone3.ipynb")
    destination.parent.mkdir(exist_ok=True)
    nbf.write(notebook, destination)
    print("Built", destination, "with", len(cells), "cells and source", digest)


if __name__ == "__main__":
    build()
