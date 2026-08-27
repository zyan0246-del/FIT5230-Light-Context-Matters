# Context-Swap Gauntlet - Interactive Challenge Specification

## Challenge objective

Determine whether a short talking-face video is real or AI-generated, identify the most suspicious time interval, and explain the forensic cue used. The challenge emphasizes context-dependent lip transitions rather than isolated frames.

## Release package planned for Milestone 2

```text
context_swap_gauntlet/
├── README.md
├── clips/
│   ├── clip_001.mp4
│   ├── clip_002.mp4
│   └── ...
├── submission_template.csv
├── starter_analysis.ipynb
└── validate_submission.py
```

Labels, generator names, and target intervals remain hidden until scoring.

## Detection track

Participants submit one row per clip:

| Field | Type | Example | Meaning |
|---|---|---|---|
| `clip_id` | string | `clip_001` | Identifier supplied with the clip |
| `prediction` | integer | `1` | `0 = real`, `1 = fake` |
| `fake_probability` | float | `0.82` | Value from 0 to 1 |
| `start_s` | float or blank | `1.20` | Start of most suspicious interval; blank for predicted real |
| `end_s` | float or blank | `1.75` | End of most suspicious interval; blank for predicted real |
| `cue` | string | `context_transition` | Primary detection cue |
| `comment` | string | `Lip rounding starts too late.` | Maximum 30 words |

Allowed `cue` values:

```text
context_transition
timing
motion_smoothness
visual_artifact
other
```

## Dataset design

- Initial release target: 24 clips, balanced as 12 real and 12 fake.
- Single visible speaking face per clip.
- Short clips, normally 3-8 seconds.
- Identity-balanced split where practical.
- Controlled prompts containing repeated central phonemes in different contexts.
- Hard negatives matched on the same central phoneme and, where practical, speaker, duration, speaking rate, pose, resolution, and encoding quality.
- No random context swaps that create obvious audio-video discontinuities or label shortcuts.
- Target two S2F/lip-sync generators; if only one is available, include multiple standardized compression conditions and disclose the limitation.
- Metadata removed and filenames randomized.
- Hidden target intervals defined around the controlled phonetic transition.

Example prompt families:

```text
bee / boo / bay / bar
meet / moon / map
feet / food / fan
peat / pool / pan
```

## Detection scoring

Total: 100 points.

1. **Classification - 70 points**  
   Macro-F1 over `real` and `fake` labels. Probabilities are used for AUROC as a tie-breaker.

2. **Temporal localization - 20 points**  
   Mean temporal intersection-over-union between the submitted interval and the hidden target interval. A real clip receives full localization credit when both interval fields are left blank.

3. **Interpretation - 10 points**  
   Cue-category agreement with the hidden annotation, with partial manual credit for a concise technically sound explanation.

## Win condition and disclosure

A team beats our initial defense when its valid submission achieves a higher total score on the same hidden pack than our released context-free lip-dynamics baseline.

We disclose the clip count, prompt families, submission schema, allowed operations, and scoring code. We withhold real/fake labels, generator identities, source filenames, and target intervals until scoring is complete.

## Optional evasion track

Participants receive selected fake clips and attempt to reduce the defense model's fake probability.

Allowed operations:

- H.264 recompression;
- resizing, with the shorter side remaining at least 256 pixels;
- mild blur or denoising;
- frame interpolation or temporal smoothing;
- combinations of the above.

Not allowed:

- replacing the clip with a real recording;
- changing the transcript;
- replacing or muting the audio;
- trimming more than 0.1 seconds from either end;
- editing file metadata without changing the media content.

An evasion submission is valid only if:

1. the transcript remains unchanged;
2. duration differs by no more than 0.1 seconds;
3. the face remains visible and recognizable; and
4. the final quality satisfies the threshold released with the scoring script.

The evasion score will combine fake-score reduction with perceptual quality. Thresholds will be calibrated on the released development clips before the final challenge.

## Fairness and leakage controls

- Do not publish labels in filenames, folder structure, notebook outputs, or metadata.
- Normalize encoding before release to reduce codec-source shortcuts.
- Avoid placing the same source recording in both public development clips and hidden evaluation clips.
- Report results by identity and generator to expose shortcut learning.
- Preserve a private answer key with clip provenance, consent/license status, generator, prompt, and target interval.
