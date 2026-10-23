# Step 4 K-Frame Dataset Creation Report

## Purpose

Create image-path lists for limited-frame experiments; no images are transformed and no model is trained.

## Input and Settings

- Input: `E:\research_university\Research\looting-detection-research\data\splits\frames_with_splits.csv`
- Device: CPU (metadata processing).
- Site key: site_uid, stored as text.
- Existing Step 3 splits are preserved; no cross-validation is performed.
- K values: [1, 2, 3, 5, 8]
- Random base seed: 42; SHA-256 of base seed and site_uid/strategy/split/K.

## Selection Strategies

- recent_window: latest K available frames; missing months need not be consecutive.
- uniform: evenly spaced frame positions, not necessarily equally spaced calendar dates.
- early_late: earliest frame plus latest K-1 frames; earliest is not guaranteed to be undamaged.
- random: one repeatable selection without replacement for each site/K, sorted chronologically.
- For K=1, recent_window, uniform, and early_late all select the latest frame.
- For K=2, uniform and early_late both select the first and last frame.
- Random selections for different K values are not required to be nested.

## Main Output Summary

- Input frame rows: 63439
- Unique sites: 675
- Total K-frame sample rows created: 13500
- Full-sequence reference rows created: 675
- Combined output rows: 14175
- Skipped rows due to insufficient frames: 0
- Data leakage check: 0 site_uid values appear in more than one split.

## Site Split Counts

| Split | Looted | Preserved | Total |
|---|---:|---:|---:|
| train | 100 | 496 | 596 |
| val | 9 | 9 | 18 |
| test | 26 | 35 | 61 |

## Validated Before Saving

- Unique input image paths and consistent path/site identities: PASS.
- One split per complete site_uid in input and output: PASS.
- Frame numbering starts at zero and follows chronological order: PASS.
- Every sequence has K distinct image paths and matching metadata-list lengths: PASS.
- Labels and splits preserved from Step 3: PASS.
- Unique sample IDs and complete eligible site/K/strategy coverage: PASS.

## Samples by K, Strategy, Split, and Label

| K | Strategy | Split | Label | Samples |
|---:|---|---|---|---:|
| 1 | early_late | test | looted | 26 |
| 1 | early_late | test | preserved | 35 |
| 1 | early_late | train | looted | 100 |
| 1 | early_late | train | preserved | 496 |
| 1 | early_late | val | looted | 9 |
| 1 | early_late | val | preserved | 9 |
| 1 | random | test | looted | 26 |
| 1 | random | test | preserved | 35 |
| 1 | random | train | looted | 100 |
| 1 | random | train | preserved | 496 |
| 1 | random | val | looted | 9 |
| 1 | random | val | preserved | 9 |
| 1 | recent_window | test | looted | 26 |
| 1 | recent_window | test | preserved | 35 |
| 1 | recent_window | train | looted | 100 |
| 1 | recent_window | train | preserved | 496 |
| 1 | recent_window | val | looted | 9 |
| 1 | recent_window | val | preserved | 9 |
| 1 | uniform | test | looted | 26 |
| 1 | uniform | test | preserved | 35 |
| 1 | uniform | train | looted | 100 |
| 1 | uniform | train | preserved | 496 |
| 1 | uniform | val | looted | 9 |
| 1 | uniform | val | preserved | 9 |
| 2 | early_late | test | looted | 26 |
| 2 | early_late | test | preserved | 35 |
| 2 | early_late | train | looted | 100 |
| 2 | early_late | train | preserved | 496 |
| 2 | early_late | val | looted | 9 |
| 2 | early_late | val | preserved | 9 |
| 2 | random | test | looted | 26 |
| 2 | random | test | preserved | 35 |
| 2 | random | train | looted | 100 |
| 2 | random | train | preserved | 496 |
| 2 | random | val | looted | 9 |
| 2 | random | val | preserved | 9 |
| 2 | recent_window | test | looted | 26 |
| 2 | recent_window | test | preserved | 35 |
| 2 | recent_window | train | looted | 100 |
| 2 | recent_window | train | preserved | 496 |
| 2 | recent_window | val | looted | 9 |
| 2 | recent_window | val | preserved | 9 |
| 2 | uniform | test | looted | 26 |
| 2 | uniform | test | preserved | 35 |
| 2 | uniform | train | looted | 100 |
| 2 | uniform | train | preserved | 496 |
| 2 | uniform | val | looted | 9 |
| 2 | uniform | val | preserved | 9 |
| 3 | early_late | test | looted | 26 |
| 3 | early_late | test | preserved | 35 |
| 3 | early_late | train | looted | 100 |
| 3 | early_late | train | preserved | 496 |
| 3 | early_late | val | looted | 9 |
| 3 | early_late | val | preserved | 9 |
| 3 | random | test | looted | 26 |
| 3 | random | test | preserved | 35 |
| 3 | random | train | looted | 100 |
| 3 | random | train | preserved | 496 |
| 3 | random | val | looted | 9 |
| 3 | random | val | preserved | 9 |
| 3 | recent_window | test | looted | 26 |
| 3 | recent_window | test | preserved | 35 |
| 3 | recent_window | train | looted | 100 |
| 3 | recent_window | train | preserved | 496 |
| 3 | recent_window | val | looted | 9 |
| 3 | recent_window | val | preserved | 9 |
| 3 | uniform | test | looted | 26 |
| 3 | uniform | test | preserved | 35 |
| 3 | uniform | train | looted | 100 |
| 3 | uniform | train | preserved | 496 |
| 3 | uniform | val | looted | 9 |
| 3 | uniform | val | preserved | 9 |
| 5 | early_late | test | looted | 26 |
| 5 | early_late | test | preserved | 35 |
| 5 | early_late | train | looted | 100 |
| 5 | early_late | train | preserved | 496 |
| 5 | early_late | val | looted | 9 |
| 5 | early_late | val | preserved | 9 |
| 5 | random | test | looted | 26 |
| 5 | random | test | preserved | 35 |
| 5 | random | train | looted | 100 |
| 5 | random | train | preserved | 496 |
| 5 | random | val | looted | 9 |
| 5 | random | val | preserved | 9 |
| 5 | recent_window | test | looted | 26 |
| 5 | recent_window | test | preserved | 35 |
| 5 | recent_window | train | looted | 100 |
| 5 | recent_window | train | preserved | 496 |
| 5 | recent_window | val | looted | 9 |
| 5 | recent_window | val | preserved | 9 |
| 5 | uniform | test | looted | 26 |
| 5 | uniform | test | preserved | 35 |
| 5 | uniform | train | looted | 100 |
| 5 | uniform | train | preserved | 496 |
| 5 | uniform | val | looted | 9 |
| 5 | uniform | val | preserved | 9 |
| 8 | early_late | test | looted | 26 |
| 8 | early_late | test | preserved | 35 |
| 8 | early_late | train | looted | 100 |
| 8 | early_late | train | preserved | 496 |
| 8 | early_late | val | looted | 9 |
| 8 | early_late | val | preserved | 9 |
| 8 | random | test | looted | 26 |
| 8 | random | test | preserved | 35 |
| 8 | random | train | looted | 100 |
| 8 | random | train | preserved | 496 |
| 8 | random | val | looted | 9 |
| 8 | random | val | preserved | 9 |
| 8 | recent_window | test | looted | 26 |
| 8 | recent_window | test | preserved | 35 |
| 8 | recent_window | train | looted | 100 |
| 8 | recent_window | train | preserved | 496 |
| 8 | recent_window | val | looted | 9 |
| 8 | recent_window | val | preserved | 9 |
| 8 | uniform | test | looted | 26 |
| 8 | uniform | test | preserved | 35 |
| 8 | uniform | train | looted | 100 |
| 8 | uniform | train | preserved | 496 |
| 8 | uniform | val | looted | 9 |
| 8 | uniform | val | preserved | 9 |

## Warnings

- No warnings from the implemented checks.

## Scope and Use

- These are multiple selections from the same sites, not new independent sites.
- Choose the intended K and strategy for each experiment and keep the existing split assignments.
- Group and join by site_uid. Never pass site_uid, sample_id, or class-bearing paths as model features.
- Random selections change from the old script because the seed now uses the complete site identity.
- Images were not reopened. Unique paths do not establish unique pixels or geographical sites.
- Step 2/3 dataset warnings, including reference image-count differences, are not resolved by this step.
- These checks do not independently establish that the original official fold memberships are correct.

## Output Files Created

- `E:\research_university\Research\looting-detection-research\data\processed\kframe_sequences\kframe_samples_all.csv`
- `E:\research_university\Research\looting-detection-research\data\processed\kframe_sequences\full_sequence_samples.csv`
- `E:\research_university\Research\looting-detection-research\data\processed\kframe_sequences\kframe_skipped_samples.csv`
- `E:\research_university\Research\looting-detection-research\data\processed\kframe_sequences\kframe_summary_by_k_strategy.csv`
- `E:\research_university\Research\looting-detection-research\data\processed\kframe_sequences\kframe_samples_K1.csv`
- `E:\research_university\Research\looting-detection-research\data\processed\kframe_sequences\kframe_samples_K2.csv`
- `E:\research_university\Research\looting-detection-research\data\processed\kframe_sequences\kframe_samples_K3.csv`
- `E:\research_university\Research\looting-detection-research\data\processed\kframe_sequences\kframe_samples_K5.csv`
- `E:\research_university\Research\looting-detection-research\data\processed\kframe_sequences\kframe_samples_K8.csv`
- `E:\research_university\Research\looting-detection-research\data\processed\kframe_sequences\kframe_samples_recent_window.csv`
- `E:\research_university\Research\looting-detection-research\data\processed\kframe_sequences\kframe_samples_uniform.csv`
- `E:\research_university\Research\looting-detection-research\data\processed\kframe_sequences\kframe_samples_early_late.csv`
- `E:\research_university\Research\looting-detection-research\data\processed\kframe_sequences\kframe_samples_random.csv`
- `E:\research_university\Research\looting-detection-research\results\tables\step4_kframe_report.md`
