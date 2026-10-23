# Step 5 Dataset Loader Check Report

## Overall Result: PASS

Every batch in every required K/strategy/split combination must pass. Missing combinations are failures.

## Settings

- Project root: E:\research_university\Research\looting-detection-research
- Data root: E:\research_university\Research\looting-detection-research\data\raw\DAFA_LS
- Batch size: 4
- Image size: 266
- Normalization: imagenet
- Seed: 42
- Workers: 0
- CPU tensor threads: 1
- Coverage: all batches; no first-batch-only shortcut
- PyTorch version: 2.7.1+cu118
- CUDA available (not exercised by this check): True
- Images and transforms run on CPU. Pinned memory does not move tensors to a GPU.
- No model training, cross-validation, or archaeological-site masking is performed.

## Current Step 3 Sites

| Split | Looted | Preserved | Total |
|---|---:|---:|---:|
| train | 100 | 496 | 596 |
| val | 9 | 9 | 18 |
| test | 26 | 35 | 61 |

## Loader Check Summary

- Successful combinations: 60/60
- Checked sequences, including reuse across experiments: 13500
- Image shape: [B, K, 3, image_size, image_size]. B may be smaller in the final batch.
- Label shape: [B], torch.long, values 0/1 and matched to site metadata.

| K | Strategy | Split | Status | Available | Checked | Batches | First image batch |
|---:|---|---|---|---:|---:|---:|---|
| 1 | recent_window | train | ok | 596 | 596 | 149 | [4, 1, 3, 266, 266] |
| 1 | recent_window | val | ok | 18 | 18 | 5 | [4, 1, 3, 266, 266] |
| 1 | recent_window | test | ok | 61 | 61 | 16 | [4, 1, 3, 266, 266] |
| 1 | uniform | train | ok | 596 | 596 | 149 | [4, 1, 3, 266, 266] |
| 1 | uniform | val | ok | 18 | 18 | 5 | [4, 1, 3, 266, 266] |
| 1 | uniform | test | ok | 61 | 61 | 16 | [4, 1, 3, 266, 266] |
| 1 | early_late | train | ok | 596 | 596 | 149 | [4, 1, 3, 266, 266] |
| 1 | early_late | val | ok | 18 | 18 | 5 | [4, 1, 3, 266, 266] |
| 1 | early_late | test | ok | 61 | 61 | 16 | [4, 1, 3, 266, 266] |
| 1 | random | train | ok | 596 | 596 | 149 | [4, 1, 3, 266, 266] |
| 1 | random | val | ok | 18 | 18 | 5 | [4, 1, 3, 266, 266] |
| 1 | random | test | ok | 61 | 61 | 16 | [4, 1, 3, 266, 266] |
| 2 | recent_window | train | ok | 596 | 596 | 149 | [4, 2, 3, 266, 266] |
| 2 | recent_window | val | ok | 18 | 18 | 5 | [4, 2, 3, 266, 266] |
| 2 | recent_window | test | ok | 61 | 61 | 16 | [4, 2, 3, 266, 266] |
| 2 | uniform | train | ok | 596 | 596 | 149 | [4, 2, 3, 266, 266] |
| 2 | uniform | val | ok | 18 | 18 | 5 | [4, 2, 3, 266, 266] |
| 2 | uniform | test | ok | 61 | 61 | 16 | [4, 2, 3, 266, 266] |
| 2 | early_late | train | ok | 596 | 596 | 149 | [4, 2, 3, 266, 266] |
| 2 | early_late | val | ok | 18 | 18 | 5 | [4, 2, 3, 266, 266] |
| 2 | early_late | test | ok | 61 | 61 | 16 | [4, 2, 3, 266, 266] |
| 2 | random | train | ok | 596 | 596 | 149 | [4, 2, 3, 266, 266] |
| 2 | random | val | ok | 18 | 18 | 5 | [4, 2, 3, 266, 266] |
| 2 | random | test | ok | 61 | 61 | 16 | [4, 2, 3, 266, 266] |
| 3 | recent_window | train | ok | 596 | 596 | 149 | [4, 3, 3, 266, 266] |
| 3 | recent_window | val | ok | 18 | 18 | 5 | [4, 3, 3, 266, 266] |
| 3 | recent_window | test | ok | 61 | 61 | 16 | [4, 3, 3, 266, 266] |
| 3 | uniform | train | ok | 596 | 596 | 149 | [4, 3, 3, 266, 266] |
| 3 | uniform | val | ok | 18 | 18 | 5 | [4, 3, 3, 266, 266] |
| 3 | uniform | test | ok | 61 | 61 | 16 | [4, 3, 3, 266, 266] |
| 3 | early_late | train | ok | 596 | 596 | 149 | [4, 3, 3, 266, 266] |
| 3 | early_late | val | ok | 18 | 18 | 5 | [4, 3, 3, 266, 266] |
| 3 | early_late | test | ok | 61 | 61 | 16 | [4, 3, 3, 266, 266] |
| 3 | random | train | ok | 596 | 596 | 149 | [4, 3, 3, 266, 266] |
| 3 | random | val | ok | 18 | 18 | 5 | [4, 3, 3, 266, 266] |
| 3 | random | test | ok | 61 | 61 | 16 | [4, 3, 3, 266, 266] |
| 5 | recent_window | train | ok | 596 | 596 | 149 | [4, 5, 3, 266, 266] |
| 5 | recent_window | val | ok | 18 | 18 | 5 | [4, 5, 3, 266, 266] |
| 5 | recent_window | test | ok | 61 | 61 | 16 | [4, 5, 3, 266, 266] |
| 5 | uniform | train | ok | 596 | 596 | 149 | [4, 5, 3, 266, 266] |
| 5 | uniform | val | ok | 18 | 18 | 5 | [4, 5, 3, 266, 266] |
| 5 | uniform | test | ok | 61 | 61 | 16 | [4, 5, 3, 266, 266] |
| 5 | early_late | train | ok | 596 | 596 | 149 | [4, 5, 3, 266, 266] |
| 5 | early_late | val | ok | 18 | 18 | 5 | [4, 5, 3, 266, 266] |
| 5 | early_late | test | ok | 61 | 61 | 16 | [4, 5, 3, 266, 266] |
| 5 | random | train | ok | 596 | 596 | 149 | [4, 5, 3, 266, 266] |
| 5 | random | val | ok | 18 | 18 | 5 | [4, 5, 3, 266, 266] |
| 5 | random | test | ok | 61 | 61 | 16 | [4, 5, 3, 266, 266] |
| 8 | recent_window | train | ok | 596 | 596 | 149 | [4, 8, 3, 266, 266] |
| 8 | recent_window | val | ok | 18 | 18 | 5 | [4, 8, 3, 266, 266] |
| 8 | recent_window | test | ok | 61 | 61 | 16 | [4, 8, 3, 266, 266] |
| 8 | uniform | train | ok | 596 | 596 | 149 | [4, 8, 3, 266, 266] |
| 8 | uniform | val | ok | 18 | 18 | 5 | [4, 8, 3, 266, 266] |
| 8 | uniform | test | ok | 61 | 61 | 16 | [4, 8, 3, 266, 266] |
| 8 | early_late | train | ok | 596 | 596 | 149 | [4, 8, 3, 266, 266] |
| 8 | early_late | val | ok | 18 | 18 | 5 | [4, 8, 3, 266, 266] |
| 8 | early_late | test | ok | 61 | 61 | 16 | [4, 8, 3, 266, 266] |
| 8 | random | train | ok | 596 | 596 | 149 | [4, 8, 3, 266, 266] |
| 8 | random | val | ok | 18 | 18 | 5 | [4, 8, 3, 266, 266] |
| 8 | random | test | ok | 61 | 61 | 16 | [4, 8, 3, 266, 266] |

## Errors

- None.

## Scope

- Metadata is compared with current Step 3; all selected sequences are loaded when the overall result is PASS.
- Source identity, labeling correctness, reference image-count differences, and scientific suitability of normalization require their separate checks.
- Equal random seeds support repeatable loading in the same environment; they do not guarantee identical model training across platforms.
- PASS confirms this loading workflow, not model accuracy or GPU operation.
