# Step 10 Proposed Change-Aware Temporal Fusion Report

## Status: COMPLETE

- Run: 20261018T103147_718144Z_all_strategies_seed42
- Completed training experiments: 48
- Model: shared ImageNet-pretrained ResNet18 + appearance/reference-difference projection + one-layer GRU
- K values: [3, 5, 8]
- Strategies: ['recent_window', 'uniform', 'early_late', 'random']
- Variants: ['no_change_no_drop', 'change_no_drop', 'no_change_frame_drop', 'proposed']
- Labels: preserved=0, looted=1
- All variants have identical parameter shapes; no-change variants replace the change half with zeros.
- GRU and projection width: 256
- Backbone frozen: False (frozen BatchNorm remains in evaluation mode)
- Image size: 266 x 266; normalization: ImageNet
- Batch size in sites: 2
- Training frame-drop probability for drop variants: 0.25
- Frame removal happens before the CNN; the loader still requires all source image files.
- The reference and at least one target are retained. The probability applies to non-reference frames.
- Projection dropout: 0.2; classifier dropout: 0.3
- Seed reset for every experiment: 42
- Epochs planned: 20; early stopping patience: 5
- Learning rate: 0.0001; weight decay: 0.0001
- Gradient clipping maximum norm: 5.0
- Device used: cuda; GPU: NVIDIA GeForce GTX 1650 with Max-Q Design
- PyTorch: 2.7.1+cu118; torchvision: 0.22.1+cu118
- Class weights [preserved, looted]: [0.6008064516129032, 2.98]
- Deterministic algorithms requested: False; precision: float32
- Checkpoint selection: highest validation F1; ties retain the earlier epoch and count toward patience.
- Configuration selection: among proposed variants only: highest best validation F1; ties: smaller K, then strategy order recent_window, uniform, early_late, random
- Selection was saved before any test predictions were generated in this run.
- Each predefined configuration was tested once with all K frames; robustness evaluations are separate.
- Prediction rule: larger logit; ties select preserved.

## Checked site counts

| Split | Preserved | Looted | Total |
|---|---:|---:|---:|
| train | 496 | 100 | 596 |
| val | 9 | 9 | 18 |
| test | 35 | 26 | 61 |

## Predefined experiments

| K | Strategy | Variant | Epochs | Best epoch | Best val F1 | Test accuracy | Test precision | Test recall | Test F1 | Test AUC | Test FAR | Selected |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | recent_window | no_change_no_drop | 9 | 4 | 0.7500 | 0.6885 | 0.7333 | 0.4231 | 0.5366 | 0.7824 | 0.1143 |  |
| 3 | recent_window | change_no_drop | 6 | 1 | 0.7500 | 0.5246 | 0.3636 | 0.1538 | 0.2162 | 0.6374 | 0.2000 |  |
| 3 | recent_window | no_change_frame_drop | 14 | 9 | 0.8000 | 0.7541 | 0.7619 | 0.6154 | 0.6809 | 0.8110 | 0.1429 |  |
| 3 | recent_window | proposed | 8 | 3 | 0.7059 | 0.6393 | 0.6429 | 0.3462 | 0.4500 | 0.6604 | 0.1429 |  |
| 3 | uniform | no_change_no_drop | 8 | 3 | 0.6250 | 0.7049 | 0.9000 | 0.3462 | 0.5000 | 0.7978 | 0.0286 |  |
| 3 | uniform | change_no_drop | 8 | 3 | 0.7500 | 0.5902 | 0.6000 | 0.1154 | 0.1935 | 0.7692 | 0.0571 |  |
| 3 | uniform | no_change_frame_drop | 8 | 3 | 0.7500 | 0.7213 | 0.6957 | 0.6154 | 0.6531 | 0.8000 | 0.2000 |  |
| 3 | uniform | proposed | 13 | 8 | 0.7368 | 0.7213 | 0.6957 | 0.6154 | 0.6531 | 0.7901 | 0.2000 |  |
| 3 | early_late | no_change_no_drop | 9 | 4 | 0.7778 | 0.7049 | 0.6333 | 0.7308 | 0.6786 | 0.7813 | 0.3143 |  |
| 3 | early_late | change_no_drop | 9 | 4 | 0.8000 | 0.7049 | 0.7857 | 0.4231 | 0.5500 | 0.6143 | 0.0857 |  |
| 3 | early_late | no_change_frame_drop | 17 | 12 | 0.8235 | 0.7213 | 0.6800 | 0.6538 | 0.6667 | 0.7593 | 0.2286 |  |
| 3 | early_late | proposed | 11 | 6 | 0.8000 | 0.6557 | 0.7778 | 0.2692 | 0.4000 | 0.7626 | 0.0571 | yes |
| 3 | random | no_change_no_drop | 9 | 4 | 0.3636 | 0.5738 | 0.5000 | 0.1923 | 0.2778 | 0.5440 | 0.1429 |  |
| 3 | random | change_no_drop | 9 | 4 | 0.2000 | 0.5574 | 0.4286 | 0.1154 | 0.1818 | 0.5703 | 0.1143 |  |
| 3 | random | no_change_frame_drop | 6 | 1 | 0.0000 | 0.5738 | 0.0000 | 0.0000 | 0.0000 | 0.5703 | 0.0000 |  |
| 3 | random | proposed | 6 | 1 | 0.0000 | 0.5738 | 0.0000 | 0.0000 | 0.0000 | 0.5615 | 0.0000 |  |
| 5 | recent_window | no_change_no_drop | 9 | 4 | 0.7778 | 0.6885 | 0.6296 | 0.6538 | 0.6415 | 0.8286 | 0.2857 |  |
| 5 | recent_window | change_no_drop | 9 | 4 | 0.7500 | 0.6557 | 0.7778 | 0.2692 | 0.4000 | 0.7077 | 0.0571 |  |
| 5 | recent_window | no_change_frame_drop | 8 | 3 | 0.7500 | 0.6721 | 0.7500 | 0.3462 | 0.4737 | 0.7604 | 0.0857 |  |
| 5 | recent_window | proposed | 10 | 5 | 0.7778 | 0.7213 | 0.8000 | 0.4615 | 0.5854 | 0.8165 | 0.0857 |  |
| 5 | uniform | no_change_no_drop | 17 | 12 | 0.7500 | 0.7541 | 0.7619 | 0.6154 | 0.6809 | 0.8264 | 0.1429 |  |
| 5 | uniform | change_no_drop | 10 | 5 | 0.7500 | 0.6557 | 0.6471 | 0.4231 | 0.5116 | 0.6692 | 0.1714 |  |
| 5 | uniform | no_change_frame_drop | 8 | 3 | 0.7778 | 0.5738 | 0.5000 | 0.1923 | 0.2778 | 0.6747 | 0.1429 |  |
| 5 | uniform | proposed | 16 | 11 | 0.7500 | 0.7541 | 0.7391 | 0.6538 | 0.6939 | 0.8319 | 0.1714 |  |
| 5 | early_late | no_change_no_drop | 8 | 3 | 0.7778 | 0.7049 | 0.7000 | 0.5385 | 0.6087 | 0.7692 | 0.1714 |  |
| 5 | early_late | change_no_drop | 8 | 3 | 0.7500 | 0.6066 | 0.5625 | 0.3462 | 0.4286 | 0.6681 | 0.2000 |  |
| 5 | early_late | no_change_frame_drop | 13 | 8 | 0.8235 | 0.7049 | 0.6667 | 0.6154 | 0.6400 | 0.7989 | 0.2286 |  |
| 5 | early_late | proposed | 14 | 9 | 0.8000 | 0.6393 | 0.7000 | 0.2692 | 0.3889 | 0.6890 | 0.0857 |  |
| 5 | random | no_change_no_drop | 8 | 3 | 0.5714 | 0.5738 | 0.5000 | 0.4231 | 0.4583 | 0.5681 | 0.3143 |  |
| 5 | random | change_no_drop | 8 | 3 | 0.4286 | 0.5574 | 0.4615 | 0.2308 | 0.3077 | 0.6275 | 0.2000 |  |
| 5 | random | no_change_frame_drop | 8 | 3 | 0.3333 | 0.4918 | 0.2222 | 0.0769 | 0.1143 | 0.5440 | 0.2000 |  |
| 5 | random | proposed | 8 | 3 | 0.3077 | 0.6393 | 0.6250 | 0.3846 | 0.4762 | 0.6967 | 0.1714 |  |
| 8 | recent_window | no_change_no_drop | 11 | 6 | 0.8000 | 0.6721 | 0.8000 | 0.3077 | 0.4444 | 0.8088 | 0.0571 |  |
| 8 | recent_window | change_no_drop | 8 | 3 | 0.8000 | 0.5902 | 0.5714 | 0.1538 | 0.2424 | 0.7407 | 0.0857 |  |
| 8 | recent_window | no_change_frame_drop | 19 | 14 | 0.7778 | 0.6393 | 0.5833 | 0.5385 | 0.5600 | 0.7451 | 0.2857 |  |
| 8 | recent_window | proposed | 9 | 4 | 0.7778 | 0.7213 | 0.7368 | 0.5385 | 0.6222 | 0.7868 | 0.1429 |  |
| 8 | uniform | no_change_no_drop | 6 | 1 | 0.8000 | 0.6393 | 0.6667 | 0.3077 | 0.4211 | 0.6681 | 0.1143 |  |
| 8 | uniform | change_no_drop | 16 | 11 | 0.8000 | 0.6885 | 0.7059 | 0.4615 | 0.5581 | 0.8198 | 0.1429 |  |
| 8 | uniform | no_change_frame_drop | 9 | 4 | 0.7778 | 0.6066 | 0.5417 | 0.5000 | 0.5200 | 0.6978 | 0.3143 |  |
| 8 | uniform | proposed | 20 | 15 | 0.8000 | 0.5574 | 0.4706 | 0.3077 | 0.3721 | 0.6670 | 0.2571 |  |
| 8 | early_late | no_change_no_drop | 9 | 4 | 0.7778 | 0.6721 | 0.6071 | 0.6538 | 0.6296 | 0.7670 | 0.3143 |  |
| 8 | early_late | change_no_drop | 9 | 4 | 0.8235 | 0.6885 | 0.6296 | 0.6538 | 0.6415 | 0.7176 | 0.2857 |  |
| 8 | early_late | no_change_frame_drop | 6 | 1 | 0.7059 | 0.5246 | 0.3636 | 0.1538 | 0.2162 | 0.5813 | 0.2000 |  |
| 8 | early_late | proposed | 8 | 3 | 0.7778 | 0.6066 | 0.6000 | 0.2308 | 0.3333 | 0.6571 | 0.1143 |  |
| 8 | random | no_change_no_drop | 13 | 8 | 0.6667 | 0.7049 | 0.7222 | 0.5000 | 0.5909 | 0.7374 | 0.1429 |  |
| 8 | random | change_no_drop | 6 | 1 | 0.3636 | 0.5902 | 0.6667 | 0.0769 | 0.1379 | 0.5681 | 0.0286 |  |
| 8 | random | no_change_frame_drop | 8 | 3 | 0.6250 | 0.5410 | 0.4000 | 0.1538 | 0.2222 | 0.5473 | 0.1714 |  |
| 8 | random | proposed | 14 | 9 | 0.5714 | 0.6557 | 0.6190 | 0.5000 | 0.5532 | 0.7264 | 0.2286 |  |

## Test counts and weighted loss

| Experiment | Samples | TN | FP | FN | TP | Loss | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| K3_recent_window_no_change_no_drop | 61 | 31 | 4 | 15 | 11 | 0.9218 | 11836226 |
| K3_recent_window_change_no_drop | 61 | 28 | 7 | 22 | 4 | 1.2289 | 11836226 |
| K3_recent_window_no_change_frame_drop | 61 | 30 | 5 | 10 | 16 | 0.9688 | 11836226 |
| K3_recent_window_proposed | 61 | 30 | 5 | 17 | 9 | 1.4924 | 11836226 |
| K3_uniform_no_change_no_drop | 61 | 34 | 1 | 17 | 9 | 1.1214 | 11836226 |
| K3_uniform_change_no_drop | 61 | 33 | 2 | 23 | 3 | 1.0511 | 11836226 |
| K3_uniform_no_change_frame_drop | 61 | 28 | 7 | 10 | 16 | 0.7503 | 11836226 |
| K3_uniform_proposed | 61 | 28 | 7 | 10 | 16 | 0.7205 | 11836226 |
| K3_early_late_no_change_no_drop | 61 | 24 | 11 | 7 | 19 | 0.6372 | 11836226 |
| K3_early_late_change_no_drop | 61 | 32 | 3 | 15 | 11 | 1.3905 | 11836226 |
| K3_early_late_no_change_frame_drop | 61 | 27 | 8 | 9 | 17 | 0.8146 | 11836226 |
| K3_early_late_proposed | 61 | 33 | 2 | 19 | 7 | 1.5146 | 11836226 |
| K3_random_no_change_no_drop | 61 | 30 | 5 | 21 | 5 | 1.4039 | 11836226 |
| K3_random_change_no_drop | 61 | 31 | 4 | 23 | 3 | 1.0502 | 11836226 |
| K3_random_no_change_frame_drop | 61 | 35 | 0 | 26 | 0 | 1.1632 | 11836226 |
| K3_random_proposed | 61 | 35 | 0 | 26 | 0 | 1.6169 | 11836226 |
| K5_recent_window_no_change_no_drop | 61 | 25 | 10 | 9 | 17 | 0.5779 | 11836226 |
| K5_recent_window_change_no_drop | 61 | 33 | 2 | 19 | 7 | 2.1537 | 11836226 |
| K5_recent_window_no_change_frame_drop | 61 | 32 | 3 | 17 | 9 | 1.0520 | 11836226 |
| K5_recent_window_proposed | 61 | 32 | 3 | 14 | 12 | 1.1220 | 11836226 |
| K5_uniform_no_change_no_drop | 61 | 30 | 5 | 10 | 16 | 0.8797 | 11836226 |
| K5_uniform_change_no_drop | 61 | 29 | 6 | 15 | 11 | 1.3360 | 11836226 |
| K5_uniform_no_change_frame_drop | 61 | 30 | 5 | 21 | 5 | 0.8407 | 11836226 |
| K5_uniform_proposed | 61 | 29 | 6 | 9 | 17 | 0.9761 | 11836226 |
| K5_early_late_no_change_no_drop | 61 | 29 | 6 | 12 | 14 | 1.0096 | 11836226 |
| K5_early_late_change_no_drop | 61 | 28 | 7 | 17 | 9 | 1.2747 | 11836226 |
| K5_early_late_no_change_frame_drop | 61 | 27 | 8 | 10 | 16 | 0.8512 | 11836226 |
| K5_early_late_proposed | 61 | 32 | 3 | 19 | 7 | 1.7671 | 11836226 |
| K5_random_no_change_no_drop | 61 | 24 | 11 | 15 | 11 | 1.1100 | 11836226 |
| K5_random_change_no_drop | 61 | 28 | 7 | 20 | 6 | 1.0977 | 11836226 |
| K5_random_no_change_frame_drop | 61 | 28 | 7 | 24 | 2 | 1.0615 | 11836226 |
| K5_random_proposed | 61 | 29 | 6 | 16 | 10 | 0.8134 | 11836226 |
| K8_recent_window_no_change_no_drop | 61 | 33 | 2 | 18 | 8 | 1.9081 | 11836226 |
| K8_recent_window_change_no_drop | 61 | 32 | 3 | 22 | 4 | 1.4629 | 11836226 |
| K8_recent_window_no_change_frame_drop | 61 | 25 | 10 | 12 | 14 | 1.2150 | 11836226 |
| K8_recent_window_proposed | 61 | 30 | 5 | 12 | 14 | 1.0045 | 11836226 |
| K8_uniform_no_change_no_drop | 61 | 31 | 4 | 18 | 8 | 1.0433 | 11836226 |
| K8_uniform_change_no_drop | 61 | 30 | 5 | 14 | 12 | 1.2979 | 11836226 |
| K8_uniform_no_change_frame_drop | 61 | 24 | 11 | 13 | 13 | 0.7444 | 11836226 |
| K8_uniform_proposed | 61 | 26 | 9 | 18 | 8 | 1.3983 | 11836226 |
| K8_early_late_no_change_no_drop | 61 | 24 | 11 | 9 | 17 | 0.9132 | 11836226 |
| K8_early_late_change_no_drop | 61 | 25 | 10 | 9 | 17 | 0.7744 | 11836226 |
| K8_early_late_no_change_frame_drop | 61 | 28 | 7 | 22 | 4 | 1.2065 | 11836226 |
| K8_early_late_proposed | 61 | 31 | 4 | 20 | 6 | 1.3173 | 11836226 |
| K8_random_no_change_no_drop | 61 | 30 | 5 | 13 | 13 | 0.9919 | 11836226 |
| K8_random_change_no_drop | 61 | 34 | 1 | 24 | 2 | 1.5797 | 11836226 |
| K8_random_no_change_frame_drop | 61 | 29 | 6 | 22 | 4 | 1.2357 | 11836226 |
| K8_random_proposed | 61 | 27 | 8 | 13 | 13 | 0.9246 | 11836226 |

## Validation-selected proposed configuration

- Experiment: K3_early_late_proposed
- Best validation F1: 0.8000
- Test F1 of that preselected configuration: 0.4000
- Weights: `E:\research_university\Research\looting-detection-research\models\proposed_change_aware_fusion\20261018T103147_718144Z_all_strategies_seed42\K3_early_late_proposed_best.pth`

## Paired simulated missing-observation robustness

Both models use the same K, strategy, sites and retained-frame masks. p=0 reuses the normal test result.
Standard deviations describe variation across missing-frame masks, not statistical significance or training-seed variation.

| Variant | Drop probability | Repeats | Mean retained frames | Actual dropped fraction | F1 mean | F1 std | Recall mean | FAR mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| change_no_drop | 0.00 | 1 | 3.000 | 0.0000 | 0.5500 | 0.0000 | 0.4231 | 0.0857 |
| change_no_drop | 0.25 | 5 | 2.541 | 0.1530 | 0.5077 | 0.0199 | 0.3769 | 0.0800 |
| change_no_drop | 0.50 | 5 | 2.243 | 0.2525 | 0.4889 | 0.0358 | 0.3615 | 0.0857 |
| proposed | 0.00 | 1 | 3.000 | 0.0000 | 0.4000 | 0.0000 | 0.2692 | 0.0571 |
| proposed | 0.25 | 5 | 2.541 | 0.1530 | 0.3011 | 0.0651 | 0.1923 | 0.0571 |
| proposed | 0.50 | 5 | 2.243 | 0.2525 | 0.2718 | 0.0619 | 0.1692 | 0.0514 |

## Baseline comparison

Comparison status: COMPLETE.
Runs were fixed before Step 10 training using compatibility and timestamp, never test scores.
Training settings differ across steps and are recorded in the comparison CSV. These are descriptive comparisons.

| Model | K | Strategy | Method | Test F1 | Test accuracy | Test recall | Test FAR |
|---|---:|---|---|---:|---:|---:|---:|
| Step 6 | 1 | recent_window | single_frame | 0.6000 | 0.6066 | 0.6923 | 0.4571 |
| Step 7 | 2 | early_late | two_frame_change | 0.7000 | 0.7049 | 0.8077 | 0.3714 |
| Step 8 | 5 | random | max | 0.4889 | 0.6230 | 0.4231 | 0.2286 |
| Step 9 | 8 | recent_window | gru | 0.7059 | 0.7541 | 0.6923 | 0.2000 |
| Step 10 | 3 | early_late | change features + GRU + frame drop | 0.4000 | 0.6557 | 0.2692 | 0.0571 |

## Files

- Model folder: `E:\research_university\Research\looting-detection-research\models\proposed_change_aware_fusion\20261018T103147_718144Z_all_strategies_seed42`
- Report folder: `E:\research_university\Research\looting-detection-research\results\proposed_change_aware_fusion\20261018T103147_718144Z_all_strategies_seed42`
- step10_summary_results.csv; step10_ablation_results.csv; step10_k_analysis_results.csv
- validation_results.csv; selected_configuration.json; run_metadata.json
- step10_frame_drop_robustness.csv; step10_robustness_summary.csv; robustness/ predictions
- final_model_comparison.csv; baseline_comparison_status.json; baseline_inputs.json
- Every experiment has history, metadata, validation predictions/metrics and test predictions/metrics.

## Scope and limitations

- Current Step 3/4 source metadata and site memberships were checked; train=596, val=18, test=61.
- Every epoch checks coverage, identities, labels, shapes and finite values. Loss uses target-weight sums.
- This is one fixed split with one training seed, not cross-validation or a statistical significance study.
- Selecting among many configurations with only 18 validation sites can overfit the validation set.
- Existing test exposure from earlier experiments cannot be undone by this script.
- Ablations share architecture and parameter shapes; zero change inputs do not have the same effective information/capacity.
- The first selected reference image may already show looting. Feature differences are not looting-specific explanations.
- GRU uses chronological order but receives no dates or time gaps; a few frames may span many years.
- K3/K5/K8 subsets and simulated dropping are evaluated; full sequences, 100-frame input and naturally mixed lengths are not.
- Reference loss, cloud corruption and missing files on disk are not simulated.
- Random strategy uses saved Step 4 selections. Training masks and image selection are different operations.
- Whole images are used, without site masking or registration. Source identities and labels need separate auditing.
- Parameter counts are recorded; runtime, energy use and GPU-memory efficiency are not benchmarked.
- Metadata hashes do not verify every raw image's content. Images must remain unchanged during experiments.
- Training metrics use augmentation and changing model weights. Seeds do not ensure identical results across platforms.
