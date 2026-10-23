# Step 9 Lightweight Temporal Fusion Baseline Report

## Status: COMPLETE

- Run: 20261014T121946_001827Z_all_strategies_seed42
- Completed experiments: 24
- Model: shared ImageNet-pretrained ResNet18 feature extractor + GRU or attention pooling
- Backbone frozen: False (when frozen, BatchNorm statistics are also fixed)
- GRU: one layer, one direction, hidden size 256
- Attention: Linear(512,128) -> Tanh -> Linear(128,1) -> softmax over frames
- K values: [3, 5, 8]
- Fusion types: ['gru', 'attention']
- Frame selection strategies: ['recent_window', 'uniform', 'early_late', 'random']
- Labels: preserved=0, looted=1
- Batch size in sites: 2
- Image size: 266 x 266
- Dropout probability: 0.3
- Normalization: ImageNet; whole images, no site masking
- Seed reset independently for every training experiment: 42
- Epochs planned per experiment: 20
- Early stopping patience: 5
- Learning rate: 0.0001
- Weight decay: 0.0001
- Device used: cuda
- GPU name: NVIDIA GeForce GTX 1650 with Max-Q Design
- PyTorch: 2.7.1+cu118
- torchvision: 0.22.1+cu118
- Class weights [preserved, looted]: [0.6008064516129032, 2.98]
- Deterministic algorithms requested: False
- Precision: float32; no automatic mixed precision
- Checkpoint selection within each experiment: validation F1; ties keep the earlier epoch
- Configuration selection: highest best validation F1; ties: smaller K, then GRU, then strategy order recent_window, uniform, early_late, random
- Configuration selection was saved before any test predictions in this run.
- Each predefined configuration was tested once after selection was fixed.

## Checked site counts

| Split | Preserved | Looted | Total |
|---|---:|---:|---:|
| train | 496 | 100 | 596 |
| val | 9 | 9 | 18 |
| test | 35 | 26 | 61 |

## Results for predefined configurations

Selected means chosen by validation F1, not by the test score.

| K | Strategy | Fusion | Epochs | Best epoch | Best val F1 | Test accuracy | Test precision | Test recall | Test F1 | Test AUC | Test FAR | Selected |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | recent_window | gru | 8 | 3 | 0.7500 | 0.5410 | 0.4286 | 0.2308 | 0.3000 | 0.5560 | 0.2286 |  |
| 3 | recent_window | attention | 6 | 1 | 0.8235 | 0.6557 | 0.5758 | 0.7308 | 0.6441 | 0.7275 | 0.4000 |  |
| 3 | uniform | gru | 8 | 3 | 0.7500 | 0.6721 | 0.6667 | 0.4615 | 0.5455 | 0.7758 | 0.1714 |  |
| 3 | uniform | attention | 10 | 5 | 0.8750 | 0.6557 | 0.6190 | 0.5000 | 0.5532 | 0.7714 | 0.2286 |  |
| 3 | early_late | gru | 9 | 4 | 0.7778 | 0.7213 | 0.7368 | 0.5385 | 0.6222 | 0.7791 | 0.1429 |  |
| 3 | early_late | attention | 7 | 2 | 0.7778 | 0.6557 | 0.6471 | 0.4231 | 0.5116 | 0.7066 | 0.1714 |  |
| 3 | random | gru | 17 | 12 | 0.7778 | 0.6393 | 0.5714 | 0.6154 | 0.5926 | 0.6670 | 0.3429 |  |
| 3 | random | attention | 8 | 3 | 0.6667 | 0.6066 | 0.5312 | 0.6538 | 0.5862 | 0.6582 | 0.4286 |  |
| 5 | recent_window | gru | 12 | 7 | 0.7500 | 0.7049 | 0.7857 | 0.4231 | 0.5500 | 0.8363 | 0.0857 |  |
| 5 | recent_window | attention | 8 | 3 | 0.7368 | 0.6885 | 0.6296 | 0.6538 | 0.6415 | 0.7319 | 0.2857 |  |
| 5 | uniform | gru | 9 | 4 | 0.8421 | 0.7049 | 0.6111 | 0.8462 | 0.7097 | 0.8044 | 0.4000 |  |
| 5 | uniform | attention | 6 | 1 | 0.7619 | 0.6393 | 0.5455 | 0.9231 | 0.6857 | 0.7769 | 0.5714 |  |
| 5 | early_late | gru | 6 | 1 | 0.7368 | 0.5410 | 0.4444 | 0.3077 | 0.3636 | 0.6011 | 0.2857 |  |
| 5 | early_late | attention | 9 | 4 | 0.8235 | 0.7213 | 0.6667 | 0.6923 | 0.6792 | 0.8077 | 0.2571 |  |
| 5 | random | gru | 20 | 16 | 0.8000 | 0.6721 | 0.6500 | 0.5000 | 0.5652 | 0.7484 | 0.2000 |  |
| 5 | random | attention | 18 | 13 | 0.7500 | 0.6721 | 0.6500 | 0.5000 | 0.5652 | 0.7703 | 0.2000 |  |
| 8 | recent_window | gru | 13 | 8 | 0.8889 | 0.7541 | 0.7200 | 0.6923 | 0.7059 | 0.7802 | 0.2000 | yes |
| 8 | recent_window | attention | 7 | 2 | 0.7273 | 0.6230 | 0.5455 | 0.6923 | 0.6102 | 0.6901 | 0.4286 |  |
| 8 | uniform | gru | 6 | 1 | 0.7778 | 0.7377 | 0.6471 | 0.8462 | 0.7333 | 0.7604 | 0.3429 |  |
| 8 | uniform | attention | 8 | 3 | 0.7500 | 0.6557 | 0.6000 | 0.5769 | 0.5882 | 0.7593 | 0.2857 |  |
| 8 | early_late | gru | 8 | 3 | 0.8235 | 0.7377 | 0.7083 | 0.6538 | 0.6800 | 0.7560 | 0.2000 |  |
| 8 | early_late | attention | 7 | 2 | 0.8235 | 0.6885 | 0.7333 | 0.4231 | 0.5366 | 0.7451 | 0.1143 |  |
| 8 | random | gru | 13 | 8 | 0.7500 | 0.7377 | 0.7273 | 0.6154 | 0.6667 | 0.7681 | 0.1714 |  |
| 8 | random | attention | 8 | 3 | 0.7059 | 0.6066 | 0.5625 | 0.3462 | 0.4286 | 0.6077 | 0.2000 |  |

## Test counts and loss

| K | Strategy | Fusion | Samples | TN | FP | FN | TP | Weighted loss |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 3 | recent_window | gru | 61 | 27 | 8 | 20 | 6 | 1.1051 |
| 3 | recent_window | attention | 61 | 21 | 14 | 7 | 19 | 0.6249 |
| 3 | uniform | gru | 61 | 29 | 6 | 14 | 12 | 0.7052 |
| 3 | uniform | attention | 61 | 27 | 8 | 13 | 13 | 0.8978 |
| 3 | early_late | gru | 61 | 30 | 5 | 12 | 14 | 0.5500 |
| 3 | early_late | attention | 61 | 29 | 6 | 15 | 11 | 0.7469 |
| 3 | random | gru | 61 | 23 | 12 | 10 | 16 | 0.6944 |
| 3 | random | attention | 61 | 20 | 15 | 9 | 17 | 0.7956 |
| 5 | recent_window | gru | 61 | 32 | 3 | 15 | 11 | 0.9699 |
| 5 | recent_window | attention | 61 | 25 | 10 | 9 | 17 | 1.0151 |
| 5 | uniform | gru | 61 | 21 | 14 | 4 | 22 | 0.4744 |
| 5 | uniform | attention | 61 | 15 | 20 | 2 | 24 | 0.4691 |
| 5 | early_late | gru | 61 | 25 | 10 | 18 | 8 | 0.8171 |
| 5 | early_late | attention | 61 | 26 | 9 | 8 | 18 | 0.5226 |
| 5 | random | gru | 61 | 28 | 7 | 13 | 13 | 0.8809 |
| 5 | random | attention | 61 | 28 | 7 | 13 | 13 | 0.7550 |
| 8 | recent_window | gru | 61 | 28 | 7 | 8 | 18 | 0.6422 |
| 8 | recent_window | attention | 61 | 20 | 15 | 8 | 18 | 0.6644 |
| 8 | uniform | gru | 61 | 23 | 12 | 4 | 22 | 0.5601 |
| 8 | uniform | attention | 61 | 25 | 10 | 11 | 15 | 0.6496 |
| 8 | early_late | gru | 61 | 28 | 7 | 9 | 17 | 0.6160 |
| 8 | early_late | attention | 61 | 31 | 4 | 15 | 11 | 0.7515 |
| 8 | random | gru | 61 | 29 | 6 | 10 | 16 | 0.6017 |
| 8 | random | attention | 61 | 28 | 7 | 17 | 9 | 0.9652 |

## Configuration selected using validation

- Experiment: K8_recent_window_gru
- Strategy: recent_window
- Best validation F1: 0.8889
- Test F1 of that preselected configuration: 0.7059
- Model weights: `E:\research_university\Research\looting-detection-research\models\lightweight_temporal_fusion\20261014T121946_001827Z_all_strategies_seed42\K8_recent_window_gru_best.pth`

## Outputs

- Model folder: `E:\research_university\Research\looting-detection-research\models\lightweight_temporal_fusion\20261014T121946_001827Z_all_strategies_seed42`
- lightweight_temporal_fusion_summary_results.csv
- validation_results.csv
- selected_configuration.json
- run_metadata.json
- Each experiment folder contains history, metadata, best validation predictions/metrics and test predictions/metrics.

## Scope

- Current Step 3/4 metadata was checked; all configurations use the same site memberships.
- Each full epoch/evaluation checks sample coverage, site_uid, label, K, tensor shapes and finite values.
- This is one fixed-split suite, not cross-validation. Validation has 18 sites.
- Selecting among many configurations with 18 validation sites can overfit that validation set.
- Random frame selection uses the saved Step 4 indices; changing the training seed does not resample them.
- GRU uses the supplied chronological frame order. Attention pooling does not encode frame order.
- Neither fusion method receives calendar dates or time gaps. Few frames may span several years.
- This suite evaluates fixed K=3/5/8 subsets, not full sequences or mixed sequence lengths.
- Attention weights are not proof of which archaeological changes caused a prediction.
- Parameter counts are recorded per experiment; runtime/memory efficiency was not benchmarked.
- Batch size is the number of sites. The CNN processes batch_size * K images together.
- Training metrics use augmented images and changing weights within each epoch.
- Weighted loss is accumulated using target class-weight sums.
- No site masking or image registration is performed. Source identities/labels require their separate audit.
- Seeds do not guarantee identical results across environments; one run does not establish statistical significance.
