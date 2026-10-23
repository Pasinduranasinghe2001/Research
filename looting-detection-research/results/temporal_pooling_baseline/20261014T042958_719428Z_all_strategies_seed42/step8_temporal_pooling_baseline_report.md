# Step 8 Temporal Pooling Baseline Report

## Status: COMPLETE

- Run: 20261014T042958_719428Z_all_strategies_seed42
- Completed experiments: 24
- Model: shared ImageNet-pretrained ResNet18 feature extractor; all parameters fine-tuned
- K values: [3, 5, 8]
- Pooling types: ['mean', 'max']
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
- Configuration selection: highest best validation F1; ties: smaller K, then mean pooling, then strategy order recent_window, uniform, early_late, random
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

| K | Strategy | Pooling | Epochs | Best epoch | Best val F1 | Test accuracy | Test precision | Test recall | Test F1 | Test AUC | Test FAR | Selected |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | recent_window | mean | 14 | 9 | 0.8421 | 0.7377 | 0.6316 | 0.9231 | 0.7500 | 0.8593 | 0.4000 |  |
| 3 | recent_window | max | 10 | 5 | 0.8000 | 0.6721 | 0.7500 | 0.3462 | 0.4737 | 0.7473 | 0.0857 |  |
| 3 | uniform | mean | 9 | 4 | 0.7368 | 0.6393 | 0.5500 | 0.8462 | 0.6667 | 0.7604 | 0.5143 |  |
| 3 | uniform | max | 9 | 4 | 0.8750 | 0.7541 | 0.7391 | 0.6538 | 0.6939 | 0.7648 | 0.1714 |  |
| 3 | early_late | mean | 9 | 4 | 0.7778 | 0.7705 | 0.7000 | 0.8077 | 0.7500 | 0.8220 | 0.2571 |  |
| 3 | early_late | max | 6 | 1 | 0.7778 | 0.6885 | 0.6522 | 0.5769 | 0.6122 | 0.7209 | 0.2286 |  |
| 3 | random | mean | 9 | 4 | 0.7619 | 0.5902 | 0.5172 | 0.5769 | 0.5455 | 0.6473 | 0.4000 |  |
| 3 | random | max | 7 | 2 | 0.7500 | 0.6066 | 0.5556 | 0.3846 | 0.4545 | 0.5846 | 0.2286 |  |
| 5 | recent_window | mean | 8 | 3 | 0.7500 | 0.6885 | 0.7692 | 0.3846 | 0.5128 | 0.7923 | 0.0857 |  |
| 5 | recent_window | max | 9 | 4 | 0.8000 | 0.6066 | 0.5278 | 0.7308 | 0.6129 | 0.7780 | 0.4857 |  |
| 5 | uniform | mean | 8 | 3 | 0.8235 | 0.7213 | 0.6800 | 0.6538 | 0.6667 | 0.7791 | 0.2286 |  |
| 5 | uniform | max | 10 | 5 | 0.8235 | 0.6393 | 0.6250 | 0.3846 | 0.4762 | 0.7110 | 0.1714 |  |
| 5 | early_late | mean | 9 | 4 | 0.7778 | 0.7377 | 0.6667 | 0.7692 | 0.7143 | 0.7736 | 0.2857 |  |
| 5 | early_late | max | 8 | 3 | 0.7500 | 0.6557 | 0.7273 | 0.3077 | 0.4324 | 0.7736 | 0.0857 |  |
| 5 | random | mean | 13 | 8 | 0.7778 | 0.6885 | 0.6061 | 0.7692 | 0.6780 | 0.7505 | 0.3714 |  |
| 5 | random | max | 13 | 8 | 0.8889 | 0.6230 | 0.5789 | 0.4231 | 0.4889 | 0.6824 | 0.2286 | yes |
| 8 | recent_window | mean | 9 | 4 | 0.7368 | 0.6721 | 0.5938 | 0.7308 | 0.6552 | 0.7560 | 0.3714 |  |
| 8 | recent_window | max | 9 | 4 | 0.7778 | 0.7049 | 0.7000 | 0.5385 | 0.6087 | 0.7846 | 0.1714 |  |
| 8 | uniform | mean | 6 | 1 | 0.7778 | 0.6885 | 0.6000 | 0.8077 | 0.6885 | 0.7165 | 0.4000 |  |
| 8 | uniform | max | 8 | 3 | 0.7500 | 0.6230 | 0.6364 | 0.2692 | 0.3784 | 0.5736 | 0.1143 |  |
| 8 | early_late | mean | 9 | 4 | 0.8000 | 0.6721 | 0.5750 | 0.8846 | 0.6970 | 0.7857 | 0.4857 |  |
| 8 | early_late | max | 6 | 1 | 0.6667 | 0.4918 | 0.4000 | 0.3846 | 0.3922 | 0.5846 | 0.4286 |  |
| 8 | random | mean | 17 | 12 | 0.8421 | 0.6721 | 0.6071 | 0.6538 | 0.6296 | 0.7440 | 0.3143 |  |
| 8 | random | max | 18 | 13 | 0.8889 | 0.6557 | 0.6316 | 0.4615 | 0.5333 | 0.7758 | 0.2000 |  |

## Test counts and loss

| K | Strategy | Pooling | Samples | TN | FP | FN | TP | Weighted loss |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 3 | recent_window | mean | 61 | 21 | 14 | 2 | 24 | 0.3470 |
| 3 | recent_window | max | 61 | 32 | 3 | 17 | 9 | 0.8689 |
| 3 | uniform | mean | 61 | 17 | 18 | 4 | 22 | 0.5078 |
| 3 | uniform | max | 61 | 29 | 6 | 9 | 17 | 0.6114 |
| 3 | early_late | mean | 61 | 26 | 9 | 5 | 21 | 0.4657 |
| 3 | early_late | max | 61 | 27 | 8 | 11 | 15 | 0.7159 |
| 3 | random | mean | 61 | 21 | 14 | 11 | 15 | 0.6521 |
| 3 | random | max | 61 | 27 | 8 | 16 | 10 | 0.8565 |
| 5 | recent_window | mean | 61 | 32 | 3 | 16 | 10 | 0.7470 |
| 5 | recent_window | max | 61 | 18 | 17 | 7 | 19 | 0.4806 |
| 5 | uniform | mean | 61 | 27 | 8 | 9 | 17 | 0.6406 |
| 5 | uniform | max | 61 | 29 | 6 | 16 | 10 | 0.9242 |
| 5 | early_late | mean | 61 | 25 | 10 | 6 | 20 | 0.5550 |
| 5 | early_late | max | 61 | 32 | 3 | 18 | 8 | 1.0262 |
| 5 | random | mean | 61 | 22 | 13 | 6 | 20 | 0.5747 |
| 5 | random | max | 61 | 27 | 8 | 15 | 11 | 0.7494 |
| 8 | recent_window | mean | 61 | 22 | 13 | 7 | 19 | 0.5739 |
| 8 | recent_window | max | 61 | 29 | 6 | 12 | 14 | 0.6284 |
| 8 | uniform | mean | 61 | 21 | 14 | 5 | 21 | 0.5615 |
| 8 | uniform | max | 61 | 31 | 4 | 19 | 7 | 1.2013 |
| 8 | early_late | mean | 61 | 18 | 17 | 3 | 23 | 0.4576 |
| 8 | early_late | max | 61 | 20 | 15 | 16 | 10 | 0.7694 |
| 8 | random | mean | 61 | 24 | 11 | 9 | 17 | 0.7516 |
| 8 | random | max | 61 | 28 | 7 | 14 | 12 | 0.7415 |

## Configuration selected using validation

- Experiment: K5_random_max
- Strategy: random
- Best validation F1: 0.8889
- Test F1 of that preselected configuration: 0.4889
- Model weights: `E:\research_university\Research\looting-detection-research\models\temporal_pooling_baseline\20261014T042958_719428Z_all_strategies_seed42\K5_random_max_best.pth`

## Outputs

- Model folder: `E:\research_university\Research\looting-detection-research\models\temporal_pooling_baseline\20261014T042958_719428Z_all_strategies_seed42`
- temporal_pooling_summary_results.csv
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
- Mean/max pooling does not explicitly model frame order; few frames may span several years.
- Batch size is the number of sites. The CNN processes batch_size * K images together.
- Training metrics use augmented images and changing weights within each epoch.
- Weighted loss is accumulated using target class-weight sums.
- No site masking or image registration is performed. Source identities/labels require their separate audit.
- Seeds do not guarantee identical results across environments; one run does not establish statistical significance.
