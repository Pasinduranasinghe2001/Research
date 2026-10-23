# Step 6 Single-Frame CNN Baseline Report

## Status: COMPLETE

- Run: 20261013T184026_546585Z_recent_window_seed42
- Model: ImageNet-pretrained ResNet18; all parameters fine-tuned
- Input: K=1; strategy=recent_window
- Labels: preserved=0; looted=1
- Image size: 266 x 266
- Normalization: ImageNet; whole images, no site masking
- Batch size: 16
- Seed: 42
- Epochs planned: 20
- Epochs completed: 10
- Best epoch: 5
- Best validation F1: 0.8421
- Stopping reason: early stopping: validation F1 did not improve
- Learning rate: 0.0001
- Weight decay: 0.0001
- Device used: cuda
- GPU name: NVIDIA GeForce GTX 1650 with Max-Q Design
- PyTorch: 2.7.1+cu118
- torchvision: 0.22.1+cu118
- Class weights [preserved, looted]: [0.6008064516129032, 2.98]
- Prediction rule: larger of the two logits (ties select preserved)
- Checkpoint selection: validation F1; test evaluated once after selection
- Equal F1 scores keep the earlier checkpoint and count toward patience.

## Checked site counts

| Split | Preserved | Looted | Total |
|---|---:|---:|---:|
| train | 496 | 100 | 596 |
| val | 9 | 9 | 18 |
| test | 35 | 26 | 61 |

## Test results

- accuracy: 0.6066
- precision: 0.5294
- recall: 0.6923
- f1: 0.6000
- roc_auc: 0.7341
- false_alarm_rate: 0.4571
- loss: 0.6112
- samples: 61
- tn: 19
- fp: 16
- fn: 8
- tp: 18

## Files in this run

- best_model.pth
- training_history.csv
- test_predictions.csv
- test_metrics.json
- run_metadata.json

## Scope

- Current Step 3/4 metadata checked before training; no site_uid split overlap.
- This is one fixed-split experiment, not cross-validation.
- Validation has 18 sites; results may vary with seed and validation partition.
- Training metrics use augmented images and changing model weights within each epoch.
- Loss combines weighted batch losses using the sum of target class weights.
- Site identities and labels are trusted from the supplied source metadata.
- Seed/determinism settings do not guarantee identical results across platforms.
