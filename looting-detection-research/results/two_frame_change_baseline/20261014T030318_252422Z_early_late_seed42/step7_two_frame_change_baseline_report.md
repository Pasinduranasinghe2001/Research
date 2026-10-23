# Step 7 Two-Frame Change Baseline Report

## Status: COMPLETE

- Run: 20261014T030318_252422Z_early_late_seed42
- Model: ImageNet-pretrained ResNet18; all parameters fine-tuned
- Input: K=2; strategy=early_late
- Labels: preserved=0; looted=1
- Input representation: early RGB + recent RGB + absolute difference; 9 channels
- Difference is calculated after identical ImageNet normalization of both frames.
- First-layer initialization: repeat pretrained RGB weights three times, divide by 3
- Image size: 266 x 266
- Normalization: ImageNet; whole images, no site masking
- Batch size: 8
- Seed: 42
- Epochs planned: 20
- Epochs completed: 10
- Best epoch: 5
- Best validation F1: 0.8235
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

- accuracy: 0.7049
- precision: 0.6176
- recall: 0.8077
- f1: 0.7000
- roc_auc: 0.7637
- false_alarm_rate: 0.3714
- loss: 0.5206
- samples: 61
- tn: 22
- fp: 13
- fn: 5
- tp: 21

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
- No image registration is performed; corresponding pixels must represent the same location.
- Pixel changes can include lighting, vegetation, clouds or other changes unrelated to looting.
- An early image is not guaranteed to show a site before looting.
- Seed/determinism settings do not guarantee identical results across platforms.
