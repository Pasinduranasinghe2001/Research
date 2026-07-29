# Step 7 Two-Frame Change Baseline Report

## Experiment Setup

- Model: ResNet18 two-frame change baseline
- Input: K = 2 satellite image frames
- Frame selection strategy: early_late
- Input representation: early frame + recent frame + absolute difference
- Input channels: 9
- Task: Binary classification
- Label 0: preserved
- Label 1: looted
- Image size: 266 x 266
- Batch size: 8
- Epochs planned: 20
- Learning rate: 0.0001
- Device: cpu

## Test Results

- Accuracy: 0.7011
- Precision: 0.5000
- Recall: 0.8846
- F1-score: 0.6389
- ROC-AUC: 0.8064
- False Alarm Rate: 0.3770
- TN: 38
- FP: 23
- FN: 3
- TP: 23

## Output Files Created

- `models\two_frame_change_resnet18_best.pth`
- `results\two_frame_change_baseline\two_frame_change_training_history.csv`
- `results\two_frame_change_baseline\two_frame_change_test_predictions.csv`
- `results\two_frame_change_baseline\two_frame_change_test_metrics.json`