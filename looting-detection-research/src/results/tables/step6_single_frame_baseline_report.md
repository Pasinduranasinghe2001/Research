# Step 6 Single-Frame CNN Baseline Report

## Experiment Setup

- Model: ResNet18
- Input: K = 1 satellite image frame
- Frame selection strategy: recent_window
- Task: Binary classification
- Label 0: preserved
- Label 1: looted
- Image size: 266 x 266
- Batch size: 16
- Epochs planned: 20
- Learning rate: 0.0001
- Device: cpu

## Test Results

- Accuracy: 0.5632
- Precision: 0.4062
- Recall: 1.0000
- F1-score: 0.5778
- ROC-AUC: 0.7926
- False Alarm Rate: 0.6230
- TN: 23
- FP: 38
- FN: 0
- TP: 26

## Output Files Created

- `models\single_frame_resnet18_best.pth`
- `results\single_frame_baseline\single_frame_training_history.csv`
- `results\single_frame_baseline\single_frame_test_predictions.csv`
- `results\single_frame_baseline\single_frame_test_metrics.json`