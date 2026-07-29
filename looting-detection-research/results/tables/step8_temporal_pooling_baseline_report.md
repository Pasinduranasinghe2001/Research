# Step 8 Short Temporal Pooling Baseline Report

## Experiment Setup

- Model: ResNet18 frame-wise feature extractor + temporal pooling
- Task: Binary classification
- Label 0: preserved
- Label 1: looted
- Frame selection strategy: uniform
- K values: [3, 5, 8]
- Pooling types: ['mean', 'max']
- Image size: 266 x 266
- Batch size: 8
- Epochs planned: 20
- Learning rate: 0.0001
- Device: cpu

## Test Results Summary

- K=3, pooling=mean: Accuracy=0.6092, Precision=0.4259, Recall=0.8846, F1=0.5750, ROC-AUC=0.7976, FAR=0.5082
- K=3, pooling=max: Accuracy=0.7356, Precision=0.6154, Recall=0.3077, F1=0.4103, ROC-AUC=0.7390, FAR=0.0820
- K=5, pooling=mean: Accuracy=0.7586, Precision=0.6316, Recall=0.4615, F1=0.5333, ROC-AUC=0.8701, FAR=0.1148
- K=5, pooling=max: Accuracy=0.7471, Precision=0.5625, Recall=0.6923, F1=0.6207, ROC-AUC=0.8613, FAR=0.2295
- K=8, pooling=mean: Accuracy=0.7241, Precision=0.5227, Recall=0.8846, F1=0.6571, ROC-AUC=0.8064, FAR=0.3443
- K=8, pooling=max: Accuracy=0.6437, Precision=0.4359, Recall=0.6538, F1=0.5231, ROC-AUC=0.7446, FAR=0.3607

## Best Temporal Pooling Result Based on F1-score

- Experiment: temporal_pooling_K8_uniform_mean
- K: 8
- Pooling: mean
- Accuracy: 0.7241
- Precision: 0.5227
- Recall: 0.8846
- F1-score: 0.6571
- ROC-AUC: 0.8064
- False Alarm Rate: 0.3443

## Output Files Created

- `results\temporal_pooling_baseline\temporal_pooling_summary_results.csv`
- `models\temporal_pooling_K3_uniform_mean_best.pth`
- `results\temporal_pooling_baseline\temporal_pooling_K3_uniform_mean_training_history.csv`
- `results\temporal_pooling_baseline\temporal_pooling_K3_uniform_mean_test_predictions.csv`
- `results\temporal_pooling_baseline\temporal_pooling_K3_uniform_mean_test_metrics.json`
- `models\temporal_pooling_K3_uniform_max_best.pth`
- `results\temporal_pooling_baseline\temporal_pooling_K3_uniform_max_training_history.csv`
- `results\temporal_pooling_baseline\temporal_pooling_K3_uniform_max_test_predictions.csv`
- `results\temporal_pooling_baseline\temporal_pooling_K3_uniform_max_test_metrics.json`
- `models\temporal_pooling_K5_uniform_mean_best.pth`
- `results\temporal_pooling_baseline\temporal_pooling_K5_uniform_mean_training_history.csv`
- `results\temporal_pooling_baseline\temporal_pooling_K5_uniform_mean_test_predictions.csv`
- `results\temporal_pooling_baseline\temporal_pooling_K5_uniform_mean_test_metrics.json`
- `models\temporal_pooling_K5_uniform_max_best.pth`
- `results\temporal_pooling_baseline\temporal_pooling_K5_uniform_max_training_history.csv`
- `results\temporal_pooling_baseline\temporal_pooling_K5_uniform_max_test_predictions.csv`
- `results\temporal_pooling_baseline\temporal_pooling_K5_uniform_max_test_metrics.json`
- `models\temporal_pooling_K8_uniform_mean_best.pth`
- `results\temporal_pooling_baseline\temporal_pooling_K8_uniform_mean_training_history.csv`
- `results\temporal_pooling_baseline\temporal_pooling_K8_uniform_mean_test_predictions.csv`
- `results\temporal_pooling_baseline\temporal_pooling_K8_uniform_mean_test_metrics.json`
- `models\temporal_pooling_K8_uniform_max_best.pth`
- `results\temporal_pooling_baseline\temporal_pooling_K8_uniform_max_training_history.csv`
- `results\temporal_pooling_baseline\temporal_pooling_K8_uniform_max_test_predictions.csv`
- `results\temporal_pooling_baseline\temporal_pooling_K8_uniform_max_test_metrics.json`