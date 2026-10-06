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

- K=3, pooling=mean: Accuracy=0.6437, Precision=0.4510, Recall=0.8846, F1=0.5974, ROC-AUC=0.8001, FAR=0.4590
- K=3, pooling=max: Accuracy=0.7471, Precision=0.5588, Recall=0.7308, F1=0.6333, ROC-AUC=0.8033, FAR=0.2459
- K=5, pooling=mean: Accuracy=0.7586, Precision=0.5758, Recall=0.7308, F1=0.6441, ROC-AUC=0.8518, FAR=0.2295
- K=5, pooling=max: Accuracy=0.6322, Precision=0.4118, Recall=0.5385, F1=0.4667, ROC-AUC=0.6021, FAR=0.3279
- K=8, pooling=mean: Accuracy=0.7701, Precision=0.6000, Recall=0.6923, F1=0.6429, ROC-AUC=0.7989, FAR=0.1967
- K=8, pooling=max: Accuracy=0.7701, Precision=0.5789, Recall=0.8462, F1=0.6875, ROC-AUC=0.8651, FAR=0.2623

## Best Temporal Pooling Result Based on F1-score

- Experiment: temporal_pooling_K8_uniform_max
- K: 8
- Pooling: max
- Accuracy: 0.7701
- Precision: 0.5789
- Recall: 0.8462
- F1-score: 0.6875
- ROC-AUC: 0.8651
- False Alarm Rate: 0.2623

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