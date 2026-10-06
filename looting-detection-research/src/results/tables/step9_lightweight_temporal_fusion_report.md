# Step 9 Lightweight Temporal Fusion Baseline Report

## Experiment Setup

- Model: ResNet18 frame-wise feature extractor + lightweight temporal fusion
- Fusion methods: GRU and attention pooling
- Task: Binary classification
- Label 0: preserved
- Label 1: looted
- Frame selection strategy: uniform
- K values: [3, 5, 8]
- Fusion types: ['gru', 'attention']
- Image size: 266 x 266
- Batch size: 8
- Epochs planned: 20
- Learning rate: 0.0001
- Freeze backbone: False
- Device: cpu

## Test Results Summary

- K=3, fusion=gru: Accuracy=0.6552, Precision=0.4333, Recall=0.5000, F1=0.4643, ROC-AUC=0.7623, FAR=0.2787
- K=3, fusion=attention: Accuracy=0.7701, Precision=0.5938, Recall=0.7308, F1=0.6552, ROC-AUC=0.8121, FAR=0.2131
- K=5, fusion=gru: Accuracy=0.7126, Precision=0.5128, Recall=0.7692, F1=0.6154, ROC-AUC=0.8531, FAR=0.3115
- K=5, fusion=attention: Accuracy=0.6552, Precision=0.4524, Recall=0.7308, F1=0.5588, ROC-AUC=0.7932, FAR=0.3770
- K=8, fusion=gru: Accuracy=0.7586, Precision=0.5676, Recall=0.8077, F1=0.6667, ROC-AUC=0.8108, FAR=0.2623
- K=8, fusion=attention: Accuracy=0.7586, Precision=0.5806, Recall=0.6923, F1=0.6316, ROC-AUC=0.8083, FAR=0.2131

## Best Lightweight Temporal Fusion Result Based on F1-score

- Experiment: lightweight_temporal_fusion_K8_uniform_gru
- K: 8
- Fusion: gru
- Accuracy: 0.7586
- Precision: 0.5676
- Recall: 0.8077
- F1-score: 0.6667
- ROC-AUC: 0.8108
- False Alarm Rate: 0.2623

## Output Files Created

- `results\lightweight_temporal_fusion\lightweight_temporal_fusion_summary_results.csv`
- `models\lightweight_temporal_fusion_K3_uniform_gru_best.pth`
- `results\lightweight_temporal_fusion\lightweight_temporal_fusion_K3_uniform_gru_training_history.csv`
- `results\lightweight_temporal_fusion\lightweight_temporal_fusion_K3_uniform_gru_test_predictions.csv`
- `results\lightweight_temporal_fusion\lightweight_temporal_fusion_K3_uniform_gru_test_metrics.json`
- `models\lightweight_temporal_fusion_K3_uniform_attention_best.pth`
- `results\lightweight_temporal_fusion\lightweight_temporal_fusion_K3_uniform_attention_training_history.csv`
- `results\lightweight_temporal_fusion\lightweight_temporal_fusion_K3_uniform_attention_test_predictions.csv`
- `results\lightweight_temporal_fusion\lightweight_temporal_fusion_K3_uniform_attention_test_metrics.json`
- `models\lightweight_temporal_fusion_K5_uniform_gru_best.pth`
- `results\lightweight_temporal_fusion\lightweight_temporal_fusion_K5_uniform_gru_training_history.csv`
- `results\lightweight_temporal_fusion\lightweight_temporal_fusion_K5_uniform_gru_test_predictions.csv`
- `results\lightweight_temporal_fusion\lightweight_temporal_fusion_K5_uniform_gru_test_metrics.json`
- `models\lightweight_temporal_fusion_K5_uniform_attention_best.pth`
- `results\lightweight_temporal_fusion\lightweight_temporal_fusion_K5_uniform_attention_training_history.csv`
- `results\lightweight_temporal_fusion\lightweight_temporal_fusion_K5_uniform_attention_test_predictions.csv`
- `results\lightweight_temporal_fusion\lightweight_temporal_fusion_K5_uniform_attention_test_metrics.json`
- `models\lightweight_temporal_fusion_K8_uniform_gru_best.pth`
- `results\lightweight_temporal_fusion\lightweight_temporal_fusion_K8_uniform_gru_training_history.csv`
- `results\lightweight_temporal_fusion\lightweight_temporal_fusion_K8_uniform_gru_test_predictions.csv`
- `results\lightweight_temporal_fusion\lightweight_temporal_fusion_K8_uniform_gru_test_metrics.json`
- `models\lightweight_temporal_fusion_K8_uniform_attention_best.pth`
- `results\lightweight_temporal_fusion\lightweight_temporal_fusion_K8_uniform_attention_training_history.csv`
- `results\lightweight_temporal_fusion\lightweight_temporal_fusion_K8_uniform_attention_test_predictions.csv`
- `results\lightweight_temporal_fusion\lightweight_temporal_fusion_K8_uniform_attention_test_metrics.json`