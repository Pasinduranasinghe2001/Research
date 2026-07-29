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

- K=3, fusion=gru: Accuracy=0.6782, Precision=0.4706, Recall=0.6154, F1=0.5333, ROC-AUC=0.7535, FAR=0.2951
- K=3, fusion=attention: Accuracy=0.7586, Precision=0.5758, Recall=0.7308, F1=0.6441, ROC-AUC=0.8159, FAR=0.2295
- K=5, fusion=gru: Accuracy=0.7701, Precision=0.5882, Recall=0.7692, F1=0.6667, ROC-AUC=0.8556, FAR=0.2295
- K=5, fusion=attention: Accuracy=0.7241, Precision=0.5333, Recall=0.6154, F1=0.5714, ROC-AUC=0.7995, FAR=0.2295
- K=8, fusion=gru: Accuracy=0.7126, Precision=0.5122, Recall=0.8077, F1=0.6269, ROC-AUC=0.8020, FAR=0.3279
- K=8, fusion=attention: Accuracy=0.7011, Precision=0.5000, Recall=0.4615, F1=0.4800, ROC-AUC=0.7837, FAR=0.1967

## Best Lightweight Temporal Fusion Result Based on F1-score

- Experiment: lightweight_temporal_fusion_K5_uniform_gru
- K: 5
- Fusion: gru
- Accuracy: 0.7701
- Precision: 0.5882
- Recall: 0.7692
- F1-score: 0.6667
- ROC-AUC: 0.8556
- False Alarm Rate: 0.2295

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