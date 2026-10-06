# Step 10 Proposed Change-Aware Lightweight Temporal Fusion Report

## Experiment Setup

- Feature extractor: pretrained ResNet18
- Change representation: absolute feature difference from the first selected frame
- Temporal fusion: one-layer GRU
- Frame selection strategy: uniform
- Proposed K values: [3, 5, 8]
- Training frame-drop probability: 0.25
- Image size: 266 x 266
- Batch size: 8
- Maximum epochs: 20
- Learning rate: 0.0001
- Freeze backbone: False
- Device: cpu

## Step 10 Test Results

- step10_K5_gru_no_change_no_drop: Accuracy=0.5977, Precision=0.4182, Recall=0.8846, F1=0.5679, ROC-AUC=0.7793, FAR=0.5246
- step10_K5_change_gru_no_drop: Accuracy=0.7931, Precision=0.6818, Recall=0.5769, F1=0.6250, ROC-AUC=0.8770, FAR=0.1148
- step10_K3_change_gru_framedrop: Accuracy=0.7241, Precision=0.5278, Recall=0.7308, F1=0.6129, ROC-AUC=0.8329, FAR=0.2787
- step10_K5_change_gru_framedrop: Accuracy=0.8046, Precision=0.6667, Recall=0.6923, F1=0.6792, ROC-AUC=0.8291, FAR=0.1475
- step10_K8_change_gru_framedrop: Accuracy=0.7471, Precision=0.6000, Recall=0.4615, F1=0.5217, ROC-AUC=0.8411, FAR=0.1311

## Best Proposed Model Based on F1-score

- Experiment: step10_K5_change_gru_framedrop
- K: 5
- Accuracy: 0.8046
- Precision: 0.6667
- Recall: 0.6923
- F1-score: 0.6792
- ROC-AUC: 0.8291
- False Alarm Rate: 0.1475

## Ablation Study

- ablation_no_change_no_drop: F1=0.5679, ROC-AUC=0.7793, FAR=0.5246
- ablation_change_no_drop: F1=0.6250, ROC-AUC=0.8770, FAR=0.1148
- proposed_change_gru_framedrop: F1=0.6792, ROC-AUC=0.8291, FAR=0.1475

## Missing-Frame Robustness Test

- Test frame-drop=0.00: Accuracy=0.8046, Recall=0.6923, F1=0.6792, FAR=0.1475
- Test frame-drop=0.25: Accuracy=0.7931, Recall=0.6923, F1=0.6667, FAR=0.1639
- Test frame-drop=0.50: Accuracy=0.7586, Recall=0.6154, F1=0.6038, FAR=0.1803

## Final Baseline Comparison

- Step 6 Single-frame ResNet18: K=1, Accuracy=0.7701, Precision=0.5714, Recall=0.9231, F1=0.7059, ROC-AUC=0.8947, FAR=0.2951
- Step 10 Proposed change-aware GRU: K=5, Accuracy=0.8046, Precision=0.6667, Recall=0.6923, F1=0.6792, ROC-AUC=0.8291, FAR=0.1475
- Step 9 Best lightweight fusion: K=5, Accuracy=0.7701, Precision=0.5882, Recall=0.7692, F1=0.6667, ROC-AUC=0.8556, FAR=0.2295
- Step 8 Best temporal pooling: K=8, Accuracy=0.7241, Precision=0.5227, Recall=0.8846, F1=0.6571, ROC-AUC=0.8064, FAR=0.3443
- Step 7 Two-frame change ResNet18: K=2, Accuracy=0.7011, Precision=0.5000, Recall=0.8846, F1=0.6389, ROC-AUC=0.8064, FAR=0.3770

## Output Files

- `results\proposed_change_aware_fusion\step10_summary_results.csv`
- `results\proposed_change_aware_fusion\step10_ablation_results.csv`
- `results\proposed_change_aware_fusion\step10_k_analysis_results.csv`
- `results\proposed_change_aware_fusion\step10_frame_drop_robustness.csv`
- `results\final_comparison\final_model_comparison.csv`