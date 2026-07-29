from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torchvision import models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from dataset_loader import create_dataloader


# ==================================================
# CONFIGURATION
# ==================================================

SEED = 42

K_VALUES = [3, 5, 8]

# Main recommended strategy for temporal pooling.
# uniform = frames selected across the whole time period.
STRATEGY = "uniform"

# Both are useful baselines.
POOLING_TYPES = ["mean", "max"]

IMAGE_SIZE = 266
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULT_DIR = Path("results/temporal_pooling_baseline")
TABLE_DIR = Path("results/tables")
MODEL_DIR = Path("models")

RESULT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# REPRODUCIBILITY
# ==================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==================================================
# MODEL
# ==================================================

class TemporalPoolingResNet18(nn.Module):
    """
    Short temporal pooling baseline.

    Input:
        x shape = [B, K, 3, H, W]

    Processing:
        1. Each frame is passed through ResNet18 feature extractor.
        2. Features are pooled across time using mean or max pooling.
        3. Final classifier predicts preserved / looted.
    """

    def __init__(self, pooling_type="mean", num_classes=2):
        super().__init__()

        if pooling_type not in ["mean", "max"]:
            raise ValueError("pooling_type must be 'mean' or 'max'")

        self.pooling_type = pooling_type

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        feature_dim = backbone.fc.in_features

        # Remove final classification layer.
        backbone.fc = nn.Identity()

        self.feature_extractor = backbone

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x):
        # x shape: [B, K, 3, H, W]
        batch_size, k, channels, height, width = x.shape

        # Combine batch and temporal dimensions:
        # [B, K, 3, H, W] -> [B*K, 3, H, W]
        x = x.view(batch_size * k, channels, height, width)

        features = self.feature_extractor(x)  # [B*K, feature_dim]

        # Restore temporal dimension:
        # [B*K, feature_dim] -> [B, K, feature_dim]
        features = features.view(batch_size, k, -1)

        if self.pooling_type == "mean":
            pooled = torch.mean(features, dim=1)

        elif self.pooling_type == "max":
            pooled, _ = torch.max(features, dim=1)

        logits = self.classifier(pooled)

        return logits


# ==================================================
# METRICS
# ==================================================

def calculate_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "false_alarm_rate": false_alarm_rate,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


# ==================================================
# TRAINING AND EVALUATION
# ==================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    all_true = []
    all_pred = []
    all_prob = []

    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["images"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        running_loss += loss.item() * images.size(0)

        all_true.extend(labels.detach().cpu().numpy())
        all_pred.extend(preds.detach().cpu().numpy())
        all_prob.extend(probs.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    metrics = calculate_metrics(all_true, all_pred, all_prob)
    metrics["loss"] = epoch_loss

    return metrics


def evaluate(model, loader, criterion, split_name="val"):
    model.eval()

    running_loss = 0.0
    all_true = []
    all_pred = []
    all_prob = []
    all_site_ids = []
    all_sample_ids = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {split_name}", leave=False):
            images = batch["images"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            running_loss += loss.item() * images.size(0)

            all_true.extend(labels.detach().cpu().numpy())
            all_pred.extend(preds.detach().cpu().numpy())
            all_prob.extend(probs.detach().cpu().numpy())
            all_site_ids.extend(batch["site_id"])
            all_sample_ids.extend(batch["sample_id"])

    epoch_loss = running_loss / len(loader.dataset)

    metrics = calculate_metrics(all_true, all_pred, all_prob)
    metrics["loss"] = epoch_loss

    predictions_df = pd.DataFrame({
        "site_id": all_site_ids,
        "sample_id": all_sample_ids,
        "true_label": all_true,
        "predicted_label": all_pred,
        "looted_probability": all_prob
    })

    return metrics, predictions_df


# ==================================================
# CLASS WEIGHTS
# ==================================================

def compute_class_weights(csv_path, strategy):
    df = pd.read_csv(csv_path)

    train_df = df[
        (df["split"].astype(str).str.lower() == "train") &
        (df["strategy"].astype(str).str.lower() == strategy.lower())
    ]

    class_counts = train_df["label_id"].value_counts().to_dict()

    preserved_count = class_counts.get(0, 1)
    looted_count = class_counts.get(1, 1)

    total = preserved_count + looted_count

    weight_preserved = total / (2 * preserved_count)
    weight_looted = total / (2 * looted_count)

    return torch.tensor([weight_preserved, weight_looted], dtype=torch.float32)


# ==================================================
# SINGLE EXPERIMENT
# ==================================================

def run_experiment(k_value, pooling_type):
    csv_path = f"data/processed/kframe_sequences/kframe_samples_K{k_value}.csv"

    print("\n" + "=" * 70)
    print(f"Running temporal pooling baseline")
    print(f"K = {k_value}")
    print(f"Strategy = {STRATEGY}")
    print(f"Pooling = {pooling_type}")
    print("=" * 70)

    experiment_name = f"temporal_pooling_K{k_value}_{STRATEGY}_{pooling_type}"

    train_loader = create_dataloader(
        csv_path=csv_path,
        split="train",
        strategy=STRATEGY,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        train=True,
        shuffle=True,
        num_workers=0,
        normalization="imagenet"
    )

    val_loader = create_dataloader(
        csv_path=csv_path,
        split="val",
        strategy=STRATEGY,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        train=False,
        shuffle=False,
        num_workers=0,
        normalization="imagenet"
    )

    test_loader = create_dataloader(
        csv_path=csv_path,
        split="test",
        strategy=STRATEGY,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        train=False,
        shuffle=False,
        num_workers=0,
        normalization="imagenet"
    )

    model = TemporalPoolingResNet18(
        pooling_type=pooling_type,
        num_classes=2
    ).to(DEVICE)

    class_weights = compute_class_weights(csv_path, STRATEGY).to(DEVICE)

    print(f"Class weights: {class_weights.detach().cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    history = []
    best_val_f1 = -1.0

    best_model_path = MODEL_DIR / f"{experiment_name}_best.pth"

    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer)
        val_metrics, _ = evaluate(model, val_loader, criterion, split_name="val")

        row = {
            "experiment_name": experiment_name,
            "k": k_value,
            "strategy": STRATEGY,
            "pooling_type": pooling_type,
            "epoch": epoch,

            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_roc_auc": train_metrics["roc_auc"],
            "train_far": train_metrics["false_alarm_rate"],

            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_far": val_metrics["false_alarm_rate"],
        }

        history.append(row)

        print(
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train F1: {train_metrics['f1']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"Best model saved: {best_model_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    # Save history
    history_df = pd.DataFrame(history)

    history_path = RESULT_DIR / f"{experiment_name}_training_history.csv"
    history_df.to_csv(history_path, index=False)

    # Test best model
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    test_metrics, test_predictions = evaluate(
        model,
        test_loader,
        criterion,
        split_name="test"
    )

    test_predictions_path = RESULT_DIR / f"{experiment_name}_test_predictions.csv"
    test_predictions.to_csv(test_predictions_path, index=False)

    test_metrics["experiment_name"] = experiment_name
    test_metrics["k"] = k_value
    test_metrics["strategy"] = STRATEGY
    test_metrics["pooling_type"] = pooling_type

    test_metrics_path = RESULT_DIR / f"{experiment_name}_test_metrics.json"

    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4)

    print(f"\nCompleted experiment: {experiment_name}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")

    return {
        "experiment_name": experiment_name,
        "k": k_value,
        "strategy": STRATEGY,
        "pooling_type": pooling_type,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "roc_auc": test_metrics["roc_auc"],
        "false_alarm_rate": test_metrics["false_alarm_rate"],
        "tn": test_metrics["tn"],
        "fp": test_metrics["fp"],
        "fn": test_metrics["fn"],
        "tp": test_metrics["tp"],
        "best_model_path": str(best_model_path),
        "history_path": str(history_path),
        "test_predictions_path": str(test_predictions_path),
        "test_metrics_path": str(test_metrics_path)
    }


# ==================================================
# MAIN
# ==================================================

def main():
    set_seed(SEED)

    print(f"Using device: {DEVICE}")

    all_results = []

    for k_value in K_VALUES:
        for pooling_type in POOLING_TYPES:
            result = run_experiment(
                k_value=k_value,
                pooling_type=pooling_type
            )
            all_results.append(result)

    results_df = pd.DataFrame(all_results)

    summary_csv_path = RESULT_DIR / "temporal_pooling_summary_results.csv"
    results_df.to_csv(summary_csv_path, index=False)

    # Create markdown report
    report = []

    report.append("# Step 8 Short Temporal Pooling Baseline Report\n")

    report.append("## Experiment Setup\n")
    report.append("- Model: ResNet18 frame-wise feature extractor + temporal pooling")
    report.append("- Task: Binary classification")
    report.append("- Label 0: preserved")
    report.append("- Label 1: looted")
    report.append(f"- Frame selection strategy: {STRATEGY}")
    report.append(f"- K values: {K_VALUES}")
    report.append(f"- Pooling types: {POOLING_TYPES}")
    report.append(f"- Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    report.append(f"- Batch size: {BATCH_SIZE}")
    report.append(f"- Epochs planned: {NUM_EPOCHS}")
    report.append(f"- Learning rate: {LEARNING_RATE}")
    report.append(f"- Device: {DEVICE}\n")

    report.append("## Test Results Summary\n")

    for _, row in results_df.iterrows():
        report.append(
            f"- K={int(row['k'])}, pooling={row['pooling_type']}: "
            f"Accuracy={row['accuracy']:.4f}, "
            f"Precision={row['precision']:.4f}, "
            f"Recall={row['recall']:.4f}, "
            f"F1={row['f1']:.4f}, "
            f"ROC-AUC={row['roc_auc']:.4f}, "
            f"FAR={row['false_alarm_rate']:.4f}"
        )

    best_row = results_df.sort_values("f1", ascending=False).iloc[0]

    report.append("\n## Best Temporal Pooling Result Based on F1-score\n")
    report.append(f"- Experiment: {best_row['experiment_name']}")
    report.append(f"- K: {int(best_row['k'])}")
    report.append(f"- Pooling: {best_row['pooling_type']}")
    report.append(f"- Accuracy: {best_row['accuracy']:.4f}")
    report.append(f"- Precision: {best_row['precision']:.4f}")
    report.append(f"- Recall: {best_row['recall']:.4f}")
    report.append(f"- F1-score: {best_row['f1']:.4f}")
    report.append(f"- ROC-AUC: {best_row['roc_auc']:.4f}")
    report.append(f"- False Alarm Rate: {best_row['false_alarm_rate']:.4f}\n")

    report.append("## Output Files Created\n")
    report.append(f"- `{summary_csv_path}`")

    for _, row in results_df.iterrows():
        report.append(f"- `{row['best_model_path']}`")
        report.append(f"- `{row['history_path']}`")
        report.append(f"- `{row['test_predictions_path']}`")
        report.append(f"- `{row['test_metrics_path']}`")

    report_path = TABLE_DIR / "step8_temporal_pooling_baseline_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\nStep 8 completed.")
    print(f"Summary results: {summary_csv_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()