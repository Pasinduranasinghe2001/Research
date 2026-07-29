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

CSV_PATH = "data/processed/kframe_sequences/kframe_samples_K1.csv"
RESULT_DIR = Path("results/single_frame_baseline")
TABLE_DIR = Path("results/tables")
MODEL_DIR = Path("models")

RESULT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY = "recent_window"
IMAGE_SIZE = 266
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

class SingleFrameResNet18(nn.Module):
    """
    Single-frame CNN baseline.
    Input shape from loader: [B, 1, 3, H, W]
    Model uses only one frame and predicts looted/preserved.
    """

    def __init__(self, num_classes=2):
        super().__init__()

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x shape: [B, 1, 3, H, W]
        x = x[:, 0, :, :, :]  # convert to [B, 3, H, W]
        logits = self.backbone(x)
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
# TRAIN / EVALUATE FUNCTIONS
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
# CLASS WEIGHT
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
# MAIN
# ==================================================

def main():
    set_seed(SEED)

    print(f"Using device: {DEVICE}")
    print(f"Training single-frame baseline with strategy: {STRATEGY}")

    train_loader = create_dataloader(
        csv_path=CSV_PATH,
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
        csv_path=CSV_PATH,
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
        csv_path=CSV_PATH,
        split="test",
        strategy=STRATEGY,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        train=False,
        shuffle=False,
        num_workers=0,
        normalization="imagenet"
    )

    model = SingleFrameResNet18(num_classes=2).to(DEVICE)

    class_weights = compute_class_weights(CSV_PATH, STRATEGY).to(DEVICE)

    print(f"Class weights: {class_weights.detach().cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    history = []

    best_val_f1 = -1.0
    best_model_path = MODEL_DIR / "single_frame_resnet18_best.pth"
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer)
        val_metrics, _ = evaluate(model, val_loader, criterion, split_name="val")

        row = {
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

    # Save training history
    history_df = pd.DataFrame(history)
    history_path = RESULT_DIR / "single_frame_training_history.csv"
    history_df.to_csv(history_path, index=False)

    # Load best model and test
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    test_metrics, test_predictions = evaluate(
        model,
        test_loader,
        criterion,
        split_name="test"
    )

    test_predictions_path = RESULT_DIR / "single_frame_test_predictions.csv"
    test_predictions.to_csv(test_predictions_path, index=False)

    test_metrics_path = RESULT_DIR / "single_frame_test_metrics.json"
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4)

    # Create markdown report
    report = []

    report.append("# Step 6 Single-Frame CNN Baseline Report\n")

    report.append("## Experiment Setup\n")
    report.append(f"- Model: ResNet18")
    report.append(f"- Input: K = 1 satellite image frame")
    report.append(f"- Frame selection strategy: {STRATEGY}")
    report.append(f"- Task: Binary classification")
    report.append(f"- Label 0: preserved")
    report.append(f"- Label 1: looted")
    report.append(f"- Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    report.append(f"- Batch size: {BATCH_SIZE}")
    report.append(f"- Epochs planned: {NUM_EPOCHS}")
    report.append(f"- Learning rate: {LEARNING_RATE}")
    report.append(f"- Device: {DEVICE}\n")

    report.append("## Test Results\n")
    report.append(f"- Accuracy: {test_metrics['accuracy']:.4f}")
    report.append(f"- Precision: {test_metrics['precision']:.4f}")
    report.append(f"- Recall: {test_metrics['recall']:.4f}")
    report.append(f"- F1-score: {test_metrics['f1']:.4f}")
    report.append(f"- ROC-AUC: {test_metrics['roc_auc']:.4f}")
    report.append(f"- False Alarm Rate: {test_metrics['false_alarm_rate']:.4f}")
    report.append(f"- TN: {test_metrics['tn']}")
    report.append(f"- FP: {test_metrics['fp']}")
    report.append(f"- FN: {test_metrics['fn']}")
    report.append(f"- TP: {test_metrics['tp']}\n")

    report.append("## Output Files Created\n")
    report.append(f"- `{best_model_path}`")
    report.append(f"- `{history_path}`")
    report.append(f"- `{test_predictions_path}`")
    report.append(f"- `{test_metrics_path}`")

    report_path = TABLE_DIR / "step6_single_frame_baseline_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\nStep 6 completed.")
    print(f"Best model: {best_model_path}")
    print(f"Training history: {history_path}")
    print(f"Test predictions: {test_predictions_path}")
    print(f"Test metrics: {test_metrics_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()