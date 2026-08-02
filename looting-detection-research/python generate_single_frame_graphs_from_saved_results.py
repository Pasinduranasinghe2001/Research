"""
Generate publication-ready figures from an already completed single-frame run.

Expected files:
  results/single_frame_baseline/single_frame_training_history.csv
  results/single_frame_baseline/single_frame_test_predictions.csv
  results/single_frame_baseline/single_frame_test_metrics.json

Run from the project root:
  python generate_single_frame_graphs_from_saved_results.py
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


SEED = 42
DECISION_THRESHOLD = 0.5
BOOTSTRAP_ITERATIONS = 1000

RESULT_DIR = Path("results/single_frame_baseline")
FIGURE_DIR = RESULT_DIR / "figures"

HISTORY_PATH = RESULT_DIR / "single_frame_training_history.csv"
PREDICTIONS_PATH = RESULT_DIR / "single_frame_test_predictions.csv"
METRICS_PATH = RESULT_DIR / "single_frame_test_metrics.json"
CI_PATH = RESULT_DIR / "single_frame_test_bootstrap_95ci.csv"


METRIC_DISPLAY = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
    "average_precision": "Average precision",
    "false_alarm_rate": "False alarm rate",
}


def configure_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig, stem):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    png = FIGURE_DIR / f"{stem}.png"
    pdf = FIGURE_DIR / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")


def calculate_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_alarm_rate": fp / (fp + tn) if (fp + tn) else 0.0,
    }

    if np.unique(y_true).size >= 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["average_precision"] = average_precision_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = np.nan
        metrics["average_precision"] = np.nan

    return metrics


def grouped_bootstrap_ci(predictions, iterations=1000):
    """Resample full sites where possible; otherwise resample individual rows."""
    rng = np.random.default_rng(SEED)
    df = predictions.reset_index(drop=True).copy()

    grouped = (
        "site_id" in df.columns
        and df["site_id"].notna().all()
        and df["site_id"].nunique() > 1
    )

    metric_names = list(METRIC_DISPLAY.keys())
    values = {name: [] for name in metric_names}

    if grouped:
        group_labels = df["site_id"].astype(str).to_numpy()
        unique_groups = np.unique(group_labels)
        group_rows = {
            group: np.flatnonzero(group_labels == group)
            for group in unique_groups
        }

    for _ in range(iterations):
        if grouped:
            sampled_groups = rng.choice(
                unique_groups, size=len(unique_groups), replace=True
            )
            sampled_indices = np.concatenate(
                [group_rows[group] for group in sampled_groups]
            )
        else:
            sampled_indices = rng.integers(0, len(df), size=len(df))

        sample = df.iloc[sampled_indices]
        sample_metrics = calculate_metrics(
            sample["true_label"],
            sample["looted_probability"],
            DECISION_THRESHOLD,
        )
        for name in metric_names:
            values[name].append(sample_metrics[name])

    full_metrics = calculate_metrics(
        df["true_label"], df["looted_probability"], DECISION_THRESHOLD
    )

    rows = []
    for name in metric_names:
        metric_values = np.asarray(values[name], dtype=float)
        metric_values = metric_values[np.isfinite(metric_values)]
        lower, upper = (
            np.percentile(metric_values, [2.5, 97.5])
            if metric_values.size
            else (np.nan, np.nan)
        )
        rows.append({
            "metric": name,
            "estimate": full_metrics[name],
            "ci_lower_95": lower,
            "ci_upper_95": upper,
            "bootstrap_valid_runs": len(metric_values),
            "bootstrap_unit": "site" if grouped else "sample",
        })

    return pd.DataFrame(rows)


def plot_loss(history):
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(history["epoch"], history["train_loss"], marker="o", label="Train")
    ax.plot(history["epoch"], history["val_loss"], marker="o", label="Validation")
    ax.set_title("Single-frame ResNet18 learning curve: loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted cross-entropy loss")
    ax.legend()
    save_figure(fig, "01_training_validation_loss")


def plot_f1(history):
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(history["epoch"], history["train_f1"], marker="o", label="Train")
    ax.plot(history["epoch"], history["val_f1"], marker="o", label="Validation")
    best_idx = history["val_f1"].idxmax()
    best_epoch = int(history.loc[best_idx, "epoch"])
    best_value = float(history.loc[best_idx, "val_f1"])
    ax.scatter(best_epoch, best_value, s=70, zorder=5, label=f"Best validation F1 = {best_value:.3f}")
    ax.axvline(best_epoch, linestyle="--", alpha=0.6)
    ax.set_title("Single-frame ResNet18 learning curve: F1-score")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    save_figure(fig, "02_training_validation_f1")


def plot_auc(history):
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(history["epoch"], history["train_roc_auc"], marker="o", label="Train")
    ax.plot(history["epoch"], history["val_roc_auc"], marker="o", label="Validation")
    ax.set_title("Single-frame ResNet18 learning curve: ROC-AUC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    save_figure(fig, "03_training_validation_roc_auc")


def draw_confusion(ax, matrix, title, fmt):
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks([0, 1], labels=["Preserved", "Looted"])
    ax.set_yticks([0, 1], labels=["Preserved", "Looted"])
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(2):
        for col in range(2):
            value = matrix[row, col]
            ax.text(
                col, row, format(value, fmt),
                ha="center", va="center", fontsize=13,
                color="white" if value > threshold else "black",
            )
    return image


def plot_confusion(predictions):
    y_true = predictions["true_label"].to_numpy(dtype=int)
    y_pred = predictions["predicted_label"].to_numpy(dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    image = draw_confusion(ax, cm, "Test confusion matrix (counts)", "d")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, "04_test_confusion_matrix_counts")

    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = cm / cm.sum(axis=1, keepdims=True)
        normalized = np.nan_to_num(normalized)

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    image = draw_confusion(
        ax, normalized, "Test confusion matrix (row-normalized)", ".2f"
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, "05_test_confusion_matrix_normalized")


def plot_roc(predictions):
    y_true = predictions["true_label"].to_numpy(dtype=int)
    y_prob = predictions["looted_probability"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    if np.unique(y_true).size >= 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_value = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, linewidth=2, label=f"ResNet18 (AUC = {auc_value:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
        ax.legend(loc="lower right")
    else:
        ax.text(0.5, 0.5, "ROC unavailable: test set has one class", ha="center")
    ax.set_title("Test receiver operating characteristic")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    save_figure(fig, "06_test_roc_curve")


def plot_pr(predictions):
    y_true = predictions["true_label"].to_numpy(dtype=int)
    y_prob = predictions["looted_probability"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    if np.unique(y_true).size >= 2:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap_value = average_precision_score(y_true, y_prob)
        prevalence = y_true.mean()
        ax.plot(recall, precision, linewidth=2, label=f"ResNet18 (AP = {ap_value:.3f})")
        ax.axhline(prevalence, linestyle="--", label=f"No-skill prevalence = {prevalence:.3f}")
        ax.legend(loc="lower left")
    else:
        ax.text(0.5, 0.5, "PR unavailable: test set has one class", ha="center")
    ax.set_title("Test precision-recall curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    save_figure(fig, "07_test_precision_recall_curve")


def plot_probability_distribution(predictions):
    preserved = predictions.loc[
        predictions["true_label"] == 0, "looted_probability"
    ]
    looted = predictions.loc[
        predictions["true_label"] == 1, "looted_probability"
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    bins = np.linspace(0.0, 1.0, 21)
    ax.hist(preserved, bins=bins, alpha=0.65, label="True preserved")
    ax.hist(looted, bins=bins, alpha=0.65, label="True looted")
    ax.axvline(0.5, linestyle="--", linewidth=2, label="Decision threshold = 0.5")
    ax.set_title("Test probability separation by true class")
    ax.set_xlabel("Predicted probability of looting")
    ax.set_ylabel("Number of samples")
    ax.legend()
    save_figure(fig, "08_test_probability_distribution")


def plot_metric_ci(ci_table):
    metrics = [
        "accuracy", "balanced_accuracy", "precision", "recall",
        "specificity", "f1", "roc_auc", "average_precision",
    ]
    selected = ci_table.set_index("metric").loc[metrics].reset_index()
    labels = [METRIC_DISPLAY[name] for name in metrics]
    estimates = selected["estimate"].to_numpy(dtype=float)
    lower = selected["ci_lower_95"].to_numpy(dtype=float)
    upper = selected["ci_upper_95"].to_numpy(dtype=float)
    errors = np.vstack([estimates - lower, upper - estimates])

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x = np.arange(len(labels))
    bars = ax.bar(x, estimates, yerr=errors, capsize=4)
    ax.set_xticks(x, labels=labels, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Score")
    unit = selected["bootstrap_unit"].iloc[0]
    ax.set_title(f"Test metrics with 95% bootstrap confidence intervals ({unit}-level)")

    for bar, value in zip(bars, estimates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.025, 1.04),
            f"{value:.3f}",
            ha="center", va="bottom", fontsize=8,
        )
    save_figure(fig, "09_test_metrics_with_95ci")


def plot_dashboard(history, predictions, metrics):
    y_true = predictions["true_label"].to_numpy(dtype=int)
    y_pred = predictions["predicted_label"].to_numpy(dtype=int)
    y_prob = predictions["looted_probability"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    axes[0, 0].plot(history["epoch"], history["train_loss"], label="Train")
    axes[0, 0].plot(history["epoch"], history["val_loss"], label="Validation")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()

    axes[0, 1].plot(history["epoch"], history["train_f1"], label="Train")
    axes[0, 1].plot(history["epoch"], history["val_f1"], label="Validation")
    axes[0, 1].set_title("F1-score")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].legend()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    draw_confusion(axes[0, 2], cm, "Confusion matrix", "d")

    if np.unique(y_true).size >= 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_value = roc_auc_score(y_true, y_prob)
        axes[1, 0].plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
        axes[1, 0].plot([0, 1], [0, 1], linestyle="--")
        axes[1, 0].legend()
    axes[1, 0].set_title("ROC curve")
    axes[1, 0].set_xlabel("False positive rate")
    axes[1, 0].set_ylabel("True positive rate")

    if np.unique(y_true).size >= 2:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap_value = average_precision_score(y_true, y_prob)
        axes[1, 1].plot(recall, precision, label=f"AP = {ap_value:.3f}")
        axes[1, 1].axhline(y_true.mean(), linestyle="--")
        axes[1, 1].legend()
    axes[1, 1].set_title("Precision-recall curve")
    axes[1, 1].set_xlabel("Recall")
    axes[1, 1].set_ylabel("Precision")

    metric_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    metric_values = [
        metrics.get("accuracy", np.nan),
        metrics.get("precision", np.nan),
        metrics.get("recall", np.nan),
        metrics.get("f1", np.nan),
        metrics.get("roc_auc", np.nan),
    ]
    axes[1, 2].bar(metric_names, metric_values)
    axes[1, 2].set_title("Test metrics")
    axes[1, 2].set_ylim(0.0, 1.05)
    axes[1, 2].tick_params(axis="x", rotation=30)

    fig.suptitle("Single-frame ResNet18 final-results dashboard", fontsize=16)
    save_figure(fig, "10_single_frame_results_dashboard")


def validate_inputs(history, predictions):
    required_history = {
        "epoch", "train_loss", "val_loss", "train_f1", "val_f1",
        "train_roc_auc", "val_roc_auc",
    }
    required_predictions = {
        "true_label", "predicted_label", "looted_probability",
    }

    missing_history = required_history - set(history.columns)
    missing_predictions = required_predictions - set(predictions.columns)

    if missing_history:
        raise ValueError(f"Training history is missing columns: {sorted(missing_history)}")
    if missing_predictions:
        raise ValueError(f"Test predictions are missing columns: {sorted(missing_predictions)}")


def main():
    for path in [HISTORY_PATH, PREDICTIONS_PATH, METRICS_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required result file not found: {path}\n"
                "Run the single-frame training script first, or update the paths at the top of this file."
            )

    configure_style()
    history = pd.read_csv(HISTORY_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    validate_inputs(history, predictions)

    # Recalculate missing modern metrics from saved probabilities.
    recalculated = calculate_metrics(
        predictions["true_label"],
        predictions["looted_probability"],
        DECISION_THRESHOLD,
    )
    for key, value in recalculated.items():
        metrics.setdefault(key, float(value) if np.isfinite(value) else None)

    ci_table = grouped_bootstrap_ci(predictions, BOOTSTRAP_ITERATIONS)
    ci_table.to_csv(CI_PATH, index=False)
    print(f"Saved: {CI_PATH}")

    plot_loss(history)
    plot_f1(history)
    plot_auc(history)
    plot_confusion(predictions)
    plot_roc(predictions)
    plot_pr(predictions)
    plot_probability_distribution(predictions)
    plot_metric_ci(ci_table)
    plot_dashboard(history, predictions, metrics)

    print(f"\nAll figures created in: {FIGURE_DIR}")
    print("Both PNG (300 dpi) and vector PDF versions were saved.")


if __name__ == "__main__":
    main()