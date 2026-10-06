from pathlib import Path
import json
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import AdamW
from torchvision import models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from dataset_loader import create_dataloader


# ==================================================
# CONFIGURATION
# ==================================================

SEED = 42

# Keep the same frame-selection strategy used in Steps 8 and 9
# so the Step 10 comparison remains fair.
STRATEGY = "uniform"

# Best Step 9 setting is used for the ablation study.
ABLATION_K = 5

# Final proposed model is checked with different short sequence lengths.
PROPOSED_K_VALUES = [3, 5, 8]

IMAGE_SIZE = 266
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5
GRAD_CLIP_NORM = 5.0

HIDDEN_DIM = 256
FRAME_DROP_PROB = 0.25

# Keep False for a fair comparison with Step 9.
FREEZE_BACKBONE = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

KFRAME_DIR = Path("data/processed/kframe_sequences")
RESULT_DIR = Path("results/proposed_change_aware_fusion")
TABLE_DIR = Path("results/tables")
MODEL_DIR = Path("models")
FINAL_COMPARISON_DIR = Path("results/final_comparison")

RESULT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FINAL_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# REPRODUCIBILITY
# ==================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==================================================
# MODEL
# ==================================================

class ChangeAwareTemporalFusionResNet18(nn.Module):
    """
    Proposed change-aware lightweight temporal-fusion model.

    Input:
        x: [B, K, 3, H, W]

    Processing:
        1. Extract a 512-dimensional ResNet18 feature for every frame.
        2. Use the first selected frame as the temporal reference.
        3. Calculate absolute feature differences:
               |feature_t - reference_feature|
        4. Concatenate appearance and change features.
        5. Project them into a compact feature space.
        6. Apply training-time frame-drop augmentation.
        7. Fuse the remaining ordered features using a GRU.
        8. Predict preserved (0) or looted (1).
    """

    def __init__(
        self,
        use_change_features: bool = True,
        hidden_dim: int = 256,
        num_classes: int = 2,
        train_frame_drop_prob: float = 0.25,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        if not 0.0 <= train_frame_drop_prob < 1.0:
            raise ValueError("train_frame_drop_prob must be in [0, 1).")

        self.use_change_features = use_change_features
        self.train_frame_drop_prob = train_frame_drop_prob
        self.freeze_backbone = freeze_backbone

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
        self.feature_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.feature_extractor = backbone

        if self.freeze_backbone:
            for parameter in self.feature_extractor.parameters():
                parameter.requires_grad = False

        fusion_input_dim = (
            self.feature_dim * 2
            if self.use_change_features
            else self.feature_dim
        )

        self.feature_projection = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

        self.temporal_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def train(self, mode: bool = True):
        """
        Keep the frozen backbone in evaluation mode so its BatchNorm
        statistics are not changed.
        """
        super().train(mode)
        if self.freeze_backbone:
            self.feature_extractor.eval()
        return self

    @staticmethod
    def _build_keep_mask(
        batch_size: int,
        k: int,
        drop_prob: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Keep the first frame as the reference frame.

        Other frames are dropped independently. At least one non-reference
        frame is always retained, so every sample contains a reference and
        at least one comparison frame.
        """
        keep_mask = torch.ones(
            (batch_size, k),
            dtype=torch.bool,
            device=device,
        )

        if drop_prob <= 0.0 or k <= 1:
            return keep_mask

        keep_mask[:, 1:] = torch.rand(
            (batch_size, k - 1),
            device=device,
        ) > drop_prob

        # Make sure at least one non-reference frame remains.
        no_target_frame = ~keep_mask[:, 1:].any(dim=1)

        for batch_index in torch.where(no_target_frame)[0].tolist():
            selected_position = torch.randint(
                low=1,
                high=k,
                size=(1,),
                device=device,
            ).item()
            keep_mask[batch_index, selected_position] = True

        return keep_mask

    @staticmethod
    def _compact_valid_frames(
        features: torch.Tensor,
        keep_mask: torch.Tensor,
    ):
        """
        Remove dropped positions while preserving temporal order.

        Returns:
            compact_features: [B, max_valid_length, D]
            lengths: [B]
        """
        batch_size, _, feature_dim = features.shape
        lengths = keep_mask.sum(dim=1)
        max_length = int(lengths.max().item())

        compact = features.new_zeros(
            (batch_size, max_length, feature_dim)
        )

        for batch_index in range(batch_size):
            valid_features = features[batch_index][keep_mask[batch_index]]
            compact[
                batch_index,
                : valid_features.size(0),
            ] = valid_features

        return compact, lengths

    def forward(
        self,
        x: torch.Tensor,
        override_frame_drop_prob: Optional[float] = None,
    ) -> torch.Tensor:
        # x: [B, K, 3, H, W]
        batch_size, k, channels, height, width = x.shape

        # [B, K, 3, H, W] -> [B*K, 3, H, W]
        flat_frames = x.reshape(
            batch_size * k,
            channels,
            height,
            width,
        )

        if self.freeze_backbone:
            with torch.no_grad():
                frame_features = self.feature_extractor(flat_frames)
        else:
            frame_features = self.feature_extractor(flat_frames)

        # [B*K, 512] -> [B, K, 512]
        frame_features = frame_features.reshape(
            batch_size,
            k,
            self.feature_dim,
        )

        if self.use_change_features:
            reference_feature = frame_features[:, 0:1, :]
            change_features = torch.abs(
                frame_features - reference_feature
            )

            combined_features = torch.cat(
                [frame_features, change_features],
                dim=-1,
            )
        else:
            combined_features = frame_features

        projected_features = self.feature_projection(
            combined_features
        )

        # During normal validation/test, no frames are dropped.
        # A value can be supplied during robustness testing.
        if override_frame_drop_prob is None:
            effective_drop_prob = (
                self.train_frame_drop_prob
                if self.training
                else 0.0
            )
        else:
            effective_drop_prob = override_frame_drop_prob

        keep_mask = self._build_keep_mask(
            batch_size=batch_size,
            k=k,
            drop_prob=effective_drop_prob,
            device=x.device,
        )

        compact_features, lengths = self._compact_valid_frames(
            projected_features,
            keep_mask,
        )

        packed_features = pack_padded_sequence(
            compact_features,
            lengths=lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.temporal_gru(packed_features)
        fused_features = hidden[-1]

        logits = self.classifier(fused_features)
        return logits


# ==================================================
# METRICS
# ==================================================

def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: List[float],
) -> Dict[str, float]:
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )
    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )
    tn, fp, fn, tp = matrix.ravel()

    false_alarm_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "false_alarm_rate": float(false_alarm_rate),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ==================================================
# TRAINING AND EVALUATION
# ==================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
):
    model.train()

    running_loss = 0.0
    all_true = []
    all_pred = []
    all_prob = []

    for batch in tqdm(
        loader,
        desc="Training",
        leave=False,
    ):
        images = batch["images"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRAD_CLIP_NORM,
        )

        optimizer.step()

        probabilities = torch.softmax(logits, dim=1)[:, 1]
        predictions = torch.argmax(logits, dim=1)

        running_loss += loss.item() * images.size(0)

        all_true.extend(labels.detach().cpu().tolist())
        all_pred.extend(predictions.detach().cpu().tolist())
        all_prob.extend(probabilities.detach().cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)

    metrics = calculate_metrics(
        all_true,
        all_pred,
        all_prob,
    )
    metrics["loss"] = float(epoch_loss)

    return metrics


def evaluate(
    model: nn.Module,
    loader,
    criterion,
    split_name: str,
    override_frame_drop_prob: Optional[float] = None,
):
    model.eval()

    running_loss = 0.0
    all_true = []
    all_pred = []
    all_prob = []
    all_site_ids = []
    all_sample_ids = []

    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc=f"Evaluating {split_name}",
            leave=False,
        ):
            images = batch["images"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(
                images,
                override_frame_drop_prob=override_frame_drop_prob,
            )
            loss = criterion(logits, labels)

            probabilities = torch.softmax(logits, dim=1)[:, 1]
            predictions = torch.argmax(logits, dim=1)

            running_loss += loss.item() * images.size(0)

            all_true.extend(labels.detach().cpu().tolist())
            all_pred.extend(predictions.detach().cpu().tolist())
            all_prob.extend(probabilities.detach().cpu().tolist())
            all_site_ids.extend(batch["site_id"])
            all_sample_ids.extend(batch["sample_id"])

    epoch_loss = running_loss / len(loader.dataset)

    metrics = calculate_metrics(
        all_true,
        all_pred,
        all_prob,
    )
    metrics["loss"] = float(epoch_loss)

    predictions_df = pd.DataFrame({
        "site_id": all_site_ids,
        "sample_id": all_sample_ids,
        "true_label": all_true,
        "predicted_label": all_pred,
        "looted_probability": all_prob,
    })

    return metrics, predictions_df


# ==================================================
# CLASS WEIGHTS
# ==================================================

def compute_class_weights(
    csv_path: str,
    strategy: str,
) -> torch.Tensor:
    dataframe = pd.read_csv(csv_path)

    train_df = dataframe[
        (
            dataframe["split"]
            .astype(str)
            .str.lower()
            == "train"
        )
        & (
            dataframe["strategy"]
            .astype(str)
            .str.lower()
            == strategy.lower()
        )
    ]

    class_counts = (
        train_df["label_id"]
        .value_counts()
        .to_dict()
    )

    preserved_count = class_counts.get(0, 1)
    looted_count = class_counts.get(1, 1)
    total = preserved_count + looted_count

    weight_preserved = total / (2 * preserved_count)
    weight_looted = total / (2 * looted_count)

    return torch.tensor(
        [weight_preserved, weight_looted],
        dtype=torch.float32,
    )


# ==================================================
# EXPERIMENT CONFIGURATION
# ==================================================

def build_experiments() -> List[Dict]:
    experiments = [
        {
            "experiment_name": (
                f"step10_K{ABLATION_K}_gru_no_change_no_drop"
            ),
            "variant": "ablation_no_change_no_drop",
            "k": ABLATION_K,
            "use_change_features": False,
            "frame_drop_prob": 0.0,
        },
        {
            "experiment_name": (
                f"step10_K{ABLATION_K}_change_gru_no_drop"
            ),
            "variant": "ablation_change_no_drop",
            "k": ABLATION_K,
            "use_change_features": True,
            "frame_drop_prob": 0.0,
        },
    ]

    for k_value in PROPOSED_K_VALUES:
        experiments.append({
            "experiment_name": (
                f"step10_K{k_value}_change_gru_framedrop"
            ),
            "variant": "proposed_change_gru_framedrop",
            "k": k_value,
            "use_change_features": True,
            "frame_drop_prob": FRAME_DROP_PROB,
        })

    return experiments


# ==================================================
# RUN ONE EXPERIMENT
# ==================================================

def run_experiment(config: Dict) -> Dict:
    set_seed(SEED)

    experiment_name = config["experiment_name"]
    k_value = int(config["k"])
    use_change_features = bool(
        config["use_change_features"]
    )
    frame_drop_prob = float(config["frame_drop_prob"])

    csv_path = (
        KFRAME_DIR
        / f"kframe_samples_K{k_value}.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"K-frame CSV not found: {csv_path}"
        )

    print("\n" + "=" * 72)
    print(f"Experiment: {experiment_name}")
    print(f"K: {k_value}")
    print(f"Strategy: {STRATEGY}")
    print(f"Change features: {use_change_features}")
    print(f"Training frame-drop probability: {frame_drop_prob}")
    print("=" * 72)

    train_loader = create_dataloader(
        csv_path=str(csv_path),
        split="train",
        strategy=STRATEGY,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        train=True,
        shuffle=True,
        num_workers=0,
        normalization="imagenet",
    )

    val_loader = create_dataloader(
        csv_path=str(csv_path),
        split="val",
        strategy=STRATEGY,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        train=False,
        shuffle=False,
        num_workers=0,
        normalization="imagenet",
    )

    test_loader = create_dataloader(
        csv_path=str(csv_path),
        split="test",
        strategy=STRATEGY,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        train=False,
        shuffle=False,
        num_workers=0,
        normalization="imagenet",
    )

    model = ChangeAwareTemporalFusionResNet18(
        use_change_features=use_change_features,
        hidden_dim=HIDDEN_DIM,
        num_classes=2,
        train_frame_drop_prob=frame_drop_prob,
        freeze_backbone=FREEZE_BACKBONE,
    ).to(DEVICE)

    class_weights = compute_class_weights(
        str(csv_path),
        STRATEGY,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history = []
    best_val_f1 = -1.0
    patience_counter = 0

    best_model_path = (
        MODEL_DIR
        / f"{experiment_name}_best.pth"
    )

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_metrics, _ = evaluate(
            model,
            val_loader,
            criterion,
            split_name="val",
        )

        history.append({
            "experiment_name": experiment_name,
            "variant": config["variant"],
            "k": k_value,
            "strategy": STRATEGY,
            "use_change_features": use_change_features,
            "frame_drop_prob": frame_drop_prob,
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
        })

        print(
            f"Train loss={train_metrics['loss']:.4f} | "
            f"Train F1={train_metrics['f1']:.4f} | "
            f"Val loss={val_metrics['loss']:.4f} | "
            f"Val F1={val_metrics['f1']:.4f} | "
            f"Val AUC={val_metrics['roc_auc']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(
                model.state_dict(),
                best_model_path,
            )
            patience_counter = 0
            print(f"Best model saved: {best_model_path}")
        else:
            patience_counter += 1
            print(
                "No validation F1 improvement. "
                f"Patience {patience_counter}/{PATIENCE}"
            )

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    history_df = pd.DataFrame(history)
    history_path = (
        RESULT_DIR
        / f"{experiment_name}_training_history.csv"
    )
    history_df.to_csv(history_path, index=False)

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=DEVICE,
        )
    )

    test_metrics, test_predictions = evaluate(
        model,
        test_loader,
        criterion,
        split_name="test",
    )

    predictions_path = (
        RESULT_DIR
        / f"{experiment_name}_test_predictions.csv"
    )
    test_predictions.to_csv(
        predictions_path,
        index=False,
    )

    test_metrics.update({
        "experiment_name": experiment_name,
        "variant": config["variant"],
        "k": k_value,
        "strategy": STRATEGY,
        "use_change_features": use_change_features,
        "frame_drop_prob": frame_drop_prob,
        "best_validation_f1": best_val_f1,
    })

    metrics_path = (
        RESULT_DIR
        / f"{experiment_name}_test_metrics.json"
    )
    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            test_metrics,
            file,
            indent=4,
        )

    print(f"\nCompleted: {experiment_name}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(
        "Test FAR: "
        f"{test_metrics['false_alarm_rate']:.4f}"
    )

    return {
        **test_metrics,
        "best_model_path": str(best_model_path),
        "history_path": str(history_path),
        "test_predictions_path": str(predictions_path),
        "test_metrics_path": str(metrics_path),
    }


# ==================================================
# ROBUSTNESS TEST
# ==================================================

def evaluate_best_model_under_missing_frames(
    best_result: Dict,
) -> pd.DataFrame:
    """
    Evaluate the best proposed model with additional missing-frame
    simulation at test time.

    This is not used to select the model. It is only a robustness test.
    """
    k_value = int(best_result["k"])
    csv_path = (
        KFRAME_DIR
        / f"kframe_samples_K{k_value}.csv"
    )

    test_loader = create_dataloader(
        csv_path=str(csv_path),
        split="test",
        strategy=STRATEGY,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        train=False,
        shuffle=False,
        num_workers=0,
        normalization="imagenet",
    )

    model = ChangeAwareTemporalFusionResNet18(
        use_change_features=bool(
            best_result["use_change_features"]
        ),
        hidden_dim=HIDDEN_DIM,
        num_classes=2,
        train_frame_drop_prob=float(
            best_result["frame_drop_prob"]
        ),
        freeze_backbone=FREEZE_BACKBONE,
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            best_result["best_model_path"],
            map_location=DEVICE,
        )
    )

    class_weights = compute_class_weights(
        str(csv_path),
        STRATEGY,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    robustness_rows = []

    for drop_probability in [0.0, 0.25, 0.50]:
        # Reset seed so the simulated missing-frame pattern is repeatable.
        set_seed(SEED)

        metrics, _ = evaluate(
            model,
            test_loader,
            criterion,
            split_name=(
                f"test_drop_{drop_probability:.2f}"
            ),
            override_frame_drop_prob=drop_probability,
        )

        robustness_rows.append({
            "experiment_name": best_result["experiment_name"],
            "k": k_value,
            "test_frame_drop_probability": drop_probability,
            **metrics,
        })

    return pd.DataFrame(robustness_rows)


# ==================================================
# FINAL BASELINE COMPARISON
# ==================================================

def _load_json_metrics(
    path: Path,
    model_name: str,
    k_value,
    method: str,
) -> Optional[Dict]:
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    return {
        "model": model_name,
        "k": k_value,
        "method": method,
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "roc_auc": metrics.get("roc_auc"),
        "false_alarm_rate": metrics.get(
            "false_alarm_rate"
        ),
        "tn": metrics.get("tn"),
        "fp": metrics.get("fp"),
        "fn": metrics.get("fn"),
        "tp": metrics.get("tp"),
    }


def create_final_comparison(
    best_proposed_result: Dict,
) -> pd.DataFrame:
    comparison_rows = []

    step6 = _load_json_metrics(
        Path(
            "results/single_frame_baseline/"
            "single_frame_test_metrics.json"
        ),
        model_name="Step 6 Single-frame ResNet18",
        k_value=1,
        method="recent_window",
    )
    if step6 is not None:
        comparison_rows.append(step6)

    step7 = _load_json_metrics(
        Path(
            "results/two_frame_change_baseline/"
            "two_frame_change_test_metrics.json"
        ),
        model_name="Step 7 Two-frame change ResNet18",
        k_value=2,
        method="early_late + absolute difference",
    )
    if step7 is not None:
        comparison_rows.append(step7)

    step8_path = Path(
        "results/temporal_pooling_baseline/"
        "temporal_pooling_summary_results.csv"
    )
    if step8_path.exists():
        step8_df = pd.read_csv(step8_path)
        if not step8_df.empty:
            row = step8_df.sort_values(
                "f1",
                ascending=False,
            ).iloc[0]
            comparison_rows.append({
                "model": "Step 8 Best temporal pooling",
                "k": int(row["k"]),
                "method": (
                    f"uniform + {row['pooling_type']} pooling"
                ),
                "accuracy": row["accuracy"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "roc_auc": row["roc_auc"],
                "false_alarm_rate": row[
                    "false_alarm_rate"
                ],
                "tn": row["tn"],
                "fp": row["fp"],
                "fn": row["fn"],
                "tp": row["tp"],
            })

    step9_path = Path(
        "results/lightweight_temporal_fusion/"
        "lightweight_temporal_fusion_summary_results.csv"
    )
    if step9_path.exists():
        step9_df = pd.read_csv(step9_path)
        if not step9_df.empty:
            row = step9_df.sort_values(
                "f1",
                ascending=False,
            ).iloc[0]
            comparison_rows.append({
                "model": "Step 9 Best lightweight fusion",
                "k": int(row["k"]),
                "method": (
                    f"uniform + {row['fusion_type']}"
                ),
                "accuracy": row["accuracy"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "roc_auc": row["roc_auc"],
                "false_alarm_rate": row[
                    "false_alarm_rate"
                ],
                "tn": row["tn"],
                "fp": row["fp"],
                "fn": row["fn"],
                "tp": row["tp"],
            })

    comparison_rows.append({
        "model": "Step 10 Proposed change-aware GRU",
        "k": int(best_proposed_result["k"]),
        "method": (
            "uniform + reference feature difference "
            "+ GRU + frame drop"
        ),
        "accuracy": best_proposed_result["accuracy"],
        "precision": best_proposed_result["precision"],
        "recall": best_proposed_result["recall"],
        "f1": best_proposed_result["f1"],
        "roc_auc": best_proposed_result["roc_auc"],
        "false_alarm_rate": best_proposed_result[
            "false_alarm_rate"
        ],
        "tn": best_proposed_result["tn"],
        "fp": best_proposed_result["fp"],
        "fn": best_proposed_result["fn"],
        "tp": best_proposed_result["tp"],
    })

    comparison_df = pd.DataFrame(comparison_rows)

    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values(
            "f1",
            ascending=False,
        ).reset_index(drop=True)

    return comparison_df


# ==================================================
# REPORT
# ==================================================

def create_markdown_report(
    results_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> Path:
    report = []

    report.append(
        "# Step 10 Proposed Change-Aware "
        "Lightweight Temporal Fusion Report\n"
    )

    report.append("## Experiment Setup\n")
    report.append(
        "- Feature extractor: pretrained ResNet18"
    )
    report.append(
        "- Change representation: absolute feature "
        "difference from the first selected frame"
    )
    report.append(
        "- Temporal fusion: one-layer GRU"
    )
    report.append(
        f"- Frame selection strategy: {STRATEGY}"
    )
    report.append(
        f"- Proposed K values: {PROPOSED_K_VALUES}"
    )
    report.append(
        f"- Training frame-drop probability: "
        f"{FRAME_DROP_PROB}"
    )
    report.append(
        f"- Image size: {IMAGE_SIZE} x {IMAGE_SIZE}"
    )
    report.append(f"- Batch size: {BATCH_SIZE}")
    report.append(
        f"- Maximum epochs: {NUM_EPOCHS}"
    )
    report.append(
        f"- Learning rate: {LEARNING_RATE}"
    )
    report.append(
        f"- Freeze backbone: {FREEZE_BACKBONE}"
    )
    report.append(f"- Device: {DEVICE}\n")

    report.append("## Step 10 Test Results\n")

    for _, row in results_df.iterrows():
        report.append(
            f"- {row['experiment_name']}: "
            f"Accuracy={row['accuracy']:.4f}, "
            f"Precision={row['precision']:.4f}, "
            f"Recall={row['recall']:.4f}, "
            f"F1={row['f1']:.4f}, "
            f"ROC-AUC={row['roc_auc']:.4f}, "
            f"FAR={row['false_alarm_rate']:.4f}"
        )

    proposed_df = results_df[
        results_df["variant"]
        == "proposed_change_gru_framedrop"
    ].copy()

    best_proposed = proposed_df.sort_values(
        "f1",
        ascending=False,
    ).iloc[0]

    report.append(
        "\n## Best Proposed Model Based on F1-score\n"
    )
    report.append(
        f"- Experiment: {best_proposed['experiment_name']}"
    )
    report.append(f"- K: {int(best_proposed['k'])}")
    report.append(
        f"- Accuracy: {best_proposed['accuracy']:.4f}"
    )
    report.append(
        f"- Precision: {best_proposed['precision']:.4f}"
    )
    report.append(
        f"- Recall: {best_proposed['recall']:.4f}"
    )
    report.append(
        f"- F1-score: {best_proposed['f1']:.4f}"
    )
    report.append(
        f"- ROC-AUC: {best_proposed['roc_auc']:.4f}"
    )
    report.append(
        "- False Alarm Rate: "
        f"{best_proposed['false_alarm_rate']:.4f}\n"
    )

    report.append("## Ablation Study\n")
    ablation_df = results_df[
        results_df["k"] == ABLATION_K
    ].copy()

    for _, row in ablation_df.iterrows():
        report.append(
            f"- {row['variant']}: "
            f"F1={row['f1']:.4f}, "
            f"ROC-AUC={row['roc_auc']:.4f}, "
            f"FAR={row['false_alarm_rate']:.4f}"
        )

    report.append(
        "\n## Missing-Frame Robustness Test\n"
    )
    for _, row in robustness_df.iterrows():
        report.append(
            f"- Test frame-drop={row['test_frame_drop_probability']:.2f}: "
            f"Accuracy={row['accuracy']:.4f}, "
            f"Recall={row['recall']:.4f}, "
            f"F1={row['f1']:.4f}, "
            f"FAR={row['false_alarm_rate']:.4f}"
        )

    report.append("\n## Final Baseline Comparison\n")
    for _, row in comparison_df.iterrows():
        report.append(
            f"- {row['model']}: "
            f"K={row['k']}, "
            f"Accuracy={row['accuracy']:.4f}, "
            f"Precision={row['precision']:.4f}, "
            f"Recall={row['recall']:.4f}, "
            f"F1={row['f1']:.4f}, "
            f"ROC-AUC={row['roc_auc']:.4f}, "
            f"FAR={row['false_alarm_rate']:.4f}"
        )

    report.append("\n## Output Files\n")
    report.append(
        f"- `{RESULT_DIR / 'step10_summary_results.csv'}`"
    )
    report.append(
        f"- `{RESULT_DIR / 'step10_ablation_results.csv'}`"
    )
    report.append(
        f"- `{RESULT_DIR / 'step10_k_analysis_results.csv'}`"
    )
    report.append(
        f"- `{RESULT_DIR / 'step10_frame_drop_robustness.csv'}`"
    )
    report.append(
        f"- `{FINAL_COMPARISON_DIR / 'final_model_comparison.csv'}`"
    )

    report_path = (
        TABLE_DIR
        / "step10_proposed_model_report.md"
    )
    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    return report_path


# ==================================================
# MAIN
# ==================================================

def main():
    set_seed(SEED)

    print(f"Using device: {DEVICE}")
    print(
        "Running Step 10 proposed "
        "change-aware temporal-fusion experiments."
    )

    experiments = build_experiments()
    results = []

    for config in experiments:
        result = run_experiment(config)
        results.append(result)

    results_df = pd.DataFrame(results)

    summary_path = (
        RESULT_DIR
        / "step10_summary_results.csv"
    )
    results_df.to_csv(summary_path, index=False)

    ablation_df = results_df[
        results_df["k"] == ABLATION_K
    ].copy()
    ablation_path = (
        RESULT_DIR
        / "step10_ablation_results.csv"
    )
    ablation_df.to_csv(ablation_path, index=False)

    k_analysis_df = results_df[
        results_df["variant"]
        == "proposed_change_gru_framedrop"
    ].copy()
    k_analysis_path = (
        RESULT_DIR
        / "step10_k_analysis_results.csv"
    )
    k_analysis_df.to_csv(
        k_analysis_path,
        index=False,
    )

    best_proposed_result = (
        k_analysis_df
        .sort_values("f1", ascending=False)
        .iloc[0]
        .to_dict()
    )

    robustness_df = (
        evaluate_best_model_under_missing_frames(
            best_proposed_result
        )
    )
    robustness_path = (
        RESULT_DIR
        / "step10_frame_drop_robustness.csv"
    )
    robustness_df.to_csv(
        robustness_path,
        index=False,
    )

    comparison_df = create_final_comparison(
        best_proposed_result
    )
    comparison_path = (
        FINAL_COMPARISON_DIR
        / "final_model_comparison.csv"
    )
    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    report_path = create_markdown_report(
        results_df=results_df,
        robustness_df=robustness_df,
        comparison_df=comparison_df,
    )

    print("\nStep 10 completed.")
    print(f"Summary results: {summary_path}")
    print(f"Ablation results: {ablation_path}")
    print(f"K analysis: {k_analysis_path}")
    print(f"Robustness results: {robustness_path}")
    print(f"Final comparison: {comparison_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
