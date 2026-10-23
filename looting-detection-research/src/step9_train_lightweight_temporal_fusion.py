"""Step 9: ResNet18 + GRU/attention fusion with checked data and selection.

Put this file beside the corrected Step 5 dataset_loader.py, in src or the
project root. From the project root:
    python src/step9_train_lightweight_temporal_fusion.py --device cuda

Default suite: K=3,5,8; GRU and attention pooling; all four frame strategies:
recent_window, uniform, early_late, random (24 experiments, run sequentially).
All models train first. The best configuration is selected by validation F1
BEFORE any test predictions. All predefined models are then tested once.
Ties: smaller K, then GRU, then the strategy order listed above.
Test scores never select a configuration.

Optional subset example: --strategies uniform early_late --k-values 3 5
The older --strategy uniform spelling also works for a single strategy.

Batch size defaults to 2 to reduce memory use for K=8. No mixed precision is
used. The backbone is fine-tuned by default. --freeze-backbone freezes its
parameters AND keeps its BatchNorm statistics in evaluation mode.
This is one fixed split, not cross-validation. GRU uses frame order; attention
pooling has no position/date encoding. Neither method receives time gaps.
These are fixed-K experiments, not full-sequence or mixed-length training.
Few images can still span a long calendar period. No old results are reused.

Models: models/lightweight_temporal_fusion/<run>/
Reports: results/lightweight_temporal_fusion/<run>/
Dependencies: existing PyTorch/torchvision, pandas, NumPy, Pillow, scikit-learn;
tqdm is optional. A first run may download ImageNet weights. Use --preflight-only
for metadata/file-existence checks without image decoding or training.
"""
from __future__ import annotations

import argparse
import gc
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
try:
    from tqdm import tqdm
except ImportError:
    # Progress bars are optional; their absence must not prevent training.
    def tqdm(iterable, **kwargs):
        return iterable

import dataset_loader as dl


# These are the agreed counts for the current experiment (validation pool 1).
# Change them only if you intentionally define and document another protocol.
EXPECTED_COUNTS = {
    "train": {"preserved": 496, "looted": 100},
    "val": {"preserved": 9, "looted": 9},
    "test": {"preserved": 35, "looted": 26},
}
SPLITS = ("train", "val", "test")
STRATEGIES = ("recent_window", "uniform", "early_late", "random")
LABELS = {"preserved": 0, "looted": 1}
K_VALUES = (3, 5, 8)
FUSION_TYPES = ("gru", "attention")
SELECTION_RULE = (
    "highest best validation F1; ties: smaller K, then GRU, then strategy order "
    "recent_window, uniform, early_late, random"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def find_project_root(explicit=None, script_path=None, working_directory=None):
    """Support both project/script.py and project/src/script.py."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    script_dir = Path(script_path or __file__).resolve().parent
    cwd = Path(working_directory or Path.cwd()).resolve()
    # Prefer this script's project over a possibly unrelated terminal folder.
    candidates = [script_dir, script_dir.parent, cwd]
    for candidate in candidates:
        if (candidate / "data/splits").is_dir():
            return candidate
    # A useful fallback when Step 3 has not yet created its outputs.
    return script_dir.parent if script_dir.name.lower() == "src" else script_dir


def check_loader_version():
    required = ("read_samples_csv", "relative_path", "integer_column", "require_torch", "create_dataloader")
    require(all(hasattr(dl, name) for name in required),
            "Use the corrected Step 5 dataset_loader.py beside this script.")
    parameters = inspect.signature(dl.create_dataloader).parameters
    require({"data_root", "seed", "k", "pin_memory"}.issubset(parameters),
            "Your dataset_loader.py is outdated. Replace it with the corrected Step 5 version.")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, value):
    # Reject NaN/Infinity instead of silently writing invalid JSON values.
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_metadata(sites, frames, samples, k):
    """Compare identities, memberships, frame paths and dates with Step 3."""
    sites, frames = sites.copy(), frames.copy()
    for table, required, name in [
        (sites, {"site_uid", "site_id", "label", "label_id", "split", "frame_count"}, "site_splits.csv"),
        (frames, {"site_uid", "site_id", "label", "label_id", "split", "relative_path",
                  "frame_index", "year", "month"}, "frames_with_splits.csv"),
    ]:
        require(not table.empty and required.issubset(table.columns),
                f"Missing corrected Step 3 data/columns in {name}.")
        require(table[list(required)].astype(str).apply(lambda c: c.str.strip().ne("")).all().all(),
                f"Blank required values in {name}.")
        require(table["site_uid"].eq(table["label"] + "/" + table["site_id"]).all(),
                f"Invalid site_uid in {name}. Rerun corrected Step 3.")
        require(table["label"].isin(LABELS).all() and table["split"].isin(SPLITS).all(),
                f"Unknown label or split in {name}.")
        table["label_id"] = dl.integer_column(table["label_id"], "label_id")
        require(table["label_id"].eq(table["label"].map(LABELS)).all(), f"Incorrect labels in {name}.")
        require(table.groupby("site_uid")["split"].nunique().eq(1).all(),
                f"A site appears in multiple splits in {name}.")
    require(sites["site_uid"].is_unique, "Duplicate site_uid in Step 3 site table.")
    require(set(sites["site_uid"]) == set(frames["site_uid"]), "Step 3 site/frame membership differs.")
    counts = {
        split: {label: int(((sites["split"] == split) & (sites["label"] == label)).sum())
                for label in LABELS} for split in SPLITS
    }
    require(counts == EXPECTED_COUNTS,
            f"Unexpected Step 3 counts: {counts}. Expected {EXPECTED_COUNTS}. "
            "Rerun corrected Steps 3-5. An old 87-site test split must not be used.")
    sites["frame_count"] = dl.integer_column(sites["frame_count"], "frame_count")
    require(sites["frame_count"].ge(k).all(), f"Every site in this experiment needs at least {k} frames.")
    for column in ("frame_index", "year", "month"):
        frames[column] = dl.integer_column(frames[column], column)
    require(frames["year"].between(1, 9999).all() and frames["month"].between(1, 12).all(),
            "Invalid frame year/month in Step 3.")
    frames["relative_path"] = frames["relative_path"].map(dl.relative_path)
    require(frames["relative_path"].str.casefold().is_unique, "Repeated Step 3 image paths.")
    for row in frames[["site_uid", "relative_path"]].itertuples(index=False):
        parts = PurePosixPath(row.relative_path).parts
        require(len(parts) >= 3 and f"{parts[0].lower()}/{parts[1]}" == row.site_uid,
                f"Frame belongs to another site: {row.relative_path}.")
    frames = frames.sort_values(["site_uid", "frame_index"]).reset_index(drop=True)
    require(frames["frame_index"].eq(frames.groupby("site_uid").cumcount()).all(),
            "Frame indices must be consecutive from zero for each site.")
    month_number = frames["year"] * 12 + frames["month"]
    require(month_number.groupby(frames["site_uid"]).diff().fillna(0).ge(0).all(),
            "Step 3 frame indices are not in chronological order.")
    lookup = sites.set_index("site_uid")
    for column in ("site_id", "label", "label_id", "split"):
        require(frames[column].eq(frames["site_uid"].map(lookup[column])).all(),
                f"Step 3 tables disagree on {column}.")
    actual_counts = frames.groupby("site_uid").size()
    require(lookup["frame_count"].sort_index().equals(actual_counts.sort_index()),
            "Step 3 frame counts disagree.")
    require(samples["k"].eq(k).all(), f"The K{k} CSV contains a different K.")
    require(set(samples["strategy"]) == set(STRATEGIES), f"The K{k} CSV is missing a required strategy.")
    for strategy in STRATEGIES:
        for split in SPLITS:
            actual = samples[(samples["strategy"] == strategy) & (samples["split"] == split)]
            expected = set(sites.loc[sites["split"] == split, "site_uid"])
            require(actual["site_uid"].is_unique and set(actual["site_uid"]) == expected,
                    f"K{k}/{strategy}/{split} differs from current Step 3. Rerun corrected Step 4.")
    frame_lookup = {
        (r.site_uid, r.frame_index): (r.relative_path, r.year, r.month)
        for r in frames.itertuples(index=False)
    }
    for row in samples.itertuples(index=False):
        site = lookup.loc[row.site_uid]
        require(row.split == site["split"] and row.label_id == site["label_id"]
                and row.available_frames == site["frame_count"],
                f"Stale Step 4 metadata: {row.sample_id}.")
        require(len(row.frame_indices) == k and row.frame_indices == sorted(set(row.frame_indices)),
                f"Exactly K different frames in date order are required: {row.sample_id}.")
        for index, path, year, month in zip(row.frame_indices, row.relative_paths, row.years, row.months):
            require(frame_lookup.get((row.site_uid, index)) == (path, year, month),
                    f"Selected frame/date disagrees with Step 3: {row.sample_id}, frame {index}.")
        n = int(site["frame_count"])
        expected_indices = None
        if row.strategy == "uniform":
            expected_indices = [round(i * (n - 1) / (k - 1)) for i in range(k)]
        elif row.strategy == "early_late":
            expected_indices = [0] + list(range(n - k + 1, n))
        elif row.strategy == "recent_window":
            expected_indices = list(range(n - k, n))
        if expected_indices is not None:
            require(row.frame_indices == expected_indices,
                    f"K{k}/{row.strategy} frame selection is incorrect: {row.sample_id}.")
    return sites, counts


def preflight(root, data_root, strategies, k_values):
    check_loader_version()
    paths = {
        "site_splits": root / "data/splits/site_splits.csv",
        "frames_with_splits": root / "data/splits/frames_with_splits.csv",
        "dataset_loader": Path(dl.__file__).resolve(),
        "training_script": Path(__file__).resolve(),
    }
    for k in k_values:
        paths[f"k{k}_samples"] = root / f"data/processed/kframe_sequences/kframe_samples_K{k}.csv"
    for path in paths.values():
        require(path.is_file(), f"Required file is missing: {path}")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    sites = pd.read_csv(paths["site_splits"], dtype=str, keep_default_na=False)
    frames = pd.read_csv(paths["frames_with_splits"], dtype=str, keep_default_na=False)
    require(data_root.is_dir(), f"Image directory is missing: {data_root}. Use --data-root if it moved.")
    tables, counts, checked_paths = {}, None, set()
    for k in k_values:
        print(f"Checking K={k} metadata and selected image paths...", flush=True)
        samples = dl.read_samples_csv(paths[f"k{k}_samples"])
        _, counts = validate_metadata(sites, frames, samples, k)
        for strategy in strategies:
            tables[(k, strategy)] = samples[samples["strategy"] == strategy].copy()
            for row in tables[(k, strategy)].itertuples(index=False):
                for relative in row.relative_paths:
                    if relative not in checked_paths:
                        path = data_root.joinpath(*PurePosixPath(relative).parts)
                        require(path.is_file(), f"Selected image is missing for K{k}/{strategy}: {path}")
                        checked_paths.add(relative)
    require(all(sha256_file(paths[name]) == digest for name, digest in hashes.items()),
            "An input file changed during preflight. Stop editing inputs and rerun.")
    inputs = {name: {"path": str(path), "sha256": hashes[name]} for name, path in paths.items()}
    return tables, counts, inputs


def class_weights_from_labels(labels):
    values = np.asarray(labels)
    require(values.ndim == 1 and values.size > 0 and np.isin(values, [0, 1]).all(),
            "Training labels must be a nonempty list of 0/1 values.")
    counts = np.bincount(values.astype(np.int64), minlength=2)
    require((counts > 0).all(), "Both preserved and looted classes must be present in training.")
    return len(values) / (2.0 * counts)


def calculate_metrics(y_true, y_pred, y_prob):
    y_true, y_pred, y_prob = (np.asarray(v) for v in (y_true, y_pred, y_prob))
    require(y_true.ndim == y_pred.ndim == y_prob.ndim == 1
            and 0 < len(y_true) == len(y_pred) == len(y_prob), "Invalid metric array shapes/counts.")
    require(np.isin(y_true, [0, 1]).all() and np.isin(y_pred, [0, 1]).all(), "Invalid metric labels.")
    require(np.isfinite(y_prob).all() and ((0 <= y_prob) & (y_prob <= 1)).all(),
            "Probabilities must be finite numbers in [0,1].")
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    # None is saved as JSON null; it does not pretend an undefined score is zero.
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else None
    if auc is not None:
        require(math.isfinite(auc), "ROC-AUC returned a non-finite value.")
    return {
        "samples": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)), "roc_auc": auc,
        "false_alarm_rate": float(fp / (fp + tn)) if fp + tn else None,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


class WeightedLossMeter:
    """Combine batch means using their weight sums, not their sample counts."""
    def __init__(self):
        self.numerator = 0.0
        self.denominator = 0.0

    def add(self, batch_mean, target_weight_sum):
        require(math.isfinite(batch_mean) and math.isfinite(target_weight_sum)
                and target_weight_sum > 0, "Non-finite loss or invalid loss denominator.")
        self.numerator += batch_mean * target_weight_sum
        self.denominator += target_weight_sum

    def mean(self):
        require(self.denominator > 0, "No samples were included in the loss.")
        result = self.numerator / self.denominator
        require(math.isfinite(result), "Epoch loss is non-finite.")
        return result


def check_temporal_shape(shape, k):
    require(len(shape) == 5 and shape[0] > 0 and shape[1] == k and shape[2] == 3
            and shape[3] > 0 and shape[4] > 0,
            f"Expected [B,{k},3,H,W], received {tuple(shape)}. K must match this experiment.")


def fuse_features(torch, features, fusion_type, fusion_module):
    """Return one feature vector per site and optional attention weights."""
    require(len(features.shape) == 3 and all(v > 0 for v in features.shape),
            "Features must have shape [B,K,D] with nonempty dimensions.")
    require(fusion_type in FUSION_TYPES, "Unknown temporal fusion type.")
    if fusion_type == "gru":
        _, hidden = fusion_module(features)
        require(len(hidden.shape) == 3 and hidden.shape[0] == 1
                and hidden.shape[1] == features.shape[0] and hidden.shape[2] > 0,
                "Expected one-layer, one-direction GRU hidden state [1,B,H].")
        return hidden[-1], None
    scores = fusion_module(features)
    require(tuple(scores.shape) == (features.shape[0], features.shape[1], 1),
            "Attention scores must have shape [B,K,1].")
    weights = torch.softmax(scores, dim=1)
    return torch.sum(features * weights, dim=1), weights


def configure_backbone(backbone, frozen, training):
    """Freezing parameters alone would still allow BatchNorm statistics to change."""
    for parameter in backbone.parameters():
        parameter.requires_grad_(not frozen)
    backbone.train(bool(training) and not frozen)


def build_model(torch, k, fusion_type, dropout=0.3, pretrained=True,
                freeze_backbone=False, gru_hidden_dim=256, attention_dim=128):
    from torchvision import models

    require(k in K_VALUES and fusion_type in FUSION_TYPES, "Invalid K/fusion configuration.")
    require(gru_hidden_dim > 0 and attention_dim > 0, "Fusion dimensions must be positive.")

    class LightweightTemporalFusionResNet18(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.k = k
            self.fusion_type = fusion_type
            self.freeze_backbone = bool(freeze_backbone)
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
            self.feature_dim = backbone.fc.in_features
            backbone.fc = torch.nn.Identity()
            self.feature_extractor = backbone
            configure_backbone(self.feature_extractor, self.freeze_backbone, self.training)
            if fusion_type == "gru":
                self.temporal_fusion = torch.nn.GRU(
                    input_size=self.feature_dim, hidden_size=gru_hidden_dim,
                    num_layers=1, batch_first=True, bidirectional=False,
                )
                classifier_input_dim = gru_hidden_dim
            else:
                # Frame scores depend on image content, without position or date encoding.
                self.temporal_fusion = torch.nn.Sequential(
                    torch.nn.Linear(self.feature_dim, attention_dim), torch.nn.Tanh(),
                    torch.nn.Linear(attention_dim, 1),
                )
                classifier_input_dim = self.feature_dim
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=dropout), torch.nn.Linear(classifier_input_dim, 2)
            )

        def train(self, mode=True):
            super().train(mode)
            configure_backbone(self.feature_extractor, self.freeze_backbone, mode)
            return self

        def forward(self, images):
            check_temporal_shape(images.shape, self.k)
            b, k_actual, channels, h, w = images.shape
            context = torch.no_grad() if self.freeze_backbone else nullcontext()
            with context:
                features = self.feature_extractor(images.reshape(b * k_actual, channels, h, w))
            features = features.reshape(b, k_actual, self.feature_dim)
            fused, _ = fuse_features(torch, features, self.fusion_type, self.temporal_fusion)
            return self.classifier(fused)

    return LightweightTemporalFusionResNet18()


def parameter_counts(model):
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "backbone": sum(p.numel() for p in model.feature_extractor.parameters()),
        "fusion": sum(p.numel() for p in model.temporal_fusion.parameters()),
        "classifier": sum(p.numel() for p in model.classifier.parameters()),
    }


def set_seed(torch, seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def choose_device(torch, requested):
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA was requested but is unavailable in this Python environment. "
                           "Use your CUDA-enabled environment, or choose --device cpu explicitly.")
    return torch.device("cuda" if requested != "cpu" and available else "cpu")


def validate_batch(torch, batch, expected, seen, split, strategy, image_size, k):
    require("images" in batch and "label" in batch, "Batch is missing images or labels.")
    images, labels = batch["images"], batch["label"]
    require(isinstance(images, torch.Tensor), "Images must be tensors.")
    check_temporal_shape(images.shape, k)
    b = images.shape[0]
    require(tuple(images.shape[-2:]) == (image_size, image_size)
            and images.dtype == torch.float32 and torch.isfinite(images).all().item(),
            "Wrong image size/dtype or non-finite pixels.")
    require(isinstance(labels, torch.Tensor) and labels.dtype == torch.long
            and tuple(labels.shape) == (b,), "Labels must have shape [B] and dtype torch.long.")
    for name in ("site_uid", "site_id", "sample_id", "split", "strategy", "k"):
        require(name in batch and len(batch[name]) == b, f"Missing/invalid batch field: {name}.")
    for i, sample_id in enumerate(batch["sample_id"]):
        require(sample_id in expected and sample_id not in seen, "Unknown or repeated sample in loader.")
        row = expected[sample_id]
        require(batch["site_uid"][i] == row["site_uid"] and batch["site_id"][i] == row["site_id"]
                and int(labels[i].item()) == row["label_id"], "Batch identity or label differs from metadata.")
        require(batch["split"][i] == split and batch["strategy"][i] == strategy and int(batch["k"][i]) == k,
                "Batch split/strategy/K differs from this experiment.")
        seen.add(sample_id)


def run_epoch(torch, model, loader, criterion, device, expected, split, strategy,
              image_size, k, optimizer=None):
    training = optimizer is not None
    require(not training or split == "train", "Only the training split may update model weights.")
    model.train(training)
    meter, seen, records = WeightedLossMeter(), set(), []
    context = nullcontext() if training else torch.inference_mode()
    with context:
        for batch in tqdm(loader, desc="Training" if training else f"Evaluating {split}", leave=False):
            validate_batch(torch, batch, expected, seen, split, strategy, image_size, k)
            images = batch["images"].to(device, non_blocking=device.type == "cuda")
            labels = batch["label"].to(device, non_blocking=device.type == "cuda")
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            require(tuple(logits.shape) == (images.shape[0], 2) and torch.isfinite(logits).all().item(),
                    "Model output must be finite logits with shape [B,2].")
            loss = criterion(logits, labels)
            # Weighted CrossEntropyLoss(mean) divides by SUM of target weights.
            denominator = float(criterion.weight[labels].sum().item())
            meter.add(float(loss.detach().item()), denominator)
            if training:
                loss.backward()
                require(all(p.grad is None or torch.isfinite(p.grad).all().item()
                            for p in model.parameters()), "Non-finite gradient. Training stopped.")
                optimizer.step()
            probabilities = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().tolist()
            predictions = logits.detach().argmax(dim=1).cpu().tolist()
            truths = labels.detach().cpu().tolist()
            for i, sample_id in enumerate(batch["sample_id"]):
                row = expected[sample_id]
                records.append({
                    "site_uid": row["site_uid"], "site_id": row["site_id"], "sample_id": sample_id,
                    "split": split, "strategy": strategy, "k": k,
                    "frame_indices": json.dumps(row["frame_indices"]),
                    "years": json.dumps(row["years"]), "months": json.dumps(row["months"]),
                    "relative_paths": json.dumps(row["relative_paths"]),
                    "true_label": int(truths[i]), "predicted_label": int(predictions[i]),
                    "looted_probability": float(probabilities[i]),
                })
    require(seen == set(expected) and len(records) == len(expected),
            "Loader did not process every expected sample exactly once.")
    predictions_df = pd.DataFrame(records)
    metrics = calculate_metrics(predictions_df["true_label"], predictions_df["predicted_label"],
                                predictions_df["looted_probability"])
    metrics["loss"] = meter.mean()
    return metrics, predictions_df


def format_score(value):
    return "N/A" if value is None else f"{value:.4f}"


def check_inputs_unchanged(inputs):
    for name, info in inputs.items():
        require(sha256_file(info["path"]) == info["sha256"], f"Input changed during the run: {name}.")


def make_loader(args, device, samples, csv_path, data_root, k, strategy, split):
    require(not samples.empty and samples["k"].eq(k).all()
            and samples["strategy"].eq(strategy).all(), "Wrong K/strategy table supplied to the loader.")
    expected = samples.loc[samples["split"] == split].set_index("sample_id").to_dict("index")
    loader = dl.create_dataloader(
        csv_path=csv_path, split=split, strategy=strategy,
        batch_size=args.batch_size, image_size=args.image_size, train=split == "train",
        shuffle=split == "train", num_workers=args.num_workers, normalization="imagenet",
        data_root=data_root, seed=args.seed, k=k, pin_memory=device.type == "cuda",
    )
    actual = loader.dataset.df.set_index("sample_id").to_dict("index")
    require(actual == expected, "Loader metadata differs from the preflight snapshot.")
    return loader, expected


def train_experiment(torch, args, device, samples, inputs, config, run_dir, model_dir, k, strategy, fusion):
    # Reset before model initialization, augmentation, and DataLoader creation.
    set_seed(torch, args.seed, args.deterministic)
    name = f"K{k}_{strategy}_{fusion}"
    folder = run_dir / name
    folder.mkdir(exist_ok=False)
    checkpoint = model_dir / f"{name}_best.pth"
    state = {"status": "TRAINING", "experiment_name": name, "k": k, "fusion_type": fusion,
             "strategy": strategy, "seed": args.seed, "config": config,
             "best_model_path": str(checkpoint)}
    state_path = folder / "experiment_metadata.json"
    json_write(state_path, state)
    try:
        loaders, expected = {}, {}
        for split in ("train", "val"):
            loaders[split], expected[split] = make_loader(
                args, device, samples, inputs[f"k{k}_samples"]["path"], config["data_root"], k, strategy, split)
        weights = class_weights_from_labels(samples.loc[samples["split"] == "train", "label_id"])
        model = build_model(torch, k, fusion, args.dropout, pretrained=True,
                            freeze_backbone=args.freeze_backbone, gru_hidden_dim=args.gru_hidden_dim,
                            attention_dim=args.attention_dim).to(device)
        parameters = parameter_counts(model)
        state["parameter_counts"] = parameters
        json_write(state_path, state)
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=args.learning_rate, weight_decay=args.weight_decay)
        history, best_val_f1, best_epoch, patience = [], -1.0, 0, 0
        best_val_metrics = None
        stopping_reason = "maximum epochs reached"
        print(f"\nTraining {name}; seed={args.seed}; batch size={args.batch_size}", flush=True)
        for epoch in range(1, args.epochs + 1):
            train_metrics, _ = run_epoch(torch, model, loaders["train"], criterion, device, expected["train"],
                                         "train", strategy, args.image_size, k, optimizer)
            val_metrics, val_predictions = run_epoch(torch, model, loaders["val"], criterion, device, expected["val"],
                                                     "val", strategy, args.image_size, k)
            history.append({"epoch": epoch, **{f"train_{n}": v for n, v in train_metrics.items()},
                            **{f"val_{n}": v for n, v in val_metrics.items()}})
            pd.DataFrame(history).to_csv(folder / "training_history.csv", index=False)
            print(f"{name} epoch {epoch}/{args.epochs}: train loss={train_metrics['loss']:.4f}, "
                  f"val F1={val_metrics['f1']:.4f}, val AUC={format_score(val_metrics['roc_auc'])}", flush=True)
            if val_metrics["f1"] > best_val_f1:
                best_val_f1, best_epoch, patience = val_metrics["f1"], epoch, 0
                best_val_metrics = val_metrics.copy()
                temporary = checkpoint.with_suffix(".tmp")
                torch.save(model.state_dict(), temporary)
                temporary.replace(checkpoint)
                val_predictions["fusion_type"] = fusion
                val_predictions.to_csv(folder / "best_validation_predictions.csv", index=False)
                json_write(folder / "best_validation_metrics.json", {"epoch": epoch, **val_metrics})
                print(f"Saved best checkpoint: epoch {epoch}", flush=True)
            else:
                patience += 1
            state["progress"] = {"epochs_completed": epoch, "best_epoch": best_epoch, "best_val_f1": best_val_f1}
            json_write(state_path, state)
            if patience >= args.patience:
                stopping_reason = "early stopping: validation F1 did not improve"
                break
        require(checkpoint.is_file() and best_val_metrics is not None, "No valid checkpoint was saved.")
        result = {"status": "TRAINED", "experiment_name": name, "k": k, "strategy": strategy,
                  "fusion_type": fusion, "seed": args.seed, "epochs_completed": len(history),
                  "best_epoch": best_epoch, "best_val_f1": best_val_f1,
                  "best_validation_metrics": best_val_metrics, "stopping_reason": stopping_reason,
                  "class_weights": weights.tolist(), "best_model_path": str(checkpoint),
                  "parameter_counts": parameters,
                  "checkpoint_sha256": sha256_file(checkpoint), "output_dir": str(folder)}
        state.update(result)
        json_write(state_path, state)
        return result
    except BaseException as exc:
        state.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
        json_write(state_path, state)
        raise


def check_experiment_coverage(results, planned):
    observed = [(r["k"], r["strategy"], r["fusion_type"]) for r in results]
    require(bool(planned) and len(planned) == len(set(planned))
            and len(observed) == len(set(observed)) and set(observed) == set(planned),
            "Missing or repeated experiments. An incomplete suite cannot be reported as complete.")


def select_configuration(results):
    """Uses only validation F1 and the predefined tie rule, never test fields."""
    require(bool(results), "There are no experiments to select from.")
    for row in results:
        score = row.get("best_val_f1")
        require(isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 1,
                "A valid best validation F1 is required for every experiment.")
        require(row["k"] in K_VALUES and row["strategy"] in STRATEGIES
                and row["fusion_type"] in FUSION_TYPES, "Invalid selection candidate.")
    return min(results, key=lambda r: (-r["best_val_f1"], r["k"],
                                       FUSION_TYPES.index(r["fusion_type"]),
                                       STRATEGIES.index(r["strategy"]))).copy()


def test_experiment(torch, args, device, samples, inputs, config, trained):
    k, strategy, fusion = trained["k"], trained["strategy"], trained["fusion_type"]
    folder, checkpoint = Path(trained["output_dir"]), Path(trained["best_model_path"])
    require(sha256_file(checkpoint) == trained["checkpoint_sha256"], "Saved checkpoint changed before testing.")
    loader, expected = make_loader(args, device, samples, inputs[f"k{k}_samples"]["path"],
                                   config["data_root"], k, strategy, "test")
    # Architecture only; trained weights are restored from the selected epoch.
    model = build_model(torch, k, fusion, args.dropout, pretrained=False,
                        freeze_backbone=args.freeze_backbone, gru_hidden_dim=args.gru_hidden_dim,
                        attention_dim=args.attention_dim).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    weights = torch.tensor(trained["class_weights"], dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    metrics, predictions = run_epoch(torch, model, loader, criterion, device, expected,
                                    "test", strategy, args.image_size, k)
    require(metrics["samples"] == 61 and metrics["tn"] + metrics["fp"] == 35
            and metrics["fn"] + metrics["tp"] == 26, "Unexpected test counts.")
    predictions["fusion_type"] = fusion
    predictions.to_csv(folder / "test_predictions.csv", index=False)
    json_write(folder / "test_metrics.json", metrics)
    state_path = folder / "experiment_metadata.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update({"status": "COMPLETE", "test_metrics": metrics})
    json_write(state_path, state)
    print(f"Tested {trained['experiment_name']}: F1={metrics['f1']:.4f}; samples=61", flush=True)
    return metrics


def summary_row(trained, metrics, selected_name):
    keys = ("experiment_name", "k", "strategy", "fusion_type", "seed", "epochs_completed",
            "best_epoch", "best_val_f1", "stopping_reason", "best_model_path", "output_dir")
    return {**{key: trained[key] for key in keys}, "status": "COMPLETE",
            "selected_by_validation": trained["experiment_name"] == selected_name,
            **{f"parameters_{key}": value for key, value in trained["parameter_counts"].items()},
            **{f"test_{key}": value for key, value in metrics.items()}}


def write_report(run_dir, model_dir, config, counts, results, selected):
    lines = ["# Step 9 Lightweight Temporal Fusion Baseline Report", "", "## Status: COMPLETE", "",
             f"- Run: {run_dir.name}", f"- Completed experiments: {len(results)}",
             "- Model: shared ImageNet-pretrained ResNet18 feature extractor + GRU or attention pooling",
             f"- Backbone frozen: {config['freeze_backbone']} (when frozen, BatchNorm statistics are also fixed)",
             f"- GRU: one layer, one direction, hidden size {config['gru_hidden_dim']}",
             f"- Attention: Linear(512,{config['attention_dim']}) -> Tanh -> Linear({config['attention_dim']},1) -> softmax over frames",
             f"- K values: {config['k_values']}", f"- Fusion types: {config['fusion_types']}",
             f"- Frame selection strategies: {config['strategies']}", "- Labels: preserved=0, looted=1",
             f"- Batch size in sites: {config['batch_size']}", f"- Image size: {config['image_size']} x {config['image_size']}",
             f"- Dropout probability: {config['dropout']}", "- Normalization: ImageNet; whole images, no site masking",
             f"- Seed reset independently for every training experiment: {config['seed']}",
             f"- Epochs planned per experiment: {config['epochs']}", f"- Early stopping patience: {config['patience']}",
             f"- Learning rate: {config['learning_rate']}", f"- Weight decay: {config['weight_decay']}",
             f"- Device used: {config['actual_device']}", f"- GPU name: {config['gpu_name'] or 'N/A'}",
             f"- PyTorch: {config['torch_version']}", f"- torchvision: {config['torchvision_version']}",
             f"- Class weights [preserved, looted]: {config['class_weights']}",
             f"- Deterministic algorithms requested: {config['deterministic']}",
             "- Precision: float32; no automatic mixed precision",
             "- Checkpoint selection within each experiment: validation F1; ties keep the earlier epoch",
             f"- Configuration selection: {SELECTION_RULE}",
             "- Configuration selection was saved before any test predictions in this run.",
             "- Each predefined configuration was tested once after selection was fixed.",
             "", "## Checked site counts", "", "| Split | Preserved | Looted | Total |", "|---|---:|---:|---:|"]
    for split in SPLITS:
        c = counts[split]
        lines.append(f"| {split} | {c['preserved']} | {c['looted']} | {sum(c.values())} |")
    lines += ["", "## Results for predefined configurations", "",
              "Selected means chosen by validation F1, not by the test score.", "",
              "| K | Strategy | Fusion | Epochs | Best epoch | Best val F1 | Test accuracy | Test precision | Test recall | Test F1 | Test AUC | Test FAR | Selected |",
              "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in results:
        scores = [format_score(row[key]) for key in ("best_val_f1", "test_accuracy", "test_precision",
                   "test_recall", "test_f1", "test_roc_auc", "test_false_alarm_rate")]
        chosen = "yes" if row["selected_by_validation"] else ""
        lines.append(f"| {row['k']} | {row['strategy']} | {row['fusion_type']} | {row['epochs_completed']} | {row['best_epoch']} | "
                     + " | ".join(scores) + f" | {chosen} |")
    lines += ["", "## Test counts and loss", "", "| K | Strategy | Fusion | Samples | TN | FP | FN | TP | Weighted loss |",
              "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['k']} | {r['strategy']} | {r['fusion_type']} | {r['test_samples']} | {r['test_tn']} | {r['test_fp']} | "
                     f"{r['test_fn']} | {r['test_tp']} | {format_score(r['test_loss'])} |")
    chosen = next(r for r in results if r["experiment_name"] == selected["experiment_name"])
    lines += ["", "## Configuration selected using validation", "",
              f"- Experiment: {selected['experiment_name']}", f"- Strategy: {selected['strategy']}",
              f"- Best validation F1: {selected['best_val_f1']:.4f}",
              f"- Test F1 of that preselected configuration: {chosen['test_f1']:.4f}",
              f"- Model weights: `{selected['best_model_path']}`", "",
              "## Outputs", "", f"- Model folder: `{model_dir}`", "- lightweight_temporal_fusion_summary_results.csv",
              "- validation_results.csv", "- selected_configuration.json", "- run_metadata.json",
              "- Each experiment folder contains history, metadata, best validation predictions/metrics and test predictions/metrics.",
              "", "## Scope", "",
              "- Current Step 3/4 metadata was checked; all configurations use the same site memberships.",
              "- Each full epoch/evaluation checks sample coverage, site_uid, label, K, tensor shapes and finite values.",
              "- This is one fixed-split suite, not cross-validation. Validation has 18 sites.",
              "- Selecting among many configurations with 18 validation sites can overfit that validation set.",
              "- Random frame selection uses the saved Step 4 indices; changing the training seed does not resample them.",
              "- GRU uses the supplied chronological frame order. Attention pooling does not encode frame order.",
              "- Neither fusion method receives calendar dates or time gaps. Few frames may span several years.",
              "- This suite evaluates fixed K=3/5/8 subsets, not full sequences or mixed sequence lengths.",
              "- Attention weights are not proof of which archaeological changes caused a prediction.",
              "- Parameter counts are recorded per experiment; runtime/memory efficiency was not benchmarked.",
              "- Batch size is the number of sites. The CNN processes batch_size * K images together.",
              "- Training metrics use augmented images and changing weights within each epoch.",
              "- Weighted loss is accumulated using target class-weight sums.",
              "- No site masking or image registration is performed. Source identities/labels require their separate audit.",
              "- Seeds do not guarantee identical results across environments; one run does not establish statistical significance."]
    (run_dir / "step9_lightweight_temporal_fusion_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def release_memory(torch, device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def execute_suite(torch, args, device, tables, inputs, counts, config, run_dir, model_dir):
    planned = [(k, strategy, fusion) for k in args.k_values
               for strategy in args.strategies for fusion in args.fusion_types]
    state = {"status": "RUNNING", "phase": "training", "started_utc": datetime.now(timezone.utc).isoformat(),
             "config": config, "site_counts": counts, "inputs": inputs,
             "planned_experiments": [{"k": k, "strategy": s, "fusion_type": p} for k, s, p in planned]}
    metadata_path = run_dir / "run_metadata.json"
    json_write(metadata_path, state)
    trained, results = [], []
    try:
        # Phase 1: no test loader or test predictions are used here.
        for number, (k, strategy, fusion) in enumerate(planned, start=1):
            print(f"\nTraining experiment {number}/{len(planned)}: K{k}/{strategy}/{fusion}", flush=True)
            trained.append(train_experiment(torch, args, device, tables[(k, strategy)], inputs, config,
                                            run_dir, model_dir, k, strategy, fusion))
            columns = ("experiment_name", "k", "strategy", "fusion_type", "seed", "best_val_f1", "best_epoch",
                       "epochs_completed", "best_model_path")
            pd.DataFrame([{key: r[key] for key in columns} for r in trained]).to_csv(
                run_dir / "validation_results.csv", index=False)
            state["training_completed"] = [r["experiment_name"] for r in trained]
            json_write(metadata_path, state)
            release_memory(torch, device)
        check_experiment_coverage(trained, planned)
        check_inputs_unchanged(inputs)
        selected = select_configuration(trained)
        selection = {"selection_rule": SELECTION_RULE, "selected_before_test_evaluation": True,
                     "selected_utc": datetime.now(timezone.utc).isoformat(),
                     "experiment_name": selected["experiment_name"], "k": selected["k"],
                     "strategy": selected["strategy"],
                     "fusion_type": selected["fusion_type"], "best_val_f1": selected["best_val_f1"],
                     "best_epoch": selected["best_epoch"], "best_model_path": selected["best_model_path"],
                     "checkpoint_sha256": selected["checkpoint_sha256"]}
        json_write(run_dir / "selected_configuration.json", selection)
        state.update({"phase": "testing", "selected_configuration": selection})
        json_write(metadata_path, state)
        print(f"\nValidation selected: {selected['experiment_name']}. Now testing the predefined suite.", flush=True)
        # Phase 2: test all predefined baselines; never reselect using test metrics.
        for number, row in enumerate(trained, start=1):
            print(f"\nTesting experiment {number}/{len(planned)}: {row['experiment_name']}", flush=True)
            metrics = test_experiment(torch, args, device, tables[(row["k"], row["strategy"])], inputs, config, row)
            results.append(summary_row(row, metrics, selected["experiment_name"]))
            pd.DataFrame(results).to_csv(run_dir / "lightweight_temporal_fusion_summary_results.csv", index=False)
            state["test_completed"] = [r["experiment_name"] for r in results]
            json_write(metadata_path, state)
            release_memory(torch, device)
        check_experiment_coverage(results, planned)
        check_inputs_unchanged(inputs)
        write_report(run_dir, model_dir, config, counts, results, selected)
        state.update({"status": "COMPLETE", "phase": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
                      "completed_experiments": len(results)})
        json_write(metadata_path, state)
        print(f"\nStep 9 COMPLETE: {len(results)}/{len(planned)} experiments; 61 test sites each.", flush=True)
        print(f"Report: {run_dir / 'step9_lightweight_temporal_fusion_report.md'}", flush=True)
        print(f"Validation-selected weights: {selected['best_model_path']}", flush=True)
        return results
    except BaseException as exc:
        message = f"{type(exc).__name__}: {exc}"
        state.update({"status": "FAILED", "error": message})
        json_write(metadata_path, state)
        (run_dir / "step9_lightweight_temporal_fusion_report.md").write_text(
            "# Step 9 Lightweight Temporal Fusion Baseline Report\n\n## Status: FAILED\n\n"
            f"- Error: {message}\n- Training completed: {len(trained)}/{len(planned)}\n"
            f"- Testing completed: {len(results)}/{len(planned)}\n"
            "- Any existing summary CSV is partial. This suite did not complete.\n", encoding="utf-8")
        raise


def main(args):
    root = find_project_root(args.project_root)
    data_root = args.data_root.expanduser().resolve() if args.data_root else root / "data/raw/DAFA_LS"
    require(args.batch_size > 0 and args.image_size >= 32 and args.epochs > 0 and args.patience > 0
            and args.num_workers >= 0 and 0 <= args.seed < 2 ** 32
            and math.isfinite(args.learning_rate) and args.learning_rate > 0
            and math.isfinite(args.weight_decay) and args.weight_decay >= 0
            and args.gru_hidden_dim > 0 and args.attention_dim > 0
            and math.isfinite(args.dropout) and 0 <= args.dropout < 1, "Invalid training settings.")
    require(bool(args.k_values) and len(args.k_values) == len(set(args.k_values))
            and set(args.k_values).issubset(K_VALUES), "Invalid or repeated K values.")
    require(bool(args.fusion_types) and len(args.fusion_types) == len(set(args.fusion_types))
            and set(args.fusion_types).issubset(FUSION_TYPES), "Invalid or repeated fusion types.")
    require(bool(args.strategies) and len(args.strategies) == len(set(args.strategies))
            and set(args.strategies).issubset(STRATEGIES), "Invalid or repeated frame-selection strategies.")
    args.k_values = sorted(args.k_values)
    args.fusion_types = sorted(args.fusion_types, key=FUSION_TYPES.index)
    args.strategies = sorted(args.strategies, key=STRATEGIES.index)
    print(f"Project root: {root}", flush=True)
    tables, counts, inputs = preflight(root, data_root, args.strategies, args.k_values)
    print("Preflight PASS: train=596, val=18, test=61 for every requested K/strategy.", flush=True)
    total = len(args.k_values) * len(args.strategies) * len(args.fusion_types)
    print(f"Planned experiments: {total} = {len(args.k_values)} K values x "
          f"{len(args.strategies)} strategies x {len(args.fusion_types)} fusion types.", flush=True)
    if args.preflight_only:
        print("Metadata and selected file existence checked. No images decoded or model training performed.")
        return 0
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    dl.require_torch()
    torch = dl.torch
    import torchvision
    device = choose_device(torch, args.device)
    first_table = tables[(args.k_values[0], args.strategies[0])]
    weights = class_weights_from_labels(first_table.query("split == 'train'")["label_id"])
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"project_root": str(root), "data_root": str(data_root), "actual_device": str(device),
                   "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "torch_version": str(torch.__version__), "torchvision_version": str(torchvision.__version__),
                   "cuda_runtime": torch.version.cuda, "class_weights": weights.tolist(),
                   "label_mapping": LABELS, "normalization": "imagenet", "selection_rule": SELECTION_RULE})
    strategy_tag = ("all_strategies" if len(args.strategies) == len(STRATEGIES)
                    else args.strategies[0] if len(args.strategies) == 1 else f"{len(args.strategies)}_strategies")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + f"_{strategy_tag}_seed{args.seed}"
    run_dir = root / "results/lightweight_temporal_fusion" / run_id
    model_dir = root / "models/lightweight_temporal_fusion" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)
    print(f"Device: {device}" + (f" ({config['gpu_name']})" if config["gpu_name"] else ""), flush=True)
    print(f"Reports: {run_dir}\nModels: {model_dir}", flush=True)
    execute_suite(torch, args, device, tables, inputs, counts, config, run_dir, model_dir)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=None, help="Auto-detected for scripts in src or project root.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--strategies", "--strategy", dest="strategies", nargs="+",
                        choices=STRATEGIES, default=list(STRATEGIES),
                        help="Frame-selection strategies to run; default is all four. --strategy is an alias.")
    parser.add_argument("--k-values", type=int, nargs="+", choices=K_VALUES, default=list(K_VALUES))
    parser.add_argument("--fusion-types", nargs="+", choices=FUSION_TYPES, default=list(FUSION_TYPES))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=2, help="Number of sites per batch; CNN processes batch_size*K frames.")
    parser.add_argument("--image-size", type=int, default=266)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--gru-hidden-dim", type=int, default=256)
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze ResNet18 parameters and BatchNorm statistics; default fine-tunes the backbone.")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", help="Require deterministic operations; unsupported operations fail.")
    parser.add_argument("--preflight-only", action="store_true", help="Check metadata and image paths without training.")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(main(parse_args()))
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Step 9 FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)

