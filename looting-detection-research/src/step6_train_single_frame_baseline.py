"""Step 6: train a checked, single-frame ResNet18 baseline.

Place this file beside the CORRECTED dataset_loader.py in the project root.
Run: python step6_train_single_frame_baseline.py --device cuda
Use --device auto to allow CPU fallback, or --preflight-only to check metadata.

Dependencies: your existing PyTorch/torchvision environment, pandas, NumPy,
Pillow, scikit-learn and tqdm. This script is compatible with PyTorch 2.7.
Each run gets its own results folder; previous reports are not overwritten.
This implements one fixed split, not cross-validation. Only images enter the
network. Class-bearing identifiers and paths are used only for checks/exports.
"""
from __future__ import annotations

import argparse
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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_loader_version():
    required = ("read_samples_csv", "relative_path", "integer_column", "require_torch")
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


def validate_metadata(sites, frames, samples):
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
    require(samples["k"].eq(1).all(), "The K1 CSV contains a non-K1 sample.")
    require(set(samples["strategy"]) == set(STRATEGIES), "The K1 CSV is missing a required strategy.")
    for strategy in STRATEGIES:
        for split in SPLITS:
            actual = samples[(samples["strategy"] == strategy) & (samples["split"] == split)]
            expected = set(sites.loc[sites["split"] == split, "site_uid"])
            require(actual["site_uid"].is_unique and set(actual["site_uid"]) == expected,
                    f"K1/{strategy}/{split} differs from current Step 3. Rerun corrected Step 4.")
    frame_lookup = {
        (r.site_uid, r.frame_index): (r.relative_path, r.year, r.month)
        for r in frames.itertuples(index=False)
    }
    for row in samples.itertuples(index=False):
        site = lookup.loc[row.site_uid]
        require(row.split == site["split"] and row.label_id == site["label_id"]
                and row.available_frames == site["frame_count"],
                f"Stale Step 4 metadata: {row.sample_id}.")
        require(frame_lookup.get((row.site_uid, row.frame_indices[0])) ==
                (row.relative_paths[0], row.years[0], row.months[0]),
                f"Selected frame/date disagrees with Step 3: {row.sample_id}.")
        if row.strategy in ("recent_window", "uniform", "early_late"):
            require(row.frame_indices == [int(site["frame_count"]) - 1],
                    f"K1/{row.strategy} must select the latest frame: {row.sample_id}.")
    return sites, counts


def preflight(root, data_root, strategy):
    check_loader_version()
    paths = {
        "site_splits": root / "data/splits/site_splits.csv",
        "frames_with_splits": root / "data/splits/frames_with_splits.csv",
        "k1_samples": root / "data/processed/kframe_sequences/kframe_samples_K1.csv",
        "dataset_loader": Path(dl.__file__).resolve(),
        "training_script": Path(__file__).resolve(),
    }
    for path in paths.values():
        require(path.is_file(), f"Required file is missing: {path}")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    sites = pd.read_csv(paths["site_splits"], dtype=str, keep_default_na=False)
    frames = pd.read_csv(paths["frames_with_splits"], dtype=str, keep_default_na=False)
    samples = dl.read_samples_csv(paths["k1_samples"])
    sites, counts = validate_metadata(sites, frames, samples)
    chosen = samples[samples["strategy"] == strategy].copy()
    require(data_root.is_dir(), f"Image directory is missing: {data_root}. Use --data-root if it moved.")
    for row in chosen.itertuples(index=False):
        path = data_root.joinpath(*PurePosixPath(row.relative_paths[0]).parts)
        require(path.is_file(), f"Selected image is missing: {path}")
    require(all(sha256_file(paths[name]) == digest for name, digest in hashes.items()),
            "An input file changed during preflight. Stop editing inputs and rerun.")
    inputs = {name: {"path": str(path), "sha256": hashes[name]} for name, path in paths.items()}
    return chosen, counts, inputs


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


def check_single_frame_shape(shape):
    require(len(shape) == 5 and shape[0] > 0 and shape[1] == 1 and shape[2] == 3,
            f"Expected [B,1,3,H,W], received {tuple(shape)}. K must equal 1.")


def build_model(torch):
    from torchvision import models

    class SingleFrameResNet18(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, 2)

        def forward(self, images):
            check_single_frame_shape(images.shape)
            return self.backbone(images[:, 0])

    return SingleFrameResNet18()


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


def validate_batch(torch, batch, expected, seen, split, strategy, image_size):
    require("images" in batch and "label" in batch, "Batch is missing images or labels.")
    images, labels = batch["images"], batch["label"]
    require(isinstance(images, torch.Tensor), "Images must be tensors.")
    check_single_frame_shape(images.shape)
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
        require(batch["split"][i] == split and batch["strategy"][i] == strategy and int(batch["k"][i]) == 1,
                "Batch split/strategy/K differs from this experiment.")
        seen.add(sample_id)


def run_epoch(torch, model, loader, criterion, device, expected, split, strategy,
              image_size, optimizer=None):
    training = optimizer is not None
    require(not training or split == "train", "Only the training split may update model weights.")
    model.train(training)
    meter, seen, records = WeightedLossMeter(), set(), []
    context = nullcontext() if training else torch.inference_mode()
    with context:
        for batch in tqdm(loader, desc="Training" if training else f"Evaluating {split}", leave=False):
            validate_batch(torch, batch, expected, seen, split, strategy, image_size)
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
                    "split": split, "strategy": strategy, "k": 1,
                    "frame_index": row["frame_indices"][0], "year": row["years"][0],
                    "month": row["months"][0], "relative_path": row["relative_paths"][0],
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


def write_report(run_dir, config, counts, summary, metrics):
    lines = ["# Step 6 Single-Frame CNN Baseline Report", "", "## Status: COMPLETE", "",
             f"- Run: {run_dir.name}", "- Model: ImageNet-pretrained ResNet18; all parameters fine-tuned",
             f"- Input: K=1; strategy={config['strategy']}", "- Labels: preserved=0; looted=1",
             f"- Image size: {config['image_size']} x {config['image_size']}",
             "- Normalization: ImageNet; whole images, no site masking",
             f"- Batch size: {config['batch_size']}", f"- Seed: {config['seed']}",
             f"- Epochs planned: {config['epochs']}", f"- Epochs completed: {summary['epochs_completed']}",
             f"- Best epoch: {summary['best_epoch']}", f"- Best validation F1: {summary['best_val_f1']:.4f}",
             f"- Stopping reason: {summary['stopping_reason']}",
             f"- Learning rate: {config['learning_rate']}", f"- Weight decay: {config['weight_decay']}",
             f"- Device used: {config['actual_device']}", f"- GPU name: {config['gpu_name'] or 'N/A'}",
             f"- PyTorch: {config['torch_version']}", f"- torchvision: {config['torchvision_version']}",
             f"- Class weights [preserved, looted]: {config['class_weights']}",
             "- Prediction rule: larger of the two logits (ties select preserved)",
             "- Checkpoint selection: validation F1; test evaluated once after selection",
             "- Equal F1 scores keep the earlier checkpoint and count toward patience.",
             "", "## Checked site counts", "", "| Split | Preserved | Looted | Total |",
             "|---|---:|---:|---:|"]
    for split in SPLITS:
        row = counts[split]
        lines.append(f"| {split} | {row['preserved']} | {row['looted']} | {sum(row.values())} |")
    lines += ["", "## Test results", ""]
    for name in ("accuracy", "precision", "recall", "f1", "roc_auc", "false_alarm_rate", "loss"):
        lines.append(f"- {name}: {format_score(metrics[name])}")
    for name in ("samples", "tn", "fp", "fn", "tp"):
        lines.append(f"- {name}: {metrics[name]}")
    lines += ["", "## Files in this run", "", "- best_model.pth",
              "- training_history.csv", "- test_predictions.csv", "- test_metrics.json",
              "- run_metadata.json", "", "## Scope", "",
              "- Current Step 3/4 metadata checked before training; no site_uid split overlap.",
              "- This is one fixed-split experiment, not cross-validation.",
              "- Validation has 18 sites; results may vary with seed and validation partition.",
              "- Training metrics use augmented images and changing model weights within each epoch.",
              "- Loss combines weighted batch losses using the sum of target class weights.",
              "- Site identities and labels are trusted from the supplied source metadata.",
              "- Seed/determinism settings do not guarantee identical results across platforms."]
    (run_dir / "step6_single_frame_baseline_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args):
    root = args.project_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve() if args.data_root else root / "data/raw/DAFA_LS"
    require(args.batch_size > 0 and args.image_size >= 32 and args.epochs > 0 and args.patience > 0
            and args.num_workers >= 0 and 0 <= args.seed < 2 ** 32
            and math.isfinite(args.learning_rate) and args.learning_rate > 0
            and math.isfinite(args.weight_decay) and args.weight_decay >= 0, "Invalid training settings.")
    samples, counts, inputs = preflight(root, data_root, args.strategy)
    print("Preflight PASS: train=596, val=18, test=61; identities and selected frames match Step 3.", flush=True)
    if args.preflight_only:
        print("Metadata and selected file existence checked. Images were not decoded; no training was performed.")
        return 0
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    dl.require_torch()
    torch = dl.torch
    import torchvision
    device = choose_device(torch, args.device)
    set_seed(torch, args.seed, args.deterministic)
    weights = class_weights_from_labels(samples.loc[samples["split"] == "train", "label_id"])
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"project_root": str(root), "data_root": str(data_root), "actual_device": str(device),
                   "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "torch_version": str(torch.__version__), "torchvision_version": str(torchvision.__version__),
                   "cuda_runtime": torch.version.cuda, "class_weights": weights.tolist(), "k": 1,
                   "label_mapping": LABELS, "normalization": "imagenet"})
    print(f"Device: {device}" + (f" ({config['gpu_name']})" if config["gpu_name"] else ""), flush=True)
    print(f"Class weights [preserved, looted]: {weights.tolist()}", flush=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + f"_{args.strategy}_seed{args.seed}"
    run_dir = root / "results/single_frame_baseline" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Results folder: {run_dir}", flush=True)
    metadata_path = run_dir / "run_metadata.json"
    state = {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
             "config": config, "site_counts": counts, "inputs": inputs}
    json_write(metadata_path, state)
    try:
        loaders, expected = {}, {}
        for split in SPLITS:
            expected[split] = samples.loc[samples["split"] == split].set_index("sample_id").to_dict("index")
            loaders[split] = dl.create_dataloader(
                csv_path=inputs["k1_samples"]["path"], split=split, strategy=args.strategy,
                batch_size=args.batch_size, image_size=args.image_size, train=split == "train",
                shuffle=split == "train", num_workers=args.num_workers, normalization="imagenet",
                data_root=data_root, seed=args.seed, k=1, pin_memory=device.type == "cuda",
            )
            actual = loaders[split].dataset.df.set_index("sample_id").to_dict("index")
            require(actual == expected[split], "Loader metadata changed after preflight.")
        model = build_model(torch).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        history, best_val_f1, best_epoch, patience_counter = [], -1.0, 0, 0
        best_path = run_dir / "best_model.pth"
        stopping_reason = "maximum epochs reached"
        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
            train_metrics, _ = run_epoch(torch, model, loaders["train"], criterion, device, expected["train"],
                                         "train", args.strategy, args.image_size, optimizer)
            val_metrics, _ = run_epoch(torch, model, loaders["val"], criterion, device, expected["val"],
                                       "val", args.strategy, args.image_size)
            history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
                            **{f"val_{k}": v for k, v in val_metrics.items()}})
            pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
            print(f"Train loss={train_metrics['loss']:.4f}, F1={train_metrics['f1']:.4f} | "
                  f"Val loss={val_metrics['loss']:.4f}, F1={val_metrics['f1']:.4f}, "
                  f"AUC={format_score(val_metrics['roc_auc'])}", flush=True)
            if val_metrics["f1"] > best_val_f1:
                best_val_f1, best_epoch, patience_counter = val_metrics["f1"], epoch, 0
                torch.save(model.state_dict(), best_path)
                print(f"Saved best model from epoch {epoch}.", flush=True)
            else:
                patience_counter += 1
            state["progress"] = {"epochs_completed": epoch, "best_epoch": best_epoch, "best_val_f1": best_val_f1}
            json_write(metadata_path, state)
            if patience_counter >= args.patience:
                stopping_reason = "early stopping: validation F1 did not improve"
                break
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        test_metrics, predictions = run_epoch(torch, model, loaders["test"], criterion, device, expected["test"],
                                               "test", args.strategy, args.image_size)
        require(test_metrics["samples"] == 61 and test_metrics["tn"] + test_metrics["fp"] == 35
                and test_metrics["fn"] + test_metrics["tp"] == 26, "Unexpected final test counts.")
        for name, info in inputs.items():
            require(sha256_file(info["path"]) == info["sha256"], f"Input changed during training: {name}.")
        predictions.to_csv(run_dir / "test_predictions.csv", index=False)
        json_write(run_dir / "test_metrics.json", test_metrics)
        summary = {"epochs_completed": len(history), "best_epoch": best_epoch,
                   "best_val_f1": best_val_f1, "stopping_reason": stopping_reason}
        write_report(run_dir, config, counts, summary, test_metrics)
        state.update({"status": "COMPLETE", "completed_utc": datetime.now(timezone.utc).isoformat(),
                      "summary": summary, "test_metrics": test_metrics})
        json_write(metadata_path, state)
        print(f"\nStep 6 COMPLETE. Best epoch: {best_epoch}; test sites: 61.", flush=True)
        print(f"Report: {run_dir / 'step6_single_frame_baseline_report.md'}", flush=True)
        return 0
    except BaseException as exc:
        state.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
        json_write(metadata_path, state)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--strategy", choices=STRATEGIES, default="recent_window")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=266)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true",
                        help="Require deterministic PyTorch operations; unsupported operations raise an error.")
    parser.add_argument("--preflight-only", action="store_true", help="Check current metadata and image paths, then exit.")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(main(parse_args()))
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Step 6 FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
