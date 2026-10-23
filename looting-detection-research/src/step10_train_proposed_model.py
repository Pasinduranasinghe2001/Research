"""Step 10: checked change-aware ResNet18 + GRU experiments.

Save beside the corrected Step 5 dataset_loader.py in your project's src folder.
Run from the project root:
    python src/step10_train_proposed_model.py --device cuda
Metadata / image-path check only:
    python src/step10_train_proposed_model.py --preflight-only
Smaller predefined suite (12 experiments):
    python src/step10_train_proposed_model.py --device cuda --strategies uniform

Default: K=3,5,8 x four strategies x four ablations = 48 training experiments.
Variants: no_change_no_drop, change_no_drop, no_change_frame_drop, proposed.
All variants use identical parameter shapes (the no-change input half is zero).
The proposed configuration is selected only by validation F1 BEFORE testing.
Every predefined experiment is subsequently tested once with all K frames.

Frame dropping selects images BEFORE ResNet18, so discarded images cannot
influence BatchNorm. The loader still reads all K source files; this simulates
missing observations and does not repair missing files on disk. The first
selected frame and at least one comparison frame always remain. Drop probability
applies to non-reference frames; actual retained counts are exported.

Robustness: selected proposed model versus its matching change_no_drop model,
using identical per-site masks, default five repeats at p=.25 and .50. The normal
p=0 evaluation is reused. These repeats measure mask variability, not independent
training runs or confidence intervals. No 100-frame or mixed raw-length study.

Baseline comparison: only compatible COMPLETE timestamped runs from corrected
Steps 6-9 are considered. The latest compatible run is chosen by run timestamp,
never test score. --step6-run through --step9-run can explicitly name run folders.
Unavailable/invalid baselines are listed as excluded; no old root-level score
files are used. Step 8/9 choices must have been saved using validation before test.

Reports: results/proposed_change_aware_fusion/<unique_run>/
Weights: models/proposed_change_aware_fusion/<unique_run>/
Each model saves a state_dict and its configuration metadata. No old run is reused.
Requires your existing torch/torchvision, pandas, NumPy, Pillow, scikit-learn.
The first training run may download ImageNet weights; tqdm is optional.
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
import dataset_loader as dl

EXPECTED_COUNTS = {
    "train": {"preserved": 496, "looted": 100},
    "val": {"preserved": 9, "looted": 9},
    "test": {"preserved": 35, "looted": 26},
}
SPLITS = ("train", "val", "test")
STRATEGIES = ("recent_window", "uniform", "early_late", "random")
LABELS = {"preserved": 0, "looted": 1}
K_VALUES = (3, 5, 8)
VARIANTS = ("no_change_no_drop", "change_no_drop", "no_change_frame_drop", "proposed")
VARIANT_FLAGS = {"no_change_no_drop": (False, False), "change_no_drop": (True, False),
                 "no_change_frame_drop": (False, True), "proposed": (True, True)}
SELECTION_RULE = ("among proposed variants only: highest best validation F1; ties: smaller K, "
                  "then strategy order recent_window, uniform, early_late, random")
REPORT_NAME = "step10_proposed_model_report.md"
SUMMARY_NAME = "step10_summary_results.csv"
BASELINE_DIRS = {6: "single_frame_baseline", 7: "two_frame_change_baseline",
                 8: "temporal_pooling_baseline", 9: "lightweight_temporal_fusion"}

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


def configure_backbone(backbone, frozen, training):
    """Freezing parameters alone would still allow BatchNorm statistics to change."""
    for parameter in backbone.parameters():
        parameter.requires_grad_(not frozen)
    backbone.train(bool(training) and not frozen)


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


def release_memory(torch, device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def keep_mask_from_uniform(uniform, probability):
    """Reference + at least one target; argmax fallback also makes masks nested."""
    values = np.asarray(uniform, dtype=float)
    require(math.isfinite(probability) and 0 <= probability <= 1, "Drop probability must be in [0,1].")
    require(values.ndim == 2 and values.shape[0] > 0 and values.shape[1] >= 2
            and np.isfinite(values).all() and ((0 <= values) & (values < 1)).all(),
            "Mask draws must be finite [B,K] values in [0,1), K >= 2.")
    keep = values >= probability
    keep[:, 0] = True
    empty = np.flatnonzero(~keep[:, 1:].any(axis=1))
    keep[empty, 1 + values[empty, 1:].argmax(axis=1)] = True
    return keep


def stable_keep_masks(site_uids, k, strategy, probability, seed, repeat):
    """Reproducible per-site masks, independent of batch order/size and model."""
    require(len(site_uids) > 0 and len(set(site_uids)) == len(site_uids), "Repeated/empty mask sites.")
    draws = []
    for uid in site_uids:
        key = json.dumps([int(seed), int(repeat), str(uid), int(k), strategy], separators=(",", ":"))
        local_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
        draws.append(np.random.default_rng(local_seed).random(k))
    return keep_mask_from_uniform(draws, probability)


def combine_features(torch, sequence, use_change):
    """Identical 1024-wide input shape in every ablation; no-change half is zero."""
    change = torch.abs(sequence - sequence[:1]) if use_change else torch.zeros_like(sequence)
    return torch.cat((sequence, change), dim=-1)


def extract_kept_features(torch, images, keep_mask, extractor):
    # Boolean indexing is BEFORE the CNN, so dropped images cannot affect BatchNorm.
    retained = images[keep_mask]
    features = extractor(retained)
    lengths = keep_mask.sum(dim=1).detach().cpu().tolist()
    return list(features.split(lengths, dim=0))


def build_model(torch, k, use_change=True, hidden_dim=256, dropout=0.3,
                projection_dropout=0.2, freeze_backbone=False, pretrained=True):
    from torchvision import models
    require(k in K_VALUES and hidden_dim > 0, "Invalid K or hidden dimension.")
    for value in (dropout, projection_dropout):
        require(math.isfinite(value) and 0 <= value < 1, "Invalid dropout probability.")

    class ChangeAwareTemporalFusionResNet18(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.k = k
            self.use_change = bool(use_change)
            self.freeze_backbone = bool(freeze_backbone)
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.feature_extractor = models.resnet18(weights=weights)
            self.feature_dim = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = torch.nn.Identity()
            self.feature_projection = torch.nn.Sequential(
                torch.nn.LayerNorm(2 * self.feature_dim),
                torch.nn.Linear(2 * self.feature_dim, hidden_dim), torch.nn.ReLU(),
                torch.nn.Dropout(projection_dropout))
            self.temporal_gru = torch.nn.GRU(hidden_dim, hidden_dim, num_layers=1,
                                            batch_first=True, bidirectional=False)
            self.classifier = torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.Linear(hidden_dim, 2))
            configure_backbone(self.feature_extractor, self.freeze_backbone, self.training)

        def train(self, mode=True):
            super().train(mode)
            configure_backbone(self.feature_extractor, self.freeze_backbone, mode)
            return self

        def forward(self, images, keep_mask=None):
            check_temporal_shape(images.shape, self.k)
            b, k_actual = images.shape[:2]
            if keep_mask is None:
                keep_mask = torch.ones((b, k_actual), dtype=torch.bool, device=images.device)
            require(keep_mask.dtype == torch.bool and tuple(keep_mask.shape) == (b, k_actual)
                    and keep_mask.device == images.device, "Invalid frame keep-mask shape/dtype/device.")
            require(keep_mask[:, 0].all().item() and (keep_mask.sum(dim=1) >= 2).all().item(),
                    "A reference and at least one target frame must be kept.")
            context = torch.no_grad() if self.freeze_backbone else nullcontext()
            with context:
                sequences = extract_kept_features(torch, images, keep_mask, self.feature_extractor)
            lengths = [sequence.shape[0] for sequence in sequences]
            combined = torch.cat([combine_features(torch, s, self.use_change) for s in sequences], dim=0)
            projected = self.feature_projection(combined)
            # split/pad/pack preserve autograd and the original chronological order.
            padded = torch.nn.utils.rnn.pad_sequence(list(projected.split(lengths, dim=0)), batch_first=True)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                padded, lengths, batch_first=True, enforce_sorted=False)
            _, hidden = self.temporal_gru(packed)
            return self.classifier(hidden[-1])

    return ChangeAwareTemporalFusionResNet18()


def parameter_counts(model):
    groups = {"backbone": model.feature_extractor, "projection": model.feature_projection,
              "gru": model.temporal_gru, "classifier": model.classifier}
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            **{name: sum(p.numel() for p in module.parameters()) for name, module in groups.items()}}


def model_for(torch, args, k, variant, pretrained):
    require(variant in VARIANTS, "Unknown ablation variant.")
    return build_model(torch, k, use_change=VARIANT_FLAGS[variant][0], hidden_dim=args.hidden_dim,
                       dropout=args.dropout, projection_dropout=args.projection_dropout,
                       freeze_backbone=args.freeze_backbone, pretrained=pretrained)


def run_epoch(torch, model, loader, criterion, device, expected, split, strategy, image_size, k,
              optimizer=None, drop_probability=0.0, mask_rng=None, mask_seed=42, mask_repeat=0,
              grad_clip_norm=5.0):
    training = optimizer is not None
    require(not training or split == "train", "Only training data may update weights.")
    require(math.isfinite(drop_probability) and 0 <= drop_probability <= 1, "Invalid frame-drop probability.")
    require(not training or mask_rng is not None, "Training needs its independent mask RNG.")
    model.train(training)
    meter, seen, records = WeightedLossMeter(), set(), []
    context = nullcontext() if training else torch.inference_mode()
    with context:
        for batch in tqdm(loader, desc="Training" if training else f"Evaluating {split}", leave=False):
            validate_batch(torch, batch, expected, seen, split, strategy, image_size, k)
            b = batch["images"].shape[0]
            if drop_probability == 0:
                keep = np.ones((b, k), dtype=bool)
            elif training:
                keep = keep_mask_from_uniform(mask_rng.random((b, k)), drop_probability)
            else:
                keep = stable_keep_masks(batch["site_uid"], k, strategy, drop_probability, mask_seed, mask_repeat)
            images = batch["images"].to(device, non_blocking=device.type == "cuda")
            labels = batch["label"].to(device, non_blocking=device.type == "cuda")
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images, keep_mask=torch.as_tensor(keep, dtype=torch.bool, device=device))
            require(tuple(logits.shape) == (b, 2) and torch.isfinite(logits).all().item(), "Invalid model logits.")
            loss = criterion(logits, labels)
            meter.add(float(loss.detach().item()), float(criterion.weight[labels].sum().item()))
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, error_if_nonfinite=True)
                optimizer.step()
            probabilities = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().tolist()
            predicted = logits.detach().argmax(dim=1).cpu().tolist()
            truths = labels.detach().cpu().tolist()
            for i, sample_id in enumerate(batch["sample_id"]):
                row = expected[sample_id]
                positions = np.flatnonzero(keep[i]).tolist()
                records.append({
                    "site_uid": row["site_uid"], "site_id": row["site_id"], "sample_id": sample_id,
                    "split": split, "strategy": strategy, "k": k,
                    "frame_indices": json.dumps(row["frame_indices"]),
                    "years": json.dumps(row["years"]), "months": json.dumps(row["months"]),
                    "relative_paths": json.dumps(row["relative_paths"]),
                    "retained_positions": json.dumps(positions), "retained_count": len(positions),
                    "retained_frame_indices": json.dumps([row["frame_indices"][j] for j in positions]),
                    "drop_probability": drop_probability, "actual_dropped_fraction": 1 - len(positions) / k,
                    "true_label": int(truths[i]), "predicted_label": int(predicted[i]),
                    "looted_probability": float(probabilities[i])})
    require(seen == set(expected) and len(records) == len(expected), "Incomplete/duplicate sample coverage.")
    predictions = pd.DataFrame(records)
    metrics = calculate_metrics(predictions.true_label, predictions.predicted_label, predictions.looted_probability)
    metrics.update({"loss": meter.mean(), "mean_retained_frames": float(predictions.retained_count.mean()),
                    "actual_dropped_fraction": float(predictions.actual_dropped_fraction.mean())})
    return metrics, predictions


def experiment_name(k, strategy, variant):
    return f"K{k}_{strategy}_{variant}"


def train_experiment(torch, args, device, samples, inputs, config, run_dir, model_dir, k, strategy, variant):
    set_seed(torch, args.seed, args.deterministic)
    mask_rng = np.random.default_rng(args.seed)
    name = experiment_name(k, strategy, variant)
    folder = run_dir / name
    folder.mkdir(exist_ok=False)
    checkpoint = model_dir / f"{name}_best.pth"
    use_change, use_drop = VARIANT_FLAGS[variant]
    probability = args.frame_drop_probability if use_drop else 0.0
    state = {"status": "TRAINING", "experiment_name": name, "k": k, "strategy": strategy,
             "variant": variant, "use_change_features": use_change, "frame_drop_probability": probability,
             "seed": args.seed, "config": config, "best_model_path": str(checkpoint)}
    state_path = folder / "experiment_metadata.json"
    json_write(state_path, state)
    try:
        loaders, expected = {}, {}
        for split in ("train", "val"):
            loaders[split], expected[split] = make_loader(args, device, samples, inputs[f"k{k}_samples"]["path"],
                                                         config["data_root"], k, strategy, split)
        weights = class_weights_from_labels(samples.loc[samples.split == "train", "label_id"])
        model = model_for(torch, args, k, variant, pretrained=True).to(device)
        parameters = parameter_counts(model)
        state["parameter_counts"] = parameters
        json_write(state_path, state)
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=args.learning_rate, weight_decay=args.weight_decay)
        history, best_score, best_epoch, patience = [], -1.0, 0, 0
        best_metrics = None
        reason = "maximum epochs reached"
        for epoch in range(1, args.epochs + 1):
            train_metrics, _ = run_epoch(torch, model, loaders["train"], criterion, device, expected["train"],
                                         "train", strategy, args.image_size, k, optimizer=optimizer,
                                         drop_probability=probability, mask_rng=mask_rng, grad_clip_norm=args.grad_clip_norm)
            val_metrics, val_predictions = run_epoch(torch, model, loaders["val"], criterion, device, expected["val"],
                                                     "val", strategy, args.image_size, k)
            history.append({"epoch": epoch, **{f"train_{n}": v for n, v in train_metrics.items()},
                            **{f"val_{n}": v for n, v in val_metrics.items()}})
            pd.DataFrame(history).to_csv(folder / "training_history.csv", index=False)
            print(f"{name}: epoch {epoch}/{args.epochs}; loss={train_metrics['loss']:.4f}; "
                  f"val F1={val_metrics['f1']:.4f}", flush=True)
            if val_metrics["f1"] > best_score:
                best_score, best_epoch, patience, best_metrics = val_metrics["f1"], epoch, 0, val_metrics.copy()
                temporary = checkpoint.with_suffix(".tmp")
                torch.save(model.state_dict(), temporary)
                temporary.replace(checkpoint)
                val_predictions.to_csv(folder / "best_validation_predictions.csv", index=False)
                json_write(folder / "best_validation_metrics.json", {"epoch": epoch, **val_metrics})
            else:
                patience += 1
            state["progress"] = {"epochs_completed": epoch, "best_epoch": best_epoch, "best_val_f1": best_score}
            json_write(state_path, state)
            if patience >= args.patience:
                reason = "early stopping: validation F1 did not improve"
                break
        require(checkpoint.is_file() and best_metrics is not None, "No valid checkpoint was saved.")
        result = {"status": "TRAINED", "experiment_name": name, "k": k, "strategy": strategy, "variant": variant,
                  "use_change_features": use_change, "frame_drop_probability": probability, "seed": args.seed,
                  "epochs_completed": len(history), "best_epoch": best_epoch, "best_val_f1": best_score,
                  "best_validation_metrics": best_metrics, "stopping_reason": reason,
                  "class_weights": weights.tolist(), "best_model_path": str(checkpoint),
                  "parameter_counts": parameters, "checkpoint_sha256": sha256_file(checkpoint), "output_dir": str(folder)}
        state.update(result)
        json_write(state_path, state)
        json_write(checkpoint.with_suffix(".json"), {"architecture": "ChangeAwareTemporalFusionResNet18",
                   "config": config, "experiment": result})
        return result
    except BaseException as exc:
        state.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
        json_write(state_path, state)
        raise


def check_experiment_coverage(results, planned):
    observed = [(r["k"], r["strategy"], r["variant"]) for r in results]
    require(planned and len(planned) == len(set(planned)) and len(observed) == len(set(observed))
            and set(observed) == set(planned), "An incomplete or duplicated suite cannot be COMPLETE.")


def select_configuration(results):
    candidates = [r for r in results if r["variant"] == "proposed"]
    require(bool(candidates), "No proposed configurations to select.")
    for row in candidates:
        score = row.get("best_val_f1")
        require(isinstance(score, (float, int)) and math.isfinite(score) and 0 <= score <= 1,
                "Invalid validation F1.")
        require(row["k"] in K_VALUES and row["strategy"] in STRATEGIES, "Invalid selection candidate.")
    return min(candidates, key=lambda r: (-r["best_val_f1"], r["k"], STRATEGIES.index(r["strategy"]))).copy()


def load_trained_model(torch, args, device, trained):
    checkpoint = Path(trained["best_model_path"])
    require(sha256_file(checkpoint) == trained["checkpoint_sha256"], "Model checkpoint changed.")
    model = model_for(torch, args, trained["k"], trained["variant"], pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(trained["class_weights"], dtype=torch.float32, device=device))
    return model, criterion


def test_experiment(torch, args, device, samples, inputs, config, trained):
    k, strategy = trained["k"], trained["strategy"]
    folder = Path(trained["output_dir"])
    loader, expected = make_loader(args, device, samples, inputs[f"k{k}_samples"]["path"],
                                   config["data_root"], k, strategy, "test")
    model, criterion = load_trained_model(torch, args, device, trained)
    metrics, predictions = run_epoch(torch, model, loader, criterion, device, expected, "test", strategy, args.image_size, k)
    require(metrics["samples"] == 61 and metrics["tn"] + metrics["fp"] == 35
            and metrics["fn"] + metrics["tp"] == 26, "Unexpected test counts.")
    predictions.to_csv(folder / "test_predictions.csv", index=False)
    json_write(folder / "test_metrics.json", metrics)
    state_path = folder / "experiment_metadata.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update({"status": "COMPLETE", "test_metrics": metrics})
    json_write(state_path, state)
    print(f"Tested {trained['experiment_name']}: F1={metrics['f1']:.4f}", flush=True)
    return metrics


def summary_row(trained, metrics, selected_name):
    keys = ("experiment_name", "k", "strategy", "variant", "use_change_features", "frame_drop_probability", "seed",
            "epochs_completed", "best_epoch", "best_val_f1", "stopping_reason", "best_model_path", "output_dir")
    return {**{key: trained[key] for key in keys}, "status": "COMPLETE",
            "selected_by_validation": trained["experiment_name"] == selected_name,
            **{f"parameters_{key}": value for key, value in trained["parameter_counts"].items()},
            **{f"test_{key}": value for key, value in metrics.items()}}


def paired_robustness(torch, args, device, tables, inputs, config, run_dir, trained, selected):
    k, strategy = selected["k"], selected["strategy"]
    matched = [next(r for r in trained if (r["k"], r["strategy"], r["variant"]) == (k, strategy, variant))
               for variant in ("change_no_drop", "proposed")]
    output = run_dir / "robustness"
    output.mkdir(exist_ok=False)
    rows, paired_masks = [], {}
    for model_row in matched:
        name = model_row["experiment_name"]
        normal_folder = Path(model_row["output_dir"])
        normal = json.loads((normal_folder / "test_metrics.json").read_text(encoding="utf-8"))
        rows.append({"experiment_name": name, "variant": model_row["variant"], "k": k, "strategy": strategy,
                     "drop_probability": 0.0, "repeat": -1, "mask_seed": args.robustness_seed,
                     "predictions_path": str(normal_folder / "test_predictions.csv"), **normal})
        model, criterion = load_trained_model(torch, args, device, model_row)
        loader, expected = make_loader(args, device, tables[(k, strategy)], inputs[f"k{k}_samples"]["path"],
                                       config["data_root"], k, strategy, "test")
        for probability in args.robustness_probabilities:
            for repeat in range(args.robustness_repeats):
                metrics, predictions = run_epoch(torch, model, loader, criterion, device, expected, "test", strategy,
                                                 args.image_size, k, drop_probability=probability,
                                                 mask_seed=args.robustness_seed, mask_repeat=repeat)
                mask_table = predictions.set_index("site_uid")["retained_positions"].sort_index().to_dict()
                key = (probability, repeat)
                if key in paired_masks:
                    require(mask_table == paired_masks[key], "Paired robustness masks do not match.")
                else:
                    paired_masks[key] = mask_table
                path = output / f"{name}_p{probability:g}_repeat{repeat}.csv"
                predictions.to_csv(path, index=False)
                rows.append({"experiment_name": name, "variant": model_row["variant"], "k": k, "strategy": strategy,
                             "drop_probability": probability, "repeat": repeat, "mask_seed": args.robustness_seed,
                             "predictions_path": str(path), **metrics})
                pd.DataFrame(rows).to_csv(run_dir / "step10_frame_drop_robustness.csv", index=False)
        del model, criterion, loader
        release_memory(torch, device)
    results = pd.DataFrame(rows)
    aggregate = []
    for (name, variant, probability), group in results.groupby(["experiment_name", "variant", "drop_probability"], sort=False):
        entry = {"experiment_name": name, "variant": variant, "k": k, "strategy": strategy,
                 "drop_probability": float(probability), "repeats": len(group), "sites_per_repeat": 61}
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "false_alarm_rate", "loss",
                       "mean_retained_frames", "actual_dropped_fraction"):
            values = group[metric].to_numpy(dtype=float)
            require(np.isfinite(values).all(), "Non-finite robustness metric.")
            entry[f"{metric}_mean"] = float(values.mean())
            entry[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        aggregate.append(entry)
    pd.DataFrame(aggregate).to_csv(run_dir / "step10_robustness_summary.csv", index=False)
    return aggregate


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def validate_baseline_run(folder, step, root, inputs, sites, args):
    """Read only one concrete run; validate identities and recompute its scores."""
    folder = Path(folder).expanduser().resolve()
    metadata_path = folder / "run_metadata.json"
    state = read_json(metadata_path)
    require(state.get("status") == "COMPLETE", "Baseline run is not COMPLETE.")
    require(state.get("site_counts") == EXPECTED_COUNTS, "Baseline split counts differ.")
    cfg = state.get("config", {})
    require(cfg.get("image_size") == args.image_size and cfg.get("normalization") == "imagenet"
            and cfg.get("label_mapping") == LABELS, "Baseline preprocessing/label mapping differs.")
    files = [metadata_path]
    for name in ("site_splits", "frames_with_splits", "dataset_loader"):
        require(state.get("inputs", {}).get(name, {}).get("sha256") == inputs[name]["sha256"],
                f"Baseline source differs: {name}.")
    if step in (8, 9):
        selection_path = folder / "selected_configuration.json"
        selection = read_json(selection_path)
        require(selection.get("selected_before_test_evaluation") is True, "Baseline was not validation-selected before testing.")
        name = selection["experiment_name"]
        require(isinstance(name, str) and Path(name).name == name and "\\" not in name, "Invalid baseline experiment name.")
        predictions_folder = folder / name
        k, strategy = int(selection["k"]), selection["strategy"]
        best_val = float(selection["best_val_f1"])
        validation_path = folder / "validation_results.csv"
        validation = pd.read_csv(validation_path)
        scores = pd.to_numeric(validation["best_val_f1"], errors="raise")
        require(np.isfinite(scores).all() and scores.between(0, 1).all(), "Invalid baseline validation scores.")
        selected_row = validation[validation.experiment_name == name]
        require(len(selected_row) == 1 and math.isclose(best_val, float(scores.max()), abs_tol=1e-10)
                and math.isclose(best_val, float(selected_row.iloc[0].best_val_f1), abs_tol=1e-10),
                "Saved baseline choice does not match highest validation F1.")
        require(int(selected_row.iloc[0].k) == k and selected_row.iloc[0].strategy == strategy,
                "Baseline selected configuration differs from validation table.")
        method = selection.get("fusion_type", selection.get("pooling_type", "unknown"))
        files.extend([selection_path, validation_path])
    else:
        predictions_folder = folder
        k, strategy = (1 if step == 6 else 2), cfg.get("strategy")
        best_val = float(state.get("summary", {}).get("best_val_f1", float("nan")))
        name = "single_frame" if step == 6 else "two_frame_change"
        method = name
    require(math.isfinite(best_val) and 0 <= best_val <= 1 and strategy in STRATEGIES, "Invalid baseline selection metadata.")
    key = f"k{k}_samples"
    source = root / f"data/processed/kframe_sequences/kframe_samples_K{k}.csv"
    require(source.is_file() and state.get("inputs", {}).get(key, {}).get("sha256") == sha256_file(source),
            "Baseline selected K-frame file differs from current Step 4.")
    files.append(source)
    prediction_path = predictions_folder / "test_predictions.csv"
    metric_path = predictions_folder / "test_metrics.json"
    predictions = pd.read_csv(prediction_path, dtype={"site_uid": str}, keep_default_na=False)
    required = {"site_uid", "split", "true_label", "predicted_label", "looted_probability", "k", "strategy"}
    require(required.issubset(predictions.columns) and predictions.site_uid.is_unique, "Baseline lacks unique site identities.")
    expected = sites.loc[sites.split == "test"].set_index("site_uid")["label_id"].astype(int).to_dict()
    actual_labels = dict(zip(predictions.site_uid, predictions.true_label))
    require(actual_labels == expected and len(predictions) == 61 and predictions.split.eq("test").all(),
            "Baseline test sites/labels differ from this experiment.")
    require(predictions.k.eq(k).all() and predictions.strategy.eq(strategy).all(), "Baseline prediction K/strategy mismatch.")
    recomputed = calculate_metrics(predictions.true_label, predictions.predicted_label, predictions.looted_probability)
    saved = read_json(metric_path)
    for metric, value in recomputed.items():
        require(metric in saved and ((value is None and saved[metric] is None)
                or (value is not None and isinstance(saved[metric], (int, float))
                    and math.isclose(float(saved[metric]), value, rel_tol=1e-9, abs_tol=1e-9))),
                f"Baseline saved {metric} disagrees with predictions.")
    files.extend([prediction_path, metric_path])
    result = {"model": f"Step {step}", "run": folder.name, "experiment_name": name, "k": k,
              "strategy": strategy, "method": method, "best_val_f1": best_val,
              "seed": cfg.get("seed"), "batch_size": cfg.get("batch_size"),
              "learning_rate": cfg.get("learning_rate"), "weight_decay": cfg.get("weight_decay"),
              "epochs_planned": cfg.get("epochs"), "patience": cfg.get("patience"),
              "freeze_backbone": cfg.get("freeze_backbone", False), "device": cfg.get("actual_device"),
              "run_dir": str(folder), **recomputed}
    hashes = {str(path): sha256_file(path) for path in files}
    return result, hashes


def prepare_baselines(root, inputs, args):
    sites = pd.read_csv(inputs["site_splits"]["path"], dtype=str, keep_default_na=False)
    results, checks, hashes = [], [], {}
    for step, directory in BASELINE_DIRS.items():
        explicit = getattr(args, f"step{step}_run")
        parent = root / "results" / directory
        # The only automatic criterion is newest timestamped compatible COMPLETE run.
        if explicit is not None:
            candidates = [explicit]
        elif parent.is_dir():
            candidates = sorted((p for p in parent.iterdir() if p.is_dir() and p.name[:8].isdigit()),
                                key=lambda p: p.name, reverse=True)
        else:
            candidates = []
        rejected = []
        for candidate in candidates:
            try:
                row, source_hashes = validate_baseline_run(candidate, step, root, inputs, sites, args)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                rejected.append({"run": str(candidate), "reason": str(exc)})
                if explicit is not None:
                    raise ValueError(f"Explicit Step {step} baseline is invalid: {exc}") from exc
                continue
            results.append(row)
            hashes.update(source_hashes)
            checks.append({"step": step, "status": "included", "run": str(candidate), "rejected_runs": rejected})
            break
        else:
            checks.append({"step": step, "status": "excluded", "reason": "No compatible COMPLETE run found.",
                           "rejected_runs": rejected})
    return {"selection_rule": "latest compatible COMPLETE timestamped run; never test score",
            "rows": results, "checks": checks, "file_hashes": hashes,
            "complete": len(results) == len(BASELINE_DIRS)}


def check_baselines_unchanged(baselines):
    for path, digest in baselines["file_hashes"].items():
        require(sha256_file(path) == digest, f"Baseline input changed after it was selected: {path}")


def create_final_comparison(run_dir, config, results, selected, baselines):
    check_baselines_unchanged(baselines)
    chosen = next(r for r in results if r["experiment_name"] == selected["experiment_name"])
    row = {"model": "Step 10", "run": run_dir.name, "experiment_name": chosen["experiment_name"],
           "k": chosen["k"], "strategy": chosen["strategy"], "method": "change features + GRU + frame drop",
           "best_val_f1": chosen["best_val_f1"], "seed": config["seed"], "batch_size": config["batch_size"],
           "learning_rate": config["learning_rate"], "weight_decay": config["weight_decay"],
           "epochs_planned": config["epochs"], "patience": config["patience"],
           "freeze_backbone": config["freeze_backbone"], "device": config["actual_device"], "run_dir": str(run_dir)}
    for key in ("samples", "accuracy", "precision", "recall", "f1", "roc_auc", "false_alarm_rate", "tn", "fp", "fn", "tp"):
        row[key] = chosen[f"test_{key}"]
    rows = baselines["rows"] + [row]
    pd.DataFrame(rows).to_csv(run_dir / "final_model_comparison.csv", index=False)
    json_write(run_dir / "baseline_comparison_status.json", {
        "status": "COMPLETE" if baselines["complete"] else "INCOMPLETE",
        "checks": baselines["checks"],
        "scope": "Matched source metadata, loader, preprocessing and test identities. Training settings are recorded; "
                 "cross-step comparisons are descriptive, not controlled ablations or significance tests."})
    return rows


def write_report(run_dir, model_dir, config, counts, results, selected, robustness, comparison, baselines):
    chosen = next(r for r in results if r["selected_by_validation"])
    lines = ["# Step 10 Proposed Change-Aware Temporal Fusion Report", "", "## Status: COMPLETE", "",
        f"- Run: {run_dir.name}", f"- Completed training experiments: {len(results)}",
        "- Model: shared ImageNet-pretrained ResNet18 + appearance/reference-difference projection + one-layer GRU",
        f"- K values: {config['k_values']}", f"- Strategies: {config['strategies']}",
        f"- Variants: {list(VARIANTS)}", "- Labels: preserved=0, looted=1",
        "- All variants have identical parameter shapes; no-change variants replace the change half with zeros.",
        f"- GRU and projection width: {config['hidden_dim']}",
        f"- Backbone frozen: {config['freeze_backbone']} (frozen BatchNorm remains in evaluation mode)",
        f"- Image size: {config['image_size']} x {config['image_size']}; normalization: ImageNet",
        f"- Batch size in sites: {config['batch_size']}",
        f"- Training frame-drop probability for drop variants: {config['frame_drop_probability']}",
        "- Frame removal happens before the CNN; the loader still requires all source image files.",
        "- The reference and at least one target are retained. The probability applies to non-reference frames.",
        f"- Projection dropout: {config['projection_dropout']}; classifier dropout: {config['dropout']}",
        f"- Seed reset for every experiment: {config['seed']}",
        f"- Epochs planned: {config['epochs']}; early stopping patience: {config['patience']}",
        f"- Learning rate: {config['learning_rate']}; weight decay: {config['weight_decay']}",
        f"- Gradient clipping maximum norm: {config['grad_clip_norm']}",
        f"- Device used: {config['actual_device']}; GPU: {config['gpu_name'] or 'N/A'}",
        f"- PyTorch: {config['torch_version']}; torchvision: {config['torchvision_version']}",
        f"- Class weights [preserved, looted]: {config['class_weights']}",
        f"- Deterministic algorithms requested: {config['deterministic']}; precision: float32",
        "- Checkpoint selection: highest validation F1; ties retain the earlier epoch and count toward patience.",
        f"- Configuration selection: {SELECTION_RULE}",
        "- Selection was saved before any test predictions were generated in this run.",
        "- Each predefined configuration was tested once with all K frames; robustness evaluations are separate.",
        "- Prediction rule: larger logit; ties select preserved.", "", "## Checked site counts", "",
        "| Split | Preserved | Looted | Total |", "|---|---:|---:|---:|"]
    for split in SPLITS:
        c = counts[split]
        lines.append(f"| {split} | {c['preserved']} | {c['looted']} | {sum(c.values())} |")
    lines += ["", "## Predefined experiments", "",
        "| K | Strategy | Variant | Epochs | Best epoch | Best val F1 | Test accuracy | Test precision | Test recall | Test F1 | Test AUC | Test FAR | Selected |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in results:
        scores = [format_score(r[key]) for key in ("best_val_f1", "test_accuracy", "test_precision", "test_recall",
                                                   "test_f1", "test_roc_auc", "test_false_alarm_rate")]
        lines.append(f"| {r['k']} | {r['strategy']} | {r['variant']} | {r['epochs_completed']} | {r['best_epoch']} | "
                     + " | ".join(scores) + f" | {'yes' if r['selected_by_validation'] else ''} |")
    lines += ["", "## Test counts and weighted loss", "",
              "| Experiment | Samples | TN | FP | FN | TP | Loss | Parameters |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['experiment_name']} | {r['test_samples']} | {r['test_tn']} | {r['test_fp']} | "
                     f"{r['test_fn']} | {r['test_tp']} | {r['test_loss']:.4f} | {r['parameters_total']} |")
    lines += ["", "## Validation-selected proposed configuration", "",
        f"- Experiment: {selected['experiment_name']}", f"- Best validation F1: {selected['best_val_f1']:.4f}",
        f"- Test F1 of that preselected configuration: {chosen['test_f1']:.4f}",
        f"- Weights: `{selected['best_model_path']}`", "",
        "## Paired simulated missing-observation robustness", "",
        "Both models use the same K, strategy, sites and retained-frame masks. p=0 reuses the normal test result.",
        "Standard deviations describe variation across missing-frame masks, not statistical significance or training-seed variation.", "",
        "| Variant | Drop probability | Repeats | Mean retained frames | Actual dropped fraction | F1 mean | F1 std | Recall mean | FAR mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in robustness:
        lines.append(f"| {r['variant']} | {r['drop_probability']:.2f} | {r['repeats']} | {r['mean_retained_frames_mean']:.3f} | "
                     f"{r['actual_dropped_fraction_mean']:.4f} | {r['f1_mean']:.4f} | {r['f1_std']:.4f} | "
                     f"{r['recall_mean']:.4f} | {r['false_alarm_rate_mean']:.4f} |")
    lines += ["", "## Baseline comparison", "",
        f"Comparison status: {'COMPLETE' if baselines['complete'] else 'INCOMPLETE: one or more baseline runs were excluded'}.",
        "Runs were fixed before Step 10 training using compatibility and timestamp, never test scores.",
        "Training settings differ across steps and are recorded in the comparison CSV. These are descriptive comparisons.", "",
        "| Model | K | Strategy | Method | Test F1 | Test accuracy | Test recall | Test FAR |",
        "|---|---:|---|---|---:|---:|---:|---:|"]
    for r in comparison:
        lines.append(f"| {r['model']} | {r['k']} | {r['strategy']} | {r['method']} | {r['f1']:.4f} | "
                     f"{r['accuracy']:.4f} | {r['recall']:.4f} | {format_score(r['false_alarm_rate'])} |")
    for item in baselines["checks"]:
        if item["status"] == "excluded":
            lines.append(f"- Step {item['step']} excluded: {item['reason']} See baseline_inputs.json for details.")
    lines += ["", "## Files", "", f"- Model folder: `{model_dir}`", f"- Report folder: `{run_dir}`",
        "- step10_summary_results.csv; step10_ablation_results.csv; step10_k_analysis_results.csv",
        "- validation_results.csv; selected_configuration.json; run_metadata.json",
        "- step10_frame_drop_robustness.csv; step10_robustness_summary.csv; robustness/ predictions",
        "- final_model_comparison.csv; baseline_comparison_status.json; baseline_inputs.json",
        "- Every experiment has history, metadata, validation predictions/metrics and test predictions/metrics.",
        "", "## Scope and limitations", "",
        "- Current Step 3/4 source metadata and site memberships were checked; train=596, val=18, test=61.",
        "- Every epoch checks coverage, identities, labels, shapes and finite values. Loss uses target-weight sums.",
        "- This is one fixed split with one training seed, not cross-validation or a statistical significance study.",
        "- Selecting among many configurations with only 18 validation sites can overfit the validation set.",
        "- Existing test exposure from earlier experiments cannot be undone by this script.",
        "- Ablations share architecture and parameter shapes; zero change inputs do not have the same effective information/capacity.",
        "- The first selected reference image may already show looting. Feature differences are not looting-specific explanations.",
        "- GRU uses chronological order but receives no dates or time gaps; a few frames may span many years.",
        "- K3/K5/K8 subsets and simulated dropping are evaluated; full sequences, 100-frame input and naturally mixed lengths are not.",
        "- Reference loss, cloud corruption and missing files on disk are not simulated.",
        "- Random strategy uses saved Step 4 selections. Training masks and image selection are different operations.",
        "- Whole images are used, without site masking or registration. Source identities and labels need separate auditing.",
        "- Parameter counts are recorded; runtime, energy use and GPU-memory efficiency are not benchmarked.",
        "- Metadata hashes do not verify every raw image's content. Images must remain unchanged during experiments.",
        "- Training metrics use augmentation and changing model weights. Seeds do not ensure identical results across platforms."]
    (run_dir / REPORT_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute_suite(torch, args, device, tables, inputs, counts, config, run_dir, model_dir, baselines):
    planned = [(k, strategy, variant) for k in args.k_values for strategy in args.strategies for variant in VARIANTS]
    state = {"status": "RUNNING", "phase": "training", "started_utc": datetime.now(timezone.utc).isoformat(),
             "config": config, "site_counts": counts, "inputs": inputs,
             "planned_experiments": [{"k": k, "strategy": s, "variant": v} for k, s, v in planned]}
    metadata_path = run_dir / "run_metadata.json"
    json_write(metadata_path, state)
    json_write(run_dir / "baseline_inputs.json", baselines)
    trained, results = [], []
    try:
        for number, (k, strategy, variant) in enumerate(planned, 1):
            print(f"\nTraining {number}/{len(planned)}: {experiment_name(k, strategy, variant)}", flush=True)
            check_inputs_unchanged(inputs)
            trained.append(train_experiment(torch, args, device, tables[(k, strategy)], inputs, config,
                                            run_dir, model_dir, k, strategy, variant))
            columns = ("experiment_name", "k", "strategy", "variant", "best_val_f1", "best_epoch", "epochs_completed",
                       "seed", "best_model_path", "checkpoint_sha256")
            pd.DataFrame([{key: r[key] for key in columns} for r in trained]).to_csv(run_dir / "validation_results.csv", index=False)
            state["training_completed"] = [r["experiment_name"] for r in trained]
            json_write(metadata_path, state)
            release_memory(torch, device)
        check_experiment_coverage(trained, planned)
        require(len({r["parameter_counts"]["total"] for r in trained}) == 1,
                "Ablation architectures unexpectedly have different parameter counts.")
        check_inputs_unchanged(inputs)
        selected = select_configuration(trained)
        selection = {"selection_rule": SELECTION_RULE, "selected_before_test_evaluation": True,
                     "selected_utc": datetime.now(timezone.utc).isoformat(),
                     **{key: selected[key] for key in ("experiment_name", "k", "strategy", "variant", "best_val_f1",
                                                       "best_epoch", "best_model_path", "checkpoint_sha256")}}
        json_write(run_dir / "selected_configuration.json", selection)
        state.update({"phase": "testing", "selected_configuration": selection})
        json_write(metadata_path, state)
        print(f"\nValidation selected {selected['experiment_name']}. Selection is now fixed.", flush=True)
        for number, row in enumerate(trained, 1):
            print(f"\nTesting {number}/{len(planned)}: {row['experiment_name']}", flush=True)
            metrics = test_experiment(torch, args, device, tables[(row["k"], row["strategy"])], inputs, config, row)
            results.append(summary_row(row, metrics, selected["experiment_name"]))
            pd.DataFrame(results).to_csv(run_dir / SUMMARY_NAME, index=False)
            state["test_completed"] = [r["experiment_name"] for r in results]
            json_write(metadata_path, state)
            release_memory(torch, device)
        check_experiment_coverage(results, planned)
        pd.DataFrame(results).to_csv(run_dir / "step10_ablation_results.csv", index=False)
        pd.DataFrame([r for r in results if r["variant"] == "proposed"]).to_csv(run_dir / "step10_k_analysis_results.csv", index=False)
        state["phase"] = "paired robustness"
        json_write(metadata_path, state)
        robustness = paired_robustness(torch, args, device, tables, inputs, config, run_dir, trained, selected)
        check_inputs_unchanged(inputs)
        comparison = create_final_comparison(run_dir, config, results, selected, baselines)
        write_report(run_dir, model_dir, config, counts, results, selected, robustness, comparison, baselines)
        state.update({"status": "COMPLETE", "phase": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
                      "completed_experiments": len(results), "baseline_comparison_complete": baselines["complete"]})
        json_write(metadata_path, state)
        print(f"\nStep 10 COMPLETE: {len(results)} experiments, 61 test sites each.", flush=True)
        print(f"Report: {run_dir / REPORT_NAME}\nValidation-selected weights: {selected['best_model_path']}", flush=True)
        return results
    except BaseException as exc:
        message = f"{type(exc).__name__}: {exc}"
        state.update({"status": "FAILED", "error": message})
        json_write(metadata_path, state)
        (run_dir / REPORT_NAME).write_text(
            "# Step 10 Proposed Model Report\n\n## Status: FAILED\n\n"
            f"- Error: {message}\n- Training completed: {len(trained)}/{len(planned)}\n"
            f"- Testing completed: {len(results)}/{len(planned)}\n- Partial output files are not a completed study.\n", encoding="utf-8")
        raise


def validate_arguments(args):
    require(args.batch_size > 0 and args.image_size >= 32 and args.epochs > 0 and args.patience > 0
            and args.hidden_dim > 0 and args.num_workers >= 0 and 0 <= args.seed < 2**32
            and 0 <= args.robustness_seed < 2**32 and args.robustness_repeats >= 2, "Invalid integer settings.")
    for name in ("learning_rate", "grad_clip_norm"):
        require(math.isfinite(getattr(args, name)) and getattr(args, name) > 0, f"Invalid {name}.")
    require(math.isfinite(args.weight_decay) and args.weight_decay >= 0, "Invalid weight decay.")
    for name in ("dropout", "projection_dropout"):
        require(math.isfinite(getattr(args, name)) and 0 <= getattr(args, name) < 1, f"Invalid {name}.")
    require(math.isfinite(args.frame_drop_probability) and 0 < args.frame_drop_probability < 1,
            "Training frame-drop probability must be strictly between zero and one for distinct ablations.")
    for values, allowed, name in ((args.k_values, K_VALUES, "K values"), (args.strategies, STRATEGIES, "strategies")):
        require(values and len(values) == len(set(values)) and set(values).issubset(allowed), f"Invalid/repeated {name}.")
    probs = args.robustness_probabilities
    require(probs and len(probs) == len(set(probs)) and all(math.isfinite(p) and 0 < p <= 1 for p in probs),
            "Robustness probabilities must be unique in (0,1]; p=0 is automatically included.")
    args.k_values = sorted(args.k_values)
    args.strategies = sorted(args.strategies, key=STRATEGIES.index)
    args.robustness_probabilities = sorted(probs)


def main(args):
    validate_arguments(args)
    root = find_project_root(args.project_root)
    data_root = args.data_root.expanduser().resolve() if args.data_root else root / "data/raw/DAFA_LS"
    print(f"Project root: {root}", flush=True)
    tables, counts, inputs = preflight(root, data_root, args.strategies, args.k_values)
    print("Preflight PASS: train=596, val=18, test=61 for every requested K/strategy.", flush=True)
    total = len(args.k_values) * len(args.strategies) * len(VARIANTS)
    print(f"Planned experiments: {total} = {len(args.k_values)} K values x {len(args.strategies)} strategies x 4 variants.", flush=True)
    if args.preflight_only:
        print("Metadata and selected image-file existence checked. No image decoding, model training or baseline comparison performed.")
        return 0
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    dl.require_torch()
    torch = dl.torch
    import torchvision
    device = choose_device(torch, args.device)
    first = tables[(args.k_values[0], args.strategies[0])]
    weights = class_weights_from_labels(first.loc[first.split == "train", "label_id"])
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"project_root": str(root), "data_root": str(data_root), "actual_device": str(device),
                   "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "torch_version": str(torch.__version__), "torchvision_version": str(torchvision.__version__),
                   "cuda_runtime": torch.version.cuda, "class_weights": weights.tolist(), "label_mapping": LABELS,
                   "normalization": "imagenet", "selection_rule": SELECTION_RULE, "variants": list(VARIANTS)})
    baselines = prepare_baselines(root, inputs, args)
    for item in baselines["checks"]:
        print(f"Step {item['step']} baseline: {item['status']}" +
              (f" ({item['run']})" if item['status'] == "included" else " - see baseline_inputs.json after this run starts."), flush=True)
    strategy_tag = "all_strategies" if len(args.strategies) == 4 else "_".join(args.strategies)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + f"_{strategy_tag}_seed{args.seed}"
    run_dir = root / "results/proposed_change_aware_fusion" / run_id
    model_dir = root / "models/proposed_change_aware_fusion" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)
    print(f"Device: {device}" + (f" ({config['gpu_name']})" if config['gpu_name'] else ""), flush=True)
    print(f"Reports: {run_dir}\nModels: {model_dir}", flush=True)
    execute_suite(torch, args, device, tables, inputs, counts, config, run_dir, model_dir, baselines)
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--strategies", "--strategy", dest="strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--k-values", type=int, nargs="+", choices=K_VALUES, default=list(K_VALUES))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=2, help="Sites per batch; defaults to 2 for your GTX 1650.")
    parser.add_argument("--image-size", type=int, default=266)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--projection-dropout", type=float, default=0.2)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--frame-drop-probability", type=float, default=0.25)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--robustness-seed", type=int, default=4242)
    parser.add_argument("--robustness-repeats", type=int, default=5)
    parser.add_argument("--robustness-probabilities", type=float, nargs="+", default=[0.25, 0.50])
    for step in BASELINE_DIRS:
        parser.add_argument(f"--step{step}-run", type=Path, default=None, help=f"Explicit compatible Step {step} run folder.")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main(parse_args()))
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(f"Step 10 FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)

