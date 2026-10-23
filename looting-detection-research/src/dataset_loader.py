"""Validated DAFA-LS K-frame loader. Requires pandas, Pillow, torch, torchvision.

Uses corrected Step 4 CSV files. Site identifiers remain text. Images and
transforms stay on the CPU; the training loop moves tensors to its device.
Whole images are used: this loader does not apply archaeological-site masks.
Never use site_uid, sample_id, or class-bearing paths as model input features.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import json
import random
import re

import pandas as pd
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms.functional as TF
    _TORCH_ERROR = ""
except (ImportError, OSError, RuntimeError) as exc:
    # Metadata inspection is still possible without an installed torch runtime.
    torch = TF = DataLoader = None
    Dataset = object
    _TORCH_ERROR = str(exc)

K_VALUES = [1, 2, 3, 5, 8]
STRATEGIES = ["recent_window", "uniform", "early_late", "random"]
SPLITS = ["train", "val", "test"]
LABEL_MAPPING = {"preserved": 0, "looted": 1}
LIST_COLUMNS = ["selected_positions", "frame_indices", "years", "months", "relative_paths", "full_paths"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_torch():
    if torch is None:
        raise RuntimeError("A working PyTorch and torchvision installation is required. " + _TORCH_ERROR)


def seed_everything(seed=42):
    """Call once at the start of a run, not again at every epoch."""
    require_torch()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """Top-level function so Windows worker processes can import it."""
    random.seed(torch.initial_seed() % (2 ** 32))


def integer_column(series, name):
    values = pd.to_numeric(series, errors="coerce")
    require(values.notna().all() and values.mod(1).eq(0).all(), f"Invalid integer values in {name}.")
    return values.astype("int64")


def relative_path(value):
    require(isinstance(value, str) and bool(value.strip()), "An image path is empty or is not text.")
    text = value.replace("\\", "/")
    path = PurePosixPath(text)
    require(not path.is_absolute() and ".." not in path.parts and not re.match(r"^[A-Za-z]:", text),
            f"Expected a dataset-relative path: {value!r}")
    return path.as_posix()


def read_samples_csv(csv_path):
    """Validate the complete CSV before filtering to one split or strategy."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    required = {"sample_id", "site_uid", "site_id", "label", "label_id", "split", "k",
                "strategy", "selected_count", "available_frames", *LIST_COLUMNS}
    missing = sorted(required - set(df.columns))
    require(not missing, f"Missing columns {missing} in {csv_path}. Rerun corrected Step 4.")
    require(not df.empty, f"No samples in {csv_path}.")
    for column in required:
        require(df[column].str.strip().ne("").all(), f"Missing values in {column}: {csv_path}")
    df["split"] = df["split"].str.strip().str.lower()
    df["strategy"] = df["strategy"].str.strip().str.lower()
    require(df["split"].isin(SPLITS).all(), "Unknown split name.")
    require(df["strategy"].isin(STRATEGIES + ["full_sequence"]).all(), "Unknown strategy name.")
    require(df["label"].isin(LABEL_MAPPING).all(), "Unknown class label.")
    require(df["site_uid"].eq(df["label"] + "/" + df["site_id"]).all(), "site_uid/label/site_id mismatch.")
    for column in ["label_id", "k", "selected_count", "available_frames"]:
        df[column] = integer_column(df[column], column)
    require(df["label_id"].eq(df["label"].map(LABEL_MAPPING)).all(), "Incorrect numeric label mapping.")
    require(df["k"].gt(0).all() and df["selected_count"].eq(df["k"]).all()
            and df["available_frames"].ge(df["k"]).all(), "Invalid K or selected/available frame counts.")
    require(df["sample_id"].is_unique, "Duplicate sample IDs.")
    require(not df.duplicated(["site_uid", "k", "strategy"]).any(), "Duplicate site/K/strategy rows.")
    require(df.groupby("site_uid")["split"].nunique().eq(1).all(), "A site_uid occurs in multiple splits.")
    require(df.groupby("site_uid")["available_frames"].nunique().eq(1).all(), "Inconsistent site frame counts.")
    for name in LIST_COLUMNS:
        parsed = []
        for sample_id, value, k in zip(df["sample_id"], df[name], df["k"]):
            try:
                values = json.loads(value)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid JSON in {name} for {sample_id}: {exc}") from exc
            require(isinstance(values, list) and len(values) == k,
                    f"{name} must contain exactly K entries: {sample_id}")
            parsed.append(values)
        df[name] = parsed
    for row in df.itertuples(index=False):
        for name in ["selected_positions", "frame_indices", "years", "months"]:
            require(all(type(v) is int for v in getattr(row, name)),
                    f"Non-integer list entries in {name}: {row.sample_id}")
        indices = row.frame_indices
        require(indices == sorted(set(indices)) and min(indices) >= 0 and max(indices) < row.available_frames,
                f"Repeated, unordered, or invalid frame indices: {row.sample_id}")
        require(row.selected_positions == indices, f"Position/index mismatch: {row.sample_id}")
        require(all(1 <= y <= 9999 and 1 <= m <= 12 for y, m in zip(row.years, row.months)),
                f"Invalid dates: {row.sample_id}")
        dates = list(zip(row.years, row.months))
        require(dates == sorted(dates), f"Selected dates are out of order: {row.sample_id}")
        paths = [relative_path(p) for p in row.relative_paths]
        require(len(set(p.casefold() for p in paths)) == row.k,
                f"Repeated image paths within sequence: {row.sample_id}")
        for path, full in zip(paths, row.full_paths):
            parts = PurePosixPath(path).parts
            require(len(parts) >= 3 and f"{parts[0].lower()}/{parts[1]}" == row.site_uid,
                    f"Image belongs to a different site: {row.sample_id}, {path}")
            require(isinstance(full, str) and full.replace("\\", "/").endswith("/" + path),
                    f"Full and relative image paths disagree: {row.sample_id}")
    df["relative_paths"] = df["relative_paths"].map(lambda paths: [relative_path(p) for p in paths])
    return df


def load_rgb_images(paths):
    """Decode detached RGB images and close file handles immediately."""
    images = []
    try:
        for path in paths:
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(f"Image not found: {path}. Check data_root and rerun Steps 3/4 if needed.")
            with Image.open(path) as source:
                rgb = source.convert("RGB")
                rgb.load()
            images.append(rgb)
        return images
    except Exception:
        for image in images:
            image.close()
        raise


class ConsistentTemporalTransform:
    """The same orientation change is applied to every frame in a sequence."""
    def __init__(self, image_size=266, train=False, normalization="imagenet"):
        require(isinstance(image_size, int) and image_size > 0, "image_size must be a positive integer.")
        require(normalization in {"imagenet", "zero_one"}, "Choose imagenet or zero_one normalization.")
        self.image_size = image_size
        self.train = train
        self.normalization = normalization

    def prepare_images(self, frames):
        """Pillow-only preparation, also useful for inspecting augmentation."""
        require(bool(frames), "Cannot transform an empty image sequence.")
        do_hflip = self.train and random.random() < 0.5
        do_vflip = self.train and random.random() < 0.5
        turns = random.choice([0, 1, 2, 3]) if self.train else 0
        rotations = {1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180, 3: Image.Transpose.ROTATE_270}
        prepared = []
        for source in frames:
            img = source.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            if do_hflip:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if do_vflip:
                img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if turns:
                img = img.transpose(rotations[turns])
            prepared.append(img)
        return prepared

    def __call__(self, frames):
        require_torch()
        prepared = self.prepare_images(frames)
        try:
            x = torch.stack([TF.to_tensor(img) for img in prepared], dim=0)
        finally:
            for image in prepared:
                image.close()
        if self.normalization == "imagenet":
            mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            x = (x - mean) / std
        return x


class KFrameLootingDataset(Dataset):
    """One item is one site sequence, with images [K,3,H,W] and a scalar label.

    data_root overrides saved full paths using the validated relative paths,
    allowing a project to move between machines. Without it, full_paths are used.
    """
    def __init__(self, csv_path, split=None, strategy=None, transform=None,
                 max_samples=None, data_root=None, k=None):
        self.csv_path = Path(csv_path)
        self.df = read_samples_csv(self.csv_path)
        if split is not None:
            self.df = self.df[self.df["split"] == split.strip().lower()]
        if strategy is not None:
            self.df = self.df[self.df["strategy"] == strategy.strip().lower()]
        if k is not None:
            self.df = self.df[self.df["k"] == k]
        if max_samples is not None:
            require(isinstance(max_samples, int) and max_samples > 0, "max_samples must be positive.")
            self.df = self.df.head(max_samples)
        self.df = self.df.reset_index(drop=True)
        require(not self.df.empty, f"No samples in {csv_path} for split={split}, strategy={strategy}, K={k}.")
        self.data_root = Path(data_root).resolve() if data_root is not None else None
        self.transform = transform if transform is not None else ConsistentTemporalTransform()

    def __len__(self):
        return len(self.df)

    def _load_paths(self, row):
        if self.data_root is not None:
            return [self.data_root.joinpath(*PurePosixPath(p).parts) for p in row["relative_paths"]]
        paths = [Path(p) for p in row["full_paths"]]
        require(all(p.is_absolute() for p in paths),
                "Saved full paths are not absolute on this platform. Supply data_root for this machine.")
        return paths

    def __getitem__(self, idx):
        require_torch()
        row = self.df.iloc[idx]
        images = load_rgb_images(self._load_paths(row))
        try:
            x = self.transform(images)
        finally:
            for image in images:
                image.close()
        require(isinstance(x, torch.Tensor) and x.ndim == 4 and x.shape[0] == row["k"]
                and x.shape[1] == 3, f"Wrong transformed sequence shape: {row['sample_id']}")
        require(x.dtype == torch.float32 and torch.isfinite(x).all().item(),
                f"Invalid image tensor dtype or non-finite pixels: {row['sample_id']}")
        return {
            "images": x, "label": torch.tensor(int(row["label_id"]), dtype=torch.long),
            "site_uid": row["site_uid"], "site_id": row["site_id"],
            "sample_id": row["sample_id"], "strategy": row["strategy"],
            "split": row["split"], "k": int(row["k"]),
        }


def create_dataloader(csv_path, split, strategy, batch_size=8, image_size=266,
                      train=False, shuffle=None, num_workers=0, normalization="imagenet",
                      data_root=None, seed=42, k=None, pin_memory=None):
    require_torch()
    require(isinstance(batch_size, int) and batch_size > 0, "batch_size must be positive.")
    require(isinstance(num_workers, int) and num_workers >= 0, "num_workers must be non-negative.")
    require(not train or split.strip().lower() == "train", "Random augmentation is only allowed on training data.")
    if shuffle is None:
        shuffle = train
    dataset = KFrameLootingDataset(
        csv_path, split=split, strategy=strategy,
        transform=ConsistentTemporalTransform(image_size, train, normalization), data_root=data_root, k=k,
    )
    require(dataset.df["k"].nunique() == 1,
            "Default batching requires one K. Use a per-K CSV or specify k; full sequences need a custom collator.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
                      worker_init_fn=seed_worker, generator=generator)
