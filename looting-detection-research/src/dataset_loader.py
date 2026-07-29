from pathlib import Path
import json
import random
from typing import List, Optional

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


class ConsistentTemporalTransform:
    """
    Applies the same spatial augmentation to every frame in one time-series sample.
    This is important because random independent augmentation per frame can damage
    temporal/change information.
    """

    def __init__(
        self,
        image_size: int = 266,
        train: bool = False,
        normalization: str = "imagenet"
    ):
        self.image_size = image_size
        self.train = train
        self.normalization = normalization

        if normalization == "imagenet":
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        elif normalization == "zero_one":
            self.mean = None
            self.std = None
        else:
            raise ValueError("normalization must be 'imagenet' or 'zero_one'")

    def __call__(self, frames: List[Image.Image]) -> torch.Tensor:
        processed_frames = []

        # Same augmentation decision for all frames in one sample
        do_hflip = self.train and random.random() < 0.5
        do_vflip = self.train and random.random() < 0.5
        rotation_angle = random.choice([0, 90, 180, 270]) if self.train else 0

        for img in frames:
            img = img.convert("RGB")
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

            if do_hflip:
                img = TF.hflip(img)

            if do_vflip:
                img = TF.vflip(img)

            if rotation_angle != 0:
                img = TF.rotate(img, rotation_angle)

            tensor = TF.to_tensor(img)  # shape: [3, H, W], values: 0 to 1
            processed_frames.append(tensor)

        x = torch.stack(processed_frames, dim=0)  # shape: [K, 3, H, W]

        if self.normalization == "imagenet":
            x = (x - self.mean) / self.std

        return x


class KFrameLootingDataset(Dataset):
    """
    Dataset class for DAFA-LS K-frame looting classification.

    Output:
        images: Tensor [K, 3, 266, 266]
        label:  Tensor scalar, 0 = preserved, 1 = looted
    """

    def __init__(
        self,
        csv_path: str,
        split: Optional[str] = None,
        strategy: Optional[str] = None,
        transform: Optional[ConsistentTemporalTransform] = None,
        max_samples: Optional[int] = None
    ):
        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        if split is not None:
            self.df = self.df[self.df["split"].astype(str).str.lower() == split.lower()]

        if strategy is not None:
            self.df = self.df[self.df["strategy"].astype(str).str.lower() == strategy.lower()]

        self.df = self.df.reset_index(drop=True)

        if max_samples is not None:
            self.df = self.df.head(max_samples).reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(
                f"No samples found in {csv_path} for split={split}, strategy={strategy}"
            )

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _load_paths(self, row):
        """
        Reads full_paths from JSON list.
        """
        paths = json.loads(row["full_paths"])

        if not isinstance(paths, list):
            raise ValueError("full_paths must be a JSON list")

        return [Path(p) for p in paths]

    def _load_images(self, paths: List[Path]) -> List[Image.Image]:
        images = []

        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")

            img = Image.open(path).convert("RGB")
            images.append(img)

        return images

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_paths = self._load_paths(row)
        images = self._load_images(image_paths)

        if self.transform is not None:
            images_tensor = self.transform(images)
        else:
            default_transform = ConsistentTemporalTransform(
                image_size=266,
                train=False,
                normalization="imagenet"
            )
            images_tensor = default_transform(images)

        label = torch.tensor(int(row["label_id"]), dtype=torch.long)

        return {
            "images": images_tensor,
            "label": label,
            "site_id": str(row["site_id"]),
            "sample_id": str(row["sample_id"]),
            "strategy": str(row["strategy"]),
            "split": str(row["split"]),
            "k": int(row["k"])
        }


def create_dataloader(
    csv_path: str,
    split: str,
    strategy: str,
    batch_size: int = 8,
    image_size: int = 266,
    train: bool = False,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    normalization: str = "imagenet"
):
    """
    Creates a PyTorch DataLoader for a selected K-frame CSV, split, and strategy.
    num_workers=0 is safer for Windows.
    """

    if shuffle is None:
        shuffle = train

    transform = ConsistentTemporalTransform(
        image_size=image_size,
        train=train,
        normalization=normalization
    )

    dataset = KFrameLootingDataset(
        csv_path=csv_path,
        split=split,
        strategy=strategy,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return loader