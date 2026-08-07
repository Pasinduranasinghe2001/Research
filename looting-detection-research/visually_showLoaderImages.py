from pathlib import Path
import json
import random

import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image

from src.dataset_loader import ConsistentTemporalTransform


# ==========================================================
# SETTINGS
# ==========================================================

CSV_PATH = Path("data/processed/kframe_sequences/kframe_samples_K3.csv")

OUTPUT_DIR = Path("results/figures/transformation_examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY = "uniform"
SPLIT = "train"

NUM_EXAMPLES = 1

# Fix random seed so results are repeatable
random.seed(42)
torch.manual_seed(42)


# ==========================================================
# FUNCTION: REVERSE IMAGENET NORMALIZATION
# ==========================================================

def denormalize_imagenet(tensor):
    """
    Convert ImageNet-normalized tensor back to displayable RGB image.
    Tensor shape: [3, H, W]
    """

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    tensor = tensor * std + mean

    # Keep values between 0 and 1
    tensor = torch.clamp(tensor, 0, 1)

    return tensor


# ==========================================================
# LOAD CSV
# ==========================================================

df = pd.read_csv(CSV_PATH)

subset = df[
    (df["strategy"].astype(str).str.lower() == STRATEGY.lower()) &
    (df["split"].astype(str).str.lower() == SPLIT.lower())
].reset_index(drop=True)

if len(subset) == 0:
    raise ValueError("No matching samples found.")


# ==========================================================
# TRANSFORM
# ==========================================================

transform = ConsistentTemporalTransform(
    image_size=266,
    train=True,                 # IMPORTANT: enables augmentation
    normalization="imagenet"
)


# ==========================================================
# CREATE EXAMPLES
# ==========================================================

for sample_index in range(min(NUM_EXAMPLES, len(subset))):

    row = subset.iloc[sample_index]

    paths = json.loads(row["full_paths"])

    original_images = [
        Image.open(path).convert("RGB")
        for path in paths
    ]

    # Apply SAME augmentation to all frames
    transformed_tensor = transform(original_images)

    k = len(original_images)

    # ------------------------------------------------------
    # Figure
    # ------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        k,
        figsize=(4 * k, 8)
    )

    # If K=1, matplotlib behaves differently
    if k == 1:
        axes = axes.reshape(2, 1)

    for i in range(k):

        # ==================================================
        # ORIGINAL
        # ==================================================

        axes[0, i].imshow(original_images[i])
        axes[0, i].axis("off")

        axes[0, i].set_title(
            f"Original Frame {i + 1}",
            fontsize=12,
            fontweight="bold"
        )

        # ==================================================
        # TRANSFORMED
        # ==================================================

        img_tensor = transformed_tensor[i]

        # Undo ImageNet normalization for visualization
        img_tensor = denormalize_imagenet(img_tensor)

        # PyTorch shape:
        # [C,H,W]
        #
        # matplotlib wants:
        # [H,W,C]

        display_img = img_tensor.permute(1, 2, 0).numpy()

        axes[1, i].imshow(display_img)
        axes[1, i].axis("off")

        axes[1, i].set_title(
            f"Transformed Frame {i + 1}",
            fontsize=12,
            fontweight="bold"
        )

    fig.suptitle(
        f"Original vs Training Transformation\n"
        f"Site: {row['site_id']} | "
        f"K={row['k']} | "
        f"Strategy: {row['strategy']}",
        fontsize=15,
        fontweight="bold"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        f"sample_{sample_index + 1}_original_vs_transformed.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("Saved:", output_path)