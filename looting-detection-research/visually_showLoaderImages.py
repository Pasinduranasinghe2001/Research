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

CSV_PATH = Path(
    "data/processed/kframe_sequences/kframe_samples_K3.csv"
)

OUTPUT_DIR = Path(
    "results/figures/transformation_examples/different_sites"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY = "uniform"
SPLIT = "train"

# How many DIFFERENT sites you want
NUM_SITES = 6

# Which frame from each K=3 sequence
# 0 = first frame
# 1 = second frame
# 2 = third frame
FRAME_INDEX = 0

random.seed(42)
torch.manual_seed(42)


# ==========================================================
# REVERSE IMAGENET NORMALIZATION
# ==========================================================

def denormalize_imagenet(tensor):

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)

    return tensor


# ==========================================================
# LOAD CSV
# ==========================================================

df = pd.read_csv(CSV_PATH)

subset = df[
    (df["strategy"].astype(str).str.lower() == STRATEGY.lower()) &
    (df["split"].astype(str).str.lower() == SPLIT.lower())
].copy()

if len(subset) == 0:
    raise ValueError("No matching samples found.")


# ==========================================================
# SELECT DIFFERENT SITES
# ==========================================================

# Keeps only ONE sample from each site
different_sites = (
    subset
    .drop_duplicates(subset=["site_id"])
    .reset_index(drop=True)
)

different_sites = different_sites.head(NUM_SITES)

print(f"Found {len(different_sites)} different sites.")


# ==========================================================
# TRANSFORM
# ==========================================================

transform = ConsistentTemporalTransform(
    image_size=266,
    train=True,
    normalization="imagenet"
)


# ==========================================================
# PROCESS EACH SITE
# ==========================================================

for index, row in different_sites.iterrows():

    site_id = str(row["site_id"])

    paths = json.loads(row["full_paths"])

    original_images = [
        Image.open(path).convert("RGB")
        for path in paths
    ]

    # Make sure selected frame exists
    if FRAME_INDEX >= len(original_images):
        print(f"Skipping site {site_id}: frame does not exist")
        continue

    # ------------------------------------------------------
    # ORIGINAL IMAGE
    # ------------------------------------------------------

    original_image = original_images[FRAME_INDEX]

    # ------------------------------------------------------
    # APPLY PREPROCESSING TO WHOLE K-FRAME SEQUENCE
    # ------------------------------------------------------

    processed_sequence = transform(original_images)

    processed_tensor = processed_sequence[FRAME_INDEX]

    # Reverse normalization ONLY for visualization
    processed_tensor = denormalize_imagenet(
        processed_tensor
    )

    display_processed = (
        processed_tensor
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


    # ======================================================
    # CREATE ONE PNG FOR THIS SITE
    # ======================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8, 4)
    )

    # ORIGINAL
    axes[0].imshow(original_image)
    axes[0].set_title(
        "Original Image",
        fontsize=14,
        fontweight="bold"
    )
    axes[0].axis("off")


    # PREPROCESSED
    axes[1].imshow(display_processed)
    axes[1].set_title(
        "After Preprocessing",
        fontsize=14,
        fontweight="bold"
    )
    axes[1].axis("off")


    # Main title
    fig.suptitle(
        f"Site {site_id}",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()


    # ======================================================
    # SAVE
    # ======================================================

    # Clean site ID so Windows filename is safe
    safe_site_id = (
        site_id
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )

    output_path = (
        OUTPUT_DIR /
        f"site_{safe_site_id}_original_vs_preprocessed.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


print("\nDone!")
print(f"Images saved in: {OUTPUT_DIR}")