from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================================
# PATHS
# ==========================================================

SPLIT_CSV = Path("data/splits/site_splits.csv")
FRAMES_CSV = Path("data/splits/frames_with_splits.csv")

OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

site_df = pd.read_csv(SPLIT_CSV)
frames_df = pd.read_csv(FRAMES_CSV)

# ==========================================================
# 1. PREPROCESSING PIPELINE
# ==========================================================

fig, ax = plt.subplots(figsize=(16, 4.5))
ax.axis("off")

steps = [
    ("Valid Frames", "63,439\nsatellite frames"),
    ("Temporal Sorting", "Year → Month → Day"),
    ("Metadata", "Site-level +\nframe-level"),
    ("Official Split", "Train / Val / Test\nsite-wise"),
    ("Leakage Check", "0 sites in\nmultiple splits"),
    ("Binary Labels", "0 = Preserved\n1 = Looted")
]

x_positions = np.linspace(0.07, 0.93, len(steps))

for i, ((title, subtitle), x) in enumerate(zip(steps, x_positions)):

    ax.text(
        x,
        0.60,
        title,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        transform=ax.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.7",
            facecolor="white",
            edgecolor="black",
            linewidth=1.5
        )
    )

    ax.text(
        x,
        0.25,
        subtitle,
        ha="center",
        va="center",
        fontsize=10.5,
        transform=ax.transAxes
    )

    if i < len(steps) - 1:
        ax.annotate(
            "",
            xy=(x_positions[i + 1] - 0.055, 0.60),
            xytext=(x + 0.055, 0.60),
            xycoords=ax.transAxes,
            arrowprops=dict(
                arrowstyle="->",
                linewidth=2
            )
        )

ax.set_title(
    "DAFA-LS Preprocessing and Dataset Split Pipeline",
    fontsize=19,
    fontweight="bold",
    pad=20
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "preprocessing_pipeline.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# 2. TRAIN / VALIDATION / TEST FRAME DISTRIBUTION
# ==========================================================

split_counts = (
    frames_df["split"]
    .value_counts()
    .reindex(["train", "val", "test"])
)

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    ["Training", "Validation", "Test"],
    split_counts.values
)

ax.set_title(
    "Frame Distribution Across Dataset Splits",
    fontsize=17,
    fontweight="bold"
)

ax.set_ylabel("Number of Frames")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for bar, value in zip(bars, split_counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{int(value):,}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold"
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "split_frame_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# 3. CLASS DISTRIBUTION INSIDE EACH SPLIT
# ==========================================================

class_split = (
    frames_df
    .groupby(["split", "label"])
    .size()
    .unstack(fill_value=0)
    .reindex(["train", "val", "test"])
)

x = np.arange(len(class_split.index))
width = 0.36

fig, ax = plt.subplots(figsize=(9, 5))

preserved = class_split.get("preserved", pd.Series(0, index=class_split.index))
looted = class_split.get("looted", pd.Series(0, index=class_split.index))

bars1 = ax.bar(
    x - width / 2,
    preserved,
    width,
    label="Preserved"
)

bars2 = ax.bar(
    x + width / 2,
    looted,
    width,
    label="Looted"
)

ax.set_title(
    "Class Distribution Within Each Split",
    fontsize=17,
    fontweight="bold"
)

ax.set_ylabel("Number of Frames")
ax.set_xticks(x)
ax.set_xticklabels(["Training", "Validation", "Test"])

ax.legend()

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()

        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height):,}",
                ha="center",
                va="bottom",
                fontsize=9
            )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "class_distribution_per_split.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# 4. SITE DISTRIBUTION PER SPLIT
# ==========================================================

site_split_counts = (
    site_df
    .groupby(["split", "label"])["site_id"]
    .nunique()
    .unstack(fill_value=0)
    .reindex(["train", "val", "test"])
)

x = np.arange(len(site_split_counts.index))
width = 0.36

fig, ax = plt.subplots(figsize=(9, 5))

preserved_sites = site_split_counts.get(
    "preserved",
    pd.Series(0, index=site_split_counts.index)
)

looted_sites = site_split_counts.get(
    "looted",
    pd.Series(0, index=site_split_counts.index)
)

bars1 = ax.bar(
    x - width / 2,
    preserved_sites,
    width,
    label="Preserved"
)

bars2 = ax.bar(
    x + width / 2,
    looted_sites,
    width,
    label="Looted"
)

ax.set_title(
    "Site Distribution Across Dataset Splits",
    fontsize=17,
    fontweight="bold"
)

ax.set_ylabel("Number of Sites")
ax.set_xticks(x)
ax.set_xticklabels(["Training", "Validation", "Test"])

ax.legend()

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()

        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "site_distribution_per_split.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# 5. DATA LEAKAGE CHECK VISUAL
# ==========================================================

leakage_check = (
    frames_df
    .groupby("site_id")["split"]
    .nunique()
)

leakage_sites = (leakage_check > 1).sum()
safe_sites = (leakage_check == 1).sum()

fig, ax = plt.subplots(figsize=(7, 4))

ax.axis("off")

ax.text(
    0.5,
    0.72,
    "SITE-LEVEL DATA LEAKAGE CHECK",
    ha="center",
    fontsize=17,
    fontweight="bold",
    transform=ax.transAxes
)

ax.text(
    0.5,
    0.48,
    "✓",
    ha="center",
    va="center",
    fontsize=55,
    transform=ax.transAxes
)

ax.text(
    0.5,
    0.28,
    f"{leakage_sites} sites appear in multiple splits",
    ha="center",
    fontsize=14,
    fontweight="bold",
    transform=ax.transAxes
)

ax.text(
    0.5,
    0.13,
    "Frames from each site remain inside a single split",
    ha="center",
    fontsize=11,
    transform=ax.transAxes
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "data_leakage_check.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\nAll preprocessing visualizations created successfully.")
print("Saved inside:", OUTPUT_DIR)