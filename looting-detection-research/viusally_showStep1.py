from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# PATHS
# ==========================================================

INVENTORY_CSV = Path("results/tables/step2_dataset_inventory.csv")
SITE_SUMMARY_CSV = Path("results/tables/step2_site_frame_summary.csv")

OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(INVENTORY_CSV)
site_df = pd.read_csv(SITE_SUMMARY_CSV)

# Main valid satellite frames
frames_df = df[
    (df["is_mask"] == False) &
    (df["is_corrupt"] == False)
].copy()


# ==========================================================
# 1. DATASET COMPOSITION
# ==========================================================

total_files = len(df)
valid_frames = len(frames_df)
mask_files = int(df["is_mask"].sum())
corrupted = int(df["is_corrupt"].sum())

labels = ["Valid Satellite Frames", "Mask Files", "Corrupted"]
values = [valid_frames, mask_files, corrupted]

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(labels, values)

ax.set_title(
    "DAFA-LS Dataset Audit",
    fontsize=18,
    fontweight="bold"
)

ax.set_ylabel("Number of Files")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# numbers above bars
for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        f"{value:,}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold"
    )

ax.text(
    0.5,
    0.93,
    f"Total image files audited: {total_files:,}",
    transform=ax.transAxes,
    ha="center",
    fontsize=12
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "dataset_composition.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# 2. SITE CLASS DISTRIBUTION
# ==========================================================

# Count unique sites per label
site_counts = (
    site_df
    .groupby("label")["site_id"]
    .nunique()
)

looted_sites = site_counts.get("looted", 0)
preserved_sites = site_counts.get("preserved", 0)

labels = ["Looted Sites", "Preserved Sites"]
values = [looted_sites, preserved_sites]

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(labels, values)

ax.set_title(
    "Detected Site Distribution",
    fontsize=18,
    fontweight="bold"
)

ax.set_ylabel("Number of Sites")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        f"{value}",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold"
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "site_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# 3. FRAME COUNT DISTRIBUTION PER SITE
# ==========================================================

fig, ax = plt.subplots(figsize=(9, 5))

ax.hist(
    site_df["frame_count"],
    bins=15,
    edgecolor="black"
)

mean_frames = site_df["frame_count"].mean()
median_frames = site_df["frame_count"].median()

ax.axvline(
    mean_frames,
    linestyle="--",
    linewidth=2,
    label=f"Mean = {mean_frames:.2f}"
)

ax.axvline(
    median_frames,
    linestyle=":",
    linewidth=2,
    label=f"Median = {median_frames:.0f}"
)

ax.set_title(
    "Frame Distribution Across Sites",
    fontsize=18,
    fontweight="bold"
)

ax.set_xlabel("Frames per Site")
ax.set_ylabel("Number of Sites")

ax.legend()

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "frames_per_site_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# 4. IMAGE RESOLUTION CHECK
# ==========================================================

resolution_counts = (
    frames_df
    .groupby(["width", "height"])
    .size()
    .reset_index(name="count")
)

resolution_counts["resolution"] = (
    resolution_counts["width"].astype(int).astype(str)
    + " × "
    + resolution_counts["height"].astype(int).astype(str)
)

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    resolution_counts["resolution"],
    resolution_counts["count"]
)

ax.set_title(
    "Image Resolution Validation",
    fontsize=18,
    fontweight="bold"
)

ax.set_xlabel("Resolution (pixels)")
ax.set_ylabel("Number of Valid Frames")

for bar, value in zip(bars, resolution_counts["count"]):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        f"{value:,}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold"
    )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "resolution_validation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# 5. DATASET PREPARATION FLOW DIAGRAM
# ==========================================================

fig, ax = plt.subplots(figsize=(16, 4.5))

ax.axis("off")

steps = [
    ("Raw DAFA-LS\nDataset", "64,114\nimage files"),
    ("Separate\nFiles", "63,439 frames\n675 masks"),
    ("Integrity\nCheck", "0 corrupted\nimages"),
    ("Resolution\nValidation", "266 × 266\npixels"),
    ("Metadata\nExtraction", "Site • Date • Path\nFrame metadata")
]

x_positions = [0.08, 0.29, 0.50, 0.71, 0.92]

for i, ((title, subtitle), x) in enumerate(zip(steps, x_positions)):

    ax.text(
        x,
        0.58,
        title,
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        transform=ax.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="white",
            edgecolor="black",
            linewidth=1.5
        )
    )

    ax.text(
        x,
        0.22,
        subtitle,
        ha="center",
        va="center",
        fontsize=11,
        transform=ax.transAxes
    )

    # arrows
    if i < len(steps) - 1:
        ax.annotate(
            "",
            xy=(x_positions[i+1] - 0.07, 0.58),
            xytext=(x + 0.07, 0.58),
            xycoords=ax.transAxes,
            arrowprops=dict(
                arrowstyle="->",
                linewidth=2
            )
        )

ax.set_title(
    "DAFA-LS Dataset Preparation Pipeline",
    fontsize=20,
    fontweight="bold",
    pad=20
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "dataset_preparation_pipeline.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\nVisualizations created successfully:")
print(OUTPUT_DIR / "dataset_composition.png")
print(OUTPUT_DIR / "site_distribution.png")
print(OUTPUT_DIR / "frames_per_site_distribution.png")
print(OUTPUT_DIR / "resolution_validation.png")
print(OUTPUT_DIR / "dataset_preparation_pipeline.png")