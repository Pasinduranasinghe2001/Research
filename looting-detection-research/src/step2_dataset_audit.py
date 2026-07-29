from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm
import json
import re
import statistics

# ==================================================
# CORRECT DATASET PATH FOR YOUR STRUCTURE
# ==================================================

DATA_ROOT = Path("data/raw/DAFA_LS")
OUTPUT_DIR = Path("results/tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

EXPECTED_TOTAL_IMAGES = 55480
EXPECTED_TOTAL_SITES = 675
EXPECTED_LOOTED_SITES = 135
EXPECTED_PRESERVED_SITES = 540
EXPECTED_IMAGE_SIZE = (266, 266)


def detect_label(path: Path):
    parts = [p.lower() for p in path.parts]

    if "looted" in parts:
        return "looted"

    if "preserved" in parts:
        return "preserved"

    return "unknown"


def detect_site_id(path: Path):
    """
    Detects site ID from folder structure.
    Expected style:
    DAFA_LS/looted/site_id/images...
    DAFA_LS/preserved/site_id/images...
    """
    relative_parts = path.relative_to(DATA_ROOT).parts

    if len(relative_parts) >= 3:
        return relative_parts[1]

    return "unknown_site"


def detect_is_mask(path: Path):
    text = str(path).lower()
    return "mask" in text


def parse_date_from_filename(filename: str):
    """
    Extract year/month from filenames where possible.
    Supports:
    2016_01, 2016-01, 201601, 2016_01_15, 2016-01-15
    """
    name = filename.lower()

    patterns = [
        r"(20[0-9]{2})[-_](0[1-9]|1[0-2])[-_](0[1-9]|[12][0-9]|3[01])",
        r"(20[0-9]{2})(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])",
        r"(20[0-9]{2})[-_](0[1-9]|1[0-2])",
        r"(20[0-9]{2})(0[1-9]|1[0-2])",
        r"(20[0-9]{2})",
    ]

    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            groups = match.groups()
            year = int(groups[0])
            month = int(groups[1]) if len(groups) >= 2 else None
            day = int(groups[2]) if len(groups) >= 3 else None
            return year, month, day

    return None, None, None


def check_image(path: Path):
    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
            channels = len(img.getbands())

        return {
            "is_corrupt": False,
            "width": width,
            "height": height,
            "mode": mode,
            "channels": channels,
            "error": ""
        }

    except Exception as e:
        return {
            "is_corrupt": True,
            "width": None,
            "height": None,
            "mode": None,
            "channels": None,
            "error": str(e)
        }


def main():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_ROOT}")

    print(f"Dataset root: {DATA_ROOT}")

    # Check important files
    important_files = ["fold_dict.json", "README.txt", "LICENSE.txt"]
    important_file_status = {}

    for file_name in important_files:
        file_path = DATA_ROOT / file_name
        important_file_status[file_name] = file_path.exists()

    # Try reading fold_dict.json
    fold_dict_status = "not_checked"
    fold_dict_keys = []

    fold_path = DATA_ROOT / "fold_dict.json"
    if fold_path.exists():
        try:
            with open(fold_path, "r", encoding="utf-8") as f:
                fold_dict = json.load(f)
            fold_dict_status = "readable"
            fold_dict_keys = list(fold_dict.keys())
        except Exception as e:
            fold_dict_status = f"error: {e}"

    # Collect image files
    image_files = [
        p for p in DATA_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"Total image files found: {len(image_files)}")
    print("Checking images...")

    records = []

    for path in tqdm(image_files):
        image_info = check_image(path)
        label = detect_label(path)
        site_id = detect_site_id(path)
        is_mask = detect_is_mask(path)
        year, month, day = parse_date_from_filename(path.name)

        records.append({
            "relative_path": str(path.relative_to(DATA_ROOT)),
            "file_name": path.name,
            "label": label,
            "site_id": site_id,
            "is_mask": is_mask,
            "year": year,
            "month": month,
            "day": day,
            "extension": path.suffix.lower(),
            "file_size_bytes": path.stat().st_size,
            "is_corrupt": image_info["is_corrupt"],
            "width": image_info["width"],
            "height": image_info["height"],
            "mode": image_info["mode"],
            "channels": image_info["channels"],
            "error": image_info["error"]
        })

    df = pd.DataFrame(records)

    inventory_path = OUTPUT_DIR / "step2_dataset_inventory.csv"
    df.to_csv(inventory_path, index=False)

    # Main satellite frames only
    frames_df = df[
        (df["is_mask"] == False) &
        (df["is_corrupt"] == False)
    ].copy()

    site_summary = (
        frames_df.groupby(["site_id", "label"])
        .agg(
            frame_count=("relative_path", "count"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            unique_years=("year", "nunique"),
            unique_months=("month", "nunique"),
            width=("width", "first"),
            height=("height", "first"),
            channels=("channels", "first")
        )
        .reset_index()
    )

    site_summary_path = OUTPUT_DIR / "step2_site_frame_summary.csv"
    site_summary.to_csv(site_summary_path, index=False)

    # Summary values
    total_image_files = len(df)
    total_valid_frames = len(frames_df)
    total_corrupt = int(df["is_corrupt"].sum())
    total_masks = int(df["is_mask"].sum())

    unique_sites = frames_df["site_id"].nunique()
    looted_sites = site_summary[site_summary["label"] == "looted"]["site_id"].nunique()
    preserved_sites = site_summary[site_summary["label"] == "preserved"]["site_id"].nunique()
    unknown_sites = site_summary[site_summary["label"] == "unknown"]["site_id"].nunique()

    frame_counts = site_summary["frame_count"].tolist()

    if frame_counts:
        min_frames = min(frame_counts)
        max_frames = max(frame_counts)
        mean_frames = round(statistics.mean(frame_counts), 2)
        median_frames = statistics.median(frame_counts)
    else:
        min_frames = max_frames = mean_frames = median_frames = 0

    size_summary = (
        frames_df.groupby(["width", "height"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    # Create markdown report
    report = []

    report.append("# Step 2 Dataset Audit Report\n")

    report.append("## Dataset Path\n")
    report.append(f"`{DATA_ROOT}`\n")

    report.append("## Important Dataset Files\n")
    for file_name, exists in important_file_status.items():
        report.append(f"- {file_name}: {'FOUND' if exists else 'MISSING'}")
    report.append(f"- fold_dict.json status: {fold_dict_status}")
    report.append(f"- fold_dict.json keys: {fold_dict_keys}\n")

    report.append("## Overall Image Summary\n")
    report.append(f"- Total image files found: {total_image_files}")
    report.append(f"- Valid non-mask frames: {total_valid_frames}")
    report.append(f"- Mask files detected: {total_masks}")
    report.append(f"- Corrupted/unreadable files: {total_corrupt}\n")

    report.append("## Site Summary\n")
    report.append(f"- Unique sites detected: {unique_sites}")
    report.append(f"- Looted sites detected: {looted_sites}")
    report.append(f"- Preserved sites detected: {preserved_sites}")
    report.append(f"- Unknown-label sites detected: {unknown_sites}\n")

    report.append("## Frame Count Per Site\n")
    report.append(f"- Minimum frames per site: {min_frames}")
    report.append(f"- Maximum frames per site: {max_frames}")
    report.append(f"- Mean frames per site: {mean_frames}")
    report.append(f"- Median frames per site: {median_frames}\n")

    report.append("## Image Size Summary\n")
    if not size_summary.empty:
        for _, row in size_summary.head(10).iterrows():
            report.append(
                f"- {int(row['width'])} x {int(row['height'])}: {int(row['count'])} images"
            )
    else:
        report.append("- No valid image sizes detected.")

    report.append("\n## Expected Full DAFA-LS Values\n")
    report.append(f"- Expected images: {EXPECTED_TOTAL_IMAGES}")
    report.append(f"- Expected sites: {EXPECTED_TOTAL_SITES}")
    report.append(f"- Expected looted sites: {EXPECTED_LOOTED_SITES}")
    report.append(f"- Expected preserved sites: {EXPECTED_PRESERVED_SITES}")
    report.append(f"- Expected image size: {EXPECTED_IMAGE_SIZE[0]} x {EXPECTED_IMAGE_SIZE[1]}\n")

    report.append("## Warnings\n")

    warnings = []

    if total_valid_frames != EXPECTED_TOTAL_IMAGES:
        warnings.append(
            f"Valid frame count is {total_valid_frames}, expected {EXPECTED_TOTAL_IMAGES}. "
            "Check whether masks are separated, whether extra files exist, or whether the dataset version differs."
        )

    if unique_sites != EXPECTED_TOTAL_SITES:
        warnings.append(
            f"Detected site count is {unique_sites}, expected {EXPECTED_TOTAL_SITES}. "
            "Check folder depth and site ID detection."
        )

    if looted_sites != EXPECTED_LOOTED_SITES:
        warnings.append(
            f"Detected looted sites is {looted_sites}, expected {EXPECTED_LOOTED_SITES}."
        )

    if preserved_sites != EXPECTED_PRESERVED_SITES:
        warnings.append(
            f"Detected preserved sites is {preserved_sites}, expected {EXPECTED_PRESERVED_SITES}."
        )

    if total_corrupt > 0:
        warnings.append(
            f"{total_corrupt} corrupted images found. These must be removed or skipped."
        )

    wrong_size = frames_df[
        (frames_df["width"] != EXPECTED_IMAGE_SIZE[0]) |
        (frames_df["height"] != EXPECTED_IMAGE_SIZE[1])
    ]

    if len(wrong_size) > 0:
        warnings.append(
            f"{len(wrong_size)} images are not {EXPECTED_IMAGE_SIZE[0]} x {EXPECTED_IMAGE_SIZE[1]}. "
            "Resize them during preprocessing."
        )

    if unknown_sites > 0:
        warnings.append(
            f"{unknown_sites} sites have unknown labels. Check folder names."
        )

    if not warnings:
        report.append("- No major warnings found.")
    else:
        for warning in warnings:
            report.append(f"- WARNING: {warning}")

    report.append("\n## Created Output Files\n")
    report.append(f"- `{inventory_path}`")
    report.append(f"- `{site_summary_path}`")

    report_path = OUTPUT_DIR / "step2_dataset_audit_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\nStep 2 completed.")
    print(f"Inventory CSV saved to: {inventory_path}")
    print(f"Site summary CSV saved to: {site_summary_path}")
    print(f"Audit report saved to: {report_path}")


if __name__ == "__main__":
    main()