"""Audit DAFA-LS before preparing model inputs.

Run from your project folder:
    python step2_dataset_audit.py

Required packages: pandas, Pillow. tqdm is optional (a progress bar).
PyTorch is optional and is used ONLY to check CUDA availability.
Pillow image checks and pandas tables still run on the CPU.
To check CUDA without scanning the dataset:
    python step2_dataset_audit.py --cuda-check-only

Expected layout: data/raw/DAFA_LS/{looted,preserved}/{site_id}/images...
Keep site_id as text. Use site_uid (for example, looted/001) in Step 03
for all site grouping, split assignments, joins, and overlap checks.
This script reads images and writes audit tables; it does not change images.
"""

from pathlib import Path
from datetime import date
import argparse
import json
import re
import statistics

import pandas as pd
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(items, **kwargs):
        return items


DATA_ROOT = Path("data/raw/DAFA_LS")
OUTPUT_DIR = Path("results/tables")
LABELS = {"looted", "preserved"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# These are reference values, not forced targets for the local dataset.
EXPECTED_TOTAL_IMAGES = 55480
EXPECTED_TOTAL_SITES = 675
EXPECTED_LOOTED_SITES = 135
EXPECTED_PRESERVED_SITES = 540
EXPECTED_IMAGE_SIZE = (266, 266)

# More specific date formats are checked first. Calendar validity is checked below.
DATE_PATTERNS = [
    re.compile(r"(?<!\d)(20\d{2})[-_](\d{2})[-_](\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(20\d{2})[-_](\d{2})(?!\d|[-_]\d)"),
    re.compile(r"(?<!\d)(20\d{2})(\d{2})(?!\d|[-_]\d)"),
    re.compile(r"(?<!\d)(20\d{2})(?!\d|[-_]\d)"),
]


def check_cuda():
    """Report GPU availability without pretending the image audit uses CUDA."""
    lines = ["Audit processing device: CPU (Pillow and pandas)"]
    try:
        import torch
    except ImportError:
        lines.append("CUDA check: unavailable because PyTorch is not installed.")
        lines.append("The dataset audit can still run without PyTorch.")
        return lines
    except Exception as exc:
        lines.append(f"CUDA check: PyTorch could not be loaded: {exc}")
        lines.append("The dataset audit can still run on the CPU.")
        return lines

    lines.append(f"PyTorch version: {torch.__version__}")
    lines.append(f"PyTorch CUDA build: {torch.version.cuda or 'none'}")
    try:
        available = torch.cuda.is_available()
        lines.append(f"CUDA available to PyTorch: {'yes' if available else 'no'}")
        if available:
            for index in range(torch.cuda.device_count()):
                lines.append(f"GPU {index}: {torch.cuda.get_device_name(index)}")
            lines.append("CUDA is available for later model training. This audit still runs on CPU.")
        else:
            lines.append("A CUDA-capable GPU, working driver, and CUDA-enabled PyTorch are needed for CUDA training.")
    except Exception as exc:
        lines.append(f"CUDA device check failed: {exc}")
        lines.append("The dataset audit can still run on the CPU.")
    return lines


def detect_site_identity(path, data_root):
    """Keep the folder ID, but qualify it with the class to avoid collisions."""
    parts = path.relative_to(data_root).parts
    label = parts[0].lower() if parts and parts[0].lower() in LABELS else "unknown"
    site_id = parts[1] if len(parts) >= 3 else None
    valid = label in LABELS and site_id is not None
    return {
        "label": label,
        "site_id": site_id,
        "site_uid": f"{label}/{site_id}" if valid else None,
        "site_folder": "/".join(parts[:2]) if len(parts) >= 3 else None,
        "is_valid_site": valid,
    }


def detect_is_mask(path, data_root):
    """Keep the existing naming rule, without matching unrelated parent folders."""
    return "mask" in path.relative_to(data_root).as_posix().lower()


def parse_date_from_filename(filename):
    """Return year, month, day and a status; never accept an impossible date.

    Supports YYYY_MM_DD, YYYY-MM-DD, YYYYMMDD, YYYY_MM, YYYY-MM,
    YYYYMM, and a year alone. A missing day stays missing.
    Filenames are still the source: this does not verify acquisition metadata.
    """
    for pattern in DATE_PATTERNS:
        match = pattern.search(Path(filename).stem)
        if match is None:
            continue
        numbers = [int(value) for value in match.groups()]
        year = numbers[0]
        month = numbers[1] if len(numbers) >= 2 else None
        day = numbers[2] if len(numbers) >= 3 else None
        try:
            if month is not None:
                date(year, month, day if day is not None else 1)
        except ValueError:
            # Do not reinterpret an invalid full date as a valid month/year.
            return None, None, None, "invalid"
        status = "full_date" if day is not None else "year_month" if month is not None else "year_only"
        return year, month, day, status
    return None, None, None, "missing"


def check_image(path):
    file_size = None
    try:
        file_size = path.stat().st_size
        with Image.open(path) as img:
            img.verify()
        # verify() requires reopening. load() also checks actual pixel decoding.
        with Image.open(path) as img:
            img.load()
            width, height = img.size
            mode = img.mode
            channels = len(img.getbands())
        return {
            "file_size_bytes": file_size, "is_corrupt": False,
            "width": width, "height": height, "mode": mode,
            "channels": channels, "error": "",
        }
    except Exception as exc:
        return {
            "file_size_bytes": file_size, "is_corrupt": True,
            "width": None, "height": None, "mode": None,
            "channels": None, "error": str(exc),
        }


def summarize_sites(frames):
    """One row per qualified site. unique_months counts distinct YYYY-MM dates."""
    summary = (
        frames.groupby(["site_uid", "label", "site_id"], sort=True)
        .agg(
            frame_count=("relative_path", "count"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            unique_years=("year", "nunique"),
            unique_months=("year_month", "nunique"),
            unique_month_numbers=("month", "nunique"),
            dated_frame_count=("year_month", "count"),
            width=("width", "first"),
            height=("height", "first"),
            channels=("channels", "first"),
        )
        .reset_index()
    )
    summary["missing_year_month_frames"] = summary["frame_count"] - summary["dated_frame_count"]
    summary["extra_frames_in_same_month"] = summary["dated_frame_count"] - summary["unique_months"]
    return summary


def main(data_root=DATA_ROOT, output_dir=OUTPUT_DIR):
    data_root, output_dir = Path(data_root), Path(output_dir)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {data_root}")

    cuda_info = check_cuda()
    for line in cuda_info:
        print(line)

    image_files = sorted(
        (p for p in data_root.rglob("*")
         if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.relative_to(data_root).as_posix(),
    )
    if not image_files:
        raise ValueError(f"No supported image files found under {data_root}")

    important_files = {name: (data_root / name).is_file()
                       for name in ["fold_dict.json", "README.txt", "LICENSE.txt"]}
    fold_status, fold_keys = "missing", []
    if important_files["fold_dict.json"]:
        try:
            with (data_root / "fold_dict.json").open("r", encoding="utf-8") as handle:
                fold_dict = json.load(handle)
            if not isinstance(fold_dict, dict):
                raise ValueError("Expected a JSON object at the top level")
            fold_status, fold_keys = "readable", list(fold_dict.keys())
        except (OSError, ValueError) as exc:
            fold_status = f"error: {exc}"

    print(f"Dataset root: {data_root}")
    print(f"Total image files found: {len(image_files)}")
    records = []
    for path in tqdm(image_files, desc="Checking images", unit="file"):
        year, month, day, date_status = parse_date_from_filename(path.name)
        record = {
            "relative_path": path.relative_to(data_root).as_posix(),
            "file_name": path.name,
            **detect_site_identity(path, data_root),
            "is_mask": detect_is_mask(path, data_root),
            "year": year, "month": month, "day": day,
            "year_month": f"{year:04d}-{month:02d}" if month is not None else None,
            "date_status": date_status,
            "extension": path.suffix.lower(),
            **check_image(path),
        }
        records.append(record)

    inventory = pd.DataFrame(records)
    # String IDs preserve leading zeros. Step 03 must read these columns as text too.
    for column in ["site_id", "site_uid", "site_folder"]:
        inventory[column] = inventory[column].astype("string")
    for column in ["year", "month", "day", "width", "height", "channels", "file_size_bytes"]:
        inventory[column] = inventory[column].astype("Int64")
    if inventory["relative_path"].duplicated().any():
        raise RuntimeError("Duplicate paths found while scanning. Audit outputs were not written.")

    readable_non_mask = inventory[~inventory["is_corrupt"] & ~inventory["is_mask"]].copy()
    # Retain invalid paths in the inventory, but exclude them from site statistics.
    frames = readable_non_mask[readable_non_mask["is_valid_site"]].copy()
    site_summary = summarize_sites(frames)

    total_sites = int(site_summary["site_uid"].nunique())
    looted_sites = int(site_summary.loc[site_summary["label"] == "looted", "site_uid"].nunique())
    preserved_sites = int(site_summary.loc[site_summary["label"] == "preserved", "site_uid"].nunique())
    bare_id_classes = site_summary.groupby("site_id")["label"].nunique()
    reused_ids = int((bare_id_classes > 1).sum())
    folder_aliases = int((frames.groupby("site_uid")["site_folder"].nunique() > 1).sum())
    counts = site_summary["frame_count"].tolist()
    min_frames = min(counts) if counts else 0
    max_frames = max(counts) if counts else 0
    mean_frames = round(statistics.mean(counts), 2) if counts else 0
    median_frames = statistics.median(counts) if counts else 0

    corrupt_count = int(inventory["is_corrupt"].sum())
    mask_count = int(inventory["is_mask"].sum())
    invalid_site_count = int((~readable_non_mask["is_valid_site"]).sum())
    missing_dates = int(frames["year_month"].isna().sum())
    invalid_dates = int((frames["date_status"] == "invalid").sum())
    wrong_size = int(((frames["width"] != EXPECTED_IMAGE_SIZE[0]) |
                      (frames["height"] != EXPECTED_IMAGE_SIZE[1])).sum())
    non_rgb = int(((frames["mode"] != "RGB") | (frames["channels"] != 3)).sum())
    repeated_months = int(site_summary["extra_frames_in_same_month"].sum())
    size_summary = frames.groupby(["width", "height"]).size().sort_values(ascending=False)

    warnings = []
    if len(frames) != EXPECTED_TOTAL_IMAGES:
        warnings.append(
            f"Found {len(frames)} usable frames; the reference count is {EXPECTED_TOTAL_IMAGES}. "
            "Investigate dataset version, dates, and file contents. Do not delete files just to match the reference."
        )
    for name, actual, expected in [
        ("total sites", total_sites, EXPECTED_TOTAL_SITES),
        ("looted sites", looted_sites, EXPECTED_LOOTED_SITES),
        ("preserved sites", preserved_sites, EXPECTED_PRESERVED_SITES),
    ]:
        if actual != expected:
            warnings.append(f"Detected {actual} {name}; expected {expected}.")
    if not len(frames):
        warnings.append("No readable, non-mask frames with recognised site folders were found.")
    if corrupt_count:
        warnings.append(f"{corrupt_count} files failed reading or pixel decoding; inspect their inventory error messages.")
    if invalid_site_count:
        warnings.append(f"{invalid_site_count} readable non-mask files have unrecognised class/site paths and were excluded from site statistics.")
    if folder_aliases:
        warnings.append(f"{folder_aliases} site UIDs map to multiple folder spellings. Resolve these ambiguous folders.")
    if missing_dates:
        warnings.append(f"{missing_dates} retained frames have no usable year-month date, including {invalid_dates} invalid dates. Check before temporal selection.")
    if wrong_size:
        warnings.append(f"{wrong_size} retained frames are not 266 x 266. Review preprocessing; this audit does not resize them.")
    if non_rgb:
        warnings.append(f"{non_rgb} retained frames are not three-channel RGB. This audit does not convert them.")
    if repeated_months:
        warnings.append(f"{repeated_months} extra frames share a site and year-month with another frame. Check their dates and contents; they are not necessarily duplicate images.")
    for name, found in important_files.items():
        if not found:
            warnings.append(f"Important file missing: {name}.")
    if fold_status.startswith("error:"):
        warnings.append(f"fold_dict.json could not be read as a JSON object: {fold_status}")

    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "step2_dataset_inventory.csv"
    summary_path = output_dir / "step2_site_frame_summary.csv"
    report_path = output_dir / "step2_dataset_audit_report.md"

    report = [
        "# Step 2 Dataset Audit Report", "", "## Dataset Path", "", f"`{data_root}`", "",
        "## CUDA Availability and Audit Device", "",
        *[f"- {line}" for line in cuda_info], "",
        "## Important Dataset Files", "",
        *[f"- {name}: {'FOUND' if found else 'MISSING'}" for name, found in important_files.items()],
        f"- fold_dict.json status: {fold_status}", f"- fold_dict.json keys: {fold_keys}", "",
        "## Overall Image Summary", "",
        f"- Total image files found: {len(inventory)}",
        f"- Readable non-mask files: {len(readable_non_mask)}",
        f"- Valid non-mask frames with recognised site folders: {len(frames)}",
        f"- Mask files detected by name: {mask_count}",
        f"- Corrupted/unreadable files: {corrupt_count}",
        f"- Readable non-mask files excluded for invalid site paths: {invalid_site_count}", "",
        "## Site Summary", "",
        f"- Unique sites detected using site_uid: {total_sites}",
        f"- Looted sites detected: {looted_sites}",
        f"- Preserved sites detected: {preserved_sites}",
        f"- Bare site IDs reused across classes: {reused_ids}",
        "- A site_uid includes the class folder: looted/0 and preserved/0 are separate identifiers.",
        "- These are folder identities; confirm their correspondence to physical sites using dataset metadata.", "",
        "## Frame Count Per Site", "",
        f"- Minimum frames per site: {min_frames}", f"- Maximum frames per site: {max_frames}",
        f"- Mean frames per site: {mean_frames}", f"- Median frames per site: {median_frames}", "",
        "## Temporal Coverage", "",
        "- unique_months counts distinct year-month dates, such as 2016-01 and 2017-01.",
        "- unique_month_numbers counts different month numbers (1 to 12).",
        f"- Retained frames without a usable year-month: {missing_dates}",
        f"- Retained frames with an invalid calendar date: {invalid_dates}",
        f"- Extra frames within an already represented site/month: {repeated_months}", "",
        "## Image Size Summary", "",
    ]
    report.extend(f"- {int(width)} x {int(height)}: {int(count)} images"
                  for (width, height), count in size_summary.items())
    if size_summary.empty:
        report.append("- No usable frame sizes found.")
    report.extend([
        f"- Retained frames that are not three-channel RGB: {non_rgb}", "",
        "## Expected Reference Values", "",
        f"- Expected images: {EXPECTED_TOTAL_IMAGES}", f"- Expected sites: {EXPECTED_TOTAL_SITES}",
        f"- Expected looted sites: {EXPECTED_LOOTED_SITES}",
        f"- Expected preserved sites: {EXPECTED_PRESERVED_SITES}",
        f"- Expected image size: {EXPECTED_IMAGE_SIZE[0]} x {EXPECTED_IMAGE_SIZE[1]}", "",
        "## Warnings", "",
    ])
    report.extend([f"- WARNING: {message}" for message in warnings] or ["- No warnings from the implemented checks."])
    report.extend([
        "", "## Scope and Next Step", "",
        "- Image files are read only. Masks are recognised by 'mask' in the dataset-relative path.",
        "- No image-content deduplication, label verification, or cloud-quality assessment is performed.",
        "- This step does not choose train/validation/test splits or validate fold membership.",
        "- Update Step 03 to use site_uid throughout. The old Step 03 still has split and merge bugs.",
        '- When reading the inventory, use pd.read_csv(path, dtype={"site_id": "string", "site_uid": "string"}).',
        "", "## Created Output Files", "",
        f"- `{inventory_path}`", f"- `{summary_path}`", f"- `{report_path}`",
    ])

    inventory.to_csv(inventory_path, index=False)
    site_summary.to_csv(summary_path, index=False)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nStep 2 completed: {total_sites} sites, {len(frames)} usable frames.")
    print(f"Inventory CSV saved to: {inventory_path}")
    print(f"Site summary CSV saved to: {summary_path}")
    print(f"Audit report saved to: {report_path}")
    if warnings:
        print(f"Review the report: {len(warnings)} warning(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit DAFA-LS and preserve unique site identities.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cuda-check-only", action="store_true",
                        help="Report CUDA availability without opening the dataset.")
    args = parser.parse_args()
    if args.cuda_check_only:
        for line in check_cuda():
            print(line)
    else:
        main(args.data_root, args.output_dir)
