from pathlib import Path
import json
import random
import re
import statistics
import pandas as pd

# ==================================================
# PATH CONFIGURATION
# ==================================================

DATA_ROOT = Path("data/raw/DAFA_LS")
STEP2_INVENTORY = Path("results/tables/step2_dataset_inventory.csv")

PROCESSED_DIR = Path("data/processed")
SPLIT_DIR = Path("data/splits")
REPORT_DIR = Path("results/tables")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FOLD_DICT_PATH = DATA_ROOT / "fold_dict.json"

SEED = 42
random.seed(SEED)

EXPECTED_IMAGE_SIZE = (266, 266)
EXPECTED_TOTAL_IMAGES = 55480
EXPECTED_TOTAL_SITES = 675
EXPECTED_LOOTED_SITES = 135
EXPECTED_PRESERVED_SITES = 540


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def clean_text(x):
    return str(x).strip()

def norm_key(x):
    return (
        str(x)
        .strip()
        .lower()
        .replace("\\", "/")
        .replace(" ", "")
    )

def infer_split_name(tokens):
    """
    Infer split name from fold_dict path tokens.
    """
    joined = "/".join([str(t).lower() for t in tokens])

    if "test" in joined:
        return "test"

    if "val" in joined or "valid" in joined or "validation" in joined:
        return "val"

    if "train" in joined:
        return "train"

    return "unknown"

def extract_site_from_value(value, known_site_lookup):
    """
    Try to convert a fold_dict value into a known site_id.
    Handles direct site IDs and file paths.
    """
    raw = clean_text(value)
    nk = norm_key(raw)

    if nk in known_site_lookup:
        return known_site_lookup[nk]

    # If it is a path such as looted/site_id/...
    parts = re.split(r"[\\/]", raw)

    for i, part in enumerate(parts):
        p = part.lower()
        if p in {"looted", "preserved"} and i + 1 < len(parts):
            candidate = parts[i + 1]
            if norm_key(candidate) in known_site_lookup:
                return known_site_lookup[norm_key(candidate)]

    # Try filename stem
    stem = Path(raw).stem
    if norm_key(stem) in known_site_lookup:
        return known_site_lookup[norm_key(stem)]

    return None

def walk_fold_dict(obj, path_tokens, known_site_lookup, rows):
    """
    Recursively parse fold_dict.json.
    This supports many possible JSON structures.
    """
    if isinstance(obj, dict):
        # Case: {"site_id": "train"} or {"site_id": 0}
        for key, value in obj.items():
            key_site = extract_site_from_value(key, known_site_lookup)

            if key_site is not None and not isinstance(value, (list, dict)):
                value_text = str(value).lower()
                if "test" in value_text:
                    split = "test"
                elif "val" in value_text or "valid" in value_text:
                    split = "val"
                elif "train" in value_text:
                    split = "train"
                else:
                    split = f"fold_{value_text}"

                rows.append({
                    "site_id": key_site,
                    "split": split,
                    "fold_source": "/".join(map(str, path_tokens)),
                    "raw_value": str(value)
                })

            else:
                walk_fold_dict(value, path_tokens + [key], known_site_lookup, rows)

    elif isinstance(obj, list):
        split = infer_split_name(path_tokens)
        fold_source = "/".join(map(str, path_tokens))

        for item in obj:
            if isinstance(item, (list, dict)):
                walk_fold_dict(item, path_tokens, known_site_lookup, rows)
            else:
                site = extract_site_from_value(item, known_site_lookup)
                if site is not None:
                    rows.append({
                        "site_id": site,
                        "split": split,
                        "fold_source": fold_source,
                        "raw_value": str(item)
                    })

def create_fallback_site_split(site_df):
    """
    Fallback site-wise split only if fold_dict cannot be parsed.
    It keeps sites separated and maintains class balance as much as possible.
    """
    train_rows = []
    val_rows = []
    test_rows = []

    for label in ["looted", "preserved"]:
        label_df = site_df[site_df["label"] == label].copy()
        ids = label_df["site_id"].tolist()
        random.shuffle(ids)

        n = len(ids)
        n_test = max(1, round(n * 0.10))
        n_val = max(1, round(n * 0.10))

        test_ids = ids[:n_test]
        val_ids = ids[n_test:n_test + n_val]
        train_ids = ids[n_test + n_val:]

        train_rows.extend([(site_id, "train", "fallback_sitewise_80_10_10") for site_id in train_ids])
        val_rows.extend([(site_id, "val", "fallback_sitewise_80_10_10") for site_id in val_ids])
        test_rows.extend([(site_id, "test", "fallback_sitewise_80_10_10") for site_id in test_ids])

    rows = train_rows + val_rows + test_rows

    return pd.DataFrame(rows, columns=["site_id", "split", "fold_source"])

def choose_final_split_from_fold_rows(fold_rows_df, site_df):
    """
    Convert parsed fold_dict rows into one practical train/val/test split.

    Priority:
    1. Use official test sites if available.
    2. Use the first available validation fold for validation.
    3. All remaining sites become train.
    """
    all_sites = set(site_df["site_id"].tolist())

    official_test = set(
        fold_rows_df[fold_rows_df["split"] == "test"]["site_id"].tolist()
    )

    val_candidates = fold_rows_df[fold_rows_df["split"] == "val"].copy()

    official_val = set()

    if len(val_candidates) > 0:
        first_val_source = sorted(val_candidates["fold_source"].unique())[0]
        official_val = set(
            val_candidates[val_candidates["fold_source"] == first_val_source]["site_id"].tolist()
        )

    train_sites = all_sites - official_test - official_val

    rows = []

    for site in sorted(train_sites):
        rows.append({
            "site_id": site,
            "split": "train",
            "fold_source": "official_fold_dict_remaining_train"
        })

    for site in sorted(official_val):
        rows.append({
            "site_id": site,
            "split": "val",
            "fold_source": "official_fold_dict_first_validation_fold"
        })

    for site in sorted(official_test):
        rows.append({
            "site_id": site,
            "split": "test",
            "fold_source": "official_fold_dict_test"
        })

    return pd.DataFrame(rows)


# ==================================================
# MAIN SCRIPT
# ==================================================

def main():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_ROOT}")

    if not STEP2_INVENTORY.exists():
        raise FileNotFoundError(
            f"Step 2 inventory not found: {STEP2_INVENTORY}\n"
            "Run Step 2 first before Step 3."
        )

    inventory = pd.read_csv(STEP2_INVENTORY)

    # Keep only valid satellite images, not masks
    frames = inventory[
        (inventory["is_corrupt"] == False) &
        (inventory["is_mask"] == False) &
        (inventory["label"].isin(["looted", "preserved"]))
    ].copy()

    # Add full path
    frames["full_path"] = frames["relative_path"].apply(
        lambda p: str(DATA_ROOT / p)
    )

    # Make sure year/month columns exist
    for col in ["year", "month", "day"]:
        if col not in frames.columns:
            frames[col] = None

    # Sort temporal order
    frames["year_sort"] = frames["year"].fillna(9999)
    frames["month_sort"] = frames["month"].fillna(99)
    frames["day_sort"] = frames["day"].fillna(99)

    frames = frames.sort_values(
        by=["label", "site_id", "year_sort", "month_sort", "day_sort", "file_name"]
    ).reset_index(drop=True)

    # Add frame index inside each site
    frames["frame_index"] = frames.groupby("site_id").cumcount()

    metadata_all_frames_path = PROCESSED_DIR / "metadata_all_frames.csv"
    frames.to_csv(metadata_all_frames_path, index=False)

    # Site-level metadata
    site_df = (
        frames.groupby(["site_id", "label"])
        .agg(
            frame_count=("relative_path", "count"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            width=("width", "first"),
            height=("height", "first"),
            channels=("channels", "first")
        )
        .reset_index()
    )

    metadata_sites_path = PROCESSED_DIR / "metadata_sites.csv"
    site_df.to_csv(metadata_sites_path, index=False)

    # Create preprocessing config
    preprocessing_config = {
        "dataset_name": "DAFA-LS",
        "task_type": "binary_classification",
        "label_mapping": {
            "preserved": 0,
            "looted": 1
        },
        "image_channels": "RGB",
        "expected_image_size": [266, 266],
        "normalization": {
            "method": "scale_pixels_to_0_1",
            "mean_std": "compute_from_training_set_in_next_step_or_use_ImageNet_for_pretrained_CNN"
        },
        "temporal_order": "sorted_by_year_month_day_filename",
        "split_type": "site_wise",
        "data_leakage_rule": "frames_from_same_site_must_not_appear_in_multiple_splits"
    }

    preprocessing_config_path = PROCESSED_DIR / "preprocessing_config.json"
    with open(preprocessing_config_path, "w", encoding="utf-8") as f:
        json.dump(preprocessing_config, f, indent=4)

    # Parse fold_dict.json if possible
    known_site_lookup = {
        norm_key(site_id): site_id for site_id in site_df["site_id"].tolist()
    }

    fold_rows = []
    fold_status = "not_found"
    split_method = "none"

    if FOLD_DICT_PATH.exists():
        try:
            with open(FOLD_DICT_PATH, "r", encoding="utf-8") as f:
                fold_dict = json.load(f)

            walk_fold_dict(fold_dict, ["fold_dict"], known_site_lookup, fold_rows)
            fold_status = f"readable_parsed_rows_{len(fold_rows)}"

        except Exception as e:
            fold_status = f"read_error_{e}"

    fold_rows_df = pd.DataFrame(fold_rows)

    official_long_path = SPLIT_DIR / "site_splits_official_long.csv"

    if len(fold_rows_df) > 0:
        fold_rows_df.drop_duplicates().to_csv(official_long_path, index=False)
        site_splits = choose_final_split_from_fold_rows(fold_rows_df, site_df)
        split_method = "official_fold_dict_parsed"
    else:
        # Fallback only if official fold cannot be parsed
        site_splits = create_fallback_site_split(site_df)
        split_method = "fallback_sitewise_80_10_10"

    # Add label to split file
    site_splits = site_splits.merge(
        site_df[["site_id", "label", "frame_count"]],
        on="site_id",
        how="left"
    )

    site_splits_path = SPLIT_DIR / "site_splits.csv"
    site_splits.to_csv(site_splits_path, index=False)

    # Attach split to each frame
    frames_with_splits = frames.merge(
        site_splits[["site_id", "split"]],
        on="site_id",
        how="left"
    )

    frames_with_splits_path = SPLIT_DIR / "frames_with_splits.csv"
    frames_with_splits.to_csv(frames_with_splits_path, index=False)

    # Statistics
    total_frames = len(frames)
    total_sites = site_df["site_id"].nunique()
    looted_sites = site_df[site_df["label"] == "looted"]["site_id"].nunique()
    preserved_sites = site_df[site_df["label"] == "preserved"]["site_id"].nunique()

    frame_counts = site_df["frame_count"].tolist()

    min_frames = min(frame_counts) if frame_counts else 0
    max_frames = max(frame_counts) if frame_counts else 0
    mean_frames = round(statistics.mean(frame_counts), 2) if frame_counts else 0
    median_frames = statistics.median(frame_counts) if frame_counts else 0

    split_summary = (
        site_splits.groupby(["split", "label"])
        .agg(
            sites=("site_id", "nunique"),
            frames=("frame_count", "sum")
        )
        .reset_index()
    )

    wrong_size = frames[
        (frames["width"] != EXPECTED_IMAGE_SIZE[0]) |
        (frames["height"] != EXPECTED_IMAGE_SIZE[1])
    ]

    missing_dates = frames[
        frames["year"].isna() | frames["month"].isna()
    ]

    # Create report
    report = []

    report.append("# Step 3 Preprocessing and Split Preparation Report\n")

    report.append("## Input Files\n")
    report.append(f"- Dataset root: `{DATA_ROOT}`")
    report.append(f"- Step 2 inventory: `{STEP2_INVENTORY}`")
    report.append(f"- fold_dict.json: `{FOLD_DICT_PATH}`")
    report.append(f"- fold_dict status: `{fold_status}`")
    report.append(f"- split method used: `{split_method}`\n")

    report.append("## Clean Dataset Summary\n")
    report.append(f"- Valid satellite frames used: {total_frames}")
    report.append(f"- Unique sites: {total_sites}")
    report.append(f"- Looted sites: {looted_sites}")
    report.append(f"- Preserved sites: {preserved_sites}")
    report.append(f"- Minimum frames per site: {min_frames}")
    report.append(f"- Maximum frames per site: {max_frames}")
    report.append(f"- Mean frames per site: {mean_frames}")
    report.append(f"- Median frames per site: {median_frames}\n")

    report.append("## Split Summary\n")
    if len(split_summary) > 0:
        for _, row in split_summary.iterrows():
            report.append(
                f"- {row['split']} / {row['label']}: "
                f"{int(row['sites'])} sites, {int(row['frames'])} frames"
            )
    else:
        report.append("- No split summary available.")

    report.append("\n## Checks\n")
    report.append(f"- Images not {EXPECTED_IMAGE_SIZE[0]}x{EXPECTED_IMAGE_SIZE[1]}: {len(wrong_size)}")
    report.append(f"- Frames with missing year or month: {len(missing_dates)}")

    leakage_check = (
        frames_with_splits.groupby("site_id")["split"]
        .nunique()
        .reset_index(name="split_count")
    )

    leakage_sites = leakage_check[leakage_check["split_count"] > 1]

    report.append(f"- Data leakage check: {len(leakage_sites)} sites appear in more than one split")

    report.append("\n## Expected DAFA-LS Reference Values\n")
    report.append(f"- Expected total images: {EXPECTED_TOTAL_IMAGES}")
    report.append(f"- Expected total sites: {EXPECTED_TOTAL_SITES}")
    report.append(f"- Expected looted sites: {EXPECTED_LOOTED_SITES}")
    report.append(f"- Expected preserved sites: {EXPECTED_PRESERVED_SITES}")

    report.append("\n## Warnings\n")
    warnings = []

    if split_method.startswith("fallback"):
        warnings.append(
            "fold_dict.json was not parsed into official splits. "
            "A fallback site-wise split was created. For final thesis comparison, inspect fold_dict.json carefully."
        )

    if len(wrong_size) > 0:
        warnings.append(
            "Some images are not 266x266. Resize them during model loading."
        )

    if len(missing_dates) > 0:
        warnings.append(
            "Some frames have missing date information from filename. Temporal order may rely on filename sorting."
        )

    if len(leakage_sites) > 0:
        warnings.append(
            "Data leakage detected. Same site appears in multiple splits. Fix before training."
        )

    if total_sites != EXPECTED_TOTAL_SITES:
        warnings.append(
            f"Detected {total_sites} sites, expected {EXPECTED_TOTAL_SITES}. Check Step 2 site_id detection."
        )

    if looted_sites != EXPECTED_LOOTED_SITES:
        warnings.append(
            f"Detected {looted_sites} looted sites, expected {EXPECTED_LOOTED_SITES}."
        )

    if preserved_sites != EXPECTED_PRESERVED_SITES:
        warnings.append(
            f"Detected {preserved_sites} preserved sites, expected {EXPECTED_PRESERVED_SITES}."
        )

    if not warnings:
        report.append("- No major warnings found.")
    else:
        for warning in warnings:
            report.append(f"- WARNING: {warning}")

    report.append("\n## Output Files Created\n")
    report.append(f"- `{metadata_all_frames_path}`")
    report.append(f"- `{metadata_sites_path}`")
    report.append(f"- `{preprocessing_config_path}`")
    report.append(f"- `{site_splits_path}`")
    report.append(f"- `{frames_with_splits_path}`")
    if len(fold_rows_df) > 0:
        report.append(f"- `{official_long_path}`")

    report_path = REPORT_DIR / "step3_preprocessing_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\nStep 3 completed.")
    print(f"Metadata frames: {metadata_all_frames_path}")
    print(f"Metadata sites: {metadata_sites_path}")
    print(f"Preprocessing config: {preprocessing_config_path}")
    print(f"Site splits: {site_splits_path}")
    print(f"Frames with splits: {frames_with_splits_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()