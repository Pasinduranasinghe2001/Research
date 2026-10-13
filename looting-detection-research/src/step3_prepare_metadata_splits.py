"""Prepare DAFA-LS metadata and one fixed site-wise train/val/test split.

Run the corrected Step 02 first. Then, from your project folder:
    python step3_prepare_metadata_splits.py

Required package: pandas. This step uses the CPU and does not modify images.
It does not perform cross-validation or apply image normalization.

Expected fold_dict.json structure (IDs may be strings or integers):
    {"train": {"looted": [...], "preserved": [...]},
     "val_1": {"looted": [...], "preserved": [...]},
     ...,
     "test": {"looted": [...], "preserved": [...]}}

Default policy: test -> test; complete val_1 -> val; all other listed sites
-> train (including val_2 through val_5, when present).
This preserves your intended single-split policy; it is not a claim that
this is the reference paper's complete evaluation procedure.

Always group/join by site_uid, never by the bare site_id. There is no
automatic random fallback if the official metadata cannot be validated.
"""

from pathlib import Path, PurePosixPath
import argparse
import json
import re

import pandas as pd


DATA_ROOT = Path("data/raw/DAFA_LS")
STEP2_INVENTORY = Path("results/tables/step2_dataset_inventory.csv")
PROCESSED_DIR = Path("data/processed")
SPLIT_DIR = Path("data/splits")
REPORT_DIR = Path("results/tables")
VALIDATION_FOLD = "val_1"
LABEL_MAPPING = {"preserved": 0, "looted": 1}
EXPECTED_IMAGE_SIZE = (266, 266)
EXPECTED_TOTAL_IMAGES = 55480
EXPECTED_TOTAL_SITES = 675
EXPECTED_LOOTED_SITES = 135
EXPECTED_PRESERVED_SITES = 540


def require(condition, message):
    """Stop on invalid data, even when Python runs with assertions disabled."""
    if not condition:
        raise ValueError(message)


def read_boolean_column(series, name):
    text = series.astype("string").str.strip().str.lower()
    values = text.map({"true": True, "false": False, "1": True, "0": False})
    require(values.notna().all(), f"Column {name} contains missing or invalid boolean values. Rerun Step 02.")
    return values.astype(bool)


def normalize_relative_path(value):
    text = str(value).replace("\\", "/")
    path = PurePosixPath(text)
    require(bool(text) and not path.is_absolute() and ".." not in path.parts
            and not re.match(r"^[A-Za-z]:", text),
            f"Expected a dataset-relative image path, got: {value!r}")
    return path.as_posix()


def load_inventory(inventory_path, data_root):
    """Read the corrected inventory and validate identities before using it."""
    inventory = pd.read_csv(
        inventory_path,
        dtype={"site_id": "string", "site_uid": "string", "label": "string",
               "relative_path": "string", "file_name": "string"},
        keep_default_na=False,
    )
    required = {
        "site_id", "site_uid", "label", "relative_path", "file_name",
        "is_corrupt", "is_mask", "is_valid_site", "year", "month", "day",
        "width", "height", "mode", "channels",
    }
    missing = sorted(required - set(inventory.columns))
    require(not missing, f"Missing inventory columns: {missing}. Run the corrected Step 02 first.")
    require("split" not in inventory.columns,
            "Use step2_dataset_inventory.csv, not an old frames_with_splits.csv.")
    for column in ["is_corrupt", "is_mask", "is_valid_site"]:
        inventory[column] = read_boolean_column(inventory[column], column)

    keep = (~inventory["is_corrupt"] & ~inventory["is_mask"] & inventory["is_valid_site"])
    require(inventory.loc[keep, "label"].isin(LABEL_MAPPING).all(),
            "An inventory row marked as a valid site has an unknown label.")
    frames = inventory.loc[keep].copy()
    require(not frames.empty, "No usable frames remain in the Step 02 inventory.")
    frames["relative_path"] = frames["relative_path"].map(normalize_relative_path)
    duplicated = frames["relative_path"].duplicated(keep=False)
    require(not duplicated.any(),
            "Duplicate image paths in the input inventory. Rerun Step 02; "
            f"examples: {frames.loc[duplicated, 'relative_path'].head(3).tolist()}")

    identities = []
    for row in frames[["relative_path", "label", "site_id", "site_uid", "file_name"]].itertuples(index=False):
        parts = PurePosixPath(row.relative_path).parts
        require(len(parts) >= 3, f"Missing class/site folders in {row.relative_path}")
        label, site_id = parts[0].lower(), parts[1]
        uid = f"{label}/{site_id}"
        require(label == row.label and site_id == row.site_id and uid == row.site_uid,
                f"Site identity disagrees with the path: {row.relative_path}. Rerun Step 02.")
        require(parts[-1] == row.file_name, f"Filename disagrees with path: {row.relative_path}")
        identities.append((uid, "/".join(parts[:2])))
    identity_table = pd.DataFrame(identities, columns=["site_uid", "folder"])
    require(not (identity_table.groupby("site_uid")["folder"].nunique() > 1).any(),
            "One site_uid points to different folder spellings. Resolve the ambiguous paths.")

    for column in ["year", "month", "day", "width", "height", "channels"]:
        text = frames[column].astype("string").str.strip()
        numeric = pd.to_numeric(text.replace("", pd.NA), errors="coerce")
        require(not (text.ne("") & numeric.isna()).any(), f"Invalid numeric values in {column}.")
        require((numeric.dropna() % 1 == 0).all(), f"Non-integer values in {column}.")
        frames[column] = numeric.astype("Int64")
    require(frames[["year", "month", "width", "height", "channels"]].notna().all().all(),
            "A retained frame has missing dates or image properties. Resolve the Step 02 warnings.")
    require((frames[["width", "height", "channels"]] > 0).all().all(),
            "Image dimensions and channel counts must be positive.")

    date_parts = pd.DataFrame({
        "year": frames["year"].astype(int),
        "month": frames["month"].astype(int),
        "day": frames["day"].fillna(1).astype(int),
    })
    dates = pd.to_datetime(date_parts, errors="coerce")
    require(dates.notna().all(), "Invalid calendar dates in the inventory. Correct them in Step 02.")
    frames["year_month"] = dates.dt.strftime("%Y-%m")
    frames["label_id"] = frames["label"].map(LABEL_MAPPING).astype(int)
    frames["full_path"] = frames["relative_path"].map(
        lambda p: str(data_root.joinpath(*PurePosixPath(p).parts))
    )
    frames["year_sort"] = frames["year"]
    frames["month_sort"] = frames["month"]
    frames["day_sort"] = frames["day"].fillna(99)
    frames = frames.sort_values(
        ["site_uid", "year_sort", "month_sort", "day_sort", "file_name", "relative_path"],
        kind="stable",
    ).reset_index(drop=True)
    frames["frame_index"] = frames.groupby("site_uid", sort=False).cumcount()
    return frames, len(inventory) - len(frames)


def summarize_sites(frames):
    return (
        frames.groupby(["site_uid", "label", "site_id"], sort=True)
        .agg(
            label_id=("label_id", "first"),
            frame_count=("relative_path", "count"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            unique_months=("year_month", "nunique"),
            width=("width", "first"),
            height=("height", "first"),
            channels=("channels", "first"),
        )
        .reset_index()
    )


def resolve_fold_site(value, label, known_uids):
    """Resolve IDs only inside their named class; never guess the class."""
    require(isinstance(value, (str, int)) and not isinstance(value, bool),
            f"Unsupported site entry under {label}: {value!r}")
    raw = str(value).strip()
    require(bool(raw), f"Empty site ID under {label}.")
    # Official lists normally contain bare IDs. Also accept qualified paths.
    uid = f"{label}/{raw}"
    if uid in known_uids:
        return uid
    parts = raw.replace("\\", "/").split("/")
    class_positions = [i for i, part in enumerate(parts) if part.lower() in LABEL_MAPPING]
    require(len(class_positions) == 1,
            f"Unknown site entry {value!r} under {label}. IDs must match the inventory exactly.")
    index = class_positions[0]
    require(parts[index].lower() == label and index + 1 < len(parts),
            f"Class or site missing/mismatched in fold entry {value!r} under {label}.")
    uid = f"{label}/{parts[index + 1]}"
    require(uid in known_uids, f"Fold entry refers to an unavailable site: {uid}")
    return uid


def read_official_folds(fold_path, site_df, validation_fold):
    with fold_path.open("r", encoding="utf-8") as handle:
        fold_dict = json.load(handle)
    require(isinstance(fold_dict, dict) and bool(fold_dict),
            "fold_dict.json must contain named partitions as a JSON object.")
    require({"train", "test", validation_fold}.issubset(fold_dict),
            f"fold_dict.json must contain train, test, and {validation_fold}.")
    known_uids = set(site_df["site_uid"])
    rows = []
    for fold_name, branch in fold_dict.items():
        require(fold_name in {"train", "test"} or re.fullmatch(r"val_[1-9]\d*", fold_name),
                f"Unsupported fold name: {fold_name!r}. Check the dataset instructions.")
        require(isinstance(branch, dict) and set(branch) == set(LABEL_MAPPING),
                f"{fold_name} must contain looted and preserved lists. No guessed parsing is used.")
        for label in ["looted", "preserved"]:
            values = branch[label]
            require(isinstance(values, list), f"{fold_name}/{label} must be a list of site IDs.")
            for value in values:
                uid = resolve_fold_site(value, label, known_uids)
                rows.append({
                    "site_uid": uid, "site_id": uid.split("/", 1)[1], "label": label,
                    "fold_name": fold_name,
                    "split": "val" if fold_name.startswith("val_") else fold_name,
                    "fold_source": f"fold_dict/{fold_name}/{label}", "raw_value": str(value),
                })
    require(bool(rows), "No site assignments were found in fold_dict.json.")
    official = pd.DataFrame(rows)
    duplicate = official["site_uid"].duplicated(keep=False)
    require(not duplicate.any(),
            "A site appears repeatedly or in multiple official partitions: "
            f"{official.loc[duplicate, ['site_uid', 'fold_name']].head(6).to_dict('records')}")
    assigned = set(official["site_uid"])
    require(assigned == known_uids,
            "Official partition coverage does not match the inventory. "
            f"Unassigned sites: {sorted(known_uids - assigned)[:10]}; "
            f"unknown sites: {sorted(assigned - known_uids)[:10]}.")
    return official


def make_splits(frames, site_df, official, validation_fold):
    sites = site_df.merge(
        official[["site_uid", "fold_name", "fold_source"]],
        on="site_uid", how="left", validate="one_to_one",
    )
    sites["split"] = sites["fold_name"].map(
        lambda fold: "test" if fold == "test" else "val" if fold == validation_fold else "train"
    )
    for split in ["train", "val", "test"]:
        require(set(sites.loc[sites["split"] == split, "label"]) == set(LABEL_MAPPING),
                f"The {split} split must have both looted and preserved sites for this experiment.")
    assigned_frames = frames.merge(
        sites[["site_uid", "split"]], on="site_uid", how="left",
        validate="many_to_one", sort=False,
    )
    require(len(assigned_frames) == len(frames), "Merging changed the frame-row count.")
    require(not assigned_frames["relative_path"].duplicated().any(), "Merging duplicated image paths.")
    require(assigned_frames["split"].notna().all(), "Some frames have no split assignment.")
    require(not (assigned_frames.groupby("site_uid")["split"].nunique() > 1).any(),
            "A site appears in multiple final splits.")
    expected_test = set(official.loc[official["fold_name"] == "test", "site_uid"])
    expected_val = set(official.loc[official["fold_name"] == validation_fold, "site_uid"])
    require(set(sites.loc[sites["split"] == "test", "site_uid"]) == expected_test,
            "Final test membership differs from the official test entries.")
    require(set(sites.loc[sites["split"] == "val", "site_uid"]) == expected_val,
            "Final validation membership differs from the complete selected fold.")
    index_check = assigned_frames.groupby("site_uid")["frame_index"].agg(["min", "max", "nunique", "size"])
    require(((index_check["min"] == 0) &
             (index_check["max"] == index_check["size"] - 1) &
             (index_check["nunique"] == index_check["size"])).all(),
            "Frame indices must run from 0 to frame_count - 1 for every site.")
    require(int(sites["frame_count"].sum()) == len(assigned_frames), "Site and frame totals disagree.")
    return sites, assigned_frames


def main(project_root=Path("."), validation_fold=VALIDATION_FOLD):
    root = Path(project_root).resolve()
    data_root = root / DATA_ROOT
    inventory_path = root / STEP2_INVENTORY
    fold_path = data_root / "fold_dict.json"
    require(re.fullmatch(r"val_[1-9]\d*", validation_fold) is not None,
            "Choose a validation-fold name such as val_1.")
    for path in [inventory_path, fold_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    frames, excluded_rows = load_inventory(inventory_path, data_root)
    site_df = summarize_sites(frames)
    official = read_official_folds(fold_path, site_df, validation_fold)
    site_splits, frames_with_splits = make_splits(frames, site_df, official, validation_fold)
    split_summary = (
        site_splits.groupby(["split", "label"])
        .agg(sites=("site_uid", "nunique"), frames=("frame_count", "sum"))
        .reset_index()
    )
    warnings = []
    totals = {
        "images": len(frames), "sites": len(site_df),
        "looted sites": int((site_df["label"] == "looted").sum()),
        "preserved sites": int((site_df["label"] == "preserved").sum()),
    }
    for name, expected in [("images", EXPECTED_TOTAL_IMAGES), ("sites", EXPECTED_TOTAL_SITES),
                           ("looted sites", EXPECTED_LOOTED_SITES), ("preserved sites", EXPECTED_PRESERVED_SITES)]:
        if totals[name] != expected:
            warnings.append(f"Found {totals[name]} {name}; the reference count is {expected}. Investigate the difference.")
    wrong_size = int(((frames["width"] != EXPECTED_IMAGE_SIZE[0]) |
                      (frames["height"] != EXPECTED_IMAGE_SIZE[1])).sum())
    non_rgb = int(((frames["channels"] != 3) | (frames["mode"] != "RGB")).sum())
    repeated_months = int(frames.duplicated(["site_uid", "year_month"]).sum())
    if wrong_size:
        warnings.append(f"{wrong_size} frames have unexpected dimensions. Review resizing during model loading.")
    if non_rgb:
        warnings.append(f"{non_rgb} frames are not RGB. Review channel conversion during model loading.")
    if repeated_months:
        warnings.append(f"{repeated_months} extra frames share a site and year-month. Decide which acquisitions to use; this script retains them.")

    paths = {
        "metadata_all_frames": root / PROCESSED_DIR / "metadata_all_frames.csv",
        "metadata_sites": root / PROCESSED_DIR / "metadata_sites.csv",
        "preprocessing_config": root / PROCESSED_DIR / "preprocessing_config.json",
        "site_splits_official_long": root / SPLIT_DIR / "site_splits_official_long.csv",
        "site_splits": root / SPLIT_DIR / "site_splits.csv",
        "frames_with_splits": root / SPLIT_DIR / "frames_with_splits.csv",
        "report": root / REPORT_DIR / "step3_preprocessing_report.md",
    }
    training_sources = sorted(site_splits.loc[site_splits["split"] == "train", "fold_name"].unique())
    config = {
        "dataset_name": "DAFA-LS", "task_type": "binary_classification",
        "label_mapping": LABEL_MAPPING, "site_key": "site_uid",
        "site_identity": "class folder plus original site_id; keep both identifiers as text",
        "split_type": "site_wise", "split_method": "official_fold_dict_fixed_holdout",
        "cross_validation": False, "validation_fold": validation_fold,
        "training_source_partitions": training_sources, "test_source_partition": "test",
        "data_leakage_rule": "Each site_uid occurs in exactly one final split",
        "temporal_order": "site_uid/year/month/day/filename; unknown day after known days within a month",
        "image_channels": "RGB", "expected_image_size": list(EXPECTED_IMAGE_SIZE),
        "image_transformations_applied": False,
        "normalization": {
            "method": "scale_pixels_to_0_1_during_model_loading",
            "mean_std": "compute_from_training_set_or_use_the_selected_pretrained_models_required_transform",
        },
        "inventory_path": str(inventory_path), "fold_dict_path": str(fold_path),
    }
    report = [
        "# Step 3 Preprocessing and Split Preparation Report", "", "## Input Files", "",
        f"- Dataset root: `{data_root}`", f"- Step 2 inventory: `{inventory_path}`",
        f"- fold_dict.json: `{fold_path}`", "- Device: CPU (metadata processing)", "",
        "## Split Policy", "",
        "- Method: fixed holdout using validated fold_dict.json entries.",
        "- Test: all sites in the test partition, preserving both classes.",
        f"- Validation: all sites in {validation_fold}, preserving both classes.",
        f"- Training source partitions: {', '.join(training_sources)}.",
        "- This is one split, not five-fold cross-validation.", "",
        "## Clean Dataset Summary", "",
        f"- Valid satellite frames used: {len(frames)}",
        f"- Excluded mask/unreadable/invalid-site inventory rows: {excluded_rows}",
        f"- Unique sites using site_uid: {len(site_df)}",
        f"- Looted sites: {totals['looted sites']}", f"- Preserved sites: {totals['preserved sites']}",
        f"- Minimum frames per site: {int(site_df['frame_count'].min())}",
        f"- Maximum frames per site: {int(site_df['frame_count'].max())}",
        f"- Mean frames per site: {site_df['frame_count'].mean():.2f}",
        f"- Median frames per site: {site_df['frame_count'].median():g}", "",
        "## Split Summary", "",
        "| Split | Label | Sites | Frames |", "|---|---|---:|---:|",
    ]
    for split in ["train", "val", "test"]:
        for row in split_summary[split_summary["split"] == split].itertuples(index=False):
            report.append(f"| {row.split} | {row.label} | {int(row.sites)} | {int(row.frames)} |")
    report.extend([
        "", "## Validated Checks", "",
        f"- Official rows parsed: {len(official)}; exactly one entry per site_uid.",
        "- All retained sites occur in fold_dict.json: PASS.",
        "- Complete selected validation and test memberships preserved: PASS.",
        "- Sites appearing in more than one final split: 0.",
        "- Frames without a split: 0.",
        "- Duplicate image paths in input and final frame tables: 0.",
        f"- Frame rows before merging: {len(frames)}; after merging: {len(frames_with_splits)}.",
        "- Frame indices restart at 0 and are consecutive within each site_uid: PASS.",
        "- Retained frames with missing year/month or invalid calendar dates: 0.",
        f"- Images not 266 x 266: {wrong_size}.", f"- Frames not three-channel RGB: {non_rgb}.", "",
        "## Warnings", "",
        *([f"- WARNING: {warning}" for warning in warnings] or ["- No warnings from the implemented checks."]),
        "", "## Scope and Next Step", "",
        "- Images were not reopened, resized, normalized, or used to train a model.",
        "- Site identity and image properties come from the corrected Step 02 inventory.",
        "- These checks do not establish that image contents, labels, or geographic sites are unique/correct.",
        "- The unresolved reference image-count difference is not hidden or automatically repaired.",
        "- Later loaders must group by site_uid and read site_id/site_uid as text.",
        "- Do not pass site_uid or its embedded class name as an input feature to the model.",
        "", "## Output Files Created", "",
        *[f"- `{path}`" for path in paths.values()],
    ])

    # All input, mapping, and merge checks have passed before any output is saved.
    for directory in {path.parent for path in paths.values()}:
        directory.mkdir(parents=True, exist_ok=True)
    frames.to_csv(paths["metadata_all_frames"], index=False)
    site_df.to_csv(paths["metadata_sites"], index=False)
    official.to_csv(paths["site_splits_official_long"], index=False)
    site_splits.to_csv(paths["site_splits"], index=False)
    frames_with_splits.to_csv(paths["frames_with_splits"], index=False)
    paths["preprocessing_config"].write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
    paths["report"].write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Step 3 completed. All site and split checks passed.")
    print(f"Sites: {len(site_df)}; frames: {len(frames_with_splits)}")
    for split in ["train", "val", "test"]:
        print(f"{split}: {int((site_splits['split'] == split).sum())} sites")
    print(f"Report: {paths['report']}")
    if warnings:
        print(f"Review the report: {len(warnings)} warning(s).")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare validated DAFA-LS site-wise metadata and splits.")
    parser.add_argument("--project-root", type=Path, default=Path("."),
                        help="Project folder containing data/ and results/ (default: current folder).")
    parser.add_argument("--validation-fold", default=VALIDATION_FOLD,
                        help="Complete fold to hold out for validation (default: val_1).")
    args = parser.parse_args()
    try:
        main(args.project_root, args.validation_fold)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Step 3 failed: {exc}\nDo not use previous Step 3 outputs as corrected results.\n")
