"""Create validated K-frame image lists from the corrected Step 3 output.

Run from your project folder:
    python step4_create_kframe_datasets.py
Or specify that folder:
    python step4_create_kframe_datasets.py --project-root "E:/path/to/project"

Requires pandas. Uses the CPU; does not open, copy, or transform images.
Keeps Step 3's existing split. Does not perform training or cross-validation.
Always use site_uid as the site key. Never give identifiers to the model as
input features: site_uid and sample_id contain the class name.
"""

from pathlib import Path, PurePosixPath, PureWindowsPath
import argparse
import hashlib
import json
import random
import re

import pandas as pd


INPUT_FRAMES = Path("data/splits/frames_with_splits.csv")
DATA_ROOT = Path("data/raw/DAFA_LS")
OUTPUT_DIR = Path("data/processed/kframe_sequences")
REPORT_DIR = Path("results/tables")
K_VALUES = [1, 2, 3, 5, 8]
STRATEGIES = ["recent_window", "uniform", "early_late", "random"]
LABEL_MAPPING = {"preserved": 0, "looted": 1}
SEED = 42
SKIPPED_COLUMNS = [
    "site_uid", "site_id", "label", "split", "k", "strategy",
    "available_frames", "reason",
]


def require(condition, message):
    """Stop on invalid data, including when Python assertions are disabled."""
    if not condition:
        raise ValueError(message)


def integer_column(series, name):
    values = pd.to_numeric(series, errors="coerce")
    require(values.notna().all(), f"Missing or invalid numbers in {name}.")
    require(((values % 1) == 0).all(), f"Non-integer values in {name}.")
    return values.astype("int64")


def normalize_relative_path(value):
    text = str(value).replace("\\", "/")
    path = PurePosixPath(text)
    require(bool(text) and not path.is_absolute() and ".." not in path.parts
            and not re.match(r"^[A-Za-z]:", text),
            f"Invalid dataset-relative path: {value!r}")
    return path.as_posix()


def load_frames(input_path, data_root):
    """Validate all input records before selecting frames or writing files."""
    # Read identifiers as text: site 001 must not silently become site 1.
    frames = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    required = {
        "site_uid", "site_id", "label", "split", "relative_path",
        "frame_index", "year", "month",
    }
    missing = sorted(required - set(frames.columns))
    require(not missing,
            f"Missing columns: {missing}. Run the corrected Step 3 first.")
    require(not frames.empty, "The Step 3 frame table is empty.")
    for column in required:
        require(frames[column].str.strip().ne("").all(),
                f"Missing values in {column}. Rerun the corrected Step 3.")
    require(frames["label"].isin(LABEL_MAPPING).all(), "Unknown classification label.")
    frames["split"] = frames["split"].str.strip().str.lower()
    require(frames["split"].isin(["train", "val", "test"]).all(),
            "Every split must be train, val, or test.")
    require(frames["site_uid"].eq(frames["label"] + "/" + frames["site_id"]).all(),
            "site_uid disagrees with label/site_id. Rerun the corrected Step 3.")

    frames["relative_path"] = frames["relative_path"].map(normalize_relative_path)
    duplicate_paths = frames["relative_path"].duplicated(keep=False)
    require(not duplicate_paths.any(),
            "Duplicate image paths in the input. Rerun the corrected Step 3. "
            f"Examples: {frames.loc[duplicate_paths, 'relative_path'].head(3).tolist()}")
    require(frames["relative_path"].str.casefold().is_unique,
            "Image paths differ only by letter case. Resolve ambiguous paths before training.")
    for row in frames[["relative_path", "site_uid"]].itertuples(index=False):
        parts = PurePosixPath(row.relative_path).parts
        require(len(parts) >= 3 and f"{parts[0].lower()}/{parts[1]}" == row.site_uid,
                f"Image path disagrees with site_uid: {row.relative_path}")

    # This check must use the class-qualified UID, not the reused bare number.
    split_counts = frames.groupby("site_uid")["split"].nunique()
    leaked = split_counts[split_counts > 1].index.tolist()
    require(not leaked, f"Real site overlap between splits: {leaked[:10]}")
    for column, expected in [("is_valid_site", True), ("is_mask", False), ("is_corrupt", False)]:
        if column in frames:
            values = frames[column].str.strip().str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            )
            require(values.notna().all() and values.eq(expected).all(),
                    f"Unexpected {column} values. Use the corrected Step 3 frame table.")
    expected_labels = frames["label"].map(LABEL_MAPPING).astype(int)
    if "label_id" in frames:
        require(integer_column(frames["label_id"], "label_id").eq(expected_labels).all(),
                "label_id disagrees with the preserved=0, looted=1 mapping.")
    frames["label_id"] = expected_labels
    for column in ["frame_index", "year", "month"]:
        frames[column] = integer_column(frames[column], column)
    day_text = frames["day"].str.strip() if "day" in frames else pd.Series("", index=frames.index)
    days = integer_column(day_text.replace("", "1"), "day")
    dates = pd.to_datetime(
        pd.DataFrame({"year": frames["year"], "month": frames["month"], "day": days}),
        errors="coerce",
    )
    require(dates.notna().all(), "Invalid calendar dates in the Step 3 table.")
    # Match Step 3: a missing day sorts after known days within the same month.
    sort_days = days.where(day_text.ne(""), 99)
    frames["_date_order"] = frames["year"] * 10000 + frames["month"] * 100 + sort_days
    frames = frames.sort_values(["site_uid", "frame_index"], kind="stable").reset_index(drop=True)
    expected_indices = frames.groupby("site_uid", sort=False).cumcount()
    require(frames["frame_index"].eq(expected_indices).all(),
            "Frame indices must be unique and consecutive from 0 within each site_uid.")
    require(all(group["_date_order"].is_monotonic_increasing
                for _, group in frames.groupby("site_uid", sort=False)),
            "Frame indices are not in chronological order. Rerun Step 3.")

    if "full_path" not in frames:
        frames["full_path"] = frames["relative_path"].map(
            lambda p: str(data_root.joinpath(*PurePosixPath(p).parts))
        )
    require(frames["full_path"].str.strip().ne("").all(), "An image has an empty full_path.")
    for full_path, relative_path in zip(frames["full_path"], frames["relative_path"]):
        require(PurePosixPath(full_path).is_absolute() or PureWindowsPath(full_path).is_absolute(),
                f"full_path must be absolute: {full_path}")
        normalized_full = str(full_path).replace("\\", "/")
        require(normalized_full.endswith("/" + relative_path),
                f"full_path disagrees with relative_path: {relative_path}")
    return frames.drop(columns="_date_order")


def stable_seed(text, base_seed=SEED):
    """A repeatable integer; avoids Python's process-dependent built-in hash."""
    digest = hashlib.sha256(f"{base_seed}_{text}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def select_indices(n, k, strategy, site_uid, split):
    require(isinstance(k, int) and k > 0, "K must be a positive integer.")
    require(strategy in STRATEGIES, f"Unknown strategy: {strategy}")
    if n < k:
        return None
    if strategy == "recent_window":
        selected = list(range(n - k, n))
    elif strategy == "uniform":
        # Equal spacing in frame positions, not necessarily in calendar time.
        selected = [n - 1] if k == 1 else [round(i * (n - 1) / (k - 1)) for i in range(k)]
    elif strategy == "early_late":
        selected = [n - 1] if k == 1 else [0] + list(range(n - k + 1, n))
    else:
        # Use the complete UID: looted/0 and preserved/0 are different sites.
        seed_text = json.dumps([site_uid, strategy, split, k], separators=(",", ":"))
        rng = random.Random(stable_seed(seed_text))
        selected = sorted(rng.sample(range(n), k))
    require(len(selected) == k and len(set(selected)) == k,
            f"Selection did not produce K distinct positions for {site_uid}.")
    require(selected == sorted(selected) and all(0 <= p < n for p in selected),
            f"Invalid selection positions for {site_uid}.")
    return selected


def make_record(group, positions, k, strategy):
    selected = group.iloc[positions]
    first = group.iloc[0]
    uid, site_id, label, split = (first[c] for c in ["site_uid", "site_id", "label", "split"])
    require(len(selected) == k and selected["relative_path"].nunique() == k,
            f"Sequence must contain K different images: {uid}, {strategy}, K={k}.")
    require(selected["site_uid"].eq(uid).all() and selected["split"].eq(split).all(),
            "A sequence contains different sites or splits.")
    suffix = "full_sequence" if strategy == "full_sequence" else f"{strategy}_K{k}"
    record = {
        "sample_id": f"{split}_{label}_{site_id}_{suffix}",
        "site_uid": uid, "site_id": site_id, "label": label,
        "label_id": LABEL_MAPPING[label], "split": split, "k": k,
        "strategy": strategy, "available_frames": len(group), "selected_count": k,
    }
    lists = {
        "selected_positions": positions,
        "frame_indices": [int(v) for v in selected["frame_index"]],
        "years": [int(v) for v in selected["year"]],
        "months": [int(v) for v in selected["month"]],
        "relative_paths": selected["relative_path"].tolist(),
        "full_paths": selected["full_path"].tolist(),
    }
    for name, values in lists.items():
        require(len(values) == k, f"Incorrect list length in {name} for {uid}.")
        record[name] = json.dumps(values, allow_nan=False)
    return record


def create_samples(frames):
    records, skipped_records = [], []
    for uid, group in frames.groupby("site_uid", sort=True):
        group = group.reset_index(drop=True)
        first, n = group.iloc[0], len(group)
        for k in K_VALUES:
            for strategy in STRATEGIES:
                positions = select_indices(n, k, strategy, uid, first["split"])
                if positions is None:
                    skipped_records.append({
                        "site_uid": uid, "site_id": first["site_id"], "label": first["label"],
                        "split": first["split"], "k": k, "strategy": strategy,
                        "available_frames": n, "reason": "not_enough_frames",
                    })
                else:
                    records.append(make_record(group, positions, k, strategy))
        records.append(make_record(group, list(range(n)), n, "full_sequence"))
    samples = pd.DataFrame(records)
    # Explicit columns keep an empty skipped CSV readable with pandas.read_csv.
    skipped = pd.DataFrame(skipped_records, columns=SKIPPED_COLUMNS)
    require(not samples.empty and samples["sample_id"].is_unique,
            "Output samples are empty or sample IDs are not unique.")
    require(not samples.duplicated(["site_uid", "strategy", "k"]).any(),
            "Duplicate site/K/strategy output records.")
    require(samples["selected_count"].eq(samples["k"]).all(), "Selected count differs from K.")
    require(samples.groupby("site_uid")["split"].nunique().eq(1).all(),
            "A site appears in multiple output splits.")
    source_sites = frames.drop_duplicates("site_uid").set_index("site_uid")
    for column in ["site_id", "label", "label_id", "split"]:
        require(samples[column].eq(samples["site_uid"].map(source_sites[column])).all(),
                f"Output {column} disagrees with Step 3.")
    full = samples[samples["strategy"] == "full_sequence"]
    require(full["site_uid"].is_unique and set(full["site_uid"]) == set(source_sites.index),
            "Missing or repeated full-sequence reference rows.")
    counts = frames.groupby("site_uid").size()
    for k in K_VALUES:
        expected_sites = set(counts[counts >= k].index)
        for strategy in STRATEGIES:
            actual = samples[(samples["k"] == k) & (samples["strategy"] == strategy)]
            require(set(actual["site_uid"]) == expected_sites and len(actual) == len(expected_sites),
                    f"Incomplete output coverage for {strategy}, K={k}.")
    expected_total = len(source_sites) * len(K_VALUES) * len(STRATEGIES)
    require(len(samples) - len(full) + len(skipped) == expected_total,
            "Created and skipped sample totals do not reconcile.")
    return samples, skipped


def main(project_root=Path(".")):
    root = Path(project_root).resolve()
    input_path = root / INPUT_FRAMES
    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}. Run the corrected Step 3 first.")
    frames = load_frames(input_path, root / DATA_ROOT)
    samples, skipped = create_samples(frames)
    limited = samples[samples["strategy"].isin(STRATEGIES)]
    full = samples[samples["strategy"] == "full_sequence"]
    summary = limited.groupby(["k", "strategy", "split", "label"]).agg(
        samples=("sample_id", "size")
    ).reset_index()
    sites = frames.drop_duplicates("site_uid")
    site_counts = sites.groupby(["split", "label"]).size()
    output_dir, report_dir = root / OUTPUT_DIR, root / REPORT_DIR
    outputs = {
        "kframe_samples_all.csv": samples,
        "full_sequence_samples.csv": full,
        "kframe_skipped_samples.csv": skipped,
        "kframe_summary_by_k_strategy.csv": summary,
    }
    for k in K_VALUES:
        outputs[f"kframe_samples_K{k}.csv"] = limited[limited["k"] == k]
    for strategy in STRATEGIES:
        outputs[f"kframe_samples_{strategy}.csv"] = limited[limited["strategy"] == strategy]

    report = [
        "# Step 4 K-Frame Dataset Creation Report", "", "## Purpose", "",
        "Create image-path lists for limited-frame experiments; no images are transformed and no model is trained.",
        "", "## Input and Settings", "",
        f"- Input: `{input_path}`", "- Device: CPU (metadata processing).",
        "- Site key: site_uid, stored as text.", "- Existing Step 3 splits are preserved; no cross-validation is performed.",
        f"- K values: {K_VALUES}", f"- Random base seed: {SEED}; SHA-256 of base seed and site_uid/strategy/split/K.",
        "", "## Selection Strategies", "",
        "- recent_window: latest K available frames; missing months need not be consecutive.",
        "- uniform: evenly spaced frame positions, not necessarily equally spaced calendar dates.",
        "- early_late: earliest frame plus latest K-1 frames; earliest is not guaranteed to be undamaged.",
        "- random: one repeatable selection without replacement for each site/K, sorted chronologically.",
        "- For K=1, recent_window, uniform, and early_late all select the latest frame.",
        "- For K=2, uniform and early_late both select the first and last frame.",
        "- Random selections for different K values are not required to be nested.",
        "", "## Main Output Summary", "",
        f"- Input frame rows: {len(frames)}", f"- Unique sites: {len(sites)}",
        f"- Total K-frame sample rows created: {len(limited)}",
        f"- Full-sequence reference rows created: {len(full)}",
        f"- Combined output rows: {len(samples)}",
        f"- Skipped rows due to insufficient frames: {len(skipped)}",
        "- Data leakage check: 0 site_uid values appear in more than one split.",
        "", "## Site Split Counts", "",
        "| Split | Looted | Preserved | Total |", "|---|---:|---:|---:|",
    ]
    for split in ["train", "val", "test"]:
        looted = int(site_counts.get((split, "looted"), 0))
        preserved = int(site_counts.get((split, "preserved"), 0))
        report.append(f"| {split} | {looted} | {preserved} | {looted + preserved} |")
    report += [
        "", "## Validated Before Saving", "",
        "- Unique input image paths and consistent path/site identities: PASS.",
        "- One split per complete site_uid in input and output: PASS.",
        "- Frame numbering starts at zero and follows chronological order: PASS.",
        "- Every sequence has K distinct image paths and matching metadata-list lengths: PASS.",
        "- Labels and splits preserved from Step 3: PASS.",
        "- Unique sample IDs and complete eligible site/K/strategy coverage: PASS.",
        "", "## Samples by K, Strategy, Split, and Label", "",
        "| K | Strategy | Split | Label | Samples |", "|---:|---|---|---|---:|",
    ]
    for row in summary.itertuples(index=False):
        report.append(f"| {row.k} | {row.strategy} | {row.split} | {row.label} | {row.samples} |")
    report += ["", "## Warnings", ""]
    report.append(
        "- Some sites lack enough frames. Compare eligible site sets across K values; see kframe_skipped_samples.csv."
        if len(skipped) else "- No warnings from the implemented checks."
    )
    report += [
        "", "## Scope and Use", "",
        "- These are multiple selections from the same sites, not new independent sites.",
        "- Choose the intended K and strategy for each experiment and keep the existing split assignments.",
        "- Group and join by site_uid. Never pass site_uid, sample_id, or class-bearing paths as model features.",
        "- Random selections change from the old script because the seed now uses the complete site identity.",
        "- Images were not reopened. Unique paths do not establish unique pixels or geographical sites.",
        "- Step 2/3 dataset warnings, including reference image-count differences, are not resolved by this step.",
        "- These checks do not independently establish that the original official fold memberships are correct.",
        "", "## Output Files Created", "",
        *[f"- `{output_dir / filename}`" for filename in outputs],
        f"- `{report_dir / 'step4_kframe_report.md'}`",
    ]

    # All content checks finish before any existing result file is overwritten.
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in outputs.items():
        table.to_csv(output_dir / filename, index=False)
    report_path = report_dir / "step4_kframe_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Step 4 completed. All implemented input and selection checks passed.")
    print(f"Sites: {len(sites)}; K-frame rows: {len(limited)}; full-sequence rows: {len(full)}")
    print("Sites shared across splits (using site_uid): 0")
    print(f"Skipped combinations: {len(skipped)}")
    print(f"Report: {report_path}")
    return {"output_dir": output_dir, "report": report_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create validated site-wise K-frame image lists.")
    parser.add_argument("--project-root", type=Path, default=Path("."),
                        help="Project folder containing data/ and results/ (default: current folder).")
    args = parser.parse_args()
    try:
        main(args.project_root)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Step 4 failed: {exc}\nDo not use old or partially written Step 4 outputs after a failed run.\n")
