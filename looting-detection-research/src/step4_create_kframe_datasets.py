from pathlib import Path
import pandas as pd
import json
import random
import hashlib
import math

# ==================================================
# PATH CONFIGURATION
# ==================================================

INPUT_FRAMES = Path("data/splits/frames_with_splits.csv")
OUTPUT_DIR = Path("data/processed/kframe_sequences")
REPORT_DIR = Path("results/tables")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES = [1, 2, 3, 5, 8]

STRATEGIES = [
    "recent_window",
    "uniform",
    "early_late",
    "random"
]

LABEL_MAPPING = {
    "preserved": 0,
    "looted": 1
}

SEED = 42


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def stable_seed(text: str, base_seed: int = SEED) -> int:
    """
    Creates a reproducible seed from text.
    This avoids Python's built-in hash randomness.
    """
    value = f"{base_seed}_{text}"
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def safe_json_list(values):
    clean_values = []
    for v in values:
        if pd.isna(v):
            clean_values.append(None)
        else:
            try:
                if isinstance(v, float) and v.is_integer():
                    clean_values.append(int(v))
                else:
                    clean_values.append(v)
            except Exception:
                clean_values.append(v)
    return json.dumps(clean_values)


def select_recent_window(n, k):
    """
    Select the latest K frames.
    Useful because recent images are more likely to show final looting state.
    """
    return list(range(n - k, n))


def select_uniform(n, k):
    """
    Select K frames evenly across the full time series.
    Useful for representing the whole temporal period with few frames.
    """
    if k == 1:
        return [n - 1]

    positions = []
    for i in range(k):
        pos = round(i * (n - 1) / (k - 1))
        positions.append(pos)

    # Remove duplicates while preserving order
    unique_positions = []
    for p in positions:
        if p not in unique_positions:
            unique_positions.append(p)

    # Fill missing positions if rounding created duplicates
    if len(unique_positions) < k:
        for p in range(n):
            if p not in unique_positions:
                unique_positions.append(p)
            if len(unique_positions) == k:
                break

    return sorted(unique_positions[:k])


def select_early_late(n, k):
    """
    Select an early reference frame and recent target frames.
    This is useful for change-aware looting detection.
    """
    if k == 1:
        return [n - 1]

    if k == 2:
        return [0, n - 1]

    # First frame + latest k-1 frames
    latest_frames = list(range(n - (k - 1), n))
    selected = [0] + latest_frames

    # Remove duplicates if time series is very short
    selected = sorted(list(dict.fromkeys(selected)))

    # Fill if needed
    if len(selected) < k:
        for p in range(n):
            if p not in selected:
                selected.append(p)
            if len(selected) == k:
                break

    return sorted(selected[:k])


def select_random(n, k, site_id, strategy, split):
    """
    Select K random frames reproducibly.
    Random selection is useful to test robustness to irregular data availability.
    """
    rng = random.Random(stable_seed(f"{site_id}_{strategy}_{split}_K{k}"))
    selected = rng.sample(range(n), k)
    return sorted(selected)


def select_indices(n, k, strategy, site_id, split):
    if n < k:
        return None

    if strategy == "recent_window":
        return select_recent_window(n, k)

    if strategy == "uniform":
        return select_uniform(n, k)

    if strategy == "early_late":
        return select_early_late(n, k)

    if strategy == "random":
        return select_random(n, k, site_id, strategy, split)

    raise ValueError(f"Unknown strategy: {strategy}")


# ==================================================
# MAIN SCRIPT
# ==================================================

def main():
    if not INPUT_FRAMES.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FRAMES}\n"
            "Run Step 3 first."
        )

    frames = pd.read_csv(INPUT_FRAMES)

    required_columns = [
        "site_id",
        "label",
        "split",
        "relative_path",
        "frame_index"
    ]

    missing_columns = [c for c in required_columns if c not in frames.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {INPUT_FRAMES}: {missing_columns}"
        )

    # If full_path is not available, use relative_path
    if "full_path" not in frames.columns:
        frames["full_path"] = frames["relative_path"]

    # Keep only required classification labels
    frames = frames[frames["label"].isin(["looted", "preserved"])].copy()

    # Clean split names
    frames["split"] = frames["split"].astype(str).str.lower().str.strip()

    # Sort frames inside each site by temporal order
    sort_columns = []

    for col in ["site_id", "frame_index"]:
        if col in frames.columns:
            sort_columns.append(col)

    frames = frames.sort_values(sort_columns).reset_index(drop=True)

    records = []
    skipped_records = []

    grouped = frames.groupby(["site_id", "label", "split"])

    for (site_id, label, split), group in grouped:
        group = group.sort_values("frame_index").reset_index(drop=True)
        n_available = len(group)

        for k in K_VALUES:
            for strategy in STRATEGIES:
                selected_indices = select_indices(
                    n=n_available,
                    k=k,
                    strategy=strategy,
                    site_id=site_id,
                    split=split
                )

                if selected_indices is None:
                    skipped_records.append({
                        "site_id": site_id,
                        "label": label,
                        "split": split,
                        "k": k,
                        "strategy": strategy,
                        "available_frames": n_available,
                        "reason": "not_enough_frames"
                    })
                    continue

                selected = group.iloc[selected_indices].copy()

                sample_id = f"{split}_{label}_{site_id}_{strategy}_K{k}"

                records.append({
                    "sample_id": sample_id,
                    "site_id": site_id,
                    "label": label,
                    "label_id": LABEL_MAPPING[label],
                    "split": split,
                    "k": k,
                    "strategy": strategy,
                    "available_frames": n_available,
                    "selected_count": len(selected),
                    "selected_positions": safe_json_list(selected_indices),
                    "frame_indices": safe_json_list(selected["frame_index"].tolist()),
                    "years": safe_json_list(selected["year"].tolist()) if "year" in selected.columns else json.dumps([]),
                    "months": safe_json_list(selected["month"].tolist()) if "month" in selected.columns else json.dumps([]),
                    "relative_paths": safe_json_list(selected["relative_path"].tolist()),
                    "full_paths": safe_json_list(selected["full_path"].tolist())
                })

        # Full sequence reference row
        selected = group.copy()
        records.append({
            "sample_id": f"{split}_{label}_{site_id}_full_sequence",
            "site_id": site_id,
            "label": label,
            "label_id": LABEL_MAPPING[label],
            "split": split,
            "k": n_available,
            "strategy": "full_sequence",
            "available_frames": n_available,
            "selected_count": n_available,
            "selected_positions": safe_json_list(list(range(n_available))),
            "frame_indices": safe_json_list(selected["frame_index"].tolist()),
            "years": safe_json_list(selected["year"].tolist()) if "year" in selected.columns else json.dumps([]),
            "months": safe_json_list(selected["month"].tolist()) if "month" in selected.columns else json.dumps([]),
            "relative_paths": safe_json_list(selected["relative_path"].tolist()),
            "full_paths": safe_json_list(selected["full_path"].tolist())
        })

    samples_df = pd.DataFrame(records)
    skipped_df = pd.DataFrame(skipped_records)

    # Save all samples
    all_samples_path = OUTPUT_DIR / "kframe_samples_all.csv"
    samples_df.to_csv(all_samples_path, index=False)

    # Save one file per K value
    for k in K_VALUES:
        k_df = samples_df[
            (samples_df["k"] == k) &
            (samples_df["strategy"].isin(STRATEGIES))
        ].copy()

        k_path = OUTPUT_DIR / f"kframe_samples_K{k}.csv"
        k_df.to_csv(k_path, index=False)

    # Save one file per strategy
    for strategy in STRATEGIES:
        strategy_df = samples_df[samples_df["strategy"] == strategy].copy()
        strategy_path = OUTPUT_DIR / f"kframe_samples_{strategy}.csv"
        strategy_df.to_csv(strategy_path, index=False)

    # Save full sequence reference
    full_df = samples_df[samples_df["strategy"] == "full_sequence"].copy()
    full_path = OUTPUT_DIR / "full_sequence_samples.csv"
    full_df.to_csv(full_path, index=False)

    # Save skipped file
    skipped_path = OUTPUT_DIR / "kframe_skipped_samples.csv"
    skipped_df.to_csv(skipped_path, index=False)

    # Summary tables
    summary_by_k_strategy = (
        samples_df[samples_df["strategy"].isin(STRATEGIES)]
        .groupby(["k", "strategy", "split", "label"])
        .agg(samples=("sample_id", "count"))
        .reset_index()
    )

    summary_path = OUTPUT_DIR / "kframe_summary_by_k_strategy.csv"
    summary_by_k_strategy.to_csv(summary_path, index=False)

    # Leakage check
    leakage_check = (
        samples_df.groupby("site_id")["split"]
        .nunique()
        .reset_index(name="split_count")
    )

    leakage_sites = leakage_check[leakage_check["split_count"] > 1]

    # Create report
    report = []

    report.append("# Step 4 K-Frame Dataset Creation Report\n")

    report.append("## Purpose\n")
    report.append(
        "This step creates limited satellite image time-series samples "
        "for archaeological looting detection under short temporal conditions.\n"
    )

    report.append("## Input File\n")
    report.append(f"- `{INPUT_FRAMES}`\n")

    report.append("## K Values Used\n")
    for k in K_VALUES:
        report.append(f"- K = {k}")
    report.append("")

    report.append("## Frame Selection Strategies\n")
    report.append("- `recent_window`: selects the latest K frames.")
    report.append("- `uniform`: selects K frames evenly across the whole time series.")
    report.append("- `early_late`: selects an early reference frame and recent target frames.")
    report.append("- `random`: selects K frames randomly using a fixed seed.\n")

    report.append("## Main Output Summary\n")
    report.append(f"- Total K-frame sample rows created: {len(samples_df[samples_df['strategy'].isin(STRATEGIES)])}")
    report.append(f"- Full-sequence reference rows created: {len(full_df)}")
    report.append(f"- Skipped rows due to insufficient frames: {len(skipped_df)}")
    report.append(f"- Data leakage check: {len(leakage_sites)} sites appear in more than one split\n")

    report.append("## Samples by K, Strategy, Split, and Label\n")
    if len(summary_by_k_strategy) > 0:
        for _, row in summary_by_k_strategy.iterrows():
            report.append(
                f"- K={int(row['k'])}, {row['strategy']}, {row['split']}, {row['label']}: "
                f"{int(row['samples'])} samples"
            )
    else:
        report.append("- No K-frame samples created.")

    report.append("\n## Output Files Created\n")
    report.append(f"- `{all_samples_path}`")
    report.append(f"- `{full_path}`")
    report.append(f"- `{skipped_path}`")
    report.append(f"- `{summary_path}`")

    for k in K_VALUES:
        report.append(f"- `{OUTPUT_DIR / f'kframe_samples_K{k}.csv'}`")

    for strategy in STRATEGIES:
        report.append(f"- `{OUTPUT_DIR / f'kframe_samples_{strategy}.csv'}`")

    report.append("\n## Warnings\n")
    warnings = []

    if len(skipped_df) > 0:
        warnings.append(
            "Some sites had fewer frames than required for certain K values. "
            "Check kframe_skipped_samples.csv."
        )

    if len(leakage_sites) > 0:
        warnings.append(
            "Data leakage detected. Same site appears in more than one split. "
            "Fix site_splits.csv before training."
        )

    if samples_df.empty:
        warnings.append("No samples were created. Check Step 3 output.")

    if not warnings:
        report.append("- No major warnings found.")
    else:
        for warning in warnings:
            report.append(f"- WARNING: {warning}")

    report_path = REPORT_DIR / "step4_kframe_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\nStep 4 completed.")
    print(f"All K-frame samples: {all_samples_path}")
    print(f"Full sequence samples: {full_path}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()