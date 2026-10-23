"""Check every batch in all 60 fixed-K experiments against corrected Step 3.

Put this file beside dataset_loader.py. Run from the project folder:
    python step5_check_loader.py
Or use --project-root "E:/path/to/project". Use --data-root for relocated images.

This is a full loader check and may take several minutes. It does not train a
model or implement cross-validation. Failure writes a FAIL report and exits 1.
"""
from pathlib import Path
import argparse
import json

import pandas as pd

import dataset_loader as dl

K_VALUES = dl.K_VALUES
STRATEGIES = dl.STRATEGIES
SPLITS = dl.SPLITS


def load_reference(project_root):
    """Use current Step 3 identities, splits, paths, dates and frame indices."""
    directory = Path(project_root) / "data/splits"
    sites = pd.read_csv(directory / "site_splits.csv", dtype=str, keep_default_na=False)
    frames = pd.read_csv(directory / "frames_with_splits.csv", dtype=str, keep_default_na=False)
    for table, columns, name in [
        (sites, {"site_uid", "site_id", "label", "label_id", "split", "frame_count"}, "site_splits.csv"),
        (frames, {"site_uid", "site_id", "label", "label_id", "split", "relative_path", "frame_index", "year", "month"}, "frames_with_splits.csv"),
    ]:
        dl.require(columns.issubset(table.columns) and not table.empty,
                   f"Missing/empty corrected Step 3 columns in {name}.")
        dl.require(table[list(columns)].ne("").all().all(), f"Blank required values in {name}.")
        dl.require(table["site_uid"].eq(table["label"] + "/" + table["site_id"]).all(), f"Invalid site identity in {name}.")
        dl.require(table["label"].isin(dl.LABEL_MAPPING).all() and table["split"].isin(SPLITS).all(),
                   f"Unknown labels or splits in {name}.")
        table["label_id"] = dl.integer_column(table["label_id"], "label_id")
        dl.require(table["label_id"].eq(table["label"].map(dl.LABEL_MAPPING)).all(), f"Wrong label IDs in {name}.")
        dl.require(table.groupby("site_uid")["split"].nunique().eq(1).all(), f"Site overlap in {name}.")
    dl.require(sites["site_uid"].is_unique, "Duplicate site_uid rows in Step 3 site table.")
    dl.require(set(sites["site_uid"]) == set(frames["site_uid"]), "Step 3 site/frame coverage differs.")
    sites["frame_count"] = dl.integer_column(sites["frame_count"], "frame_count")
    frames["relative_path"] = frames["relative_path"].map(dl.relative_path)
    dl.require(frames["relative_path"].str.casefold().is_unique, "Duplicate Step 3 image paths.")
    for column in ["frame_index", "year", "month"]:
        frames[column] = dl.integer_column(frames[column], column)
    frames = frames.sort_values(["site_uid", "frame_index"]).reset_index(drop=True)
    dl.require(frames["frame_index"].eq(frames.groupby("site_uid").cumcount()).all(), "Invalid Step 3 frame numbering.")
    site_lookup = sites.set_index("site_uid")
    for column in ["site_id", "label", "label_id", "split"]:
        dl.require(frames[column].eq(frames["site_uid"].map(site_lookup[column])).all(), f"Step 3 {column} tables disagree.")
    counts = frames.groupby("site_uid").size()
    dl.require(site_lookup["frame_count"].sort_index().equals(counts.sort_index()), "Step 3 frame counts disagree.")
    dl.require(set(sites["split"]) == set(SPLITS), "All three Step 3 splits are required.")
    for split in SPLITS:
        dl.require(set(sites.loc[sites["split"] == split, "label"]) == set(dl.LABEL_MAPPING),
                   f"Step 3 {split} must contain both classes.")
    frame_lookup = {
        (r.site_uid, r.frame_index): (r.relative_path, r.year, r.month)
        for r in frames.itertuples(index=False)
    }
    return sites, site_lookup, frame_lookup


def check_against_reference(samples, k, sites, site_lookup, frame_lookup):
    dl.require(samples["k"].eq(k).all(), f"A K{k} file contains a different K.")
    dl.require(set(samples["strategy"]) == set(STRATEGIES), f"K{k} is missing a required strategy.")
    eligible = sites[sites["frame_count"] >= k]
    for strategy in STRATEGIES:
        for split in SPLITS:
            expected = set(eligible.loc[eligible["split"] == split, "site_uid"])
            actual = samples[(samples["strategy"] == strategy) & (samples["split"] == split)]
            dl.require(bool(expected), f"No eligible sites for required K{k}/{strategy}/{split}.")
            dl.require(actual["site_uid"].is_unique and set(actual["site_uid"]) == expected,
                       f"K{k}/{strategy}/{split} site membership differs from current Step 3. Rerun corrected Step 4.")
    for row in samples.itertuples(index=False):
        dl.require(row.site_uid in site_lookup.index, f"Unknown site {row.site_uid}.")
        site = site_lookup.loc[row.site_uid]
        dl.require(row.split == site["split"] and row.label_id == site["label_id"]
                   and row.available_frames == site["frame_count"], f"Stale or incorrect metadata for {row.sample_id}.")
        for index, path, year, month in zip(row.frame_indices, row.relative_paths, row.years, row.months):
            dl.require(frame_lookup.get((row.site_uid, index)) == (path, year, month),
                       f"Selected frame/date differs from current Step 3: {row.sample_id}, frame {index}.")


def validate_batch(batch, k, image_size, split, strategy, expected_rows, batch_size, seen):
    """Raise on every shape, label, identity, or completeness failure."""
    torch = dl.torch
    images, labels = batch["images"], batch["label"]
    dl.require(isinstance(images, torch.Tensor) and images.ndim == 5, "Images must be a five-dimensional tensor.")
    b = images.shape[0]
    dl.require(0 < b <= batch_size and tuple(images.shape[1:]) == (k, 3, image_size, image_size),
               f"Wrong image shape: {list(images.shape)}.")
    dl.require(images.dtype == torch.float32 and torch.isfinite(images).all().item(), "Invalid image dtype or non-finite values.")
    dl.require(isinstance(labels, torch.Tensor) and labels.dtype == torch.long and tuple(labels.shape) == (b,),
               f"Wrong label dtype/shape: {getattr(labels, 'shape', None)}.")
    dl.require(((labels == 0) | (labels == 1)).all().item(), "Labels must be 0 or 1.")
    for name in ["site_uid", "site_id", "sample_id", "split", "strategy", "k"]:
        dl.require(name in batch and len(batch[name]) == b, f"Missing or wrong-length batch field {name}.")
    for i, sample_id in enumerate(batch["sample_id"]):
        dl.require(sample_id in expected_rows.index and sample_id not in seen, "Unknown or repeated sample in loader.")
        row = expected_rows.loc[sample_id]
        dl.require(batch["site_uid"][i] == row["site_uid"] and batch["site_id"][i] == row["site_id"], "Batch site identity mismatch.")
        dl.require(batch["split"][i] == split and batch["strategy"][i] == strategy and int(batch["k"][i]) == k,
                   "Batch split/strategy/K mismatch.")
        dl.require(int(labels[i].item()) == row["label_id"], "Batch label differs from expected site label.")
        seen.add(sample_id)
    return b


def write_results(report_dir, rows, global_errors, settings, sites):
    columns = ["k", "strategy", "split", "status", "samples", "checked_samples", "checked_batches",
               "batch_shape", "labels_shape", "error"]
    checks = pd.DataFrame(rows, columns=columns)
    passed = (not global_errors and len(checks) == len(K_VALUES) * len(STRATEGIES) * len(SPLITS)
              and checks["status"].eq("ok").all() and checks["checked_samples"].eq(checks["samples"]).all())
    report = [
        "# Step 5 Dataset Loader Check Report", "", f"## Overall Result: {'PASS' if passed else 'FAIL'}", "",
        "Every batch in every required K/strategy/split combination must pass. Missing combinations are failures.",
        "", "## Settings", "", *[f"- {name}: {value}" for name, value in settings.items()],
        "- Images and transforms run on CPU. Pinned memory does not move tensors to a GPU.",
        "- No model training, cross-validation, or archaeological-site masking is performed.",
        "", "## Current Step 3 Sites", "", "| Split | Looted | Preserved | Total |", "|---|---:|---:|---:|",
    ]
    if sites is not None:
        counts = sites.groupby(["split", "label"]).size()
        for split in SPLITS:
            looted, preserved = int(counts.get((split, "looted"), 0)), int(counts.get((split, "preserved"), 0))
            report.append(f"| {split} | {looted} | {preserved} | {looted + preserved} |")
    report += [
        "", "## Loader Check Summary", "",
        f"- Successful combinations: {int(checks['status'].eq('ok').sum())}/{len(K_VALUES)*len(STRATEGIES)*len(SPLITS)}",
        f"- Checked sequences, including reuse across experiments: {int(checks['checked_samples'].sum())}",
        "- Image shape: [B, K, 3, image_size, image_size]. B may be smaller in the final batch.",
        "- Label shape: [B], torch.long, values 0/1 and matched to site metadata.",
        "", "| K | Strategy | Split | Status | Available | Checked | Batches | First image batch |",
        "|---:|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        report.append(f"| {row['k']} | {row['strategy']} | {row['split']} | {row['status']} | {row['samples']} | {row['checked_samples']} | {row['checked_batches']} | {row['batch_shape']} |")
    problems = list(global_errors) + [f"K{r['k']}/{r['strategy']}/{r['split']}: {r['error']}" for r in rows if r["status"] != "ok"]
    report += ["", "## Errors", "", *([f"- {p}" for p in problems] or ["- None."]),
               "", "## Scope", "",
               "- Metadata is compared with current Step 3; all selected sequences are loaded when the overall result is PASS.",
               "- Source identity, labeling correctness, reference image-count differences, and scientific suitability of normalization require their separate checks.",
               "- Equal random seeds support repeatable loading in the same environment; they do not guarantee identical model training across platforms.",
               "- PASS confirms this loading workflow, not model accuracy or GPU operation."]
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "step5_loader_checks.csv"
    report_path = report_dir / "step5_loader_check_report.md"
    checks.to_csv(csv_path, index=False)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return passed, report_path


def main(project_root=Path("."), data_root=None, batch_size=4, image_size=266,
         num_workers=0, normalization="imagenet", seed=42, num_threads=1):
    root = Path(project_root).resolve()
    data_root = Path(data_root).resolve() if data_root is not None else root / "data/raw/DAFA_LS"
    report_dir = root / "results/tables"
    rows, errors, sites = [], [], None
    settings = {"Project root": str(root), "Data root": str(data_root), "Batch size": batch_size,
                "Image size": image_size, "Normalization": normalization, "Seed": seed,
                "Workers": num_workers, "CPU tensor threads": num_threads,
                "Coverage": "all batches; no first-batch-only shortcut"}
    try:
        dl.require(batch_size > 0 and image_size > 0 and num_workers >= 0 and num_threads > 0,
                   "Invalid batch size, image size, worker count, or thread count.")
        sites, site_lookup, frame_lookup = load_reference(root)
        dl.require_torch()
        dl.seed_everything(seed)
        dl.torch.set_num_threads(num_threads)
        settings["PyTorch version"] = dl.torch.__version__
        settings["CUDA available (not exercised by this check)"] = dl.torch.cuda.is_available()
    except Exception as exc:
        errors.append(f"Preflight failed: {type(exc).__name__}: {exc}")
    if not errors:
        for k in K_VALUES:
            csv_path = root / "data/processed/kframe_sequences" / f"kframe_samples_K{k}.csv"
            preflight_error = None
            try:
                metadata = dl.read_samples_csv(csv_path)
                check_against_reference(metadata, k, sites, site_lookup, frame_lookup)
            except Exception as exc:
                preflight_error = f"{type(exc).__name__}: {exc}"
            for strategy in STRATEGIES:
                for split in SPLITS:
                    row = {"k": k, "strategy": strategy, "split": split, "status": "error",
                           "samples": 0, "checked_samples": 0, "checked_batches": 0,
                           "batch_shape": "", "labels_shape": "", "error": ""}
                    try:
                        if preflight_error:
                            raise ValueError(preflight_error)
                        expected = metadata[(metadata["strategy"] == strategy) & (metadata["split"] == split)].set_index("sample_id")
                        row["samples"] = len(expected)
                        dl.require(len(expected) > 0, "No samples in a required combination.")
                        loader = dl.create_dataloader(
                            str(csv_path), split, strategy, batch_size=batch_size, image_size=image_size,
                            train=(split == "train"), shuffle=False, num_workers=num_workers,
                            normalization=normalization, data_root=data_root, seed=seed, k=k, pin_memory=False,
                        )
                        seen = set()
                        for batch in loader:
                            b = validate_batch(batch, k, image_size, split, strategy, expected, batch_size, seen)
                            if row["checked_batches"] == 0:
                                row["batch_shape"] = str(list(batch["images"].shape))
                                row["labels_shape"] = str(list(batch["label"].shape))
                            row["checked_samples"] += b
                            row["checked_batches"] += 1
                        dl.require(seen == set(expected.index) and row["checked_samples"] == len(expected),
                                   "Loader did not visit every expected sample exactly once.")
                        dl.require(row["checked_batches"] == (len(expected) + batch_size - 1) // batch_size,
                                   "Wrong number of batches, possibly a dropped final batch.")
                        row["status"] = "ok"
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    rows.append(row)
                    print(f"K{k} {strategy} {split}: {row['status']}; checked {row['checked_samples']}/{row['samples']}", flush=True)
    passed, report_path = write_results(report_dir, rows, errors, settings, sites)
    print(f"\nStep 5 {'PASS' if passed else 'FAIL'}. Report: {report_path}")
    if passed:
        print("All required sequences passed the loader checks. You can proceed to model setup.")
    else:
        print("Fix the reported errors before training. This run is not approved by the checker.")
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check all Step 5 sequences against current Step 3.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=266)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--normalization", choices=["imagenet", "zero_one"], default="imagenet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=1)
    args = parser.parse_args()
    raise SystemExit(main(args.project_root, args.data_root, args.batch_size, args.image_size,
                          args.num_workers, args.normalization, args.seed, args.num_threads))
