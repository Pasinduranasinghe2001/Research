from pathlib import Path
import pandas as pd
import torch

from dataset_loader import create_dataloader


K_VALUES = [1, 2, 3, 5, 8]
STRATEGIES = ["recent_window", "uniform", "early_late", "random"]
SPLITS = ["train", "val", "test"]

KFRAME_DIR = Path("data/processed/kframe_sequences")
REPORT_DIR = Path("results/tables")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 4
IMAGE_SIZE = 266


def main():
    report = []

    report.append("# Step 5 Dataset Loader Check Report\n")

    report.append("## Purpose\n")
    report.append(
        "This step checks whether K-frame satellite image samples can be loaded "
        "correctly as PyTorch tensors for binary looting classification.\n"
    )

    report.append("## Expected Tensor Shape\n")
    report.append("Each batch should have this shape:\n")
    report.append("```text")
    report.append("[batch_size, K, 3, 266, 266]")
    report.append("```\n")

    all_checks = []
    errors = []

    for k in K_VALUES:
        csv_path = KFRAME_DIR / f"kframe_samples_K{k}.csv"

        if not csv_path.exists():
            errors.append(f"Missing CSV file: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        for strategy in STRATEGIES:
            for split in SPLITS:
                subset = df[
                    (df["strategy"].astype(str).str.lower() == strategy.lower()) &
                    (df["split"].astype(str).str.lower() == split.lower())
                ]

                if len(subset) == 0:
                    all_checks.append({
                        "k": k,
                        "strategy": strategy,
                        "split": split,
                        "status": "skipped_no_samples",
                        "samples": 0,
                        "batch_shape": "",
                        "labels_shape": ""
                    })
                    continue

                try:
                    loader = create_dataloader(
                        csv_path=str(csv_path),
                        split=split,
                        strategy=strategy,
                        batch_size=BATCH_SIZE,
                        image_size=IMAGE_SIZE,
                        train=(split == "train"),
                        shuffle=False,
                        num_workers=0,
                        normalization="imagenet"
                    )

                    batch = next(iter(loader))

                    images = batch["images"]
                    labels = batch["label"]

                    correct_shape = (
                        images.ndim == 5 and
                        images.shape[1] == k and
                        images.shape[2] == 3 and
                        images.shape[3] == IMAGE_SIZE and
                        images.shape[4] == IMAGE_SIZE
                    )

                    status = "ok" if correct_shape else "wrong_shape"

                    all_checks.append({
                        "k": k,
                        "strategy": strategy,
                        "split": split,
                        "status": status,
                        "samples": len(subset),
                        "batch_shape": str(list(images.shape)),
                        "labels_shape": str(list(labels.shape))
                    })

                except Exception as e:
                    errors.append(
                        f"K={k}, strategy={strategy}, split={split}: {str(e)}"
                    )

                    all_checks.append({
                        "k": k,
                        "strategy": strategy,
                        "split": split,
                        "status": "error",
                        "samples": len(subset),
                        "batch_shape": "",
                        "labels_shape": ""
                    })

    checks_df = pd.DataFrame(all_checks)
    checks_csv_path = REPORT_DIR / "step5_loader_checks.csv"
    checks_df.to_csv(checks_csv_path, index=False)

    report.append("## Loader Check Summary\n")

    if len(checks_df) > 0:
        for _, row in checks_df.iterrows():
            report.append(
                f"- K={row['k']}, {row['strategy']}, {row['split']}: "
                f"{row['status']}, samples={row['samples']}, "
                f"batch={row['batch_shape']}"
            )
    else:
        report.append("- No loader checks were completed.")

    report.append("\n## Errors\n")

    if errors:
        for error in errors:
            report.append(f"- ERROR: {error}")
    else:
        report.append("- No errors found.")

    report.append("\n## Output Files Created\n")
    report.append(f"- `{checks_csv_path}`")

    report_path = REPORT_DIR / "step5_loader_check_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\nStep 5 completed.")
    print(f"Loader check CSV: {checks_csv_path}")
    print(f"Loader check report: {report_path}")

    if errors:
        print("\nErrors found. Open the report and fix them before Step 6.")
    else:
        print("\nNo loader errors found. You can continue to Step 6.")


if __name__ == "__main__":
    main()