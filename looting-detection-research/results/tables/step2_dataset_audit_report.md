# Step 2 Dataset Audit Report

## Dataset Path

`data\raw\DAFA_LS`

## CUDA Availability and Audit Device

- Audit processing device: CPU (Pillow and pandas)
- PyTorch version: 2.7.1+cu118
- PyTorch CUDA build: 11.8
- CUDA available to PyTorch: yes
- GPU 0: NVIDIA GeForce GTX 1650 with Max-Q Design
- CUDA is available for later model training. This audit still runs on CPU.

## Important Dataset Files

- fold_dict.json: FOUND
- README.txt: FOUND
- LICENSE.txt: FOUND
- fold_dict.json status: readable
- fold_dict.json keys: ['train', 'val_1', 'val_2', 'val_3', 'val_4', 'val_5', 'test']

## Overall Image Summary

- Total image files found: 64114
- Readable non-mask files: 63439
- Valid non-mask frames with recognised site folders: 63439
- Mask files detected by name: 675
- Corrupted/unreadable files: 0
- Readable non-mask files excluded for invalid site paths: 0

## Site Summary

- Unique sites detected using site_uid: 675
- Looted sites detected: 135
- Preserved sites detected: 540
- Bare site IDs reused across classes: 135
- A site_uid includes the class folder: looted/0 and preserved/0 are separate identifiers.
- These are folder identities; confirm their correspondence to physical sites using dataset metadata.

## Frame Count Per Site

- Minimum frames per site: 81
- Maximum frames per site: 95
- Mean frames per site: 93.98
- Median frames per site: 94

## Temporal Coverage

- unique_months counts distinct year-month dates, such as 2016-01 and 2017-01.
- unique_month_numbers counts different month numbers (1 to 12).
- Retained frames without a usable year-month: 0
- Retained frames with an invalid calendar date: 0
- Extra frames within an already represented site/month: 0

## Image Size Summary

- 266 x 266: 63439 images
- Retained frames that are not three-channel RGB: 0

## Expected Reference Values

- Expected images: 55480
- Expected sites: 675
- Expected looted sites: 135
- Expected preserved sites: 540
- Expected image size: 266 x 266

## Warnings

- WARNING: Found 63439 usable frames; the reference count is 55480. Investigate dataset version, dates, and file contents. Do not delete files just to match the reference.

## Scope and Next Step

- Image files are read only. Masks are recognised by 'mask' in the dataset-relative path.
- No image-content deduplication, label verification, or cloud-quality assessment is performed.
- This step does not choose train/validation/test splits or validate fold membership.
- Update Step 03 to use site_uid throughout. The old Step 03 still has split and merge bugs.
- When reading the inventory, use pd.read_csv(path, dtype={"site_id": "string", "site_uid": "string"}).

## Created Output Files

- `results\tables\step2_dataset_inventory.csv`
- `results\tables\step2_site_frame_summary.csv`
- `results\tables\step2_dataset_audit_report.md`
