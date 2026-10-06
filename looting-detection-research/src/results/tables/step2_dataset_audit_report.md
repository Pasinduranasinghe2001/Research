# Step 2 Dataset Audit Report

## Dataset Path

`data\raw\DAFA_LS`

## Important Dataset Files

- fold_dict.json: FOUND
- README.txt: FOUND
- LICENSE.txt: FOUND
- fold_dict.json status: readable
- fold_dict.json keys: ['train', 'val_1', 'val_2', 'val_3', 'val_4', 'val_5', 'test']

## Overall Image Summary

- Total image files found: 64114
- Valid non-mask frames: 63439
- Mask files detected: 675
- Corrupted/unreadable files: 0

## Site Summary

- Unique sites detected: 540
- Looted sites detected: 135
- Preserved sites detected: 540
- Unknown-label sites detected: 0

## Frame Count Per Site

- Minimum frames per site: 81
- Maximum frames per site: 95
- Mean frames per site: 93.98
- Median frames per site: 94

## Image Size Summary

- 266 x 266: 63439 images

## Expected Full DAFA-LS Values

- Expected images: 55480
- Expected sites: 675
- Expected looted sites: 135
- Expected preserved sites: 540
- Expected image size: 266 x 266

## Warnings

- WARNING: Valid frame count is 63439, expected 55480. Check whether masks are separated, whether extra files exist, or whether the dataset version differs.
- WARNING: Detected site count is 540, expected 675. Check folder depth and site ID detection.

## Created Output Files

- `results\tables\step2_dataset_inventory.csv`
- `results\tables\step2_site_frame_summary.csv`