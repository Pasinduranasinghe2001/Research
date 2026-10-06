# Step 3 Preprocessing and Split Preparation Report

## Input Files

- Dataset root: `data\raw\DAFA_LS`
- Step 2 inventory: `results\tables\step2_dataset_inventory.csv`
- fold_dict.json: `data\raw\DAFA_LS\fold_dict.json`
- fold_dict status: `readable_parsed_rows_675`
- split method used: `official_fold_dict_parsed`

## Clean Dataset Summary

- Valid satellite frames used: 63439
- Unique sites: 540
- Looted sites: 135
- Preserved sites: 540
- Minimum frames per site: 81
- Maximum frames per site: 95
- Mean frames per site: 93.98
- Median frames per site: 94

## Split Summary

- test / looted: 26 sites, 2453 frames
- test / preserved: 61 sites, 5752 frames
- train / looted: 100 sites, 9441 frames
- train / preserved: 470 sites, 44097 frames
- val / looted: 9 sites, 852 frames
- val / preserved: 9 sites, 844 frames

## Checks

- Images not 266x266: 0
- Frames with missing year or month: 0
- Data leakage check: 0 sites appear in more than one split

## Expected DAFA-LS Reference Values

- Expected total images: 55480
- Expected total sites: 675
- Expected looted sites: 135
- Expected preserved sites: 540

## Warnings

- WARNING: Detected 540 sites, expected 675. Check Step 2 site_id detection.

## Output Files Created

- `data\processed\metadata_all_frames.csv`
- `data\processed\metadata_sites.csv`
- `data\processed\preprocessing_config.json`
- `data\splits\site_splits.csv`
- `data\splits\frames_with_splits.csv`
- `data\splits\site_splits_official_long.csv`