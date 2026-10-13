# Step 3 Preprocessing and Split Preparation Report

## Input Files

- Dataset root: `E:\research_university\Research\looting-detection-research\data\raw\DAFA_LS`
- Step 2 inventory: `E:\research_university\Research\looting-detection-research\results\tables\step2_dataset_inventory.csv`
- fold_dict.json: `E:\research_university\Research\looting-detection-research\data\raw\DAFA_LS\fold_dict.json`
- Device: CPU (metadata processing)

## Split Policy

- Method: fixed holdout using validated fold_dict.json entries.
- Test: all sites in the test partition, preserving both classes.
- Validation: all sites in val_1, preserving both classes.
- Training source partitions: train, val_2, val_3, val_4, val_5.
- This is one split, not five-fold cross-validation.

## Clean Dataset Summary

- Valid satellite frames used: 63439
- Excluded mask/unreadable/invalid-site inventory rows: 675
- Unique sites using site_uid: 675
- Looted sites: 135
- Preserved sites: 540
- Minimum frames per site: 81
- Maximum frames per site: 95
- Mean frames per site: 93.98
- Median frames per site: 94

## Split Summary

| Split | Label | Sites | Frames |
|---|---|---:|---:|
| train | looted | 100 | 9441 |
| train | preserved | 496 | 46540 |
| val | looted | 9 | 852 |
| val | preserved | 9 | 846 |
| test | looted | 26 | 2453 |
| test | preserved | 35 | 3307 |

## Validated Checks

- Official rows parsed: 675; exactly one entry per site_uid.
- All retained sites occur in fold_dict.json: PASS.
- Complete selected validation and test memberships preserved: PASS.
- Sites appearing in more than one final split: 0.
- Frames without a split: 0.
- Duplicate image paths in input and final frame tables: 0.
- Frame rows before merging: 63439; after merging: 63439.
- Frame indices restart at 0 and are consecutive within each site_uid: PASS.
- Retained frames with missing year/month or invalid calendar dates: 0.
- Images not 266 x 266: 0.
- Frames not three-channel RGB: 0.

## Warnings

- WARNING: Found 63439 images; the reference count is 55480. Investigate the difference.

## Scope and Next Step

- Images were not reopened, resized, normalized, or used to train a model.
- Site identity and image properties come from the corrected Step 02 inventory.
- These checks do not establish that image contents, labels, or geographic sites are unique/correct.
- The unresolved reference image-count difference is not hidden or automatically repaired.
- Later loaders must group by site_uid and read site_id/site_uid as text.
- Do not pass site_uid or its embedded class name as an input feature to the model.

## Output Files Created

- `E:\research_university\Research\looting-detection-research\data\processed\metadata_all_frames.csv`
- `E:\research_university\Research\looting-detection-research\data\processed\metadata_sites.csv`
- `E:\research_university\Research\looting-detection-research\data\processed\preprocessing_config.json`
- `E:\research_university\Research\looting-detection-research\data\splits\site_splits_official_long.csv`
- `E:\research_university\Research\looting-detection-research\data\splits\site_splits.csv`
- `E:\research_university\Research\looting-detection-research\data\splits\frames_with_splits.csv`
- `E:\research_university\Research\looting-detection-research\results\tables\step3_preprocessing_report.md`
