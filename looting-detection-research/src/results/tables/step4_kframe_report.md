# Step 4 K-Frame Dataset Creation Report

## Purpose

This step creates limited satellite image time-series samples for archaeological looting detection under short temporal conditions.

## Input File

- `data\splits\frames_with_splits.csv`

## K Values Used

- K = 1
- K = 2
- K = 3
- K = 5
- K = 8

## Frame Selection Strategies

- `recent_window`: selects the latest K frames.
- `uniform`: selects K frames evenly across the whole time series.
- `early_late`: selects an early reference frame and recent target frames.
- `random`: selects K frames randomly using a fixed seed.

## Main Output Summary

- Total K-frame sample rows created: 13500
- Full-sequence reference rows created: 675
- Skipped rows due to insufficient frames: 0
- Data leakage check: 0 sites appear in more than one split

## Samples by K, Strategy, Split, and Label

- K=1, early_late, test, looted: 26 samples
- K=1, early_late, test, preserved: 61 samples
- K=1, early_late, train, looted: 100 samples
- K=1, early_late, train, preserved: 470 samples
- K=1, early_late, val, looted: 9 samples
- K=1, early_late, val, preserved: 9 samples
- K=1, random, test, looted: 26 samples
- K=1, random, test, preserved: 61 samples
- K=1, random, train, looted: 100 samples
- K=1, random, train, preserved: 470 samples
- K=1, random, val, looted: 9 samples
- K=1, random, val, preserved: 9 samples
- K=1, recent_window, test, looted: 26 samples
- K=1, recent_window, test, preserved: 61 samples
- K=1, recent_window, train, looted: 100 samples
- K=1, recent_window, train, preserved: 470 samples
- K=1, recent_window, val, looted: 9 samples
- K=1, recent_window, val, preserved: 9 samples
- K=1, uniform, test, looted: 26 samples
- K=1, uniform, test, preserved: 61 samples
- K=1, uniform, train, looted: 100 samples
- K=1, uniform, train, preserved: 470 samples
- K=1, uniform, val, looted: 9 samples
- K=1, uniform, val, preserved: 9 samples
- K=2, early_late, test, looted: 26 samples
- K=2, early_late, test, preserved: 61 samples
- K=2, early_late, train, looted: 100 samples
- K=2, early_late, train, preserved: 470 samples
- K=2, early_late, val, looted: 9 samples
- K=2, early_late, val, preserved: 9 samples
- K=2, random, test, looted: 26 samples
- K=2, random, test, preserved: 61 samples
- K=2, random, train, looted: 100 samples
- K=2, random, train, preserved: 470 samples
- K=2, random, val, looted: 9 samples
- K=2, random, val, preserved: 9 samples
- K=2, recent_window, test, looted: 26 samples
- K=2, recent_window, test, preserved: 61 samples
- K=2, recent_window, train, looted: 100 samples
- K=2, recent_window, train, preserved: 470 samples
- K=2, recent_window, val, looted: 9 samples
- K=2, recent_window, val, preserved: 9 samples
- K=2, uniform, test, looted: 26 samples
- K=2, uniform, test, preserved: 61 samples
- K=2, uniform, train, looted: 100 samples
- K=2, uniform, train, preserved: 470 samples
- K=2, uniform, val, looted: 9 samples
- K=2, uniform, val, preserved: 9 samples
- K=3, early_late, test, looted: 26 samples
- K=3, early_late, test, preserved: 61 samples
- K=3, early_late, train, looted: 100 samples
- K=3, early_late, train, preserved: 470 samples
- K=3, early_late, val, looted: 9 samples
- K=3, early_late, val, preserved: 9 samples
- K=3, random, test, looted: 26 samples
- K=3, random, test, preserved: 61 samples
- K=3, random, train, looted: 100 samples
- K=3, random, train, preserved: 470 samples
- K=3, random, val, looted: 9 samples
- K=3, random, val, preserved: 9 samples
- K=3, recent_window, test, looted: 26 samples
- K=3, recent_window, test, preserved: 61 samples
- K=3, recent_window, train, looted: 100 samples
- K=3, recent_window, train, preserved: 470 samples
- K=3, recent_window, val, looted: 9 samples
- K=3, recent_window, val, preserved: 9 samples
- K=3, uniform, test, looted: 26 samples
- K=3, uniform, test, preserved: 61 samples
- K=3, uniform, train, looted: 100 samples
- K=3, uniform, train, preserved: 470 samples
- K=3, uniform, val, looted: 9 samples
- K=3, uniform, val, preserved: 9 samples
- K=5, early_late, test, looted: 26 samples
- K=5, early_late, test, preserved: 61 samples
- K=5, early_late, train, looted: 100 samples
- K=5, early_late, train, preserved: 470 samples
- K=5, early_late, val, looted: 9 samples
- K=5, early_late, val, preserved: 9 samples
- K=5, random, test, looted: 26 samples
- K=5, random, test, preserved: 61 samples
- K=5, random, train, looted: 100 samples
- K=5, random, train, preserved: 470 samples
- K=5, random, val, looted: 9 samples
- K=5, random, val, preserved: 9 samples
- K=5, recent_window, test, looted: 26 samples
- K=5, recent_window, test, preserved: 61 samples
- K=5, recent_window, train, looted: 100 samples
- K=5, recent_window, train, preserved: 470 samples
- K=5, recent_window, val, looted: 9 samples
- K=5, recent_window, val, preserved: 9 samples
- K=5, uniform, test, looted: 26 samples
- K=5, uniform, test, preserved: 61 samples
- K=5, uniform, train, looted: 100 samples
- K=5, uniform, train, preserved: 470 samples
- K=5, uniform, val, looted: 9 samples
- K=5, uniform, val, preserved: 9 samples
- K=8, early_late, test, looted: 26 samples
- K=8, early_late, test, preserved: 61 samples
- K=8, early_late, train, looted: 100 samples
- K=8, early_late, train, preserved: 470 samples
- K=8, early_late, val, looted: 9 samples
- K=8, early_late, val, preserved: 9 samples
- K=8, random, test, looted: 26 samples
- K=8, random, test, preserved: 61 samples
- K=8, random, train, looted: 100 samples
- K=8, random, train, preserved: 470 samples
- K=8, random, val, looted: 9 samples
- K=8, random, val, preserved: 9 samples
- K=8, recent_window, test, looted: 26 samples
- K=8, recent_window, test, preserved: 61 samples
- K=8, recent_window, train, looted: 100 samples
- K=8, recent_window, train, preserved: 470 samples
- K=8, recent_window, val, looted: 9 samples
- K=8, recent_window, val, preserved: 9 samples
- K=8, uniform, test, looted: 26 samples
- K=8, uniform, test, preserved: 61 samples
- K=8, uniform, train, looted: 100 samples
- K=8, uniform, train, preserved: 470 samples
- K=8, uniform, val, looted: 9 samples
- K=8, uniform, val, preserved: 9 samples

## Output Files Created

- `data\processed\kframe_sequences\kframe_samples_all.csv`
- `data\processed\kframe_sequences\full_sequence_samples.csv`
- `data\processed\kframe_sequences\kframe_skipped_samples.csv`
- `data\processed\kframe_sequences\kframe_summary_by_k_strategy.csv`
- `data\processed\kframe_sequences\kframe_samples_K1.csv`
- `data\processed\kframe_sequences\kframe_samples_K2.csv`
- `data\processed\kframe_sequences\kframe_samples_K3.csv`
- `data\processed\kframe_sequences\kframe_samples_K5.csv`
- `data\processed\kframe_sequences\kframe_samples_K8.csv`
- `data\processed\kframe_sequences\kframe_samples_recent_window.csv`
- `data\processed\kframe_sequences\kframe_samples_uniform.csv`
- `data\processed\kframe_sequences\kframe_samples_early_late.csv`
- `data\processed\kframe_sequences\kframe_samples_random.csv`

## Warnings

- No major warnings found.