# Step 5 Dataset Loader Check Report

## Purpose

This step checks whether K-frame satellite image samples can be loaded correctly as PyTorch tensors for binary looting classification.

## Expected Tensor Shape

Each batch should have this shape:

```text
[batch_size, K, 3, 266, 266]
```

## Loader Check Summary

- K=1, recent_window, train: ok, samples=570, batch=[4, 1, 3, 266, 266]
- K=1, recent_window, val: ok, samples=18, batch=[4, 1, 3, 266, 266]
- K=1, recent_window, test: ok, samples=87, batch=[4, 1, 3, 266, 266]
- K=1, uniform, train: ok, samples=570, batch=[4, 1, 3, 266, 266]
- K=1, uniform, val: ok, samples=18, batch=[4, 1, 3, 266, 266]
- K=1, uniform, test: ok, samples=87, batch=[4, 1, 3, 266, 266]
- K=1, early_late, train: ok, samples=570, batch=[4, 1, 3, 266, 266]
- K=1, early_late, val: ok, samples=18, batch=[4, 1, 3, 266, 266]
- K=1, early_late, test: ok, samples=87, batch=[4, 1, 3, 266, 266]
- K=1, random, train: ok, samples=570, batch=[4, 1, 3, 266, 266]
- K=1, random, val: ok, samples=18, batch=[4, 1, 3, 266, 266]
- K=1, random, test: ok, samples=87, batch=[4, 1, 3, 266, 266]
- K=2, recent_window, train: ok, samples=570, batch=[4, 2, 3, 266, 266]
- K=2, recent_window, val: ok, samples=18, batch=[4, 2, 3, 266, 266]
- K=2, recent_window, test: ok, samples=87, batch=[4, 2, 3, 266, 266]
- K=2, uniform, train: ok, samples=570, batch=[4, 2, 3, 266, 266]
- K=2, uniform, val: ok, samples=18, batch=[4, 2, 3, 266, 266]
- K=2, uniform, test: ok, samples=87, batch=[4, 2, 3, 266, 266]
- K=2, early_late, train: ok, samples=570, batch=[4, 2, 3, 266, 266]
- K=2, early_late, val: ok, samples=18, batch=[4, 2, 3, 266, 266]
- K=2, early_late, test: ok, samples=87, batch=[4, 2, 3, 266, 266]
- K=2, random, train: ok, samples=570, batch=[4, 2, 3, 266, 266]
- K=2, random, val: ok, samples=18, batch=[4, 2, 3, 266, 266]
- K=2, random, test: ok, samples=87, batch=[4, 2, 3, 266, 266]
- K=3, recent_window, train: ok, samples=570, batch=[4, 3, 3, 266, 266]
- K=3, recent_window, val: ok, samples=18, batch=[4, 3, 3, 266, 266]
- K=3, recent_window, test: ok, samples=87, batch=[4, 3, 3, 266, 266]
- K=3, uniform, train: ok, samples=570, batch=[4, 3, 3, 266, 266]
- K=3, uniform, val: ok, samples=18, batch=[4, 3, 3, 266, 266]
- K=3, uniform, test: ok, samples=87, batch=[4, 3, 3, 266, 266]
- K=3, early_late, train: ok, samples=570, batch=[4, 3, 3, 266, 266]
- K=3, early_late, val: ok, samples=18, batch=[4, 3, 3, 266, 266]
- K=3, early_late, test: ok, samples=87, batch=[4, 3, 3, 266, 266]
- K=3, random, train: ok, samples=570, batch=[4, 3, 3, 266, 266]
- K=3, random, val: ok, samples=18, batch=[4, 3, 3, 266, 266]
- K=3, random, test: ok, samples=87, batch=[4, 3, 3, 266, 266]
- K=5, recent_window, train: ok, samples=570, batch=[4, 5, 3, 266, 266]
- K=5, recent_window, val: ok, samples=18, batch=[4, 5, 3, 266, 266]
- K=5, recent_window, test: ok, samples=87, batch=[4, 5, 3, 266, 266]
- K=5, uniform, train: ok, samples=570, batch=[4, 5, 3, 266, 266]
- K=5, uniform, val: ok, samples=18, batch=[4, 5, 3, 266, 266]
- K=5, uniform, test: ok, samples=87, batch=[4, 5, 3, 266, 266]
- K=5, early_late, train: ok, samples=570, batch=[4, 5, 3, 266, 266]
- K=5, early_late, val: ok, samples=18, batch=[4, 5, 3, 266, 266]
- K=5, early_late, test: ok, samples=87, batch=[4, 5, 3, 266, 266]
- K=5, random, train: ok, samples=570, batch=[4, 5, 3, 266, 266]
- K=5, random, val: ok, samples=18, batch=[4, 5, 3, 266, 266]
- K=5, random, test: ok, samples=87, batch=[4, 5, 3, 266, 266]
- K=8, recent_window, train: ok, samples=570, batch=[4, 8, 3, 266, 266]
- K=8, recent_window, val: ok, samples=18, batch=[4, 8, 3, 266, 266]
- K=8, recent_window, test: ok, samples=87, batch=[4, 8, 3, 266, 266]
- K=8, uniform, train: ok, samples=570, batch=[4, 8, 3, 266, 266]
- K=8, uniform, val: ok, samples=18, batch=[4, 8, 3, 266, 266]
- K=8, uniform, test: ok, samples=87, batch=[4, 8, 3, 266, 266]
- K=8, early_late, train: ok, samples=570, batch=[4, 8, 3, 266, 266]
- K=8, early_late, val: ok, samples=18, batch=[4, 8, 3, 266, 266]
- K=8, early_late, test: ok, samples=87, batch=[4, 8, 3, 266, 266]
- K=8, random, train: ok, samples=570, batch=[4, 8, 3, 266, 266]
- K=8, random, val: ok, samples=18, batch=[4, 8, 3, 266, 266]
- K=8, random, test: ok, samples=87, batch=[4, 8, 3, 266, 266]

## Errors

- No errors found.

## Output Files Created

- `results\tables\step5_loader_checks.csv`