# Research Foundation

## Title
Looting Detection Using Satellite Images Under Limited Time-Series Conditions

## Research Problem
Archaeological looting causes permanent damage to cultural heritage sites, but many sites are located in remote or insecure areas where regular field inspection is difficult. Satellite image time series can support remote monitoring, but current state-of-the-art approaches depend on long temporal sequences. In practical situations, long satellite image histories may not always be available due to missing acquisitions, cloud cover, limited archives, or incomplete temporal coverage. Therefore, this research investigates how archaeological looting can be detected effectively using only a limited number of satellite image frames.

## Aim
To develop and evaluate a deep-learning-based method for detecting archaeological looting from satellite imagery under limited time-series conditions.

## Objectives
1. Study the DAFA-LS dataset.
2. Prepare limited time-series versions of the dataset.
3. Implement baseline models for short-sequence looting detection.
4. Develop a change-aware lightweight temporal fusion model.
5. Evaluate models using Accuracy, Precision, Recall, F1-score, ROC-AUC, and FAR.
6. Analyze the effect of frame count and frame-selection strategy.

## Dataset
DAFA-LS contains 55,480 satellite images collected monthly from 2016 to 2023 across 675 Afghan archaeological sites, including 135 looted and 540 preserved sites.

## Task
Binary classification:
0 = Preserved
1 = Looted

## Main Research Gap
Existing methods depend on long satellite image time series, but this research focuses on looting detection using limited satellite image frames.

## Main Contribution
A short time-series looting detection method using change-aware features and lightweight temporal fusion.