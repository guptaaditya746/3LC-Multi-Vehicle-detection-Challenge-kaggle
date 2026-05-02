# 3LC Workflow

This repository is organized around a repeated 3LC inspection loop rather than one-off model training.

## Core 3LC Concepts

### Tables

Use tables to register train and validation data so images, labels, embeddings, and predictions are all inspectable in one place.

Recommended naming:

- `vehicle_train`
- `vehicle_val`

### Revisions

Use table revisions to version label changes over time. Each revision should correspond to a documented cleanup pass or a focused annotation campaign.

Good revision examples:

- fix missing buses in crowded validation scenes
- remove false positives caused by parked roadside objects
- tighten large loose boxes on trucks

### Runs

Treat each training or inference job as a run with linked metadata:

- code version or git commit
- data revision used
- hyperparameters
- output predictions
- local metrics

### Dashboard

The Dashboard is the main place to inspect:

- false positives
- false negatives
- poor localization
- class confusion
- duplicate or suspicious images

## Useful Filters

Suggested filtering views to save and revisit:

- false positives at moderate or high confidence
- false negatives on validation scenes with visible vehicles
- low-confidence detections near decision thresholds
- low-IoU predictions where class is correct but box placement is weak
- images with many objects or dense traffic
- images touched in the latest label revision

## Suggested Iteration Loop

1. Train the current baseline on the latest train revision.
2. Run validation predictions and register the outputs in 3LC.
3. Sort failure cases by confidence, IoU, and class.
4. Inspect clusters of similar errors.
5. Decide whether the issue is mostly label quality, split quality, or model limitation.
6. Apply targeted label fixes through a new table revision.
7. Retrain and compare local mAP@0.5 and Kaggle score.
8. Record what improved and what did not.

## What To Save For Final Judging

Save screenshots or exports that show:

- example false positives before cleanup
- example false negatives before cleanup
- annotation fixes across revisions
- a table or view showing revision history
- run comparison views
- any cluster or embedding view that motivated a data-cleaning decision

## Minimal Evidence Checklist

- [ ] Table names and revision IDs recorded
- [ ] Training runs linked to data revisions
- [ ] Screenshots saved under `reports/screenshots/`
- [ ] Key findings summarized in `docs/DATA_CARD.md`
- [ ] Experiment and submission logs updated
