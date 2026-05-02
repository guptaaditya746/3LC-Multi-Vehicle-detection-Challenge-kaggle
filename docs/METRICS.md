# Metrics Guide

## mAP@0.5

The competition score is **mean Average Precision at IoU 0.5**. In practice, this measures how well predictions match the ground truth when a predicted box overlaps the correct box by at least 50 percent and has the right class label.

Why it matters here:

- missing boxes reduce recall
- false boxes reduce precision
- badly placed boxes can fail the IoU threshold even when the class is correct

## IoU

**Intersection over Union (IoU)** compares the overlap between a predicted box and a ground-truth box.

Formula:

```text
IoU = area(prediction ∩ ground_truth) / area(prediction ∪ ground_truth)
```

Interpretation:

- higher IoU means tighter localization
- IoU below 0.5 will not count as a true positive for the competition metric

## Precision And Recall

- **Precision** answers: of the boxes we predicted, how many were correct?
- **Recall** answers: of the true objects present, how many did we find?

Label quality directly affects both:

- missing labels can make correct detections look like false positives
- bad boxes can lower IoU and suppress true positives
- wrong classes create avoidable confusion between vehicle types

## Confidence Threshold

Every predicted box has a confidence score in `[0, 1]`.

What to watch:

- too high a threshold can drop valid detections and hurt recall
- too low a threshold can flood the output with false positives
- validation review in 3LC should include low-confidence and high-confidence mistakes

## Why Label Quality Matters

This challenge rewards data-centric improvements. Better labels can improve:

- localization consistency
- class separation
- calibration of confidence scores
- the reliability of local validation metrics

## What To Log After Each Experiment

Record the following after every serious run:

| Field | Why it matters |
|---|---|
| Experiment ID | Stable reference across notes and submissions |
| Date | Ordering and traceability |
| Git commit | Code reproducibility |
| Data revision | Connects model behavior to label state |
| Run name | Human-readable identifier |
| Image size | Must remain 640 px |
| Epochs and batch size | Training budget context |
| Optimizer and learning rate | Hyperparameter trace |
| Local mAP@0.5 | Main offline metric |
| Public Kaggle score | Submission feedback |
| Notes | Failure modes, label fixes, anomalies |

## Recommended Review Questions

- Did the latest label revision improve precision, recall, or both?
- Are gains concentrated in one class or scene type?
- Are local metrics moving in the same direction as Kaggle scores?
- Which failure mode should drive the next 3LC review cycle?
