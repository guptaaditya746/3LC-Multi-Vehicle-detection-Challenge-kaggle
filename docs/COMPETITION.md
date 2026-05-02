# Competition Summary

## Challenge Goal

Build a vehicle detector for traffic-camera images under a strict model budget. The target workflow is intentionally data-centric: use **YOLOv8n from scratch**, inspect outputs in **3LC**, repair labels, retrain, and improve submission quality over repeated cycles.

## Classes

| Class ID | Class name |
|---:|---|
| 0 | truck |
| 1 | car |
| 2 | van |
| 3 | bus |

## Metric

- Primary competition metric: **mAP@0.5**
- Evaluation set: hidden Kaggle test set
- Practical implication: both localization and class correctness matter, but IoU is judged at a single threshold of 0.5

## Allowed Workflow

1. Register training and validation data in 3LC tables.
2. Train **YOLOv8n** from scratch without pretrained weights.
3. Inspect predictions in the 3LC Dashboard.
4. Fix label issues through table revisions.
5. Retrain with the latest data revision.
6. Predict on test images.
7. Submit `submission.csv` to Kaggle.
8. Repeat with documented evidence.

## Important Deadlines

- Competition start: TODO
- Team registration deadline: TODO
- Final submission deadline: TODO
- Final repo hand-in deadline: TODO

Always confirm dates on Kaggle before planning late-stage work.

## Judging Expectations

Final judging should be prepared to show:

- reproducible training and inference code
- clear experiment logs
- data-centric iteration notes
- evidence of 3LC usage
- label-fix rationale and revisions
- valid Kaggle submissions
- a brief final write-up

## Rules Checklist

- [x] Use YOLOv8n only
- [x] Train from scratch
- [x] Use 640 px input resolution
- [x] No pretrained weights
- [x] No ensembles
- [x] No test-time augmentation
- [x] No pseudo-labeling
- [x] Keep the workflow centered on 3LC-based inspection and revision
- [ ] Confirm any extra organizer rules in the official competition page
