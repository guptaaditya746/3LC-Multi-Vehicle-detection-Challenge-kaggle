# Submission Format

## Required CSV Schema

The submission file must contain exactly these columns:

| Column | Description |
|---|---|
| `id` | Row identifier matching `sample_submission.csv` |
| `image_id` | Image identifier matching `sample_submission.csv` |
| `prediction_string` | Concatenated detections or `no box` |

## `prediction_string` Rules

Each detection is a group of six values:

```text
class_id confidence x_center y_center width height
```

Formatting rules:

- use normalized YOLO coordinates in `[0, 1]`
- keep all values on one line
- concatenate multiple detections with spaces
- use `no box` exactly when there are no detections

## Validation Checklist

- [ ] Columns are exactly `id,image_id,prediction_string`
- [ ] Row count matches `sample_submission.csv`
- [ ] `id` values match the sample exactly
- [ ] `image_id` values match the sample exactly
- [ ] Every detection row contains groups of six values
- [ ] `class_id` is an integer from 0 to 3
- [ ] Confidence and box coordinates are floats in `[0, 1]`
- [ ] Rows with no detections use `no box`
- [ ] File passes `scripts/validate_submission.py`

## Common Rejection Reasons

- wrong column order or extra columns
- missing rows
- mismatched `id` or `image_id`
- malformed `prediction_string`
- out-of-range class IDs or coordinates
- using anything other than `no box` for empty predictions
- line breaks inside prediction strings

## Example Rows

```csv
id,image_id,prediction_string
1,frame_000001.jpg,no box
2,frame_000002.jpg,1 0.92 0.4312 0.5520 0.1200 0.0800 0 0.88 0.7120 0.4200 0.1800 0.1400
```

## Recommended Workflow

1. Generate predictions from the latest approved experiment.
2. Build `submission.csv`.
3. Run the validator against `sample_submission.csv`.
4. Log the submission in `experiments/submission_log.csv`.
5. Upload to Kaggle and record the public score.
