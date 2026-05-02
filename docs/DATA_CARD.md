# Data Card Template

Use this document to track what data was used, what issues were found, and how the dataset changed across iterations.

## Dataset Overview

| Field | Value |
|---|---|
| Dataset source | TODO |
| Competition split package version | TODO |
| Download date | TODO |
| Storage location | TODO |
| Notes | TODO |

## Splits

| Split | Images | Label files | Notes |
|---|---:|---:|---|
| Train | TODO | TODO | TODO |
| Validation | TODO | TODO | TODO |
| Test | TODO | N/A | Hidden labels |

## Label Format

- Annotation format: YOLO normalized bounding boxes
- Row structure: `class_id x_center y_center width height`
- Class mapping:

| Class ID | Class name |
|---:|---|
| 0 | truck |
| 1 | car |
| 2 | van |
| 3 | bus |

## Known Label Issues

| Issue ID | Type | Split | Severity | Description | Status |
|---|---|---|---|---|---|
| TODO | missing box | train | TODO | TODO | open |

## Duplicate Images

| Duplicate group | Split(s) | Detection method | Action taken | Notes |
|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO |

## Class Imbalance

| Class | Count | Relative frequency | Notes |
|---|---:|---:|---|
| truck | TODO | TODO | TODO |
| car | TODO | TODO | TODO |
| van | TODO | TODO | TODO |
| bus | TODO | TODO | TODO |

## Difficult Cases

- Heavy occlusion: TODO
- Tiny or distant vehicles: TODO
- Motion blur or compression artifacts: TODO
- Truncated objects at image borders: TODO
- Ambiguous car versus van cases: TODO
- Dense traffic scenes: TODO

## 3LC Table Revision History

| Revision ID | Date | Split | Summary of changes | Images touched | Reviewer | Notes |
|---|---|---|---|---:|---|---|
| TODO | TODO | train | TODO | TODO | TODO | TODO |

## Open Questions

- TODO: decide whether near-duplicate images need explicit filtering
- TODO: quantify class imbalance after the first labeling cleanup
- TODO: document any test-set assumptions that may affect submission strategy
