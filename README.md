# 3LC Multi-Vehicle Detection Challenge

Data-centric object detection repository for the Kaggle-based **3LC Multi-Vehicle Detection Challenge**. This project is built around a simple constraint set: train **YOLOv8n from scratch** at **640 px**, inspect model behavior in **3LC**, fix labels, retrain, and submit.

## Team

| Field | Value |
|---|---|
| Team name | TODO |
| Member name | TODO |
| Kaggle username | TODO |
| Contact | TODO |
| Repository visibility | Private until final judging |

## Competition Objective

Detect on-road vehicles in traffic-camera images using the four competition classes below. The official evaluation metric is **mAP@0.5** on a hidden test set.

## Data-Centric Workflow

```text
train -> inspect in 3LC -> fix labels -> retrain -> submit -> repeat
```

Recommended loop:

1. Register train and validation data in 3LC tables.
2. Train a YOLOv8n baseline from scratch.
3. Inspect predictions and failure modes in the 3LC Dashboard.
4. Fix label issues through table revisions.
5. Retrain on the latest revisions.
6. Predict on the hidden-test image set.
7. Validate `submission.csv`.
8. Upload to Kaggle and document the result.

## Class Mapping

| Class ID | Class name |
|---:|---|
| 0 | truck |
| 1 | car |
| 2 | van |
| 3 | bus |

## Repository Layout

```text
.
├── README.md
├── LICENSE
├── .gitignore
├── Makefile
├── requirements.txt
├── requirements-gpu-cu121.txt
├── config/
│   └── config.example.yaml
├── docs/
│   ├── COMPETITION.md
│   ├── DATA_CARD.md
│   ├── LABELING_PROTOCOL.md
│   ├── 3LC_WORKFLOW.md
│   ├── METRICS.md
│   ├── SUBMISSION.md
│   └── FINAL_REPORT_TEMPLATE.md
├── experiments/
│   ├── README.md
│   ├── experiment_log.csv
│   └── submission_log.csv
├── notebooks/
│   └── README.md
├── scripts/
│   ├── validate_submission.py
│   ├── summarize_yolo_dataset.py
│   └── make_experiment_entry.py
├── src/
│   └── README.md
├── reports/
│   ├── figures/
│   │   └── .gitkeep
│   └── screenshots/
│       └── .gitkeep
└── .github/
    ├── pull_request_template.md
    └── ISSUE_TEMPLATE/
        ├── label_issue.md
        ├── experiment.md
        └── submission.md
```

Local-only working directories such as `data/`, `runs/`, `models/`, `weights/`, and `submissions/` are intentionally ignored and may exist outside the tracked scaffold.

## Environment Setup

### Requirements

- Python 3.9+
- Manual access to competition data from Kaggle
- 3LC account and local service access

### Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### CPU install

```bash
make install
```

### CUDA 12.1 GPU install

```bash
make install-gpu-cu121
```

### 3LC login

```bash
3lc login
```

### Start the 3LC local service

```bash
make service
```

## Reproduction Checklist

- [ ] Download the competition data manually from Kaggle.
- [ ] Copy official starter-kit scripts into this repo if the organizers release or require them.
- [ ] Copy `config/config.example.yaml` to `config/config.yaml` and fill in local paths.
- [ ] Register train and validation tables in 3LC.
- [ ] Train the YOLOv8n baseline from scratch.
- [ ] Inspect predictions in the 3LC Dashboard.
- [ ] Apply label fixes through table revisions.
- [ ] Retrain on the latest data revision.
- [ ] Predict on test images.
- [ ] Validate `submission.csv` with `make validate-submission`.
- [ ] Upload the validated file to Kaggle.
- [ ] Record experiment details, leaderboard scores, and revision notes.

## Constraints Checklist

- [x] YOLOv8n only
- [x] Train from scratch only
- [x] Input size fixed at 640 px
- [x] No pretrained weights
- [x] No ensembles
- [x] No TTA
- [x] No pseudo-labels

## Current Status

- [ ] TODO: add the official Kaggle competition link and starter assets if they become available.
- [ ] TODO: register the first train and validation tables in 3LC.
- [ ] TODO: train the first scratch baseline.
- [ ] TODO: capture first-pass error analysis screenshots.
- [ ] TODO: document the first label revision.
- [ ] TODO: produce the first valid Kaggle submission.

## Metrics Placeholder

| Experiment ID | Data revision | Epochs | Local mAP@0.5 | Public score | Notes |
|---|---|---:|---:|---:|---|
| TODO | TODO | TODO | TODO | TODO | Baseline pending |

## Submission Placeholder

| Submission ID | Experiment ID | Date | Public score | Private score | Notes |
|---|---|---|---:|---:|---|
| TODO | TODO | TODO | TODO | TODO | No submissions recorded yet |

## Links And Resources

- Kaggle competition: TODO
- Official starter kit: TODO
- 3LC documentation: TODO
- 3LC Dashboard guide: TODO
- PyTorch install page: https://pytorch.org/get-started/locally/
- Ultralytics documentation: TODO
- Team communication or Discord: TODO

## Security And Data Notes

- Do not commit raw competition data.
- Do not commit trained weights unless intentionally curated for sharing.
- Do not commit API keys, `kaggle.json`, `.env` files, or local 3LC artifacts.
- Do not commit raw submissions unless they are intentionally sanitized and part of the final evidence package.
- The repository is released under the MIT License, but the competition dataset may have separate non-commercial, research-only, or platform-specific restrictions that are **not** covered by the repo license.
