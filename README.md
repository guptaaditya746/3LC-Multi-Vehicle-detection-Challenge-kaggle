# 3LC Multi-Vehicle Detection Challenge

Data-centric object detection workflow for the Kaggle-based **3LC Multi-Vehicle Detection Challenge**.
The repository now keeps source code and configuration at the repo level while the raw starter data stays under `data/competition_starter/`.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
make install-gpu-cu121        # or: make install

make verify                   # syntax check repo scripts
make verify-setup             # check packages, GPU, config, and data files
3lc login                     # one-time per machine
make register-tables          # create/reuse 3LC train + val tables
make train                    # train YOLOv8n from scratch
make predict                  # write submissions/submission.csv
make validate-submission
```

All workflow commands use `config/competition.yaml` by default. Override with:

```bash
make train CONFIG=config/config.yaml
```

## Where Things Go

```text
.
├── config/
│   ├── competition.yaml       # tracked default workflow config
│   ├── config.example.yaml    # private override template
│   └── dataset.yaml           # YOLO dataset file pointing at data/competition_starter
├── data/
│   └── competition_starter/   # ignored; raw competition starter data stays here
│       ├── sample_submission.csv
│       └── data/
│           ├── train/images
│           ├── train/labels
│           ├── val/images
│           ├── val/labels
│           └── test/images
├── scripts/
│   ├── verify_setup.py
│   ├── register_tables.py
│   ├── train.py
│   ├── predict.py
│   ├── validate_submission.py
│   ├── summarize_yolo_dataset.py
│   └── make_experiment_entry.py
├── src/ua_detrac_starter/     # shared config + 3LC compatibility helpers
├── experiments/               # experiment and submission logs
├── docs/                      # workflow, labeling, metrics, and report notes
├── runs/                      # ignored training outputs
└── submissions/               # ignored Kaggle CSV outputs
```

The old self-contained starter-kit files can remain in `data/competition_starter/`, but the canonical scripts are now the repo-level files in `scripts/`.

## Workflow

1. Put the downloaded starter data under `data/competition_starter/`.
2. Confirm paths in `config/competition.yaml`.
3. Run `make verify-setup` and fix any `[FAIL]` items.
4. Run `make register-tables` to create or reuse 3LC train/val tables.
5. Run `make train`; weights are written to `runs/detect/<run_name>/weights/best.pt`.
6. Run `make predict`; predictions are written to `submissions/submission.csv`.
7. Run `make validate-submission` before uploading to Kaggle.
8. Record experiments in `experiments/experiment_log.csv` and submissions in `experiments/submission_log.csv`.

## Constraints

- Model architecture: **YOLOv8n only**
- Initialization: **from scratch** via `yolov8n.yaml`
- Input size: **640 px**
- Not allowed: pretrained weights, ensembles, TTA, pseudo-labels
- Test images are not registered in 3LC; they are read directly from disk by `scripts/predict.py`

## Submission Format

`submissions/submission.csv` must contain exactly:

```text
id,image_id,prediction_string
```

Each `prediction_string` is repeated groups of:

```text
class_id confidence x_center y_center width height
```

Use `no box` for images without detections.

## Class Mapping

| Class ID | Class name |
|---:|---|
| 0 | truck |
| 1 | car |
| 2 | van |
| 3 | bus |

## Security And Data Notes

- Do not commit raw competition data, trained weights, submissions, API keys, `kaggle.json`, or local 3LC artifacts.
- `data/`, `runs/`, `submissions/`, `predictions/`, model weights, and private config overrides are ignored.
- Follow the official Kaggle competition rules and dataset license.
