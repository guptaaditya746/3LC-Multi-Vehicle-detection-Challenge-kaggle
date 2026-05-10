# TODO: Experiment Ideas

Use this as the working backlog. Move useful findings into `experiments/experiment_log.csv`, `experiments/submission_log.csv`, and the docs when they affect decisions.

## Immediate Setup

- [ ] Put the raw starter data under `data/competition_starter/`.
- [ ] Install the environment with `make install-gpu-cu121` or `make install`.
- [ ] Run `make verify` and `make verify-setup`.
- [ ] Open `notebooks/00_data_audit.ipynb` and record class counts plus obvious label issues.
- [ ] Run `notebooks/03_validation_map_failure_analysis.ipynb` after the first training run.
- [ ] Run `notebooks/04_label_quality_mining.ipynb` before the first 3LC cleanup pass.
- [ ] Use `notebooks/05_run_comparison_action_plan.ipynb` before choosing the next run.
- [ ] Register train/val tables with `make register-tables`.

## Baseline

- [ ] Train `exp001_baseline` with the tracked `config/competition.yaml`.
- [ ] Generate `submissions/submission.csv` with `make predict`.
- [ ] Validate with `make validate-submission`.
- [ ] Submit with `make kaggle-submit KAGGLE_MESSAGE="exp001 baseline"`.
- [ ] Review predictions in `notebooks/01_submission_review.ipynb`.
- [ ] Log the run with `scripts/make_experiment_entry.py`.

## Data-Centric Passes

- [ ] In 3LC, filter high-confidence false positives and check for missing labels.
- [ ] In 3LC, filter low-IoU detections and tighten loose boxes.
- [ ] Review class confusion: car vs van, truck vs bus.
- [ ] Inspect crowded scenes for missed small vehicles.
- [ ] Create a named 3LC revision for each cleanup pass and record the revision ID.
- [ ] Retrain after each meaningful label revision; avoid mixing multiple unlabeled changes in one experiment.

## Training Variants Within Rules

- [ ] Try longer training: 20, 40, and 60 epochs with the same data revision.
- [ ] Try smaller/larger batch sizes that fit GPU memory.
- [ ] Try lower `lr0` values such as `0.005` and `0.002`.
- [ ] Compare augmentation on/off after the baseline is stable.
- [ ] Keep `model: yolov8n`, `image_size: 640`, and no pretrained weights.

## Prediction Variants

- [ ] Sweep confidence thresholds: `0.10`, `0.15`, `0.20`, `0.25`, `0.30`.
- [ ] Sweep IoU thresholds: `0.50`, `0.60`, `0.70`.
- [ ] Compare `predict.pipeline: memory` vs `txt` only if outputs look suspicious.
- [ ] Check whether missed detections or false positives dominate before choosing the final threshold.

## Documentation And Final Package

- [ ] Update `docs/DATA_CARD.md` after every label cleanup pass.
- [ ] Save useful 3LC screenshots under `reports/screenshots/`.
- [ ] Record every Kaggle upload in `experiments/submission_log.csv`.
- [ ] Fill `docs/FINAL_REPORT_TEMPLATE.md` before final submission.
- [ ] Confirm the final repo contains code, configs, logs, and evidence but no raw data, weights, API keys, or local artifacts.
