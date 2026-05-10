# Notebooks

Keep notebooks lightweight and exploratory.

- `00_data_audit.ipynb` checks configured data paths, label quality, class balance, and random boxed examples.
- `01_submission_review.ipynb` validates and visualizes generated Kaggle submissions.
- `02_experiment_workbench.ipynb` creates an ignored local config and runs/logs experiments when requested.
- `03_validation_map_failure_analysis.ipynb` computes validation mAP-style errors and visualizes false positives, false negatives, localization errors, and class confusion.
- `04_label_quality_mining.ipynb` finds suspicious labels and high-value 3LC review buckets.
- `05_run_comparison_action_plan.ipynb` compares training curves and turns evidence into the next experiment plan.

Rules:

- Prefer scripts in `scripts/` or modules in `src/` for anything needed again.
- Avoid storing outputs that make diffs noisy.
- If a notebook supports a key decision, summarize the conclusion in `docs/` or `experiments/`.
