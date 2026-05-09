# Notebooks

Keep notebooks lightweight and exploratory.

- `00_data_audit.ipynb` checks configured data paths, label quality, class balance, and random boxed examples.
- `01_submission_review.ipynb` validates and visualizes generated Kaggle submissions.
- `02_experiment_workbench.ipynb` creates an ignored local config and runs/logs experiments when requested.

Rules:

- Prefer scripts in `scripts/` or modules in `src/` for anything needed again.
- Avoid storing outputs that make diffs noisy.
- If a notebook supports a key decision, summarize the conclusion in `docs/` or `experiments/`.
