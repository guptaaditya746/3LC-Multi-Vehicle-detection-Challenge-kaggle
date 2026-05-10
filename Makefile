PYTHON ?= python3
CONFIG ?= config/competition.yaml
SAMPLE_SUBMISSION ?= data/competition_starter/sample_submission.csv
SUBMISSION ?= submissions/submission.csv
LABELS_DIR ?= data/competition_starter/data/train/labels
CLASSES ?= truck,car,van,bus
KAGGLE_COMPETITION ?= 3lc-multi-vehicle-detection-challenge
KAGGLE_MESSAGE ?= $(shell git rev-parse --short HEAD 2>/dev/null || date +%Y-%m-%d) $(notdir $(SUBMISSION))
SCRIPTS := $(wildcard scripts/*.py)

.PHONY: install install-gpu-cu121 verify verify-setup service register-tables train predict validate-submission kaggle-submit summarize-labels

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

install-gpu-cu121:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
	$(PYTHON) -m pip install -r requirements-gpu-cu121.txt

verify:
	$(PYTHON) -m py_compile $(SCRIPTS)

verify-setup:
	$(PYTHON) scripts/verify_setup.py --config $(CONFIG)

service:
	3lc service

register-tables:
	$(PYTHON) scripts/register_tables.py --config $(CONFIG)

train:
	$(PYTHON) scripts/train.py --config $(CONFIG)

predict:
	$(PYTHON) scripts/predict.py --config $(CONFIG)

validate-submission:
	$(PYTHON) scripts/validate_submission.py --sample $(SAMPLE_SUBMISSION) --submission $(SUBMISSION)

kaggle-submit: validate-submission
	@command -v kaggle >/dev/null || { echo "ERROR: kaggle CLI not found. Install it with: $(PYTHON) -m pip install kaggle"; exit 1; }
	@test -n "$(KAGGLE_COMPETITION)" || { echo "ERROR: set KAGGLE_COMPETITION=<competition-slug>"; exit 1; }
	kaggle competitions submit -c $(KAGGLE_COMPETITION) -f $(SUBMISSION) -m "$(KAGGLE_MESSAGE)"

summarize-labels:
	$(PYTHON) scripts/summarize_yolo_dataset.py --labels $(LABELS_DIR) --classes "$(CLASSES)"
