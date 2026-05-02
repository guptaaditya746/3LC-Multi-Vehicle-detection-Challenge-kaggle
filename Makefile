PYTHON ?= python
SAMPLE_SUBMISSION ?= data/sample_submission.csv
SUBMISSION ?= submission.csv
LABELS_DIR ?= data/train/labels
CLASSES ?= truck,car,van,bus

.PHONY: install install-gpu-cu121 verify service validate-submission summarize-labels

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

install-gpu-cu121:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
	$(PYTHON) -m pip install -r requirements-gpu-cu121.txt

verify:
	$(PYTHON) -m py_compile scripts/validate_submission.py scripts/summarize_yolo_dataset.py scripts/make_experiment_entry.py

service:
	3lc service

validate-submission:
	$(PYTHON) scripts/validate_submission.py --sample $(SAMPLE_SUBMISSION) --submission $(SUBMISSION)

summarize-labels:
	$(PYTHON) scripts/summarize_yolo_dataset.py --labels $(LABELS_DIR) --classes "$(CLASSES)"

# Add official Kaggle starter-kit targets later when the organizer scripts are available.
