#!/usr/bin/env python3
"""Train YOLOv8n from scratch on the latest 3LC train/val table revisions."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import tlc
from tlc_ultralytics import YOLO, Settings


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ua_detrac_starter.config import config_path, load_config  # noqa: E402
from ua_detrac_starter.tlc_compat import apply_ultralytics_83_compat  # noqa: E402


LOCKED_ARCH = "yolov8n"
YOLOV8N_YAML = "yolov8n.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8n with 3LC tables.")
    parser.add_argument(
        "--config",
        default="config/competition.yaml",
        help="Path to the workflow config file.",
    )
    return parser.parse_args()


def normalize_model_stem(name: str) -> str:
    stem = str(name).lower().strip()
    for suffix in (".pt", ".yaml", ".yml"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def assert_yolov8n_only(training: dict) -> None:
    raw_model = training.get("model")
    if raw_model is None:
        return
    stem = normalize_model_stem(str(raw_model))
    if stem != LOCKED_ARCH:
        raise SystemExit(
            f"Training is locked to YOLOv8n. training.model={raw_model!r} "
            f"resolves to {stem!r}; set it to yolov8n."
        )


def reject_pretrained_config(training: dict) -> None:
    if training.get("pretrained"):
        raise SystemExit(
            "Pretrained weights are not allowed. Remove `training.pretrained`; "
            f"training always starts from {YOLOV8N_YAML}."
        )


def apply_seed(seed: int | None) -> None:
    if seed is None:
        return
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def check_umap(embedding_dim: int, reducer: str) -> None:
    if embedding_dim <= 0 or reducer != "umap":
        return
    try:
        import umap  # noqa: F401
    except ImportError:
        raise SystemExit(
            "\n  umap-learn is required for image embeddings.\n"
            "  Fix: pip install umap-learn\n"
            "  Or: set tlc.image_embeddings_dim: 0 in config/competition.yaml.\n"
        )


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    apply_ultralytics_83_compat()

    cfg, _ = load_config(args.config)
    tlc_cfg = cfg.get("tlc", {})
    training = cfg.get("training", {})
    repro = cfg.get("reproducibility", {})

    assert_yolov8n_only(training)
    reject_pretrained_config(training)
    apply_seed(repro.get("seed"))

    project_name = str(tlc_cfg.get("project_name", "ua_detrac_vehicle_detection"))
    dataset_name = str(tlc_cfg.get("dataset_name", "ua_detrac_10k"))
    train_name = str(tlc_cfg.get("train_table_name", f"{dataset_name}-train"))
    val_name = str(tlc_cfg.get("val_table_name", f"{dataset_name}-val"))
    embedding_dim = int(tlc_cfg.get("image_embeddings_dim", 3))
    embedding_reducer = str(tlc_cfg.get("image_embeddings_reducer", "umap"))
    check_umap(embedding_dim, embedding_reducer)

    run_name = str(training.get("run_name", "yolov8n_baseline"))
    run_description = str(training.get("run_description", ""))
    epochs = int(training.get("epochs", 10))
    batch_size = int(training.get("batch_size", 16))
    image_size = int(training.get("image_size", 640))
    device = training.get("device", 0)
    workers = int(training.get("workers", 2))
    lr0 = float(training.get("lr0", 0.01))
    patience = int(training.get("patience", 20))
    use_augmentation = bool(training.get("use_augmentation", True))
    exist_ok = bool(training.get("exist_ok", True))
    runs_root = config_path(cfg, "runs_detect_root", "runs/detect")

    print("=" * 70)
    print("TRAINING (YOLOv8n + 3LC)")
    print("=" * 70)
    print(f"\n  PyTorch: {torch.__version__}, 3LC: {tlc.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")

    print("\n  Loading latest table revisions...")
    train_table = tlc.Table.from_names(
        project_name=project_name,
        dataset_name=dataset_name,
        table_name=train_name,
    ).latest()
    val_table = tlc.Table.from_names(
        project_name=project_name,
        dataset_name=dataset_name,
        table_name=val_name,
    ).latest()
    print(f"  Train: {len(train_table)} | Val: {len(val_table)}")

    tables_used = config_path(cfg, "tables_used", "runs/tables_used.txt")
    tables_used.parent.mkdir(parents=True, exist_ok=True)
    tables_used.write_text(
        f"train_url={train_table.url}\nval_url={val_table.url}\n",
        encoding="utf-8",
    )
    print(f"  Wrote {tables_used}")

    print(
        f"\n  Model: YOLOv8n from scratch ({YOLOV8N_YAML}) | run: {run_name} | "
        f"epochs: {epochs} | batch: {batch_size} | imgsz: {image_size}"
    )

    settings = Settings(
        project_name=project_name,
        run_name=run_name,
        run_description=run_description,
        image_embeddings_dim=embedding_dim,
        image_embeddings_reducer=embedding_reducer,
    )

    model = YOLO(YOLOV8N_YAML)
    train_args: dict = {
        "tables": {"train": train_table, "val": val_table},
        "project": str(runs_root),
        "name": run_name,
        "epochs": epochs,
        "imgsz": image_size,
        "batch": batch_size,
        "device": device,
        "workers": workers,
        "lr0": lr0,
        "patience": patience,
        "settings": settings,
        "val": True,
        "exist_ok": exist_ok,
    }
    if use_augmentation:
        train_args.update({"mosaic": 1.0, "mixup": 0.05, "copy_paste": 0.1})

    model.train(**train_args)

    print("\n" + "=" * 70)
    print("OK — TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Weights: {runs_root / run_name / 'weights' / 'best.pt'}")
    print("  Next: python scripts/predict.py --config config/competition.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
