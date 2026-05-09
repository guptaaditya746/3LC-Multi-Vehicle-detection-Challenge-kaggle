#!/usr/bin/env python3
"""Register train and validation YOLO splits as 3LC tables."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ua_detrac_starter.config import (  # noqa: E402
    config_path,
    dataset_split_path,
    load_config,
    load_yaml,
)


warnings.filterwarnings(
    "ignore",
    message=".*from_yolo.*deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings("ignore", message=".*from_yolo.*deprecated.*", category=UserWarning)
logging.getLogger("3lc").setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or reuse 3LC train/val tables.")
    parser.add_argument(
        "--config",
        default="config/competition.yaml",
        help="Path to the workflow config file.",
    )
    return parser.parse_args()


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise SystemExit(f"ERROR: Expected directory missing: {path}")


def latest_table(tlc, project_name: str, dataset_name: str, table_name: str):
    return tlc.Table.from_names(
        project_name=project_name,
        dataset_name=dataset_name,
        table_name=table_name,
    ).latest()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)

    import tlc

    cfg, _ = load_config(args.config)
    tlc_cfg = cfg.get("tlc", {})

    dataset_yaml = config_path(cfg, "dataset_yaml", "config/dataset.yaml")
    if not dataset_yaml.is_file():
        print(f"ERROR: dataset yaml not found: {dataset_yaml}", file=sys.stderr)
        return 1

    dataset_cfg = load_yaml(dataset_yaml)
    train_images = dataset_split_path(dataset_yaml, dataset_cfg, "train")
    val_images = dataset_split_path(dataset_yaml, dataset_cfg, "val")
    for path in (
        train_images,
        train_images.parent / "labels",
        val_images,
        val_images.parent / "labels",
    ):
        require_dir(path)

    project_name = str(tlc_cfg.get("project_name", "ua_detrac_vehicle_detection"))
    dataset_name = str(tlc_cfg.get("dataset_name", "ua_detrac_10k"))
    train_name = str(tlc_cfg.get("train_table_name", f"{dataset_name}-train"))
    val_name = str(tlc_cfg.get("val_table_name", f"{dataset_name}-val"))

    print("=" * 70)
    print("REGISTER 3LC TABLES")
    print("=" * 70)
    print(f"  dataset_yaml: {dataset_yaml}")
    print(f"  project: {project_name}")
    print(f"  dataset: {dataset_name}")

    created_any = False
    try:
        train_table = latest_table(tlc, project_name, dataset_name, train_name)
        print("\n  Train table exists — using latest revision.")
    except Exception:
        print("\n  Creating train table from YOLO...")
        train_table = tlc.Table.from_yolo(
            dataset_yaml_file=str(dataset_yaml),
            split="train",
            task="detect",
            dataset_name=dataset_name,
            project_name=project_name,
            table_name=train_name,
        )
        created_any = True
        print(f"  Train samples: {len(train_table)}")

    try:
        val_table = latest_table(tlc, project_name, dataset_name, val_name)
        print("\n  Val table exists — using latest revision.")
    except Exception:
        print("\n  Creating val table from YOLO...")
        val_table = tlc.Table.from_yolo(
            dataset_yaml_file=str(dataset_yaml),
            split="val",
            task="detect",
            dataset_name=dataset_name,
            project_name=project_name,
            table_name=val_name,
        )
        created_any = True
        print(f"  Val samples: {len(val_table)}")

    print(f"\n  Train: {train_table.url}")
    print(f"  Val:   {val_table.url}")
    print("\n  Test images stay on disk and are used by scripts/predict.py.")

    print("\n" + "=" * 70)
    print("OK — TABLES READY" if created_any else "OK — EXISTING TABLES READY")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
