#!/usr/bin/env python3
"""Pre-flight environment check for the competition workflow."""

from __future__ import annotations

import argparse
import importlib
import sys
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


PASS = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"

fail_count = 0
warn_count = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check packages, GPU, and data paths.")
    parser.add_argument(
        "--config",
        default="config/competition.yaml",
        help="Path to the workflow config file.",
    )
    return parser.parse_args()


def check(label: str, ok: bool, detail: str, *, warn_only: bool = False) -> None:
    global fail_count, warn_count
    if ok:
        print(f"  {PASS} {label}: {detail}")
    elif warn_only:
        warn_count += 1
        print(f"  {WARN} {label}: {detail}")
    else:
        fail_count += 1
        print(f"  {FAIL} {label}: {detail}")


def try_import(module: str) -> tuple[bool, str]:
    try:
        imported = importlib.import_module(module)
        version = getattr(imported, "__version__", "installed")
        return True, str(version)
    except ImportError:
        return False, "not installed"


def check_python() -> None:
    version = sys.version_info
    version_text = f"{version.major}.{version.minor}.{version.micro}"
    ok = version.major == 3 and 9 <= version.minor < 14
    detail = version_text if ok else f"{version_text} — need 3.9 to 3.13"
    check("Python version", ok, detail)


def check_packages() -> None:
    required = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("3lc (tlc)", "tlc"),
        ("3lc-ultralytics", "tlc_ultralytics"),
        ("ultralytics", "ultralytics"),
        ("umap-learn", "umap"),
        ("PyYAML", "yaml"),
        ("tqdm", "tqdm"),
        ("pandas", "pandas"),
    ]
    for display, module in required:
        ok, detail = try_import(module)
        if not ok and module == "umap":
            detail += " — install umap-learn or set image_embeddings_dim: 0"
        check(display, ok, detail)


def check_gpu() -> None:
    try:
        import torch

        has_cuda = torch.cuda.is_available()
        if has_cuda:
            check("CUDA GPU", True, torch.cuda.get_device_name(0))
        else:
            check(
                "CUDA GPU",
                False,
                "not available — training will run on CPU",
                warn_only=True,
            )

        cuda_version = getattr(torch.version, "cuda", None)
        if cuda_version is None:
            check(
                "PyTorch CUDA build",
                False,
                f"torch {torch.__version__} is CPU-only",
                warn_only=True,
            )
        else:
            check("PyTorch CUDA build", True, f"CUDA {cuda_version}")
    except Exception as exc:
        check("CUDA GPU", False, str(exc), warn_only=True)


def check_file(path: Path, label: str) -> None:
    check(label, path.is_file(), "found" if path.is_file() else f"MISSING: {path}")


def check_dir(path: Path, label: str) -> None:
    if path.is_dir():
        count = sum(1 for _ in path.iterdir())
        check(label, True, f"{count} files")
    else:
        check(label, False, f"MISSING: {path}")


def check_config_and_data(raw_config_path: str) -> None:
    try:
        cfg, resolved_config = load_config(raw_config_path)
    except Exception as exc:
        check("Config", False, str(exc))
        return

    check_file(resolved_config, "Config")
    for script_name in (
        "register_tables.py",
        "train.py",
        "predict.py",
        "validate_submission.py",
    ):
        check_file(REPO_ROOT / "scripts" / script_name, f"Script: {script_name}")

    dataset_yaml = config_path(cfg, "dataset_yaml", "config/dataset.yaml")
    sample_submission = config_path(
        cfg,
        "sample_submission",
        "data/competition_starter/sample_submission.csv",
    )
    check_file(dataset_yaml, "Dataset YAML")
    check_file(sample_submission, "Sample submission")

    if not dataset_yaml.is_file():
        return

    try:
        dataset_cfg = load_yaml(dataset_yaml)
        train_images = dataset_split_path(dataset_yaml, dataset_cfg, "train")
        val_images = dataset_split_path(dataset_yaml, dataset_cfg, "val")
        test_images = dataset_split_path(dataset_yaml, dataset_cfg, "test")
    except Exception as exc:
        check("Dataset YAML parse", False, str(exc))
        return

    check_dir(train_images, "Dir: train images")
    check_dir(train_images.parent / "labels", "Dir: train labels")
    check_dir(val_images, "Dir: val images")
    check_dir(val_images.parent / "labels", "Dir: val labels")
    check_dir(test_images, "Dir: test images")


def main() -> int:
    args = parse_args()
    print("=" * 65)
    print("  ENVIRONMENT CHECK")
    print("=" * 65)

    print("\n-- Python --")
    check_python()
    print("\n-- Packages --")
    check_packages()
    print("\n-- GPU --")
    check_gpu()
    print("\n-- Files and data --")
    check_config_and_data(args.config)

    print("\n" + "=" * 65)
    if fail_count == 0 and warn_count == 0:
        print("  ALL CHECKS PASSED")
    elif fail_count == 0:
        print(f"  PASSED with {warn_count} warning(s)")
    else:
        print(f"  {fail_count} FAILED, {warn_count} warning(s)")
    print("=" * 65)
    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())

