#!/usr/bin/env python3
"""Run inference on test images and write a Kaggle submission CSV."""

from __future__ import annotations

import argparse
import csv
import gc
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from tlc_ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ua_detrac_starter.config import config_path, load_config, resolve_repo_path  # noqa: E402
from ua_detrac_starter.tlc_compat import apply_ultralytics_83_compat  # noqa: E402


try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate submission.csv from test images.")
    parser.add_argument(
        "--config",
        default="config/competition.yaml",
        help="Path to the workflow config file.",
    )
    return parser.parse_args()


def resolve_weights(cfg: dict) -> Path:
    training = cfg.get("training", {})
    predict_cfg = cfg.get("predict", {})
    runs_root = config_path(cfg, "runs_detect_root", "runs/detect")
    run_name = str(training.get("run_name", "yolov8n_baseline"))

    explicit = predict_cfg.get("weights")
    if explicit:
        explicit_path = resolve_repo_path(str(explicit))
        if explicit_path.is_file():
            return explicit_path
        raise FileNotFoundError(f"Configured predict.weights not found: {explicit_path}")

    primary = runs_root / run_name / "weights" / "best.pt"
    if primary.is_file():
        return primary

    candidates = sorted(
        runs_root.glob("*/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "No weights found. Run scripts/train.py first or set predict.weights."
    )


def find_image(test_dir: Path, image_id: str) -> Path | None:
    raw = Path(image_id)
    if raw.suffix:
        direct = test_dir / raw.name
        return direct if direct.is_file() else None

    for extension in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        candidate = test_dir / f"{image_id}{extension}"
        if candidate.is_file():
            return candidate
    return None


def label_file_to_prediction_string(label_path: Path) -> str:
    if not label_path.is_file() or label_path.stat().st_size == 0:
        return "no box"

    tokens: list[str] = []
    with label_path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 6:
                class_id, xc, yc, width, height, conf = parts[:6]
                tokens.append(f"{class_id} {conf} {xc} {yc} {width} {height}")
    return " ".join(tokens) if tokens else "no box"


def result_to_prediction_string(result) -> str:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return "no box"

    xywhn = boxes.xywhn.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    order = np.argsort(-conf)

    parts: list[str] = []
    for index in order:
        class_id = int(cls[index])
        confidence = float(conf[index])
        x_center, y_center, width, height = (float(value) for value in xywhn[index])
        x_center = min(1.0, max(0.0, x_center))
        y_center = min(1.0, max(0.0, y_center))
        width = min(1.0, max(0.0, width))
        height = min(1.0, max(0.0, height))
        parts.extend(
            [
                str(class_id),
                f"{confidence:.6f}",
                f"{x_center:.6f}",
                f"{y_center:.6f}",
                f"{width:.6f}",
                f"{height:.6f}",
            ]
        )
    return " ".join(parts)


def image_pairs(test_dir: Path, image_ids: list[str]) -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []
    for image_id in image_ids:
        path = find_image(test_dir, image_id)
        if path is not None:
            pairs.append((image_id, path))
    if not pairs:
        raise RuntimeError(f"No test images found under {test_dir}")
    return pairs


def pipeline_txt(
    model: YOLO,
    test_dir: Path,
    image_ids: list[str],
    pred_dir_name: str,
    *,
    imgsz: int,
    conf: float,
    iou: float,
    device,
    max_det: int,
    batch: int,
) -> dict[str, str]:
    pred_root = resolve_repo_path(pred_dir_name)
    if pred_root.exists():
        shutil.rmtree(pred_root)

    pairs = image_pairs(test_dir, image_ids)
    paths = [path for _, path in pairs]
    batch_size = max(1, int(batch))

    results = model.predict(
        source=[str(path) for path in paths],
        save=False,
        save_txt=True,
        save_conf=True,
        conf=float(conf),
        iou=float(iou),
        imgsz=int(imgsz),
        device=device,
        max_det=int(max_det),
        batch=batch_size,
        project=str(pred_root.parent),
        name=pred_root.name,
        exist_ok=False,
        verbose=False,
        stream=True,
    )
    for _ in tqdm(results, total=len(paths), desc="Predicting", unit="img"):
        pass

    labels_dir = pred_root / "labels"
    predictions = {
        image_id: label_file_to_prediction_string(labels_dir / f"{path.stem}.txt")
        for image_id, path in pairs
    }

    if pred_root.exists():
        shutil.rmtree(pred_root)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return predictions


def pipeline_memory(
    model: YOLO,
    test_dir: Path,
    image_ids: list[str],
    *,
    imgsz: int,
    conf: float,
    iou: float,
    device,
    max_det: int,
    batch: int,
) -> dict[str, str]:
    pairs = image_pairs(test_dir, image_ids)
    predictions: dict[str, str] = {}
    batch_size = max(1, int(batch))
    total_chunks = (len(pairs) + batch_size - 1) // batch_size

    for start in tqdm(
        range(0, len(pairs), batch_size),
        total=total_chunks,
        desc="Predicting",
        unit="batch",
    ):
        chunk = pairs[start : start + batch_size]
        results = model.predict(
            source=[str(path) for _, path in chunk],
            imgsz=int(imgsz),
            conf=float(conf),
            iou=float(iou),
            device=device,
            max_det=int(max_det),
            batch=min(batch_size, len(chunk)),
            verbose=False,
        )
        for result, (image_id, _) in zip(results, chunk):
            predictions[image_id] = result_to_prediction_string(result)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return predictions


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    apply_ultralytics_83_compat()
    cfg, _ = load_config(args.config)

    training = cfg.get("training", {})
    predict_cfg = cfg.get("predict", {})

    sample_path = config_path(
        cfg,
        "sample_submission",
        "data/competition_starter/sample_submission.csv",
    )
    out_path = config_path(cfg, "submission_csv", "submissions/submission.csv")
    test_dir = config_path(cfg, "test_images", "data/competition_starter/data/test/images")

    conf = float(predict_cfg.get("conf", 0.25))
    iou = float(predict_cfg.get("iou", 0.7))
    imgsz = int(predict_cfg.get("imgsz") or training.get("image_size", 640))
    device = (
        predict_cfg["device"]
        if predict_cfg.get("device") is not None
        else training.get("device", 0)
    )
    max_det = int(predict_cfg.get("max_det", 300))
    batch = int(predict_cfg.get("batch", 1))
    pipeline = str(predict_cfg.get("pipeline", "memory")).lower()
    pred_dir_name = str(predict_cfg.get("pred_dir_name", "predictions"))

    print("=" * 70)
    print("PREDICTIONS → SUBMISSION CSV")
    print("=" * 70)

    if not sample_path.is_file():
        print(f"ERROR: sample submission not found: {sample_path}", file=sys.stderr)
        return 1
    if not test_dir.is_dir():
        print(f"ERROR: test images directory not found: {test_dir}", file=sys.stderr)
        return 1
    if pipeline not in {"memory", "txt"}:
        print("ERROR: predict.pipeline must be `memory` or `txt`", file=sys.stderr)
        return 1

    try:
        weights = resolve_weights(cfg)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    rows_in: list[dict[str, str]] = []
    with sample_path.open(newline="", encoding="utf-8") as handle:
        rows_in.extend(csv.DictReader(handle))

    image_ids = [str(row["image_id"]).strip() for row in rows_in]
    print(f"\n  Rows: {len(rows_in)}")
    print(f"  Test dir: {test_dir}")
    print(f"  Weights: {weights}")
    print(f"  Pipeline: {pipeline}")
    print(f"  conf={conf}, iou={iou}, imgsz={imgsz}, batch={batch}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = YOLO(str(weights))
    print("  Model loaded.")

    if pipeline == "txt":
        predictions = pipeline_txt(
            model,
            test_dir,
            image_ids,
            pred_dir_name,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            max_det=max_det,
            batch=batch,
        )
    else:
        predictions = pipeline_memory(
            model,
            test_dir,
            image_ids,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            max_det=max_det,
            batch=batch,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "image_id", "prediction_string"],
        )
        writer.writeheader()
        for row, image_id in tqdm(
            zip(rows_in, image_ids),
            total=len(rows_in),
            desc="Writing CSV",
            unit="row",
        ):
            writer.writerow(
                {
                    "id": row["id"],
                    "image_id": image_id,
                    "prediction_string": predictions.get(image_id, "no box"),
                }
            )

    nonempty = sum(
        1 for image_id in image_ids if predictions.get(image_id, "no box") != "no box"
    )
    box_count = sum(
        len(predictions.get(image_id, "").split()) // 6
        for image_id in image_ids
        if predictions.get(image_id, "no box") != "no box"
    )

    print("\n" + "=" * 70)
    print("OK — SUBMISSION READY")
    print("=" * 70)
    print(f"  File: {out_path}")
    print(f"  Images with boxes: {nonempty} / {len(rows_in)}")
    print(f"  Total boxes approx: {box_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

