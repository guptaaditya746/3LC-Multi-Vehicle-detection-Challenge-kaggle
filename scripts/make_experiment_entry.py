#!/usr/bin/env python3
"""Append a simple experiment entry to experiments/experiment_log.csv."""

from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path


FIELDNAMES = [
    "experiment_id",
    "date",
    "git_commit",
    "run_name",
    "data_revision",
    "train_table",
    "val_table",
    "imgsz",
    "epochs",
    "batch",
    "seed",
    "optimizer",
    "lr",
    "public_score",
    "local_map50",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append a row to experiments/experiment_log.csv."
    )
    parser.add_argument("--experiment-id", required=True, help="Unique experiment identifier.")
    parser.add_argument("--run-name", default="", help="Human-readable run name.")
    parser.add_argument("--data-revision", default="", help="3LC data revision identifier.")
    parser.add_argument("--epochs", default="", help="Number of training epochs.")
    parser.add_argument("--batch", default="", help="Batch size.")
    parser.add_argument("--seed", default="", help="Random seed.")
    parser.add_argument("--notes", default="", help="Short notes for this run.")
    return parser.parse_args()


def ensure_header(csv_path: Path) -> None:
    if csv_path.exists():
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()


def build_row(args: argparse.Namespace) -> dict[str, str]:
    row = {field: "" for field in FIELDNAMES}
    row.update(
        {
            "experiment_id": args.experiment_id,
            "date": date.today().isoformat(),
            "run_name": args.run_name,
            "data_revision": args.data_revision,
            "epochs": str(args.epochs),
            "batch": str(args.batch),
            "seed": str(args.seed),
            "notes": args.notes,
        }
    )
    return row


def append_row(csv_path: Path, row: dict[str, str]) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writerow(row)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "experiments" / "experiment_log.csv"
    args = parse_args()

    ensure_header(csv_path)
    row = build_row(args)
    append_row(csv_path, row)

    print(f"Appended experiment entry to {csv_path}")
    print(f"experiment_id={row['experiment_id']}")
    print(f"date={row['date']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
