#!/usr/bin/env python3
"""Summarize YOLO label files for quick dataset sanity checks."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a directory of YOLO label files.")
    parser.add_argument("--labels", required=True, help="Path to a directory containing YOLO .txt labels.")
    parser.add_argument(
        "--classes",
        default="truck,car,van,bus",
        help="Comma-separated class names in class-id order.",
    )
    return parser.parse_args()


def load_class_names(raw_names: str) -> list[str]:
    names = [name.strip() for name in raw_names.split(",")]
    return [name for name in names if name]


def summarize(labels_dir: Path, class_names: list[str]) -> int:
    if not labels_dir.exists():
        print(f"Labels directory not found: {labels_dir}")
        return 1

    if not labels_dir.is_dir():
        print(f"Labels path is not a directory: {labels_dir}")
        return 1

    label_files = sorted(labels_dir.rglob("*.txt"))
    class_counts: Counter[int] = Counter()
    invalid_rows = 0
    invalid_normalized_coordinates = 0
    empty_label_files = 0
    total_boxes = 0
    invalid_examples: list[str] = []

    for label_file in label_files:
        raw_lines = label_file.read_text(encoding="utf-8").splitlines()
        lines = [line.strip() for line in raw_lines if line.strip()]
        if not lines:
            empty_label_files += 1
            continue

        for line_number, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) != 5:
                invalid_rows += 1
                invalid_examples.append(
                    f"{label_file}:{line_number} expected 5 columns, found {len(parts)}"
                )
                continue

            class_raw, x_raw, y_raw, w_raw, h_raw = parts
            if not class_raw.lstrip("-").isdigit():
                invalid_rows += 1
                invalid_examples.append(
                    f"{label_file}:{line_number} invalid class id '{class_raw}'"
                )
                continue

            class_id = int(class_raw)
            try:
                coords = [float(x_raw), float(y_raw), float(w_raw), float(h_raw)]
            except ValueError:
                invalid_rows += 1
                invalid_examples.append(
                    f"{label_file}:{line_number} contains non-numeric coordinates"
                )
                continue

            if class_names and not 0 <= class_id < len(class_names):
                invalid_rows += 1
                invalid_examples.append(
                    f"{label_file}:{line_number} has unmapped class id {class_id}"
                )
                continue

            if not all(0.0 <= value <= 1.0 for value in coords):
                invalid_rows += 1
                invalid_normalized_coordinates += 1
                invalid_examples.append(
                    f"{label_file}:{line_number} has coordinates outside [0, 1]"
                )
                continue

            class_counts[class_id] += 1
            total_boxes += 1

    print("YOLO dataset summary")
    print("--------------------")
    print(f"Labels directory: {labels_dir}")
    print(f"Label files: {len(label_files)}")
    print(f"Empty label files: {empty_label_files}")
    print(f"Total valid boxes: {total_boxes}")
    print(f"Invalid rows: {invalid_rows}")
    print(f"Invalid normalized coordinates: {invalid_normalized_coordinates}")
    print("Boxes per class:")

    for class_id, class_name in enumerate(class_names):
        print(f"  {class_id} ({class_name}): {class_counts[class_id]}")

    if invalid_examples:
        print("Invalid row examples:")
        for example in invalid_examples[:10]:
            print(f"  - {example}")
        if len(invalid_examples) > 10:
            print(f"  - ... {len(invalid_examples) - 10} more")

    if not label_files:
        print("Warning: no .txt label files were found.")

    return 0


def main() -> int:
    args = parse_args()
    class_names = load_class_names(args.classes)
    return summarize(Path(args.labels), class_names)


if __name__ == "__main__":
    raise SystemExit(main())
