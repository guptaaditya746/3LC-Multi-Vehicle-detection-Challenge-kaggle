#!/usr/bin/env python3
"""Validate a Kaggle submission file for the 3LC Multi-Vehicle Detection Challenge."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import pandas as pd


EXPECTED_COLUMNS = ["id", "image_id", "prediction_string"]
VALID_CLASS_IDS = {0, 1, 2, 3}


@dataclass
class ValidationResult:
    errors: list[str]
    checked_rows: int = 0
    no_box_rows: int = 0
    detection_rows: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a submission.csv file against sample_submission.csv."
    )
    parser.add_argument("--sample", required=True, help="Path to sample_submission.csv")
    parser.add_argument("--submission", required=True, help="Path to submission.csv")
    return parser.parse_args()


def read_csv(path: str, label: str) -> tuple[pd.DataFrame | None, list[str]]:
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    except FileNotFoundError:
        return None, [f"{label} file not found: {path}"]
    except Exception as exc:  # pragma: no cover - defensive
        return None, [f"Failed to read {label} file '{path}': {exc}"]
    return frame, []


def validate_columns(frame: pd.DataFrame, label: str) -> list[str]:
    columns = list(frame.columns)
    if columns != EXPECTED_COLUMNS:
        return [
            f"{label} columns must be exactly {EXPECTED_COLUMNS}, found {columns}."
        ]
    return []


def csv_row_number(index: int) -> int:
    return index + 2


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def validate_prediction_string(row_index: int, value: str) -> list[str]:
    errors: list[str] = []
    row_number = csv_row_number(row_index)

    if "\n" in value or "\r" in value:
        errors.append(
            f"Row {row_number}: prediction_string must not contain line breaks."
        )

    if value == "no box":
        return errors

    parts = value.split()
    if not parts:
        errors.append(
            f"Row {row_number}: prediction_string is empty; use 'no box' for empty detections."
        )
        return errors

    if len(parts) % 6 != 0:
        errors.append(
            f"Row {row_number}: prediction_string must contain groups of 6 values, found {len(parts)} token(s)."
        )
        return errors

    for start in range(0, len(parts), 6):
        chunk = parts[start : start + 6]
        detection_number = (start // 6) + 1

        class_id_raw = chunk[0]
        if not class_id_raw.lstrip("-").isdigit():
            errors.append(
                f"Row {row_number}, detection {detection_number}: class_id must be an integer from 0 to 3, found '{class_id_raw}'."
            )
            continue

        class_id = int(class_id_raw)
        if class_id not in VALID_CLASS_IDS:
            errors.append(
                f"Row {row_number}, detection {detection_number}: class_id must be in [0, 3], found {class_id}."
            )

        labels = [
            ("confidence", chunk[1]),
            ("x_center", chunk[2]),
            ("y_center", chunk[3]),
            ("width", chunk[4]),
            ("height", chunk[5]),
        ]
        for field_name, raw_value in labels:
            numeric_value = parse_float(raw_value)
            if numeric_value is None:
                errors.append(
                    f"Row {row_number}, detection {detection_number}: {field_name} must be a float in [0, 1], found '{raw_value}'."
                )
                continue
            if not 0.0 <= numeric_value <= 1.0:
                errors.append(
                    f"Row {row_number}, detection {detection_number}: {field_name} must be in [0, 1], found {numeric_value}."
                )

    return errors


def validate_submission(sample: pd.DataFrame, submission: pd.DataFrame) -> ValidationResult:
    errors: list[str] = []

    errors.extend(validate_columns(sample, "Sample"))
    errors.extend(validate_columns(submission, "Submission"))
    if errors:
        return ValidationResult(errors=errors)

    if len(submission) != len(sample):
        errors.append(
            f"Submission row count ({len(submission)}) does not match sample row count ({len(sample)})."
        )

    common_length = min(len(sample), len(submission))
    for index in range(common_length):
        sample_id = sample.at[index, "id"]
        submission_id = submission.at[index, "id"]
        if submission_id != sample_id:
            errors.append(
                f"Row {csv_row_number(index)}: id mismatch. Expected '{sample_id}', found '{submission_id}'."
            )

        sample_image_id = sample.at[index, "image_id"]
        submission_image_id = submission.at[index, "image_id"]
        if submission_image_id != sample_image_id:
            errors.append(
                f"Row {csv_row_number(index)}: image_id mismatch. Expected '{sample_image_id}', found '{submission_image_id}'."
            )

    no_box_rows = 0
    detection_rows = 0
    for index, prediction_string in enumerate(submission["prediction_string"]):
        row_errors = validate_prediction_string(index, prediction_string)
        errors.extend(row_errors)
        if prediction_string == "no box":
            no_box_rows += 1
        else:
            detection_rows += 1

    return ValidationResult(
        errors=errors,
        checked_rows=len(submission),
        no_box_rows=no_box_rows,
        detection_rows=detection_rows,
    )


def main() -> int:
    args = parse_args()

    sample, sample_errors = read_csv(args.sample, "Sample")
    submission, submission_errors = read_csv(args.submission, "Submission")
    all_errors = sample_errors + submission_errors

    if all_errors:
        print("Submission validation failed.")
        for error in all_errors:
            print(f"- {error}")
        return 1

    assert sample is not None
    assert submission is not None

    result = validate_submission(sample, submission)
    if result.errors:
        print("Submission validation failed.")
        print(f"Checked rows: {result.checked_rows}")
        for error in result.errors:
            print(f"- {error}")
        return 1

    print("Submission validation passed.")
    print(f"Checked rows: {result.checked_rows}")
    print(f"Rows with 'no box': {result.no_box_rows}")
    print(f"Rows with detections: {result.detection_rows}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
