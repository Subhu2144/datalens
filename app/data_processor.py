from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}


class DatasetValidationError(Exception):
    """Raised when an uploaded dataset is invalid or unsupported."""


def validate_file(file_name: str, file_size: int) -> None:
    """
    Validate the uploaded file before attempting to read it.

    Args:
        file_name: Original uploaded file name.
        file_size: File size in bytes.

    Raises:
        DatasetValidationError: If the file is unsupported or invalid.
    """
    extension = Path(file_name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DatasetValidationError(
            f"Unsupported file type '{extension or 'unknown'}'. "
            f"Please upload one of: {supported}."
        )

    if file_size == 0:
        raise DatasetValidationError(
            "The uploaded file is empty. Please upload a dataset containing data."
        )


def _read_csv(file_bytes: bytes) -> pd.DataFrame:
    """Read CSV data into a pandas DataFrame."""
    try:
        return pd.read_csv(BytesIO(file_bytes), low_memory=False)

    except UnicodeDecodeError:
        try:
            return pd.read_csv(
                BytesIO(file_bytes),
                encoding="latin-1",
                low_memory=False,
            )
        except Exception as exc:
            raise DatasetValidationError(
                "The CSV file could not be decoded or read. "
                "Please check that it is a valid CSV file."
            ) from exc

    except pd.errors.EmptyDataError as exc:
        raise DatasetValidationError(
            "The CSV file does not contain any data."
        ) from exc

    except pd.errors.ParserError as exc:
        raise DatasetValidationError(
            "The CSV file could not be parsed. "
            "Please check its formatting."
        ) from exc


def _read_excel(file_bytes: bytes) -> pd.DataFrame:
    """Read the first worksheet of an Excel file into a pandas DataFrame."""
    try:
        return pd.read_excel(
            BytesIO(file_bytes),
            engine="openpyxl",
        )

    except ValueError as exc:
        raise DatasetValidationError(
            "The Excel file does not contain a readable worksheet."
        ) from exc

    except Exception as exc:
        raise DatasetValidationError(
            "The Excel file could not be read. "
            "Please make sure it is a valid .xlsx file."
        ) from exc


def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate a loaded DataFrame.

    Args:
        df: DataFrame to validate.

    Raises:
        DatasetValidationError: If the DataFrame is invalid.
    """
    if df.empty:
        raise DatasetValidationError(
            "The dataset contains no rows. Please upload a dataset with data."
        )

    if df.shape[1] == 0:
        raise DatasetValidationError(
            "The dataset contains no columns."
        )

    if all(str(column).strip() == "" for column in df.columns):
        raise DatasetValidationError(
            "The dataset does not contain valid column names."
        )


def load_dataset(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    """
    Load a CSV or Excel dataset into a pandas DataFrame.

    Args:
        file_name: Original uploaded file name.
        file_bytes: Raw bytes of the uploaded file.

    Returns:
        A validated pandas DataFrame.

    Raises:
        DatasetValidationError: If the file or dataset is invalid.
    """
    validate_file(file_name, len(file_bytes))

    extension = Path(file_name).suffix.lower()

    if extension == ".csv":
        dataframe = _read_csv(file_bytes)
    else:
        dataframe = _read_excel(file_bytes)

    validate_dataframe(dataframe)

    return dataframe


def get_dataset_summary(df: pd.DataFrame) -> dict[str, int]:
    """
    Return basic dataset dimensions.

    Args:
        df: Dataset DataFrame.

    Returns:
        Dictionary containing row and column counts.
    """
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
    }


def _clean_statistic_value(value: Any) -> Any:
    """
    Convert pandas/NumPy values into safe Python values.

    This prevents NaN and NumPy scalar values from leaking into
    the profiling result.
    """
    if pd.isna(value):
        return None

    if isinstance(value, np.generic):
        return value.item()

    return value


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """
    Generate a complete profile of the dataset.

    The profile contains:
        - dataset summary
        - column-level information
        - missing-value information
        - duplicate information
        - numeric statistics

    Args:
        df: Dataset DataFrame.

    Returns:
        A structured dictionary containing profiling information.

    Raises:
        DatasetValidationError: If the DataFrame is invalid.
    """
    validate_dataframe(df)

    total_cells = df.shape[0] * df.shape[1]

    missing_total = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    memory_usage_bytes = int(
        df.memory_usage(deep=True).sum()
    )

    column_details: list[dict[str, Any]] = []

    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        non_null_count = int(df[column].notna().sum())
        unique_count = int(df[column].nunique(dropna=True))

        missing_percentage = (
            (missing_count / len(df)) * 100
            if len(df) > 0
            else 0.0
        )

        column_details.append(
            {
                "column": str(column),
                "data_type": str(df[column].dtype),
                "non_null": non_null_count,
                "missing": missing_count,
                "missing_percentage": round(missing_percentage, 2),
                "unique_values": unique_count,
            }
        )

    missing_by_column = [
        {
            "column": str(column),
            "missing": int(df[column].isna().sum()),
            "missing_percentage": round(
                (df[column].isna().sum() / len(df)) * 100,
                2,
            ),
        }
        for column in df.columns
        if df[column].isna().sum() > 0
    ]

    numeric_statistics: dict[str, dict[str, Any]] = {}

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:
        series = df[column]

        numeric_statistics[str(column)] = {
            "count": int(series.count()),
            "mean": _clean_statistic_value(series.mean()),
            "median": _clean_statistic_value(series.median()),
            "std": _clean_statistic_value(series.std()),
            "min": _clean_statistic_value(series.min()),
            "max": _clean_statistic_value(series.max()),
        }

    duplicate_percentage = (
        (duplicate_rows / len(df)) * 100
        if len(df) > 0
        else 0.0
    )

    missing_percentage = (
        (missing_total / total_cells) * 100
        if total_cells > 0
        else 0.0
    )

    return {
        "summary": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "total_cells": int(total_cells),
            "memory_usage_bytes": memory_usage_bytes,
            "missing_cells": missing_total,
            "missing_percentage": round(missing_percentage, 2),
            "duplicate_rows": duplicate_rows,
            "duplicate_percentage": round(
                duplicate_percentage,
                2,
            ),
        },
        "column_details": column_details,
        "missing_values": missing_by_column,
        "numeric_statistics": numeric_statistics,
    }

def run_quality_checks(df):
    """
    Run deterministic data-quality checks on a dataframe.

    Returns:
        dict: Quality score, detected issues, and summary.
    """

    issues = []

    # ---------------------------------------------------------
    # 1. Missing-value checks
    # ---------------------------------------------------------
    total_cells = df.shape[0] * df.shape[1]

    if total_cells > 0:
        total_missing = int(df.isna().sum().sum())
        total_missing_percentage = round(
            (total_missing / total_cells) * 100, 2
        )
    else:
        total_missing = 0
        total_missing_percentage = 0.0

    for column in df.columns:
        missing_count = int(df[column].isna().sum())

        if missing_count == 0:
            continue

        missing_percentage = round(
            (missing_count / len(df)) * 100, 2
        ) if len(df) > 0 else 0.0

        if missing_percentage <= 5:
            severity = "low"
        elif missing_percentage <= 20:
            severity = "medium"
        else:
            severity = "high"

        issues.append(
            {
                "type": "missing_values",
                "column": str(column),
                "count": missing_count,
                "percentage": missing_percentage,
                "severity": severity,
                "message": (
                    f"{column} has {missing_count} missing value(s) "
                    f"({missing_percentage}%)."
                ),
            }
        )

    # ---------------------------------------------------------
    # 2. Duplicate-row check
    # ---------------------------------------------------------
    duplicate_rows = int(df.duplicated().sum())

    duplicate_percentage = round(
        (duplicate_rows / len(df)) * 100, 2
    ) if len(df) > 0 else 0.0

    if duplicate_rows > 0:

        if duplicate_percentage <= 5:
            severity = "low"
        elif duplicate_percentage <= 20:
            severity = "medium"
        else:
            severity = "high"

        issues.append(
            {
                "type": "duplicate_rows",
                "column": None,
                "count": duplicate_rows,
                "percentage": duplicate_percentage,
                "severity": severity,
                "message": (
                    f"Dataset contains {duplicate_rows} duplicate row(s) "
                    f"({duplicate_percentage}%)."
                ),
            }
        )

    # ---------------------------------------------------------
    # 3. Constant-column check
    # ---------------------------------------------------------
    constant_columns = []

    for column in df.columns:
        unique_count = df[column].nunique(dropna=False)

        if unique_count <= 1:
            constant_columns.append(str(column))

            issues.append(
                {
                    "type": "constant_column",
                    "column": str(column),
                    "count": 1,
                    "percentage": 100.0,
                    "severity": "medium",
                    "message": (
                        f"{column} contains only one unique value "
                        "and may not provide useful analytical information."
                    ),
                }
            )

    # ---------------------------------------------------------
    # 4. Potential numeric outliers using IQR
    # ---------------------------------------------------------
    outlier_columns = []

    numeric_columns = df.select_dtypes(include=np.number).columns

    for column in numeric_columns:
        values = df[column].dropna()

        # Need enough observations for a meaningful check
        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        # No meaningful spread
        if iqr <= 0:
            continue

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        outlier_mask = (
            (values < lower_bound)
            | (values > upper_bound)
        )

        outlier_count = int(outlier_mask.sum())

        if outlier_count == 0:
            continue

        outlier_percentage = round(
            (outlier_count / len(values)) * 100, 2
        )

        outlier_columns.append(str(column))

        if outlier_percentage <= 5:
            severity = "low"
        elif outlier_percentage <= 20:
            severity = "medium"
        else:
            severity = "high"

        issues.append(
            {
                "type": "potential_outliers",
                "column": str(column),
                "count": outlier_count,
                "percentage": outlier_percentage,
                "severity": severity,
                "message": (
                    f"{column} contains {outlier_count} potential "
                    f"outlier(s) ({outlier_percentage}%) based on the IQR method."
                ),
            }
        )

    # ---------------------------------------------------------
    # 5. Quality score
    # ---------------------------------------------------------
    # This is an app-defined heuristic score, not a statistical
    # or industry-standard data-quality score.

    missing_penalty = min(
        30,
        round(total_missing_percentage * 0.75)
    )

    duplicate_penalty = min(
        20,
        round(duplicate_percentage * 0.5)
    )

    constant_penalty = min(
        15,
        len(constant_columns) * 3
    )

    outlier_penalty = min(
        20,
        len(outlier_columns) * 3
    )

    total_penalty = (
        missing_penalty
        + duplicate_penalty
        + constant_penalty
        + outlier_penalty
    )

    quality_score = max(
        0,
        min(100, 100 - total_penalty)
    )

    # ---------------------------------------------------------
    # 6. Summary
    # ---------------------------------------------------------
    summary = {
        "missing_issues": sum(
            1 for issue in issues
            if issue["type"] == "missing_values"
        ),
        "duplicate_rows": duplicate_rows,
        "constant_columns": len(constant_columns),
        "outlier_columns": len(outlier_columns),
        "total_issues": len(issues),
        "total_missing_cells": total_missing,
        "total_missing_percentage": total_missing_percentage,
    }

    return {
        "score": quality_score,
        "issues": issues,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------

def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataframe with duplicate rows removed."""
    validate_dataframe(df)
    return df.drop_duplicates().reset_index(drop=True)


def fill_numeric_missing_with_median(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in numeric columns with each column's median."""
    validate_dataframe(df)

    cleaned_df = df.copy()

    numeric_columns = cleaned_df.select_dtypes(include=np.number).columns

    for column in numeric_columns:
        if cleaned_df[column].isna().any():
            median = cleaned_df[column].median()
            if not pd.isna(median):
                cleaned_df[column] = cleaned_df[column].fillna(median)

    return cleaned_df


def fill_categorical_missing_with_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing categorical values with each column's mode."""
    validate_dataframe(df)

    cleaned_df = df.copy()

    categorical_columns = cleaned_df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    for column in categorical_columns:
        if cleaned_df[column].isna().any():
            mode = cleaned_df[column].mode(dropna=True)

            if not mode.empty:
                cleaned_df[column] = cleaned_df[column].fillna(mode.iloc[0])

    return cleaned_df


def drop_rows_with_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with rows containing one or more missing values removed."""
    validate_dataframe(df)
    return df.dropna().reset_index(drop=True)


def apply_cleaning(
    df: pd.DataFrame,
    remove_duplicates: bool = False,
    fill_numeric_median: bool = False,
    fill_categorical_mode: bool = False,
    drop_missing_rows: bool = False,
) -> pd.DataFrame:
    """
    Apply selected deterministic cleaning operations to a dataframe.

    Operations are applied in a predictable order:
    1. Fill numeric missing values with median.
    2. Fill categorical missing values with mode.
    3. Remove duplicate rows.
    4. Drop rows that still contain missing values.

    The original dataframe is never modified.
    """
    validate_dataframe(df)

    cleaned_df = df.copy()

    if fill_numeric_median:
        cleaned_df = fill_numeric_missing_with_median(cleaned_df)

    if fill_categorical_mode:
        cleaned_df = fill_categorical_missing_with_mode(cleaned_df)

    if remove_duplicates:
        cleaned_df = remove_duplicate_rows(cleaned_df)

    if drop_missing_rows:
        cleaned_df = drop_rows_with_missing_values(cleaned_df)

    return cleaned_df
