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