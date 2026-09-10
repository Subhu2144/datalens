import io

import pandas as pd
import pytest

from app.data_processor import (
    DatasetValidationError,
    get_dataset_summary,
    load_dataset,
    profile_dataset,
    run_quality_checks,
    validate_dataframe,
    validate_file,
)


# ============================================================
# File validation tests
# ============================================================

def test_validate_file_accepts_csv():
    validate_file("sales.csv", 1024)


def test_validate_file_accepts_excel():
    validate_file("sales.xlsx", 1024)


def test_validate_file_rejects_unsupported_extension():
    with pytest.raises(DatasetValidationError):
        validate_file("sales.txt", 1024)


def test_validate_file_rejects_empty_file():
    with pytest.raises(DatasetValidationError):
        validate_file("sales.csv", 0)


# ============================================================
# DataFrame validation tests
# ============================================================

def test_validate_dataframe_accepts_valid_dataframe():
    df = pd.DataFrame(
        {
            "Sales": [100, 200, 300],
            "Region": ["Pune", "Mumbai", "Delhi"],
        }
    )

    validate_dataframe(df)


def test_validate_dataframe_rejects_empty_dataframe():
    df = pd.DataFrame()

    with pytest.raises(DatasetValidationError):
        validate_dataframe(df)


# ============================================================
# Dataset loading tests
# ============================================================

def test_load_dataset_csv():
    csv_content = b"Name,Sales\nA,100\nB,200\n"

    df = load_dataset("sales.csv", csv_content)

    assert len(df) == 2
    assert list(df.columns) == ["Name", "Sales"]


def test_load_dataset_excel():
    buffer = io.BytesIO()

    source_df = pd.DataFrame(
        {
            "Name": ["A", "B"],
            "Sales": [100, 200],
        }
    )

    source_df.to_excel(
        buffer,
        index=False,
        engine="openpyxl",
    )

    df = load_dataset("sales.xlsx", buffer.getvalue())

    assert len(df) == 2
    assert list(df.columns) == ["Name", "Sales"]


# ============================================================
# Dataset summary tests
# ============================================================

def test_get_dataset_summary():
    df = pd.DataFrame(
        {
            "Sales": [100, 200, 300],
            "Region": ["Pune", "Mumbai", "Pune"],
        }
    )

    summary = get_dataset_summary(df)

    assert summary["rows"] == 3
    assert summary["columns"] == 2


# ============================================================
# Profiling tests
# ============================================================

def test_profile_dataset_detects_missing_values():
    df = pd.DataFrame(
        {
            "Sales": [100, 200, None, 400],
            "Region": ["Pune", "Mumbai", "Pune", "Mumbai"],
        }
    )

    profile = profile_dataset(df)

    assert profile["summary"]["missing_cells"] == 1

    sales_details = next(
        item
        for item in profile["column_details"]
        if item["column"] == "Sales"
    )

    assert sales_details["missing"] == 1
    assert sales_details["missing_percentage"] == 25.0


def test_profile_dataset_detects_numeric_statistics():
    df = pd.DataFrame(
        {
            "Sales": [100, 200, 300, 400],
        }
    )

    profile = profile_dataset(df)

    sales_stats = profile["numeric_statistics"]["Sales"]

    assert sales_stats["count"] == 4
    assert sales_stats["mean"] == 250.0
    assert sales_stats["median"] == 250.0
    assert sales_stats["min"] == 100
    assert sales_stats["max"] == 400


# ============================================================
# Quality engine tests
# ============================================================

def test_quality_checks_detect_missing_values():
    df = pd.DataFrame(
        {
            "Sales": [100, 200, None, 400],
            "Region": ["Pune", "Mumbai", "Pune", "Mumbai"],
        }
    )

    result = run_quality_checks(df)

    assert result["summary"]["missing_issues"] == 1
    assert result["summary"]["total_missing_cells"] == 1

    issue = next(
        issue
        for issue in result["issues"]
        if issue["type"] == "missing_values"
    )

    assert issue["column"] == "Sales"
    assert issue["count"] == 1
    assert issue["severity"] == "high"


def test_quality_checks_detect_duplicate_rows():
    df = pd.DataFrame(
        {
            "Name": ["A", "B", "A"],
            "Sales": [100, 200, 100],
        }
    )

    result = run_quality_checks(df)

    assert result["summary"]["duplicate_rows"] == 1

    issue = next(
        issue
        for issue in result["issues"]
        if issue["type"] == "duplicate_rows"
    )

    assert issue["count"] == 1


def test_quality_checks_detect_constant_column():
    df = pd.DataFrame(
        {
            "Sales": [100, 200, 300, 400],
            "Status": ["Active", "Active", "Active", "Active"],
        }
    )

    result = run_quality_checks(df)

    assert result["summary"]["constant_columns"] == 1

    issue = next(
        issue
        for issue in result["issues"]
        if issue["type"] == "constant_column"
    )

    assert issue["column"] == "Status"


def test_quality_checks_detect_potential_outliers():
    df = pd.DataFrame(
        {
            "Sales": [
                100,
                110,
                120,
                130,
                140,
                150,
                10000,
            ]
        }
    )

    result = run_quality_checks(df)

    assert result["summary"]["outlier_columns"] == 1

    issue = next(
        issue
        for issue in result["issues"]
        if issue["type"] == "potential_outliers"
    )

    assert issue["column"] == "Sales"
    assert issue["count"] == 1


def test_quality_checks_clean_dataset_has_no_issues():
    df = pd.DataFrame(
        {
            "Sales": [100, 200, 300, 400, 500],
            "Region": ["Pune", "Mumbai", "Delhi", "Pune", "Mumbai"],
        }
    )

    result = run_quality_checks(df)

    assert result["score"] == 100
    assert result["issues"] == []
    assert result["summary"]["total_issues"] == 0


def test_quality_score_is_bounded():
    df = pd.DataFrame(
        {
            "A": [None, None, None, None, None],
            "B": ["X", "X", "X", "X", "X"],
            "C": [1, 1, 1, 1, 1],
        }
    )

    result = run_quality_checks(df)

    assert 0 <= result["score"] <= 100


def test_quality_checks_handles_small_numeric_columns():
    df = pd.DataFrame(
        {
            "Sales": [100, 200, 300],
        }
    )

    result = run_quality_checks(df)

    # Outlier detection should not run on fewer than 4 observations.
    assert result["summary"]["outlier_columns"] == 0