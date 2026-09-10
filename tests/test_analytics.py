import pandas as pd
import pytest

from app.analytics import (
    AnalyticsError,
    calculate_average,
    calculate_correlation,
    calculate_maximum,
    calculate_minimum,
    calculate_summary,
    calculate_total,
    group_by_aggregation,
    top_n,
)


# ============================================================
# Test data
# ============================================================

@pytest.fixture
def sales_df():
    return pd.DataFrame(
        {
            "Product": ["A", "B", "C", "D", "E"],
            "Region": [
                "Pune",
                "Mumbai",
                "Pune",
                "Mumbai",
                "Delhi",
            ],
            "Sales": [100, 200, 300, 400, 500],
            "Quantity": [1, 2, 3, 4, 5],
        }
    )


# ============================================================
# Summary tests
# ============================================================

def test_calculate_summary(sales_df):
    result = calculate_summary(
        sales_df,
        "Sales",
    )

    assert result["operation"] == "summary"
    assert result["column"] == "Sales"
    assert result["count"] == 5
    assert result["sum"] == 1500
    assert result["mean"] == 300.0
    assert result["median"] == 300.0
    assert result["min"] == 100
    assert result["max"] == 500


def test_calculate_summary_ignores_missing_values():
    df = pd.DataFrame(
        {
            "Sales": [100, 200, None, 400],
        }
    )

    result = calculate_summary(
        df,
        "Sales",
    )

    assert result["count"] == 3
    assert result["sum"] == 700
    assert result["mean"] == pytest.approx(233.333333)


# ============================================================
# Total tests
# ============================================================

def test_calculate_total(sales_df):
    result = calculate_total(
        sales_df,
        "Sales",
    )

    assert result["operation"] == "total"
    assert result["column"] == "Sales"
    assert result["value"] == 1500


def test_calculate_total_ignores_missing_values():
    df = pd.DataFrame(
        {
            "Sales": [100, None, 300],
        }
    )

    result = calculate_total(
        df,
        "Sales",
    )

    assert result["value"] == 400


# ============================================================
# Average tests
# ============================================================

def test_calculate_average(sales_df):
    result = calculate_average(
        sales_df,
        "Sales",
    )

    assert result["operation"] == "average"
    assert result["column"] == "Sales"
    assert result["value"] == 300.0


# ============================================================
# Minimum / Maximum tests
# ============================================================

def test_calculate_minimum(sales_df):
    result = calculate_minimum(
        sales_df,
        "Sales",
    )

    assert result["operation"] == "minimum"
    assert result["value"] == 100


def test_calculate_maximum(sales_df):
    result = calculate_maximum(
        sales_df,
        "Sales",
    )

    assert result["operation"] == "maximum"
    assert result["value"] == 500


# ============================================================
# Group-by tests
# ============================================================

def test_group_by_sum(sales_df):
    result = group_by_aggregation(
        sales_df,
        "Region",
        "Sales",
        "sum",
    )

    data = result["data"]

    assert data == [
        {"Region": "Delhi", "value": 500},
        {"Region": "Mumbai", "value": 600},
        {"Region": "Pune", "value": 400},
    ]


def test_group_by_mean(sales_df):
    result = group_by_aggregation(
        sales_df,
        "Region",
        "Sales",
        "mean",
    )

    data = result["data"]

    assert data == [
        {"Region": "Delhi", "value": 500.0},
        {"Region": "Mumbai", "value": 300.0},
        {"Region": "Pune", "value": 200.0},
    ]


def test_group_by_min(sales_df):
    result = group_by_aggregation(
        sales_df,
        "Region",
        "Sales",
        "min",
    )

    data = result["data"]

    assert data == [
        {"Region": "Delhi", "value": 500},
        {"Region": "Mumbai", "value": 200},
        {"Region": "Pune", "value": 100},
    ]


def test_group_by_max(sales_df):
    result = group_by_aggregation(
        sales_df,
        "Region",
        "Sales",
        "max",
    )

    data = result["data"]

    assert data == [
        {"Region": "Delhi", "value": 500},
        {"Region": "Mumbai", "value": 400},
        {"Region": "Pune", "value": 300},
    ]


def test_group_by_count(sales_df):
    result = group_by_aggregation(
        sales_df,
        "Region",
        "Sales",
        "count",
    )

    data = result["data"]

    assert data == [
        {"Region": "Delhi", "value": 1},
        {"Region": "Mumbai", "value": 2},
        {"Region": "Pune", "value": 2},
    ]


# ============================================================
# Top / Bottom N tests
# ============================================================

def test_top_n(sales_df):
    result = top_n(
        sales_df,
        "Sales",
        n=3,
    )

    assert result["operation"] == "top_n"
    assert result["column"] == "Sales"
    assert result["n"] == 3

    assert result["data"] == [
        {"rank": 1, "value": 500},
        {"rank": 2, "value": 400},
        {"rank": 3, "value": 300},
    ]


def test_bottom_n(sales_df):
    result = top_n(
        sales_df,
        "Sales",
        n=2,
        ascending=True,
    )

    assert result["operation"] == "bottom_n"
    assert result["data"] == [
        {"rank": 1, "value": 100},
        {"rank": 2, "value": 200},
    ]


# ============================================================
# Correlation tests
# ============================================================

def test_calculate_correlation(sales_df):
    result = calculate_correlation(
        sales_df,
        "Sales",
        "Quantity",
    )

    assert result["operation"] == "correlation"
    assert result["column_a"] == "Sales"
    assert result["column_b"] == "Quantity"
    assert result["value"] == pytest.approx(1.0)


def test_calculate_correlation_handles_missing_values():
    df = pd.DataFrame(
        {
            "Sales": [100, 200, None, 400],
            "Quantity": [1, 2, 3, 4],
        }
    )

    result = calculate_correlation(
        df,
        "Sales",
        "Quantity",
    )

    assert result["value"] == pytest.approx(1.0)


# ============================================================
# Validation tests
# ============================================================

def test_invalid_column_raises_error(sales_df):
    with pytest.raises(AnalyticsError):
        calculate_total(
            sales_df,
            "Revenue",
        )


def test_non_numeric_column_raises_error(sales_df):
    with pytest.raises(AnalyticsError):
        calculate_average(
            sales_df,
            "Region",
        )


def test_invalid_aggregation_raises_error(sales_df):
    with pytest.raises(AnalyticsError):
        group_by_aggregation(
            sales_df,
            "Region",
            "Sales",
            "median",
        )


def test_invalid_n_raises_error(sales_df):
    with pytest.raises(AnalyticsError):
        top_n(
            sales_df,
            "Sales",
            n=0,
        )


def test_non_integer_n_raises_error(sales_df):
    with pytest.raises(AnalyticsError):
        top_n(
            sales_df,
            "Sales",
            n=2.5,
        )


def test_correlation_requires_two_valid_rows():
    df = pd.DataFrame(
        {
            "Sales": [100, None],
            "Quantity": [1, None],
        }
    )

    with pytest.raises(AnalyticsError):
        calculate_correlation(
            df,
            "Sales",
            "Quantity",
        )


def test_empty_numeric_column_raises_error():
    df = pd.DataFrame(
        {
            "Sales": [None, None, None],
        }
    )

    with pytest.raises(AnalyticsError):
        calculate_total(
            df,
            "Sales",
        )