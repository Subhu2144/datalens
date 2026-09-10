import numpy as np
import pandas as pd


class AnalyticsError(Exception):
    """Raised when an analytics operation cannot be completed."""


def _validate_column(df, column):
    """Validate that a requested column exists."""
    if column not in df.columns:
        raise AnalyticsError(
            f"Column '{column}' does not exist in the dataset."
        )


def _validate_numeric_column(df, column):
    """Validate that a requested column is numeric."""
    _validate_column(df, column)

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise AnalyticsError(
            f"Column '{column}' must be numeric for this operation."
        )


def _clean_value(value):
    """Convert NumPy/Pandas values into JSON-friendly Python values."""

    if pd.isna(value):
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


def calculate_summary(df, column):
    """
    Calculate basic summary statistics for a numeric column.

    Returns:
        dict containing count, sum, mean, median, min, max and std.
    """

    _validate_numeric_column(df, column)

    values = df[column].dropna()

    if values.empty:
        raise AnalyticsError(
            f"Column '{column}' contains no valid numeric values."
        )

    return {
        "operation": "summary",
        "column": column,
        "count": int(values.count()),
        "sum": _clean_value(values.sum()),
        "mean": _clean_value(values.mean()),
        "median": _clean_value(values.median()),
        "min": _clean_value(values.min()),
        "max": _clean_value(values.max()),
        "std": _clean_value(values.std()),
    }


def calculate_total(df, column):
    """Calculate the total of a numeric column."""

    _validate_numeric_column(df, column)

    values = df[column].dropna()

    if values.empty:
        raise AnalyticsError(
            f"Column '{column}' contains no valid numeric values."
        )

    return {
        "operation": "total",
        "column": column,
        "value": _clean_value(values.sum()),
    }


def calculate_average(df, column):
    """Calculate the average of a numeric column."""

    _validate_numeric_column(df, column)

    values = df[column].dropna()

    if values.empty:
        raise AnalyticsError(
            f"Column '{column}' contains no valid numeric values."
        )

    return {
        "operation": "average",
        "column": column,
        "value": _clean_value(values.mean()),
    }


def calculate_minimum(df, column):
    """Find the minimum value of a numeric column."""

    _validate_numeric_column(df, column)

    values = df[column].dropna()

    if values.empty:
        raise AnalyticsError(
            f"Column '{column}' contains no valid numeric values."
        )

    return {
        "operation": "minimum",
        "column": column,
        "value": _clean_value(values.min()),
    }


def calculate_maximum(df, column):
    """Find the maximum value of a numeric column."""

    _validate_numeric_column(df, column)

    values = df[column].dropna()

    if values.empty:
        raise AnalyticsError(
            f"Column '{column}' contains no valid numeric values."
        )

    return {
        "operation": "maximum",
        "column": column,
        "value": _clean_value(values.max()),
    }


def group_by_aggregation(
    df,
    group_column,
    value_column,
    aggregation="sum",
):
    """
    Aggregate a numeric column by a categorical/group column.

    Supported aggregations:
        sum, mean, min, max, count
    """

    _validate_column(df, group_column)
    _validate_column(df, value_column)

    if aggregation not in {
        "sum",
        "mean",
        "min",
        "max",
        "count",
    }:
        raise AnalyticsError(
            "Unsupported aggregation. "
            "Use sum, mean, min, max, or count."
        )

    if aggregation != "count":
        _validate_numeric_column(df, value_column)

    if aggregation == "sum":
        grouped = (
            df.groupby(
                group_column,
                dropna=False,
                observed=True,
            )[value_column]
            .sum()
            .reset_index()
        )

    elif aggregation == "mean":
        grouped = (
            df.groupby(
                group_column,
                dropna=False,
                observed=True,
            )[value_column]
            .mean()
            .reset_index()
        )

    elif aggregation == "min":
        grouped = (
            df.groupby(
                group_column,
                dropna=False,
                observed=True,
            )[value_column]
            .min()
            .reset_index()
        )

    elif aggregation == "max":
        grouped = (
            df.groupby(
                group_column,
                dropna=False,
                observed=True,
            )[value_column]
            .max()
            .reset_index()
        )

    else:
        grouped = (
            df.groupby(
                group_column,
                dropna=False,
                observed=True,
            )[value_column]
            .count()
            .reset_index()
        )

    grouped.columns = [
        group_column,
        "value",
    ]

    grouped["value"] = grouped["value"].apply(
        _clean_value
    )

    records = grouped.to_dict(
        orient="records"
    )

    return {
        "operation": "group_by",
        "group_column": group_column,
        "value_column": value_column,
        "aggregation": aggregation,
        "data": records,
    }


def top_n(
    df,
    column,
    n=10,
    ascending=False,
):
    """
    Return the top or bottom N values from a numeric column.
    """

    _validate_numeric_column(df, column)

    if not isinstance(n, int) or isinstance(n, bool):
        raise AnalyticsError(
            "n must be an integer."
        )

    if n <= 0:
        raise AnalyticsError(
            "n must be greater than zero."
        )

    values = (
        df[column]
        .dropna()
        .sort_values(
            ascending=ascending
        )
        .head(n)
    )

    records = [
        {
            "rank": index + 1,
            "value": _clean_value(value),
        }
        for index, value in enumerate(values.tolist())
    ]

    return {
        "operation": "top_n" if not ascending else "bottom_n",
        "column": column,
        "n": n,
        "data": records,
    }


def calculate_correlation(
    df,
    column_a,
    column_b,
):
    """Calculate Pearson correlation between two numeric columns."""

    _validate_numeric_column(df, column_a)
    _validate_numeric_column(df, column_b)

    valid_data = df[
        [column_a, column_b]
    ].dropna()

    if len(valid_data) < 2:
        raise AnalyticsError(
            "At least two valid rows are required "
            "to calculate correlation."
        )

    correlation = valid_data[
        column_a
    ].corr(
        valid_data[column_b]
    )

    return {
        "operation": "correlation",
        "column_a": column_a,
        "column_b": column_b,
        "value": _clean_value(correlation),
    }