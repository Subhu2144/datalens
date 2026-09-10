import pandas as pd
import plotly.express as px


def generate_auto_charts(df):
    """
    Generate useful automatic Plotly charts from a dataframe.

    Returns:
        list[dict]: Chart metadata and Plotly figures.
    """

    charts = []

    if df.empty:
        return charts

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # --------------------------------------------------------
    # 1. Numeric distribution
    # --------------------------------------------------------

    if numeric_columns:
        column = numeric_columns[0]

        chart_df = df[[column]].dropna()

        if not chart_df.empty:
            figure = px.histogram(
                chart_df,
                x=column,
                title=f"Distribution of {column}",
            )

            charts.append(
                {
                    "type": "histogram",
                    "title": f"Distribution of {column}",
                    "column": column,
                    "figure": figure,
                }
            )

    # --------------------------------------------------------
    # 2. Categorical frequency
    # --------------------------------------------------------

    if categorical_columns:
        column = categorical_columns[0]

        counts = (
            df[column]
            .dropna()
            .astype(str)
            .value_counts()
            .head(10)
            .reset_index()
        )

        counts.columns = [
            column,
            "Count",
        ]

        if not counts.empty:
            figure = px.bar(
                counts,
                x=column,
                y="Count",
                title=f"Top Values in {column}",
            )

            charts.append(
                {
                    "type": "bar",
                    "title": f"Top Values in {column}",
                    "column": column,
                    "figure": figure,
                }
            )

    # --------------------------------------------------------
    # 3. Numeric comparison by category
    # --------------------------------------------------------

    if numeric_columns and categorical_columns:
        numeric_column = numeric_columns[0]
        categorical_column = categorical_columns[0]

        grouped = (
            df.groupby(
                categorical_column,
                dropna=False,
                observed=True,
            )[numeric_column]
            .mean()
            .reset_index()
            .sort_values(
                numeric_column,
                ascending=False,
            )
            .head(10)
        )

        if not grouped.empty:
            figure = px.bar(
                grouped,
                x=categorical_column,
                y=numeric_column,
                title=(
                    f"Average {numeric_column} "
                    f"by {categorical_column}"
                ),
            )

            charts.append(
                {
                    "type": "grouped_bar",
                    "title": (
                        f"Average {numeric_column} "
                        f"by {categorical_column}"
                    ),
                    "column": numeric_column,
                    "group_by": categorical_column,
                    "figure": figure,
                }
            )

    # --------------------------------------------------------
    # 4. Numeric correlation
    # --------------------------------------------------------

    if len(numeric_columns) >= 2:
        correlation_df = df[numeric_columns].corr()

        figure = px.imshow(
            correlation_df,
            text_auto=True,
            title="Numeric Column Correlation",
            aspect="auto",
        )

        charts.append(
            {
                "type": "correlation",
                "title": "Numeric Column Correlation",
                "columns": numeric_columns,
                "figure": figure,
            }
        )

    return charts