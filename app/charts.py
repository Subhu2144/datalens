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


# ============================================================
# Natural-Language Analysis Charts
# ============================================================

def generate_analysis_chart(
    result,
    intent,
):
    """
    Generate a controlled Plotly chart from an
    already calculated analytical result.

    The chart is created only from the validated
    analysis intent and deterministic Python result.

    Gemini does not generate chart code.

    Returns:
        dict | None:
            Chart metadata and Plotly figure,
            or None when a chart is not appropriate.
    """

    if not result or not intent:
        return None

    operation = result.get("operation")

    # --------------------------------------------------------
    # 1. Group By → Bar Chart
    # --------------------------------------------------------

    if operation == "group_by":

        data = result.get("data")

        if data is None:
            return None

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        if data.empty or len(data.columns) < 2:
            return None

        group_column = intent.group_column

        value_column = intent.value_column

        if (
            group_column not in data.columns
            or value_column not in data.columns
        ):
            return None

        chart_df = data[
            [group_column, value_column]
        ].copy()

        chart_df = chart_df.dropna(
            subset=[group_column, value_column]
        )

        if chart_df.empty:
            return None

        figure = px.bar(
            chart_df,
            x=group_column,
            y=value_column,
            title=(
                f"{intent.aggregation.title()} "
                f"{value_column} by {group_column}"
            ),
        )

        return {
            "type": "bar",
            "title": (
                f"{intent.aggregation.title()} "
                f"{value_column} by {group_column}"
            ),
            "figure": figure,
        }

    # --------------------------------------------------------
    # 2. Top N → Bar Chart
    # --------------------------------------------------------

    if operation == "top_n":

        data = result.get("data")

        if data is None:
            return None

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        if data.empty or len(data.columns) < 2:
            return None

        column = intent.column

        if column not in data.columns:
            return None

        chart_df = data.copy()

        category_column = chart_df.columns[0]

        if category_column == column:
            return None

        figure = px.bar(
            chart_df,
            x=category_column,
            y=column,
            title=f"Top {intent.n} by {column}",
        )

        return {
            "type": "bar",
            "title": f"Top {intent.n} by {column}",
            "figure": figure,
        }

    # --------------------------------------------------------
    # 3. Bottom N → Bar Chart
    # --------------------------------------------------------

    if operation == "bottom_n":

        data = result.get("data")

        if data is None:
            return None

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        if data.empty or len(data.columns) < 2:
            return None

        column = intent.column

        if column not in data.columns:
            return None

        chart_df = data.copy()

        category_column = chart_df.columns[0]

        if category_column == column:
            return None

        figure = px.bar(
            chart_df,
            x=category_column,
            y=column,
            title=f"Bottom {intent.n} by {column}",
        )

        return {
            "type": "bar",
            "title": f"Bottom {intent.n} by {column}",
            "figure": figure,
        }

    # --------------------------------------------------------
    # 4. Correlation → Scatter Plot
    # --------------------------------------------------------

    if operation == "correlation":

        column_a = intent.column_a

        column_b = intent.column_b

        if not column_a or not column_b:
            return None

        if column_a == column_b:
            return None

        # Correlation result currently contains only
        # the calculated correlation value, not the
        # original dataframe. Therefore the scatter
        # chart will be handled separately when the
        # dataframe is available.
        return None

    # --------------------------------------------------------
    # 5. Scalar Results → No Chart
    # --------------------------------------------------------

    if operation in {
        "total",
        "average",
        "minimum",
        "maximum",
        "summary",
    }:

        return None

    return None