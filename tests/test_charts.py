import pandas as pd

from app.charts import generate_analysis_chart
from app.models import AnalysisIntent


def make_intent(operation, **kwargs):
    return AnalysisIntent(
        operation=operation,
        **kwargs,
    )


def test_group_by_generates_bar_chart():
    result = {
        "operation": "group_by",
        "group_column": "City",
        "value_column": "Sales",
        "aggregation": "sum",
        "data": pd.DataFrame(
            {
                "City": ["Pune", "Mumbai", "Delhi"],
                "Sales": [1000, 800, 600],
            }
        ),
    }

    intent = make_intent(
        "group_by",
        group_column="City",
        value_column="Sales",
        aggregation="sum",
    )

    chart = generate_analysis_chart(result, intent)

    assert chart is not None
    assert chart["type"] == "bar"
    assert chart["figure"] is not None
    assert len(chart["figure"].data) == 1


def test_top_n_generates_bar_chart():
    result = {
        "operation": "top_n",
        "column": "Sales",
        "n": 3,
        "data": pd.DataFrame(
            {
                "Product": ["A", "B", "C"],
                "Sales": [1000, 800, 600],
            }
        ),
    }

    intent = make_intent(
        "top_n",
        column="Sales",
        n=3,
    )

    chart = generate_analysis_chart(result, intent)

    assert chart is not None
    assert chart["type"] == "bar"
    assert chart["figure"] is not None
    assert len(chart["figure"].data) == 1


def test_bottom_n_generates_bar_chart():
    result = {
        "operation": "bottom_n",
        "column": "Sales",
        "n": 3,
        "data": pd.DataFrame(
            {
                "Product": ["A", "B", "C"],
                "Sales": [100, 200, 300],
            }
        ),
    }

    intent = make_intent(
        "bottom_n",
        column="Sales",
        n=3,
    )

    chart = generate_analysis_chart(result, intent)

    assert chart is not None
    assert chart["type"] == "bar"
    assert chart["figure"] is not None
    assert len(chart["figure"].data) == 1


def test_total_does_not_generate_chart():
    result = {
        "operation": "total",
        "column": "Sales",
        "value": 2400,
    }

    intent = make_intent(
        "total",
        column="Sales",
    )

    chart = generate_analysis_chart(result, intent)

    assert chart is None


def test_average_does_not_generate_chart():
    result = {
        "operation": "average",
        "column": "Sales",
        "value": 800,
    }

    intent = make_intent(
        "average",
        column="Sales",
    )

    chart = generate_analysis_chart(result, intent)

    assert chart is None


def test_minimum_does_not_generate_chart():
    result = {
        "operation": "minimum",
        "column": "Sales",
        "value": 100,
    }

    intent = make_intent(
        "minimum",
        column="Sales",
    )

    chart = generate_analysis_chart(result, intent)

    assert chart is None


def test_maximum_does_not_generate_chart():
    result = {
        "operation": "maximum",
        "column": "Sales",
        "value": 1500,
    }

    intent = make_intent(
        "maximum",
        column="Sales",
    )

    chart = generate_analysis_chart(result, intent)

    assert chart is None


def test_summary_does_not_generate_chart():
    result = {
        "operation": "summary",
        "data": {
            "rows": 100,
            "columns": 5,
        },
    }

    intent = make_intent("summary")

    chart = generate_analysis_chart(result, intent)

    assert chart is None


def test_invalid_result_returns_none():
    result = {}

    intent = make_intent("group_by")

    chart = generate_analysis_chart(result, intent)

    assert chart is None