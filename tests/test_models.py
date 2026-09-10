import pytest
from pydantic import ValidationError

from app.models import AnalysisIntent


def test_valid_summary_intent():
    intent = AnalysisIntent(
        operation="summary",
        column="Sales",
    )

    assert intent.operation == "summary"
    assert intent.column == "Sales"


def test_valid_total_intent():
    intent = AnalysisIntent(
        operation="total",
        column="Sales",
    )

    assert intent.operation == "total"
    assert intent.column == "Sales"


def test_valid_group_by_intent():
    intent = AnalysisIntent(
        operation="group_by",
        group_column="City",
        value_column="Sales",
        aggregation="sum",
    )

    assert intent.operation == "group_by"
    assert intent.group_column == "City"
    assert intent.value_column == "Sales"
    assert intent.aggregation == "sum"


def test_valid_top_n_intent():
    intent = AnalysisIntent(
        operation="top_n",
        column="Sales",
        n=5,
    )

    assert intent.operation == "top_n"
    assert intent.column == "Sales"
    assert intent.n == 5


def test_valid_bottom_n_intent():
    intent = AnalysisIntent(
        operation="bottom_n",
        column="Sales",
        n=3,
    )

    assert intent.operation == "bottom_n"
    assert intent.n == 3


def test_valid_correlation_intent():
    intent = AnalysisIntent(
        operation="correlation",
        column_a="Sales",
        column_b="Quantity",
    )

    assert intent.operation == "correlation"
    assert intent.column_a == "Sales"
    assert intent.column_b == "Quantity"


def test_all_basic_operations_are_supported():
    operations = [
        "summary",
        "total",
        "average",
        "minimum",
        "maximum",
        "group_by",
        "top_n",
        "bottom_n",
        "correlation",
    ]

    for operation in operations:
        intent = AnalysisIntent(
            operation=operation,
        )

        assert intent.operation == operation


def test_invalid_operation_is_rejected():
    with pytest.raises(ValidationError):
        AnalysisIntent(
            operation="delete_data",
        )


def test_zero_n_is_rejected():
    with pytest.raises(ValidationError):
        AnalysisIntent(
            operation="top_n",
            column="Sales",
            n=0,
        )


def test_negative_n_is_rejected():
    with pytest.raises(ValidationError):
        AnalysisIntent(
            operation="top_n",
            column="Sales",
            n=-5,
        )


def test_valid_n_is_accepted():
    intent = AnalysisIntent(
        operation="top_n",
        column="Sales",
        n=10,
    )

    assert intent.n == 10


def test_optional_fields_default_to_none():
    intent = AnalysisIntent(
        operation="average",
        column="Sales",
    )

    assert intent.group_column is None
    assert intent.value_column is None
    assert intent.aggregation is None
    assert intent.n is None
    assert intent.column_a is None
    assert intent.column_b is None
    assert intent.question is None


def test_invalid_aggregation_is_rejected():
    with pytest.raises(ValidationError):
        AnalysisIntent(
            operation="group_by",
            group_column="City",
            value_column="Sales",
            aggregation="median",
        )


def test_extra_fields_are_ignored():
    intent = AnalysisIntent(
        operation="summary",
        column="Sales",
        unexpected_field="something",
    )

    assert intent.operation == "summary"
    assert intent.column == "Sales"
    assert not hasattr(intent, "unexpected_field")