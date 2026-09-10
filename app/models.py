from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


AnalysisOperation = Literal[
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


class AnalysisIntent(BaseModel):
    operation: AnalysisOperation

    column: Optional[str] = None
    group_column: Optional[str] = None
    value_column: Optional[str] = None

    aggregation: Optional[
        Literal["sum", "mean", "min", "max", "count"]
    ] = None

    n: Optional[int] = None

    column_a: Optional[str] = None
    column_b: Optional[str] = None

    question: Optional[str] = None

    @field_validator("n")
    @classmethod
    def validate_n(cls, value):
        if value is not None and value <= 0:
            raise ValueError("n must be greater than zero.")
        return value