import os
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.models import AnalysisIntent


# ============================================================
# Environment
# ============================================================

load_dotenv()


class AgentError(Exception):
    """Raised when the DataLens agent cannot process a request."""


# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL = "gemini-3.6-flash"


def get_gemini_model():
    """Return configured Gemini model name."""

    return os.getenv(
        "GEMINI_MODEL",
        DEFAULT_MODEL,
    )


def get_gemini_api_key():
    """Return Gemini API key."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise AgentError(
            "GEMINI_API_KEY is not configured. "
            "Add it to the .env file."
        )

    return api_key


# ============================================================
# Prompt
# ============================================================

SYSTEM_INSTRUCTION = """
You are the planning component of DataLens,
an Intelligent Data Investigation Assistant.

Your job is to convert a user's natural-language
data question into a structured AnalysisIntent.

You DO NOT calculate any numbers.

You DO NOT invent dataset values.

You DO NOT write Python code.

You only identify the analytical operation and
the relevant dataset columns.

Supported operations:

- summary
- total
- average
- minimum
- maximum
- group_by
- top_n
- bottom_n
- correlation

Rules:

1. Use only columns that exist in the provided dataset.

2. For summary, total, average, minimum, maximum:
   use the requested numeric column as "column".

3. For group_by:
   ALWAYS provide all three fields:
   - group_column
   - value_column
   - aggregation

4. For group_by questions:
   if the question says "by Region", "grouped by Region",
   "per Region", or similar wording, use "Region" as
   group_column if that column exists in the dataset.

5. For top_n:
   use "column" and "n".

6. For bottom_n:
   use "column" and "n".

7. For correlation:
   use "column_a" and "column_b".

8. If the user asks for total by category,
   use group_by with aggregation="sum".

9. If the user asks for average by category,
   use group_by with aggregation="mean".

10. If the user asks "top N", use top_n.

11. If the user asks "bottom N", use bottom_n.

12. Preserve the original user question in "question".

13. NEVER omit a field that is required for the
    selected operation.

14. Return only the structured AnalysisIntent.
"""


# ============================================================
# Dataset Context
# ============================================================

def build_dataset_context(df):
    """
    Build a compact dataset schema for the LLM.

    Only column names and basic type information are sent.
    Actual numerical data is NOT sent to the planner.
    """

    columns = []

    for column in df.columns:
        columns.append(
            {
                "name": str(column),
                "dtype": str(df[column].dtype),
            }
        )

    return {
        "rows": int(len(df)),
        "columns": columns,
    }


# ============================================================
# Intent Normalization
# ============================================================

def _find_column_in_question(question, columns, patterns):
    """
    Find a dataset column referenced by a natural-language
    grouping phrase such as:

    - by Region
    - grouped by Region
    - per Region
    """

    question_lower = question.lower()

    for pattern in patterns:

        match = re.search(
            pattern,
            question_lower,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        mentioned_text = match.group(1).strip()

        # Exact dataset-column matching.
        for column in columns:

            if str(column).lower() == mentioned_text:
                return column

    return None


def normalize_intent(intent, question, df):
    """
    Validate and safely complete an AnalysisIntent.

    The LLM remains responsible for planning.
    This function only resolves missing references
    using the known dataset schema.
    """

    columns = df.columns.tolist()

    # --------------------------------------------------------
    # Group By
    # --------------------------------------------------------

    if (
        intent.operation == "group_by"
        and not intent.group_column
    ):

        group_column = _find_column_in_question(
            question,
            columns,
            patterns=[
                r"\bby\s+(.+?)(?:\?|$)",
                r"\bgrouped\s+by\s+(.+?)(?:\?|$)",
                r"\bgroup\s+by\s+(.+?)(?:\?|$)",
                r"\bper\s+(.+?)(?:\?|$)",
            ],
        )

        if group_column is not None:

            intent = intent.model_copy(
                update={
                    "group_column": str(group_column),
                }
            )

    # --------------------------------------------------------
    # Validate referenced columns
    # --------------------------------------------------------

    valid_columns = {
        str(column).lower(): str(column)
        for column in columns
    }

    referenced_columns = [
        ("column", intent.column),
        ("group_column", intent.group_column),
        ("value_column", intent.value_column),
        ("column_a", intent.column_a),
        ("column_b", intent.column_b),
    ]

    for field_name, value in referenced_columns:

        if value is None:
            continue

        if str(value).lower() not in valid_columns:

            raise AgentError(
                f"Column '{value}' from the analysis plan "
                f"does not exist in the dataset."
            )

    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    if intent.operation == "group_by":

        if not intent.group_column:

            raise AgentError(
                "Could not identify the group column. "
                "Please mention the grouping column clearly."
            )

        if not intent.value_column:

            raise AgentError(
                "Could not identify the value column."
            )

        if not intent.aggregation:

            raise AgentError(
                "Could not identify the aggregation."
            )

    if intent.operation in {
        "summary",
        "total",
        "average",
        "minimum",
        "maximum",
    }:

        if not intent.column:

            raise AgentError(
                "Could not identify the numeric column."
            )

    if intent.operation in {
        "top_n",
        "bottom_n",
    }:

        if not intent.column:

            raise AgentError(
                "Could not identify the numeric column."
            )

        if not intent.n:

            raise AgentError(
                "Could not identify the requested N."
            )

    if intent.operation == "correlation":

        if not intent.column_a:

            raise AgentError(
                "Could not identify the first column."
            )

        if not intent.column_b:

            raise AgentError(
                "Could not identify the second column."
            )

    return intent


def _format_gemini_error(exc: Exception, operation: str) -> str:
    """Convert common Gemini API failures into user-friendly messages."""
    error_text = str(exc)
    normalized = error_text.upper()

    if "429" in normalized or "RESOURCE_EXHAUSTED" in normalized:
        return (
            "Gemini API quota is temporarily exhausted. "
            "Please try again later or check your Gemini API quota and billing settings."
        )

    return f"Gemini {operation} failed: {exc}"


# ============================================================
# Intent Generation
# ============================================================

def generate_intent(
    question,
    df,
):
    """
    Convert a natural-language question into AnalysisIntent.

    The LLM only plans the operation.
    Numerical calculations remain inside analytics.py.
    """

    if not question or not question.strip():

        raise AgentError(
            "Please enter a data question."
        )

    if df is None or df.empty:

        raise AgentError(
            "A non-empty dataset is required."
        )

    api_key = get_gemini_api_key()

    client = genai.Client(
        api_key=api_key
    )

    dataset_context = build_dataset_context(
        df
    )

    prompt = f"""
{SYSTEM_INSTRUCTION}

Dataset schema:

{dataset_context}

User question:

{question.strip()}
"""

    try:

        response = client.models.generate_content(
            model=get_gemini_model(),
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AnalysisIntent,
                temperature=0,
            ),
        )

    except Exception as exc:

        raise AgentError(
            _format_gemini_error(exc, "request")
        ) from exc

    try:

        if hasattr(
            response,
            "parsed",
        ) and response.parsed:

            intent = response.parsed

            if not isinstance(
                intent,
                AnalysisIntent,
            ):

                intent = AnalysisIntent.model_validate(
                    intent
                )

        else:

            intent = AnalysisIntent.model_validate_json(
                response.text
            )

    except Exception as exc:

        raise AgentError(
            f"Gemini returned an invalid analysis intent: {exc}"
        ) from exc

    # --------------------------------------------------------
    # Normalize and validate the generated intent.
    # --------------------------------------------------------

    return normalize_intent(
        intent,
        question,
        df,
    )


# ============================================================
# Intent Execution
# ============================================================

def execute_intent(intent, df):
    """
    Execute a validated AnalysisIntent using deterministic
    analytics functions.

    Gemini only decides WHAT operation to perform.
    analytics.py performs the actual calculation.
    """

    from app import analytics

    try:

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        if intent.operation == "summary":

            if not intent.column:

                raise AgentError(
                    "A column is required for summary."
                )

            return analytics.calculate_summary(
                df,
                intent.column,
            )

        # ----------------------------------------------------
        # Total
        # ----------------------------------------------------

        if intent.operation == "total":

            if not intent.column:

                raise AgentError(
                    "A column is required for total."
                )

            return analytics.calculate_total(
                df,
                intent.column,
            )

        # ----------------------------------------------------
        # Average
        # ----------------------------------------------------

        if intent.operation == "average":

            if not intent.column:

                raise AgentError(
                    "A column is required for average."
                )

            return analytics.calculate_average(
                df,
                intent.column,
            )

        # ----------------------------------------------------
        # Minimum
        # ----------------------------------------------------

        if intent.operation == "minimum":

            if not intent.column:

                raise AgentError(
                    "A column is required for minimum."
                )

            return analytics.calculate_minimum(
                df,
                intent.column,
            )

        # ----------------------------------------------------
        # Maximum
        # ----------------------------------------------------

        if intent.operation == "maximum":

            if not intent.column:

                raise AgentError(
                    "A column is required for maximum."
                )

            return analytics.calculate_maximum(
                df,
                intent.column,
            )

        # ----------------------------------------------------
        # Group By
        # ----------------------------------------------------

        if intent.operation == "group_by":

            if not intent.group_column:

                raise AgentError(
                    "A group column is required."
                )

            if not intent.value_column:

                raise AgentError(
                    "A value column is required."
                )

            if not intent.aggregation:

                raise AgentError(
                    "An aggregation is required."
                )

            return analytics.group_by_aggregation(
                df,
                group_column=intent.group_column,
                value_column=intent.value_column,
                aggregation=intent.aggregation,
            )

        # ----------------------------------------------------
        # Top N
        # ----------------------------------------------------

        if intent.operation == "top_n":

            if not intent.column:

                raise AgentError(
                    "A column is required for top_n."
                )

            if not intent.n:

                raise AgentError(
                    "n is required for top_n."
                )

            return analytics.top_n(
                df,
                column=intent.column,
                n=intent.n,
                ascending=False,
            )

        # ----------------------------------------------------
        # Bottom N
        # ----------------------------------------------------

        if intent.operation == "bottom_n":

            if not intent.column:

                raise AgentError(
                    "A column is required for bottom_n."
                )

            if not intent.n:

                raise AgentError(
                    "n is required for bottom_n."
                )

            return analytics.top_n(
                df,
                column=intent.column,
                n=intent.n,
                ascending=True,
            )

        # ----------------------------------------------------
        # Correlation
        # ----------------------------------------------------

        if intent.operation == "correlation":

            if not intent.column_a:

                raise AgentError(
                    "column_a is required for correlation."
                )

            if not intent.column_b:

                raise AgentError(
                    "column_b is required for correlation."
                )

            return analytics.calculate_correlation(
                df,
                intent.column_a,
                intent.column_b,
            )

        # ----------------------------------------------------
        # Unsupported Operation
        # ----------------------------------------------------

        raise AgentError(
            f"Unsupported analysis operation: {intent.operation}"
        )

    except analytics.AnalyticsError as exc:

        raise AgentError(
            str(exc)
        ) from exc



# ============================================================
# Result Explanation
# ============================================================

def generate_explanation(
    question,
    intent,
    result,
):
    """
    Generate a concise natural-language explanation
    from an already-calculated analytics result.

    Gemini does NOT calculate or modify the result.
    It only explains the values produced by analytics.py.
    """

    if not question or not question.strip():
        raise AgentError(
            "A question is required for explanation."
        )

    if intent is None:
        raise AgentError(
            "An analysis intent is required for explanation."
        )

    if result is None:
        raise AgentError(
            "An analysis result is required for explanation."
        )

    api_key = get_gemini_api_key()

    client = genai.Client(
        api_key=api_key
    )

    prompt = f"""
You are the explanation component of DataLens,
an Intelligent Data Investigation Assistant.

Your job is to explain an analysis result that has
ALREADY been calculated by Python/Pandas.

You MUST follow these rules:

1. Do NOT recalculate any values.
2. Do NOT invent any values.
3. Do NOT modify any values.
4. Use ONLY the provided analysis result.
5. Clearly answer the user's original question.
6. Mention the most important insight when it is
   directly supported by the result.
7. Keep the explanation concise and professional.
8. If the result contains rankings or groups,
   mention the highest and lowest when clearly available.
9. Do not mention internal implementation details
   unless necessary.
10. Return plain text only.

Original question:
{question.strip()}

Analysis intent:
{intent.model_dump()}

Calculated result:
{result}

Write a concise answer for the user.
"""

    try:

        response = client.models.generate_content(
            model=get_gemini_model(),
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
            ),
        )

    except Exception as exc:

        raise AgentError(
            _format_gemini_error(exc, "explanation request")
        ) from exc

    explanation = response.text.strip()

    if not explanation:
        raise AgentError(
            "Gemini returned an empty explanation."
        )

    return explanation