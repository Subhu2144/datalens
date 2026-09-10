import pandas as pd
import streamlit as st

from app.agent import (
    AgentError,
    execute_intent,
    generate_explanation,
    generate_intent,
)

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

from app.charts import (
    generate_analysis_chart,
    generate_auto_charts,
)

from app.data_processor import (
    DatasetValidationError,
    apply_cleaning,
    get_dataset_summary,
    load_dataset,
    profile_dataset,
    run_quality_checks,
)


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="DataLens",
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# Session State
# ============================================================

def initialize_session_state():

    if "dataframe" not in st.session_state:
        st.session_state.dataframe = None

    if "file_name" not in st.session_state:
        st.session_state.file_name = None

    if "profile" not in st.session_state:
        st.session_state.profile = None

    if "quality_report" not in st.session_state:
        st.session_state.quality_report = None

    if "charts" not in st.session_state:
        st.session_state.charts = []

    if "cleaned_dataframe" not in st.session_state:
        st.session_state.cleaned_dataframe = None


# ============================================================
# Header
# ============================================================

def display_header():

    st.title("🔎 DataLens")

    st.caption(
        "Intelligent Data Investigation Assistant"
    )

    st.write(
        "Upload a CSV or Excel dataset to profile, "
        "validate, visualize, and investigate your data."
    )


# ============================================================
# Dataset Upload
# ============================================================

def display_upload_section():

    st.subheader("📁 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx"],
        help="Supported formats: CSV and XLSX",
    )

    if uploaded_file is None:
        return

    try:

        file_bytes = uploaded_file.getvalue()

        dataframe = load_dataset(
            uploaded_file.name,
            file_bytes,
        )

        st.session_state.dataframe = dataframe
        st.session_state.cleaned_dataframe = None

        st.session_state.file_name = uploaded_file.name

        st.session_state.profile = profile_dataset(
            dataframe
        )

        st.session_state.quality_report = run_quality_checks(
            dataframe
        )

        st.session_state.charts = generate_auto_charts(
            dataframe
        )

        st.success(
            f"Successfully loaded {uploaded_file.name}"
        )

    except DatasetValidationError as exc:

        st.error(str(exc))

        st.session_state.dataframe = None
        st.session_state.file_name = None
        st.session_state.profile = None
        st.session_state.quality_report = None
        st.session_state.charts = []
        st.session_state.cleaned_dataframe = None

    except Exception as exc:

        st.error(
            f"Unexpected error while loading dataset: {exc}"
        )

        st.session_state.dataframe = None
        st.session_state.file_name = None
        st.session_state.profile = None
        st.session_state.quality_report = None
        st.session_state.charts = []
        st.session_state.cleaned_dataframe = None


# ============================================================
# Dataset Cleaning
# ============================================================

def display_cleaning_dashboard():
    dataframe = st.session_state.dataframe

    if dataframe is None:
        return

    st.subheader("🧹 Data Cleaning")

    st.caption(
        "Create a cleaned copy of your dataset without modifying "
        "the original uploaded data."
    )

    duplicate_count = int(dataframe.duplicated().sum())
    missing_count = int(dataframe.isna().sum().sum())

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Missing Cells", missing_count)

    with col2:
        st.metric("Duplicate Rows", duplicate_count)

    if missing_count == 0 and duplicate_count == 0:
        st.success("No missing values or duplicate rows detected.")
        return

    st.markdown("### Cleaning Options")

    options = st.multiselect(
        "Select cleaning operations",
        [
            "Remove duplicate rows",
            "Fill numeric missing values with median",
            "Fill categorical missing values with mode",
            "Drop rows with missing values",
        ],
        key="cleaning_options",
    )

    if (
        "Fill numeric missing values with median" in options
        and not dataframe.select_dtypes(include="number").columns.tolist()
    ):
        st.warning("No numeric columns are available for median filling.")

    if (
        "Fill categorical missing values with mode" in options
        and not dataframe.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
    ):
        st.warning("No categorical columns are available for mode filling.")

    if st.button("Preview Cleaning", key="preview_cleaning_button"):
        cleaned = apply_cleaning(
            dataframe,
            remove_duplicates="Remove duplicate rows" in options,
            fill_numeric_median=(
                "Fill numeric missing values with median" in options
            ),
            fill_categorical_mode=(
                "Fill categorical missing values with mode" in options
            ),
            drop_missing_rows="Drop rows with missing values" in options,
        )

        st.session_state.cleaned_dataframe = cleaned

    cleaned_dataframe = st.session_state.cleaned_dataframe

    if cleaned_dataframe is None:
        if not options:
            st.info("Select at least one cleaning operation.")
        return

    st.markdown("### Cleaning Preview")

    original_missing = int(dataframe.isna().sum().sum())
    cleaned_missing = int(cleaned_dataframe.isna().sum().sum())

    original_duplicates = int(dataframe.duplicated().sum())
    cleaned_duplicates = int(cleaned_dataframe.duplicated().sum())

    comparison_col1, comparison_col2 = st.columns(2)

    with comparison_col1:
        st.markdown("**Before**")
        st.write(f"Rows: {len(dataframe)}")
        st.write(f"Missing cells: {original_missing}")
        st.write(f"Duplicate rows: {original_duplicates}")

    with comparison_col2:
        st.markdown("**After**")
        st.write(f"Rows: {len(cleaned_dataframe)}")
        st.write(f"Missing cells: {cleaned_missing}")
        st.write(f"Duplicate rows: {cleaned_duplicates}")

    st.dataframe(
        cleaned_dataframe.head(10),
        use_container_width=True,
    )

    if st.button("Apply Cleaning", key="apply_cleaning_button"):
        st.session_state.dataframe = cleaned_dataframe.copy()
        st.session_state.cleaned_dataframe = None

        st.session_state.profile = profile_dataset(
            st.session_state.dataframe
        )

        st.session_state.quality_report = run_quality_checks(
            st.session_state.dataframe
        )

        st.session_state.charts = generate_auto_charts(
            st.session_state.dataframe
        )

        st.success("Cleaning applied successfully.")
        st.rerun()


# ============================================================
# Dataset Overview
# ============================================================

def display_dataset_overview():

    dataframe = st.session_state.dataframe

    if dataframe is None:
        return

    summary = get_dataset_summary(dataframe)

    st.subheader("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Rows",
            summary["rows"],
        )

    with col2:

        st.metric(
            "Columns",
            summary["columns"],
        )

    with col3:

        st.metric(
            "File",
            st.session_state.file_name,
        )


# ============================================================
# Dataset Preview
# ============================================================

def display_dataset_preview():

    dataframe = st.session_state.dataframe

    if dataframe is None:
        return

    st.subheader("👀 Dataset Preview")

    st.dataframe(
        dataframe.head(10),
        use_container_width=True,
    )


# ============================================================
# Profiling Dashboard
# ============================================================

def display_profiling_dashboard():

    profile = st.session_state.profile

    if profile is None:
        return

    st.subheader("🔬 Data Profile")

    summary = profile["summary"]

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Cells",
            summary["total_cells"],
        )

    with col2:

        st.metric(
            "Missing Cells",
            summary["missing_cells"],
        )

    with col3:

        st.metric(
            "Missing %",
            f'{summary["missing_percentage"]}%',
        )

    with col4:

        st.metric(
            "Duplicate Rows",
            summary["duplicate_rows"],
        )

    st.markdown("### Column Information")

    column_details = profile["column_details"]

    if column_details:

        st.dataframe(
            column_details,
            use_container_width=True,
        )

    st.markdown("### Missing Values")

    missing_values = profile["missing_values"]

    if missing_values:

        st.dataframe(
            missing_values,
            use_container_width=True,
        )

    else:

        st.success(
            "No missing values found."
        )

    st.markdown("### Numeric Statistics")

    numeric_statistics = profile["numeric_statistics"]

    if numeric_statistics:

        statistics_rows = []

        for column, statistics in numeric_statistics.items():

            statistics_rows.append(
                {
                    "Column": column,
                    **statistics,
                }
            )

        st.dataframe(
            statistics_rows,
            use_container_width=True,
        )

    else:

        st.info(
            "No numeric columns found."
        )


# ============================================================
# Data Quality Dashboard
# ============================================================

def display_quality_dashboard():

    quality_report = st.session_state.quality_report

    if quality_report is None:
        return

    st.subheader("🛡️ Data Quality")

    score = quality_report["score"]

    summary = quality_report["summary"]

    issues = quality_report["issues"]

    if score >= 90:

        score_status = "Excellent"

    elif score >= 75:

        score_status = "Good"

    elif score >= 50:

        score_status = "Needs Attention"

    else:

        score_status = "Poor"

    col1, col2 = st.columns([1, 2])

    with col1:

        st.metric(
            "Quality Score",
            f"{score}/100",
        )

    with col2:

        st.write(
            f"**Status:** {score_status}"
        )

        st.progress(
            score / 100
        )

    st.caption(
        "Quality score is an app-defined heuristic based on "
        "missing values, duplicate rows, constant columns, "
        "and potential outliers."
    )

    st.markdown("### Quality Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Missing Issues",
            summary["missing_issues"],
        )

    with col2:

        st.metric(
            "Duplicate Rows",
            summary["duplicate_rows"],
        )

    with col3:

        st.metric(
            "Constant Columns",
            summary["constant_columns"],
        )

    with col4:

        st.metric(
            "Outlier Columns",
            summary["outlier_columns"],
        )

    st.markdown("### Detected Issues")

    if not issues:

        st.success(
            "No major data-quality issues detected."
        )

        return

    for issue in issues:

        severity = issue["severity"]

        if severity == "high":

            st.error(
                f"🔴 **{issue['message']}**"
            )

        elif severity == "medium":

            st.warning(
                f"🟠 **{issue['message']}**"
            )

        else:

            st.info(
                f"🔵 **{issue['message']}**"
            )

    st.markdown("### Issue Details")

    issue_rows = []

    for issue in issues:

        issue_rows.append(
            {
                "Type": issue["type"],
                "Column": issue["column"] or "-",
                "Count": issue["count"],
                "Percentage": f'{issue["percentage"]}%',
                "Severity": issue["severity"],
                "Message": issue["message"],
            }
        )

    st.dataframe(
        issue_rows,
        use_container_width=True,
    )


# ============================================================
# Automatic Visualization Dashboard
# ============================================================

def display_visualization_dashboard():

    charts = st.session_state.charts

    if not charts:
        return

    st.subheader("📈 Automatic Visualizations")

    st.caption(
        "DataLens automatically selects basic visualizations "
        "based on the detected column types."
    )

    for index, chart in enumerate(charts):

        if index > 0:
            st.divider()

        st.markdown(
            f"### {chart['title']}"
        )

        st.plotly_chart(
            chart["figure"],
            use_container_width=True,
        )


# ============================================================
# Analytics Helpers
# ============================================================

def get_numeric_columns(df):

    return df.select_dtypes(
        include="number"
    ).columns.tolist()


def get_all_columns(df):

    return df.columns.tolist()


# ============================================================
# Analytics Result Display
# ============================================================

def display_analytics_result(result):

    operation = result["operation"]

    if operation == "summary":

        st.dataframe(
            [
                {
                    "Metric": "Count",
                    "Value": result["count"],
                },
                {
                    "Metric": "Sum",
                    "Value": result["sum"],
                },
                {
                    "Metric": "Mean",
                    "Value": result["mean"],
                },
                {
                    "Metric": "Median",
                    "Value": result["median"],
                },
                {
                    "Metric": "Minimum",
                    "Value": result["min"],
                },
                {
                    "Metric": "Maximum",
                    "Value": result["max"],
                },
                {
                    "Metric": "Standard Deviation",
                    "Value": result["std"],
                },
            ],
            use_container_width=True,
        )

    elif operation in {
        "total",
        "average",
        "minimum",
        "maximum",
    }:

        st.metric(
            operation.replace(
                "_",
                " "
            ).title(),
            result["value"],
        )

    elif operation == "group_by":

        st.dataframe(
            result["data"],
            use_container_width=True,
        )

    elif operation in {
        "top_n",
        "bottom_n",
    }:

        st.dataframe(
            result["data"],
            use_container_width=True,
        )

    elif operation == "correlation":

        st.metric(
            "Pearson Correlation",
            round(
                result["value"],
                4,
            ),
        )


# ============================================================
# Analytics Dashboard
# ============================================================

def display_analytics_dashboard():

    dataframe = st.session_state.dataframe

    if dataframe is None:
        return

    st.subheader("🧮 Data Analytics")

    st.caption(
        "Run deterministic analytical operations using "
        "Pandas. No LLM is used for the calculations."
    )

    numeric_columns = get_numeric_columns(
        dataframe
    )

    all_columns = get_all_columns(
        dataframe
    )

    operation = st.selectbox(
        "Choose an analysis",
        [
            "Summary",
            "Total",
            "Average",
            "Minimum",
            "Maximum",
            "Group By",
            "Top / Bottom N",
            "Correlation",
        ],
    )

    # --------------------------------------------------------
    # Summary / Total / Average / Min / Max
    # --------------------------------------------------------

    if operation in {
        "Summary",
        "Total",
        "Average",
        "Minimum",
        "Maximum",
    }:

        if not numeric_columns:

            st.warning(
                "No numeric columns are available for this analysis."
            )

            return

        column = st.selectbox(
            "Select numeric column",
            numeric_columns,
        )

        if st.button(
            "Run Analysis",
            key="basic_analytics_button",
        ):

            try:

                if operation == "Summary":

                    result = calculate_summary(
                        dataframe,
                        column,
                    )

                elif operation == "Total":

                    result = calculate_total(
                        dataframe,
                        column,
                    )

                elif operation == "Average":

                    result = calculate_average(
                        dataframe,
                        column,
                    )

                elif operation == "Minimum":

                    result = calculate_minimum(
                        dataframe,
                        column,
                    )

                else:

                    result = calculate_maximum(
                        dataframe,
                        column,
                    )

                display_analytics_result(
                    result
                )

            except AnalyticsError as exc:

                st.error(
                    str(exc)
                )

    # --------------------------------------------------------
    # Group By
    # --------------------------------------------------------

    elif operation == "Group By":

        if not numeric_columns:

            st.warning(
                "No numeric columns are available for aggregation."
            )

            return

        group_column = st.selectbox(
            "Group by column",
            all_columns,
            key="group_column",
        )

        value_column = st.selectbox(
            "Value column",
            numeric_columns,
            key="group_value_column",
        )

        aggregation = st.selectbox(
            "Aggregation",
            [
                "sum",
                "mean",
                "min",
                "max",
                "count",
            ],
        )

        if st.button(
            "Run Group Analysis",
            key="group_analytics_button",
        ):

            try:

                result = group_by_aggregation(
                    dataframe,
                    group_column,
                    value_column,
                    aggregation,
                )

                display_analytics_result(
                    result
                )

            except AnalyticsError as exc:

                st.error(
                    str(exc)
                )

    # --------------------------------------------------------
    # Top / Bottom N
    # --------------------------------------------------------

    elif operation == "Top / Bottom N":

        if not numeric_columns:

            st.warning(
                "No numeric columns are available."
            )

            return

        column = st.selectbox(
            "Select numeric column",
            numeric_columns,
            key="top_n_column",
        )

        direction = st.radio(
            "Ranking",
            [
                "Top",
                "Bottom",
            ],
            horizontal=True,
        )

        n = st.number_input(
            "Number of records",
            min_value=1,
            max_value=100,
            value=5,
            step=1,
        )

        if st.button(
            "Run Ranking",
            key="top_n_button",
        ):

            try:

                result = top_n(
                    dataframe,
                    column,
                    n=int(n),
                    ascending=(
                        direction == "Bottom"
                    ),
                )

                display_analytics_result(
                    result
                )

            except AnalyticsError as exc:

                st.error(
                    str(exc)
                )

    # --------------------------------------------------------
    # Correlation
    # --------------------------------------------------------

    elif operation == "Correlation":

        if len(numeric_columns) < 2:

            st.warning(
                "At least two numeric columns are required "
                "for correlation analysis."
            )

            return

        column_a = st.selectbox(
            "First numeric column",
            numeric_columns,
            key="correlation_a",
        )

        column_b = st.selectbox(
            "Second numeric column",
            numeric_columns,
            key="correlation_b",
        )

        if st.button(
            "Calculate Correlation",
            key="correlation_button",
        ):

            try:

                result = calculate_correlation(
                    dataframe,
                    column_a,
                    column_b,
                )

                display_analytics_result(
                    result
                )

            except AnalyticsError as exc:

                st.error(
                    str(exc)
                )


# ============================================================
# Natural Language Investigation
# ============================================================

def display_investigation_dashboard():

    dataframe = st.session_state.dataframe

    if dataframe is None:
        return

    st.subheader("🤖 Ask DataLens")

    st.caption(
        "Ask a natural-language question about your dataset. "
        "Gemini plans the analysis, while Pandas performs "
        "the actual calculation."
    )

    question = st.text_input(
        "Ask a question",
        placeholder=(
            "Example: What is the total Sales by City?"
        ),
        key="investigation_question",
    )

    if st.button(
        "Investigate",
        key="investigation_button",
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

            return

        try:

            # ------------------------------------------------
            # Step 1: Generate Intent
            # ------------------------------------------------

            with st.spinner(
                "Understanding your question..."
            ):

                intent = generate_intent(
                    question,
                    dataframe,
                )

            # ------------------------------------------------
            # Step 2: Display Analysis Plan
            # ------------------------------------------------

            st.markdown(
                "### 🧠 Analysis Plan"
            )

            st.write(
                f"**Operation:** `{intent.operation}`"
            )

            if intent.column:

                st.write(
                    f"**Column:** `{intent.column}`"
                )

            if intent.group_column:

                st.write(
                    f"**Group Column:** `{intent.group_column}`"
                )

            if intent.value_column:

                st.write(
                    f"**Value Column:** `{intent.value_column}`"
                )

            if intent.aggregation:

                st.write(
                    f"**Aggregation:** `{intent.aggregation}`"
                )

            if intent.n:

                st.write(
                    f"**N:** `{intent.n}`"
                )

            if intent.column_a:

                st.write(
                    f"**Column A:** `{intent.column_a}`"
                )

            if intent.column_b:

                st.write(
                    f"**Column B:** `{intent.column_b}`"
                )

            # ------------------------------------------------
            # Step 3: Execute Deterministic Analysis
            # ------------------------------------------------

            with st.spinner(
                "Analyzing your data..."
            ):

                result = execute_intent(
                    intent,
                    dataframe,
                )

            # ------------------------------------------------
            # Step 4: Display Result
            # ------------------------------------------------

            st.markdown(
                "### 📊 Result"
            )

            display_analytics_result(
                result
            )

            # ------------------------------------------------
            # Step 5: Generate Analysis Chart
            # ------------------------------------------------

            chart = generate_analysis_chart(
                result,
                intent,
            )

            if chart is not None:
                st.markdown(
                    "### 📈 Visualization"
                )

                st.plotly_chart(
                    chart["figure"],
                    use_container_width=True,
                )

            # ------------------------------------------------
            # Step 6: Generate Explanation
            # ------------------------------------------------

            with st.spinner(
                "Generating insights..."
            ):

                explanation = generate_explanation(
                    question,
                    intent,
                    result,
                )

            st.markdown(
                "### 💡 DataLens Insight"
            )

            st.info(
                explanation
            )

        except AgentError as exc:

            st.error(
                str(exc)
            )

        except Exception as exc:

            st.error(
                f"Unexpected investigation error: {exc}"
            )


# ============================================================
# Main Application
# ============================================================

def main():

    initialize_session_state()

    display_header()

    st.divider()

    display_upload_section()

    if st.session_state.dataframe is None:

        st.info(
            "Upload a dataset to start the investigation."
        )

        return

    st.divider()

    display_dataset_overview()

    st.divider()

    display_dataset_preview()

    st.divider()

    display_cleaning_dashboard()

    st.divider()

    display_profiling_dashboard()

    st.divider()

    display_quality_dashboard()

    st.divider()

    display_visualization_dashboard()

    st.divider()

    display_analytics_dashboard()

    st.divider()

    display_investigation_dashboard()


# ============================================================
# Application Entry Point
# ============================================================

if __name__ == "__main__":

    main()