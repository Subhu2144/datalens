from __future__ import annotations

import streamlit as st

from app.data_processor import (
    DatasetValidationError,
    get_dataset_summary,
    load_dataset,
    profile_dataset,
)


def initialize_session_state() -> None:
    """Initialize values stored in Streamlit session state."""

    if "dataframe" not in st.session_state:
        st.session_state.dataframe = None

    if "file_name" not in st.session_state:
        st.session_state.file_name = None


def display_upload_section() -> None:
    """Display the dataset upload interface."""

    st.subheader("Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel dataset",
        type=["csv", "xlsx"],
        help="Supported formats: CSV and Excel (.xlsx)",
    )

    if uploaded_file is None:
        return

    try:
        dataframe = load_dataset(
            file_name=uploaded_file.name,
            file_bytes=uploaded_file.getvalue(),
        )

        st.session_state.dataframe = dataframe
        st.session_state.file_name = uploaded_file.name

        st.success(
            f"Successfully loaded **{uploaded_file.name}**."
        )

    except DatasetValidationError as exc:
        st.session_state.dataframe = None
        st.session_state.file_name = None

        st.error(str(exc))

    except Exception:
        st.session_state.dataframe = None
        st.session_state.file_name = None

        st.error(
            "Something went wrong while processing the dataset. "
            "Please check the file and try again."
        )


def display_dataset_overview() -> None:
    """Display the high-level dataset overview."""

    dataframe = st.session_state.dataframe

    if dataframe is None:
        return

    summary = get_dataset_summary(dataframe)

    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rows",
            f"{summary['rows']:,}",
        )

    with col2:
        st.metric(
            "Columns",
            f"{summary['columns']:,}",
        )

    with col3:
        st.metric(
            "File",
            st.session_state.file_name,
        )


def display_dataset_preview() -> None:
    """Display a preview of the uploaded dataset."""

    dataframe = st.session_state.dataframe

    if dataframe is None:
        return

    st.subheader("Dataset Preview")

    st.dataframe(
        dataframe.head(10),
        width="stretch",
        hide_index=True,
    )


def display_profile_summary(profile: dict) -> None:
    """Display dataset-level profiling metrics."""

    summary = profile["summary"]

    st.subheader("Data Profile")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Cells",
            f"{summary['total_cells']:,}",
        )

    with col2:
        st.metric(
            "Missing Cells",
            f"{summary['missing_cells']:,}",
        )

    with col3:
        st.metric(
            "Missing %",
            f"{summary['missing_percentage']:.2f}%",
        )

    with col4:
        st.metric(
            "Duplicate Rows",
            f"{summary['duplicate_rows']:,}",
        )


def display_column_information(profile: dict) -> None:
    """Display column-level information."""

    st.subheader("Column Information")

    column_details = profile["column_details"]

    if not column_details:
        st.info("No column information is available.")
        return

    st.dataframe(
        column_details,
        width="stretch",
        hide_index=True,
    )


def display_missing_values(profile: dict) -> None:
    """Display columns containing missing values."""

    st.subheader("Missing Values")

    missing_values = profile["missing_values"]

    if not missing_values:
        st.success("No missing values detected.")
        return

    st.dataframe(
        missing_values,
        width="stretch",
        hide_index=True,
    )


def display_numeric_statistics(profile: dict) -> None:
    """Display statistics for numeric columns."""

    st.subheader("Numeric Statistics")

    numeric_statistics = profile["numeric_statistics"]

    if not numeric_statistics:
        st.info("No numeric columns were detected.")
        return

    statistics_rows = []

    for column, statistics in numeric_statistics.items():
        statistics_rows.append(
            {
                "Column": column,
                "Count": statistics["count"],
                "Mean": statistics["mean"],
                "Median": statistics["median"],
                "Std": statistics["std"],
                "Min": statistics["min"],
                "Max": statistics["max"],
            }
        )

    st.dataframe(
        statistics_rows,
        width="stretch",
        hide_index=True,
    )


def display_profiling_dashboard() -> None:
    """Generate and display the complete profiling dashboard."""

    dataframe = st.session_state.dataframe

    if dataframe is None:
        return

    try:
        profile = profile_dataset(dataframe)

        st.divider()

        display_profile_summary(profile)

        st.divider()

        display_column_information(profile)

        st.divider()

        display_missing_values(profile)

        st.divider()

        display_numeric_statistics(profile)

    except DatasetValidationError as exc:
        st.error(
            f"Unable to profile the dataset: {exc}"
        )

    except Exception:
        st.error(
            "Something went wrong while generating the data profile."
        )


def main() -> None:
    """Run the DataLens Streamlit application."""

    st.set_page_config(
        page_title="DataLens",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_session_state()

    st.title("📊 DataLens")
    st.subheader("Intelligent Data Investigation Assistant")

    st.write(
        "Upload a CSV or Excel dataset, explore its quality, "
        "visualize important patterns, and ask questions using "
        "natural language."
    )

    st.divider()

    display_upload_section()

    if st.session_state.dataframe is not None:
        display_dataset_overview()
        display_dataset_preview()
        display_profiling_dashboard()

    else:
        st.info(
            "Upload a dataset to begin your investigation."
        )


if __name__ == "__main__":
    main()