from __future__ import annotations

import streamlit as st

from app.data_processor import (
    DatasetValidationError,
    get_dataset_summary,
    load_dataset,
)


def initialize_session_state() -> None:
    """Initialize values stored in Streamlit session state."""

    if "dataframe" not in st.session_state:
        st.session_state.dataframe = None

    if "file_name" not in st.session_state:
        st.session_state.file_name = None


def display_dataset_overview() -> None:
    """Display basic information about the currently loaded dataset."""

    dataframe = st.session_state.dataframe

    if dataframe is None:
        return

    summary = get_dataset_summary(dataframe)

    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", f"{summary['rows']:,}")

    with col2:
        st.metric("Columns", f"{summary['columns']:,}")

    with col3:
        st.metric("File", st.session_state.file_name)

    st.subheader("Dataset Preview")

    st.dataframe(
        dataframe.head(10),
        width="stretch",
        hide_index=True,
    )


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
        st.divider()
        display_dataset_overview()
    else:
        st.info(
            "Upload a dataset to begin your investigation."
        )


if __name__ == "__main__":
    main()