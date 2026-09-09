import streamlit as st


def main() -> None:
    """Run the DataLens Streamlit application."""

    st.set_page_config(
        page_title="DataLens",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📊 DataLens")
    st.subheader("Intelligent Data Investigation Assistant")

    st.write(
        "Upload a CSV or Excel dataset, explore its quality, "
        "visualize important patterns, and ask questions using natural language."
    )

    st.divider()

    st.info(
        "🚀 DataLens is being built step-by-step. "
        "Dataset upload and analysis features will be added in the next phases."
    )

    st.markdown("### Planned capabilities")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🔍 Data Profiling")
        st.write(
            "Understand rows, columns, data types, missing values, "
            "duplicates, and statistics."
        )

    with col2:
        st.markdown("#### 📈 Automatic EDA")
        st.write(
            "Generate useful charts and discover patterns "
            "in the uploaded dataset."
        )

    with col3:
        st.markdown("#### 🤖 Ask DataLens")
        st.write(
            "Ask questions in natural language and get "
            "data-backed analytical answers."
        )


if __name__ == "__main__":
    main()