# 🔎 DataLens — Intelligent Data Investigation Assistant

DataLens is an AI-powered data investigation assistant that helps users
upload, profile, clean, analyze, visualize, and investigate datasets using
natural-language questions.

The application combines **Gemini AI** with deterministic **Pandas/NumPy**
analytics so that the LLM plans the analysis while Python performs the
actual calculations.

---

## 🚀 Features

- Upload CSV and Excel (`.xlsx`) datasets
- Automatic dataset profiling
- Row and column statistics
- Data-type detection
- Missing-value analysis
- Duplicate-row detection
- Numeric statistical summary
- Data-quality checks
- Missing-value and duplicate cleaning
- Automatic visualizations
- Deterministic data analytics
- Summary, total, average, minimum and maximum calculations
- Group-by analysis
- Top-N and Bottom-N analysis
- Correlation analysis
- Natural-language data investigation
- Gemini-powered analysis intent generation
- Pydantic-based structured intent validation
- AI-generated explanations and insights
- User-friendly API quota error handling
- Automated test suite

---

## 🧠 How DataLens Works

DataLens follows a simple and safe architecture:

```text
User
  ↓
Streamlit UI
  ↓
Dataset Upload / Question
  ↓
Gemini Structured Intent
  ↓
Pydantic Validation
  ↓
Deterministic Pandas/NumPy Analytics
  ↓
Result Validation
  ↓
Plotly Visualization
  ↓
Gemini Explanation
  ↓
Final Insight