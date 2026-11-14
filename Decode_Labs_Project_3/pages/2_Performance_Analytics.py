
import streamlit as st
import pandas as pd
from pathlib import Path


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="SmartRecommend AI Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Performance Analytics")
st.write(
    "Monitor and evaluate the performance "
    "of the SmartRecommend AI recommendation engine."
)


# ==========================================
# 2. LOAD EVALUATION RESULTS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

REPORT_PATH = BASE_DIR / "evaluation_results.csv"

if not REPORT_PATH.exists():

    st.warning(
        "No evaluation report found. "
        "Run real_evaluation.py first."
    )

    st.stop()

report = pd.read_csv(REPORT_PATH)

if report.empty:

    st.warning("The evaluation report is empty.")

    st.stop()

results = report.iloc[0]


# ==========================================
# 3. DISPLAY EVALUATION METRICS
# ==========================================

st.subheader("Recommendation Performance")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Precision@5",
    f"{results['Precision@K']:.2%}"
)

col2.metric(
    "Recall@5",
    f"{results['Recall@K']:.2%}"
)

col3.metric(
    "F1 Score",
    f"{results['F1 Score']:.2%}"
)

col4, col5 = st.columns(2)

col4.metric(
    "Catalogue Coverage",
    f"{results['Catalogue Coverage']:.2%}"
)

if "RMSE" in report.columns and pd.notna(results["RMSE"]):

    col5.metric(
        "RMSE",
        f"{results['RMSE']:.4f}",
        help="Lower values indicate smaller rating prediction errors."
    )

else:

    col5.metric("RMSE", "Not available")


# ==========================================
# 4. DATASET INFORMATION
# ==========================================

st.divider()

st.subheader("Dataset Overview")

info1, info2, info3 = st.columns(3)

info1.metric(
    "Total Ratings",
    int(results["Total Ratings"])
)

info2.metric(
    "Registered Users",
    int(results["Total Users"])
)

info3.metric(
    "Users Evaluated",
    int(results["Users Evaluated"])
)


# ==========================================
# 5. PERFORMANCE CHART
# ==========================================

st.divider()

st.subheader("Recommendation Quality")

metric_data = pd.DataFrame({
    "Metric": [
        "Precision@K",
        "Recall@K",
        "F1 Score",
        "Catalogue Coverage"
    ],
    "Score": [
        results["Precision@K"],
        results["Recall@K"],
        results["F1 Score"],
        results["Catalogue Coverage"]
    ]
})

st.bar_chart(
    metric_data.set_index("Metric"),
    y="Score",
    y_label="Score",
    height=350
)


# ==========================================
# 6. EVALUATION INTERPRETATION
# ==========================================

st.divider()

st.subheader("Performance Interpretation")

precision = results["Precision@K"]
recall = results["Recall@K"]
coverage = results["Catalogue Coverage"]

if precision >= 0.7:
    st.success("The recommendation precision is strong.")
elif precision >= 0.4:
    st.info("The recommendation precision has room for improvement.")
else:
    st.warning("The recommendation precision is currently low.")

if recall >= 0.7:
    st.success("The system is identifying a good proportion of relevant items.")
else:
    st.info("The system could identify more relevant items.")

if coverage >= 0.7:
    st.success("The catalogue has good recommendation coverage.")
else:
    st.info("The system is recommending a relatively small portion of the catalogue.")


# ==========================================
# 7. REPORT INFORMATION
# ==========================================

st.divider()

st.caption(
    "SmartRecommend AI | Model Evaluation Dashboard"
)

st.caption(
    "Metrics are based on the latest saved real-data "
    "evaluation. Results depend on the available user ratings."
)
