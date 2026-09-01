# =====================================================
# pages/2_Analysis.py
# =====================================================

import streamlit as st

from agents.analyze_agent import DatasetAnalysisAgent



# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Analysis",
    page_icon="📊",
    layout="wide"
)



# =====================================================
# TITLE
# =====================================================

st.title(
    "📊 Dataset Analysis"
)


st.write(
    "Automatically analyze dataset quality, "
    "structure and statistical information."
)


st.divider()



# =====================================================
# CHECK UPLOADED DATASET
# =====================================================

if "raw_dataset" not in st.session_state:


    st.warning(
        "⚠ Please upload a dataset from Upload page first."
    )


    st.stop()



df = st.session_state[
    "raw_dataset"
]



# =====================================================
# SHOW INPUT DATA
# =====================================================

st.subheader(
    "📂 Uploaded Dataset"
)



col1, col2, col3 = st.columns(3)



col1.metric(
    "Rows",
    df.shape[0]
)


col2.metric(
    "Columns",
    df.shape[1]
)


col3.metric(
    "Missing Values",
    int(
        df.isnull()
        .sum()
        .sum()
    )
)



st.dataframe(
    df.head(),
    use_container_width=True
)



st.divider()



# =====================================================
# RUN ANALYSIS
# =====================================================

if st.button(
    "🔍 Run Analysis"
):


    with st.spinner(
        "Analyzing dataset..."
    ):


        agent = DatasetAnalysisAgent()


        report = agent.analyze(
            df
        )


        st.session_state[
            "analysis_report"
        ] = report



    st.success(
        "✅ Analysis completed"
    )



# =====================================================
# DISPLAY REPORT
# =====================================================

if "analysis_report" in st.session_state:


    report = st.session_state[
        "analysis_report"
    ]



    # =============================================
    # DATASET OVERVIEW
    # =============================================


    st.subheader(
        "📌 Dataset Overview"
    )



    col1, col2, col3 = st.columns(3)



    col1.metric(
        "Rows",
        report["rows"]
    )


    col2.metric(
        "Columns",
        report["columns"]
    )


    col3.metric(
        "Duplicate Rows",
        report["duplicate_rows"]
    )



    st.divider()



    # =============================================
    # COLUMN NAMES
    # =============================================


    st.subheader(
        "📝 Column Names"
    )


    st.write(
        report["column_names"]
    )



    # =============================================
    # DATA TYPES
    # =============================================


    st.subheader(
        "🔤 Data Types"
    )


    st.dataframe(
        report["data_types"],
        use_container_width=True
    )



    # =============================================
    # MISSING VALUES
    # =============================================


    st.subheader(
        "❌ Missing Values"
    )



    if len(report["missing_values"]) > 0:


        st.dataframe(
            report["missing_values"],
            use_container_width=True
        )


    else:


        st.success(
            "No missing values found"
        )



    # =============================================
    # STATISTICAL SUMMARY
    # =============================================


    st.subheader(
        "📈 Statistical Summary"
    )


    st.dataframe(
        report["summary"],
        use_container_width=True
    )