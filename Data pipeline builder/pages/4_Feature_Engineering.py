# =====================================================
# pages/4_Feature_Engineering.py
# =====================================================

import streamlit as st

from agents.feature_engineering_agent import (
    AdvancedFeatureEngineeringAgent
)



# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Feature Engineering",
    page_icon="⚙️",
    layout="wide"
)



# =====================================================
# TITLE
# =====================================================

st.title(
    "⚙️ Feature Engineering"
)


st.write(
    "Automatically transform raw cleaned data into "
    "machine learning ready features."
)


st.divider()



# =====================================================
# CHECK CLEANED DATASET
# =====================================================

if "cleaned_dataset" not in st.session_state:


    st.warning(
        "⚠ Please complete Data Cleaning first."
    )


    st.stop()



df = st.session_state[
    "cleaned_dataset"
]



# =====================================================
# CLEANED DATA INFORMATION
# =====================================================

st.subheader(
    "📊 Cleaned Dataset"
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
# RUN FEATURE ENGINEERING
# =====================================================

if st.button(
    "⚙️ Generate Features"
):


    with st.spinner(
        "Generating features..."
    ):



        feature_agent = (
            AdvancedFeatureEngineeringAgent()
        )



        X, y, feature_report, target_column = (
            feature_agent.process(df)
        )



        # Save outputs

        st.session_state[
            "feature_dataset"
        ] = X



        st.session_state[
            "target"
        ] = y



        st.session_state[
            "target_column"
        ] = target_column



        st.session_state[
            "feature_report"
        ] = feature_report



    st.success(
        "✅ Feature Engineering Completed"
    )



# =====================================================
# DISPLAY FEATURE OUTPUT
# =====================================================

if "feature_dataset" in st.session_state:



    X = st.session_state[
        "feature_dataset"
    ]



    st.divider()



    st.subheader(
        "🚀 Machine Learning Features"
    )



    col1, col2 = st.columns(2)



    col1.metric(
        "Rows",
        X.shape[0]
    )


    col2.metric(
        "Features",
        X.shape[1]
    )



    st.dataframe(
        X.head(),
        use_container_width=True
    )



# =====================================================
# TARGET DISPLAY
# =====================================================

if "target" in st.session_state:



    st.subheader(
        "🎯 Target Variable"
    )



    target_column = st.session_state[
        "target_column"
    ]



    st.write(
        "Target Column:",
        target_column
    )



    st.dataframe(
        st.session_state["target"].head()
    )



# =====================================================
# FEATURE REPORT
# =====================================================

if "feature_report" in st.session_state:



    st.divider()



    st.subheader(
        "📋 Feature Engineering Report"
    )



    st.json(
        st.session_state[
            "feature_report"
        ]
    )