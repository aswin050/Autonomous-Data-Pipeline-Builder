# =====================================================
# pages/3_Cleaning.py
# =====================================================

import streamlit as st

from agents.cleaning_agent import CleaningAgent


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Data Cleaning",
    page_icon="🧹",
    layout="wide"
)


# =====================================================
# TITLE
# =====================================================

st.title("🧹 Data Cleaning")

st.write(
    "Automatically clean missing values, duplicates, "
    "inconsistent columns and data quality issues."
)

st.divider()



# =====================================================
# CHECK DATASET FROM UPLOAD PAGE
# =====================================================

if "raw_dataset" not in st.session_state:

    st.warning(
        "⚠ Please upload a dataset from Upload page first."
    )

    st.stop()



df = st.session_state["raw_dataset"]



# =====================================================
# DATASET INFORMATION
# =====================================================

st.subheader("📊 Original Dataset")


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
# RUN CLEANING
# =====================================================

if st.button(
    "🧹 Clean Dataset"
):


    with st.spinner(
        "Cleaning dataset..."
    ):


        cleaner = CleaningAgent()


        cleaned_df, cleaning_report = cleaner.clean(
            df.copy()
        )


        # Store output for next pages

        st.session_state[
            "cleaned_dataset"
        ] = cleaned_df


        st.session_state[
            "cleaning_report"
        ] = cleaning_report



    st.success(
        "✅ Dataset cleaned successfully"
    )



# =====================================================
# DISPLAY CLEANED DATA
# =====================================================

if "cleaned_dataset" in st.session_state:


    cleaned_df = st.session_state[
        "cleaned_dataset"
    ]


    st.divider()


    st.subheader(
        "✨ Cleaned Dataset"
    )



    col1, col2, col3 = st.columns(3)



    col1.metric(
        "Rows",
        cleaned_df.shape[0]
    )


    col2.metric(
        "Columns",
        cleaned_df.shape[1]
    )


    col3.metric(
        "Missing Values",
        int(
            cleaned_df.isnull()
            .sum()
            .sum()
        )
    )



    st.dataframe(
        cleaned_df.head(),
        use_container_width=True
    )



# =====================================================
# CLEANING REPORT
# =====================================================

if "cleaning_report" in st.session_state:


    st.divider()


    st.subheader(
        "📋 Cleaning Report"
    )


    st.json(
        st.session_state[
            "cleaning_report"
        ]
    )