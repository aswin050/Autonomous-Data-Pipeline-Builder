import streamlit as st

from agents.DataQualityAgent import DataQualityAgent


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Data Quality Advisor",
    page_icon="🤖",
    layout="wide"
)


# =====================================================
# TITLE
# =====================================================

st.title(
    "🤖 AI Data Quality Advisor"
)

st.write(
    "Automatically identify data quality problems "
    "and receive intelligent recommendations."
)

st.divider()


# =====================================================
# CHECK DATASET
# =====================================================

if "raw_dataset" not in st.session_state:

    st.warning(
        "⚠ Please upload a dataset from the Upload page first."
    )

    st.stop()


df = st.session_state["raw_dataset"]


# =====================================================
# CHECK ANALYSIS
# =====================================================

if "analysis_report" not in st.session_state:

    st.info(
        "Please run Dataset Analysis first."
    )

    st.stop()


analysis_report = st.session_state[
    "analysis_report"
]


# =====================================================
# DATASET INFORMATION
# =====================================================

st.subheader(
    "📊 Dataset"
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
    "Duplicate Rows",
    int(df.duplicated().sum())
)


st.divider()


# =====================================================
# RUN QUALITY ANALYSIS
# =====================================================

if st.button(
    "🤖 Analyze Data Quality",
    type="primary"
):

    with st.spinner(
        "AI Data Quality Advisor is analyzing your dataset..."
    ):

        agent = DataQualityAgent()


        quality_report = agent.analyze(
            df,
            analysis_report
        )


        st.session_state[
            "quality_report"
        ] = quality_report


    st.success(
        "✅ Data quality analysis completed."
    )


# =====================================================
# DISPLAY RESULT
# =====================================================

if "quality_report" in st.session_state:

    report = st.session_state[
        "quality_report"
    ]


    # =================================================
    # HEALTH SCORE
    # =================================================

    st.subheader(
        "❤️ Dataset Health Score"
    )


    score = report[
        "health_score"
    ]


    col1, col2 = st.columns(
        [1, 2]
    )


    with col1:

        st.metric(
            "Health Score",
            f"{score}/100"
        )


    with col2:

        st.progress(
            score / 100
        )


    if score >= 90:

        st.success(
            "Excellent dataset quality."
        )

    elif score >= 75:

        st.info(
            "Good dataset quality with some issues."
        )

    elif score >= 50:

        st.warning(
            "Dataset requires cleaning."
        )

    else:

        st.error(
            "Poor dataset quality. "
            "Significant cleaning is recommended."
        )


    st.divider()


    # =================================================
    # ISSUE SUMMARY
    # =================================================

    st.subheader(
        "🔎 Detected Issues"
    )


    total = report[
        "total_issues"
    ]


    if total == 0:

        st.success(
            "🎉 No major data quality problems detected."
        )


    else:

        st.warning(
            f"{total} issue(s) detected."
        )


    # =================================================
    # ISSUE DETAILS
    # =================================================

    for index, issue in enumerate(
        report["issues"],
        start=1
    ):

        issue_type = issue[
            "type"
        ]

        severity = issue[
            "severity"
        ]


        with st.expander(
            f"{index}. {issue_type} — {severity}"
        ):


            if "column" in issue:

                st.write(
                    f"**Column:** "
                    f"{issue['column']}"
                )


            if "count" in issue:

                st.write(
                    f"**Affected Records:** "
                    f"{issue['count']}"
                )


            if "percentage" in issue:

                st.write(
                    f"**Percentage:** "
                    f"{issue['percentage']}%"
                )


            st.write(
                f"**Severity:** "
                f"{severity}"
            )


            st.write(
                "**🤖 Recommendation:**"
            )


            st.info(
                issue["recommendation"]
            )


            st.write(
                f"**Confidence:** "
                f"{issue['confidence']}%"
            )