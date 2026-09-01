# =====================================================
# pages/8_Model_Evaluation.py
# =====================================================

import streamlit as st
import pandas as pd
import joblib
import os



# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📊",
    layout="wide"
)



# =====================================================
# TITLE
# =====================================================

st.title(
    "📊 Model Evaluation Dashboard"
)


st.write(
    "Analyze trained model performance, "
    "comparison and model information."
)


st.divider()



# =====================================================
# CHECK TRAINING
# =====================================================

if (

    "best_result" not in st.session_state

    and

    not os.path.exists(
        "models/model_metadata.pkl"
    )

):


    st.warning(
        "⚠ Please train a model first."
    )


    st.stop()



# =====================================================
# BEST MODEL METRICS
# =====================================================


st.subheader(
    "🏆 Best Model Performance"
)



if "best_result" in st.session_state:


    result = st.session_state[
        "best_result"
    ]


else:


    result = {}



# =====================================================
# SEPARATE NUMERIC AND NON NUMERIC RESULTS
# =====================================================


numeric_metrics = {}

other_results = {}



for key, value in result.items():


    if isinstance(
        value,
        (int, float)
    ):

        numeric_metrics[key] = value


    else:

        other_results[key] = value



# =====================================================
# DISPLAY NUMERIC METRICS
# =====================================================


if numeric_metrics:


    cols = st.columns(
        len(numeric_metrics)
    )


    for i, (key,value) in enumerate(
        numeric_metrics.items()
    ):


        if isinstance(
            value,
            float
        ):

            value = round(
                value,
                4
            )


        cols[i].metric(
            key.replace("_"," ").title(),
            value
        )



else:


    st.info(
        "Metrics not available."
    )



# =====================================================
# DISPLAY OTHER RESULTS
# =====================================================


for key,value in other_results.items():


    st.divider()


    st.subheader(
        key.replace("_"," ").title()
    )



    # Confusion Matrix

    if key == "confusion_matrix":


        matrix_df = pd.DataFrame(
            value
        )


        st.dataframe(
            matrix_df,
            use_container_width=True
        )


    else:


        st.write(
            value
        )



st.divider()



# =====================================================
# MODEL COMPARISON
# =====================================================


st.subheader(
    "🤖 Model Comparison"
)



if "model_report" in st.session_state:


    report = st.session_state[
        "model_report"
    ]



    comparison = pd.DataFrame(

        list(report.items()),

        columns=[
            "Model",
            "Score"
        ]

    )



    comparison = comparison.sort_values(

        by="Score",

        ascending=False

    )



    st.dataframe(

        comparison,

        use_container_width=True

    )



    st.bar_chart(

        comparison.set_index(
            "Model"
        )

    )



else:


    st.info(
        "Model comparison data not available."
    )



st.divider()



# =====================================================
# MODEL METADATA
# =====================================================


st.subheader(
    "📁 Model Metadata"
)



metadata_path = (
    "models/model_metadata.pkl"
)



if os.path.exists(
    metadata_path
):


    metadata = joblib.load(
        metadata_path
    )


    st.json(
        metadata
    )


else:


    st.info(
        "Metadata file not available."
    )



st.divider()



# =====================================================
# SAVED FILE STATUS
# =====================================================


st.subheader(
    "💾 Saved Model Status"
)



files = {

    "Best Model":
        "models/best_model.pkl",

    "Feature Transformer":
        "models/feature_transformer.pkl",

    "Feature Columns":
        "models/feature_columns.pkl",

    "Metadata":
        "models/model_metadata.pkl"

}



for name,path in files.items():


    if os.path.exists(path):

        st.success(
            f"✅ {name} available"
        )

    else:

        st.warning(
            f"⚠ {name} missing"
        )