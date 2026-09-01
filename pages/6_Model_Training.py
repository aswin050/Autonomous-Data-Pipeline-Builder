# =====================================================
# pages/6_Model_Training.py
# =====================================================

import streamlit as st
import pandas as pd

from ML_agent.training_agent import TrainingAgent



# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Model Training",
    page_icon="🤖",
    layout="wide"
)



# =====================================================
# TITLE
# =====================================================

st.title(
    "🤖 AutoML Model Training"
)


st.write(
    "Automatically select, train and evaluate "
    "multiple machine learning algorithms."
)


st.divider()



# =====================================================
# CHECK FEATURE DATA
# =====================================================


required = [

    "feature_dataset",
    "target",
    "target_column"

]


for item in required:


    if item not in st.session_state:


        st.warning(
            f"⚠ Missing {item}. Complete previous stages first."
        )

        st.stop()



# =====================================================
# LOAD DATA
# =====================================================


X = st.session_state[
    "feature_dataset"
]


y = st.session_state[
    "target"
]


target_column = st.session_state[
    "target_column"
]



if isinstance(y, pd.DataFrame):

    y = y.iloc[:,0]



# =====================================================
# TRAINING DATA INFORMATION
# =====================================================


st.subheader(
    "📊 Training Dataset"
)



col1, col2, col3 = st.columns(3)



col1.metric(
    "Samples",
    X.shape[0]
)


col2.metric(
    "Features",
    X.shape[1]
)


col3.metric(
    "Target Classes",
    y.nunique()
)



st.write(
    "Target Column:",
    target_column
)



st.dataframe(
    X.head(),
    use_container_width=True
)



st.divider()



# =====================================================
# START TRAINING
# =====================================================


if st.button(
    "🚀 Start AutoML Training"
):


    with st.spinner(
        "Training multiple models..."
    ):


        trainer = TrainingAgent()



        best_model, model_report, best_result = trainer.train(

            X,

            y,

            target_column

        )



        st.session_state[
            "best_model"
        ] = best_model


        st.session_state[
            "model_report"
        ] = model_report


        st.session_state[
            "best_result"
        ] = best_result



    st.success(
        "✅ Model training completed successfully"
    )



# =====================================================
# RESULTS
# =====================================================


if "best_result" in st.session_state:


    st.divider()


    st.subheader(
        "🏆 Best Model Performance"
    )


    st.json(
        st.session_state[
            "best_result"
        ]
    )



# =====================================================
# MODEL COMPARISON
# =====================================================


if "model_report" in st.session_state:


    st.subheader(
        "📊 Model Comparison"
    )


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