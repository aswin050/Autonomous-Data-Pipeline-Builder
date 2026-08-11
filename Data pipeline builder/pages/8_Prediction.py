# =====================================================
# pages/7_Prediction.py
# =====================================================

import streamlit as st
import pandas as pd
import joblib
import os


from ML_agent.predictor import Predictor



# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Prediction",
    page_icon="🎯",
    layout="wide"
)



# =====================================================
# TITLE
# =====================================================

st.title(
    "🎯 Model Prediction"
)


st.write(
    "Upload new unseen data and generate predictions "
    "using the trained ML model."
)


st.divider()



# =====================================================
# CHECK MODEL
# =====================================================


MODEL_PATH = (
    "models/best_model.pkl"
)



if not os.path.exists(MODEL_PATH):


    st.warning(
        "⚠ No trained model found. "
        "Please complete Model Training first."
    )


    st.stop()



# =====================================================
# LOAD MODEL
# =====================================================


model = joblib.load(
    MODEL_PATH
)


predictor = Predictor()



st.success(
    "✅ Trained model loaded successfully"
)



st.divider()



# =====================================================
# MODEL INFORMATION
# =====================================================


if os.path.exists(
    "models/model_metadata.pkl"
):


    metadata = joblib.load(
        "models/model_metadata.pkl"
    )


    st.subheader(
        "🤖 Model Information"
    )


    st.json(
        metadata
    )



st.divider()



# =====================================================
# UPLOAD NEW DATA
# =====================================================


uploaded_file = st.file_uploader(
    "Upload Prediction Dataset",
    type=[
        "csv",
        "xlsx",
        "json"
    ]
)



if uploaded_file is None:

    st.info(
        "Upload a dataset to continue."
    )

    st.stop()



# =====================================================
# READ FILE
# =====================================================


if uploaded_file.name.endswith(".csv"):


    df = pd.read_csv(
        uploaded_file
    )


elif uploaded_file.name.endswith(".xlsx"):


    df = pd.read_excel(
        uploaded_file
    )


elif uploaded_file.name.endswith(".json"):


    df = pd.read_json(
        uploaded_file
    )



st.subheader(
    "📂 Prediction Dataset"
)


st.write(
    "Rows:",
    df.shape[0]
)


st.write(
    "Columns:",
    df.shape[1]
)



st.dataframe(
    df.head(),
    use_container_width=True
)



st.divider()



# =====================================================
# PREDICT
# =====================================================


if st.button(
    "🚀 Generate Prediction"
):


    with st.spinner(
        "Running prediction..."
    ):


        try:


            prediction = predictor.predict(
                model,
                df
            )



            result = df.copy()



            result["Prediction"] = prediction



            st.success(
                "✅ Prediction completed"
            )



            st.subheader(
                "🎯 Prediction Results"
            )



            st.dataframe(
                result,
                use_container_width=True
            )



            csv = result.to_csv(
                index=False
            )



            st.download_button(

                label="📥 Download Predictions",

                data=csv,

                file_name="predictions.csv",

                mime="text/csv"

            )



        except Exception as e:


            st.error(
                "Prediction Failed"
            )


            st.exception(e)