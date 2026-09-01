import streamlit as st
import pandas as pd


st.title("📂 Dataset Upload")


uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=[
        "csv",
        "xlsx",
        "json"
    ]
)



if uploaded_file:


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


    st.session_state["raw_dataset"] = df


    st.session_state["dataset_name"] = (
        uploaded_file.name
    )


    st.success(
        "Dataset loaded successfully"
    )


    st.dataframe(
        df.head()
    )