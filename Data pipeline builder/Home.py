# import streamlit as st
# import pandas as pd
# import os

# # =====================================================
# # PAGE CONFIG
# # =====================================================
# st.set_page_config(
#     page_title="Home",
#     page_icon="🏠",
#     layout="wide"
# )

# # =====================================================
# # TITLE
# # =====================================================
# st.title("🤖 Autonomous Data Pipeline Builder")
# st.markdown(
#     "### Welcome! Build complete AI pipelines automatically from your dataset."
# )

# st.write("---")

# # =====================================================
# # PROJECT OVERVIEW
# # =====================================================
# st.subheader("📌 Project Overview")

# st.info(
#     """
# This platform automates the complete machine learning workflow.

# ✅ Smart Dataset Validation

# ✅ Data Analysis

# ✅ Data Cleaning

# ✅ Feature Engineering

# ✅ AutoML Model Training

# ✅ Data Visualization

# ✅ PDF Report Generation
# """
# )

# st.write("---")

# # =====================================================
# # DATASET UPLOAD
# # =====================================================
# st.subheader("📂 Upload Dataset")

# uploaded_file = st.file_uploader(
#     "Upload CSV, Excel or JSON",
#     type=["csv", "xlsx", "json"]
# )

# if uploaded_file is not None:

#     extension = uploaded_file.name.split(".")[-1].lower()

#     if extension == "csv":
#         df = pd.read_csv(uploaded_file)

#     elif extension == "xlsx":
#         df = pd.read_excel(uploaded_file)

#     elif extension == "json":
#         df = pd.read_json(uploaded_file)

#     # Save dataframe globally
#     st.session_state["dataset"] = df
#     st.session_state["dataset_name"] = uploaded_file.name

#     st.success("✅ Dataset uploaded successfully!")

# st.write("---")

# # =====================================================
# # DATASET INFORMATION
# # =====================================================
# st.subheader("📊 Dataset Information")

# if "dataset" in st.session_state:

#     df = st.session_state["dataset"]

#     col1, col2, col3, col4 = st.columns(4)

#     col1.metric("Rows", df.shape[0])
#     col2.metric("Columns", df.shape[1])
#     col3.metric("Missing Values", int(df.isnull().sum().sum()))
#     col4.metric("Duplicate Rows", int(df.duplicated().sum()))

#     st.write("### Preview")

#     st.dataframe(df.head())

# else:

#     st.warning("No dataset uploaded yet.")

# st.write("---")

# # =====================================================
# # PIPELINE WORKFLOW
# # =====================================================
# st.subheader("🚀 Pipeline Workflow")

# st.markdown("""
# 1️⃣ Upload Dataset

# ⬇️

# 2️⃣ Analyze Dataset

# ⬇️

# 3️⃣ Clean Dataset

# ⬇️

# 4️⃣ Feature Engineering

# ⬇️

# 5️⃣ Train Machine Learning Models

# ⬇️

# 6️⃣ Generate Visualizations

# ⬇️

# 7️⃣ Download PDF Report
# """)