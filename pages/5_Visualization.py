# =====================================================
# pages/4_Visualization.py
# =====================================================

import streamlit as st
import os

from agents.visualization_agent import VisualizationAgent


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Visualization",
    page_icon="📊",
    layout="wide"
)


# =====================================================
# TITLE
# =====================================================

st.title("📊 Data Visualization Dashboard")

st.write(
    "Visualize dataset patterns, distributions, correlations and data quality insights."
)

st.divider()



# =====================================================
# CHECK DATASET
# =====================================================

if "cleaned_dataset" not in st.session_state:

    st.warning(
        "⚠ Please complete Data Cleaning first."
    )

    st.stop()



df = st.session_state["cleaned_dataset"]



# =====================================================
# LOAD TARGET AND MODEL RESULTS
# =====================================================

target = None


if "target" in st.session_state:

    target = st.session_state["target"]



model_report = None


if "model_report" in st.session_state:

    model_report = st.session_state["model_report"]




# =====================================================
# RUN VISUALIZATION
# =====================================================

if st.button(
    "📊 Generate Visualizations"
):


    with st.spinner(
        "Creating charts..."
    ):


        visualizer = VisualizationAgent()



        output = visualizer.generate(

            df,

            y=target,

            model_report=model_report

        )



        st.session_state[
            "visualization_output"
        ] = output



    st.success(
        "✅ Visualizations generated successfully"
    )




# =====================================================
# DISPLAY VISUALIZATIONS
# =====================================================


if "visualization_output" in st.session_state:


    plots = st.session_state[
        "visualization_output"
    ]



    st.subheader(
        "📈 Generated Charts"
    )



    for title, image_path in plots.items():


        st.markdown(
            f"### {title}"
        )



        if os.path.exists(image_path):


            st.image(

                image_path,

                use_container_width=True

            )


        else:


            st.warning(

                f"{image_path} not found"

            )



        st.divider()



else:


    st.info(
        "Click 'Generate Visualizations' to create charts."
    )




# =====================================================
# QUICK DATA INSIGHTS
# =====================================================


st.subheader(
    "📌 Dataset Information"
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

    int(df.isnull().sum().sum())

)



# =====================================================
# DATA PREVIEW
# =====================================================


st.subheader(
    "Dataset Preview"
)



st.dataframe(

    df.head(),

    use_container_width=True

)