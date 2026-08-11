# =====================================================
# App.py
# =====================================================

import os
import pandas as pd
import streamlit as st
import base64

from main import main



# =====================================================
# IMAGE CONVERTER
# =====================================================

def image_to_base64(path):

    with open(path, "rb") as img:

        return base64.b64encode(
            img.read()
        ).decode()



# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(

    page_title="Autonomous Data Pipeline Builder",

    page_icon="🤖",

    layout="wide"

)



# =====================================================
# TITLE
# =====================================================

st.title(
    "🤖 Autonomous Data Pipeline Builder"
)


st.write(
"""
Upload a dataset and let the AI pipeline automatically:

✅ Validate Dataset

✅ Analyze Data Quality

✅ Clean Data

✅ Engineer Features

✅ Train ML Models

✅ Generate Visualizations

✅ Create PDF Report
"""
)


st.divider()



# =====================================================
# DATASET UPLOAD
# =====================================================

uploaded_file = st.file_uploader(

    "📂 Upload Dataset",

    type=[
        "csv",
        "xlsx",
        "json",
        "parquet"
    ]

)



if uploaded_file:


    os.makedirs(
        "datasets",
        exist_ok=True
    )


    dataset_path = os.path.join(

        "datasets",

        uploaded_file.name

    )



    with open(

        dataset_path,

        "wb"

    ) as f:


        f.write(

            uploaded_file.getbuffer()

        )



    st.success(
        "Dataset Uploaded Successfully ✅"
    )



    # ===============================================
    # RUN PIPELINE
    # ===============================================


    if st.button(

        "🚀 Run Autonomous Pipeline",

        use_container_width=True

    ):



        progress = st.progress(0)

        status = st.empty()



        try:


            status.info(
                "🔍 Validator Agent Running..."
            )


            progress.progress(10)



            result = main(

                dataset_path

            )



            progress.progress(100)



            status.success(

                "Pipeline Completed Successfully 🎉"

            )



            # =========================================
            # SAVE RESULTS TO SESSION STATE
            # =========================================


            for key,value in result.items():

                st.session_state[key] = value



            st.session_state["pipeline_completed"] = True



        except Exception as e:


            progress.progress(0)


            status.error(
                "Pipeline Failed ❌"
            )


            st.exception(e)




# =====================================================
# DISPLAY RESULTS AFTER PIPELINE
# =====================================================


if st.session_state.get(
    "pipeline_completed"
):


    st.divider()


    st.header(
        "📋 Pipeline Results"
    )



    # =================================================
    # CLEANED DATASET
    # =================================================


    if "cleaned_dataset" in st.session_state:


        cleaned_df = st.session_state[
            "cleaned_dataset"
        ]


        st.subheader(
            "🧹 Cleaned Dataset"
        )


        st.dataframe(

            cleaned_df.head(),

            use_container_width=True

        )



        c1,c2,c3,c4 = st.columns(4)



        c1.metric(

            "Rows",

            cleaned_df.shape[0]

        )


        c2.metric(

            "Columns",

            cleaned_df.shape[1]

        )


        c3.metric(

            "Missing Values",

            int(
                cleaned_df.isnull()
                .sum()
                .sum()
            )

        )


        c4.metric(

            "Duplicates",

            int(
                cleaned_df.duplicated()
                .sum()
            )

        )



    # =================================================
    # FEATURE DATASET
    # =================================================


    if "feature_dataset" in st.session_state:


        st.subheader(

            "⚙ Feature Engineering Output"

        )


        feature_df = st.session_state[
            "feature_dataset"
        ]



        st.write(

            "Feature Shape:",

            feature_df.shape

        )


        st.dataframe(

            feature_df.head(),

            use_container_width=True

        )



    # =================================================
    # MODEL RESULT
    # =================================================


    if "model_report" in st.session_state:


        st.subheader(

            "🏆 Model Performance"

        )



        model_df = pd.DataFrame(

            st.session_state[
                "model_report"
            ].items(),

            columns=[

                "Model",

                "Score"

            ]

        )



        st.dataframe(

            model_df,

            use_container_width=True

        )


        st.bar_chart(

            model_df.set_index(
                "Model"
            )

        )




    # =================================================
    # VISUALIZATION DASHBOARD
    # =================================================


    st.subheader(

        "📊 Visualization Dashboard"

    )



    html_path = (

        "dashboard/charts_layout.html"

    )


    plot_folder = (

        "output/plots"

    )



    if os.path.exists(html_path):


        with open(

            html_path,

            "r",

            encoding="utf-8"

        ) as file:


            html=file.read()



        charts=[


            "datatype_distribution.png",

            "missing_values.png",

            "correlation.png",

            "target_distribution.png",

            "target_pie_chart.png",

            "model_comparison.png"

        ]



        for chart in charts:



            chart_path=os.path.join(

                plot_folder,

                chart

            )



            if os.path.exists(chart_path):


                encoded=image_to_base64(

                    chart_path

                )


                html=html.replace(

                    f'src="{chart}"',

                    f'src="data:image/png;base64,{encoded}"'

                )



        st.components.v1.html(

            html,

            height=1400,

            scrolling=True

        )


    else:


        st.warning(

            "Visualization dashboard not generated"

        )





    # =================================================
    # DOWNLOAD SECTION
    # =================================================


    st.subheader(

        "⬇ Downloads"

    )



    d1,d2,d3 = st.columns(3)



    # Cleaned dataset


    if "cleaned_dataset" in st.session_state:


        csv = st.session_state[
            "cleaned_dataset"
        ].to_csv(

            index=False

        )


        d1.download_button(

            "📄 Cleaned Dataset",

            csv,

            "cleaned_dataset.csv",

            "text/csv"

        )



    # Feature dataset


    if "feature_dataset" in st.session_state:


        csv = st.session_state[
            "feature_dataset"
        ].to_csv(

            index=False

        )


        d2.download_button(

            "⚙ Feature Dataset",

            csv,

            "feature_dataset.csv",

            "text/csv"

        )



    # PDF


    if "report" in st.session_state:


        with open(

            st.session_state["report"],

            "rb"

        ) as f:


            d3.download_button(

                "📑 PDF Report",

                f,

                "pipeline_report.pdf"

            )



    st.success(

        "🤖 Autonomous Data Pipeline Completed"

    )