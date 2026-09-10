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


            for key, value in result.items():

                st.session_state[key] = value

            st.session_state["pipeline_completed"] = bool(
                result.get("success", False)
            )



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
    # AI QUALITY ADVISOR
    # =================================================

    if "ai_advice" in st.session_state:

        st.subheader(
            "🤖 AI Quality Advisor"
        )

        ai_advice = st.session_state[
            "ai_advice"
        ]

        if isinstance(ai_advice, dict):

            if ai_advice.get("status") == "success":

                st.success(
                    "Gemini AI analysis completed successfully ✅"
                )

                st.markdown(
                    ai_advice.get(
                        "advice",
                        "No AI advice available."
                    )
                )

            else:

                st.warning(
                    ai_advice.get(
                        "message",
                        "AI Quality Advisor failed."
                    )
                )

        else:

            # In case the agent returns plain text
            st.markdown(
                str(ai_advice)
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
# 🤖 DATASET ASSISTANT
# =================================================

if st.session_state.get("pipeline_completed"):

    st.divider()

    st.header("🤖 Dataset Assistant")

    st.write(
        "Ask questions about your dataset using natural language."
    )

    question = st.text_input(
        "💬 Ask a question",
        placeholder=(
            "Example: What is the average Fare by passenger class?"
        ),
        key="dataset_assistant_question"
    )

    if st.button(
        "🚀 Ask Dataset Assistant",
        use_container_width=True
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            try:

                from agents.dataset_assistant_agent import (
                    DatasetAssistantAgent
                )

                dataset = st.session_state.get(
                    "dataset"
                )

                analysis_report = st.session_state.get(
                    "analysis_report",
                    {}
                )

                quality_report = st.session_state.get(
                    "quality_report",
                    {}
                )

                model_report = st.session_state.get(
                    "model_report",
                    {}
                )

                if dataset is None:

                    st.error(
                        "Dataset is not available."
                    )

                else:

                    with st.spinner(
                        "🤖 Dataset Assistant is analyzing..."
                    ):

                        assistant = DatasetAssistantAgent()

                        assistant_result = assistant.ask(
                            question=question,
                            df=dataset,
                            analysis_report=analysis_report,
                            quality_report=quality_report,
                            model_report=model_report
                        )

                    # =========================================
                    # CHECK FOR ERROR
                    # =========================================

                    if assistant_result.get("error"):

                        st.error(
                            f"Dataset Assistant failed: "
                            f"{assistant_result['error']}"
                        )

                    else:

                        # =====================================
                        # SAVE RESULT
                        # =====================================

                        st.session_state[
                            "assistant_result"
                        ] = assistant_result


            except Exception as e:

                st.error(
                    f"Dataset Assistant failed: {e}"
                )


# =================================================
# DISPLAY DATASET ASSISTANT RESULT
# =================================================

if "assistant_result" in st.session_state:

    assistant_result = st.session_state[
        "assistant_result"
    ]

    st.divider()

    st.subheader(
        "💡 Assistant Answer"
    )

    # =============================================
    # ANSWER
    # =============================================

    answer = assistant_result.get(
        "answer"
    )

    if answer:

        # Fix literal \n characters
        answer = answer.replace(
            "\\n",
            "\n"
        )

        # Render Markdown properly
        st.markdown(
            answer
        )

    else:

        st.warning(
            "No answer was generated."
        )


    # =============================================
    # CHART
    # =============================================

    chart_path = assistant_result.get(
        "chart"
    )

    if chart_path:

        # Normalize Windows path
        chart_path = os.path.normpath(
            chart_path
        )

        if os.path.exists(chart_path):

            st.subheader(
                "📊 Visualization"
            )

            st.image(
                chart_path,
                use_container_width=True
            )

        else:

            st.warning(
                f"Chart was generated but could not be found: "
                f"{chart_path}"
            )


    # =============================================
    # OPTIONAL: SHOW ANALYSIS DETAILS
    # =============================================

    with st.expander(
        "🔍 Analysis Details"
    ):

        st.json(
            assistant_result.get(
                "plan",
                {}
            )
        )

        st.json(
            assistant_result.get(
                "result",
                {}
            )
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


    if "report_path" in st.session_state:


        with open(

            st.session_state["report_path"],

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