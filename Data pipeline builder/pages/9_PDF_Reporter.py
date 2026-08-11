# =====================================================
# pages/9_PDF_Report.py
# =====================================================

import streamlit as st
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak
)


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="PDF Report",
    page_icon="📄",
    layout="wide"
)


# =====================================================
# TITLE
# =====================================================

st.title("📄 Final PDF Report")

st.write(
    "Generate a complete automated report containing "
    "dataset analysis, cleaning, feature engineering, "
    "visualizations and model performance."
)

st.divider()


# =====================================================
# CHECK DATASET
# =====================================================

if "dataset" not in st.session_state:

    st.warning(
        "⚠ Please upload and process a dataset first."
    )

    st.stop()


df = st.session_state["dataset"]


# =====================================================
# REPORT INFORMATION
# =====================================================

st.subheader("📊 Report Information")

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


st.divider()


# =====================================================
# GENERATE PDF FUNCTION
# =====================================================

def generate_pdf():

    os.makedirs(
        "output/reports",
        exist_ok=True
    )

    pdf_path = (
        "output/reports/"
        "autonomous_data_pipeline_report.pdf"
    )

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=22,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=15,
        spaceBefore=15,
        spaceAfter=10
    )

    body_style = styles["BodyText"]

    story = []


    # =================================================
    # TITLE
    # =================================================

    story.append(
        Paragraph(
            "AUTONOMOUS DATA PIPELINE REPORT",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Automated Dataset Analysis, "
            "Cleaning, Feature Engineering and ML Training",
            body_style
        )
    )

    story.append(
        Spacer(1, 20)
    )


    # =================================================
    # DATASET OVERVIEW
    # =================================================

    story.append(
        Paragraph(
            "1. Dataset Overview",
            heading_style
        )
    )

    overview_data = [

        ["Property", "Value"],

        ["Rows", str(df.shape[0])],

        ["Columns", str(df.shape[1])],

        [
            "Missing Values",
            str(
                int(
                    df.isnull()
                    .sum()
                    .sum()
                )
            )
        ],

        [
            "Duplicate Rows",
            str(
                int(
                    df.duplicated()
                    .sum()
                )
            )
        ]

    ]


    table = Table(
        overview_data,
        colWidths=[
            2.5 * inch,
            2.5 * inch
        ]
    )


    table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                8
            )

        ])
    )


    story.append(table)

    story.append(
        Spacer(1, 15)
    )


    # =================================================
    # COLUMN INFORMATION
    # =================================================

    story.append(
        Paragraph(
            "2. Dataset Columns",
            heading_style
        )
    )


    column_data = [
        ["Column", "Data Type"]
    ]


    for column in df.columns:

        column_data.append(
            [
                str(column),
                str(df[column].dtype)
            ]
        )


    column_table = Table(
        column_data,
        repeatRows=1
    )


    column_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                5
            )

        ])
    )


    story.append(
        column_table
    )


    # =================================================
    # ANALYSIS REPORT
    # =================================================

    if "analysis_report" in st.session_state:

        report = st.session_state[
            "analysis_report"
        ]

        story.append(
            Paragraph(
                "3. Dataset Analysis",
                heading_style
            )
        )

        story.append(
            Paragraph(
                f"Rows: {report.get('rows', df.shape[0])}",
                body_style
            )
        )

        story.append(
            Paragraph(
                f"Columns: {report.get('columns', df.shape[1])}",
                body_style
            )
        )

        story.append(
            Paragraph(
                f"Duplicate Rows: "
                f"{report.get('duplicate_rows', 0)}",
                body_style
            )
        )


    # =================================================
    # CLEANING REPORT
    # =================================================

    if "cleaning_report" in st.session_state:

        story.append(
            Paragraph(
                "4. Data Cleaning",
                heading_style
            )
        )

        cleaning_report = st.session_state[
            "cleaning_report"
        ]


        if isinstance(
            cleaning_report,
            dict
        ):

            for key, value in cleaning_report.items():

                story.append(
                    Paragraph(
                        f"<b>{key}</b>: {value}",
                        body_style
                    )
                )


    # =================================================
    # FEATURE ENGINEERING
    # =================================================

    if "feature_report" in st.session_state:

        story.append(
            Paragraph(
                "5. Feature Engineering",
                heading_style
            )
        )

        feature_report = st.session_state[
            "feature_report"
        ]


        if isinstance(
            feature_report,
            dict
        ):

            for key, value in feature_report.items():

                story.append(
                    Paragraph(
                        f"<b>{key}</b>: {value}",
                        body_style
                    )
                )


    # =================================================
    # TARGET INFORMATION
    # =================================================

    if "target_column" in st.session_state:

        target_column = st.session_state[
            "target_column"
        ]

        story.append(
            Paragraph(
                "6. Target Variable",
                heading_style
            )
        )

        story.append(
            Paragraph(
                f"Target Column: {target_column}",
                body_style
            )
        )


    # =================================================
    # MODEL PERFORMANCE
    # =================================================

    if "best_result" in st.session_state:

        story.append(
            Paragraph(
                "7. Model Performance",
                heading_style
            )
        )

        result = st.session_state[
            "best_result"
        ]


        if isinstance(
            result,
            dict
        ):

            model_data = [
                ["Metric", "Value"]
            ]


            for key, value in result.items():

                if isinstance(
                    value,
                    (int, float)
                ):

                    model_data.append(
                        [
                            str(key),
                            f"{value:.4f}"
                        ]
                    )

                else:

                    model_data.append(
                        [
                            str(key),
                            str(value)
                        ]
                    )


            model_table = Table(
                model_data,
                repeatRows=1
            )


            model_table.setStyle(
                TableStyle([

                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey
                    ),

                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey
                    ),

                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        7
                    )

                ])
            )


            story.append(
                model_table
            )


    # =================================================
    # MODEL COMPARISON
    # =================================================

    if "model_report" in st.session_state:

        story.append(
            Paragraph(
                "8. Model Comparison",
                heading_style
            )
        )


        model_report = st.session_state[
            "model_report"
        ]


        comparison_data = [
            ["Model", "Score"]
        ]


        for model, score in model_report.items():

            comparison_data.append(
                [
                    str(model),
                    f"{score:.4f}"
                ]
            )


        comparison_table = Table(
            comparison_data,
            repeatRows=1
        )


        comparison_table.setStyle(
            TableStyle([

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),

                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    7
                )

            ])
        )


        story.append(
            comparison_table
        )


    # =================================================
    # VISUALIZATIONS
    # =================================================

    if "visualization_output" in st.session_state:

        story.append(
            PageBreak()
        )

        story.append(
            Paragraph(
                "9. Visualizations",
                heading_style
            )
        )


        plots = st.session_state[
            "visualization_output"
        ]


        for title, image_path in plots.items():

            if os.path.exists(image_path):

                story.append(
                    Paragraph(
                        str(title),
                        styles["Heading3"]
                    )
                )

                story.append(
                    Image(
                        image_path,
                        width=6.5 * inch,
                        height=4 * inch
                    )
                )

                story.append(
                    Spacer(1, 15)
                )


    # =================================================
    # FINAL SUMMARY
    # =================================================

    story.append(
        Paragraph(
            "10. Pipeline Summary",
            heading_style
        )
    )


    story.append(
        Paragraph(
            "The dataset was processed through the "
            "automated data pipeline consisting of "
            "dataset analysis, data cleaning, feature "
            "engineering, visualization and machine "
            "learning model training.",
            body_style
        )
    )


    # =================================================
    # BUILD PDF
    # =================================================

    doc.build(
        story
    )


    return pdf_path


# =====================================================
# GENERATE BUTTON
# =====================================================

if st.button(
    "📄 Generate Final PDF Report"
):

    with st.spinner(
        "Generating comprehensive PDF report..."
    ):

        try:

            pdf_path = generate_pdf()

            st.success(
                "✅ PDF report generated successfully!"
            )

            with open(
                pdf_path,
                "rb"
            ) as file:

                pdf_data = file.read()


            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_data,
                file_name="autonomous_data_pipeline_report.pdf",
                mime="application/pdf"
            )


        except Exception as e:

            st.error(
                f"❌ PDF generation failed: {e}"
            )

