import os

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)


class PDFReportAgent:

    def __init__(self):

        self.styles = getSampleStyleSheet()


    def _section_title(self, text):

        return Paragraph(
            f"<b>{text}</b>",
            self.styles["Heading2"]
        )


    def _dict_table(self, data):

        rows = [["Parameter", "Value"]]

        for k, v in data.items():
            rows.append([str(k), str(v)])

        table = Table(rows)

        table.setStyle(

            TableStyle([

                ("BACKGROUND", (0,0), (-1,0), colors.darkblue),

                ("TEXTCOLOR", (0,0), (-1,0), colors.white),

                ("GRID", (0,0), (-1,-1), 1, colors.black),

                ("BACKGROUND", (0,1), (-1,-1), colors.beige),

                ("BOTTOMPADDING", (0,0), (-1,0), 10),

            ])

        )

        return table


    def generate_report(

            self,

            dataset_name,

            validator_report,

            loader_report,

            analysis_report,

            cleaning_report,

            feature_report,

            model_report,

            output_path="output/Pipeline_Report.pdf"

    ):

        doc = SimpleDocTemplate(output_path)

        story = []


        story.append(
            Paragraph(
                "AUTONOMOUS DATA PIPELINE REPORT",
                self.styles["Title"]
            )
        )

        story.append(Spacer(1,20))


        story.append(
            Paragraph(
                f"<b>Dataset :</b> {dataset_name}",
                self.styles["Normal"]
            )
        )

        story.append(Spacer(1,20))


        sections = [

            ("Validator Report", validator_report),

            ("Loader Report", loader_report),

            ("Analysis Report", analysis_report),

            ("Cleaning Report", cleaning_report),

            ("Feature Engineering", feature_report),

            ("Model Performance", model_report)

        ]


        for title, report in sections:

            story.append(
                self._section_title(title)
            )

            story.append(
                self._dict_table(report)
            )

            story.append(
                Spacer(1,20)
            )


        plots = [

            "output/plots/dataset_overview.png",

            "output/plots/missing_values.png",

            "output/plots/correlation_heatmap.png",

            "output/plots/target_distribution.png"

        ]


        story.append(
            self._section_title("Visualizations")
        )


        for plot in plots:

            if os.path.exists(plot):

                story.append(
                    Image(
                        plot,
                        width=450,
                        height=250
                    )
                )

                story.append(
                    Spacer(1,12)
                )


        doc.build(story)

        print(f"PDF Report Saved : {output_path}")