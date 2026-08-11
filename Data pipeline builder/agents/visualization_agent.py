import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



class VisualizationAgent:


    def __init__(self):

        self.output_folder = "output/plots"

        os.makedirs(
            self.output_folder,
            exist_ok=True
        )



    # =====================================================
    # Dataset Overview
    # =====================================================

    def dataset_overview(self, df):


        plt.figure(
            figsize=(8,5)
        )


        dtype_counts = (
            df.dtypes
            .value_counts()
        )


        dtype_counts.plot(
            kind="bar"
        )


        plt.title(
            "Data Type Distribution"
        )


        plt.xlabel(
            "Data Types"
        )


        plt.ylabel(
            "Count"
        )


        path = os.path.join(
            self.output_folder,
            "datatype_distribution.png"
        )


        plt.savefig(
            path,
            bbox_inches="tight"
        )


        plt.close()


        return path




    # =====================================================
    # Missing Values
    # =====================================================

    def missing_values(self, df):


        missing = (
            df.isnull()
            .sum()
        )


        missing = missing[
            missing > 0
        ]



        plt.figure(
            figsize=(10,5)
        )



        if len(missing) == 0:


            plt.text(

                0.5,
                0.5,

                "No Missing Values",

                ha="center",

                va="center",

                fontsize=15

            )


            plt.axis("off")



        else:


            missing.plot(
                kind="bar"
            )


            plt.title(
                "Missing Values"
            )


            plt.xlabel(
                "Columns"
            )


            plt.ylabel(
                "Missing Count"
            )



        path = os.path.join(

            self.output_folder,

            "missing_values.png"

        )



        plt.savefig(

            path,

            bbox_inches="tight"

        )


        plt.close()



        return path





    # =====================================================
    # Correlation Heatmap
    # =====================================================

    def correlation_heatmap(self, df):


        numeric_df = df.select_dtypes(

            include="number"

        )


        plt.figure(
            figsize=(10,8)
        )



        if numeric_df.shape[1] < 2:


            plt.text(

                0.5,
                0.5,

                "Not Enough Numeric Features",

                ha="center",

                va="center",

                fontsize=15

            )


            plt.axis("off")



        else:


            sns.heatmap(

                numeric_df.corr(),

                annot=True

            )


            plt.title(

                "Feature Correlation"

            )



        path = os.path.join(

            self.output_folder,

            "correlation.png"

        )



        plt.savefig(

            path,

            bbox_inches="tight"

        )


        plt.close()



        return path





    # =====================================================
    # Target Distribution
    # =====================================================

    def target_distribution(self, y):


        plt.figure(
            figsize=(6,4)
        )



        if y is None:


            plt.text(

                0.5,
                0.5,

                "Target Not Available",

                ha="center",

                va="center"

            )


            plt.axis("off")



        else:


            y.value_counts().plot(

                kind="bar"

            )


            plt.title(

                "Target Distribution"

            )



        path = os.path.join(

            self.output_folder,

            "target_distribution.png"

        )


        plt.savefig(

            path,

            bbox_inches="tight"

        )


        plt.close()



        return path





    # =====================================================
    # Target Pie Chart
    # =====================================================

    def target_pie_chart(self, y):


        plt.figure(

            figsize=(7,7)

        )



        if y is None:


            plt.text(

                0.5,
                0.5,

                "Target Not Available",

                ha="center",

                va="center"

            )


            plt.axis("off")



        else:


            values = y.value_counts()



            plt.pie(

                values.values,

                labels=values.index,

                autopct="%1.1f%%",

                startangle=90

            )


            plt.title(

                "Target Class Distribution"

            )



        path = os.path.join(

            self.output_folder,

            "target_pie_chart.png"

        )



        plt.savefig(

            path,

            bbox_inches="tight"

        )


        plt.close()



        return path





    # =====================================================
    # Model Comparison
    # =====================================================

    def model_comparison(self, model_report):


        plt.figure(

            figsize=(8,5)

        )



        if not model_report:


            plt.text(

                0.5,
                0.5,

                "Model Results Not Available",

                ha="center",

                va="center",

                fontsize=15

            )


            plt.axis("off")



        else:


            models = list(

                model_report.keys()

            )


            scores = list(

                model_report.values()

            )


            plt.bar(

                models,

                scores

            )


            plt.xticks(

                rotation=45,

                ha="right"

            )


            plt.xlabel(

                "Models"

            )


            plt.ylabel(

                "Score"

            )


            plt.title(

                "Model Comparison"

            )



        plt.tight_layout()



        path = os.path.join(

            self.output_folder,

            "model_comparison.png"

        )



        plt.savefig(

            path,

            bbox_inches="tight"

        )


        plt.close()



        return path

    # =====================================================
    # MAIN VISUALIZATION PIPELINE
    # =====================================================

    def generate(
            self,
            df,
            y=None,
            model_report=None
    ):


        print("="*60)
        print("VISUALIZATION AGENT STARTED")
        print("="*60)


        plots = {}


        # Dataset Overview

        plots["Data Type Distribution"] = (
            self.dataset_overview(df)
        )


        # Missing Values

        plots["Missing Values"] = (
            self.missing_values(df)
        )


        # Correlation

        plots["Correlation Heatmap"] = (
            self.correlation_heatmap(df)
        )



        # Target charts

        plots["Target Distribution"] = (
            self.target_distribution(y)
        )


        plots["Target Pie Chart"] = (
            self.target_pie_chart(y)
        )



        # Model comparison after training

        if model_report is not None:

            plots["Model Comparison"] = (
                self.model_comparison(model_report)
            )



        print(
            "Visualization Completed"
        )


        return plots