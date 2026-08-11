import os
import pandas as pd


# =====================================================
# VALIDATOR
# =====================================================

from validators.smart_validator import SmartDatasetValidator


# =====================================================
# AGENTS
# =====================================================

from agents.dataset_loader_agent import DatasetLoaderAgent
from agents.analyze_agent import DatasetAnalysisAgent
from agents.cleaning_agent import CleaningAgent
from agents.feature_engineering_agent import (
    AdvancedFeatureEngineeringAgent
)

from ML_agent.training_agent import TrainingAgent

from agents.visualization_agent import VisualizationAgent
from agents.pdf_report_agent import PDFReportAgent


# =====================================================
# CREATE REQUIRED DIRECTORIES
# =====================================================

REQUIRED_FOLDERS = [
    "output",
    "output/plots",
    "models",
    "logs"
]

for folder in REQUIRED_FOLDERS:

    os.makedirs(
        folder,
        exist_ok=True
    )


# =====================================================
# MAIN AUTONOMOUS PIPELINE
# =====================================================

def main(dataset_path):

    print("\n")
    print("=" * 70)
    print("        AUTONOMOUS DATA PIPELINE BUILDER")
    print("=" * 70)

    # =================================================
    # RESULT CONTAINER
    # =================================================

    pipeline_result = {

        "success": False,

        "dataset_path": dataset_path,

        "dataset": None,

        "cleaned_dataset": None,

        "feature_dataset": None,

        "target": None,

        "target_column": None,

        "validator_report": {},

        "loader_report": {},

        "analysis_report": {},

        "cleaning_report": {},

        "feature_report": {},

        "model_report": {},

        "best_result": {},

        "best_model": None,

        "plots": {},

        "report_path": None

    }


    # =================================================
    # 1. SMART DATASET VALIDATOR
    # =================================================

    print("\n")
    print("1. SMART DATASET VALIDATOR")
    print("=" * 70)

    try:

        validator = SmartDatasetValidator()

        valid_path, validator_report = validator.validate(
            dataset_path
        )

        pipeline_result["validator_report"] = (
            validator_report
        )

        if valid_path is None:

            print("❌ Dataset rejected")

            pipeline_result["error"] = (
                "Dataset rejected by Smart Dataset Validator."
            )

            return pipeline_result

    except Exception as e:

        print(
            f"❌ Validator failed: {e}"
        )

        pipeline_result["error"] = str(e)

        return pipeline_result


    # =================================================
    # 2. DATASET LOADER
    # =================================================

    print("\n")
    print("2. DATASET LOADER AGENT")
    print("=" * 70)

    try:

        loader = DatasetLoaderAgent()

        df = loader.load(
            valid_path
        )

        if df is None:

            print("❌ Dataset loading failed")

            pipeline_result["error"] = (
                "Dataset loading failed."
            )

            return pipeline_result

        pipeline_result["dataset"] = df.copy()

        pipeline_result["loader_report"] = (
            loader.get_report()
        )

        print(
            "Dataset Shape:",
            df.shape
        )

    except Exception as e:

        print(
            f"❌ Loader failed: {e}"
        )

        pipeline_result["error"] = str(e)

        return pipeline_result


    # =================================================
    # 3. DATASET ANALYSIS
    # =================================================

    print("\n")
    print("3. DATASET ANALYSIS AGENT")
    print("=" * 70)

    try:

        analyzer = DatasetAnalysisAgent()

        analysis_report = analyzer.analyze(
            df
        )

        pipeline_result["analysis_report"] = (
            analysis_report
        )

        print(
            "✔ Dataset analysis completed"
        )

    except Exception as e:

        print(
            f"⚠ Analysis failed: {e}"
        )

        pipeline_result["analysis_report"] = {
            "error": str(e)
        }


    # =================================================
    # 4. DATA CLEANING
    # =================================================

    print("\n")
    print("4. DATA CLEANING AGENT")
    print("=" * 70)

    try:

        cleaner = CleaningAgent()

        cleaned_df, cleaning_report = cleaner.clean(
            df.copy()
        )

        pipeline_result["cleaned_dataset"] = (
            cleaned_df.copy()
        )

        pipeline_result["cleaning_report"] = (
            cleaning_report
        )

        # Save cleaned dataset

        cleaned_path = (
            "output/cleaned_dataset.csv"
        )

        cleaned_df.to_csv(
            cleaned_path,
            index=False
        )

        print(
            "✔ Cleaned dataset saved:",
            cleaned_path
        )

    except Exception as e:

        print(
            f"❌ Cleaning failed: {e}"
        )

        pipeline_result["error"] = str(e)

        return pipeline_result


    # =================================================
    # 5. FEATURE ENGINEERING
    # =================================================

    print("\n")
    print("5. FEATURE ENGINEERING AGENT")
    print("=" * 70)

    try:

        feature_agent = (
            AdvancedFeatureEngineeringAgent()
        )

        X, y, feature_report, target_column = (
            feature_agent.process(
                cleaned_df.copy()
            )
        )

        if X is None:

            print(
                "❌ Feature engineering failed"
            )

            pipeline_result["error"] = (
                "Feature engineering returned no features."
            )

            return pipeline_result

        pipeline_result["feature_dataset"] = (
            X.copy()
        )

        pipeline_result["target"] = (
            y.copy()
            if y is not None
            else None
        )

        pipeline_result["target_column"] = (
            target_column
        )

        pipeline_result["feature_report"] = (
            feature_report
        )

        print(
            "Feature Shape:",
            X.shape
        )

        print(
            "Target Column:",
            target_column
        )

        # Save feature dataset

        feature_df = X.copy()

        if y is not None:

            feature_df[
                "__target__"
            ] = y.values

        feature_path = (
            "output/feature_dataset.csv"
        )

        feature_df.to_csv(
            feature_path,
            index=False
        )

        print(
            "✔ Feature dataset saved:",
            feature_path
        )

    except Exception as e:

        print(
            f"❌ Feature engineering failed: {e}"
        )

        pipeline_result["error"] = str(e)

        return pipeline_result


    # =================================================
    # 6. AUTOML TRAINING
    # =================================================

    print("\n")
    print("6. AUTO ML TRAINING AGENT")
    print("=" * 70)

    if y is not None and target_column is not None:

        try:

            trainer = TrainingAgent()

            (
                best_model,
                model_report,
                best_result
            ) = trainer.train(
                X.copy(),
                y.copy(),
                target_column
            )

            pipeline_result["best_model"] = (
                best_model
            )

            pipeline_result["model_report"] = (
                model_report
            )

            pipeline_result["best_result"] = (
                best_result
            )

            print(
                "\n✔ AutoML training completed"
            )

            print(
                "Best Model:",
                type(best_model).__name__
                if best_model is not None
                else "None"
            )

            print(
                "Best Result:",
                best_result
            )

        except Exception as e:

            print(
                f"⚠ Model training failed: {e}"
            )

            pipeline_result["training_error"] = (
                str(e)
            )

    else:

        print(
            "⚠ Target not detected."
        )

        print(
            "⚠ Model training skipped."
        )


    # =================================================
    # 7. VISUALIZATION
    # =================================================

    print("\n")
    print("7. VISUALIZATION AGENT")
    print("=" * 70)

    try:

        visualizer = VisualizationAgent()

        plots = {}

        # ---------------------------------------------
        # Dataset Overview
        # ---------------------------------------------

        plots["Data Type Distribution"] = (
            visualizer.dataset_overview(
                cleaned_df
            )
        )

        # ---------------------------------------------
        # Missing Values
        # ---------------------------------------------

        plots["Missing Values"] = (
            visualizer.missing_values(
                cleaned_df
            )
        )

        # ---------------------------------------------
        # Correlation
        # ---------------------------------------------

        plots["Correlation Heatmap"] = (
            visualizer.correlation_heatmap(
                cleaned_df
            )
        )

        # ---------------------------------------------
        # Target Distribution
        # ---------------------------------------------

        if y is not None:

            plots["Target Distribution"] = (
                visualizer.target_distribution(
                    y
                )
            )

            plots["Target Pie Chart"] = (
                visualizer.target_pie_chart(
                    y
                )
            )

        # ---------------------------------------------
        # Model Comparison
        # ---------------------------------------------

        if pipeline_result["model_report"]:

            plots["Model Comparison"] = (
                visualizer.model_comparison(
                    pipeline_result[
                        "model_report"
                    ]
                )
            )

        pipeline_result["plots"] = plots

        print(
            "✔ Visualization completed"
        )

    except Exception as e:

        print(
            f"⚠ Visualization failed: {e}"
        )

        pipeline_result["visualization_error"] = (
            str(e)
        )


    # =================================================
    # 8. PDF REPORT
    # =================================================

    print("\n")
    print("8. PDF REPORT AGENT")
    print("=" * 70)

    try:

        pdf = PDFReportAgent()

        report_path = pdf.generate_report(

            dataset_name=os.path.basename(
                dataset_path
            ),

            validator_report=(
                pipeline_result[
                    "validator_report"
                ]
            ),

            loader_report=(
                pipeline_result[
                    "loader_report"
                ]
            ),

            analysis_report=(
                pipeline_result[
                    "analysis_report"
                ]
            ),

            cleaning_report=(
                pipeline_result[
                    "cleaning_report"
                ]
            ),

            feature_report=(
                pipeline_result[
                    "feature_report"
                ]
            ),

            model_report=(
                pipeline_result[
                    "best_result"
                ]
            )
        )

        pipeline_result["report_path"] = (
            report_path
        )

        print(
            "✔ PDF Generated:",
            report_path
        )

    except Exception as e:

        print(
            f"⚠ PDF generation failed: {e}"
        )

        pipeline_result["pdf_error"] = (
            str(e)
        )


    # =================================================
    # PIPELINE COMPLETED
    # =================================================

    pipeline_result["success"] = True

    print("\n")
    print("=" * 70)
    print("        AUTONOMOUS PIPELINE COMPLETED")
    print("=" * 70)

    print(
        "Dataset:",
        df.shape
    )

    print(
        "Cleaned Dataset:",
        cleaned_df.shape
    )

    print(
        "Feature Dataset:",
        X.shape
    )

    print(
        "Target:",
        target_column
    )

    print(
        "Model:",
        type(
            pipeline_result["best_model"]
        ).__name__
        if pipeline_result["best_model"] is not None
        else "Not trained"
    )

    print(
        "PDF:",
        pipeline_result["report_path"]
    )

    print(
        "=" * 70
    )


    # =================================================
    # RETURN EVERYTHING TO STREAMLIT
    # =================================================

    return pipeline_result


# =====================================================
# DIRECT PYTHON TEST
# =====================================================

if __name__ == "__main__":

    dataset = "datasets/train.csv"

    result = main(
        dataset
    )

    if result.get("success"):

        print(
            "\n✅ Pipeline test completed successfully."
        )

    else:

        print(
            "\n❌ Pipeline failed."
        )

        if "error" in result:

            print(
                "Error:",
                result["error"]
            )
