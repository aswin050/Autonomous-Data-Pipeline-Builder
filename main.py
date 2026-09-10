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
from agents.DataQualityAgent import DataQualityAgent
from agents.ai_quality_advisor_agent import AIQualityAdvisorAgent
from agents.cleaning_agent import CleaningAgent

from agents.feature_engineering_agent import (
    AdvancedFeatureEngineeringAgent
)

from ML_agent.training_agent import TrainingAgent

from agents.dataset_assistant_agent import (
    DatasetAssistantAgent
)

from agents.visualization_agent import VisualizationAgent
from agents.pdf_report_agent import PDFReportAgent


# =====================================================
# REQUIRED DIRECTORIES
# =====================================================

REQUIRED_FOLDERS = [
    "output",
    "output/plots",
    "output/assistant_charts",
    "models",
    "logs"
]

for folder in REQUIRED_FOLDERS:
    os.makedirs(folder, exist_ok=True)


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

        # -------------------------------------------------
        # DATASETS
        # -------------------------------------------------

        "dataset": None,
        "cleaned_dataset": None,
        "feature_dataset": None,

        "cleaned_dataset_path": None,
        "feature_dataset_path": None,

        # -------------------------------------------------
        # TARGET
        # -------------------------------------------------

        "target": None,
        "target_column": None,

        # -------------------------------------------------
        # REPORTS
        # -------------------------------------------------

        "validator_report": {},
        "loader_report": {},
        "analysis_report": {},
        "quality_report": {},
        "ai_advice": {},
        "cleaning_report": {},
        "feature_report": {},
        "model_report": {},
        "best_result": {},

        # -------------------------------------------------
        # MODEL
        # -------------------------------------------------

        "best_model": None,

        # -------------------------------------------------
        # VISUALIZATION
        # -------------------------------------------------

        "plots": {},

        # -------------------------------------------------
        # DATASET ASSISTANT
        # -------------------------------------------------

        "assistant": None,

        # -------------------------------------------------
        # PDF
        # -------------------------------------------------

        "report_path": None,

        # -------------------------------------------------
        # ERRORS
        # -------------------------------------------------

        "errors": {}
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

        print(f"❌ Validator failed: {e}")

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

        df = loader.load(valid_path)

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

        print(f"❌ Loader failed: {e}")

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

        analysis_report = analyzer.analyze(df)

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

        analysis_report = {
            "error": str(e)
        }

        pipeline_result["analysis_report"] = (
            analysis_report
        )

        pipeline_result["errors"]["analysis"] = (
            str(e)
        )

    # =================================================
    # 4. DATA QUALITY + AI QUALITY ADVISOR
    # =================================================

    print("\n")
    print(
        "4. DATA QUALITY ANALYSIS + AI QUALITY ADVISOR"
    )
    print("=" * 70)

    quality_report = {}

    try:

        # -------------------------------------------------
        # DATA QUALITY
        # -------------------------------------------------

        quality_agent = DataQualityAgent()

        quality_report = quality_agent.analyze(
            df,
            analysis_report
        )

        pipeline_result["quality_report"] = (
            quality_report
        )

        print(
            "✔ Data quality analysis completed"
        )

        print(
            "Health Score:",
            quality_report.get(
                "health_score",
                "N/A"
            )
        )

        print(
            "Issues Detected:",
            quality_report.get(
                "total_issues",
                0
            )
        )

        # -------------------------------------------------
        # DISPLAY QUALITY ISSUES
        # -------------------------------------------------

        for issue in quality_report.get(
            "issues",
            []
        ):

            print(
                f"\n⚠ {issue.get('type', 'Unknown Issue')}"
            )

            if "column" in issue:
                print(
                    "Column:",
                    issue["column"]
                )

            if "count" in issue:
                print(
                    "Affected Records:",
                    issue["count"]
                )

            if "percentage" in issue:
                print(
                    "Percentage:",
                    issue["percentage"],
                    "%"
                )

            print(
                "Severity:",
                issue.get(
                    "severity",
                    "Unknown"
                )
            )

            print(
                "Recommendation:",
                issue.get(
                    "recommendation",
                    "N/A"
                )
            )

            if "confidence" in issue:
                print(
                    "Confidence:",
                    issue["confidence"],
                    "%"
                )

    except Exception as e:

        print(
            f"⚠ Data Quality failed: {e}"
        )

        pipeline_result["errors"]["quality"] = (
            str(e)
        )

    # -------------------------------------------------
    # AI QUALITY ADVISOR
    # -------------------------------------------------

    try:

        print("\n")
        print("🤖 AI QUALITY ADVISOR")
        print("-" * 70)

        advisor = AIQualityAdvisorAgent()

        ai_advice = advisor.generate_advice(
            quality_report
        )

        pipeline_result["ai_advice"] = (
            ai_advice
        )

        if ai_advice.get("status") == "success":

            print(
                "✔ AI quality advice generated"
            )

            print(
                ai_advice.get(
                    "advice",
                    "No advice generated."
                )
            )

        else:

            print(
                "⚠ AI advice unavailable:"
            )

            print(
                ai_advice.get(
                    "message",
                    "Unknown error."
                )
            )

    except Exception as e:

        print(
            f"⚠ AI Quality Advisor failed: {e}"
        )

        pipeline_result["errors"]["ai_advisor"] = (
            str(e)
        )

    # =================================================
    # 5. DATA CLEANING
    # =================================================

    print("\n")
    print("5. DATA CLEANING AGENT")
    print("=" * 70)

    try:

        cleaner = CleaningAgent()

        cleaned_df, cleaning_report = (
            cleaner.clean(
                df.copy()
            )
        )

        pipeline_result["cleaned_dataset"] = (
            cleaned_df.copy()
        )

        pipeline_result["cleaning_report"] = (
            cleaning_report
        )

        # -------------------------------------------------
        # SAVE CLEANED DATASET
        # -------------------------------------------------

        cleaned_path = (
            "output/cleaned_dataset.csv"
        )

        cleaned_df.to_csv(
            cleaned_path,
            index=False
        )

        pipeline_result[
            "cleaned_dataset_path"
        ] = cleaned_path

        print(
            "✔ Cleaned dataset saved:",
            cleaned_path
        )

    except Exception as e:

        print(
            f"❌ Cleaning failed: {e}"
        )

        pipeline_result["errors"]["cleaning"] = (
            str(e)
        )

        pipeline_result["error"] = str(e)

        return pipeline_result

    # =================================================
    # 6. FEATURE ENGINEERING
    # =================================================

    print("\n")
    print("6. FEATURE ENGINEERING AGENT")
    print("=" * 70)

    X = None
    y = None
    target_column = None

    try:

        feature_agent = (
            AdvancedFeatureEngineeringAgent()
        )

        (
            X,
            y,
            feature_report,
            target_column
        ) = feature_agent.process(
            cleaned_df.copy()
        )

        if X is None:

            print(
                "⚠ Feature engineering returned no features."
            )

            pipeline_result["errors"][
                "feature_engineering"
            ] = (
                "No feature dataset returned."
            )

        else:

            pipeline_result[
                "feature_dataset"
            ] = X.copy()

            pipeline_result[
                "target"
            ] = (
                y.copy()
                if y is not None
                else None
            )

            pipeline_result[
                "target_column"
            ] = target_column

            pipeline_result[
                "feature_report"
            ] = feature_report

            print(
                "Feature Shape:",
                X.shape
            )

            print(
                "Target Column:",
                target_column
            )

            # -------------------------------------------------
            # SAVE FEATURE DATASET
            # -------------------------------------------------

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

            pipeline_result[
                "feature_dataset_path"
            ] = feature_path

            print(
                "✔ Feature dataset saved:",
                feature_path
            )

    except Exception as e:

        print(
            f"⚠ Feature engineering failed: {e}"
        )

        pipeline_result["errors"][
            "feature_engineering"
        ] = str(e)

    # =================================================
    # 7. AUTOML TRAINING
    # =================================================

    print("\n")
    print("7. AUTO ML TRAINING AGENT")
    print("=" * 70)

    if (
        X is not None
        and y is not None
        and target_column is not None
    ):

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

            pipeline_result[
                "best_model"
            ] = best_model

            pipeline_result[
                "model_report"
            ] = model_report

            pipeline_result[
                "best_result"
            ] = best_result

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

            pipeline_result["errors"][
                "training"
            ] = str(e)

    else:

        print(
            "⚠ No valid target detected."
        )

        print(
            "⚠ Model training skipped."
        )

    # =================================================
    # 8. DATASET ASSISTANT
    # =================================================

    print("\n")
    print("8. DATASET ASSISTANT AGENT")
    print("=" * 70)

    try:

        assistant = DatasetAssistantAgent()

        pipeline_result[
            "assistant"
        ] = True

        print(
            "✔ Dataset Assistant initialized"
        )

        print(
            "✔ Natural-language dataset analysis is ready"
        )

    except Exception as e:

        print(
            f"⚠ Dataset Assistant initialization failed: {e}"
        )

        pipeline_result["errors"][
            "assistant"
        ] = str(e)

    # =================================================
    # 9. VISUALIZATION
    # =================================================

    print("\n")
    print("9. VISUALIZATION AGENT")
    print("=" * 70)

    try:

        visualizer = VisualizationAgent()

        plots = {}

        # -------------------------------------------------
        # DATA TYPE DISTRIBUTION
        # -------------------------------------------------

        try:

            plots[
                "Data Type Distribution"
            ] = visualizer.dataset_overview(
                cleaned_df
            )

        except Exception as e:

            print(
                f"⚠ Data type visualization failed: {e}"
            )

        # -------------------------------------------------
        # MISSING VALUES DONUT
        # -------------------------------------------------

        try:

            plots[
                "Missing Values"
            ] = visualizer.missing_values(
                cleaned_df
            )

        except Exception as e:

            print(
                f"⚠ Missing-value visualization failed: {e}"
            )

        # -------------------------------------------------
        # CORRELATION HEATMAP
        # -------------------------------------------------

        try:

            plots[
                "Correlation Heatmap"
            ] = visualizer.correlation_heatmap(
                cleaned_df
            )

        except Exception as e:

            print(
                f"⚠ Correlation visualization failed: {e}"
            )

        # -------------------------------------------------
        # TARGET DISTRIBUTION
        # -------------------------------------------------

        if y is not None:

            try:

                plots[
                    "Target Distribution"
                ] = visualizer.target_distribution(
                    y
                )

            except Exception as e:

                print(
                    f"⚠ Target distribution failed: {e}"
                )

            try:

                plots[
                    "Target Pie Chart"
                ] = visualizer.target_pie_chart(
                    y
                )

            except Exception as e:

                print(
                    f"⚠ Target pie chart failed: {e}"
                )

        # -------------------------------------------------
        # MODEL COMPARISON
        # -------------------------------------------------

        if pipeline_result[
            "model_report"
        ]:

            try:

                plots[
                    "Model Comparison"
                ] = visualizer.model_comparison(
                    pipeline_result[
                        "model_report"
                    ]
                )

            except Exception as e:

                print(
                    f"⚠ Model comparison failed: {e}"
                )

        pipeline_result[
            "plots"
        ] = plots

        print(
            "✔ Visualization completed"
        )

    except Exception as e:

        print(
            f"⚠ Visualization failed: {e}"
        )

        pipeline_result[
            "errors"
        ]["visualization"] = str(e)

    # =================================================
    # 10. PDF REPORT
    # =================================================

    print("\n")
    print("10. PDF REPORT AGENT")
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

            # IMPORTANT:
            # Pass the complete model report
            model_report=(
                pipeline_result[
                    "model_report"
                ]
            )
        )

        pipeline_result[
            "report_path"
        ] = report_path

        print(
            "✔ PDF Generated:",
            report_path
        )

    except Exception as e:

        print(
            f"⚠ PDF generation failed: {e}"
        )

        pipeline_result[
            "errors"
        ]["pdf"] = str(e)

    # =================================================
    # PIPELINE COMPLETED
    # =================================================

    pipeline_result[
        "success"
    ] = True

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

    if X is not None:

        print(
            "Feature Dataset:",
            X.shape
        )

    else:

        print(
            "Feature Dataset:",
            "Not available"
        )

    print(
        "Target:",
        target_column
        if target_column
        else "Not detected"
    )

    print(
        "Data Quality Score:",
        pipeline_result[
            "quality_report"
        ].get(
            "health_score",
            "N/A"
        )
    )

    print(
        "Quality Issues:",
        pipeline_result[
            "quality_report"
        ].get(
            "total_issues",
            0
        )
    )

    print(
        "Model:",
        type(
            pipeline_result[
                "best_model"
            ]
        ).__name__
        if pipeline_result[
            "best_model"
        ] is not None
        else "Not trained"
    )

    print(
        "Dataset Assistant:",
        "Ready"
        if pipeline_result.get(
            "assistant"
        )
        else "Unavailable"
    )

    print(
        "PDF:",
        pipeline_result[
            "report_path"
        ]
    )

    if pipeline_result["errors"]:

        print(
            "\n⚠ Non-critical errors:"
        )

        for name, error in pipeline_result[
            "errors"
        ].items():

            print(
                f"   {name}: {error}"
            )

    print(
        "=" * 70
    )

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