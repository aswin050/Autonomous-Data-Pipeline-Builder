import pandas as pd
import numpy as np


class DataQualityAgent:

    def __init__(self):

        self.report = {
            "health_score": 100,
            "issues": [],
            "recommendations": []
        }


    def analyze(self, df, analysis_report):

        if not isinstance(df, pd.DataFrame):

            return None


        self.report = {
            "health_score": 100,
            "issues": [],
            "recommendations": []
        }


        # =====================================================
        # 1. MISSING VALUE ANALYSIS
        # =====================================================

        missing_values = analysis_report.get(
            "missing_values",
            pd.Series(dtype=int)
        )


        for column, count in missing_values.items():

            percentage = (
                count / len(df)
            ) * 100


            if percentage >= 50:

                severity = "High"

                deduction = 20

                if pd.api.types.is_numeric_dtype(
                    df[column]
                ):

                    recommendation = (
                        "Consider median imputation "
                        "or investigate whether the column "
                        "should be removed."
                    )

                else:

                    recommendation = (
                        "Consider mode imputation "
                        "or evaluate whether the column "
                        "should be removed."
                    )


                confidence = 95


            elif percentage >= 20:

                severity = "Medium"

                deduction = 10


                if pd.api.types.is_numeric_dtype(
                    df[column]
                ):

                    recommendation = (
                        "Use median imputation "
                        "for the missing values."
                    )

                else:

                    recommendation = (
                        "Use mode imputation "
                        "for the missing values."
                    )


                confidence = 94


            else:

                severity = "Low"

                deduction = 5


                if pd.api.types.is_numeric_dtype(
                    df[column]
                ):

                    recommendation = (
                        "Use median or mean imputation."
                    )

                else:

                    recommendation = (
                        "Use mode imputation."
                    )


                confidence = 97


            self.report["health_score"] -= deduction


            self.report["issues"].append({

                "type": "Missing Values",

                "column": column,

                "count": int(count),

                "percentage": round(
                    percentage,
                    2
                ),

                "severity": severity,

                "recommendation": recommendation,

                "confidence": confidence

            })


        # =====================================================
        # 2. DUPLICATE ANALYSIS
        # =====================================================

        duplicates = analysis_report.get(
            "duplicate_rows",
            0
        )


        if duplicates > 0:

            duplicate_percentage = (
                duplicates / len(df)
            ) * 100


            if duplicate_percentage >= 10:

                severity = "High"

                deduction = 15

            elif duplicate_percentage >= 5:

                severity = "Medium"

                deduction = 10

            else:

                severity = "Low"

                deduction = 5


            self.report["health_score"] -= deduction


            self.report["issues"].append({

                "type": "Duplicate Rows",

                "count": int(duplicates),

                "percentage": round(
                    duplicate_percentage,
                    2
                ),

                "severity": severity,

                "recommendation":
                    "Remove duplicate rows "
                    "before model training.",

                "confidence": 98

            })


        # =====================================================
        # 3. OUTLIER ANALYSIS
        # =====================================================

        numerical_columns = df.select_dtypes(
            include=np.number
        ).columns


        for column in numerical_columns:

            series = df[column].dropna()


            if len(series) < 5:

                continue


            Q1 = series.quantile(0.25)

            Q3 = series.quantile(0.75)

            IQR = Q3 - Q1


            if IQR == 0:

                continue


            lower_bound = Q1 - (
                1.5 * IQR
            )

            upper_bound = Q3 + (
                1.5 * IQR
            )


            outliers = series[
                (series < lower_bound) |
                (series > upper_bound)
            ]


            outlier_count = len(outliers)


            if outlier_count > 0:

                percentage = (
                    outlier_count / len(series)
                ) * 100


                if percentage >= 10:

                    severity = "High"

                    deduction = 10

                elif percentage >= 5:

                    severity = "Medium"

                    deduction = 5

                else:

                    severity = "Low"

                    deduction = 2


                self.report["health_score"] -= deduction


                self.report["issues"].append({

                    "type": "Outliers",

                    "column": column,

                    "count": int(
                        outlier_count
                    ),

                    "percentage": round(
                        percentage,
                        2
                    ),

                    "severity": severity,

                    "recommendation":
                        "Review extreme values. "
                        "Consider transformation, "
                        "capping or removal depending "
                        "on domain context.",

                    "confidence": 88

                })


        # =====================================================
        # 4. DATATYPE ANALYSIS
        # =====================================================

        for column in df.columns:

            unique_types = (
                df[column]
                .dropna()
                .apply(type)
                .nunique()
            )


            if unique_types > 1:

                self.report["health_score"] -= 10


                self.report["issues"].append({

                    "type":
                        "Datatype Inconsistency",

                    "column":
                        column,

                    "severity":
                        "High",

                    "recommendation":
                        "Standardize the column "
                        "datatype before processing.",

                    "confidence":
                        97

                })


        # =====================================================
        # 5. HEALTH SCORE LIMIT
        # =====================================================

        self.report["health_score"] = max(
            0,
            min(
                100,
                self.report["health_score"]
            )
        )


        # =====================================================
        # 6. SUMMARY
        # =====================================================

        self.report[
            "total_issues"
        ] = len(
            self.report["issues"]
        )


        return self.report