import os
import json
import re
import hashlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from google import genai


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# DATASET ASSISTANT AGENT
# ============================================================

class DatasetAssistantAgent:

    def __init__(self):

        # ----------------------------------------------------
        # GEMINI API KEY
        # ----------------------------------------------------

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set."
            )

        # ----------------------------------------------------
        # GEMINI CLIENT
        # ----------------------------------------------------

        self.client = genai.Client(
            api_key=api_key
        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        self.model = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.6-flash"
        )

        # ----------------------------------------------------
        # ALLOWED OPERATIONS
        # ----------------------------------------------------

        self.allowed_operations = [

            "dataset_info",

            "missing_values",

            "duplicate_count",

            "column_mean",

            "column_median",

            "column_min",

            "column_max",

            "unique_count",

            "value_counts",

            "groupby_mean",

            "groupby_count",

            "correlation",

            "distribution",

            "groupby_rate"
        ]

        # ----------------------------------------------------
        # CHART DIRECTORY
        # ----------------------------------------------------

        self.chart_folder = (
            "output/assistant_charts"
        )

        os.makedirs(
            self.chart_folder,
            exist_ok=True
        )


    # ========================================================
    # CREATE SAFE CHART FILENAME
    # ========================================================

    def _chart_filename(
        self,
        operation,
        question=""
    ):

        text = (
            operation
            + "_"
            + question
        )

        hash_value = hashlib.md5(
            text.encode("utf-8")
        ).hexdigest()[:8]

        return os.path.join(
            self.chart_folder,
            f"{operation}_{hash_value}.png"
        )


    # ========================================================
    # GEMINI QUESTION PLANNER
    # ========================================================

    def create_analysis_plan(
        self,
        df,
        question
    ):

        # ----------------------------------------------------
        # DATASET VALIDATION
        # ----------------------------------------------------

        if df is None:

            raise ValueError(
                "Dataset is None."
            )

        if not isinstance(df, pd.DataFrame):

            raise TypeError(
                "df must be a pandas DataFrame."
            )

        if df.empty:

            raise ValueError(
                "The dataset is empty."
            )

        # ----------------------------------------------------
        # DATASET METADATA
        # ----------------------------------------------------

        columns = [
            str(column)
            for column in df.columns
        ]

        column_types = {

            str(column):
            str(df[column].dtype)

            for column in df.columns
        }

        numeric_columns = [

            str(column)

            for column in df.columns

            if pd.api.types.is_numeric_dtype(
                df[column]
            )
        ]

        categorical_columns = [

            str(column)

            for column in df.columns

            if (
                df[column].dtype == "object"
                or
                str(df[column].dtype).startswith(
                    "category"
                )
                or
                pd.api.types.is_bool_dtype(
                    df[column]
                )
            )
        ]

        # ----------------------------------------------------
        # PROMPT
        # ----------------------------------------------------

        prompt = f"""
You are a dataset analysis planning assistant.

Your job is to understand the user's question
and convert it into ONE safe analysis operation.

Python will perform the actual calculation.

====================================================
DATASET INFORMATION
====================================================

Rows:
{len(df)}

Columns:
{columns}

Column Types:
{json.dumps(column_types, indent=2)}

Numeric Columns:
{numeric_columns}

Categorical Columns:
{categorical_columns}

====================================================
USER QUESTION
====================================================

{question}

====================================================
AVAILABLE OPERATIONS
====================================================

1. dataset_info
2. missing_values
3. duplicate_count
4. column_mean
5. column_median
6. column_min
7. column_max
8. unique_count
9. value_counts
10. groupby_mean
11. groupby_count
12. correlation
13. distribution
14. groupby_rate

====================================================
RETURN FORMAT
====================================================

Return ONLY valid JSON.

{{
    "operation": "operation_name",
    "column": "column_name_or_null",
    "group_by": "column_name_or_null",
    "reason": "short explanation"
}}

====================================================
RULES
====================================================

- operation MUST be one of the available operations.
- Never invent a column name.
- column must exactly match an existing dataset column.
- group_by must exactly match an existing dataset column.
- Do not calculate anything yourself.
- Python will perform the actual calculation.

Use:

dataset_info
→ rows, columns, data types, dataset structure,
  dataset size, overview.

missing_values
→ missing/null value questions.

duplicate_count
→ duplicate record questions.

column_mean
→ average of one numeric column.

column_median
→ median of one numeric column.

column_min
→ minimum of one numeric column.

column_max
→ maximum of one numeric column.

unique_count
→ number of unique values in a column.

value_counts
→ frequency/count of values or categories.

groupby_mean
→ average numeric value grouped by another column.

groupby_count
→ number of records grouped by another column.

groupby_rate
→ rate/proportion grouped by another column.

correlation
→ relationships/correlation between numeric columns.

distribution
→ statistical distribution of one numeric column.

====================================================
IMPORTANT COLUMN RULES
====================================================

For numeric statistics:

column MUST be numeric.

For correlation:

no specific column is required.

For dataset_info:

column and group_by should be null.

For missing_values:

column and group_by should normally be null.

For duplicate_count:

column and group_by should normally be null.

For groupby operations:

column is the value being measured.

group_by is the grouping/category column.

If the user asks a general question such as:

"Which category appears most?"

use value_counts.

If the user asks:

"What is the average salary by experience?"

use groupby_mean.

If the user asks:

"What is the survival rate by gender?"

use groupby_rate.

Return ONLY JSON.
"""

        # ----------------------------------------------------
        # GEMINI REQUEST
        # ----------------------------------------------------

        response = self.client.models.generate_content(

            model=self.model,

            contents=prompt
        )

        # ----------------------------------------------------
        # RESPONSE TEXT
        # ----------------------------------------------------

        text = response.text.strip()

        # ----------------------------------------------------
        # REMOVE MARKDOWN FENCES
        # ----------------------------------------------------

        text = re.sub(
            r"```json",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"```",
            "",
            text
        )

        text = text.strip()

        # ----------------------------------------------------
        # EXTRACT JSON
        # ----------------------------------------------------

        match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL
        )

        if not match:

            raise ValueError(
                "Gemini did not return a valid JSON plan."
            )

        try:

            plan = json.loads(
                match.group()
            )

        except json.JSONDecodeError as e:

            raise ValueError(
                f"Invalid JSON returned by Gemini: {e}"
            )

        # ----------------------------------------------------
        # VALIDATE OPERATION
        # ----------------------------------------------------

        operation = plan.get(
            "operation"
        )

        if operation not in self.allowed_operations:

            raise ValueError(
                f"Unsupported operation: {operation}"
            )

        # ----------------------------------------------------
        # NORMALIZE NULL VALUES
        # ----------------------------------------------------

        column = plan.get(
            "column"
        )

        group_by = plan.get(
            "group_by"
        )

        if column in [
            "",
            "null",
            "None",
            "none"
        ]:

            plan["column"] = None

        if group_by in [
            "",
            "null",
            "None",
            "none"
        ]:

            plan["group_by"] = None

        return plan


    # ========================================================
    # VALIDATE COLUMN
    # ========================================================

    def validate_column(
        self,
        df,
        column
    ):

        if column is None:

            raise ValueError(
                "A column was not specified."
            )

        if column not in df.columns:

            raise ValueError(
                f"Column '{column}' does not exist "
                f"in the dataset."
            )


    # ========================================================
    # VALIDATE NUMERIC COLUMN
    # ========================================================

    def validate_numeric_column(
        self,
        df,
        column
    ):

        self.validate_column(
            df,
            column
        )

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):

            raise ValueError(
                f"Column '{column}' is not numeric."
            )


    # ========================================================
    # EXECUTE ANALYSIS
    # ========================================================

    def execute_operation(
        self,
        df,
        plan
    ):

        operation = plan.get(
            "operation"
        )

        column = plan.get(
            "column"
        )

        group_by = plan.get(
            "group_by"
        )

        # ====================================================
        # DATASET INFORMATION
        # ====================================================

        if operation == "dataset_info":

            return {

                "rows":
                    int(df.shape[0]),

                "columns":
                    int(df.shape[1]),

                "column_names":
                    [
                        str(col)
                        for col in df.columns
                    ],

                "data_types": {

                    str(col):
                    str(df[col].dtype)

                    for col in df.columns
                },

                "missing_values":
                    int(
                        df.isnull()
                        .sum()
                        .sum()
                    ),

                "duplicate_rows":
                    int(
                        df.duplicated()
                        .sum()
                    )
            }

        # ====================================================
        # MISSING VALUES
        # ====================================================

        if operation == "missing_values":

            missing = (
                df.isnull()
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            result = {

                str(col):
                int(count)

                for col, count
                in missing.items()

                if count > 0
            }

            return {

                "missing_values":
                    result,

                "total_missing":
                    int(
                        df.isnull()
                        .sum()
                        .sum()
                    )
            }

        # ====================================================
        # DUPLICATES
        # ====================================================

        if operation == "duplicate_count":

            return {

                "duplicate_rows":
                    int(
                        df.duplicated()
                        .sum()
                    )
            }

        # ====================================================
        # COLUMN VALIDATION
        # ====================================================

        if operation in [

            "column_mean",
            "column_median",
            "column_min",
            "column_max",
            "unique_count",
            "value_counts",
            "distribution"
        ]:

            self.validate_column(
                df,
                column
            )

        # ====================================================
        # MEAN
        # ====================================================

        if operation == "column_mean":

            self.validate_numeric_column(
                df,
                column
            )

            value = df[column].mean()

            return {

                "column":
                    column,

                "mean":
                    round(
                        float(value),
                        6
                    )
            }

        # ====================================================
        # MEDIAN
        # ====================================================

        if operation == "column_median":

            self.validate_numeric_column(
                df,
                column
            )

            value = df[column].median()

            return {

                "column":
                    column,

                "median":
                    round(
                        float(value),
                        6
                    )
            }

        # ====================================================
        # MIN
        # ====================================================

        if operation == "column_min":

            self.validate_numeric_column(
                df,
                column
            )

            value = df[column].min()

            return {

                "column":
                    column,

                "minimum":
                    float(value)
                    if pd.notna(value)
                    else None
            }

        # ====================================================
        # MAX
        # ====================================================

        if operation == "column_max":

            self.validate_numeric_column(
                df,
                column
            )

            value = df[column].max()

            return {

                "column":
                    column,

                "maximum":
                    float(value)
                    if pd.notna(value)
                    else None
            }

        # ====================================================
        # UNIQUE COUNT
        # ====================================================

        if operation == "unique_count":

            return {

                "column":
                    column,

                "unique_values":
                    int(
                        df[column]
                        .nunique(
                            dropna=True
                        )
                    )
            }

        # ====================================================
        # VALUE COUNTS
        # ====================================================

        if operation == "value_counts":

            counts = (
                df[column]
                .value_counts(
                    dropna=False
                )
                .head(20)
            )

            return {

                "column":
                    column,

                "value_counts": {

                    str(index):
                    int(value)

                    for index, value
                    in counts.items()
                }
            }

        # ====================================================
        # GROUPBY VALIDATION
        # ====================================================

        if operation in [

            "groupby_mean",
            "groupby_count",
            "groupby_rate"
        ]:

            self.validate_column(
                df,
                group_by
            )

            self.validate_column(
                df,
                column
            )

        # ====================================================
        # GROUPBY MEAN
        # ====================================================

        if operation == "groupby_mean":

            self.validate_numeric_column(
                df,
                column
            )

            grouped = (
                df.groupby(
                    group_by,
                    dropna=False
                )[column]
                .mean()
                .sort_values(
                    ascending=False
                )
            )

            return {

                "group_by":
                    group_by,

                "column":
                    column,

                "mean": {

                    str(k):
                    round(
                        float(v),
                        4
                    )

                    for k, v
                    in grouped.items()

                    if pd.notna(v)
                }
            }

        # ====================================================
        # GROUPBY COUNT
        # ====================================================

        if operation == "groupby_count":

            grouped = (
                df.groupby(
                    group_by,
                    dropna=False
                )[column]
                .count()
                .sort_values(
                    ascending=False
                )
            )

            return {

                "group_by":
                    group_by,

                "column":
                    column,

                "count": {

                    str(k):
                    int(v)

                    for k, v
                    in grouped.items()
                }
            }

        # ====================================================
        # GROUPBY RATE
        # ====================================================

        if operation == "groupby_rate":

            self.validate_numeric_column(
                df,
                column
            )

            grouped = (
                df.groupby(
                    group_by,
                    dropna=False
                )[column]
                .mean()
                .sort_values(
                    ascending=False
                )
            )

            return {

                "group_by":
                    group_by,

                "column":
                    column,

                "rate": {

                    str(k):
                    round(
                        float(v),
                        6
                    )

                    for k, v
                    in grouped.items()

                    if pd.notna(v)
                }
            }

        # ====================================================
        # CORRELATION
        # ====================================================

        if operation == "correlation":

            numeric_df = (
                df.select_dtypes(
                    include=np.number
                )
            )

            if numeric_df.shape[1] < 2:

                raise ValueError(
                    "At least two numerical columns "
                    "are required for correlation analysis."
                )

            correlation = (
                numeric_df.corr()
                .round(3)
            )

            return {

                "columns":
                    [
                        str(col)
                        for col
                        in correlation.columns
                    ],

                "correlation":
                    correlation.to_dict()
            }

        # ====================================================
        # DISTRIBUTION
        # ====================================================

        if operation == "distribution":

            self.validate_numeric_column(
                df,
                column
            )

            series = (
                df[column]
                .dropna()
            )

            if len(series) == 0:

                raise ValueError(
                    f"Column '{column}' contains "
                    "no valid numeric values."
                )

            return {

                "column":
                    column,

                "count":
                    int(
                        len(series)
                    ),

                "mean":
                    round(
                        float(
                            series.mean()
                        ),
                        4
                    ),

                "median":
                    round(
                        float(
                            series.median()
                        ),
                        4
                    ),

                "std":
                    round(
                        float(
                            series.std()
                        ),
                        4
                    ),

                "min":
                    float(
                        series.min()
                    ),

                "max":
                    float(
                        series.max()
                    )
            }

        raise ValueError(
            f"Unknown operation: {operation}"
        )


    # ========================================================
    # CREATE CHART
    # ========================================================

    def create_chart(
        self,
        df,
        plan,
        result,
        question=""
    ):

        operation = plan.get(
            "operation"
        )

        column = plan.get(
            "column"
        )

        group_by = plan.get(
            "group_by"
        )

        # ====================================================
        # MISSING VALUES — DONUT CHART
        # ====================================================

        if operation == "missing_values":

            total_cells = (
                df.shape[0]
                *
                df.shape[1]
            )

            total_missing = int(
                df.isnull()
                .sum()
                .sum()
            )

            # ------------------------------------------------
            # EMPTY DATASET
            # ------------------------------------------------

            if total_cells == 0:

                return None

            total_complete = (
                total_cells
                -
                total_missing
            )

            # ------------------------------------------------
            # DONUT VALUES
            # ------------------------------------------------

            values = [

                total_complete,

                total_missing
            ]

            labels = [

                "Complete",

                "Missing"
            ]

            # ------------------------------------------------
            # CREATE FIGURE
            # ------------------------------------------------

            fig, ax = plt.subplots(
                figsize=(8, 8)
            )

            # ------------------------------------------------
            # DONUT
            # ------------------------------------------------

            ax.pie(

                values,

                labels=labels,

                autopct="%1.1f%%",

                startangle=90,

                wedgeprops={
                    "width": 0.40,
                    "edgecolor": "white"
                },

                pctdistance=0.78
            )

            # ------------------------------------------------
            # CENTER TEXT
            # ------------------------------------------------

            complete_percentage = (
                100
                *
                total_complete
                /
                total_cells
            )

            ax.text(

                0,

                0,

                f"{complete_percentage:.1f}%\nComplete",

                ha="center",

                va="center",

                fontsize=14,

                fontweight="bold"
            )

            ax.set_title(
                "Dataset Completeness",
                fontsize=16,
                fontweight="bold"
            )

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            plt.tight_layout()

            path = self._chart_filename(
                "missing_values",
                question
            )

            plt.savefig(
                path,
                dpi=160,
                bbox_inches="tight"
            )

            plt.close(fig)

            return path

        # ====================================================
        # VALUE COUNTS
        # ====================================================

        if operation == "value_counts":

            counts = (
                df[column]
                .value_counts(
                    dropna=False
                )
                .head(20)
            )

            if counts.empty:

                return None

            # ------------------------------------------------
            # LIMIT LONG CATEGORY LABELS
            # ------------------------------------------------

            labels = [

                str(value)[:30]

                for value
                in counts.index
            ]

            fig, ax = plt.subplots(
                figsize=(11, 6)
            )

            ax.bar(
                labels,
                counts.values
            )

            ax.set_title(
                f"Value Distribution — {column}",
                fontsize=15,
                fontweight="bold"
            )

            ax.set_xlabel(
                column
            )

            ax.set_ylabel(
                "Count"
            )

            plt.xticks(
                rotation=45,
                ha="right"
            )

            plt.tight_layout()

            path = self._chart_filename(
                "value_counts",
                question
            )

            plt.savefig(
                path,
                dpi=160,
                bbox_inches="tight"
            )

            plt.close(fig)

            return path

        # ====================================================
        # DISTRIBUTION HISTOGRAM
        # ====================================================

        if operation == "distribution":

            series = (
                df[column]
                .dropna()
            )

            if series.empty:

                return None

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            ax.hist(
                series,
                bins=30
            )

            ax.set_title(
                f"Distribution of {column}",
                fontsize=15,
                fontweight="bold"
            )

            ax.set_xlabel(
                column
            )

            ax.set_ylabel(
                "Frequency"
            )

            plt.tight_layout()

            path = self._chart_filename(
                "distribution",
                question
            )

            plt.savefig(
                path,
                dpi=160,
                bbox_inches="tight"
            )

            plt.close(fig)

            return path

        # ====================================================
        # GROUPBY MEAN
        # ====================================================

        if operation == "groupby_mean":

            values = (
                df.groupby(
                    group_by,
                    dropna=False
                )[column]
                .mean()
                .dropna()
                .sort_values(
                    ascending=False
                )
                .head(20)
            )

            if values.empty:

                return None

            labels = [
                str(value)[:30]
                for value
                in values.index
            ]

            fig, ax = plt.subplots(
                figsize=(11, 6)
            )

            ax.bar(
                labels,
                values.values
            )

            ax.set_title(
                f"Average {column} by {group_by}",
                fontsize=15,
                fontweight="bold"
            )

            ax.set_xlabel(
                group_by
            )

            ax.set_ylabel(
                f"Average {column}"
            )

            plt.xticks(
                rotation=45,
                ha="right"
            )

            plt.tight_layout()

            path = self._chart_filename(
                "groupby_mean",
                question
            )

            plt.savefig(
                path,
                dpi=160,
                bbox_inches="tight"
            )

            plt.close(fig)

            return path

        # ====================================================
        # GROUPBY COUNT
        # ====================================================

        if operation == "groupby_count":

            values = (
                df.groupby(
                    group_by,
                    dropna=False
                )[column]
                .count()
                .sort_values(
                    ascending=False
                )
                .head(20)
            )

            if values.empty:

                return None

            labels = [
                str(value)[:30]
                for value
                in values.index
            ]

            fig, ax = plt.subplots(
                figsize=(11, 6)
            )

            ax.bar(
                labels,
                values.values
            )

            ax.set_title(
                f"{column} Count by {group_by}",
                fontsize=15,
                fontweight="bold"
            )

            ax.set_xlabel(
                group_by
            )

            ax.set_ylabel(
                "Count"
            )

            plt.xticks(
                rotation=45,
                ha="right"
            )

            plt.tight_layout()

            path = self._chart_filename(
                "groupby_count",
                question
            )

            plt.savefig(
                path,
                dpi=160,
                bbox_inches="tight"
            )

            plt.close(fig)

            return path

        # ====================================================
        # GROUPBY RATE
        # ====================================================

        if operation == "groupby_rate":

            values = (
                df.groupby(
                    group_by,
                    dropna=False
                )[column]
                .mean()
                .dropna()
                .sort_values(
                    ascending=False
                )
                .head(20)
            )

            if values.empty:

                return None

            labels = [
                str(value)[:30]
                for value
                in values.index
            ]

            # Convert proportion to percentage
            percentage_values = (
                values * 100
            )

            fig, ax = plt.subplots(
                figsize=(11, 6)
            )

            ax.bar(
                labels,
                percentage_values.values
            )

            ax.set_title(
                f"{column} Rate by {group_by}",
                fontsize=15,
                fontweight="bold"
            )

            ax.set_xlabel(
                group_by
            )

            ax.set_ylabel(
                "Rate (%)"
            )

            plt.xticks(
                rotation=45,
                ha="right"
            )

            plt.tight_layout()

            path = self._chart_filename(
                "groupby_rate",
                question
            )

            plt.savefig(
                path,
                dpi=160,
                bbox_inches="tight"
            )

            plt.close(fig)

            return path

        # ====================================================
        # CORRELATION HEATMAP
        # ====================================================

        if operation == "correlation":

            numeric_df = (
                df.select_dtypes(
                    include=np.number
                )
            )

            if numeric_df.shape[1] < 2:

                return None

            correlation = (
                numeric_df.corr()
            )

            fig, ax = plt.subplots(
                figsize=(10, 8)
            )

            image = ax.imshow(
                correlation,
                interpolation="nearest",
                aspect="auto"
            )

            fig.colorbar(
                image,
                ax=ax
            )

            ax.set_xticks(
                range(
                    len(
                        correlation.columns
                    )
                )
            )

            ax.set_xticklabels(
                correlation.columns,
                rotation=45,
                ha="right"
            )

            ax.set_yticks(
                range(
                    len(
                        correlation.columns
                    )
                )
            )

            ax.set_yticklabels(
                correlation.columns
            )

            ax.set_title(
                "Correlation Matrix",
                fontsize=15,
                fontweight="bold"
            )

            plt.tight_layout()

            path = self._chart_filename(
                "correlation",
                question
            )

            plt.savefig(
                path,
                dpi=160,
                bbox_inches="tight"
            )

            plt.close(fig)

            return path

        # ====================================================
        # DATASET INFO
        # ====================================================

        if operation == "dataset_info":

            # Dataset overview is better represented by
            # the main dashboard visualizations.
            return None

        # ====================================================
        # DUPLICATES
        # ====================================================

        if operation == "duplicate_count":

            return None

        # ====================================================
        # SIMPLE STATISTICS
        # ====================================================

        if operation in [

            "column_mean",
            "column_median",
            "column_min",
            "column_max",
            "unique_count"
        ]:

            return None

        return None


    # ========================================================
    # GENERATE NATURAL LANGUAGE ANSWER
    # ========================================================

    def generate_answer(
        self,
        question,
        plan,
        result,
        pipeline_context=None
    ):

        if pipeline_context is None:

            pipeline_context = {}

        analysis_report = (
            pipeline_context.get(
                "analysis_report",
                {}
            )
        )

        quality_report = (
            pipeline_context.get(
                "quality_report",
                {}
            )
        )

        model_report = (
            pipeline_context.get(
                "model_report",
                {}
            )
        )

        # ----------------------------------------------------
        # RESPONSE PROMPT
        # ----------------------------------------------------

        prompt = f"""
You are the response formatter for an AI Dataset Assistant.

Your job is to convert VERIFIED dataset analysis results
into a clear answer for the user.

Python has already performed all calculations.

You must NOT perform new calculations.

====================================================
USER QUESTION
====================================================

{question}

====================================================
ANALYSIS PLAN
====================================================

{json.dumps(plan, indent=2, default=str)}

====================================================
VERIFIED PYTHON RESULT
====================================================

{json.dumps(result, indent=2, default=str)}

====================================================
PIPELINE CONTEXT
====================================================

DATASET ANALYSIS:
{json.dumps(analysis_report, indent=2, default=str)}

DATA QUALITY:
{json.dumps(quality_report, indent=2, default=str)}

MODEL PERFORMANCE:
{json.dumps(model_report, indent=2, default=str)}

====================================================
STRICT RULES
====================================================

1. ONLY use information present in the verified result
   or pipeline context.

2. NEVER invent numbers.

3. NEVER modify numerical values.

4. NEVER perform additional calculations.

5. Use simple, beginner-friendly language.

6. Keep the response concise.

7. Use Markdown.

8. Use headings when useful.

9. Use tables for multiple values.

10. Use bullet points where appropriate.

11. Put dataset column names inside backticks.

12. Do not mention Gemini.

13. Do not mention Python.

14. Do not mention agents.

15. Do not mention prompts.

16. Do not mention internal pipeline implementation.

17. Do not say "according to the AI".

18. Do not repeat the question.

19. Do not create unsupported conclusions.

20. If the requested information is unavailable, say:

"That information is not available in the current analysis."

====================================================
FORMATTING RULES
====================================================

Large numbers:

16599 → 16,599

Rates/proportions:

0.8717 → 87.17%

Do NOT convert ordinary decimal statistics
into percentages unless they represent a rate.

====================================================
OPERATION FORMATS
====================================================

dataset_info:

### 📊 Dataset Overview

| Metric | Value |
|---|---:|
| Rows | ... |
| Columns | ... |
| Missing Values | ... |
| Duplicate Rows | ... |

Then:

### 📋 Columns

| Column | Data Type |
|---|---|
| ... | ... |

If the user asks for an overview, also explain
what the dataset appears to contain using the
available column names.

----------------------------------------------------

missing_values:

### 🔍 Missing Values

If total missing is zero:

**✅ No missing values were found.**

Otherwise:

| Column | Missing Values |
|---|---:|
| ... | ... |

Then provide one short conclusion.

----------------------------------------------------

duplicate_count:

### 🔄 Duplicate Records

**Duplicate rows:** X

----------------------------------------------------

column_mean:

### 📊 Average

**Column:** `column`

**Mean:** value

----------------------------------------------------

column_median:

### 📊 Median

**Column:** `column`

**Median:** value

----------------------------------------------------

column_min:

### 📉 Minimum

**Column:** `column`

**Minimum:** value

----------------------------------------------------

column_max:

### 📈 Maximum

**Column:** `column`

**Maximum:** value

----------------------------------------------------

unique_count:

### 🔢 Unique Values

`column` contains **X unique values**.

----------------------------------------------------

value_counts:

### 📊 Value Distribution

| Value | Count |
|---|---:|
| ... | ... |

Identify the most common value only if clearly supported.

----------------------------------------------------

groupby_mean:

### 📊 Average `column` by `group_by`

| `group_by` | Average `column` |
|---|---:|
| ... | ... |

Mention the highest group only if clearly available.

----------------------------------------------------

groupby_count:

### 📊 Count by `group_by`

| `group_by` | Count |
|---|---:|
| ... | ... |

Mention the largest group if appropriate.

----------------------------------------------------

groupby_rate:

### 📊 Rate by `group_by`

| `group_by` | Rate |
|---|---:|
| ... | ... |

Represent rates as percentages.

----------------------------------------------------

correlation:

### 🔗 Correlation

Present important relationships in a concise table.

General interpretation:

- values near +1 indicate strong positive relationship
- values near -1 indicate strong negative relationship
- values near 0 indicate weak linear relationship

Only mention relationships actually present
in the verified result.

----------------------------------------------------

distribution:

### 📊 Distribution of `column`

| Statistic | Value |
|---|---:|
| Count | ... |
| Mean | ... |
| Median | ... |
| Standard Deviation | ... |
| Minimum | ... |
| Maximum | ... |

Then provide 1-2 observations that are directly
supported by the values.

====================================================
DATASET OVERVIEW QUESTIONS
====================================================

If the user asks:

"explain this dataset"

"tell me about this dataset"

"give me an overview"

"describe the dataset"

use:

### 📊 Dataset Overview

| Metric | Value |
|---|---:|
| Rows | ... |
| Columns | ... |
| Missing Values | ... |
| Duplicate Rows | ... |

### 📌 What This Dataset Contains

Brief explanation based ONLY on column names.

### 📋 Main Categories

Group related columns logically.

Only create categories supported by the actual
dataset.

### 🔍 Data Quality

Include only when quality information exists.

### 🤖 Model Performance

Include only when relevant to the question
or when a complete pipeline overview is requested.

====================================================

Return ONLY the final user-facing Markdown answer.
"""

        response = self.client.models.generate_content(

            model=self.model,

            contents=prompt
        )

        answer = response.text.strip()

        # ----------------------------------------------------
        # REMOVE CODE FENCES
        # ----------------------------------------------------

        answer = re.sub(
            r"^```(?:markdown)?",
            "",
            answer,
            flags=re.IGNORECASE
        )

        answer = re.sub(
            r"```$",
            "",
            answer
        )

        return answer.strip()


    # ========================================================
    # ASK DATASET ASSISTANT
    # ========================================================

    def ask(
        self,
        question,
        df,
        analysis_report=None,
        quality_report=None,
        model_report=None
    ):

        try:

            # ------------------------------------------------
            # VALIDATE DATAFRAME
            # ------------------------------------------------

            if df is None:

                raise ValueError(
                    "Dataset is not available."
                )

            if not isinstance(
                df,
                pd.DataFrame
            ):

                raise TypeError(
                    "Dataset must be a pandas DataFrame."
                )

            if df.empty:

                raise ValueError(
                    "The dataset is empty."
                )

            # ------------------------------------------------
            # PIPELINE CONTEXT
            # ------------------------------------------------

            pipeline_context = {

                "analysis_report":
                    analysis_report or {},

                "quality_report":
                    quality_report or {},

                "model_report":
                    model_report or {}
            }

            # ------------------------------------------------
            # CREATE PLAN
            # ------------------------------------------------

            plan = self.create_analysis_plan(
                df,
                question
            )

            # ------------------------------------------------
            # ADDITIONAL PYTHON-SIDE VALIDATION
            # ------------------------------------------------

            operation = plan.get(
                "operation"
            )

            column = plan.get(
                "column"
            )

            group_by = plan.get(
                "group_by"
            )

            # ------------------------------------------------
            # REQUIRED COLUMN VALIDATION
            # ------------------------------------------------

            column_operations = [

                "column_mean",
                "column_median",
                "column_min",
                "column_max",
                "unique_count",
                "value_counts",
                "distribution"
            ]

            if operation in column_operations:

                self.validate_column(
                    df,
                    column
                )

            # ------------------------------------------------
            # GROUPBY VALIDATION
            # ------------------------------------------------

            group_operations = [

                "groupby_mean",
                "groupby_count",
                "groupby_rate"
            ]

            if operation in group_operations:

                self.validate_column(
                    df,
                    column
                )

                self.validate_column(
                    df,
                    group_by
                )

            # ------------------------------------------------
            # EXECUTE PYTHON ANALYSIS
            # ------------------------------------------------

            result = self.execute_operation(
                df,
                plan
            )

            # ------------------------------------------------
            # CREATE VISUALIZATION
            # ------------------------------------------------

            chart = self.create_chart(
                df,
                plan,
                result,
                question
            )

            # ------------------------------------------------
            # GENERATE ANSWER
            # ------------------------------------------------

            answer = self.generate_answer(
                question=question,
                plan=plan,
                result=result,
                pipeline_context=pipeline_context
            )

            # ------------------------------------------------
            # RETURN COMPLETE RESPONSE
            # ------------------------------------------------

            return {

                "question":
                    question,

                "plan":
                    plan,

                "result":
                    result,

                "answer":
                    answer,

                "chart":
                    chart,

                "error":
                    None
            }

        except Exception as e:

            return {

                "question":
                    question,

                "plan":
                    None,

                "result":
                    None,

                "answer":
                    None,

                "chart":
                    None,

                "error":
                    str(e)
            }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Dataset Assistant Agent loaded successfully."
    )