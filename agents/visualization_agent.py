import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# VISUALIZATION AGENT
# ============================================================

class VisualizationAgent:

    def __init__(
        self,
        output_folder="output/plots"
    ):

        self.output_folder = output_folder

        os.makedirs(
            self.output_folder,
            exist_ok=True
        )

    # ========================================================
    # SAVE FIGURE
    # ========================================================

    def _save_figure(
        self,
        filename
    ):

        path = os.path.join(
            self.output_folder,
            filename
        )

        plt.savefig(
            path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        return path

    # ========================================================
    # DATA TYPE DISTRIBUTION
    # ========================================================

    def dataset_overview(
        self,
        df
    ):

        numeric = 0
        categorical = 0
        datetime = 0
        boolean = 0
        other = 0

        for dtype in df.dtypes:

            if pd.api.types.is_bool_dtype(dtype):

                boolean += 1

            elif pd.api.types.is_numeric_dtype(dtype):

                numeric += 1

            elif pd.api.types.is_datetime64_any_dtype(dtype):

                datetime += 1

            elif pd.api.types.is_object_dtype(dtype):

                categorical += 1

            else:

                other += 1

        labels = []
        values = []

        type_data = {

            "Numeric": numeric,
            "Categorical": categorical,
            "Datetime": datetime,
            "Boolean": boolean,
            "Other": other
        }

        for label, value in type_data.items():

            if value > 0:

                labels.append(label)
                values.append(value)

        if not values:

            return None

        plt.figure(
            figsize=(9, 7)
        )

        plt.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={
                "width": 0.42
            }
        )

        plt.title(
            "Data Type Distribution"
        )

        plt.text(
            0,
            0,
            f"{df.shape[1]}\nColumns",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold"
        )

        plt.tight_layout()

        return self._save_figure(
            "data_type_distribution.png"
        )

    # ========================================================
    # MISSING VALUES — DONUT CHART
    # ========================================================

    def missing_values(
        self,
        df
    ):

        total_cells = (
            df.shape[0] *
            df.shape[1]
        )

        if total_cells == 0:

            return None

        total_missing = int(
            df.isnull()
            .sum()
            .sum()
        )

        total_complete = (
            total_cells -
            total_missing
        )

        values = [
            total_complete,
            total_missing
        ]

        labels = [
            "Complete",
            "Missing"
        ]

        # ----------------------------------------------------
        # DONUT
        # ----------------------------------------------------

        plt.figure(
            figsize=(9, 8)
        )

        plt.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={
                "width": 0.42
            },
            pctdistance=0.78
        )

        # ----------------------------------------------------
        # CENTER TEXT
        # ----------------------------------------------------

        completeness = (
            total_complete /
            total_cells
        ) * 100

        plt.text(
            0,
            0,
            f"{completeness:.1f}%\nComplete",
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold"
        )

        plt.title(
            "Dataset Completeness"
        )

        plt.tight_layout()

        return self._save_figure(
            "missing_values.png"
        )

    # ========================================================
    # CORRELATION HEATMAP
    # ========================================================

    def correlation_heatmap(
        self,
        df
    ):

        numeric_df = df.select_dtypes(
            include=np.number
        )

        if numeric_df.shape[1] < 2:

            return None

        correlation = (
            numeric_df.corr()
        )

        plt.figure(
            figsize=(10, 8)
        )

        plt.imshow(
            correlation,
            interpolation="nearest"
        )

        plt.colorbar()

        plt.xticks(
            range(
                len(
                    correlation.columns
                )
            ),
            correlation.columns,
            rotation=45,
            ha="right"
        )

        plt.yticks(
            range(
                len(
                    correlation.columns
                )
            ),
            correlation.columns
        )

        plt.title(
            "Correlation Matrix"
        )

        plt.tight_layout()

        return self._save_figure(
            "correlation_heatmap.png"
        )

    # ========================================================
    # TARGET DISTRIBUTION
    # ========================================================

    def target_distribution(
        self,
        y
    ):

        if y is None:

            return None

        series = pd.Series(
            y
        ).dropna()

        if series.empty:

            return None

        counts = (
            series
            .value_counts()
            .head(20)
        )

        plt.figure(
            figsize=(10, 6)
        )

        counts.plot(
            kind="bar"
        )

        plt.title(
            "Target Distribution"
        )

        plt.xlabel(
            "Target"
        )

        plt.ylabel(
            "Count"
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.tight_layout()

        return self._save_figure(
            "target_distribution.png"
        )

    # ========================================================
    # TARGET DONUT CHART
    # ========================================================

    def target_pie_chart(
        self,
        y
    ):

        if y is None:

            return None

        series = pd.Series(
            y
        ).dropna()

        if series.empty:

            return None

        counts = (
            series
            .value_counts()
            .head(10)
        )

        plt.figure(
            figsize=(9, 8)
        )

        plt.pie(
            counts.values,
            labels=[
                str(x)
                for x in counts.index
            ],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={
                "width": 0.42
            }
        )

        plt.title(
            "Target Distribution"
        )

        plt.tight_layout()

        return self._save_figure(
            "target_pie_chart.png"
        )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    def model_comparison(
        self,
        model_report
    ):

        if not model_report:

            return None

        # ----------------------------------------------------
        # POSSIBLE MODEL CONTAINERS
        # ----------------------------------------------------

        models = None

        if isinstance(
            model_report,
            dict
        ):

            for key in [
                "models",
                "results",
                "model_results",
                "performance"
            ]:

                if key in model_report:

                    models = model_report[key]

                    break

        if not models:

            return None

        if not isinstance(
            models,
            dict
        ):

            return None

        names = []
        scores = []

        # ----------------------------------------------------
        # EXTRACT ACCURACY
        # ----------------------------------------------------

        for name, info in models.items():

            if not isinstance(
                info,
                dict
            ):

                continue

            score = None

            for key in [
                "accuracy",
                "Accuracy",
                "test_accuracy",
                "test_score"
            ]:

                if key in info:

                    score = info[key]

                    break

            if score is None:

                continue

            try:

                score = float(score)

            except (
                TypeError,
                ValueError
            ):

                continue

            names.append(
                str(name)
            )

            scores.append(
                score
            )

        if not scores:

            return None

        # ----------------------------------------------------
        # SORT
        # ----------------------------------------------------

        combined = sorted(
            zip(
                names,
                scores
            ),
            key=lambda x: x[1],
            reverse=True
        )

        names = [
            x[0]
            for x in combined
        ]

        scores = [
            x[1]
            for x in combined
        ]

        # ----------------------------------------------------
        # LIMIT
        # ----------------------------------------------------

        names = names[:15]
        scores = scores[:15]

        # ----------------------------------------------------
        # PLOT
        # ----------------------------------------------------

        plt.figure(
            figsize=(11, 7)
        )

        plt.bar(
            names,
            scores
        )

        plt.title(
            "Model Performance Comparison"
        )

        plt.xlabel(
            "Model"
        )

        plt.ylabel(
            "Accuracy / Score"
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.tight_layout()

        return self._save_figure(
            "model_comparison.png"
        )