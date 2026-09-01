import pandas as pd
import numpy as np


class CleaningAgent:

    def __init__(self):
        self.report = {}


    def clean(self, df):

        # Copy dataset
        df = df.copy()


        print("=" * 60)
        print("=" * 60)


        self.report["initial_shape"] = df.shape


        # ------------------------------------------------
        # 1. Standardize column names
        # ------------------------------------------------

        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )


        print("✔ Column names standardized")



        # ------------------------------------------------
        # 2. Remove duplicate rows
        # ------------------------------------------------

        duplicate_count = (
            df.duplicated()
            .sum()
        )


        df = (
            df.drop_duplicates()
            .reset_index(drop=True)
        )


        self.report["duplicates_removed"] = int(
            duplicate_count
        )


        print(
            f"✔ Removed duplicates: {duplicate_count}"
        )



        # ------------------------------------------------
        # 3. Handle missing values
        # ------------------------------------------------

        missing_before = (
            df.isnull()
            .sum()
        )


        self.report["missing_before"] = (
            missing_before[
                missing_before > 0
            ]
            .to_dict()
        )


        print("\nHandling Missing Values:")



        for column in df.columns:


            missing = (
                df[column]
                .isnull()
                .sum()
            )


            if missing == 0:
                continue



            # Numeric columns

            if pd.api.types.is_numeric_dtype(
                df[column]
            ):


                median_value = (
                    df[column]
                    .median()
                )


                if pd.notna(median_value):

                    df[column] = (
                        df[column]
                        .fillna(
                            median_value
                        )
                    )


                    print(
                        f"✔ {column}: filled with median"
                    )


            # Categorical columns

            else:


                mode = (
                    df[column]
                    .mode()
                )


                if not mode.empty:


                    df[column] = (
                        df[column]
                        .fillna(
                            mode.iloc[0]
                        )
                    )


                    print(
                        f"✔ {column}: filled with mode"
                    )


                else:


                    df[column] = (
                        df[column]
                        .fillna(
                            "Unknown"
                        )
                    )



        missing_after = (
            df.isnull()
            .sum()
        )


        self.report["missing_after"] = (
            missing_after[
                missing_after > 0
            ]
            .to_dict()
        )



        # ------------------------------------------------
        # 4. Remove constant columns
        # ------------------------------------------------


        constant_columns = [

            col

            for col in df.columns

            if df[col]
            .nunique(
                dropna=False
            ) <= 1

        ]



        if constant_columns:


            df = df.drop(
                columns=constant_columns
            )



        self.report[
            "constant_columns_removed"
        ] = constant_columns



        print(
            f"✔ Removed constant columns: {constant_columns}"
        )



        # ------------------------------------------------
        # 5. Smart Outlier Handling (FIXED)
        # ------------------------------------------------


        numerical_columns = (
            df.select_dtypes(
                include=np.number
            )
            .columns
        )


        outlier_report = {}



        print("\nHandling Outliers:")



        for column in numerical_columns:


            Q1 = (
                df[column]
                .quantile(0.25)
            )


            Q3 = (
                df[column]
                .quantile(0.75)
            )


            IQR = Q3 - Q1



            if IQR == 0 or pd.isna(IQR):

                outlier_report[column] = 0

                continue



            lower = (
                Q1 - 1.5 * IQR
            )


            upper = (
                Q3 + 1.5 * IQR
            )



            outliers = (

                (
                    df[column] < lower
                )

                |

                (
                    df[column] > upper
                )

            ).sum()



            outlier_report[column] = int(
                outliers
            )



            # Store original datatype

            original_dtype = (
                df[column]
                .dtype
            )



            # Convert safely

            df[column] = (

                df[column]
                .astype(float)
                .clip(
                    lower,
                    upper
                )

            )



            # Restore integer columns

            if np.issubdtype(
                original_dtype,
                np.integer
            ):


                df[column] = (

                    df[column]
                    .round()
                    .astype(int)

                )



        self.report[
            "outliers_handled"
        ] = outlier_report



        print(
            "✔ Outliers handled:",
            outlier_report
        )



        # ------------------------------------------------
        # 6. High Cardinality Detection
        # ------------------------------------------------


        high_cardinality = []



        for column in (
            df.select_dtypes(
                include="object"
            )
            .columns
        ):


            unique_ratio = (

                df[column]
                .nunique()
                /
                len(df)

            )


            if unique_ratio > 0.80:


                high_cardinality.append(
                    column
                )



        self.report[
            "high_cardinality_columns"
        ] = high_cardinality



        print(
            "✔ High-cardinality columns:",
            high_cardinality
        )



        # ------------------------------------------------
        # Final Report
        # ------------------------------------------------


        self.report[
            "final_shape"
        ] = df.shape



        print("\nCleaning Completed")

        print(
            "Before:",
            self.report["initial_shape"]
        )

        print(
            "After :",
            self.report["final_shape"]
        )


        print("=" * 60)



        return df, self.report