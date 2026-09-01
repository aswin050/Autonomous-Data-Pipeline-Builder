import pandas as pd


class DatasetAnalysisAgent:


    def __init__(self):

        self.report = {}



    def analyze(self, df):


        if not isinstance(df, pd.DataFrame):

            print(
                "Analyze Agent received invalid input"
            )

            return None



        print("="*60)
        print("="*60)



        rows = df.shape[0]
        columns = df.shape[1]


        print(
            f"Rows               : {rows}"
        )


        print(
            f"Columns            : {columns}"
        )



        print("\nColumn Names:")

        print(
            df.columns.tolist()
        )



        print("\nData Types:")

        print(
            df.dtypes
        )



        print("\nMissing Values:")

        missing = df.isnull().sum()

        print(
            missing
        )



        print("\nDuplicate Rows:")

        duplicates = df.duplicated().sum()

        print(
            duplicates
        )



        print("\nStatistical Summary:")

        summary = df.describe(
            include="all"
        )

        print(
            summary
        )



        print("="*60)



        self.report = {


            "rows":
                rows,


            "columns":
                columns,


            "column_names":
                df.columns.tolist(),


            "data_types":
                df.dtypes.astype(str),


            "missing_values":
                missing[missing>0],


            "duplicate_rows":
                int(duplicates),


            "summary":
                summary

        }



        return self.report