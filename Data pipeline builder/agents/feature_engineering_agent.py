import pandas as pd
import numpy as np
import joblib
import os


from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import VarianceThreshold




class AdvancedFeatureEngineeringAgent:


    def __init__(self):

        self.report = {}

        self.transformer = None

        self.selector = None


        self.target_keywords = [

            "target",
            "label",
            "class",
            "output",
            "result",
            "survived",
            "churn",
            "outcome",
            "price",
            "score"

        ]




    # =====================================================
    # TARGET DETECTION
    # =====================================================


    def detect_target(self, df):

        candidates = []


        # =====================================
        # 1. Name based target detection
        # =====================================

        target_keywords = [

            "target",
            "label",
            "class",
            "output",
            "result",
            "outcome",
            "response",
            "status",
            "category",
            "sales",
            "profit",
            "price",
            "churn",
            "attrition",
            "survived",
            "diagnosis"

        ]


        for col in df.columns:

            col_name = col.lower()

            for keyword in target_keywords:

                if keyword in col_name:

                    candidates.append(col)

                    break



        # =====================================
        # 2. Low cardinality detection
        # =====================================

        for col in df.columns:


            unique = df[col].nunique()


            if unique <= 10:


                if col not in candidates:

                    candidates.append(col)



        # =====================================
        # 3. Last column fallback
        # =====================================

        if len(candidates) == 0:


            candidates.append(
                df.columns[-1]
            )



        print(
            "Target Candidates:",
            candidates
        )


        # choose first candidate

        target = candidates[0]


        self.report["target"] = target


        print(
            "Selected Target:",
            target
        )


        return target





    # =====================================================
    # MISSING VALUE CHECK
    # =====================================================


    def verify_missing_values(
            self,
            df,
            stage
    ):


        missing=df.isnull().sum()


        missing=missing[
            missing>0
        ].to_dict()



        self.report[
            "missing_"+stage
        ]=missing



        print(
            f"{stage} missing:",
            missing
        )





    # =====================================================
    # TYPE CONVERSION
    # =====================================================


    def convert_data_types(self,df):


        for col in df.columns:


            if df[col].dtype=="object":


                numeric=pd.to_numeric(
                    df[col],
                    errors="coerce"
                )


                if numeric.notna().sum()>0:


                    df[col]=numeric



        return df





    # =====================================================
    # REMOVE ID COLUMNS
    # =====================================================


    def remove_identifier_columns(self,df):


        remove=[]


        for col in df.columns:


            name=col.lower()


            if (

                "id" in name
                or "uuid" in name
                or "ticket" in name
                or "name" in name

            ):

                remove.append(col)



        df.drop(

            columns=remove,

            inplace=True,

            errors="ignore"

        )

        self.report["removed_columns"] = remove

        print(
            "Removed:",
            remove
        )


        return df





    # =====================================================
    # FEATURE CREATION
    # =====================================================


    def create_features(self,df):
        created = []

        if (

            "sibsp" in df.columns
            and
            "parch" in df.columns

        ):


            df["family_size"]=(

                df["sibsp"]
                +
                df["parch"]
                +
                1

            )

            created.append("family_size")

        self.report["created_features"] = created

        return df





    # =====================================================
    # HIGH CARDINALITY
    # =====================================================


    def remove_high_cardinality(self,df):


        remove=[]


        for col in df.select_dtypes(
            include="object"
        ).columns:


            ratio=(

                df[col].nunique()
                /
                len(df)

            )


            if ratio>0.8:

                remove.append(col)



        df.drop(

            columns=remove,

            inplace=True,

            errors="ignore"

        )

        self.report["high_cardinality_removed"] = remove

        print("High Cardinality Removed:", remove)


        return df





    # =====================================================
    # COLUMN DETECTION
    # =====================================================


    def detect_columns(self,df):


        numerical=df.select_dtypes(
            include=np.number
        ).columns.tolist()



        categorical=df.select_dtypes(
            include=[
                "object",
                "category"
            ]
        ).columns.tolist()



        return numerical,categorical





    # =====================================================
    # PIPELINE
    # =====================================================


    def build_pipeline(
            self,
            numerical,
            categorical
    ):


        transformers=[]


        if numerical:


            transformers.append(

                (

                "num",

                Pipeline([

                    (
                    "scaler",
                    StandardScaler()
                    )

                ]),

                numerical

                )

            )



        if categorical:


            transformers.append(

                (

                "cat",

                Pipeline([

                    (

                    "encoder",

                    OneHotEncoder(

                    handle_unknown="ignore",

                    sparse_output=False

                    )

                    )

                ]),

                categorical

                )

            )



        return ColumnTransformer(

            transformers

        )





    # =====================================================
    # SAVE TRANSFORMER
    # =====================================================


    def save_transformer(
            self,
            path="models/feature_transformer.pkl"
    ):


        os.makedirs(
            "models",
            exist_ok=True
        )



        joblib.dump(

            {

                "transformer": self.transformer,

                "selector": self.selector,

                "training_columns":
                    self.report.get(
                        "training_columns",
                        []
                    ),

                "selected_columns":
                    self.report.get(
                        "selected_columns",
                        []
                    ),

                "target_column":
                    self.report.get(
                        "target",
                        None
                    ),

                "removed_columns":
                    self.report.get(
                        "removed_columns",
                        []
                    ),

                "high_cardinality_removed":
                    self.report.get(
                        "high_cardinality_removed",
                        []
                    ),

                "created_features":
                    self.report.get(
                        "created_features",
                        []
                    )
                    

            },

            path

        )



        print(
            f"✔ Transformer saved : {path}"
        )





    # =====================================================
    # MAIN PROCESS
    # =====================================================


    def process(self,df):


        print("="*60)

        print(
            "FEATURE ENGINEERING AGENT"
        )

        print("="*60)



        df=df.copy()



        self.report["initial_shape"]=df.shape



        target_column=self.detect_target(df)



        y=None



        if target_column:


            y=df[target_column]


            df.drop(

                columns=[target_column],

                inplace=True

            )




        self.verify_missing_values(
            df,
            "before"
        )



        df=self.convert_data_types(df)



        df=self.remove_identifier_columns(df)



        df=self.create_features(df)



        df=self.remove_high_cardinality(df)




        numerical,categorical=self.detect_columns(df)

        self.report["training_columns"] = (
            df.columns.tolist()
        )


        print(
            "Numerical:",
            numerical
        )

        print(
            "Categorical:",
            categorical
        )


        self.report["final_shape"] = df.shape


        print(
            "Final Shape:",
            df.shape
        )


        print(
            "Feature Engineering Completed"
        )


        return (

            df,

            y,

            self.report,

            target_column

        )