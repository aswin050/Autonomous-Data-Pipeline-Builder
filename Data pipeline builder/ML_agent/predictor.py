# =====================================================
# agents/predictor.py
# =====================================================

import pandas as pd
import joblib



class Predictor:


    def __init__(self):

        self.transformer_path = (
            "models/feature_transformer.pkl"
        )

        self.feature_columns_path = (
            "models/feature_columns.pkl"
        )

        self.metadata_path = (
            "models/model_metadata.pkl"
        )



    # =====================================================
    # PREDICT FUNCTION
    # =====================================================

    def predict(
            self,
            model,
            data
    ):


        print("="*60)
        print("PREDICTOR AGENT STARTED")
        print("="*60)



        # =================================================
        # LOAD SAVED OBJECTS
        # =================================================


        saved = joblib.load(
            self.transformer_path
        )


        transformer = saved["transformer"]

        selector = saved["selector"]



        feature_columns = joblib.load(
            self.feature_columns_path
        )


        metadata = joblib.load(
            self.metadata_path
        )


        target_column = metadata.get(
            "target_column"
        )



        print(
            "✔ Transformer Loaded"
        )

        print(
            "✔ Feature Columns Loaded"
        )

        print(
            "✔ Target:",
            target_column
        )



        # =================================================
        # STANDARDIZE COLUMN NAMES
        # =================================================


        data.columns = (

            data.columns
            .str.lower()
            .str.strip()
            .str.replace(
                " ",
                "_"
            )

        )



        print("\nInput Columns:")

        print(
            data.columns.tolist()
        )



        # =================================================
        # REMOVE TARGET COLUMN
        # =================================================


        if target_column in data.columns:


            data = data.drop(

                columns=[target_column]

            )


            print(
                "✔ Removed target:",
                target_column
            )



        # =================================================
        # CHECK REQUIRED FEATURES
        # =================================================


        missing_columns = (

            set(feature_columns)
            -
            set(data.columns)

        )



        if missing_columns:


            raise ValueError(

                f"""

Missing required features:

{missing_columns}


Prediction dataset does not match
training dataset.

Expected features:

{feature_columns}

Received features:

{data.columns.tolist()}

"""

            )



        print(
            "✔ Feature columns matched"
        )



        # =================================================
        # KEEP TRAINING FEATURES ONLY
        # =================================================


        data = data[
            feature_columns
        ]



        print(
            "\nFinal Prediction Input:"
        )


        print(
            data.columns.tolist()
        )



        # =================================================
        # TRANSFORMATION
        # =================================================


        transformed = transformer.transform(
            data
        )


        print(
            "✔ Data Transformed"
        )



        # =================================================
        # FEATURE SELECTION
        # =================================================


        selected = selector.transform(
            transformed
        )


        print(
            "✔ Feature Selection Applied"
        )



        # =================================================
        # PREDICTION
        # =================================================


        prediction = model.predict(
            selected
        )


        print(
            "✔ Prediction Completed"
        )



        return prediction