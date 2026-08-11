# =====================================================
# ML_agent/training_agent.py
# =====================================================

import os
import joblib
import numpy as np
import pandas as pd


from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    OneHotEncoder
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold


from ML_agent.model_selector import ModelSelector
from ML_agent.hyperparameter_tuner import HyperparameterTuner
from ML_agent.evaluator import Evaluator
from ML_agent.model_saver import ModelSaver


from ML_agent.utils import (
    detect_problem_type,
    split_data
)



class TrainingAgent:


    def __init__(self):

        os.makedirs(
            "models",
            exist_ok=True
        )



    # =====================================================
    # BUILD PREPROCESSOR
    # =====================================================

    def build_preprocessor(self, X):


        numerical = X.select_dtypes(
            include=np.number
        ).columns.tolist()


        categorical = X.select_dtypes(
            include=[
                "object",
                "category"
            ]
        ).columns.tolist()



        transformers = []



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



        transformer = ColumnTransformer(

            transformers

        )


        return transformer





    # =====================================================
    # TRAIN FUNCTION
    # =====================================================


    def train(
            self,
            X,
            y,
            target_column
    ):


        print("\n")
        print("="*60)
        print("TRAINING AGENT STARTED")
        print("="*60)



        print(
            "Target Column:",
            target_column
        )



        # Remove target if accidentally present

        if target_column in X.columns:


            X = X.drop(
                columns=[target_column]
            )


            print(
                "✔ Target removed from features"
            )



        print(
            "Input Features:"
        )


        print(
            X.columns.tolist()
        )



        # =================================================
        # PROBLEM TYPE
        # =================================================


        task = detect_problem_type(
            y
        )


        print(
            "Task:",
            task
        )



        # =================================================
        # TARGET ENCODING
        # =================================================


        if task == "classification":


            if y.dtype == "object":


                label_encoder = LabelEncoder()


                y = label_encoder.fit_transform(
                    y
                )


                joblib.dump(

                    label_encoder,

                    "models/label_encoder.pkl"

                )


                print(
                    "✔ Label Encoder Saved"
                )



        # =================================================
        # SAVE RAW FEATURES
        # =================================================


        joblib.dump(

            X.columns.tolist(),

            "models/feature_columns.pkl"

        )


        print(
            "✔ Raw feature columns saved"
        )



        # =================================================
        # TRAIN TEST SPLIT
        # =================================================


        X_train, X_test, y_train, y_test = split_data(
            X,
            y
        )



        print(
            "Train:",
            X_train.shape
        )

        print(
            "Test:",
            X_test.shape
        )



        # =================================================
        # PREPROCESSING
        # =================================================


        transformer = self.build_preprocessor(
            X_train
        )


        X_train = transformer.fit_transform(
            X_train
        )


        X_test = transformer.transform(
            X_test
        )


        print(
            "✔ Preprocessing Completed"
        )



        # =================================================
        # FEATURE SELECTION
        # =================================================


        feature_selector = VarianceThreshold(
            threshold=0.01
        )


        X_train = feature_selector.fit_transform(
            X_train
        )


        X_test = feature_selector.transform(
            X_test
        )


        print(
            "✔ Feature Selection Completed"
        )



        # =================================================
        # MODEL SELECTION
        # =================================================


        model_selector = ModelSelector()


        models = model_selector.get_models(
            task
        )


        print(
            "Models:"
        )


        for model in models:

            print(
                "-",
                model
            )



        tuner = HyperparameterTuner()

        evaluator = Evaluator()



        best_model = None

        best_score = -1

        best_result = {}

        model_report = {}



        # =================================================
        # TRAIN MODELS
        # =================================================


        for name, model in models.items():


            print("\nTraining:", name)


            try:


                trained_model = tuner.tune(

                    model,

                    X_train,

                    y_train

                )



                result = evaluator.evaluate(

                    trained_model,

                    X_test,

                    y_test,

                    task

                )


                print(
                    result
                )



                if task == "classification":

                    score = result.get(
                        "accuracy",
                        0
                    )


                else:

                    score = result.get(
                        "r2_score",
                        0
                    )



                model_report[name] = score



                if score > best_score:


                    best_score = score

                    best_model = trained_model

                    best_result = result



            except Exception as e:


                print(
                    f"❌ {name} failed:",
                    e
                )



        # =================================================
        # CHECK MODEL
        # =================================================


        if best_model is None:


            raise Exception(
                "All models failed during training"
            )



        print(
            "Best Model:",
            type(best_model).__name__
        )


        print(
            "Best Score:",
            best_score
        )



        # =================================================
        # SAVE FILES
        # =================================================


        saver = ModelSaver()


        saver.save(
            best_model
        )



        joblib.dump(

            {

                "transformer": transformer,

                "selector": feature_selector

            },

            "models/feature_transformer.pkl"

        )


        print(
            "✔ Transformer saved"
        )



        metadata = {


            "task": task,

            "target_column": target_column,

            "features": X.columns.tolist()

        }



        joblib.dump(

            metadata,

            "models/model_metadata.pkl"

        )

        # =============================================
        # SAVE TRAINING REPORT
        # =============================================

        training_report = {

            "best_result": best_result,

            "model_report": model_report,

            "best_score": best_score,

            "best_model": type(best_model).__name__

        }


        joblib.dump(

            training_report,

            "models/training_report.pkl"

        )


        print(
            "✔ Training report saved"
        )



        print(
            "✔ Metadata saved"
        )



        return (

            best_model,

            model_report,

            best_result

        )