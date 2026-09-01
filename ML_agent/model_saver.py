# =====================================================
# agents/model_saver.py
# =====================================================

import joblib
import os


class ModelSaver:


    def __init__(self):

        os.makedirs(
            "models",
            exist_ok=True
        )



    # ==========================================
    # SAVE MODEL
    # ==========================================

    def save(
            self,
            model,
            path="models/best_model.pkl"
    ):

        joblib.dump(
            model,
            path
        )

        print(
            "✔ Model saved:",
            path
        )



    # ==========================================
    # SAVE TRANSFORMER
    # ==========================================

    def save_transformer(
            self,
            transformer,
            selector,
            path="models/feature_transformer.pkl"
    ):


        joblib.dump(

            {
                "transformer": transformer,
                "selector": selector
            },

            path
        )


        print(
            "✔ Transformer saved:",
            path
        )



    # ==========================================
    # LOAD
    # ==========================================

    def load(
            self,
            path
    ):

        return joblib.load(path)