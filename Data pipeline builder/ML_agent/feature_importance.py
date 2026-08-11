# agents/feature_importance.py


import pandas as pd



class FeatureImportance:


    def get_importance(
            self,
            model,
            features
    ):


        if hasattr(
            model,
            "feature_importances_"
        ):


            importance = pd.DataFrame({

                "feature":features,

                "importance":
                model.feature_importances_

            })


            return importance.sort_values(
                "importance",
                ascending=False
            )


        return None