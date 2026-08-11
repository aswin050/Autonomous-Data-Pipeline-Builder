from sklearn.model_selection import RandomizedSearchCV


class HyperparameterTuner:


    def tune(
            self,
            model,
            X,
            y
    ):


        try:

            model.fit(
                X,
                y
            )

            return model


        except Exception as e:

            print(
                "Training error:",
                e
            )

            return model