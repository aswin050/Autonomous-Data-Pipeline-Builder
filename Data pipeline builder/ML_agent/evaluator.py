# agents/evaluator.py


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


class Evaluator:


    def evaluate(
            self,
            model,
            X_test,
            y_test,
            task
    ):


        prediction = model.predict(X_test)

        probability = None

        if task == "classification":

            if hasattr(model, "predict_proba"):

                probability = model.predict_proba(X_test)


        report={}



        if task == "classification":

            report["accuracy"] = accuracy_score(
                y_test,
                prediction
            )

            report["precision"] = precision_score(
                y_test,
                prediction,
                average="weighted",
                zero_division = 0
            )

            report["recall"] = recall_score(
                y_test,
                prediction,
                average="weighted",
                zero_division=0
            )

            report["f1_score"] = f1_score(
                y_test,
                prediction,
                average="weighted",
                zero_division=0
            )

            report["confusion_matrix"] = confusion_matrix(
                y_test,
                prediction
            ).tolist()

            report["classification_report"] = classification_report(
                y_test,
                prediction,
                output_dict=True
            )

            if probability is not None and len(set(y_test)) == 2:

                report["roc_auc"] = roc_auc_score(
                    y_test,
                    probability[:, 1]
                )

        else:

            report["MAE"] = mean_absolute_error(
                y_test,
                prediction
            )

            report["RMSE"] = mean_squared_error(
                y_test,
                prediction,
                squared=False
            )

            report["R2"] = r2_score(
                y_test,
                prediction
            )

        return report