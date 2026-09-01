# agents/model_selector.py

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
    ExtraTreesClassifier
)

from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


class ModelSelector:


    def get_models(self, task):

        models = {}


        if task == "classification":

            models = {

                "Logistic Regression":
                    LogisticRegression(max_iter=1000),

                "Random Forest":
                    RandomForestClassifier(),

                "Gradient Boosting":
                    GradientBoostingClassifier(),

                "AdaBoost":
                    AdaBoostClassifier(),

                "Extra Trees":
                    ExtraTreesClassifier(),

                "SVM":
                    SVC(probability=True),

                "KNN":
                    KNeighborsClassifier(),

                "XGBoost":
                    XGBClassifier(),

                "LightGBM":
                    LGBMClassifier(verbose=-1),

                "CatBoost":
                    CatBoostClassifier(verbose=0)

            }


        else:

            models = {


                "Linear Regression":
                    LinearRegression(),


                "Random Forest":
                    RandomForestRegressor(),


                "Gradient Boosting":
                    GradientBoostingRegressor(),


                "SVR":
                    SVR(),


                "KNN":
                    KNeighborsRegressor(),


                "XGBoost":
                    XGBRegressor(),


                "LightGBM":
                    LGBMRegressor(),


                "CatBoost":
                    CatBoostRegressor(verbose=0)

            }


        return models