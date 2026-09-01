from sklearn.model_selection import train_test_split
import pandas as pd



def detect_problem_type(y):


    # Object/string target

    if y.dtype == "object":

        return "classification"



    # Category target

    if str(y.dtype) == "category":

        return "classification"



    # Small number of unique values

    if y.nunique() <= 20:

        return "classification"



    return "regression"





def split_data(
        X,
        y
):


    task = detect_problem_type(y)



    if task == "classification":


        return train_test_split(

            X,

            y,

            test_size=0.2,

            random_state=42,

            stratify=y

        )


    else:


        return train_test_split(

            X,

            y,

            test_size=0.2,

            random_state=42

        )