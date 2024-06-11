from typing import Dict, Tuple, Union

import numpy as np
import xgboost as xgb


import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

import pandas as pd
from pandas import Series, DataFrame

from scipy.sparse._csr import csr_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from pandas import Series
from scipy.sparse._csr import csr_matrix

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def hyperparameter_tuning(
    data,
    **kwargs,
) -> Tuple[
    Dict[str, Union[bool, float, int, str]],
    csr_matrix,
    Series,
]:

    X_train, y_train, dv, model_class = data


    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("homework03")
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        mlflow.set_tag("developer", "jelambrar")

        rf = LinearRegression()
        rf.fit(X_train, y_train)

        mlflow.sklearn.log_model(
            sk_model=rf,
            artifact_path="sklearn-model",
            registered_model_name="sk-learn-linear-reg-model",
        )

    return X_train, y_train