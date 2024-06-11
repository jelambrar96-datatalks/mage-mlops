from typing import Tuple, Dict, Union, Callable

import pandas as pd
from pandas import Series, DataFrame


import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature


from scipy.sparse._csr import csr_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from mlops.utils.data_preparation.cleaning import clean
from mlops.utils.data_preparation.feature_engineering import combine_features
from mlops.utils.data_preparation.feature_selector import select_features
from mlops.utils.data_preparation.splitters import split_on_value

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform(
    df1, **kwargs
):


    target='duration'
    # df1['PU_DO'] = df1['PULocationID'].astype(str) + '_' + df1['DOLocationID'].astype(str)
    df1['PU'] = 'PU_' + df1['PULocationID'].astype(str)
    df1['DO'] = 'DO_' + df1['DOLocationID'].astype(str)


    categorical = ['PU', 'DO']
    # categorical = ['PU_DO']
    numerical = []

    dv = DictVectorizer()

    train_dicts = df1[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    y_train = df1[target].values

    # print(type(y_train))
    # print(type(X_train))



    # model_class.fit(df_train, y_train)
    
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

    return X_train, y_train, dv, rf