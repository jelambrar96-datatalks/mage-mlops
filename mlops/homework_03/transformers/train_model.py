from typing import Callable, Dict, Tuple, Union

from pandas import Series, DataFrame
from scipy.sparse._csr import csr_matrix

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error

from mlops.utils.models.sklearn import load_class, tune_hyperparameters, train_model

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform(
    training_set,
    *args,
    **kwargs,
):

    X_train, y_train, dv = training_set
    df_train = DataFrame(X_train.toarray())

    # model_class = load_class(model_class_name)
    model_class = LinearRegression()
    # model_class.fit(df_train, y_train)

    model_class.fit(X_train.toarray(), y_train)

    return model_class