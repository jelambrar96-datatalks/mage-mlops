from typing import Tuple, Dict, Union, Callable

import pandas as pd
from pandas import Series, DataFrame

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
    data: Dict[str, Union[DataFrame, csr_matrix]], **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_on_feature = kwargs.get('split_on_feature', 'tpep_pickup_location')
    split_on_feature_value = kwargs.get('split_on_feature_value', '2024-03-01')
    target = kwargs.get('target', 'duration')

    [df1] = data['data_ingest_export']



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

    print(type(y_train))
    print(type(X_train))


    model_class = LinearRegression()
    # model_class.fit(df_train, y_train)

    model_class.fit(X_train.toarray(), y_train)

    return X_train, y_train, dv, model_class