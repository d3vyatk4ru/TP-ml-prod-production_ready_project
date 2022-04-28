import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

import logging

from src.entity.feature_params import FeatureParams


logger = logging.getLogger(__name__)


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:

    logger.info('Start categorical pipeline...')

    categorical_pipeline = build_categorical_pipeline()

    categorical_df = categorical_pipeline.fit_transform(categorical_df).toarray()

    logger.info('Finished categorical pipeline!')

    return pd.DataFrame(categorical_df)


def build_categorical_pipeline() -> Pipeline:
    ''' Make one hot encoding for categorical features '''

    categorical_pipeline = Pipeline([
            ('OHE', OneHotEncoder()),
        ])

    return categorical_pipeline

def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    ''' Get target column from dataset'''
    return df[params.target_col]

def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    ''' Make transform with input pd.DataFrame'''
    return transformer.transform(df)

def drop_target(df: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    ''' Delete target column from pf.DataFrame '''
    df = df.drop(columns=[params.target_col])
    return df

def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer([
        (
            'categorical_pipeline',
            build_categorical_pipeline(),
            params.categorical_features,
        )
    ])

    return transformer
