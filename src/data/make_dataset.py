import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.entity.split_params import SplittingParams


logger = logging.getLogger(__name__)


def read_dataset(path: str) -> pd.DataFrame:
    ''' Read dataset from csv file '''

    logger.info(f'Loading dataset from {path}...')

    data = pd.read_csv(path)

    logger.info(f'Finished loading dataset from {path}!')
    logger.info(f'The dataset has {data.shape} size')

    return data


def split_train_val_data(data: pd.DataFrame, params: SplittingParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ''' Split data to train and validation '''

    logger.info('Splitting dataset to train and test...')

    train_data, test_data = train_test_split(
        data,
        test_size=params.test_size,
        random_state=params.random_state,
    )

    logger.info('Finished splitting dataset!')

    return (train_data, test_data)
