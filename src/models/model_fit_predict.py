import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

# logging
import logging

# types
from typing import Dict, Union

from src.entity.train_params import TrainingParams

SklearnClassificationModel = Union[LogisticRegression, RandomForestClassifier]


logger = logging.getLogger(__name__)

def train_model(features: pd.DataFrame, target: pd.Series, train_params: TrainingParams) -> SklearnClassificationModel:
    
    logger.info('Start loading model...')

    if train_params.model_type == 'LogisticRegression':
        model = LogisticRegression()
    else:
        logger.exception('Selected model is incorrect')
        raise NotImplementedError()

    logger.info('Finished loading model!')

    logger.info('Start model fitting...')

    model.fit(features, target)

    logger.info('Finished model fitting!')

    return model


def predict_model(model: Pipeline, feature: pd.DataFrame) -> np.ndarray:

    logger.info('Start model predict...')

    predict = model.predict(feature)

    logger.info('Finished model predict!')

    return predict

def evaluate_model(predict: np.ndarray, target: pd.Series) -> Dict[str, float]:

    logger.info('Start calculate metrics for model')

    return {
        'acc': accuracy_score(target.toarray(), predict),
        'f1': f1_score(target.toarray(), predict, average='macro'),
        'roc_auc': roc_auc_score(target.toarray(), predict),
    }