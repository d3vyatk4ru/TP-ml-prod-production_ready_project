import logging
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)

from src.entity.custom_transformer_params import TransformerParams


logger = logging.getLogger(__name__)


class CustomTransformes(BaseEstimator, TransformerMixin):
    ''' Class realization custom version of transfomer '''

    def __init__(self, params: TransformerParams) -> None:
        self.params = params
        self.scaler = StandardScaler()

    # def fit(data: pd.DataFrame, )