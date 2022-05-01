""" Make custom transformer """

import logging

from sklearn.preprocessing import StandardScaler
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)

# from ..entity.custom_transformer_params import TransformerParams


logger = logging.getLogger(__name__)


class CustomTransformer(BaseEstimator, TransformerMixin):
    """ Class realization custom version of transformer """

    def __init__(self,
                 params,
                 ) -> None:
        self.params = params
        self.scaler = StandardScaler()

    # def fit(data: pd.DataFrame, )
