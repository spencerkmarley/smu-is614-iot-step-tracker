import pandas as pd
import warnings
from datetime import datetime
from typing import List, Optional, Tuple

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from src.config import PATHS, MLCONFIG


# Setup the hyperparameter grid
log_reg_param_grid = {
    # regularization param: higher C = less regularization
    "log_reg__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    # specifies kernel type to be used
    "log_reg__penalty": ["l1", "l2", "none"],
}

mm_scale = MinMaxScaler(feature_range=(0, 1))


class BaseModel(BaseEstimator):
    def __init__(self, model_type, params: dict = None) -> None:
        self.model_type = model_type

        if params is None:
            params = {}

        self.params = params

    def __repr__(self):
        return f"{self.__class__.__name__} (model_type={self.model_type})"

    def get_params(self):
        pass

    def fit(self):
        pass

    def predict_proba(self):
        pass

    def score(self, X, y):
        pass

    def summary(self):
        pass


if __name__ == "__main__":
    mdl = BaseModel(model_type="Logistic Regression")
    print(repr(mdl))
