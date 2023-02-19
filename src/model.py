import pandas as pd
import warnings
from datetime import datetime
from typing import List, Optional, Tuple

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from src.config import PATHS, MLCONFIG
from sklearn.metrics import roc_auc_score


# Setup the hyperparameter grid
log_reg_param_grid = {
    # regularization param: higher C = less regularization
    "log_reg__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    # specifies kernel type to be used
    "log_reg__penalty": ["l1", "l2", "none"],
}

mm_scale = MinMaxScaler(feature_range=(0, 1))
model_types = {"Logistic Regression": LogisticRegression()}


class BaseModel(BaseEstimator):
    def __init__(
        self,
        model_type="Logistic Regression",
        scaler=mm_scale,
        params: dict = None,
    ) -> None:
        self.model_type = model_type
        self.model = model_types.get("Logistic Regression")

        if params is None:
            params = log_reg_param_grid

        self.params = params

        self.scaler = scaler

    def __repr__(self):
        return f"{self.__class__.__name__} (model_type={self.model_type})"

    # def get_params(self, deep=True) -> dict:
    #     return {
    #         "model_type": self.model_type,
    #         "params": self.params,
    #         "scaler": self.scaler,
    #     }

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.set_params(**self.params)
        self.model.fit(X_scaled, y)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def score(self, X, y):
        X_scaled = self.scaler.transform(X)
        return roc_auc_score(y, self.model.predict_proba(X_scaled)[:, 1])

    def summary(self):
        model_params = self.get_params()
        model_name = model_params.pop("model_type")
        scaler_name = str(type(self.scaler)).split(".")[-1][:-2]

        print(f"{model_name} ({scaler_name} scaler) model parameters:")
        for param, value in model_params.items():
            print(f"\t{param}: {value}")


if __name__ == "__main__":
    mdl = BaseModel(model_type="Logistic Regression")
    print(repr(mdl))

    ### test methods
    print(f"Starting params CHECK")
    for param, value in mdl.get_params(deep=True).items():
        print(f"{param} : {value}")

    # mdl.summary()
