from pathlib import Path
from typing import Dict, Tuple
import numpy.typing as npt
from sklearn.base import TransformerMixin
from sklearn.model_selection import BaseShuffleSplit
import joblib
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
import pandas as pd

from src.config import MLCONFIG
from src.model import BaseModel


class Trainer:
    def __init__(
        self,
        scaler: TransformerMixin = MLCONFIG.SCALERS.get("Quantile"),
        hyperparam_space: Dict = MLCONFIG.HYPERPARAMETERS.get("LogisticRegression"),
        model_config: BaseModel = BaseModel(),
        eval_config: Dict = MLCONFIG.BASE_SCORER,
        cv_splitter: BaseShuffleSplit = MLCONFIG.CV_SPLIT,
    ) -> None:
        self.model_config = model_config
        self.eval_config = eval_config
        self.hyperparam_space = hyperparam_space
        self.scaler = scaler
        self.pipe = Pipeline([("scl", self.scaler), ("clf", self.model_config)])
        self.grid_search_cv = GridSearchCV(
            estimator=self.pipe,
            param_grid=self.hyperparam_space,
            scoring=eval_config,
            refit="AUC",
            cv=cv_splitter,
            return_train_score=True,
            n_jobs=-1,
            verbose=10,
        )

    def __repr__(self) -> str:
        return f"""
        {self.__class__.__name__} (model_config={self.model_config}, eval_config={self.eval_config}, hyperparam_space={self.hyperparam_space})
        """

    def fit(self, X: pd.DataFrame, y: npt.ArrayLike):
        self.grid_search_cv.fit(X, y)
        return self

    def evaluate(self) -> Tuple:
        best_params = self.grid_search_cv.best_params_
        best_est = self.grid_search_cv.best_estimator_
        best_score = self.grid_search_cv.best_score_
        return best_params, best_est, best_score

    # def export_best_model(self, dir_path: Path, name: str) -> None:
    #     """
    #     Exports a trained model to the given directory.

    #     Parameters
    #     ----------
    #     model : The trained model to be exported
    #     dir_path : Target directory
    #     name : File name to save the model as
    #     """

    #     path = f"{dir_path}/{name}"
    #     joblib.dump(self.model_config, path)

    #     ### To change
    #     print(f"Model saved to 'path'")

    def predict(self):
        # To edit
        pass


if __name__ == "__main__":
    mdl = BaseModel()
    scaler = MLCONFIG.SCALERS.get("Quantile")
    hyperparams = MLCONFIG.HYPERPARAMETERS.get("LogisticRegression")
    eval_config = MLCONFIG.BASE_SCORER
    cv_splitter = MLCONFIG.CV_SPLIT

    train_run = Trainer(
        model_config=mdl,
        scaler=scaler,
        hyperparam_space=hyperparams,
        eval_config=eval_config,
        cv_splitter=cv_splitter,
    )

    # load some sample data
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data[:, :2], columns=["a", "b"])
    y = iris.target
    train_run.fit(X, y)
    print(repr(train_run))
