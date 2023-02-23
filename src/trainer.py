from src.model import BaseModel
from typing import Any
from pathlib import Path
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import mlflow
import mlflow.sklearn
from typing import Any, Dict, Union
from pathlib import Path
from src.config import MLCONFIG
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


class Trainer:
    def __init__(
        self,
        scaler: Union[MinMaxScaler, StandardScaler] = MinMaxScaler(
            feature_range=(0, 1)
        ),
        hyperparam_space: Dict = MLCONFIG.HYPERPARAMETERS.get("log_reg_param_grid"),
        model_config: BaseModel = BaseModel(),
        eval_config: Dict = MLCONFIG.BASE_SCORER,
        cv_splitter: Any = MLCONFIG.CV_SPLIT,
    ) -> None:
        self.model_config = model_config
        self.eval_config = eval_config
        self.hyperparam_space = hyperparam_space
        self.scaler = scaler
        self.pipe = Pipeline([("slr", self.scaler), ("clf", self.model_config)])
        # To edit
        # self.results = "results"

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

    def fit(self, X, y):
        self.grid_search_cv.fit(X, y)
        return self

    def evaluate(self):
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
    scaler = MinMaxScaler(feature_range=(0, 1))
    hyperparams = MLCONFIG.HYPERPARAMETERS.get("log_reg_param_grid")
    eval_config = MLCONFIG.BASE_SCORER
    cv_splitter = MLCONFIG.CV_SPLIT

    train_run = Trainer(
        model_config=mdl,
        scaler=scaler,
        hyperparam_space=hyperparams,
        eval_config=eval_config,
        cv_splitter=cv_splitter,
    )

    print(repr(train_run))
