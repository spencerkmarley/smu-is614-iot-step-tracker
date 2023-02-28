from typing import Dict, Tuple
import numpy.typing as npt
from sklearn.base import TransformerMixin
from sklearn.model_selection import BaseShuffleSplit
from datetime import datetime
from src.config import PATHS
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
import pandas as pd
import argparse

from src.config import MLCONFIG
from src.model import BaseModel
from src.dataloader import DataLoader
from src.feature_generator import FeatureEngineering

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
        {self.__class__.__name__}
        \tmodel_config={self.model_config}
        \teval_config={self.eval_config}
        \thyperparam_space={self.hyperparam_space}
        """

    def fit(self, X: pd.DataFrame, y: npt.ArrayLike):
        self.grid_search_cv.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> npt.ArrayLike:
        return self.grid_search_cv.predict(X)

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


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base", type=str, default="lr", required=False, choices=["lr", "rf"]
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="mm",
        required=False,
        choices=["mm", "ss", "qt"],
    )
    parser.add_argument(
        "--window", type=float, default=4, required=False
    )

    args = parser.parse_args()
    scaler = {"mm": "MinMax", "ss": "Standard", "qt": "Quantile"}
    base = {"lr": "LogisticRegression", "rf": "RandomForest"}

    # load data via athena
    dataloader = DataLoader()
    # TODO: To revise this
    QUERY = """
        SELECT
            *,
            case when uuid like '%_walk_%' then true else false end as target
        FROM
            "smu-iot"."microbit"
        WHERE
            seconds IS NOT null AND uuid = 'songhan_walk_1'
        ORDER BY
            uuid, timestamp, seconds    
    """
    df = dataloader.load_data(QUERY, 'smu-iot')
    feature_eng = FeatureEngineering()
    X, y = feature_eng.transform(df)
    ID = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up experiment
    mlflow.set_tracking_uri(f"{PATHS.ROOT_DIR}/mlflow/mlruns")
    print(f"{PATHS.ROOT_DIR}/mlflow/mlruns")
    experiment = mlflow.set_experiment(experiment_name="ml_experiments")

    with mlflow.start_run(
        experiment_id=experiment.experiment_id, run_name=f"model_{ID}"
    ):
        MODEL = BaseModel()
        SCALER = MLCONFIG.SCALERS.get(scaler.get(args.scaler))
        HYP = MLCONFIG.HYPERPARAMETERS.get(base.get(args.base))
        trainer = Trainer(model_config=MODEL, scaler=SCALER, hyperparam_space=HYP)

        trainer.fit(X, y)
        best_params, best_est, best_score = trainer.evaluate()

        mlflow.log_params(params=best_params)
        mlflow.log_metric(key="AUC", value=best_score)
        mlflow.set_tags(
            tags={
                "Base Model": base.get(args.base),
                "Scaler": scaler.get(args.scaler),
            }
        )
        mlflow.sklearn.log_model(best_est, "model")
