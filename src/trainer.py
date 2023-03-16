import boto3
import numpy.typing as npt
from typing import Dict, Tuple
from sklearn.base import TransformerMixin
from sklearn.model_selection import BaseShuffleSplit
from datetime import datetime
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd
import argparse

from src.config import MLCONFIG, KEYS
from src.model import BaseModel
from src.dataloader import DataLoader
from src.feature_generator import FeatureEngineering


class Trainer:
    def __init__(
        self,
        scaler: TransformerMixin = MLCONFIG.SCALERS.get("Standard"),
        hyperparam_space: Dict = MLCONFIG.HYPERPARAMETERS.get("LogisticRegression"),
        model_config: BaseModel = BaseModel(),
        eval_config: Dict = MLCONFIG.BASE_SCORER,
        cv_splitter: BaseShuffleSplit = MLCONFIG.CV_SPLIT,
    ) -> None:
        self.model_config = model_config
        self.eval_config = eval_config
        self.hyperparam_space = hyperparam_space
        self.scaler = scaler
        self.pipe = Pipeline(
            [
                ("scl", self.scaler),
                ("clf", self.model_config),
            ]
        )
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


if __name__ == "__main__":
    # 1. Parse arguments
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
    parser.add_argument("--window", type=float, default=4, required=False)

    args = parser.parse_args()
    scaler = {"mm": "MinMax", "ss": "Standard", "qt": "Quantile"}
    base = {"lr": "LogisticRegression", "rf": "RandomForest"}

    # 2. Load data via athena
    session = boto3.setup_default_session(
        region_name=KEYS.AWS_DEFAULT_REGION,
        aws_access_key_id=KEYS.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=KEYS.AWS_SECRET_ACCESS_KEY,
    )
    dataloader = DataLoader(session=session)
     
    QUERY = """
        SELECT
            *,
            case when uuid like '%_walk_%' then true else false end as target
        FROM
            "smu-iot"."microbit"
        WHERE
            seconds IS NOT null
        ORDER BY
            uuid, timestamp, seconds
    """
    df = dataloader.load_data(QUERY, "smu-iot")

    # 3. Perform feature engineering
    fe_settings = {
        "upload_to_s3": False, # s3://smu-is614-iot-step-tracker/inference/interim/1678953483395.csv
        "apply_smooth_filter": True,
        "apply_median_filter": False,
        "apply_savgol_filter": True,
        "extract_features": True,
        "window_duration": 4,
        "step_seconds": 0.07,
    }
    FE = FeatureEngineering(**fe_settings)
    X, y = FE.fit_transform(df, df.uuid)


    # Set up experiment 
    # Local 
    # mlflow.set_tracking_uri(f"{PATHS.ROOT_DIR}/mlflow/mlruns")

    ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    mlflow.set_tracking_uri(
        "http://ec2-3-1-195-80.ap-southeast-1.compute.amazonaws.com:5000/"
    )
    experiment = mlflow.set_experiment(experiment_name="ml_experiments")

    with mlflow.start_run(
        experiment_id=experiment.experiment_id, run_name=f"model_{ID}"
    ):

        MODEL = BaseModel()
        SCALER = MLCONFIG.SCALERS.get(scaler.get(args.scaler))
        HYP = MLCONFIG.HYPERPARAMETERS.get(base.get(args.base))

        trainer = Trainer(
            model_config=MODEL,
            scaler=SCALER,
            hyperparam_space=HYP,
        )

        trainer.fit(X, y)
        best_params, best_est, best_score = trainer.evaluate()

        mlflow.log_params(params=best_params)
        mlflow.log_params(params=fe_settings)
        mlflow.log_metric(key="AUC", value=best_score)
        mlflow.set_tags(
            tags={
                "Base Model": base.get(args.base),
                "Scaler": scaler.get(args.scaler),
            }
        )
        mlflow.sklearn.log_model(best_est, "model")
