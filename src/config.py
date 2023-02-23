import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class PATHS:
    ROOT_DIR = Path(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
    DATA_DIR = ROOT_DIR / "data"
    MODEL_DIR = ROOT_DIR / "models"
    MISC_DIR = ROOT_DIR / "misc"


class MLCONFIG:
    RANDOM_STATE = 123
    CV_SPLIT = StratifiedShuffleSplit(
        n_splits=5, test_size=0.2, train_size=0.8, random_state=RANDOM_STATE
    )
    BASE_SCORER = {"AUC": "roc_auc_ovr", "F_score": "f1_weighted"}
    HYPERPARAMETERS = {
        "log_reg_param_grid": {
            "clf": [LogisticRegression(max_iter=1000, solver="liblinear")],
            # regularization param: higher C = less regularization
            "clf__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            # specifies kernel type to be used
            "clf__penalty": ["l1", "l2"],
        },
        "random_forest_param_grid": {
            "clf": [RandomForestClassifier()],
            # select whe
            "clf__criterion": ["gini", "entropy"],
            # num of base learners
            "clf__n_estimators": np.arange(100, 500, 100),
            # max tree depth of base learner
            "clf__max_depth": np.arange(50, 80, 10),
            # max num of features per subset
            "clf__max_features": ["log2", "sqrt"],
            "clf__bootstrap": [True, False],
        },
    }


class QUERY:
    RAW_DATA = """
      SELECT
        *
      FROM "smu-iot"."microbit"
      """


def print_config() -> None:
    print(
        f"""
    Parameters
    --------------------
    ROOT_DIR: {PATHS.ROOT_DIR}
    DATA_DIR: {PATHS.DATA_DIR}
    MODEL_DIR: {PATHS.MODEL_DIR}
    MISC_DIR: {PATHS.MISC_DIR}

    RANDOM_STATE: {MLCONFIG.RANDOM_STATE}
    CV_SPLIT: {MLCONFIG.CV_SPLIT}
    BASE_SCORER: {MLCONFIG.BASE_SCORER}
    HYPERPARAMETERS: {MLCONFIG.HYPERPARAMETERS}

    QUERY RAW DATA: {QUERY.RAW_DATA}
    """
    )


if __name__ == "__main__":
    print_config()
