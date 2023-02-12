import os
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit


class PATHS:
    ROOT_DIR = Path(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
    DATA_DIR = ROOT_DIR / "data"
    MODEL_DIR = ROOT_DIR / "models"
    MISC_DIR = ROOT_DIR / "misc"


class MLCONFIG:
    RANDOM_STATE = 123
    CV_SPLIT = TimeSeriesSplit(n_splits=5)


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
    """
    )


if __name__ == "__main__":
    print_config()
