import pandas as pd
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class BaseModel(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator: BaseEstimator = None) -> None:
        self.estimator = estimator

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: {repr(self.estimator)})"

    def fit(self, X, y: npt.ArrayLike = None) -> BaseEstimator:
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.estimator.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> npt.ArrayLike:
        check_is_fitted(self)
        X = check_array(X)
        return self.estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> npt.ArrayLike:
        check_is_fitted(self)
        X = check_array(X)
        return self.estimator.predict_proba(X)

    def score(self, X: pd.DataFrame, y: npt.ArrayLike) -> float:
        check_is_fitted(self)
        X = check_array(X)
        return self.estimator.score(X, y)

    def summary(self) -> None:
        model_params = self.estimator.get_params()
        print(repr(self))
        for param, value in model_params.items():
            print(f"\t{param}: {value}")


def main():
    mdl = BaseModel(LogisticRegression())
    mdl.summary()


if __name__ == "__main__":
    main()
