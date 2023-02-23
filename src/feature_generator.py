from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


# instantiate class wrapper to perform data transformations
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame):
        pass

    def transform(self, X: pd.DataFrame):
        X_eng = X.copy()
        # processing here

        return X_eng


if __name__ == "__main__":
    feature_eng = FeatureEngineering()
