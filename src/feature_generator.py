import pandas as pd
import numpy as np
import awswrangler as wr
from src.config import MLCONFIG, PATHS
from sklearn.base import BaseEstimator, TransformerMixin
import numpy.typing as npt
from typing import List, Tuple, Dict
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh import extract_features
from scipy.signal import savgol_filter, medfilt


# instantiate class wrapper to perform data transformations
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        label_encoding_map: Dict = MLCONFIG.LABEL_ENCODING_MAP,
        steps_count_features: List = MLCONFIG.STEPS_COUNT_FEATURES,
        base_features: List = MLCONFIG.BASE_FEATURES,
        top_features: List = MLCONFIG.TOP_FEATURES,
        **kwargs,
    ) -> None:
        self.label_encoding_map = (
            label_encoding_map  # {"walk": 1, "dynamic": 2, "box": 2}
        )
        self.base_features = base_features
        self.post_feature_names = [f"{col}_post" for col in base_features]
        self.top_features = top_features
        self.steps_count_features = steps_count_features

        self.upload_to_s3 = False
        self.apply_smooth_filter = True
        self.apply_median_filter = False
        self.apply_savgol_filter = True
        self.extract_features = True
        self.window_duration = 4
        self.step_seconds = 0.07

        allowed_keys = list(self.__dict__.keys())
        self.__dict__.update(
            (key, value) for key, value in kwargs.items() if key in allowed_keys
        )

    def print_attributes(self):
        print(f"Class attributes for {self}:")
        for name, val in vars(self).items():
            print(f" - {name}: {val}")

    def fit(self, X: pd.DataFrame, y: npt.ArrayLike = None):
        return self

    def transform(
        self, X: pd.DataFrame, y: npt.ArrayLike = None
    ) -> Tuple[pd.DataFrame]:
        """
        Data transformation pipeline

        Args:
            X: Input DataFrame with base features
            window_duration: Time window used for feature extraction (seconds)

        Returns:
            X, y: Input features, multi-class label

        """

        X_eng = X.copy()

        # apply label encoding

        X_eng["target_label"] = X_eng["uuid"].str.split("_").str[1]
        X_eng["target_label"] = (
            X_eng["target_label"].map(self.label_encoding_map).fillna(0)
        )

        # apply smoothing
        if self.apply_smooth_filter:
            print("Applying smoothing")
            X_eng = self._smooth_signal(
                X_eng=X_eng, apply_median_filter=True, apply_savgol_filter=True
            )

        # feature extraction
        if self.extract_features:
            X_eng = self._extract_features(X_eng)

        # upload to S3
        if self.upload_to_s3:
            print("Uploading to S3")
            wr.s3.to_csv(
                df=X_eng,
                path=f"s3://{PATHS.INTERIM}/{X_eng['timestamp'].max()}.csv",
                index=False,
            )

        return X_eng[self.top_features], X_eng["target_label"]

    def _smooth_signal(
        self,
        X_eng: pd.DataFrame,
        apply_median_filter: bool = True,
        apply_savgol_filter: bool = True,
    ) -> pd.DataFrame:
        """
        Apply smooth filter to remove spike or noisy data points

        Args:
            X_eng (pd.DataFrame): Input dataframe with raw features
            apply_median_filter (bool): bool flag whether to remove spike with median filter
            apply_savgol_filter (bool): bool flag whether to remove noise with Savitzky-Golay filter

        Returns:
            X_eng: pd.DataFrame

        """
        ts_id = 0
        X_eng_post = X_eng.copy().sort_index().reset_index(drop=True)

        self.window_n = int(self.window_duration // self.step_seconds)
        self.savgol_window_length = min(20, self.window_n)

        for item in X_eng_post["uuid"].unique():
            start_idx = X_eng_post.query(f"uuid == '{item}'").index[0]
            end_idx = X_eng_post.query(f"uuid == '{item}'").index[-1]

            for idx in range(start_idx, end_idx, self.window_n):
                next_idx = min(idx + self.window_n - 1, end_idx)
                if apply_median_filter:
                    X_eng_post.loc[idx:next_idx, self.post_feature_names] = (
                        X_eng_post.loc[idx:next_idx, self.base_features]
                        .apply(medfilt, axis=0)
                        .values
                    )
                if apply_savgol_filter:
                    window_length = min(next_idx - idx, self.savgol_window_length)
                    polyorder = window_length // 2
                    X_eng_post.loc[idx:next_idx, self.post_feature_names] = (
                        X_eng_post.loc[idx:next_idx, self.base_features]
                        .apply(
                            savgol_filter,
                            window_length=window_length,
                            polyorder=polyorder,
                            axis=0,
                        )
                        .values
                    )

                X_eng_post.loc[idx:next_idx, "ts_id"] = ts_id
                ts_id += 1

        return X_eng_post

    def _extract_features(self, X_eng: pd.DataFrame) -> pd.DataFrame:
        """
        Apply time-series feature extraction

        Args:
            X_eng (pd.DataFrame): Input pd.DataFrame() with smooth filter applied to base features

        Returns:
            X_eng (pd.DataFrame): Input data + top-50 important features + `number of walking steps estimated`

        """

        features = extract_features(
            timeseries_container=X_eng[["seconds", "ts_id", *self.post_feature_names]],
            column_id="ts_id",
            column_sort="seconds",
        )
        features["n_steps"] = (
            features[self.steps_count_features].median(axis=1).astype(np.int)
        )
        X_eng_post = (
            X_eng.groupby(["ts_id"])
            .max()
            .join(
                other=features[np.append(self.top_features, ["n_steps"])],
                on=["ts_id"],
            )
        )

        return X_eng_post


if __name__ == "__main__":
    fe_settings = {
        "upload_to_s3": False,
        "apply_smooth_filter": True,
        "apply_median_filter": False,
        "apply_savgol_filter": True,
        "extract_features": True,
        "window_duration": 4,
        "step_seconds": 0.07,
    }
    fe = FeatureEngineering(**fe_settings)

    fe.print_attributes()

    df = pd.read_csv(PATHS.DATA_DIR / "test.csv")
    X, y = df, df.uuid
    X_proc, y_proc = fe.fit_transform(X, y)
    print(len(X_proc), len(y_proc))
