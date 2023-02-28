import pandas as pd
from dataloader import DataLoader
from config import MLCONFIG
from sklearn.base import BaseEstimator, TransformerMixin
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh import extract_features, feature_extraction
from scipy.signal import savgol_filter, medfilt
from scipy import fftpack


# instantiate class wrapper to perform data transformations
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.POST_FEATURE_NAMES = [c + "_post" for c in MLCONFIG.BASE_FEATURES]
        self.WINDOW_N = MLCONFIG.WINDOW_N
        self.window_length = 20

    def fit(self, X: pd.DataFrame):
        pass

    def transform(self, X: pd.DataFrame, **kwargs):
        X_eng = X.copy()
        # processing here
        # apply smoothing
        X_eng = self._smooth_signal(X_eng, **kwargs)

        return X_eng

    def _time_window(self, interval: int = 5):
        return interval

    def _fourier_transform(self):  # required args for fft here
        pass

    def _smooth_signal(
        self,
        X_eng: pd.DataFrame,
        apply_median: bool = True,
        apply_sg: bool = True,
    ) -> pd.DataFrame:
        """
        Args:
            X_eng: pd.DataFrame
            apply_median: bool flag whether to remove spike with median filter
            apply_sg: bool flag whether to remove noise with Savitzky-Golay filter

        Returns:
            X_eng: pd.DataFrame

        """
        ts_id = 0
        start_idx, end_idx = X_eng.index[0], X_eng.index[-1]

        for idx in range(start_idx, end_idx, self.WINDOW_N):
            next_idx = min(idx + self.WINDOW_N - 1, end_idx)
            if apply_median:
                X_eng.loc[idx:next_idx, self.POST_FEATURE_NAMES] = (
                    X_eng.loc[idx:next_idx, MLCONFIG.BASE_FEATURES]
                    .apply(medfilt, axis=0)
                    .values
                )
            if apply_sg:
                window_length = min(next_idx - idx, self.window_length)
                polyorder = window_length // 2
                X_eng.loc[idx:next_idx, self.POST_FEATURE_NAMES] = (
                    X_eng.loc[idx:next_idx, MLCONFIG.BASE_FEATURES]
                    .apply(
                        savgol_filter,
                        window_length=window_length,
                        polyorder=polyorder,
                        axis=0,
                    )
                    .values
                )

            X_eng.loc[idx:next_idx, "ts_id"] = ts_id
            ts_id += 1

        return X_eng


if __name__ == "__main__":
    dataloader = DataLoader()
    QUERY = """
        SELECT
            *,
            case when uuid like '%_walk_%' then true else false end as target
        FROM
            "smu-iot"."microbit"
        WHERE
            seconds IS NOT null and uuid = 'songhan_walk_1'
        ORDER BY
            uuid, timestamp, seconds
    """
    df = dataloader.load_data(QUERY, "smu-iot")
    feature_eng = FeatureEngineering()
    print(feature_eng.transform(df))
