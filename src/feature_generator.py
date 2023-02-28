import pandas as pd
import numpy as np
import awswrangler as wr
from dataloader import DataLoader
from config import MLCONFIG, PATHS
from sklearn.base import BaseEstimator, TransformerMixin
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh import extract_features, feature_extraction
from scipy.signal import savgol_filter, medfilt
from scipy import fftpack

# instantiate class wrapper to perform data transformations
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.POST_FEATURE_NAMES = [c + '_post' for c in MLCONFIG.BASE_FEATURES]
        self.TOP_FEATURES = MLCONFIG.TOP_FEATURES
        self.STEP_COUNT_FEATURES = MLCONFIG.STEPS_COUNT_FEATURES

    def fit(self, X: pd.DataFrame):
        pass

    def transform(self, X: pd.DataFrame,
                  window_duration=4,
                  **kwargs) -> (pd.DataFrame, pd.DataFrame):
        """
        Data transformation pipeline

        Args:
            X: Input DataFrame with base features
            window_duration: Time window used for feature extraction (seconds)
            **kwargs:

        Returns:
            X, y: Input features, multi-class label

        """


        self.WINDOW_N = int(window_duration // MLCONFIG.STEP_SECONDS)
        self.sg_window_length = min(20, self.WINDOW_N)

        X_eng = X.copy()
        # processing here
        # apply label encoding
        X_eng['target_label'] = X_eng['uuid'].apply(lambda x: MLCONFIG.LABEL_ENCODING_MAP.get(x.split('_')[1], 0))
        # apply smoothing
        X_eng = self._smooth_signal(X_eng, **kwargs)
        # feature extraction
        X_eng = self._extract_features(X_eng)
        # upload to S3
        wr.s3.to_csv(X_eng,
                     f"s3://{PATHS.INTERIM}/{X_eng.timestamp.max()}.csv",
                     index=False)


        return X_eng[self.TOP_FEATURES], X_eng[['target_label']]

    def _smooth_signal(self,
                       X_eng: pd.DataFrame,
                       apply_median=True,
                       apply_sg=True) -> pd.DataFrame:
        """
        Apply smooth filter to remove spike or noisy data points

        Args:
            X_eng: pd.DataFrame
            apply_median: bool flag whether to remove spike with median filter
            apply_sg: bool flag whether to remove noise with Savitzky-Golay filter

        Returns:
            X_eng: pd.DataFrame

        """
        ts_id = 0

        for item in X_eng.uuid.unique():
            start_idx = X_eng.query(f"uuid == '{item}'").index[0]
            end_idx = X_eng.query(f"uuid == '{item}'").index[-1]

            for idx in range(start_idx, end_idx, self.WINDOW_N):
                next_idx = min(idx + self.WINDOW_N - 1, end_idx)
                if apply_median:
                    X_eng.loc[idx:next_idx, self.POST_FEATURE_NAMES] = \
                    X_eng.loc[idx:next_idx, MLCONFIG.BASE_FEATURES].apply(medfilt, axis=0).values
                if apply_sg:
                    window_length = min(next_idx - idx, self.sg_window_length)
                    polyorder = window_length // 2
                    X_eng.loc[idx:next_idx, self.POST_FEATURE_NAMES] = \
                    X_eng.loc[idx:next_idx, MLCONFIG.BASE_FEATURES]\
                        .apply(savgol_filter, window_length=window_length, polyorder=polyorder, axis=0).values

                X_eng.loc[idx:next_idx, 'ts_id'] = ts_id
                ts_id += 1

        return X_eng

    def _extract_features(self,
                          X_eng: pd.DataFrame
                          ) -> pd.DataFrame:
        """
        Apply time-series feature extraction

        Args:
            X_eng: Input pd.DataFrame() with smooth filter applied to base features

        Returns:
            X_eng: Input data + top-50 important features + `number of walking steps estimated`

        """

        features = extract_features(X_eng[['seconds', 'ts_id', *self.POST_FEATURE_NAMES]],
                                    column_id='ts_id',
                                    column_sort='seconds')
        features['n_steps'] = features[self.STEP_COUNT_FEATURES].median(axis=1).astype(np.int)
        X_eng_post = X_eng.groupby(['ts_id']).max().join(features[np.append(self.TOP_FEATURES, ['n_steps'])], on=['ts_id'])

        return X_eng_post

if __name__ == "__main__":
    dataloader = DataLoader()
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
