import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    QuantileTransformer,
)


class PATHS:
    ROOT_DIR = Path(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
    DATA_DIR = ROOT_DIR / "data"
    MODEL_DIR = ROOT_DIR / "models"
    MISC_DIR = ROOT_DIR / "misc"
    BUCKET = "smu-is614-iot-step-tracker"
    RAW = f"{BUCKET}/raw"
    QUERIES = f"{BUCKET}/queries"
    MODELS = f"{BUCKET}/models"
    PROCESSED = f"{BUCKET}/processed"
    RESULT = f"{BUCKET}/inference/result"
    INTERIM = f"{BUCKET}/inference/interim"
    DB_TEST = "smu-iot"


class MLCONFIG:
    RANDOM_STATE = 123
    CV_SPLIT = StratifiedShuffleSplit(
        n_splits=5, test_size=0.2, train_size=0.8, random_state=RANDOM_STATE
    )
    BASE_SCORER = {"AUC": "roc_auc_ovr", "F_score": "f1_weighted"}
    HYPERPARAMETERS = {
        "LogisticRegression": {
            "clf": [LogisticRegression(max_iter=1000, solver="liblinear")],
            # regularization param: higher C = less regularization
            "clf__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            # specifies kernel type to be used
            "clf__penalty": ["l1", "l2"],
        },
        "RandomForest": {
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

    SCALERS = {
        # Scales each feature to the range of 0 and 1
        "MinMax": MinMaxScaler(feature_range=(0, 1)),
        # Standardize features by removing the mean and scaling to unit variance.
        #   z = (x - u) / s
        "Standard": StandardScaler(),
        # Transform features using quantile information to follow a normal distribution
        "Quantile": QuantileTransformer(
            n_quantiles=1000,
            output_distribution="normal",
            random_state=RANDOM_STATE,
        ),
    }

    STEP_SECONDS = 0.07
    BASE_FEATURES = [
        "accel_x",
        "gyro_x",
        "accel_y",
        "gyro_y",
        "accel_z",
        "gyro_z",
    ]
    LABEL_ENCODING_MAP = {"walk": 1, "dynamic": 2, "box": 2}
    STEPS_COUNT_FEATURES = [
        "gyro_x_post__number_peaks__n_5",
        "gyro_y_post__number_peaks__n_5",
        "gyro_z_post__number_peaks__n_5",
    ]
    TOP_FEATURES = [
        "accel_x_post__maximum",
        "accel_y_post__count_above__t_0",
        "accel_y_post__quantile__q_0.2",
        "accel_x_post__abs_energy",
        'gyro_y_post__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.2',
        "gyro_y_post__ratio_beyond_r_sigma__r_0.5",
        # "accel_y_post__autocorrelation__lag_4",
        'accel_x_post__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.2',
        "gyro_z_post__lempel_ziv_complexity__bins_10",
        "gyro_z_post__lempel_ziv_complexity__bins_3",
        # 'gyro_x_post__fft_coefficient__attr_"abs"__coeff_5',
        "accel_y_post__minimum",
        "accel_x_post__absolute_maximum",
        # "accel_z_post__autocorrelation__lag_5",
        # "accel_x_post__mean_n_absolute_max__number_of_maxima_7",
        "gyro_y_post__percentage_of_reoccurring_datapoints_to_all_datapoints",
        'accel_x_post__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.0',
        "accel_y_post__lempel_ziv_complexity__bins_5",
        "gyro_z_post__large_standard_deviation__r_0.25",
        "accel_y_post__quantile__q_0.3",
        "accel_y_post__longest_strike_below_mean",
        'gyro_z_post__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.4',
        'accel_y_post__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.2',
        'accel_z_post__linear_trend__attr_"pvalue"',
        "gyro_x_post__sum_values",
        'gyro_z_post__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.2',
        "gyro_y_post__ratio_beyond_r_sigma__r_3",
        "gyro_z_post__kurtosis",
        "accel_y_post__permutation_entropy__dimension_3__tau_1",
        # "gyro_x_post__cwt_coefficients__coeff_7__w_10__widths_(2, 5, 10, 20)",
        # 'accel_x_post__fft_coefficient__attr_"abs"__coeff_3',
        "accel_y_post__binned_entropy__max_bins_10",
        'gyro_z_post__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.0',
        "accel_y_post__ratio_beyond_r_sigma__r_0.5",
        'accel_x_post__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.4',
        # 'accel_x_post__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"min"',
        "accel_z_post__spkt_welch_density__coeff_2",
        "accel_z_post__longest_strike_below_mean",
        "accel_y_post__cid_ce__normalize_True",
        'gyro_z_post__fft_coefficient__attr_"abs"__coeff_1',
        'gyro_x_post__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.4',
        "gyro_x_post__lempel_ziv_complexity__bins_2",
        "accel_y_post__approximate_entropy__m_2__r_0.5",
        "gyro_z_post__ratio_beyond_r_sigma__r_1",
        "accel_y_post__energy_ratio_by_chunks__num_segments_10__segment_focus_6",
        "gyro_x_post__lempel_ziv_complexity__bins_5",
        "accel_y_post__quantile__q_0.1",
        "gyro_x_post__count_below__t_0",
        "gyro_z_post__lempel_ziv_complexity__bins_100",
        "gyro_z_post__approximate_entropy__m_2__r_0.5",
    ]


class KEYS:
    AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
    AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
    AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]


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
    BUCKET: {PATHS.BUCKET}
    RAW: {PATHS.RAW}
    QUERIES: {PATHS.QUERIES}
    MODELS: {PATHS.MODELS}
    PROCESSED: {PATHS.PROCESSED}

    RANDOM_STATE: {MLCONFIG.RANDOM_STATE}
    CV_SPLIT: {MLCONFIG.CV_SPLIT}
    BASE_SCORER: {MLCONFIG.BASE_SCORER}
    HYPERPARAMETERS: {MLCONFIG.HYPERPARAMETERS}

    QUERY RAW DATA: {QUERY.RAW_DATA}
    """
    )


if __name__ == "__main__":
    print_config()
