import os
import boto3
from pathlib import Path
import numpy as np
from botocore.exceptions import ClientError
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

    FEATURE_SETTINGS = {'sum_values': None, 'abs_energy': None, 'kurtosis': None, 'longest_strike_below_mean': None,
                  'percentage_of_reoccurring_datapoints_to_all_datapoints': None, 'maximum': None,
                  'absolute_maximum': None, 'minimum': None, 'cid_ce': [{'normalize': True}, {'normalize': False}],
                  'large_standard_deviation': [{'r': 0.05}, {'r': 0.1}, {'r': 0.15000000000000002}, {'r': 0.2},
                                               {'r': 0.25}, {'r': 0.30000000000000004}, {'r': 0.35000000000000003},
                                               {'r': 0.4}, {'r': 0.45}, {'r': 0.5}, {'r': 0.55},
                                               {'r': 0.6000000000000001}, {'r': 0.65}, {'r': 0.7000000000000001},
                                               {'r': 0.75}, {'r': 0.8}, {'r': 0.8500000000000001}, {'r': 0.9},
                                               {'r': 0.9500000000000001}],
                  'quantile': [{'q': 0.1}, {'q': 0.2}, {'q': 0.3}, {'q': 0.4}, {'q': 0.6}, {'q': 0.7}, {'q': 0.8},
                               {'q': 0.9}], 'binned_entropy': [{'max_bins': 10}],
                  'spkt_welch_density': [{'coeff': 2}, {'coeff': 5}, {'coeff': 8}],
                  'change_quantiles': [{'ql': 0.0, 'qh': 0.2, 'isabs': False, 'f_agg': 'mean'},
                                       {'ql': 0.0, 'qh': 0.2, 'isabs': False, 'f_agg': 'var'},
                                       {'ql': 0.0, 'qh': 0.2, 'isabs': True, 'f_agg': 'mean'},
                                       {'ql': 0.0, 'qh': 0.2, 'isabs': True, 'f_agg': 'var'},
                                       {'ql': 0.0, 'qh': 0.4, 'isabs': False, 'f_agg': 'mean'},
                                       {'ql': 0.0, 'qh': 0.4, 'isabs': False, 'f_agg': 'var'},
                                       {'ql': 0.0, 'qh': 0.4, 'isabs': True, 'f_agg': 'mean'},
                                       {'ql': 0.0, 'qh': 0.4, 'isabs': True, 'f_agg': 'var'},
                                       {'ql': 0.0, 'qh': 0.6, 'isabs': False, 'f_agg': 'mean'},
                                       {'ql': 0.0, 'qh': 0.6, 'isabs': False, 'f_agg': 'var'},
                                       {'ql': 0.0, 'qh': 0.6, 'isabs': True, 'f_agg': 'mean'},
                                       {'ql': 0.0, 'qh': 0.6, 'isabs': True, 'f_agg': 'var'},
                                       {'ql': 0.0, 'qh': 0.8, 'isabs': False, 'f_agg': 'mean'},
                                       {'ql': 0.0, 'qh': 0.8, 'isabs': False, 'f_agg': 'var'},
                                       {'ql': 0.0, 'qh': 0.8, 'isabs': True, 'f_agg': 'mean'},
                                       {'ql': 0.0, 'qh': 0.8, 'isabs': True, 'f_agg': 'var'},
                                       {'ql': 0.0, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'},
                                       {'ql': 0.0, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'},
                                       {'ql': 0.0, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'},
                                       {'ql': 0.0, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'},
                                       {'ql': 0.2, 'qh': 0.4, 'isabs': False, 'f_agg': 'mean'},
                                       {'ql': 0.2, 'qh': 0.4, 'isabs': False, 'f_agg': 'var'},
                                       {'ql': 0.2, 'qh': 0.4, 'isabs': True, 'f_agg': 'mean'},
                                       {'ql': 0.2, 'qh': 0.4, 'isabs': True, 'f_agg': 'var'},
                                       {'ql': 0.2, 'qh': 0.6, 'isabs': False, 'f_agg': 'mean'},
                                       {'ql': 0.2, 'qh': 0.6, 'isabs': False, 'f_agg': 'var'},
                                       {'ql': 0.2, 'qh': 0.6, 'isabs': True, 'f_agg': 'mean'},
                                       {'ql': 0.2, 'qh': 0.6, 'isabs': True, 'f_agg': 'var'},
                                       {'ql': 0.2, 'qh': 0.8, 'isabs': False, 'f_agg': 'mean'},
                                       {'ql': 0.2, 'qh': 0.8, 'isabs': False, 'f_agg': 'var'},
                                       {'ql': 0.2, 'qh': 0.8, 'isabs': True, 'f_agg': 'mean'},
                                       {'ql': 0.2, 'qh': 0.8, 'isabs': True, 'f_agg': 'var'},
                                       {'ql': 0.2, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'},
                                       {'ql': 0.2, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'},
                                       {'ql': 0.2, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'},
                                       {'ql': 0.2, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'},
                                       {'ql': 0.4, 'qh': 0.6, 'isabs': False, 'f_agg': 'mean'},
                                       {'ql': 0.4, 'qh': 0.6, 'isabs': False, 'f_agg': 'var'},
                                       {'ql': 0.4, 'qh': 0.6, 'isabs': True, 'f_agg': 'mean'},
                                       {'ql': 0.4, 'qh': 0.6, 'isabs': True, 'f_agg': 'var'},
                                       {'ql': 0.4, 'qh': 0.8, 'isabs': False, 'f_agg': 'mean'},
                                       {'ql': 0.4, 'qh': 0.8, 'isabs': False, 'f_agg': 'var'},
                                       {'ql': 0.4, 'qh': 0.8, 'isabs': True, 'f_agg': 'mean'},
                                       {'ql': 0.4, 'qh': 0.8, 'isabs': True, 'f_agg': 'var'},
                                       {'ql': 0.4, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'},
                                       {'ql': 0.4, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'},
                                       {'ql': 0.4, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'},
                                       {'ql': 0.4, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'},
                                       {'ql': 0.6, 'qh': 0.8, 'isabs': False, 'f_agg': 'mean'},
                                       {'ql': 0.6, 'qh': 0.8, 'isabs': False, 'f_agg': 'var'},
                                       {'ql': 0.6, 'qh': 0.8, 'isabs': True, 'f_agg': 'mean'},
                                       {'ql': 0.6, 'qh': 0.8, 'isabs': True, 'f_agg': 'var'},
                                       {'ql': 0.6, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'},
                                       {'ql': 0.6, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'},
                                       {'ql': 0.6, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'},
                                       {'ql': 0.6, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'},
                                       {'ql': 0.8, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'},
                                       {'ql': 0.8, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'},
                                       {'ql': 0.8, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'},
                                       {'ql': 0.8, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'}],
                  'fft_coefficient': [{'coeff': 0, 'attr': 'real'}, {'coeff': 1, 'attr': 'real'},
                                      {'coeff': 2, 'attr': 'real'}, {'coeff': 3, 'attr': 'real'},
                                      {'coeff': 4, 'attr': 'real'}, {'coeff': 5, 'attr': 'real'},
                                      {'coeff': 6, 'attr': 'real'}, {'coeff': 7, 'attr': 'real'},
                                      {'coeff': 8, 'attr': 'real'}, {'coeff': 9, 'attr': 'real'},
                                      {'coeff': 10, 'attr': 'real'}, {'coeff': 11, 'attr': 'real'},
                                      {'coeff': 12, 'attr': 'real'}, {'coeff': 13, 'attr': 'real'},
                                      {'coeff': 14, 'attr': 'real'}, {'coeff': 15, 'attr': 'real'},
                                      {'coeff': 16, 'attr': 'real'}, {'coeff': 17, 'attr': 'real'},
                                      {'coeff': 18, 'attr': 'real'}, {'coeff': 19, 'attr': 'real'},
                                      {'coeff': 20, 'attr': 'real'}, {'coeff': 21, 'attr': 'real'},
                                      {'coeff': 22, 'attr': 'real'}, {'coeff': 23, 'attr': 'real'},
                                      {'coeff': 24, 'attr': 'real'}, {'coeff': 25, 'attr': 'real'},
                                      {'coeff': 26, 'attr': 'real'}, {'coeff': 27, 'attr': 'real'},
                                      {'coeff': 28, 'attr': 'real'}, {'coeff': 29, 'attr': 'real'},
                                      {'coeff': 30, 'attr': 'real'}, {'coeff': 31, 'attr': 'real'},
                                      {'coeff': 32, 'attr': 'real'}, {'coeff': 33, 'attr': 'real'},
                                      {'coeff': 34, 'attr': 'real'}, {'coeff': 35, 'attr': 'real'},
                                      {'coeff': 36, 'attr': 'real'}, {'coeff': 37, 'attr': 'real'},
                                      {'coeff': 38, 'attr': 'real'}, {'coeff': 39, 'attr': 'real'},
                                      {'coeff': 40, 'attr': 'real'}, {'coeff': 41, 'attr': 'real'},
                                      {'coeff': 42, 'attr': 'real'}, {'coeff': 43, 'attr': 'real'},
                                      {'coeff': 44, 'attr': 'real'}, {'coeff': 45, 'attr': 'real'},
                                      {'coeff': 46, 'attr': 'real'}, {'coeff': 47, 'attr': 'real'},
                                      {'coeff': 48, 'attr': 'real'}, {'coeff': 49, 'attr': 'real'},
                                      {'coeff': 50, 'attr': 'real'}, {'coeff': 51, 'attr': 'real'},
                                      {'coeff': 52, 'attr': 'real'}, {'coeff': 53, 'attr': 'real'},
                                      {'coeff': 54, 'attr': 'real'}, {'coeff': 55, 'attr': 'real'},
                                      {'coeff': 56, 'attr': 'real'}, {'coeff': 57, 'attr': 'real'},
                                      {'coeff': 58, 'attr': 'real'}, {'coeff': 59, 'attr': 'real'},
                                      {'coeff': 60, 'attr': 'real'}, {'coeff': 61, 'attr': 'real'},
                                      {'coeff': 62, 'attr': 'real'}, {'coeff': 63, 'attr': 'real'},
                                      {'coeff': 64, 'attr': 'real'}, {'coeff': 65, 'attr': 'real'},
                                      {'coeff': 66, 'attr': 'real'}, {'coeff': 67, 'attr': 'real'},
                                      {'coeff': 68, 'attr': 'real'}, {'coeff': 69, 'attr': 'real'},
                                      {'coeff': 70, 'attr': 'real'}, {'coeff': 71, 'attr': 'real'},
                                      {'coeff': 72, 'attr': 'real'}, {'coeff': 73, 'attr': 'real'},
                                      {'coeff': 74, 'attr': 'real'}, {'coeff': 75, 'attr': 'real'},
                                      {'coeff': 76, 'attr': 'real'}, {'coeff': 77, 'attr': 'real'},
                                      {'coeff': 78, 'attr': 'real'}, {'coeff': 79, 'attr': 'real'},
                                      {'coeff': 80, 'attr': 'real'}, {'coeff': 81, 'attr': 'real'},
                                      {'coeff': 82, 'attr': 'real'}, {'coeff': 83, 'attr': 'real'},
                                      {'coeff': 84, 'attr': 'real'}, {'coeff': 85, 'attr': 'real'},
                                      {'coeff': 86, 'attr': 'real'}, {'coeff': 87, 'attr': 'real'},
                                      {'coeff': 88, 'attr': 'real'}, {'coeff': 89, 'attr': 'real'},
                                      {'coeff': 90, 'attr': 'real'}, {'coeff': 91, 'attr': 'real'},
                                      {'coeff': 92, 'attr': 'real'}, {'coeff': 93, 'attr': 'real'},
                                      {'coeff': 94, 'attr': 'real'}, {'coeff': 95, 'attr': 'real'},
                                      {'coeff': 96, 'attr': 'real'}, {'coeff': 97, 'attr': 'real'},
                                      {'coeff': 98, 'attr': 'real'}, {'coeff': 99, 'attr': 'real'},
                                      {'coeff': 0, 'attr': 'imag'}, {'coeff': 1, 'attr': 'imag'},
                                      {'coeff': 2, 'attr': 'imag'}, {'coeff': 3, 'attr': 'imag'},
                                      {'coeff': 4, 'attr': 'imag'}, {'coeff': 5, 'attr': 'imag'},
                                      {'coeff': 6, 'attr': 'imag'}, {'coeff': 7, 'attr': 'imag'},
                                      {'coeff': 8, 'attr': 'imag'}, {'coeff': 9, 'attr': 'imag'},
                                      {'coeff': 10, 'attr': 'imag'}, {'coeff': 11, 'attr': 'imag'},
                                      {'coeff': 12, 'attr': 'imag'}, {'coeff': 13, 'attr': 'imag'},
                                      {'coeff': 14, 'attr': 'imag'}, {'coeff': 15, 'attr': 'imag'},
                                      {'coeff': 16, 'attr': 'imag'}, {'coeff': 17, 'attr': 'imag'},
                                      {'coeff': 18, 'attr': 'imag'}, {'coeff': 19, 'attr': 'imag'},
                                      {'coeff': 20, 'attr': 'imag'}, {'coeff': 21, 'attr': 'imag'},
                                      {'coeff': 22, 'attr': 'imag'}, {'coeff': 23, 'attr': 'imag'},
                                      {'coeff': 24, 'attr': 'imag'}, {'coeff': 25, 'attr': 'imag'},
                                      {'coeff': 26, 'attr': 'imag'}, {'coeff': 27, 'attr': 'imag'},
                                      {'coeff': 28, 'attr': 'imag'}, {'coeff': 29, 'attr': 'imag'},
                                      {'coeff': 30, 'attr': 'imag'}, {'coeff': 31, 'attr': 'imag'},
                                      {'coeff': 32, 'attr': 'imag'}, {'coeff': 33, 'attr': 'imag'},
                                      {'coeff': 34, 'attr': 'imag'}, {'coeff': 35, 'attr': 'imag'},
                                      {'coeff': 36, 'attr': 'imag'}, {'coeff': 37, 'attr': 'imag'},
                                      {'coeff': 38, 'attr': 'imag'}, {'coeff': 39, 'attr': 'imag'},
                                      {'coeff': 40, 'attr': 'imag'}, {'coeff': 41, 'attr': 'imag'},
                                      {'coeff': 42, 'attr': 'imag'}, {'coeff': 43, 'attr': 'imag'},
                                      {'coeff': 44, 'attr': 'imag'}, {'coeff': 45, 'attr': 'imag'},
                                      {'coeff': 46, 'attr': 'imag'}, {'coeff': 47, 'attr': 'imag'},
                                      {'coeff': 48, 'attr': 'imag'}, {'coeff': 49, 'attr': 'imag'},
                                      {'coeff': 50, 'attr': 'imag'}, {'coeff': 51, 'attr': 'imag'},
                                      {'coeff': 52, 'attr': 'imag'}, {'coeff': 53, 'attr': 'imag'},
                                      {'coeff': 54, 'attr': 'imag'}, {'coeff': 55, 'attr': 'imag'},
                                      {'coeff': 56, 'attr': 'imag'}, {'coeff': 57, 'attr': 'imag'},
                                      {'coeff': 58, 'attr': 'imag'}, {'coeff': 59, 'attr': 'imag'},
                                      {'coeff': 60, 'attr': 'imag'}, {'coeff': 61, 'attr': 'imag'},
                                      {'coeff': 62, 'attr': 'imag'}, {'coeff': 63, 'attr': 'imag'},
                                      {'coeff': 64, 'attr': 'imag'}, {'coeff': 65, 'attr': 'imag'},
                                      {'coeff': 66, 'attr': 'imag'}, {'coeff': 67, 'attr': 'imag'},
                                      {'coeff': 68, 'attr': 'imag'}, {'coeff': 69, 'attr': 'imag'},
                                      {'coeff': 70, 'attr': 'imag'}, {'coeff': 71, 'attr': 'imag'},
                                      {'coeff': 72, 'attr': 'imag'}, {'coeff': 73, 'attr': 'imag'},
                                      {'coeff': 74, 'attr': 'imag'}, {'coeff': 75, 'attr': 'imag'},
                                      {'coeff': 76, 'attr': 'imag'}, {'coeff': 77, 'attr': 'imag'},
                                      {'coeff': 78, 'attr': 'imag'}, {'coeff': 79, 'attr': 'imag'},
                                      {'coeff': 80, 'attr': 'imag'}, {'coeff': 81, 'attr': 'imag'},
                                      {'coeff': 82, 'attr': 'imag'}, {'coeff': 83, 'attr': 'imag'},
                                      {'coeff': 84, 'attr': 'imag'}, {'coeff': 85, 'attr': 'imag'},
                                      {'coeff': 86, 'attr': 'imag'}, {'coeff': 87, 'attr': 'imag'},
                                      {'coeff': 88, 'attr': 'imag'}, {'coeff': 89, 'attr': 'imag'},
                                      {'coeff': 90, 'attr': 'imag'}, {'coeff': 91, 'attr': 'imag'},
                                      {'coeff': 92, 'attr': 'imag'}, {'coeff': 93, 'attr': 'imag'},
                                      {'coeff': 94, 'attr': 'imag'}, {'coeff': 95, 'attr': 'imag'},
                                      {'coeff': 96, 'attr': 'imag'}, {'coeff': 97, 'attr': 'imag'},
                                      {'coeff': 98, 'attr': 'imag'}, {'coeff': 99, 'attr': 'imag'},
                                      {'coeff': 0, 'attr': 'abs'}, {'coeff': 1, 'attr': 'abs'},
                                      {'coeff': 2, 'attr': 'abs'}, {'coeff': 3, 'attr': 'abs'},
                                      {'coeff': 4, 'attr': 'abs'}, {'coeff': 5, 'attr': 'abs'},
                                      {'coeff': 6, 'attr': 'abs'}, {'coeff': 7, 'attr': 'abs'},
                                      {'coeff': 8, 'attr': 'abs'}, {'coeff': 9, 'attr': 'abs'},
                                      {'coeff': 10, 'attr': 'abs'}, {'coeff': 11, 'attr': 'abs'},
                                      {'coeff': 12, 'attr': 'abs'}, {'coeff': 13, 'attr': 'abs'},
                                      {'coeff': 14, 'attr': 'abs'}, {'coeff': 15, 'attr': 'abs'},
                                      {'coeff': 16, 'attr': 'abs'}, {'coeff': 17, 'attr': 'abs'},
                                      {'coeff': 18, 'attr': 'abs'}, {'coeff': 19, 'attr': 'abs'},
                                      {'coeff': 20, 'attr': 'abs'}, {'coeff': 21, 'attr': 'abs'},
                                      {'coeff': 22, 'attr': 'abs'}, {'coeff': 23, 'attr': 'abs'},
                                      {'coeff': 24, 'attr': 'abs'}, {'coeff': 25, 'attr': 'abs'},
                                      {'coeff': 26, 'attr': 'abs'}, {'coeff': 27, 'attr': 'abs'},
                                      {'coeff': 28, 'attr': 'abs'}, {'coeff': 29, 'attr': 'abs'},
                                      {'coeff': 30, 'attr': 'abs'}, {'coeff': 31, 'attr': 'abs'},
                                      {'coeff': 32, 'attr': 'abs'}, {'coeff': 33, 'attr': 'abs'},
                                      {'coeff': 34, 'attr': 'abs'}, {'coeff': 35, 'attr': 'abs'},
                                      {'coeff': 36, 'attr': 'abs'}, {'coeff': 37, 'attr': 'abs'},
                                      {'coeff': 38, 'attr': 'abs'}, {'coeff': 39, 'attr': 'abs'},
                                      {'coeff': 40, 'attr': 'abs'}, {'coeff': 41, 'attr': 'abs'},
                                      {'coeff': 42, 'attr': 'abs'}, {'coeff': 43, 'attr': 'abs'},
                                      {'coeff': 44, 'attr': 'abs'}, {'coeff': 45, 'attr': 'abs'},
                                      {'coeff': 46, 'attr': 'abs'}, {'coeff': 47, 'attr': 'abs'},
                                      {'coeff': 48, 'attr': 'abs'}, {'coeff': 49, 'attr': 'abs'},
                                      {'coeff': 50, 'attr': 'abs'}, {'coeff': 51, 'attr': 'abs'},
                                      {'coeff': 52, 'attr': 'abs'}, {'coeff': 53, 'attr': 'abs'},
                                      {'coeff': 54, 'attr': 'abs'}, {'coeff': 55, 'attr': 'abs'},
                                      {'coeff': 56, 'attr': 'abs'}, {'coeff': 57, 'attr': 'abs'},
                                      {'coeff': 58, 'attr': 'abs'}, {'coeff': 59, 'attr': 'abs'},
                                      {'coeff': 60, 'attr': 'abs'}, {'coeff': 61, 'attr': 'abs'},
                                      {'coeff': 62, 'attr': 'abs'}, {'coeff': 63, 'attr': 'abs'},
                                      {'coeff': 64, 'attr': 'abs'}, {'coeff': 65, 'attr': 'abs'},
                                      {'coeff': 66, 'attr': 'abs'}, {'coeff': 67, 'attr': 'abs'},
                                      {'coeff': 68, 'attr': 'abs'}, {'coeff': 69, 'attr': 'abs'},
                                      {'coeff': 70, 'attr': 'abs'}, {'coeff': 71, 'attr': 'abs'},
                                      {'coeff': 72, 'attr': 'abs'}, {'coeff': 73, 'attr': 'abs'},
                                      {'coeff': 74, 'attr': 'abs'}, {'coeff': 75, 'attr': 'abs'},
                                      {'coeff': 76, 'attr': 'abs'}, {'coeff': 77, 'attr': 'abs'},
                                      {'coeff': 78, 'attr': 'abs'}, {'coeff': 79, 'attr': 'abs'},
                                      {'coeff': 80, 'attr': 'abs'}, {'coeff': 81, 'attr': 'abs'},
                                      {'coeff': 82, 'attr': 'abs'}, {'coeff': 83, 'attr': 'abs'},
                                      {'coeff': 84, 'attr': 'abs'}, {'coeff': 85, 'attr': 'abs'},
                                      {'coeff': 86, 'attr': 'abs'}, {'coeff': 87, 'attr': 'abs'},
                                      {'coeff': 88, 'attr': 'abs'}, {'coeff': 89, 'attr': 'abs'},
                                      {'coeff': 90, 'attr': 'abs'}, {'coeff': 91, 'attr': 'abs'},
                                      {'coeff': 92, 'attr': 'abs'}, {'coeff': 93, 'attr': 'abs'},
                                      {'coeff': 94, 'attr': 'abs'}, {'coeff': 95, 'attr': 'abs'},
                                      {'coeff': 96, 'attr': 'abs'}, {'coeff': 97, 'attr': 'abs'},
                                      {'coeff': 98, 'attr': 'abs'}, {'coeff': 99, 'attr': 'abs'},
                                      {'coeff': 0, 'attr': 'angle'}, {'coeff': 1, 'attr': 'angle'},
                                      {'coeff': 2, 'attr': 'angle'}, {'coeff': 3, 'attr': 'angle'},
                                      {'coeff': 4, 'attr': 'angle'}, {'coeff': 5, 'attr': 'angle'},
                                      {'coeff': 6, 'attr': 'angle'}, {'coeff': 7, 'attr': 'angle'},
                                      {'coeff': 8, 'attr': 'angle'}, {'coeff': 9, 'attr': 'angle'},
                                      {'coeff': 10, 'attr': 'angle'}, {'coeff': 11, 'attr': 'angle'},
                                      {'coeff': 12, 'attr': 'angle'}, {'coeff': 13, 'attr': 'angle'},
                                      {'coeff': 14, 'attr': 'angle'}, {'coeff': 15, 'attr': 'angle'},
                                      {'coeff': 16, 'attr': 'angle'}, {'coeff': 17, 'attr': 'angle'},
                                      {'coeff': 18, 'attr': 'angle'}, {'coeff': 19, 'attr': 'angle'},
                                      {'coeff': 20, 'attr': 'angle'}, {'coeff': 21, 'attr': 'angle'},
                                      {'coeff': 22, 'attr': 'angle'}, {'coeff': 23, 'attr': 'angle'},
                                      {'coeff': 24, 'attr': 'angle'}, {'coeff': 25, 'attr': 'angle'},
                                      {'coeff': 26, 'attr': 'angle'}, {'coeff': 27, 'attr': 'angle'},
                                      {'coeff': 28, 'attr': 'angle'}, {'coeff': 29, 'attr': 'angle'},
                                      {'coeff': 30, 'attr': 'angle'}, {'coeff': 31, 'attr': 'angle'},
                                      {'coeff': 32, 'attr': 'angle'}, {'coeff': 33, 'attr': 'angle'},
                                      {'coeff': 34, 'attr': 'angle'}, {'coeff': 35, 'attr': 'angle'},
                                      {'coeff': 36, 'attr': 'angle'}, {'coeff': 37, 'attr': 'angle'},
                                      {'coeff': 38, 'attr': 'angle'}, {'coeff': 39, 'attr': 'angle'},
                                      {'coeff': 40, 'attr': 'angle'}, {'coeff': 41, 'attr': 'angle'},
                                      {'coeff': 42, 'attr': 'angle'}, {'coeff': 43, 'attr': 'angle'},
                                      {'coeff': 44, 'attr': 'angle'}, {'coeff': 45, 'attr': 'angle'},
                                      {'coeff': 46, 'attr': 'angle'}, {'coeff': 47, 'attr': 'angle'},
                                      {'coeff': 48, 'attr': 'angle'}, {'coeff': 49, 'attr': 'angle'},
                                      {'coeff': 50, 'attr': 'angle'}, {'coeff': 51, 'attr': 'angle'},
                                      {'coeff': 52, 'attr': 'angle'}, {'coeff': 53, 'attr': 'angle'},
                                      {'coeff': 54, 'attr': 'angle'}, {'coeff': 55, 'attr': 'angle'},
                                      {'coeff': 56, 'attr': 'angle'}, {'coeff': 57, 'attr': 'angle'},
                                      {'coeff': 58, 'attr': 'angle'}, {'coeff': 59, 'attr': 'angle'},
                                      {'coeff': 60, 'attr': 'angle'}, {'coeff': 61, 'attr': 'angle'},
                                      {'coeff': 62, 'attr': 'angle'}, {'coeff': 63, 'attr': 'angle'},
                                      {'coeff': 64, 'attr': 'angle'}, {'coeff': 65, 'attr': 'angle'},
                                      {'coeff': 66, 'attr': 'angle'}, {'coeff': 67, 'attr': 'angle'},
                                      {'coeff': 68, 'attr': 'angle'}, {'coeff': 69, 'attr': 'angle'},
                                      {'coeff': 70, 'attr': 'angle'}, {'coeff': 71, 'attr': 'angle'},
                                      {'coeff': 72, 'attr': 'angle'}, {'coeff': 73, 'attr': 'angle'},
                                      {'coeff': 74, 'attr': 'angle'}, {'coeff': 75, 'attr': 'angle'},
                                      {'coeff': 76, 'attr': 'angle'}, {'coeff': 77, 'attr': 'angle'},
                                      {'coeff': 78, 'attr': 'angle'}, {'coeff': 79, 'attr': 'angle'},
                                      {'coeff': 80, 'attr': 'angle'}, {'coeff': 81, 'attr': 'angle'},
                                      {'coeff': 82, 'attr': 'angle'}, {'coeff': 83, 'attr': 'angle'},
                                      {'coeff': 84, 'attr': 'angle'}, {'coeff': 85, 'attr': 'angle'},
                                      {'coeff': 86, 'attr': 'angle'}, {'coeff': 87, 'attr': 'angle'},
                                      {'coeff': 88, 'attr': 'angle'}, {'coeff': 89, 'attr': 'angle'},
                                      {'coeff': 90, 'attr': 'angle'}, {'coeff': 91, 'attr': 'angle'},
                                      {'coeff': 92, 'attr': 'angle'}, {'coeff': 93, 'attr': 'angle'},
                                      {'coeff': 94, 'attr': 'angle'}, {'coeff': 95, 'attr': 'angle'},
                                      {'coeff': 96, 'attr': 'angle'}, {'coeff': 97, 'attr': 'angle'},
                                      {'coeff': 98, 'attr': 'angle'}, {'coeff': 99, 'attr': 'angle'}],
                  'approximate_entropy': [{'m': 2, 'r': 0.1}, {'m': 2, 'r': 0.3}, {'m': 2, 'r': 0.5},
                                          {'m': 2, 'r': 0.7}, {'m': 2, 'r': 0.9}],
                  'linear_trend': [{'attr': 'pvalue'}, {'attr': 'rvalue'}, {'attr': 'intercept'}, {'attr': 'slope'},
                                   {'attr': 'stderr'}],
                  'number_peaks': [{'n': 1}, {'n': 3}, {'n': 5}, {'n': 10}, {'n': 50}],
                  'energy_ratio_by_chunks': [{'num_segments': 10, 'segment_focus': 0},
                                             {'num_segments': 10, 'segment_focus': 1},
                                             {'num_segments': 10, 'segment_focus': 2},
                                             {'num_segments': 10, 'segment_focus': 3},
                                             {'num_segments': 10, 'segment_focus': 4},
                                             {'num_segments': 10, 'segment_focus': 5},
                                             {'num_segments': 10, 'segment_focus': 6},
                                             {'num_segments': 10, 'segment_focus': 7},
                                             {'num_segments': 10, 'segment_focus': 8},
                                             {'num_segments': 10, 'segment_focus': 9}],
                  'ratio_beyond_r_sigma': [{'r': 0.5}, {'r': 1}, {'r': 1.5}, {'r': 2}, {'r': 2.5}, {'r': 3}, {'r': 5},
                                           {'r': 6}, {'r': 7}, {'r': 10}], 'count_above': [{'t': 0}],
                  'count_below': [{'t': 0}],
                  'lempel_ziv_complexity': [{'bins': 2}, {'bins': 3}, {'bins': 5}, {'bins': 10}, {'bins': 100}],
                  'permutation_entropy': [{'tau': 1, 'dimension': 3}, {'tau': 1, 'dimension': 4},
                                          {'tau': 1, 'dimension': 5}, {'tau': 1, 'dimension': 6},
                                          {'tau': 1, 'dimension': 7}]}


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
