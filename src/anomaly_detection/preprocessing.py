"""Data preprocessing (dissertation Section IV-B).

Offline: the cleaning pipeline that turns the raw 590-feature UCI-SECOM file into
the standardized, feature-selected training set.
Online: ``transform()`` applies the same feature selection and scaling to a single
raw record, as used by the real-time API.
"""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from . import config


def load_data(file_path):
    data = pd.read_csv(file_path, encoding="utf-8")
    return data


def get_test_data(df, n_normal=7, n_anomaly=3, target_col=config.TARGET_COL):
    """Hold out raw samples for online validation; return (holdout, remaining)."""
    df_remained = df.copy()
    normal_data = df_remained[df_remained[target_col] == config.NORMAL_LABEL].sample(n=n_normal, random_state=config.SEED)
    normal_idx = normal_data.index
    anomaly_data = df_remained[df_remained[target_col] == config.ANOMALY_LABEL].sample(n=n_anomaly, random_state=config.SEED)
    anomaly_idx = anomaly_data.index

    test_data = pd.concat([normal_data, anomaly_data]).reset_index(drop=True)

    indices_to_drop = normal_idx.union(anomaly_idx)
    df_remained.drop(indices_to_drop, inplace=True)

    return test_data, df_remained


def handle_missing_value(df, max_missing_ratio=0.2):
    """Drop features with too many missing values, mean-impute the rest."""
    numeric_cols = df.select_dtypes(include=['float64']).columns
    cols_to_drop = []
    for col in numeric_cols:
        null_proportion = df[col].isnull().sum() / df.shape[0]
        if null_proportion > max_missing_ratio:
            cols_to_drop.append(col)
        else:
            df[col] = df[col].fillna(df[col].mean())
    df = df.drop(columns=cols_to_drop)
    return df


def drop_datetime(df, time_col=config.TIME_COL):
    df = df.drop(columns=[time_col])
    return df


def remove_constant_columns(df):
    columns_to_drop = [col for col in df.columns if df[col].nunique() == 1]
    df_cleaned = df.drop(columns=columns_to_drop)
    return df_cleaned


def remove_low_variance(df, threshold=0.05):
    sel = VarianceThreshold(threshold=threshold)
    sel.fit(df)
    mask = sel.get_support()
    drop_cols = df.columns[~mask]
    df_reduced = df.drop(columns=drop_cols)
    return df_reduced


def remove_low_corr_with_target(df, threshold, target_col):
    num_cols = df.select_dtypes(include=['float64']).drop(columns=[target_col], errors='ignore').columns
    correlations = df[num_cols].corrwith(df[target_col]).abs()
    low_corr_cols = correlations[correlations < threshold].index.tolist()
    df_reduced = df.drop(columns=low_corr_cols)
    return df_reduced


def remove_high_corr(df, threshold, target_col):
    """For each feature pair with |r| >= threshold, drop the later feature."""
    data_num = df.select_dtypes(include=['float64']).drop(columns=[target_col], errors='ignore')
    corr = data_num.corr().abs()
    drop_cols = set()
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if corr.iloc[i, j] >= threshold:
                drop_cols.add(corr.columns[j])
    df_reduced = df.drop(columns=drop_cols)
    return df_reduced


def data_standardization(df, target_col):
    """Z-score all features, with the scaler fitted on normal samples only."""
    num_cols = df.select_dtypes(include=['float64']).drop(columns=[target_col], errors='ignore').columns
    normal_df = df[df[target_col] == config.NORMAL_LABEL][num_cols]
    scaler = StandardScaler()
    scaler.fit(normal_df)
    df[num_cols] = scaler.transform(df[num_cols])
    return df, scaler


def save_real_anomaly_data(df, target_col):
    fail_df = df[df[target_col] == config.ANOMALY_LABEL]
    return fail_df


# ----------------------------------------------------------------------------
# Online transform (used by the API)
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_transform_artifacts():
    idx = np.load(config.FEATURE_IDX)
    scaler = joblib.load(config.SCALER)
    all_cols = np.load(config.ALL_COLS)
    return idx, scaler, all_cols


def selected_feature_names():
    """Names of the features kept by preprocessing, in model input order."""
    idx, _, all_cols = _load_transform_artifacts()
    return all_cols[idx].astype(str)


def transform(raw_list):
    """Map one raw 590-value record to the standardized model input (shape ``(1, n_features)``).

    Missing values are filled with 0.0 before scaling. Note that the offline
    pipeline mean-imputes missing values instead; this function keeps the
    behaviour used for the online validation reported in the dissertation.
    """
    idx, scaler, all_cols = _load_transform_artifacts()
    if len(raw_list) != len(all_cols):
        raise ValueError(f"Input length {len(raw_list)} does not match expected {len(all_cols)}")

    cleaned = [
        0.0 if (x in ("", None) or (isinstance(x, float) and np.isnan(x))) else float(x)
        for x in raw_list
    ]

    x_full = pd.DataFrame([cleaned], columns=all_cols, dtype=np.float32)
    x_sel = x_full.iloc[:, idx]
    x_standard = scaler.transform(x_sel)
    return x_standard
