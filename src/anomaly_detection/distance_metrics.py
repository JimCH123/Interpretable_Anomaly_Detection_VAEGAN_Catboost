"""Per-feature distances between real and generated anomalies (dissertation Section V-B).

Each function compares the two samples feature by feature and returns
``(mean over features, sum over features)``. The label column is ignored.
"""
import numpy as np
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel

from . import config


def _features(df, target_col=config.TARGET_COL):
    return df.drop(columns=[target_col], errors='ignore')


def kl_divergence(anomaly_df, synthetic_df, bins=50):
    """Histogram-based KL(real || generated), Eq. 12."""
    anomaly_df = _features(anomaly_df)
    synthetic_df = _features(synthetic_df)
    kl_divs = []
    for feature_idx in range(anomaly_df.shape[1]):
        anomaly_hist, bin_edges = np.histogram(anomaly_df.iloc[:, feature_idx], bins=bins, density=True)
        synthetic_hist, _ = np.histogram(synthetic_df.iloc[:, feature_idx], bins=bin_edges, density=True)
        anomaly_hist = np.where(anomaly_hist == 0, 1e-10, anomaly_hist)
        synthetic_hist = np.where(synthetic_hist == 0, 1e-10, synthetic_hist)
        kl_div = entropy(anomaly_hist, synthetic_hist)
        if np.isinf(kl_div) or np.isnan(kl_div):
            kl_div = 10
        kl_divs.append(kl_div)
    return np.mean(kl_divs), np.sum(kl_divs)


def js_divergence(anomaly_df, synthetic_df, bins=50):
    """Histogram-based Jensen-Shannon divergence, Eq. 13."""
    anomaly_df = _features(anomaly_df)
    synthetic_df = _features(synthetic_df)
    js_divs = []
    for feature_idx in range(anomaly_df.shape[1]):
        anomaly_hist, bin_edges = np.histogram(anomaly_df.iloc[:, feature_idx], bins=bins, density=True)
        synthetic_hist, _ = np.histogram(synthetic_df.iloc[:, feature_idx], bins=bin_edges, density=True)
        anomaly_hist = np.where(anomaly_hist == 0, 1e-10, anomaly_hist)
        synthetic_hist = np.where(synthetic_hist == 0, 1e-10, synthetic_hist)
        m = (anomaly_hist + synthetic_hist) * 0.5
        js_div = (entropy(anomaly_hist, m) + entropy(synthetic_hist, m)) * 0.5
        js_divs.append(js_div)
    return np.mean(js_divs), np.sum(js_divs)


def compute_wasserstein_distance(anomaly_df, synthetic_df):
    """1-D Wasserstein distance per feature, Eq. 14."""
    anomaly_df = _features(anomaly_df)
    synthetic_df = _features(synthetic_df)
    distances = []
    for feature_idx in range(anomaly_df.shape[1]):
        distance = wasserstein_distance(anomaly_df.iloc[:, feature_idx], synthetic_df.iloc[:, feature_idx])
        distances.append(distance)
    return np.mean(distances), np.sum(distances)


def maximum_mean_discrepancy(anomaly_df, synthetic_df, sigma=1):
    """Unbiased RBF-kernel MMD per feature, Eq. 15."""
    anomaly_df = _features(anomaly_df)
    synthetic_df = _features(synthetic_df)
    m = anomaly_df.shape[0]
    n = synthetic_df.shape[0]
    gamma = 1 / (2 * sigma ** 2)
    mmds = []
    for feature_idx in range(anomaly_df.shape[1]):
        anomaly_feature = anomaly_df.iloc[:, feature_idx].values.reshape(-1, 1)
        synthetic_feature = synthetic_df.iloc[:, feature_idx].values.reshape(-1, 1)
        Kxx = rbf_kernel(anomaly_feature, anomaly_feature, gamma=gamma)
        Kyy = rbf_kernel(synthetic_feature, synthetic_feature, gamma=gamma)
        Kxy = rbf_kernel(anomaly_feature, synthetic_feature, gamma=gamma)

        np.fill_diagonal(Kxx, 0)
        mean_Kxx = np.sum(Kxx) / (m * (m - 1))
        np.fill_diagonal(Kyy, 0)
        mean_Kyy = np.sum(Kyy) / (n * (n - 1))
        mean_Kxy = np.mean(Kxy)

        mmd_squared = mean_Kxx + mean_Kyy - 2 * mean_Kxy
        mmds.append(np.sqrt(max(mmd_squared, 0)))
    return np.mean(mmds), np.sum(mmds)


METRICS = {
    "KL Divergence": kl_divergence,
    "JS Divergence": js_divergence,
    "Wasserstein Distance": compute_wasserstein_distance,
    "MMD": maximum_mean_discrepancy,
}


def compare(anomaly_df, synthetic_df):
    """Return ``{metric name: (mean, total)}`` for all four distances."""
    return {name: fn(anomaly_df, synthetic_df) for name, fn in METRICS.items()}
