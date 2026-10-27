"""TreeSHAP explanations of the CatBoost detector (dissertation Section IV-E)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import config
from .visualization import finish


def background_data(X_train, y_train, size=config.N_BACKGROUND, seed=config.SEED):
    """Sample normal training points as the TreeSHAP background and save them."""
    normal_idx = np.where(y_train == 0)[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(normal_idx, size=size, replace=False)
    background = X_train[idx].astype(np.float32)
    np.save(config.BACKGROUND, background)
    print(f"Saved background data {background.shape} to {config.BACKGROUND.relative_to(config.ROOT)}")
    return background


def shap_explain(clf, X_test, y_test, feature_names, background, max_display=10, show=False):
    """Global (bar, beeswarm) and local (waterfall of the first test anomaly) explanations."""
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    explainer = shap.TreeExplainer(clf, background)
    shap_values = explainer(X_test_df)

    # Features kept in their original column order, as in Figs. 12-13
    shap.summary_plot(shap_values, X_test_df, plot_type="bar", feature_names=feature_names,
                      max_display=max_display, sort=False, show=False)
    finish("shap_summary_bar.png", show)
    shap.summary_plot(shap_values, X_test_df, feature_names=feature_names,
                      max_display=max_display, sort=False, show=False)
    finish("shap_summary_beeswarm.png", show)

    # Features sorted by mean |SHAP|
    shap.plots.bar(shap_values, show=False)
    finish("shap_bar_sorted.png", show)
    shap.plots.beeswarm(shap_values, show=False)
    finish("shap_beeswarm_sorted.png", show)

    anomaly_idx = np.where(y_test == 1)[0]
    if len(anomaly_idx):
        plt.figure()
        shap.plots.waterfall(explainer(X_test_df.iloc[[anomaly_idx[0]]])[0], show=False)
        finish("shap_waterfall_first_anomaly.png", show)
    return shap_values
