"""All plotting helpers. Every figure is saved to ``outputs/figures/`` and only
displayed on screen when ``show=True``."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import KernelPCA
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)

from . import config


def finish(filename, show=False):
    """Save the current figure to ``outputs/figures/<filename>`` and optionally show it."""
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURE_DIR / filename
    plt.savefig(path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close("all")
    print(f"Saved figure: {path.relative_to(config.ROOT)}")
    return path


# ----------------------------------------------------------------------------
# EDA (Figs. 2-3)
# ----------------------------------------------------------------------------
def plot_pass_fail_distribution(df, target_col, show=False):
    size = df[target_col].value_counts()
    labels = ["Normal", "Anomalies"]
    colors = ["blue", "red"]
    explode = (0, 0.2)
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, _, autotexts = ax.pie(
        size,
        labels=None,
        colors=colors,
        explode=explode,
        autopct="%2.2f%%",
        pctdistance=1.15,
        startangle=0,
        shadow=False
    )
    fig.legend(wedges, labels, title="Class", loc="upper right", fontsize=12)
    ax.set_title("Proportion of Normal vs. Anomalies")
    plt.tight_layout()
    finish("eda_class_proportion.png", show)

    plt.figure(figsize=(8, 6))
    ax = df[target_col].value_counts().plot(
        kind="bar",
        color=['blue', 'red']
    )
    for i, v in enumerate(df[target_col].value_counts()):
        ax.text(i, v + 0.5, str(v), ha='center', va='bottom')
    plt.title("Pass/Fail Counts")
    plt.xlabel("Pass/Fail")
    plt.ylabel("Count")
    plt.xticks(ticks=[0, 1], labels=['Pass', 'Fail'], rotation=0)
    finish("eda_class_counts.png", show)


def plot_correlation_heatmap(df, target_col, show=False):
    data_num = (
        df
        .select_dtypes(include=['float64'])
        .drop(columns=[target_col], errors='ignore')
    )
    data_corr = data_num.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(data_corr, cmap="YlGnBu")
    plt.title('Data Correlation', fontsize=20)
    finish("eda_correlation_heatmap.png", show)


def plot_feature_distribution(df, target_col, n_features=3, show=False):
    pass_data = df[df[target_col] == config.NORMAL_LABEL]
    fail_data = df[df[target_col] == config.ANOMALY_LABEL]
    for feature_idx in range(n_features):
        plt.figure(figsize=(8, 6))
        plt.hist(pass_data.iloc[:, feature_idx], bins=50, density=True, alpha=0.5, label='Pass Data', color='red')
        plt.hist(fail_data.iloc[:, feature_idx], bins=50, density=True, alpha=0.5, label='Fail Data', color='blue')
        sns.kdeplot(pass_data.iloc[:, feature_idx], label='Pass Data Distribution', color='red')
        sns.kdeplot(fail_data.iloc[:, feature_idx], label='Fail Data Distribution', color='blue')
        plt.title(f'Distribution of Feature {feature_idx}')
        plt.xlabel('Feature Value')
        plt.ylabel('Density')
        plt.legend()
        finish(f"eda_feature_{feature_idx}_distribution.png", show)


def plot_kpca_normal_vs_anomaly(standard_df, target_col, show=False):
    normal_data = standard_df[standard_df[target_col] == config.NORMAL_LABEL].drop(columns=[target_col])
    anomaly_data = standard_df[standard_df[target_col] == config.ANOMALY_LABEL].drop(columns=[target_col])
    pca = KernelPCA(n_components=2, kernel='rbf')
    normal_pca = pca.fit_transform(normal_data.values)
    anomaly_pca = pca.fit_transform(anomaly_data.values)
    plt.figure(figsize=(8, 6))
    plt.scatter(normal_pca[:, 0], normal_pca[:, 1], label="Normal Data", alpha=0.5, color="blue")
    plt.scatter(anomaly_pca[:, 0], anomaly_pca[:, 1], label="Anomaly Data", alpha=0.8, color="red", marker="x")
    plt.title("KernelPCA Distribution of Normal vs. Anomalies")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    finish("eda_kpca_normal_vs_anomaly.png", show)


# ----------------------------------------------------------------------------
# Generated anomalies (Figs. 4-8)
# ----------------------------------------------------------------------------
def plot_kde_real_vs_generated(fail_df, synthetic_df, model_name, show=False):
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.ravel()

    for feature_idx in range(4):
        ax = axes[feature_idx]

        ax.hist(fail_df.iloc[:, feature_idx],
                bins=50, density=True, alpha=0.5,
                label='Real Anomalies', color='red')
        ax.hist(synthetic_df.iloc[:, feature_idx],
                bins=50, density=True, alpha=0.5,
                label='Generated Anomalies', color='blue')

        sns.kdeplot(fail_df.iloc[:, feature_idx],
                    label='Real Anomalies', color='red', ax=ax)
        sns.kdeplot(synthetic_df.iloc[:, feature_idx],
                    label='Generated Anomalies', color='blue', ax=ax)

        ax.set_title(f'Feature {feature_idx}: Real vs. Generated', fontsize=11)
        ax.set_xlabel('Feature Value')
    axes[1].legend()
    plt.tight_layout()
    finish(f"generation_kde_{model_name}.png", show)


def plot_kpca_real_vs_generated(fail_df, synthetic_df, model_name, target_col=config.TARGET_COL, show=False):
    synthetic_data = synthetic_df.drop(columns=[target_col]).values
    fail_data = fail_df.values
    synthetic_num = len(synthetic_data)
    combined_data = np.vstack([synthetic_data, fail_data])
    kpca = KernelPCA(n_components=2, kernel='rbf')
    combined_pca = kpca.fit_transform(combined_data)
    synthetic_pca = combined_pca[:synthetic_num]
    reserved_pca = combined_pca[synthetic_num:]
    plt.figure(figsize=(8, 6))
    plt.scatter(synthetic_pca[:, 0], synthetic_pca[:, 1], label="Generated Anomalies", alpha=0.5, color="blue")
    plt.scatter(reserved_pca[:, 0], reserved_pca[:, 1], label="Real Anomalies", alpha=0.8, color="red", marker="x")
    plt.title("KernelPCA Distribution of Generated vs. Real Anomalies")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    finish(f"generation_kpca_{model_name}.png", show)


def plot_distance_comparison(vae_totals, vaegan_totals, show=False):
    """Grouped bars of the summed per-feature distances (KL, JS, WD, MMD)."""
    metrics = ['KL Divergence', 'JS Divergence', 'Wasserstein Distance', 'MMD']
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width / 2, vae_totals, width, label='VAE', color='blue')
    rects2 = ax.bar(x + width / 2, vaegan_totals, width, label='VAE-GAN', color='green')
    ax.set_ylabel('Total Value')
    ax.set_title('Comparison of VAE and VAE-GAN Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
    plt.tight_layout()
    finish("generation_distance_comparison.png", show)


# ----------------------------------------------------------------------------
# Classification (Figs. 9-11)
# ----------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", filename="confusion_matrix.png", show=False):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"])
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    finish(filename, show)


def plot_score_histogram(y_test, y_pred, y_scores, threshold=0.5, filename="score_histogram.png", show=False):
    tn_indices = (y_test == 0) & (y_pred == 0)
    fp_indices = (y_test == 0) & (y_pred == 1)
    fn_indices = (y_test == 1) & (y_pred == 0)
    tp_indices = (y_test == 1) & (y_pred == 1)
    plt.figure(figsize=(12, 7))
    plt.hist(y_scores[tn_indices], bins=50, alpha=0.5, label='True Negative (Normal)', color='green')
    plt.hist(y_scores[fp_indices], bins=50, alpha=0.5, label='False Positive (Normal misclassified)', color='orange')
    plt.hist(y_scores[fn_indices], bins=50, alpha=0.5, label='False Negative (Anomaly missed)', color='red')
    plt.hist(y_scores[tp_indices], bins=50, alpha=0.5, label='True Positive (Anomaly)', color='blue')
    plt.axvline(threshold, color='black', linestyle='--', label=f'Default Threshold ({threshold})')
    plt.title('Distribution of Anomaly Scores with Classification Results (CatBoost)')
    plt.xlabel('Anomaly Score (Probability)')
    plt.ylabel('Frequency')
    plt.legend()
    finish(filename, show)


def plot_roc_curve(y_true, y_scores, title="ROC Curve", filename="roc_curve.png", show=False):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    finish(filename, show)


def plot_pr_curve(y_true, y_scores, title="Precision-Recall Curve", filename="pr_curve.png", show=False):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    finish(filename, show)


def plot_oversampling_ratio(df, show=False):
    """Horizontal grouped bars of each metric for every oversampling ratio (Fig. 11).

    ``df`` is indexed by the OS ratio label with one column per metric.
    """
    df_T = df.T
    fig, ax = plt.subplots(figsize=(10, 6))
    df_T.plot(kind="barh", ax=ax, edgecolor="black")

    ax.invert_yaxis()  # Precision on top
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", padding=4, label_type="edge", fontsize=8)

    ax.set_ylabel("Evaluation Metric", fontsize=12)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_title("Evaluation Metrics Scores vs. Oversampling Ratio", pad=10)
    ax.set_xlim(0, df_T.values.max() * 1.25)  # 25% headroom for labels
    ax.legend(title="OS Ratio", loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True)
    plt.tight_layout()
    finish("oversampling_ratio.png", show)
