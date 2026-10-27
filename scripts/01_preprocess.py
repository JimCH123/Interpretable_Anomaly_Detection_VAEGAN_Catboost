"""Step 1 - Clean the raw UCI-SECOM data (dissertation Section IV-B).

Inputs : data/raw/uci-secom.csv
Outputs: data/streaming/test_data.csv          10 raw records for the API demo
         data/processed/uci-secom-cleaned.csv  standardized, feature-selected data
         data/processed/anomaly_data.csv       real anomalies only
         artifacts/preprocessing/{all_cols.npy, idx.npy, scaler.pkl}
"""
import argparse

import joblib
import numpy as np

from anomaly_detection import config, visualization as viz
from anomaly_detection.preprocessing import (data_standardization, drop_datetime, get_test_data,
                                             handle_missing_value, load_data, remove_constant_columns,
                                             remove_high_corr, remove_low_corr_with_target,
                                             remove_low_variance, save_real_anomaly_data)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", action="store_true", help="display figures as well as saving them")
    parser.add_argument("--no-plots", action="store_true", help="skip the EDA figures")
    args = parser.parse_args()

    p = config.PREPROCESS
    target = config.TARGET_COL
    for path in (config.CLEANED_DATA, config.STREAMING_DATA, config.SCALER):
        path.parent.mkdir(parents=True, exist_ok=True)

    data = load_data(config.RAW_DATA)
    print("Data loaded, shape:", data.shape)

    test_data, data = get_test_data(data, p["n_normal_holdout"], p["n_anomaly_holdout"])
    test_data.to_csv(config.STREAMING_DATA, index=False)
    print(f"Held out {p['n_normal_holdout']} normal + {p['n_anomaly_holdout']} anomalous raw samples "
          f"for online validation, shape: {test_data.shape}")
    print("Remaining data, shape:", data.shape)

    data = drop_datetime(data)
    print("After dropping the time column, shape:", data.shape)

    all_cols = data.drop(columns=target).columns.tolist()
    np.save(config.ALL_COLS, all_cols)

    data = handle_missing_value(data, p["max_missing_ratio"])
    print("After handling missing values, shape:", data.shape)

    data_filter = remove_constant_columns(data)
    print("After removing constant features, shape:", data_filter.shape)

    data_filter = remove_low_variance(data_filter, p["variance_threshold"])
    print("After removing low-variance features, shape:", data_filter.shape)

    data_filter = remove_low_corr_with_target(data_filter, p["target_corr_threshold"], target)
    print("After removing features weakly correlated with the label, shape:", data_filter.shape)

    data_filter = remove_high_corr(data_filter, p["feature_corr_threshold"], target)
    print("After removing highly correlated features, shape:", data_filter.shape)

    data_standard, scaler = data_standardization(data_filter, target)
    joblib.dump(scaler, config.SCALER)
    data_standard.to_csv(config.CLEANED_DATA, index=False)
    print(f"Cleaned dataset {data_standard.shape} saved to {config.CLEANED_DATA.relative_to(config.ROOT)}")

    fail_df = save_real_anomaly_data(data_standard, target)
    fail_df.to_csv(config.ANOMALY_DATA, index=False)
    print(f"Real anomalies {fail_df.shape} saved to {config.ANOMALY_DATA.relative_to(config.ROOT)}")

    # Positions of the kept features within the raw feature vector, used by transform()
    remain_features = fail_df.drop(columns=target).columns.tolist()
    idx = np.array([all_cols.index(name) for name in remain_features], dtype=np.int32)
    np.save(config.FEATURE_IDX, idx)
    print(f"Saved preprocessing artifacts to {config.SCALER.parent.relative_to(config.ROOT)}/")

    if not args.no_plots:
        viz.plot_pass_fail_distribution(data_standard, target, show=args.show)
        viz.plot_kpca_normal_vs_anomaly(data_standard, target, show=args.show)
        viz.plot_feature_distribution(data_standard, target, show=args.show)


if __name__ == "__main__":
    main()
