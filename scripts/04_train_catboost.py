"""Step 4 - Train and evaluate the CatBoost detector, then explain it with TreeSHAP
(dissertation Sections IV-D, IV-E).

Inputs : data/synthetic/combined_data_<model>.csv
Outputs: outputs/metrics_<model>.json and figures in outputs/figures/
         with --save-model: artifacts/model/{model.cbm, threshold.npy, bg_1000.npy}
"""
import argparse
import json

import numpy as np

from anomaly_detection import config, visualization as viz
from anomaly_detection.classifier import (bayesian_search_params, evaluate_detection, load_dataset,
                                          predict, save_model, split, train_catboost)
from anomaly_detection.explain import background_data, shap_explain


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("vae", "vaegan"), default="vaegan",
                        help="which augmented dataset to train on")
    parser.add_argument("--ratio", type=float, default=1.0, help="oversampling ratio of the dataset")
    parser.add_argument("--tune", action="store_true",
                        help="run the Bayesian hyperparameter search and train with its best parameters")
    parser.add_argument("--shap", action="store_true", help="produce the TreeSHAP figures")
    parser.add_argument("--save-model", action="store_true",
                        help="overwrite artifacts/model/ (used by the API)")
    parser.add_argument("--show", action="store_true", help="display figures as well as saving them")
    args = parser.parse_args()

    tag = args.model if args.ratio == 1.0 else f"{args.model}_r{args.ratio:g}"
    data, X, y = load_dataset(config.combined_path(args.model, args.ratio))
    X_train, X_test, y_train, y_test = split(X, y)
    print(f"Dataset {data.shape}: train {X_train.shape}, test {X_test.shape}")

    params = dict(config.CATBOOST_PARAMS)
    if args.tune:
        best_params, _ = bayesian_search_params(X_train, y_train)
        params.update(best_params)
    print("CatBoost parameters:", params)

    clf = train_catboost(X_train, y_train, params)
    y_pred, y_scores = predict(clf, X_test)

    metrics = evaluate_detection(y_test, y_pred, y_scores)
    print("Detection performance on the test set:")
    for name, value in metrics.items():
        print(f"  {name:<9} {value:.3f}")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = config.OUTPUT_DIR / f"metrics_{tag}.json"
    out.write_text(json.dumps({"params": params, "metrics": metrics}, indent=2))
    print(f"Saved metrics to {out.relative_to(config.ROOT)}")

    viz.plot_score_histogram(y_test, y_pred, y_scores, filename=f"score_histogram_{tag}.png", show=args.show)
    viz.plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix for CatBoost ({tag})",
                              filename=f"confusion_matrix_{tag}.png", show=args.show)
    viz.plot_roc_curve(y_test, y_scores, title=f"ROC Curve for CatBoost ({tag})",
                       filename=f"roc_curve_{tag}.png", show=args.show)
    viz.plot_pr_curve(y_test, y_scores, title=f"Precision-Recall Curve for CatBoost ({tag})",
                      filename=f"pr_curve_{tag}.png", show=args.show)

    background = None
    if args.save_model:
        save_model(clf)
        print(f"Saved model to {config.MODEL.relative_to(config.ROOT)}")
        background = background_data(X_train, y_train)

    if args.shap:
        if background is None:
            background = np.load(config.BACKGROUND)
        feature_names = data.drop(columns=[config.TARGET_COL]).columns
        shap_explain(clf, X_test, y_test, feature_names, background, show=args.show)


if __name__ == "__main__":
    main()
