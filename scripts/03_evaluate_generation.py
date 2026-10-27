"""Step 3 - Compare VAE and VAE-GAN anomalies with real ones (dissertation Section V-B).

Inputs : data/processed/anomaly_data.csv, data/synthetic/synthetic_{vae,vaegan}.csv
Outputs: outputs/generation_distances.csv, outputs/figures/generation_distance_comparison.png
"""
import argparse

import pandas as pd

from anomaly_detection import config, visualization as viz
from anomaly_detection.distance_metrics import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", action="store_true", help="display figures as well as saving them")
    args = parser.parse_args()

    anomaly_df = pd.read_csv(config.ANOMALY_DATA)
    print("Real anomalies, shape:", anomaly_df.shape)

    rows = []
    for model in ("vae", "vaegan"):
        synthetic_df = pd.read_csv(config.synthetic_path(model))
        print(f"Synthetic {model.upper()} anomalies, shape:", synthetic_df.shape)
        for metric, (mean, total) in compare(anomaly_df, synthetic_df).items():
            rows.append({"model": model, "metric": metric, "mean": mean, "total": total})

    results = pd.DataFrame(rows)
    table = results.pivot(index="metric", columns="model", values="total").reindex(
        ["KL Divergence", "JS Divergence", "Wasserstein Distance", "MMD"])
    print("\nTotal distance over all features (lower is closer to the real anomalies):")
    print(table.round(3).to_string())

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = config.OUTPUT_DIR / "generation_distances.csv"
    results.to_csv(out, index=False)
    print(f"\nSaved to {out.relative_to(config.ROOT)}")

    viz.plot_distance_comparison(table["vae"].values, table["vaegan"].values, show=args.show)


if __name__ == "__main__":
    main()
