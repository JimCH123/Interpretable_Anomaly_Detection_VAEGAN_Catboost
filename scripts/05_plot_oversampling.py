"""Step 5 - Plot detection performance against the oversampling ratio (dissertation Fig. 11).

Inputs : results/oversampling_ratio.csv
Outputs: outputs/figures/oversampling_ratio.png

The CSV holds the results reported in the dissertation. To reproduce one row, run
  python scripts/02_generate_anomalies.py --model vaegan --ratio 0.3
  python scripts/04_train_catboost.py     --model vaegan --ratio 0.3
"""
import argparse

import pandas as pd

from anomaly_detection import config, visualization as viz


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", action="store_true", help="display the figure as well as saving it")
    args = parser.parse_args()

    df = pd.read_csv(config.RESULTS_DIR / "oversampling_ratio.csv", index_col="OS Ratio")
    print(df.to_string())
    viz.plot_oversampling_ratio(df, show=args.show)


if __name__ == "__main__":
    main()
