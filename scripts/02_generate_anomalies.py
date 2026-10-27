"""Step 2 - Generate synthetic anomalies with VAE or VAE-GAN (dissertation Section IV-C).

Inputs : data/processed/uci-secom-cleaned.csv
Outputs: data/synthetic/synthetic_<model>.csv      generated anomalies
         data/synthetic/combined_data_<model>.csv  real data + generated anomalies

--ratio scales the number of generated samples (1.0 = balance both classes);
non-default ratios get a "_r<ratio>" file suffix, e.g. combined_data_vaegan_r0.3.csv.
"""
import argparse

import pandas as pd

from anomaly_detection import config, visualization as viz
from anomaly_detection.generation import (MODELS, feature_columns, generate_anomaly_data,
                                          prepare_vae_data, set_seed)
from anomaly_detection.preprocessing import load_data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, default="vaegan")
    parser.add_argument("--ratio", type=float, default=1.0, help="oversampling ratio (default: 1.0)")
    parser.add_argument("--epochs", type=int, default=None, help="override the epochs set in config.py")
    parser.add_argument("--fix-adversarial", action="store_true",
                        help="VAE-GAN only: let the adversarial term update the decoder "
                             "(see VAEGAN docstring; off reproduces the dissertation)")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--show", action="store_true", help="display figures as well as saving them")
    args = parser.parse_args()

    set_seed(args.seed)
    data = load_data(config.CLEANED_DATA)
    print("Data loaded, shape:", data.shape)

    fail_df, synthetic_num = prepare_vae_data(data)
    synthetic_num = int(synthetic_num * args.ratio)
    print("Real anomalies used for training, shape:", fail_df.shape)
    print(f"Number of anomalies to generate (ratio {args.ratio:g}):", synthetic_num)

    if synthetic_num > 0:
        synthetic_df = generate_anomaly_data(args.model, fail_df, synthetic_num, feature_columns(data),
                                             epochs=args.epochs, adversarial_decoder=args.fix_adversarial)
    else:
        synthetic_df = data.iloc[:0].copy()

    synthetic_path = config.synthetic_path(args.model, args.ratio)
    synthetic_df.to_csv(synthetic_path, index=False)
    print(f"Synthetic anomalies {synthetic_df.shape} saved to {synthetic_path.relative_to(config.ROOT)}")

    combined_df = pd.concat([data, synthetic_df], ignore_index=True)
    combined_path = config.combined_path(args.model, args.ratio)
    combined_df.to_csv(combined_path, index=False)
    print(f"Combined data {combined_df.shape} saved to {combined_path.relative_to(config.ROOT)}")

    if synthetic_num > 0:
        viz.plot_kde_real_vs_generated(fail_df, synthetic_df, args.model, show=args.show)
        viz.plot_kpca_real_vs_generated(fail_df, synthetic_df, args.model, show=args.show)


if __name__ == "__main__":
    main()
