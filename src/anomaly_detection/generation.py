"""Synthetic anomaly generation with VAE or VAE-GAN (dissertation Section IV-C)."""
import numpy as np
import pandas as pd
import tensorflow as tf

from . import config
from .models import VAEGAN, VAEGenerator

MODELS = ("vae", "vaegan")


def prepare_vae_data(df, target_col=config.TARGET_COL):
    """Return the real anomalies (features only) and how many samples to generate.

    The default count (normal - anomaly) balances the two classes once the
    synthetic anomalies are added to the real ones.
    """
    fail_df = df[df[target_col] == config.ANOMALY_LABEL].select_dtypes(include=['float64', 'int64']).drop(target_col, axis=1)
    synthetic_num = df.shape[0] - len(fail_df) - len(fail_df)
    return fail_df, synthetic_num


def build_generator(model_name, input_dim, adversarial_decoder=False):
    """Create and compile a generator with the hyperparameters from ``config``."""
    if model_name == "vae":
        p = config.VAE_PARAMS
        model = VAEGenerator(input_dim=input_dim, latent_dim=p["latent_dim"], beta=p["beta"])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=p["learning_rate"]))
    elif model_name == "vaegan":
        p = config.VAEGAN_PARAMS
        model = VAEGAN(input_dim=input_dim, latent_dim=p["latent_dim"],
                       feature_layer_idx=p["feature_layer_idx"], beta=p["beta"], gamma=p["gamma"],
                       adversarial_decoder=adversarial_decoder)
        model.compile(
            enc_optimizer=tf.keras.optimizers.Adam(learning_rate=p["enc_learning_rate"]),
            dec_optimizer=tf.keras.optimizers.Adam(learning_rate=p["dec_learning_rate"]),
            discriminator_optimizer=tf.keras.optimizers.Adam(learning_rate=p["dis_learning_rate"])
        )
    else:
        raise ValueError(f"Unknown model '{model_name}', expected one of {MODELS}")
    return model, p


def generate_anomaly_data(model_name, fail_df, synthetic_num, feature_columns,
                          epochs=None, adversarial_decoder=False, target_col=config.TARGET_COL):
    """Train ``model_name`` on the real anomalies and sample ``synthetic_num`` new ones."""
    fail_data = fail_df.values
    model, params = build_generator(model_name, fail_data.shape[1], adversarial_decoder)
    model.fit(fail_data, epochs=epochs or params["epochs"], batch_size=params["batch_size"], verbose=1)

    synthetic_anomalies = model.generate(synthetic_num, scale=1)

    synthetic_df = pd.DataFrame(synthetic_anomalies, columns=feature_columns)
    synthetic_df[target_col] = config.ANOMALY_LABEL
    return synthetic_df


def feature_columns(df, target_col=config.TARGET_COL):
    return df.select_dtypes(include=['float64', 'int64']).drop(labels=[target_col], axis=1).columns


def set_seed(seed=config.SEED):
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
