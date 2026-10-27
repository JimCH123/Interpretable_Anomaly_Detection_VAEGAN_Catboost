"""Baseline VAE used to generate synthetic anomalies (dissertation Section IV-C, Eq. 7)."""
import tensorflow as tf
from tensorflow.keras import Sequential, layers, Model, metrics


class VAEGenerator(Model):
    """Fully connected VAE: encoder 512→32 (+ latent), decoder mirrors the encoder.

    Loss: ``L = L_recon + beta * L_prior`` (sum-of-squares reconstruction + KL).
    """

    def __init__(self, input_dim, latent_dim, beta=0.01):
        super(VAEGenerator, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta

        self.total_loss_tracker = metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = metrics.Mean(name='kl_loss')

        self.metrics_list = [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]

        self.encoder = Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dense(latent_dim + latent_dim)
        ])
        self.decoder = Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(center=False, scale=False),
            layers.Dense(input_dim, activation=None)
        ])

    @property
    def metrics(self):
        return self.metrics_list

    def encode(self, x):
        mean_log_var = self.encoder(x)
        mean, log_var = tf.split(mean_log_var, num_or_size_splits=2, axis=1)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * eps

    def decode(self, z):
        return self.decoder(z)

    def call(self, input):
        mean, log_var = self.encode(input)
        z = self.reparameterize(mean, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mean, log_var

    def train_step(self, data):
        with tf.GradientTape() as tape:
            reconstructed, mean, log_var = self(data, training=True)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(data - reconstructed), axis=1)
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1)
            )
            total_loss = reconstruction_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def generate(self, num_samples, scale):
        """Decode ``num_samples`` latent vectors drawn from N(0, scale^2 I)."""
        z_samples = tf.random.normal([num_samples, self.latent_dim]) * scale
        synthetic_data = self.decoder(z_samples).numpy()
        return synthetic_data
