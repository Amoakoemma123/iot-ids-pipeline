"""
Hybrid Autoencoder + Random Forest (AE-RF) model.

Stage 1: Train the autoencoder on NORMAL (benign) traffic only — anomaly detection style.
Stage 2: Use the frozen encoder as a feature extractor, then train
         a Random Forest on the encoded representations (supervised, all classes).

Architecture (§3.4.5)
---------------------
Encoder: GaussianNoise(0.1) → Dense(20, ReLU) → Dense(16, ReLU) [latent]
Decoder: Dense(20, ReLU) → Dense(n_features, linear)
RF     : n_estimators=200, max_depth=20, min_samples_leaf=2
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import GaussianNoise
from sklearn.ensemble import RandomForestClassifier


def build_autoencoder(n_features: int,
                      encoding_dim: int = 16) -> tuple:
    """
    Returns (autoencoder, encoder) — both Keras models.

    Parameters
    ----------
    n_features   : input / output dimensionality
    encoding_dim : size of the bottleneck latent space (16 as used in paper)
    """
    inp = layers.Input(shape=(n_features,), name='ae_input')

    # Encoder
    x = GaussianNoise(0.1)(inp)
    x = layers.Dense(20, activation='relu', name='enc1')(x)
    latent = layers.Dense(encoding_dim, activation='relu', name='latent')(x)

    # Decoder
    x = layers.Dense(20, activation='relu', name='dec1')(latent)
    reconstruction = layers.Dense(n_features, activation='linear', name='ae_output')(x)

    autoencoder = models.Model(inputs=inp, outputs=reconstruction, name='Autoencoder')
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

    encoder = models.Model(inputs=inp, outputs=latent, name='Encoder')
    return autoencoder, encoder


class AutoencoderRF:
    """
    Hybrid AE-RF pipeline.

    The autoencoder is trained on normal (benign) traffic only.
    The RF classifier is trained on encoder representations of the full training set.
    """

    def __init__(self, n_features: int, encoding_dim: int = 16,
                 n_estimators: int = 200, seed: int = 42, class_weight=None):
        self.n_features   = n_features
        self.encoding_dim = encoding_dim
        self.seed         = seed
        self.autoencoder_, self.encoder_ = build_autoencoder(n_features, encoding_dim)
        self.rf_ = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight=class_weight,
            random_state=seed,
        )
        self.normal_class_ = None   # set during fit

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            epochs: int = 50, batch_size: int = 256,
            X_val: np.ndarray | None = None,
            normal_class: int = 0) -> 'AutoencoderRF':
        """
        Parameters
        ----------
        X_train      : scaled training features (all classes)
        y_train      : integer class labels
        normal_class : integer label that represents benign/normal traffic
        """
        self.normal_class_ = normal_class

        # Stage 1: train AE on normal traffic only
        normal_mask = (y_train == normal_class)
        X_normal = X_train[normal_mask]
        print(f"  [AE-RF] Training autoencoder on {len(X_normal):,} normal samples…")

        val_data = (X_val, X_val) if X_val is not None else None
        self.autoencoder_.fit(
            X_normal, X_normal,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1 if val_data is None else 0.0,
            validation_data=val_data,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True)],
            verbose=0,
        )

        # Stage 2: encode full training set → train RF
        print("  [AE-RF] Training Random Forest on encoded features…")
        X_enc = self.encoder_.predict(X_train, verbose=0)
        self.rf_.fit(X_enc, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_enc = self.encoder_.predict(X, verbose=0)
        return self.rf_.predict(X_enc)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_enc = self.encoder_.predict(X, verbose=0)
        return self.rf_.predict_proba(X_enc)


def build_autoencoder_rf(n_features: int, n_classes: int = None,
                         encoding_dim: int = 16, seed: int = 42,
                         class_weight=None) -> AutoencoderRF:
    """Factory function consistent with other build_* interfaces."""
    return AutoencoderRF(n_features=n_features, encoding_dim=encoding_dim,
                         seed=seed, class_weight=class_weight)
