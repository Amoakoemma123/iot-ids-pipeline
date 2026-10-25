"""
src/training/trainer.py
Unified training interface for all five model architectures.
"""
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import joblib
import tensorflow as tf
from pathlib import Path

from ..models.lstm import make_sequences


# Suppress TF logging
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

# Default Keras callbacks (matches notebook: patience=15, ReduceLR patience=7)
def _default_callbacks(patience: int = 15):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0),
    ]


class Trainer:
    """
    Fits any of the five baseline models and saves artefacts.

    Usage
    -----
    trainer = Trainer(output_dir='models/')
    trainer.fit_sklearn(model, X_train, y_train, name='lr')
    trainer.fit_cnn(model, X_train, y_train, X_val, y_val)
    ...
    """

    def __init__(self, output_dir: str | Path = 'models'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Sklearn models (LR, RF) ───────────────────────────────────────────────

    def fit_sklearn(self, model, X_train: np.ndarray, y_train: np.ndarray,
                    name: str = 'model') -> object:
        """Fit an sklearn estimator and save to disk."""
        print(f"  Training {name}…")
        model.fit(X_train, y_train)
        out = self.output_dir / f'model_{name}.pkl'
        joblib.dump(model, out)
        print(f"  Saved → {out}")
        return model

    # ── CNN ───────────────────────────────────────────────────────────────────

    def fit_cnn(self, model: tf.keras.Model,
                X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                epochs: int = 100, batch_size: int = 256,
                patience: int = 15, name: str = 'cnn',
                class_weights: dict | None = None) -> tf.keras.Model:
        """Fit CNN. Input arrays are automatically reshaped to (N, n_features, 1)."""
        print(f"  Training {name}…")
        Xtr = X_train.reshape(-1, X_train.shape[1], 1)
        Xvl = X_val.reshape(-1, X_val.shape[1], 1)

        model.fit(
            Xtr, y_train,
            validation_data=(Xvl, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=_default_callbacks(patience),
            verbose=0,
        )
        out = self.output_dir / f'model_{name}.h5'
        model.save(str(out))
        print(f"  Saved → {out}")
        return model

    # ── LSTM ──────────────────────────────────────────────────────────────────

    def fit_lstm(self, model: tf.keras.Model,
                 X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 window: int = 10,
                 epochs: int = 100, batch_size: int = 128,
                 patience: int = 15,
                 subsample: int | None = 80_000,
                 name: str = 'lstm',
                 device: str = '/CPU:0',
                 class_weights: dict | None = None) -> tf.keras.Model:
        """
        Fit LSTM on windowed sequences.

        Parameters
        ----------
        subsample    : cap training sequences (GPU VRAM safety); None = use all
        device       : '/CPU:0' recommended to avoid VRAM OOM on GTX 1650
        class_weights: optional dict for imbalanced classes
        """
        print(f"  Training {name}…")
        X_seq, y_seq = make_sequences(X_train, y_train, window)
        X_vseq, y_vseq = make_sequences(X_val, y_val, window)

        if subsample is not None and len(X_seq) > subsample:
            idx = np.random.choice(len(X_seq), subsample, replace=False)
            X_seq, y_seq = X_seq[idx], y_seq[idx]

        with tf.device(device):
            model.fit(
                X_seq, y_seq,
                validation_data=(X_vseq, y_vseq),
                epochs=epochs,
                batch_size=batch_size,
                class_weight=class_weights,
                callbacks=_default_callbacks(patience),
                verbose=0,
            )
        out = self.output_dir / f'model_{name}.h5'
        model.save(str(out))
        print(f"  Saved → {out}")
        return model

    # ── AE-RF ─────────────────────────────────────────────────────────────────

    def fit_aerf(self, model,
                 X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray | None = None,
                 ae_epochs: int = 50, ae_batch: int = 256,
                 normal_class: int = 0,
                 name: str = 'aerf') -> object:
        """
        Fit Autoencoder-RF hybrid model.

        Parameters
        ----------
        normal_class : integer label for benign/normal traffic.
                       The AE is trained on this class only.
        """
        print(f"  Training {name}…")
        model.fit(X_train, y_train,
                  epochs=ae_epochs, batch_size=ae_batch,
                  X_val=X_val, normal_class=normal_class)
        # Save encoder (Keras) and RF (joblib) separately
        enc_out = self.output_dir / 'model_ae.h5'
        rf_out  = self.output_dir / 'model_aerf.pkl'
        model.autoencoder_.save(str(enc_out))
        joblib.dump(model.rf_, rf_out)
        print(f"  Saved → {enc_out}, {rf_out}")
        return model
