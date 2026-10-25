"""
Two-layer stacked LSTM for sequential network flow classification.
Note: temporal ordering is only meaningful when flows are from a single
continuous capture session (see §3.4.4 and §4.8 for caveats).
"""
import tensorflow as tf
from tensorflow.keras import layers, models


def build_lstm(n_features: int, n_classes: int,
               units1: int = 64, units2: int = 32,
               dense_units: int = 64,
               dropout_rate: float = 0.4,
               seq_len: int = 10,
               lr: float = 1e-3) -> tf.keras.Model:
    """
    Stacked LSTM with LayerNormalization (§3.4.4).

    Input shape: (seq_len, n_features) — a sliding window of consecutive flows.

    Parameters
    ----------
    n_features   : number of features per timestep
    n_classes    : number of output classes
    units1/2     : LSTM layer sizes
    dropout_rate : dropout applied after the dense layer
    seq_len      : sliding window length (must match make_sequences window)
    """
    inp = layers.Input(shape=(seq_len, n_features), name='input')

    x = layers.LSTM(units1, return_sequences=True, name='lstm1')(inp)
    x = layers.LayerNormalization(name='ln1')(x)
    x = layers.LSTM(units2, return_sequences=False, name='lstm2')(x)
    x = layers.LayerNormalization(name='ln2')(x)
    x = layers.Dense(dense_units, activation='relu', name='dense1')(x)
    x = layers.Dropout(dropout_rate, name='dropout')(x)
    out = layers.Dense(n_classes, activation='softmax', name='output')(x)

    model = models.Model(inputs=inp, outputs=out, name='LightweightLSTM')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def make_sequences(X: 'np.ndarray', y: 'np.ndarray',
                   window: int = 10) -> tuple:
    """
    Convert a flat 2-D array into overlapping windows for LSTM input.

    Returns
    -------
    X_seq : shape (N - window, window, n_features)
    y_seq : label of the LAST record in each window
    """
    import numpy as np
    seqs, labels = [], []
    for i in range(len(X) - window + 1):
        seqs.append(X[i: i + window])
        labels.append(y[i + window - 1])
    return np.array(seqs, dtype=np.float32), np.array(labels)
