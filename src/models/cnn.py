"""
Lightweight 1-D CNN for IoT network intrusion detection.
~24k parameters — deployable on edge hardware.
"""
import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn(n_features: int, n_classes: int,
              lr: float = 1e-3) -> tf.keras.Model:
    """
    Two convolutional blocks followed by a two-layer dense classification head.

    Architecture (§3.4.3)
    ---------------------
    Input (n_features, 1)
    → Conv1D(32, k=3, ReLU, same) + BatchNorm + MaxPool
    → Conv1D(64, k=3, ReLU, same) + BatchNorm + MaxPool
    → GlobalAveragePooling1D
    → Dense(128, ReLU) + Dropout(0.3)
    → Dense(64,  ReLU) + Dropout(0.3)
    → Dense(n_classes, softmax)

    Parameters
    ----------
    n_features : number of input features (after preprocessing)
    n_classes  : number of attack classes including normal
    lr         : Adam learning rate
    """
    inp = layers.Input(shape=(n_features, 1), name='input')

    # Block 1
    x = layers.Conv1D(32, 3, activation='relu', padding='same', name='conv1')(inp)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.MaxPooling1D(2, name='pool1')(x)

    # Block 2
    x = layers.Conv1D(64, 3, activation='relu', padding='same', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.MaxPooling1D(2, name='pool2')(x)

    # Classification head
    x = layers.GlobalAveragePooling1D(name='gap')(x)
    x = layers.Dense(128, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.3, name='drop1')(x)
    x = layers.Dense(64, activation='relu', name='dense2')(x)
    x = layers.Dropout(0.3, name='drop2')(x)
    out = layers.Dense(n_classes, activation='softmax', name='output')(x)

    model = models.Model(inputs=inp, outputs=out, name='LightweightCNN')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model
