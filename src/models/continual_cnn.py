"""
src/models/continual_cnn.py
Thin wrapper around the baseline CNN for continual learning experiments.

No architectural changes — the same lightweight 1-D CNN (~24k params) is used.
This wrapper adds:
  - expand_output_head(): extend the softmax layer when new classes arrive
  - get_feature_extractor(): return a sub-model up to GlobalAveragePooling for probing
  - clone(): deep-copy weights into a new model instance (used by LwF)
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from src.models.cnn import build_cnn


def build_continual_cnn(n_features: int, n_classes: int,
                         lr: float = 1e-3) -> tf.keras.Model:
    """
    Build the baseline CNN for use in CL experiments.
    Identical architecture to build_cnn(); returns a compiled model.
    """
    return build_cnn(n_features, n_classes, lr=lr)


def get_feature_extractor(model: tf.keras.Model) -> tf.keras.Model:
    """
    Return a sub-model from input → GlobalAveragePooling output.
    Used for probing task-wise representation drift.
    """
    gap_output = model.get_layer('gap').output
    return tf.keras.Model(inputs=model.input, outputs=gap_output,
                          name='feature_extractor')


def expand_output_head(model: tf.keras.Model,
                       n_new_classes: int,
                       lr: float = 5e-4) -> tf.keras.Model:
    """
    Expand the output Dense layer from n_old to n_old + n_new_classes.

    The existing weights for previously-seen classes are preserved; the new
    class weights are initialised from the Glorot uniform distribution.

    Parameters
    ----------
    model         : trained model with current output head
    n_new_classes : number of new output units to add
    lr            : learning rate for the rebuilt model

    Returns
    -------
    New compiled model with expanded output (same feature extractor weights).
    """
    old_output = model.get_layer('output')
    n_old = old_output.units
    n_total = n_old + n_new_classes

    # Extract body weights (all except output layer)
    new_model = build_cnn(
        n_features=model.input_shape[1],
        n_classes=n_total,
        lr=lr,
    )

    # Copy all weights except the output layer
    for new_layer in new_model.layers:
        if new_layer.name == 'output':
            continue
        try:
            old_layer = model.get_layer(new_layer.name)
            new_layer.set_weights(old_layer.get_weights())
        except ValueError:
            pass   # layer not in old model — skip

    # Transfer old output weights, zero-pad for new classes
    old_kernel, old_bias = old_output.get_weights()  # (n_hidden, n_old), (n_old,)
    n_hidden = old_kernel.shape[0]

    init = tf.keras.initializers.GlorotUniform()
    new_kernel = np.concatenate([
        old_kernel,
        init(shape=(n_hidden, n_new_classes)).numpy()
    ], axis=1)
    new_bias = np.concatenate([old_bias, np.zeros(n_new_classes, dtype=np.float32)])

    new_model.get_layer('output').set_weights([new_kernel, new_bias])
    print(f"  [ContinualCNN] Output head expanded: {n_old} → {n_total} classes")
    return new_model
