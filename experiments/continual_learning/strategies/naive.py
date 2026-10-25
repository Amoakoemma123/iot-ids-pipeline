"""
experiments/continual_learning/strategies/naive.py
Naive fine-tuning baseline — no forgetting protection.

The model is simply trained on each new task's data without any
regularisation or replay. This provides the lower bound on retention
(maximum catastrophic forgetting) against which EWC and Replay are compared.
"""
import tensorflow as tf
from .base import CLStrategy


class NaiveFineTuning(CLStrategy):
    """
    Standard SGD fine-tuning on each new task.
    No memory, no penalty — catastrophic forgetting baseline.
    """

    def before_task(self, task_id: int, task) -> None:
        self.task_id = task_id

    def compute_loss(self,
                     model: tf.keras.Model,
                     X: tf.Tensor,
                     y: tf.Tensor) -> tf.Tensor:
        logits = model(X, training=True)
        return tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y, logits)
        )

    def after_task(self, task_id: int, task, model: tf.keras.Model) -> None:
        pass   # nothing to consolidate
