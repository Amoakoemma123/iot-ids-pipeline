"""
experiments/continual_learning/strategies/lwf.py
Learning without Forgetting (LwF).

Reference: Li & Hoiem (2018) "Learning without Forgetting."
IEEE TPAMI 40(12):2935-2947.

LwF uses knowledge distillation: before training on a new task, the old model
predictions (soft targets) on the new task's data are recorded. During training,
the new model is regularised to match those soft targets, preserving the
output distribution for previously learned classes.

Loss = (1 - α) * CE(f(x), y_new) + α * KL(σ(f_old(x)/T) || σ(f(x)/T))
where T = temperature, α = distillation weight
"""
import numpy as np
import tensorflow as tf
from .base import CLStrategy


class LwF(CLStrategy):
    """
    Learning without Forgetting via output distillation.

    Parameters
    ----------
    temperature : softening temperature for old-model logits (default 2.0)
    alpha       : weight of distillation loss (1-alpha = CE weight)
    """

    def __init__(self, model: tf.keras.Model, n_classes: int,
                 temperature: float = 2.0,
                 alpha: float = 0.5,
                 seed: int = 42):
        super().__init__(model, n_classes, seed)
        self.temperature = temperature
        self.alpha       = alpha
        self._old_logits_cache = {}   # {sample_hash: logit vector} — keyed by batch index

        # Snapshot of old model weights (set in before_task)
        self._old_weights = None

    # ── Interface ─────────────────────────────────────────────────────────────

    def before_task(self, task_id: int, task) -> None:
        """Save a snapshot of the current model weights as the 'teacher'."""
        self.task_id = task_id
        if task_id > 0:
            self._old_weights = [w.numpy().copy()
                                  for w in self.model.trainable_variables]
            print(f"  [LwF] Teacher snapshot saved before task {task_id}")

    def compute_loss(self,
                     model: tf.keras.Model,
                     X: tf.Tensor,
                     y: tf.Tensor) -> tf.Tensor:
        """CE loss on current task + KL distillation from frozen teacher."""
        logits = model(X, training=True)
        ce_loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y, logits)
        )

        if self._old_weights is None or self.task_id == 0:
            return ce_loss

        # Teacher forward pass (frozen weights applied temporarily)
        old_logits = self._teacher_predict(model, X)

        # Soft targets: KL divergence between softened old and new logits
        T = self.temperature
        old_soft = tf.nn.softmax(old_logits / T)
        new_soft = tf.nn.log_softmax(logits / T)
        kl_loss  = tf.reduce_mean(
            tf.keras.losses.KLDivergence()(old_soft, tf.exp(new_soft))
        ) * (T ** 2)   # scale KL by T^2 as per Hinton et al.

        return (1.0 - self.alpha) * ce_loss + self.alpha * kl_loss

    def after_task(self, task_id: int, task, model: tf.keras.Model) -> None:
        pass   # snapshot is taken in before_task of the *next* task

    # ── Internals ─────────────────────────────────────────────────────────────

    def _teacher_predict(self, model: tf.keras.Model,
                         X: tf.Tensor) -> tf.Tensor:
        """
        Run X through the teacher (old weights) without modifying the live model.
        We temporarily swap weights, run inference, then restore.
        """
        current_weights = [w.numpy().copy() for w in model.trainable_variables]
        for w, old_w in zip(model.trainable_variables, self._old_weights):
            w.assign(old_w)
        old_logits = model(X, training=False)
        for w, cur_w in zip(model.trainable_variables, current_weights):
            w.assign(cur_w)
        return old_logits
