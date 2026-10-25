"""
experiments/continual_learning/strategies/ewc.py
Elastic Weight Consolidation (EWC).

Reference: Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in
neural networks." PNAS 114(13):3521-3526.

EWC adds a quadratic penalty to the loss that discourages large changes to
weights that were important for previous tasks. Importance is measured by the
diagonal of the Fisher Information Matrix (FIM), estimated on a random sample
of the previous task's training data.

Loss = CE(f(x), y) + (λ/2) * Σ_i F_i * (θ_i - θ*_i)^2
where F_i  = diagonal FIM entry for weight i
      θ*_i = optimal weight value after previous task
      λ    = regularisation strength (lambda_ewc in config)
"""
import numpy as np
import tensorflow as tf
from .base import CLStrategy


class EWC(CLStrategy):
    """
    Online EWC: accumulates Fisher and anchor weights across all seen tasks.
    Each new task adds its FIM to the running total, so the penalty grows with
    the number of tasks seen.
    """

    def __init__(self, model: tf.keras.Model, n_classes: int,
                 lambda_ewc: float = 5000,
                 fisher_samples: int = 500,
                 seed: int = 42):
        super().__init__(model, n_classes, seed)
        self.lambda_ewc     = lambda_ewc
        self.fisher_samples = fisher_samples

        # Accumulated Fisher and anchors across all tasks
        # Each element is a list of per-layer tensors
        self._fisher_accum = None   # summed FIM diagonal (list of np arrays)
        self._anchors      = None   # θ* after the last consolidated task

    # ── Interface ─────────────────────────────────────────────────────────────

    def before_task(self, task_id: int, task) -> None:
        self.task_id = task_id

    def compute_loss(self,
                     model: tf.keras.Model,
                     X: tf.Tensor,
                     y: tf.Tensor) -> tf.Tensor:
        logits = model(X, training=True)
        ce_loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y, logits)
        )

        if self._fisher_accum is None:
            return ce_loss   # no previous task yet

        ewc_penalty = self._compute_ewc_penalty(model)
        return ce_loss + (self.lambda_ewc / 2.0) * ewc_penalty

    def after_task(self, task_id: int, task, model: tf.keras.Model) -> None:
        """Estimate FIM on a subsample of task data and update accumulators."""
        print(f"  [EWC] Estimating Fisher Information Matrix after task {task_id}…")
        rng = np.random.default_rng(self.seed + task_id)
        n = min(self.fisher_samples, len(task.X_train))
        idx = rng.choice(len(task.X_train), n, replace=False)
        X_sample = task.X_train[idx]
        y_sample = task.y_train[idx]

        fisher_diag = self._estimate_fisher(model, X_sample, y_sample)

        # Accumulate (sum) Fisher across tasks
        if self._fisher_accum is None:
            self._fisher_accum = fisher_diag
        else:
            self._fisher_accum = [f + g
                                   for f, g in zip(self._fisher_accum, fisher_diag)]

        # Store current weights as anchors
        self._anchors = [w.numpy().copy() for w in model.trainable_variables]
        print(f"  [EWC] FIM updated — {len(self._anchors)} weight tensors consolidated")

    # ── Internals ─────────────────────────────────────────────────────────────

    def _estimate_fisher(self, model, X_sample, y_sample):
        """
        Diagonal FIM via squared gradients of log-likelihood.
        F_i = E[ (∂ log p(y|x,θ) / ∂θ_i)^2 ]
        Approximated by averaging over fisher_samples data points.
        """
        fisher_diag = [np.zeros(w.shape, dtype=np.float32)
                       for w in model.trainable_variables]

        dataset = tf.data.Dataset.from_tensor_slices(
            (X_sample.astype(np.float32), y_sample.astype(np.int32))
        ).batch(32)

        n_batches = 0
        for X_b, y_b in dataset:
            X_b_exp = tf.expand_dims(X_b, -1)
            with tf.GradientTape() as tape:
                logits = model(X_b_exp, training=False)
                log_probs = tf.nn.log_softmax(logits)
                # Use true label log-probability as proxy for log-likelihood
                idx_range = tf.range(tf.shape(y_b)[0])
                indices = tf.stack([idx_range, y_b], axis=1)
                log_likelihood = tf.reduce_mean(tf.gather_nd(log_probs, indices))

            grads = tape.gradient(log_likelihood, model.trainable_variables)
            for i, g in enumerate(grads):
                if g is not None:
                    fisher_diag[i] += g.numpy() ** 2
            n_batches += 1

        # Average over batches
        fisher_diag = [f / max(n_batches, 1) for f in fisher_diag]
        return fisher_diag

    def _compute_ewc_penalty(self, model) -> tf.Tensor:
        """
        Σ_i F_i * (θ_i - θ*_i)^2  summed over all parameters.
        """
        penalty = tf.constant(0.0)
        for w, anchor, fisher in zip(
                model.trainable_variables, self._anchors, self._fisher_accum):
            diff = w - tf.constant(anchor, dtype=tf.float32)
            f    = tf.constant(fisher, dtype=tf.float32)
            penalty += tf.reduce_sum(f * tf.square(diff))
        return penalty
