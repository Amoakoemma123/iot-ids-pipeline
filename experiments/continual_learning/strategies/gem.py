"""
experiments/continual_learning/strategies/gem.py
Gradient Episodic Memory (GEM).

Reference: Lopez-Paz & Ranzato (2017) "Gradient Episodic Memory for
Continual Learning." NeurIPS 30.

GEM stores a small episodic memory M_k for each past task k. Before each
gradient update, it projects the current gradient g onto the feasible cone:
    g~ = argmin ||g~ - g||^2  subject to  <g~, g_k> >= 0  for all k

This ensures the update does not increase the loss on any previous task
(no catastrophic forgetting by construction), while still allowing the
model to improve on the new task.

Projection is solved via a dual quadratic programme on the memory gradients.
"""
import numpy as np
import tensorflow as tf
from .base import CLStrategy


def _project_gradients(g: np.ndarray, G: np.ndarray) -> np.ndarray:
    """
    Project gradient g onto the GEM feasibility cone defined by past-task
    gradients G (shape: n_tasks × n_params).

    Uses the dual QP formulation from Lopez-Paz & Ranzato (2017) Appendix A:
      min  0.5 * v^T G G^T v - g^T G^T v
      s.t. v >= 0
    Solved with gradient descent (projected GD) for efficiency on small memory.

    Parameters
    ----------
    g : (n_params,) current task gradient (flattened)
    G : (n_tasks, n_params) past-task reference gradients

    Returns
    -------
    g_proj : (n_params,) projected gradient
    """
    n_tasks = G.shape[0]
    # Check if projection is needed (all inner products >= 0 → already feasible)
    dots = G @ g  # (n_tasks,)
    if np.all(dots >= 0):
        return g  # no constraint violated

    # Dual QP via projected gradient descent
    GGT = G @ G.T   # (n_tasks, n_tasks)
    Gg  = G @ g     # (n_tasks,)
    v   = np.zeros(n_tasks, dtype=np.float64)
    lr  = 1.0 / (np.linalg.norm(GGT) + 1e-8)

    for _ in range(200):
        grad_v = GGT @ v - Gg
        v = np.maximum(0.0, v - lr * grad_v)

    # Recover primal
    g_proj = g - G.T @ v
    return g_proj.astype(np.float32)


class GEM(CLStrategy):
    """
    Gradient Episodic Memory.

    Parameters
    ----------
    memory_per_task : number of samples stored per past task
    """

    def __init__(self, model: tf.keras.Model, n_classes: int,
                 memory_per_task: int = 200,
                 seed: int = 42):
        super().__init__(model, n_classes, seed)
        self.memory_per_task = memory_per_task
        self._rng = np.random.default_rng(seed)

        # Per-task episodic memories
        self._mem_X: list = []   # list of arrays, one per past task
        self._mem_y: list = []

    # ── Interface ─────────────────────────────────────────────────────────────

    def before_task(self, task_id: int, task) -> None:
        self.task_id = task_id

    def compute_loss(self,
                     model: tf.keras.Model,
                     X: tf.Tensor,
                     y: tf.Tensor) -> tf.Tensor:
        """
        Compute CE loss and project gradients if any past task constraint
        is violated. Returns the (possibly modified) loss.

        Note: full GEM projection happens outside compute_loss via a custom
        training step in run_continual.py when GEM is active.
        For compatibility with the shared trainer, this returns plain CE loss;
        gradient projection is applied in the GEM-specific training loop.
        """
        logits = model(X, training=True)
        return tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y, logits)
        )

    def after_task(self, task_id: int, task, model: tf.keras.Model) -> None:
        """Store a fixed-size random subset of task data as episodic memory."""
        n = min(self.memory_per_task, len(task.X_train))
        idx = self._rng.choice(len(task.X_train), n, replace=False)
        self._mem_X.append(task.X_train[idx].copy())
        self._mem_y.append(task.y_train[idx].copy())
        print(f"  [GEM] Episodic memory: task {task_id} stored {n} samples "
              f"(total tasks in memory: {len(self._mem_X)})")

    # ── GEM-specific gradient step ─────────────────────────────────────────────

    def gem_gradient_step(self,
                           model: tf.keras.Model,
                           optimizer: tf.keras.optimizers.Optimizer,
                           X_new: tf.Tensor,
                           y_new: tf.Tensor) -> float:
        """
        Full GEM update step:
          1. Compute current-task gradient g
          2. Compute per-past-task reference gradients G
          3. Project g onto feasibility cone
          4. Apply projected gradient
        """
        if not self._mem_X:
            # Task 0 — no past tasks, plain gradient step
            with tf.GradientTape() as tape:
                logits = model(X_new, training=True)
                loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(y_new, logits)
                )
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return float(loss)

        # Step 1: current-task gradient
        with tf.GradientTape() as tape:
            logits = model(X_new, training=True)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y_new, logits)
            )
        grads_new = tape.gradient(loss, model.trainable_variables)
        g = np.concatenate([gv.numpy().flatten() for gv in grads_new
                            if gv is not None])

        # Step 2: reference gradients per past task
        G_rows = []
        for mem_X, mem_y in zip(self._mem_X, self._mem_y):
            idx = self._rng.choice(len(mem_X),
                                   size=min(32, len(mem_X)), replace=False)
            X_m = tf.constant(np.expand_dims(mem_X[idx], -1), dtype=tf.float32)
            y_m = tf.constant(mem_y[idx], dtype=tf.int32)
            with tf.GradientTape() as tape:
                logits_m = model(X_m, training=True)
                loss_m = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(y_m, logits_m)
                )
            grads_m = tape.gradient(loss_m, model.trainable_variables)
            gm = np.concatenate([gv.numpy().flatten() for gv in grads_m
                                  if gv is not None])
            G_rows.append(gm)
        G = np.stack(G_rows)   # (n_past_tasks, n_params)

        # Step 3: project
        g_proj = _project_gradients(g, G)

        # Step 4: write projected gradient back and apply
        ptr = 0
        proj_grads = []
        for gv in grads_new:
            if gv is None:
                proj_grads.append(None)
                continue
            size = gv.numpy().size
            proj_grads.append(
                tf.constant(g_proj[ptr:ptr+size].reshape(gv.shape), dtype=tf.float32)
            )
            ptr += size
        optimizer.apply_gradients(
            [(pg, v) for pg, v in zip(proj_grads, model.trainable_variables)
             if pg is not None]
        )
        return float(loss)
