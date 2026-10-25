"""
experiments/continual_learning/strategies/replay.py
Experience Replay with a fixed-size ring buffer.

A random subset of each completed task's training data is stored in a
memory buffer. When training on a new task, each minibatch is augmented
with samples drawn uniformly from the buffer (replay_ratio controls the mix).

Buffer management: reservoir sampling ensures uniform coverage across all
seen tasks regardless of class imbalance within each task.
"""
import numpy as np
import tensorflow as tf
from .base import CLStrategy


class ExperienceReplay(CLStrategy):
    """
    Fixed-size episodic memory with reservoir sampling.

    Parameters
    ----------
    memory_size  : total number of samples stored across all tasks
    replay_ratio : fraction of each training batch drawn from replay memory
    """

    def __init__(self, model: tf.keras.Model, n_classes: int,
                 memory_size: int = 2000,
                 replay_ratio: float = 0.3,
                 seed: int = 42):
        super().__init__(model, n_classes, seed)
        self.memory_size  = memory_size
        self.replay_ratio = replay_ratio
        self._rng         = np.random.default_rng(seed)

        # Ring buffer (pre-allocated when first task is seen)
        self._mem_X    = None
        self._mem_y    = None
        self._mem_ptr  = 0       # next write position
        self._mem_n    = 0       # samples currently in buffer

    # ── Interface ─────────────────────────────────────────────────────────────

    def before_task(self, task_id: int, task) -> None:
        self.task_id = task_id
        if self._mem_X is None and task_id > 0:
            return   # buffer was never seeded — nothing to replay

    def compute_loss(self,
                     model: tf.keras.Model,
                     X: tf.Tensor,
                     y: tf.Tensor) -> tf.Tensor:
        """
        Mix current-task minibatch with replay samples, compute CE loss on all.
        """
        if self._mem_n == 0 or self.task_id == 0:
            # No replay data yet — pure CE on current batch
            logits = model(X, training=True)
            return tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y, logits)
            )

        # Sample from memory
        n_replay = max(1, int(self.replay_ratio * X.shape[0]))
        X_rep, y_rep = self._sample_memory(n_replay)
        # X arrives as [B, n_features, 1] (expanded in train loop); match rank
        X_rep = tf.expand_dims(tf.constant(X_rep, dtype=tf.float32), -1)
        y_rep = tf.constant(y_rep, dtype=tf.int32)

        # Concatenate current + replay
        X_mix = tf.concat([X, X_rep], axis=0)
        y_mix = tf.concat([y, y_rep], axis=0)

        logits = model(X_mix, training=True)
        return tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y_mix, logits)
        )

    def after_task(self, task_id: int, task, model: tf.keras.Model) -> None:
        """Store a representative subset of task data into the buffer."""
        self._update_memory(task.X_train, task.y_train)
        print(f"  [Replay] Buffer updated after task {task_id}: "
              f"{self._mem_n}/{self.memory_size} samples stored")

    # ── Internals ─────────────────────────────────────────────────────────────

    def _update_memory(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Reservoir sampling: each new sample has equal probability of being
        stored, maintaining uniform coverage across all tasks seen so far.
        """
        n_features = X.shape[1]
        if self._mem_X is None:
            self._mem_X = np.zeros((self.memory_size, n_features), dtype=np.float32)
            self._mem_y = np.zeros(self.memory_size, dtype=np.int32)

        for i in range(len(X)):
            if self._mem_n < self.memory_size:
                # Buffer not full — append directly
                self._mem_X[self._mem_n] = X[i]
                self._mem_y[self._mem_n] = y[i]
                self._mem_n += 1
            else:
                # Reservoir sampling: replace with probability memory_size / (n+1)
                j = self._rng.integers(0, self._mem_n + i + 1)
                if j < self.memory_size:
                    self._mem_X[j] = X[i]
                    self._mem_y[j] = y[i]

    def _sample_memory(self, n: int):
        """Sample n rows uniformly from the current buffer."""
        idx = self._rng.choice(self._mem_n, size=min(n, self._mem_n), replace=False)
        return self._mem_X[idx], self._mem_y[idx]
