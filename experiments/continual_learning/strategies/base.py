"""
experiments/continual_learning/strategies/base.py
Abstract base class for all continual learning strategies.
"""
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf


class CLStrategy(ABC):
    """
    All CL strategies share the same interface:
      - before_task(task_id, task)  — called once before training a new task
      - compute_loss(model, X, y)   — returns total loss tensor (CE + any penalty)
      - after_task(task_id, task, model) — called once after training completes
    """

    def __init__(self, model: tf.keras.Model, n_classes: int, seed: int = 42):
        self.model = model
        self.n_classes = n_classes
        self.seed = seed
        self.task_id = -1

    @abstractmethod
    def before_task(self, task_id: int, task) -> None:
        """Hook called before training on a new task begins."""

    @abstractmethod
    def compute_loss(self,
                     model: tf.keras.Model,
                     X: tf.Tensor,
                     y: tf.Tensor) -> tf.Tensor:
        """
        Compute the total training loss for a minibatch.
        Must include cross-entropy + any regularisation terms.
        """

    @abstractmethod
    def after_task(self, task_id: int, task, model: tf.keras.Model) -> None:
        """Hook called after training on a task completes."""

    def name(self) -> str:
        return self.__class__.__name__
