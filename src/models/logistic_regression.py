"""Logistic Regression baseline model."""
from sklearn.linear_model import LogisticRegression


def build_lr(seed: int = 42, class_weight=None) -> LogisticRegression:
    """
    Multi-class Logistic Regression (one-vs-rest).
    Serves as the computational baseline for comparison.
    """
    return LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        multi_class='auto',
        class_weight=class_weight,
        n_jobs=1,
        random_state=seed,
    )
