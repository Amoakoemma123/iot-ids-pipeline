"""Random Forest model — strongest overall performer across all three benchmarks."""
from sklearn.ensemble import RandomForestClassifier


def build_rf(n_estimators: int = 200, seed: int = 42,
             class_weight=None) -> RandomForestClassifier:
    """
    Random Forest with 200 trees, max_depth=20, min_samples_leaf=2.
    Achieves highest macro F1 across TON_IoT, UNSW-NB15, and CICIDS-2017.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=-1,
        class_weight=class_weight,
        random_state=seed,
    )
