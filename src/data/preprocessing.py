"""
src/data/preprocessing.py
Two-stage feature selection (MI → RFECV) + MinMax scaling + SMOTE.
Mirrors the pipeline described in §3.2 of the paper.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import joblib
from pathlib import Path

from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE


class PreprocessingPipeline:
    """
    Fit on training data; transform train / val / test identically.

    Stages
    ------
    1. Mutual Information (MI) — drop features with score <= mi_threshold
    2. RFECV               — recursive elimination with CV to find optimal subset
    3. MinMax scaling      — scale to [0, 1]
    4. SMOTE               — oversample minority classes (training set only)

    Parameters
    ----------
    mi_threshold      : minimum MI score to keep a feature (paper uses 0.05)
    rfecv_subsample   : max samples used to fit RFECV (50k as in notebook for speed)
    rfecv_cv          : number of CV folds for RFECV
    smote_k           : k_neighbors for SMOTE
    smote_max_per_class: if set, cap each class at this count during SMOTE so that
                         already-large classes are not oversampled and rare classes
                         are oversampled to this target. Prevents RF model explosion
                         on severely imbalanced datasets (CICIDS-2017, UNSW-NB15).
    mi_subsample      : if set, subsample training data to this many rows before
                         computing MI scores (for speed on large datasets like CICIDS2017).
    seed              : random seed
    """

    def __init__(self,
                 mi_threshold: float = 0.05,
                 rfecv_subsample: int = 50_000,
                 rfecv_cv: int = 3,
                 skip_rfecv: bool = False,
                 smote_k: int = 5,
                 smote_max_per_class: int | None = None,
                 mi_subsample: int | None = None,
                 seed: int = 42):

        self.mi_threshold       = mi_threshold
        self.rfecv_subsample    = rfecv_subsample
        self.rfecv_cv           = rfecv_cv
        self.skip_rfecv         = skip_rfecv
        self.smote_k            = smote_k
        self.smote_max_per_class= smote_max_per_class
        self.mi_subsample       = mi_subsample
        self.seed               = seed

        # Fitted artefacts (set during .fit())
        self.mi_mask_:           np.ndarray | None = None
        self.rfecv_:             RFECV | None      = None
        self.scaler_:            MinMaxScaler | None = None
        self.selected_features_: list | None        = None
        self.n_features_:        int               = 0

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            feature_names: list | None = None) -> 'PreprocessingPipeline':
        """Fit all pipeline stages on training data. Returns self."""
        names = feature_names or [f'f{i}' for i in range(X_train.shape[1])]

        # Stage 1: Mutual Information filter
        print("  [Pipeline] Stage 1: Mutual Information filter…")
        if self.mi_subsample is not None and len(X_train) > self.mi_subsample:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(X_train), self.mi_subsample, replace=False)
            X_mi_fit, y_mi_fit = X_train[idx], y_train[idx]
            print(f"    MI subsample: {len(X_train):,} → {self.mi_subsample:,} rows")
        else:
            X_mi_fit, y_mi_fit = X_train, y_train
        mi_scores = mutual_info_classif(X_mi_fit, y_mi_fit, random_state=self.seed)
        self.mi_mask_ = mi_scores > self.mi_threshold
        X_mi     = X_train[:, self.mi_mask_]
        names_mi = [n for n, keep in zip(names, self.mi_mask_) if keep]
        print(f"    MI: {X_train.shape[1]} → {X_mi.shape[1]} features")

        # Stage 2: RFECV (subsample for speed, as in notebook)
        if self.skip_rfecv:
            print(f"  [Pipeline] Stage 2: RFECV skipped — keeping all {X_mi.shape[1]} MI-filtered features")
            self.rfecv_ = None
            self.selected_features_ = names_mi
            self.n_features_ = len(names_mi)
            X_rfecv_out = X_mi
        else:
            print(f"  [Pipeline] Stage 2: RFECV (subsample={min(self.rfecv_subsample, len(X_mi)):,})…")
            if len(X_mi) > self.rfecv_subsample:
                rng = np.random.default_rng(self.seed)
                idx = rng.choice(len(X_mi), self.rfecv_subsample, replace=False)
                X_rfecv_fit, y_rfecv_fit = X_mi[idx], y_train[idx]
            else:
                X_rfecv_fit, y_rfecv_fit = X_mi, y_train

            estimator = RandomForestClassifier(
                n_estimators=50, n_jobs=-1, random_state=self.seed)
            self.rfecv_ = RFECV(
                estimator=estimator,
                cv=StratifiedKFold(self.rfecv_cv),
                step=1,
                n_jobs=-1,
                scoring='f1_macro',
                min_features_to_select=10,
            )
            self.rfecv_.fit(X_rfecv_fit, y_rfecv_fit)

            self.selected_features_ = [n for n, s in zip(names_mi, self.rfecv_.support_) if s]
            self.n_features_ = len(self.selected_features_)
            print(f"    RFECV: {X_mi.shape[1]} → {self.n_features_} features selected")
            X_rfecv_out = X_mi

        # Stage 3: MinMax Scaling (fit on full training set after selection)
        print("  [Pipeline] Stage 3: MinMax scaling…")
        X_sel = X_rfecv_out if self.skip_rfecv else self.rfecv_.transform(X_mi)
        self.scaler_ = MinMaxScaler()
        self.scaler_.fit(X_sel)

        return self

    def transform(self, X: np.ndarray, apply_smote: bool = False,
                  y: np.ndarray | None = None) -> tuple:
        """
        Apply fitted pipeline to X (and optionally y).

        Returns
        -------
        (X_out, y_out) — y_out equals input y unless SMOTE was applied
        """
        self._check_fitted()
        X_mi     = X[:, self.mi_mask_]
        skip_rfecv = getattr(self, 'skip_rfecv', False)
        X_rfecv  = X_mi if skip_rfecv else self.rfecv_.transform(X_mi)
        X_scaled = self.scaler_.transform(X_rfecv)

        if apply_smote:
            if y is None:
                raise ValueError("y must be provided when apply_smote=True")
            X_scaled, y = self._apply_smote(X_scaled, y)

        return X_scaled.astype(np.float32), y

    def fit_transform(self, X_train: np.ndarray, y_train: np.ndarray,
                      feature_names: list | None = None,
                      apply_smote: bool = True) -> tuple:
        """Convenience: fit then transform training data."""
        self.fit(X_train, y_train, feature_names)
        return self.transform(X_train, apply_smote=apply_smote, y=y_train)

    def save(self, path: str | Path) -> None:
        """Persist the fitted pipeline to disk."""
        joblib.dump(self, path)
        print(f"  Pipeline saved → {path}")

    @staticmethod
    def load(path: str | Path) -> 'PreprocessingPipeline':
        return joblib.load(path)

    # ── internals ─────────────────────────────────────────────────────────────

    def _apply_smote(self, X: np.ndarray, y: np.ndarray) -> tuple:
        print("  [Pipeline] Stage 4: SMOTE oversampling…")
        counts = np.bincount(y)

        smote_max = getattr(self, 'smote_max_per_class', None)
        if smote_max is not None:
            # Build a sampling_strategy that only oversamples classes below the cap.
            # Classes already above smote_max_per_class are left untouched.
            cap = smote_max
            majority_count = int(counts.max())
            target = min(cap, majority_count)   # can't exceed majority class size
            strategy = {
                cls: target
                for cls, cnt in enumerate(counts)
                if cnt < target and cnt >= self.smote_k + 1   # SMOTE requires ≥ k+1 samples
            }
            if not strategy:
                print(f"    SMOTE: no classes need oversampling (all ≥ cap={cap})")
                return X.astype(np.float32), y
            smote = SMOTE(sampling_strategy=strategy,
                          k_neighbors=self.smote_k,
                          random_state=self.seed)
        else:
            smote = SMOTE(k_neighbors=self.smote_k, random_state=self.seed)

        X_res, y_res = smote.fit_resample(X, y)
        print(f"    SMOTE: {len(X):,} → {len(X_res):,} samples")
        return X_res.astype(np.float32), y_res

    def _check_fitted(self):
        if self.scaler_ is None:
            raise RuntimeError("Pipeline not fitted. Call .fit() first.")
