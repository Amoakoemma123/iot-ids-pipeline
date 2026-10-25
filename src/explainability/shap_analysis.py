"""
src/explainability/shap_analysis.py
SHAP-based explainability for all five baseline models (§4.4).
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def _shap_values_sklearn(model, X_background: np.ndarray,
                         X_explain: np.ndarray) -> np.ndarray:
    """TreeExplainer for RF; LinearExplainer for LR."""
    import shap
    if hasattr(model, 'estimators_'):          # Random Forest
        explainer = shap.TreeExplainer(model, X_background)
    else:                                       # Logistic Regression
        explainer = shap.LinearExplainer(model, X_background,
                                         feature_perturbation='interventional')
    sv = explainer.shap_values(X_explain)
    # For multi-class TreeExplainer returns a list; stack → (N, F, C)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    return sv


def _shap_values_keras(model, X_background: np.ndarray,
                       X_explain: np.ndarray,
                       reshape_fn=None) -> np.ndarray:
    """DeepExplainer for Keras (CNN / LSTM) models."""
    import shap
    import tensorflow as tf

    bg = X_background if reshape_fn is None else reshape_fn(X_background)
    ex = X_explain   if reshape_fn is None else reshape_fn(X_explain)

    explainer = shap.DeepExplainer(model, bg)
    sv = explainer.shap_values(ex)              # list[C] each (N, F[, 1])
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    # Collapse trailing reshape dim if present (CNN: (N,F,1,C) → (N,F,C))
    if sv.ndim == 4:
        sv = sv[:, :, 0, :]
    return sv


def _shap_values_aerf(model, X_background: np.ndarray,
                      X_explain: np.ndarray) -> np.ndarray:
    """KernelExplainer for the AE-RF hybrid (slow; uses small background)."""
    import shap
    # Use a plain numpy subsample — kmeans DenseData conflicts with the AE encoder step
    bg_summary = X_background[:min(50, len(X_background))]
    explainer  = shap.KernelExplainer(model.predict_proba, bg_summary)
    sv = explainer.shap_values(X_explain, nsamples=100)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    return sv


def global_importance(shap_values: np.ndarray,
                      feature_names: list) -> dict:
    """
    Mean |SHAP| per feature, collapsed across classes.

    Parameters
    ----------
    shap_values  : (N, n_features) or (N, n_features, n_classes)
    feature_names: list of length n_features

    Returns
    -------
    dict sorted by importance descending: {feature_name: mean_abs_shap}
    """
    sv = np.abs(shap_values)
    if sv.ndim == 3:
        sv = sv.mean(axis=-1)       # average over classes → (N, F)
    mean_sv = sv.mean(axis=0)       # → (F,)
    ranked  = sorted(zip(feature_names, mean_sv.tolist()),
                     key=lambda x: x[1], reverse=True)
    return dict(ranked)


def per_class_importance(shap_values: np.ndarray,
                         feature_names: list,
                         class_names: list) -> dict:
    """
    Mean |SHAP| per feature per class.

    Returns
    -------
    dict: {class_name: {feature_name: mean_abs_shap}}
    """
    if shap_values.ndim == 2:
        # Binary or single output → wrap in class dim
        sv = shap_values[:, :, np.newaxis]
    else:
        sv = shap_values                        # (N, F, C)

    result = {}
    for c, cname in enumerate(class_names):
        mean_sv = np.abs(sv[:, :, c]).mean(axis=0)
        ranked  = sorted(zip(feature_names, mean_sv.tolist()),
                         key=lambda x: x[1], reverse=True)
        result[cname] = dict(ranked)
    return result


def run_shap(model_name: str,
             model,
             X_train: np.ndarray,
             X_test: np.ndarray,
             feature_names: list,
             class_names: list,
             n_background: int = 200,
             n_explain: int = 500,
             seed: int = 42,
             window: int = 10) -> dict:
    """
    Compute SHAP values and derived importance summaries for one model.

    Parameters
    ----------
    model_name   : one of 'lr', 'rf', 'cnn', 'lstm', 'aerf'
    model        : fitted model / estimator
    X_train      : unscaled or scaled training data (background reference)
    X_test       : samples to explain
    feature_names: list of feature name strings
    class_names  : list of class name strings
    n_background : size of background dataset for explainer
    n_explain    : number of test samples to explain
    window       : LSTM sequence window (used only when model_name='lstm')

    Returns
    -------
    dict with: shap_values, global_importance, per_class_importance
    """
    rng = np.random.default_rng(seed)

    # Sample background
    bg_idx = rng.choice(len(X_train), size=min(n_background, len(X_train)),
                        replace=False)
    X_bg   = X_train[bg_idx]

    # Sample explain set
    ex_idx = rng.choice(len(X_test), size=min(n_explain, len(X_test)),
                        replace=False)
    X_ex   = X_test[ex_idx]

    print(f"  Computing SHAP for {model_name} "
          f"(bg={len(X_bg)}, explain={len(X_ex)})…")

    name = model_name.lower()

    if name in ('lr', 'rf'):
        sv = _shap_values_sklearn(model, X_bg, X_ex)

    elif name == 'cnn':
        reshape = lambda x: x.reshape(-1, x.shape[1], 1)
        sv = _shap_values_keras(model, X_bg, X_ex, reshape_fn=reshape)

    elif name == 'lstm':
        import shap, tensorflow as tf
        from ..models.lstm import make_sequences
        # Build dummy y separately for bg and explain sets (lengths differ)
        X_bg_seq, _ = make_sequences(X_bg, np.zeros(len(X_bg)), window)
        X_ex_seq, _ = make_sequences(X_ex, np.zeros(len(X_ex)), window)
        # GradientExplainer works with LayerNormalization; DeepExplainer does not
        explainer = shap.GradientExplainer(model, X_bg_seq)
        sv = explainer.shap_values(X_ex_seq)
        if isinstance(sv, list):
            sv = np.stack(sv, axis=-1)
        # sv shape: (N, window, F) or (N, window, F, C) — mean over window dim
        if sv.ndim == 4:
            sv = sv.mean(axis=1)        # → (N, F, C)
        elif sv.ndim == 3:
            sv = sv.mean(axis=1)        # → (N, F)

    elif name == 'aerf':
        sv = _shap_values_aerf(model, X_bg, X_ex)

    else:
        raise ValueError(f"Unknown model_name '{model_name}'. "
                         "Expected one of: lr, rf, cnn, lstm, aerf.")

    return dict(
        shap_values         = sv,
        global_importance   = global_importance(sv, feature_names),
        per_class_importance= per_class_importance(sv, feature_names,
                                                   class_names),
    )


def run_shap_all(models: dict,
                 X_train: np.ndarray,
                 X_test: np.ndarray,
                 feature_names: list,
                 class_names: list,
                 **kwargs) -> dict:
    """
    Run SHAP for every model in `models` dict.

    Parameters
    ----------
    models : {model_name: fitted_model}

    Returns
    -------
    {model_name: shap_result_dict}
    """
    results = {}
    for name, model in models.items():
        try:
            results[name] = run_shap(
                name, model, X_train, X_test,
                feature_names, class_names, **kwargs)
        except Exception as e:
            print(f"  SHAP failed for {name}: {e}")
            results[name] = None
    return results
