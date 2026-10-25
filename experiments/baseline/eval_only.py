"""
experiments/baseline/eval_only.py
Load saved models + pipeline for one dataset and run evaluation, SHAP, and figures.
Use this to resume after a crash during step 4-6 without retraining.

Usage
-----
    cd IDS_IOT_Research
    python -m experiments.baseline.eval_only --dataset CICIDS2017
"""
import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
try:
    import tensorflow as tf
    for _gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data                import load_dataset, PreprocessingPipeline
from src.evaluation          import evaluate_model, mcnemar_test
from src.evaluation.metrics  import get_confusion_matrix
from src.evaluation.latency  import benchmark_all
from src.explainability      import run_shap_all
from src.models.lstm         import make_sequences

from experiments.baseline.config import (
    DATASETS, SEED, SPLIT, PREPROCESSING, PREPROCESSING_OVERRIDES,
    CNN_TRAIN, LSTM_TRAIN,
    LATENCY, SHAP,
    RESULTS_DIR, MODELS_DIR,
)
from experiments.baseline.run_experiment import _save_json, _save_figures


def eval_one(dataset_name: str, csv_path: Path, label_col: str):
    print(f"\n{'='*60}")
    print(f" Eval-only: {dataset_name}")
    print(f"{'='*60}")

    out_dir    = RESULTS_DIR / dataset_name
    models_dir = MODELS_DIR  / dataset_name

    # ── 1. Load data (need X_test, y_test, class info) ───────────────────────
    print("\n[1] Loading data…")
    data = load_dataset(csv_path, label_col=label_col, **SPLIT)
    X_train_raw = data['X_train']
    X_test_raw  = data['X_test']
    y_test      = data['y_test']
    feature_names = data['feature_names']
    class_names   = data['class_names']
    label_encoder = data['label_encoder']
    n_classes     = len(class_names)

    normal_label = 'normal' if 'normal' in class_names else class_names[0]
    normal_class = int(np.where(label_encoder.classes_ == normal_label)[0][0])
    print(f"  Test={len(X_test_raw):,}  Classes={n_classes}  Normal='{normal_label}'")

    # ── 2. Load pipeline and transform test set ───────────────────────────────
    print("\n[2] Loading pipeline…")
    pipeline = PreprocessingPipeline.load(out_dir / 'pipeline.pkl')
    X_test, _ = pipeline.transform(X_test_raw, y=y_test)
    X_train, _ = pipeline.transform(X_train_raw, y=data['y_train'])
    selected_features = pipeline.selected_features_
    n_features = X_test.shape[1]
    print(f"  Features: {n_features}")

    # ── 3. Load models ────────────────────────────────────────────────────────
    print("\n[3] Loading models…")
    import joblib
    from tensorflow import keras

    lr_model   = joblib.load(models_dir / 'model_lr.pkl')
    rf_model   = joblib.load(models_dir / 'model_rf.pkl')
    cnn_model  = keras.models.load_model(models_dir / 'model_cnn.h5')
    lstm_model = keras.models.load_model(models_dir / 'model_lstm.h5')

    ae_model   = keras.models.load_model(models_dir / 'model_ae.h5')
    aerf_clf   = joblib.load(models_dir / 'model_aerf.pkl')

    # Reconstruct encoder from saved autoencoder (trainer saves autoencoder_, not encoder_)
    # The encoder outputs the 'latent' layer (16-dim bottleneck)
    encoder_model = keras.Model(
        inputs=ae_model.input,
        outputs=ae_model.get_layer('latent').output,
        name='Encoder',
    )

    class AERFWrapper:
        def __init__(self, encoder, rf):
            self.encoder = encoder
            self.rf = rf
        def predict(self, X):
            return self.rf.predict(self.encoder.predict(X, verbose=0))
        def predict_proba(self, X):
            return self.rf.predict_proba(self.encoder.predict(X, verbose=0))

    aerf_model = AERFWrapper(encoder_model, aerf_clf)
    print("  All models loaded.")

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    print("\n[4] Evaluating…")
    window = LSTM_TRAIN['window']

    def _cnn_predict(X):
        return np.argmax(cnn_model.predict(X.reshape(-1, n_features, 1), verbose=0), axis=1)

    def _cnn_proba(X):
        return cnn_model.predict(X.reshape(-1, n_features, 1), verbose=0)

    def _lstm_predict(X):
        X_seq, _ = make_sequences(X, np.zeros(len(X)), window)
        with tf.device('/CPU:0'):
            return np.argmax(lstm_model.predict(X_seq, verbose=0), axis=1)

    def _lstm_proba(X):
        X_seq, _ = make_sequences(X, np.zeros(len(X)), window)
        with tf.device('/CPU:0'):
            return lstm_model.predict(X_seq, verbose=0)

    preds = {
        'lr':   (lr_model.predict(X_test),   lr_model.predict_proba(X_test)),
        'rf':   (rf_model.predict(X_test),   rf_model.predict_proba(X_test)),
        'cnn':  (_cnn_predict(X_test),        _cnn_proba(X_test)),
        'lstm': (_lstm_predict(X_test),       _lstm_proba(X_test)),
        'aerf': (aerf_model.predict(X_test), aerf_model.predict_proba(X_test)),
    }

    y_test_lstm = y_test[window - 1:]

    metrics_all = {}
    for name, (y_pred, y_proba) in preds.items():
        yt = y_test_lstm if name == 'lstm' else y_test
        metrics_all[name] = evaluate_model(yt, y_pred, y_proba, class_names)
        m   = metrics_all[name]
        auc = f"{m.get('roc_auc', float('nan')):.4f}"
        print(f"  {name:6s}  acc={m['accuracy']:.4f}  f1={m['macro_f1']:.4f}  auc={auc}")

    _save_json(metrics_all, out_dir / 'metrics.json')

    mcnemar_results = {}
    y_pred_rf = preds['rf'][0]
    for name in ('lr', 'cnn', 'aerf'):
        mcnemar_results[f'rf_vs_{name}'] = mcnemar_test(y_test, y_pred_rf, preds[name][0])
    _save_json(mcnemar_results, out_dir / 'mcnemar.json')

    cm_dict = {}
    for name, (y_pred, _) in preds.items():
        yt = y_test_lstm if name == 'lstm' else y_test
        cm_dict[name] = get_confusion_matrix(yt, y_pred, normalize='true').tolist()
    _save_json(cm_dict, out_dir / 'confusion_matrices.json')

    print("  Benchmarking latency…")
    latency_results = benchmark_all(
        {'lr': lr_model.predict, 'rf': rf_model.predict,
         'cnn': _cnn_predict, 'aerf': aerf_model.predict},
        X_test, **LATENCY)
    _save_json(latency_results, out_dir / 'latency.json')

    # ── 5. SHAP ───────────────────────────────────────────────────────────────
    print("\n[5] SHAP explainability…")
    shap_results = run_shap_all(
        {'lr': lr_model, 'rf': rf_model,
         'cnn': cnn_model, 'lstm': lstm_model, 'aerf': aerf_model},
        X_train, X_test,
        feature_names=selected_features,
        class_names=class_names,
        window=window,
        **SHAP,
    )
    shap_summary = {
        name: {
            'global_importance':    res['global_importance'],
            'per_class_importance': res['per_class_importance'],
        }
        for name, res in shap_results.items()
        if res is not None
    }
    _save_json(shap_summary, out_dir / 'shap_importance.json')

    # ── 6. Figures ────────────────────────────────────────────────────────────
    print("\n[6] Generating figures…")
    _save_figures(
        metrics_all, latency_results, mcnemar_results,
        class_names, selected_features, preds, X_test, y_test,
        y_test_lstm, window, dataset_name, out_dir,
    )

    print(f"\nDone: {dataset_name}. Results → {out_dir}")
    return metrics_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CICIDS2017',
                        choices=list(DATASETS.keys()),
                        help='Dataset to evaluate (default: CICIDS2017)')
    args = parser.parse_args()
    name, (path, label_col) = args.dataset, DATASETS[args.dataset]
    if not Path(path).exists():
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    eval_one(name, path, label_col)


if __name__ == '__main__':
    main()
