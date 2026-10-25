"""
experiments/baseline/run_experiment.py
End-to-end baseline experiment: load → preprocess → train → evaluate → SHAP → figures → save.

Usage
-----
    cd IDS_IOT_Research
    python -m experiments.baseline.run_experiment --dataset TON_IoT
    python -m experiments.baseline.run_experiment --dataset all
"""
import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from pathlib import Path

# Allow TF GPU memory to grow incrementally (prevents OOM / segfault on 4 GB GTX 1650)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
try:
    import tensorflow as tf
    for _gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass

# ── Add repo root to path so `src` is importable ─────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data                import load_dataset, PreprocessingPipeline
from src.models              import build_lr, build_rf, build_cnn, build_lstm, build_autoencoder_rf
from src.training            import Trainer
from src.evaluation          import evaluate_model, mcnemar_test
from src.evaluation.metrics  import get_confusion_matrix
from src.evaluation.latency  import benchmark_all
from src.explainability      import run_shap_all
from src.models.lstm         import make_sequences

from experiments.baseline.config import (
    DATASETS, SEED, SPLIT, PREPROCESSING, PREPROCESSING_OVERRIDES,
    LR_PARAMS, RF_PARAMS,
    CNN_TRAIN, LSTM_TRAIN, AERF_TRAIN,
    LATENCY, SHAP,
    RESULTS_DIR, MODELS_DIR,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_json(obj: dict, path: Path):
    def _default(o):
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        raise TypeError(f"Not serialisable: {type(o)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=_default)
    print(f"  Saved → {path}")


def _save_figures(metrics_all, latency_results, mcnemar_results,
                  class_names, selected_features, preds, X_test, y_test,
                  y_test_lstm, window, dataset_name, out_dir):
    """Generate and save all figures for one dataset."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from sklearn.metrics import ConfusionMatrixDisplay

    figs_dir = out_dir / 'figures'
    figs_dir.mkdir(exist_ok=True)

    MODEL_LABELS = {'lr': 'LR', 'rf': 'RF', 'cnn': 'CNN',
                    'lstm': 'LSTM', 'aerf': 'AE-RF'}
    MODELS  = ['lr', 'rf', 'cnn', 'lstm', 'aerf']
    COLORS  = ['#1a73e8', '#2e7d32', '#c62828', '#f9a825', '#6a1b9a']
    METRICS = ['accuracy', 'precision', 'recall', 'macro_f1', 'roc_auc']
    MLABELS = ['Accuracy', 'Precision', 'Recall', 'Macro F1', 'ROC-AUC']

    # ── Fig 1: Grouped bar — all metrics ─────────────────────────────────────
    x     = np.arange(len(METRICS))
    n     = len(MODELS)
    width = 0.14
    offs  = np.linspace(-(n-1)/2*width, (n-1)/2*width, n)

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, m in enumerate(MODELS):
        vals = [metrics_all.get(m, {}).get(k, 0) for k in METRICS]
        bars = ax.bar(x + offs[i], vals, width, color=COLORS[i],
                      label=MODEL_LABELS[m], zorder=3, edgecolor='white', lw=0.5)
        for bar, v in zip(bars, vals):
            if v > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=5.5, rotation=90)

    ax.set_xticks(x); ax.set_xticklabels(MLABELS, fontsize=10)
    ax.set_ylim(0, 1.18); ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'{dataset_name} — Model Performance Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=5, framealpha=0.9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(figs_dir / 'fig1_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Fig 2: F1 and AUC side-by-side bars ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, title in zip(axes, ['macro_f1', 'roc_auc'], ['Macro F1', 'ROC-AUC']):
        vals  = [metrics_all.get(m, {}).get(key, 0) for m in MODELS]
        mlabs = [MODEL_LABELS[m] for m in MODELS]
        bars  = ax.bar(mlabs, vals, color=COLORS, edgecolor='white', lw=0.5, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylim(0, 1.12); ax.set_title(f'{title}', fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=10)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.suptitle(f'{dataset_name} — F1 and ROC-AUC', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(figs_dir / 'fig2_f1_auc.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Fig 3: Latency bar chart ──────────────────────────────────────────────
    lat_models = ['lr', 'rf', 'cnn', 'aerf']
    lat_labels = ['LR', 'RF', 'CNN', 'AE-RF']
    lat_colors = ['#1a73e8', '#2e7d32', '#c62828', '#6a1b9a']
    lat_vals   = [latency_results.get(m, {}).get('mean_ms', 0) for m in lat_models]
    lat_errs   = [latency_results.get(m, {}).get('std_ms',  0) for m in lat_models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(lat_labels, lat_vals, color=lat_colors, edgecolor='white',
                  lw=0.5, zorder=3, yerr=lat_errs, capsize=5, error_kw={'lw':1.5})
    for bar, v in zip(bars, lat_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(lat_vals)*0.02,
                f'{v:.4f} ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel('Mean latency (ms / sample)', fontsize=11)
    ax.set_title(f'{dataset_name} — Inference Latency (mean ± std)', fontsize=12, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(figs_dir / 'fig3_latency.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Fig 4: Confusion matrices (RF + CNN side by side) ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(class_names)*2.2), 6))
    short_names = [c[:12] for c in class_names]
    for ax, name in zip(axes, ['rf', 'cnn']):
        yt = y_test_lstm if name == 'lstm' else y_test
        yp = preds[name][0][:len(yt)]
        cm = get_confusion_matrix(yt, yp, normalize='true')
        disp = ConfusionMatrixDisplay(cm, display_labels=short_names)
        disp.plot(ax=ax, colorbar=True, cmap='Blues', xticks_rotation=45)
        ax.set_title(f'{MODEL_LABELS[name]} Confusion Matrix', fontsize=11, fontweight='bold')
    fig.suptitle(f'{dataset_name} — Normalised Confusion Matrices (RF & CNN)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(figs_dir / 'fig4_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Fig 5: Per-class F1 for RF (from confusion matrix diagonal) ───────────
    try:
        yp_rf = preds['rf'][0]
        cm_rf = get_confusion_matrix(y_test, yp_rf, normalize='true')
        per_class_f1 = np.diag(cm_rf)   # recall per class (approx F1 from CM diagonal)
        fig, ax = plt.subplots(figsize=(max(10, len(class_names)*0.9), 5))
        bars = ax.bar(range(len(class_names)), per_class_f1,
                      color=['#2e7d32' if v >= 0.7 else '#f9a825' if v >= 0.4 else '#c62828'
                             for v in per_class_f1],
                      edgecolor='white', lw=0.5, zorder=3)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Recall (true-positive rate)', fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.set_title(f'{dataset_name} RF — Per-Class Detection Rate', fontsize=12, fontweight='bold')
        ax.axhline(0.7, color='green', linestyle='--', lw=1, alpha=0.7, label='0.70 threshold')
        ax.axhline(0.4, color='orange', linestyle='--', lw=1, alpha=0.7, label='0.40 threshold')
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.tight_layout()
        fig.savefig(figs_dir / 'fig5_rf_per_class.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception:
        pass

    # ── Fig 6: McNemar n01 vs n10 (RF superiority chart) ─────────────────────
    try:
        comps   = ['rf_vs_lr', 'rf_vs_cnn', 'rf_vs_aerf']
        comp_lb = ['RF vs LR', 'RF vs CNN', 'RF vs AE-RF']
        n01s    = [mcnemar_results.get(k, {}).get('n01', 0) for k in comps]
        n10s    = [mcnemar_results.get(k, {}).get('n10', 0) for k in comps]
        x_pos   = np.arange(len(comp_lb))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x_pos - 0.2, n01s, 0.35, label='n01: RF correct, other wrong',
               color='#2e7d32', zorder=3, edgecolor='white')
        ax.bar(x_pos + 0.2, n10s, 0.35, label='n10: other correct, RF wrong',
               color='#c62828', zorder=3, edgecolor='white')
        ax.set_xticks(x_pos); ax.set_xticklabels(comp_lb, fontsize=11)
        ax.set_ylabel('Sample count', fontsize=10)
        ax.set_title(f"{dataset_name} — McNemar's Test: RF vs Alternatives", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.tight_layout()
        fig.savefig(figs_dir / 'fig6_mcnemar.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception:
        pass

    print(f"  Figures saved → {figs_dir}")


# ── Main experiment ───────────────────────────────────────────────────────────

def run_one(dataset_name: str, csv_path: Path, label_col: str = 'type'):
    print(f"\n{'='*60}")
    print(f" Dataset: {dataset_name}")
    print(f"{'='*60}")

    out_dir    = RESULTS_DIR / dataset_name
    models_dir = MODELS_DIR  / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("\n[1/6] Loading data…")
    data = load_dataset(csv_path, label_col=label_col, **SPLIT)
    X_train_raw, y_train = data['X_train'], data['y_train']
    X_val_raw,   y_val   = data['X_val'],   data['y_val']
    X_test_raw,  y_test  = data['X_test'],  data['y_test']
    feature_names        = data['feature_names']
    class_names          = data['class_names']
    label_encoder        = data['label_encoder']
    n_classes            = len(class_names)

    print(f"  Train={len(X_train_raw):,}  Val={len(X_val_raw):,}  "
          f"Test={len(X_test_raw):,}  Features={len(feature_names)}  "
          f"Classes={n_classes}")

    normal_label = 'normal' if 'normal' in class_names else class_names[0]
    normal_class = int(np.where(label_encoder.classes_ == normal_label)[0][0])
    print(f"  Normal class: '{normal_label}' → label {normal_class}")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    print("\n[2/6] Preprocessing…")
    pp_cfg = {**PREPROCESSING, **PREPROCESSING_OVERRIDES.get(dataset_name, {})}
    pipeline = PreprocessingPipeline(
        mi_threshold        = pp_cfg['mi_threshold'],
        rfecv_subsample     = pp_cfg['rfecv_subsample'],
        skip_rfecv          = pp_cfg.get('skip_rfecv', False),
        smote_k             = pp_cfg['smote_k'],
        smote_max_per_class = pp_cfg.get('smote_max_per_class'),
        mi_subsample        = pp_cfg.get('mi_subsample'),
        seed                = pp_cfg['seed'],
    )
    X_train, y_train_sm = pipeline.fit_transform(
        X_train_raw, y_train,
        feature_names=feature_names,
        apply_smote=pp_cfg['apply_smote'],
    )
    X_train_sc, _ = pipeline.transform(X_train_raw, y=y_train)
    X_val,      _ = pipeline.transform(X_val_raw,   y=y_val)
    X_test,     _ = pipeline.transform(X_test_raw,  y=y_test)

    selected_features = pipeline.selected_features_
    n_features        = X_train.shape[1]
    print(f"  Features after selection: {n_features} (was {len(feature_names)})")
    pipeline.save(out_dir / 'pipeline.pkl')

    # ── 3. Train ──────────────────────────────────────────────────────────────
    print("\n[3/6] Training models…")
    trainer = Trainer(output_dir=models_dir)

    # class_weight only needed when SMOTE is fully disabled (no oversampling at all)
    cw_param = None  # SMOTE (full or capped) handles imbalance

    lr_model = build_lr(**LR_PARAMS, class_weight=cw_param)
    trainer.fit_sklearn(lr_model, X_train, y_train_sm, name='lr')

    rf_model = build_rf(**RF_PARAMS, class_weight=cw_param)
    trainer.fit_sklearn(rf_model, X_train, y_train_sm, name='rf')

    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight('balanced', classes=np.unique(y_train_sm), y=y_train_sm)
    cnn_class_weights = dict(enumerate(cw))

    cnn_model = build_cnn(n_features, n_classes)
    trainer.fit_cnn(cnn_model, X_train, y_train_sm,
                    X_val, y_val, name='cnn',
                    class_weights=cnn_class_weights, **CNN_TRAIN)

    cw_lstm = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    lstm_class_weights = dict(enumerate(cw_lstm))

    lstm_model = build_lstm(n_features, n_classes, seq_len=LSTM_TRAIN['window'])
    trainer.fit_lstm(lstm_model, X_train_sc, y_train,
                     X_val, y_val, name='lstm',
                     class_weights=lstm_class_weights, **LSTM_TRAIN)

    aerf_model = build_autoencoder_rf(n_features, n_classes, class_weight=cw_param)
    trainer.fit_aerf(aerf_model, X_train, y_train_sm,
                     X_val=X_val, name='aerf',
                     normal_class=normal_class, **AERF_TRAIN)

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    print("\n[4/6] Evaluating…")
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
        mcnemar_results[f'rf_vs_{name}'] = mcnemar_test(
            y_test, y_pred_rf, preds[name][0])
    _save_json(mcnemar_results, out_dir / 'mcnemar.json')

    cm_dict = {}
    for name, (y_pred, _) in preds.items():
        yt = y_test_lstm if name == 'lstm' else y_test
        cm_dict[name] = get_confusion_matrix(yt, y_pred, normalize='true').tolist()
    _save_json(cm_dict, out_dir / 'confusion_matrices.json')

    print("  Benchmarking latency…")
    latency_results = benchmark_all(
        {'lr': lr_model.predict,
         'rf': rf_model.predict,
         'cnn': _cnn_predict,
         'aerf': aerf_model.predict},
        X_test, **LATENCY)
    _save_json(latency_results, out_dir / 'latency.json')

    # ── 5. SHAP ───────────────────────────────────────────────────────────────
    print("\n[5/6] SHAP explainability…")
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
    print("\n[6/6] Generating figures…")
    _save_figures(
        metrics_all, latency_results, mcnemar_results,
        class_names, selected_features, preds, X_test, y_test,
        y_test_lstm, window, dataset_name, out_dir,
    )

    print(f"\nDone: {dataset_name}. Results → {out_dir}")
    return metrics_all


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run baseline IDS experiment on one or all datasets.')
    parser.add_argument('--dataset', default='TON_IoT',
                        choices=list(DATASETS.keys()) + ['all'],
                        help='Dataset to run (default: TON_IoT)')
    args = parser.parse_args()

    targets = list(DATASETS.items()) if args.dataset == 'all' \
              else [(args.dataset, DATASETS[args.dataset])]

    all_results = {}
    for name, (path, label_col) in targets:
        if not Path(path).exists():
            print(f"[SKIP] {name}: file not found at {path}")
            continue
        all_results[name] = run_one(name, path, label_col)

    if len(all_results) > 1:
        _save_json(all_results, RESULTS_DIR / 'all_metrics.json')

    print("\nExperiment complete.")


if __name__ == '__main__':
    main()
