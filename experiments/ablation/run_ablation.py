"""
experiments/ablation/run_ablation.py
2x2 factorial ablation study on TON_IoT.

Conditions:
  A: MI + RFECV + SMOTE   (full pipeline)
  B: MI + RFECV, no SMOTE
  C: MI + SMOTE, no RFECV
  D: All features, no SMOTE

Usage
-----
    cd IDS_IOT_Research
    python -m experiments.ablation.run_ablation
    python -m experiments.ablation.run_ablation --conditions A B C D
"""
import argparse
import csv
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

from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

from src.data           import load_dataset
from src.models         import build_rf, build_cnn
from src.training       import Trainer
from src.evaluation     import evaluate_model

from experiments.ablation.config import (
    DATASET, SEED, SPLIT, CONDITIONS,
    RF_PARAMS, CNN_TRAIN, LR_PARAMS,
    RESULTS_DIR, MODELS_DIR,
)


# ── Custom preprocessing for ablation ─────────────────────────────────────────

def ablation_preprocess(X_train, y_train, X_val, X_test,
                        feature_names, condition: dict):
    """
    Apply MI filter → optional RFECV → MinMax scale → optional SMOTE.
    Returns (X_train_out, y_train_out, X_val_out, X_test_out, selected_names).
    """
    cfg = condition
    names = list(feature_names)

    # ── Stage 1: Mutual Information filter ──────────────────────────────────
    if cfg['mi_threshold'] > 0.0:
        print(f"    MI filter (threshold={cfg['mi_threshold']})…")
        mi_scores = mutual_info_classif(X_train, y_train, random_state=cfg['seed'])
        mi_mask   = mi_scores > cfg['mi_threshold']
        X_train   = X_train[:, mi_mask]
        X_val     = X_val[:, mi_mask]
        X_test    = X_test[:, mi_mask]
        names     = [n for n, k in zip(names, mi_mask) if k]
        print(f"    MI: kept {X_train.shape[1]} features")
    else:
        print(f"    MI filter skipped (threshold=0.0 → keep all {X_train.shape[1]} features)")

    # ── Stage 2: Optional RFECV ──────────────────────────────────────────────
    if not cfg['skip_rfecv']:
        n_sub = min(cfg['rfecv_subsample'], len(X_train))
        print(f"    RFECV (subsample={n_sub:,})…")
        rng = np.random.default_rng(cfg['seed'])
        idx = rng.choice(len(X_train), n_sub, replace=False)
        est = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=cfg['seed'])
        rfecv = RFECV(
            estimator=est,
            cv=StratifiedKFold(3),
            step=1, n_jobs=-1,
            scoring='f1_macro',
            min_features_to_select=10,
        )
        rfecv.fit(X_train[idx], y_train[idx])
        X_train = rfecv.transform(X_train)
        X_val   = rfecv.transform(X_val)
        X_test  = rfecv.transform(X_test)
        names   = [n for n, s in zip(names, rfecv.support_) if s]
        print(f"    RFECV: kept {X_train.shape[1]} features")
    else:
        print(f"    RFECV skipped → {X_train.shape[1]} features retained")

    # ── Stage 3: MinMax scaling ──────────────────────────────────────────────
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)
    X_train_unsmoted = X_train.copy()

    # ── Stage 4: Optional SMOTE ──────────────────────────────────────────────
    if cfg['apply_smote']:
        print(f"    SMOTE oversampling…")
        smote = SMOTE(k_neighbors=cfg['smote_k'], random_state=cfg['seed'])
        X_train, y_train = smote.fit_resample(X_train, y_train)
        X_train = X_train.astype(np.float32)
        print(f"    SMOTE: {len(X_train_unsmoted):,} → {len(X_train):,} samples")
    else:
        print(f"    SMOTE skipped → {len(X_train):,} training samples")

    return X_train, y_train, X_train_unsmoted, X_val, X_test, names


# ── JSON helper ───────────────────────────────────────────────────────────────

def _save_json(obj, path):
    def _default(o):
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        raise TypeError(type(o))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=_default)


# ── Run one condition ─────────────────────────────────────────────────────────

def run_condition(cond_id: str, data: dict):
    condition = CONDITIONS[cond_id]
    print(f"\n{'─'*55}")
    print(f"  Condition {cond_id}: {condition['description']}")
    print(f"{'─'*55}")

    out_dir    = RESULTS_DIR / f'condition_{cond_id}'
    models_dir = MODELS_DIR  / f'condition_{cond_id}'
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    X_train_raw = data['X_train']
    y_train_raw = data['y_train']
    X_val_raw   = data['X_val']
    y_val       = data['y_val']
    X_test_raw  = data['X_test']
    y_test      = data['y_test']
    feature_names = data['feature_names']
    class_names   = data['class_names']
    n_classes     = len(class_names)

    # Preprocess
    (X_train, y_train_sm,
     X_train_unsmoted,
     X_val, X_test,
     sel_names) = ablation_preprocess(
        X_train_raw.copy(), y_train_raw.copy(),
        X_val_raw.copy(), X_test_raw.copy(),
        feature_names, condition,
    )

    n_features = X_train.shape[1]
    print(f"  Final: {n_features} features, {len(X_train):,} train samples")

    trainer = Trainer(output_dir=models_dir)

    # RF
    from sklearn.utils.class_weight import compute_class_weight
    cw_param = None if condition['apply_smote'] else 'balanced'

    rf_model = build_rf(**RF_PARAMS, class_weight=cw_param)
    trainer.fit_sklearn(rf_model, X_train, y_train_sm, name='rf')
    rf_pred  = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)
    rf_metrics = evaluate_model(y_test, rf_pred, rf_proba, class_names)

    # CNN
    cw_arr = compute_class_weight('balanced',
                                  classes=np.unique(y_train_sm), y=y_train_sm)
    cnn_cw = dict(enumerate(cw_arr))
    cnn_model = build_cnn(n_features, n_classes)
    trainer.fit_cnn(cnn_model, X_train, y_train_sm, X_val, y_val,
                    name='cnn', class_weights=cnn_cw, **CNN_TRAIN)

    def _cnn_predict(X):
        return np.argmax(cnn_model.predict(X.reshape(-1, n_features, 1), verbose=0), axis=1)
    def _cnn_proba(X):
        return cnn_model.predict(X.reshape(-1, n_features, 1), verbose=0)

    cnn_pred   = _cnn_predict(X_test)
    cnn_proba  = _cnn_proba(X_test)
    cnn_metrics = evaluate_model(y_test, cnn_pred, cnn_proba, class_names)

    metrics = {'rf': rf_metrics, 'cnn': cnn_metrics}
    _save_json(metrics, out_dir / 'metrics.json')

    summary = {
        'condition':   cond_id,
        'description': condition['description'],
        'n_features':  n_features,
        'n_train':     len(X_train),
        'apply_smote': condition['apply_smote'],
        'skip_rfecv':  condition['skip_rfecv'],
        'rf_accuracy': rf_metrics['accuracy'],
        'rf_f1':       rf_metrics['macro_f1'],
        'cnn_accuracy':cnn_metrics['accuracy'],
        'cnn_f1':      cnn_metrics['macro_f1'],
    }

    print(f"\n  Results:")
    print(f"    RF  — acc={rf_metrics['accuracy']:.4f}  f1={rf_metrics['macro_f1']:.4f}")
    print(f"    CNN — acc={cnn_metrics['accuracy']:.4f}  f1={cnn_metrics['macro_f1']:.4f}")

    return summary


# ── Figures ───────────────────────────────────────────────────────────────────

def _save_ablation_figures(summaries: list):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figs_dir = RESULTS_DIR / 'figures'
    figs_dir.mkdir(exist_ok=True)

    cond_ids  = [s['condition']   for s in summaries]
    descs     = [f"Cond {s['condition']}\n{s['description'].split(',')[0]}"
                 for s in summaries]
    rf_f1s    = [s['rf_f1']       for s in summaries]
    cnn_f1s   = [s['cnn_f1']      for s in summaries]
    n_feats   = [s['n_features']  for s in summaries]
    n_trains  = [s['n_train']     for s in summaries]

    x = np.arange(len(summaries))
    w = 0.35

    # Paper targets (from results_ablation.csv)
    paper_rf  = {'A': 0.9652, 'B': 0.9691, 'C': 0.9653, 'D': 0.9666}
    paper_cnn = {'A': 0.8672, 'B': 0.8168, 'C': 0.9051, 'D': 0.6047}

    # Fig 1: RF and CNN F1 by condition
    fig, ax = plt.subplots(figsize=(11, 6))
    bars_rf  = ax.bar(x - w/2, rf_f1s,  w, label='RF (ours)',  color='#2e7d32', zorder=3)
    bars_cnn = ax.bar(x + w/2, cnn_f1s, w, label='CNN (ours)', color='#c62828', zorder=3)

    # Paper target markers
    for i, cid in enumerate(cond_ids):
        ax.scatter(i - w/2, paper_rf.get(cid, 0),  marker='*', s=120, color='lime',
                   zorder=5, label='RF (paper)' if i == 0 else '')
        ax.scatter(i + w/2, paper_cnn.get(cid, 0), marker='*', s=120, color='orange',
                   zorder=5, label='CNN (paper)' if i == 0 else '')

    for bar, v in zip(bars_rf, rf_f1s):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar, v in zip(bars_cnn, cnn_f1s):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([s['description'] for s in summaries], fontsize=9)
    ax.set_ylabel('Macro F1', fontsize=11)
    ax.set_ylim(0.5, 1.08)
    ax.set_title('TON_IoT Ablation — RF & CNN Macro F1 (★ = paper target)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, ncol=4)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(figs_dir / 'ablation_f1.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Fig 2: Feature count and training-set size by condition
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes
    ax1.bar(cond_ids, n_feats,  color='#1a73e8', zorder=3, edgecolor='white')
    for i, (c, v) in enumerate(zip(cond_ids, n_feats)):
        ax1.text(i, v+0.3, str(v), ha='center', fontsize=10, fontweight='bold')
    ax1.set_ylabel('# features selected', fontsize=11)
    ax1.set_title('Features per Condition', fontsize=12, fontweight='bold')
    ax1.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

    ax2.bar(cond_ids, [n/1000 for n in n_trains],
            color='#f9a825', zorder=3, edgecolor='white')
    for i, (c, v) in enumerate(zip(cond_ids, n_trains)):
        ax2.text(i, v/1000+1, f'{v:,}', ha='center', fontsize=9, fontweight='bold')
    ax2.set_ylabel('Training samples (K)', fontsize=11)
    ax2.set_title('Training Set Size per Condition', fontsize=12, fontweight='bold')
    ax2.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    fig.suptitle('TON_IoT Ablation Study', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(figs_dir / 'ablation_data_stats.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Fig 3: Gap to paper target (delta F1)
    rf_gaps  = [v - paper_rf.get(cid, 0)  for v, cid in zip(rf_f1s,  cond_ids)]
    cnn_gaps = [v - paper_cnn.get(cid, 0) for v, cid in zip(cnn_f1s, cond_ids)]
    colors_rf  = ['#2e7d32' if g >= 0 else '#c62828' for g in rf_gaps]
    colors_cnn = ['#2e7d32' if g >= 0 else '#c62828' for g in cnn_gaps]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, gaps, colors, title, model in zip(
            axes, [rf_gaps, cnn_gaps], [colors_rf, colors_cnn],
            ['RF Macro F1 Gap vs Paper', 'CNN Macro F1 Gap vs Paper'],
            ['RF', 'CNN']):
        ax.bar(cond_ids, gaps, color=colors, zorder=3, edgecolor='white')
        ax.axhline(0, color='black', lw=1)
        for i, g in enumerate(gaps):
            ax.text(i, g + (0.002 if g >= 0 else -0.005),
                    f'{g:+.4f}', ha='center', va='bottom' if g >= 0 else 'top',
                    fontsize=9, fontweight='bold')
        ax.set_ylabel('Delta F1 (ours − paper)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    fig.suptitle('TON_IoT Ablation — Gap to Paper Target (green=above, red=below)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(figs_dir / 'ablation_gap_to_paper.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Ablation figures saved → {figs_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conditions', nargs='+', default=['A', 'B', 'C', 'D'],
                        choices=['A', 'B', 'C', 'D'])
    args = parser.parse_args()

    csv_path, label_col = DATASET
    print(f"\n{'='*55}")
    print(f"  TON_IoT Ablation Study")
    print(f"  Conditions: {args.conditions}")
    print(f"{'='*55}")

    print("\n[1] Loading TON_IoT data…")
    data = load_dataset(csv_path, label_col=label_col, **SPLIT)
    print(f"  Train={len(data['X_train']):,}  Val={len(data['X_val']):,}  "
          f"Test={len(data['X_test']):,}  Features={len(data['feature_names'])}")

    summaries = []
    for cond_id in args.conditions:
        summary = run_condition(cond_id, data)
        summaries.append(summary)

    # Write CSV summary
    csv_path_out = RESULTS_DIR / 'ablation_results.csv'
    csv_path_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['condition', 'description', 'n_features', 'n_train',
                  'apply_smote', 'skip_rfecv',
                  'rf_accuracy', 'rf_f1', 'cnn_accuracy', 'cnn_f1']
    with open(csv_path_out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\nAblation CSV → {csv_path_out}")

    # Write JSON summary
    _save_json(summaries, RESULTS_DIR / 'ablation_results.json')

    # Figures
    print("\nGenerating ablation figures…")
    _save_ablation_figures(summaries)

    # Print comparison table
    print(f"\n{'='*75}")
    print(f"  ABLATION SUMMARY vs PAPER (Baseline_LWIDS/results_ablation.csv)")
    print(f"{'='*75}")
    paper_rf  = {'A': 0.9652, 'B': 0.9691, 'C': 0.9653, 'D': 0.9666}
    paper_cnn = {'A': 0.8672, 'B': 0.8168, 'C': 0.9051, 'D': 0.6047}
    print(f"  {'Cond':<6} {'Description':<30} {'RF F1':>8} {'ΔRF':>8} {'CNN F1':>8} {'ΔCNN':>8}")
    print(f"  {'-'*70}")
    for s in summaries:
        c = s['condition']
        rf_gap  = s['rf_f1']  - paper_rf.get(c, 0)
        cnn_gap = s['cnn_f1'] - paper_cnn.get(c, 0)
        print(f"  {c:<6} {s['description'][:30]:<30} "
              f"{s['rf_f1']:>8.4f} {rf_gap:>+8.4f} "
              f"{s['cnn_f1']:>8.4f} {cnn_gap:>+8.4f}")
    print(f"{'='*75}")


if __name__ == '__main__':
    main()
