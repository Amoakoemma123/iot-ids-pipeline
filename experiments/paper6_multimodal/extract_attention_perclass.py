"""
experiments/paper6_multimodal/extract_attention_perclass.py

Extract per-class attention weights (alpha_flow, alpha_beh) from the trained
attention_fusion model for Paper 6.

Retrains attention_fusion with seed=42 (matching the main paper results) then
intercepts the modality_attention Dense(2,softmax) layer on the test set to
compute per-class mean alpha values.

Outputs:
  results/{dataset}/attention_weights_perclass.json  — per-class alpha means/stds
  experiments/paper6_multimodal/figures/{dataset}/fig_attention_weights.png

Usage:
  python experiments/paper6_multimodal/extract_attention_perclass.py --dataset TON_IoT
  python experiments/paper6_multimodal/extract_attention_perclass.py --dataset UNSW_NB15
  python experiments/paper6_multimodal/extract_attention_perclass.py --dataset CICIDS2017
  python experiments/paper6_multimodal/extract_attention_perclass.py --all
"""

import argparse, json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import joblib
from tensorflow.keras.models import Model
from experiments.baseline.config import DATASETS, SEED
from src.data.loader import load_dataset
from experiments.paper6_multimodal.run_multimodal_experiments import (
    FEATURE_PARTITIONS, TRAIN_SUBSAMPLE,
    build_attention_fusion, compile_and_train, get_feature_indices,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')


# ── Data loader ───────────────────────────────────────────────────────────────

def load_data(dataset_name):
    csv_path, label_col = DATASETS[dataset_name]
    pipe_path = os.path.join(ROOT, 'experiments', 'baseline', 'results',
                             dataset_name, 'pipeline.pkl')
    print(f"  Loading pipeline: {pipe_path}")
    pipe = joblib.load(pipe_path)

    print(f"  Loading dataset: {csv_path}")
    raw = load_dataset(csv_path, label_col=label_col, seed=SEED)

    X_train, y_tr_smote = pipe.transform(raw['X_train'], apply_smote=True,  y=raw['y_train'])
    X_val,   _          = pipe.transform(raw['X_val'],   apply_smote=False)
    X_test,  _          = pipe.transform(raw['X_test'],  apply_smote=False)
    feature_names       = pipe.selected_features_

    y_tr = y_tr_smote.astype(int)
    y_v  = raw['y_val'].astype(int)
    y_te = raw['y_test'].astype(int)
    le   = raw['label_encoder']
    n_classes = len(le.classes_)

    subsample_cap = TRAIN_SUBSAMPLE.get(dataset_name)
    if subsample_cap and X_train.shape[0] > subsample_cap:
        rng = np.random.default_rng(SEED)
        keep = rng.choice(X_train.shape[0], subsample_cap, replace=False)
        keep.sort()
        X_train = X_train[keep]
        y_tr    = y_tr[keep]
        print(f"  Post-SMOTE subsample: {X_train.shape[0]:,} (cap={subsample_cap:,})")

    partition = FEATURE_PARTITIONS[dataset_name]
    flow_idx  = get_feature_indices(feature_names, partition['flow'])
    beh_idx   = get_feature_indices(feature_names, partition['behavioral'])
    missing   = set(range(X_train.shape[1])) - set(flow_idx + beh_idx)
    if missing:
        beh_idx = beh_idx + sorted(missing)

    n_flow = len(flow_idx)
    n_beh  = len(beh_idx)
    print(f"  {n_flow} flow + {n_beh} behavioral features | {n_classes} classes")
    print(f"  Train: {X_train.shape[0]:,}  Val: {X_val.shape[0]:,}  Test: {X_test.shape[0]:,}")

    def s(X, idx): return X[:, idx][..., np.newaxis]

    return {
        'X_tr_flow':  s(X_train, flow_idx),
        'X_val_flow': s(X_val,   flow_idx),
        'X_te_flow':  s(X_test,  flow_idx),
        'X_tr_beh':   s(X_train, beh_idx),
        'X_val_beh':  s(X_val,   beh_idx),
        'X_te_beh':   s(X_test,  beh_idx),
        'y_tr': y_tr, 'y_v': y_v, 'y_te': y_te,
        'le': le, 'n_classes': n_classes,
        'n_flow': n_flow, 'n_beh': n_beh,
    }


# ── Attention extraction ──────────────────────────────────────────────────────

def extract_perclass_alphas(model, X_te_flow, X_te_beh, y_te, le):
    """
    Run the test set through the modality_attention sub-model and compute
    per-class mean alpha_flow and alpha_beh.
    Returns dict keyed by class name.
    """
    attn_model = Model(
        inputs=model.inputs,
        outputs=model.get_layer('modality_attention').output,
    )
    alpha = attn_model.predict(
        [X_te_flow, X_te_beh], verbose=0, batch_size=1024
    )  # shape: (n_test, 2)

    alpha_flow = alpha[:, 0]
    alpha_beh  = alpha[:, 1]

    overall = {
        'alpha_flow_mean': round(float(alpha_flow.mean()), 4),
        'alpha_flow_std':  round(float(alpha_flow.std()),  4),
        'alpha_beh_mean':  round(float(alpha_beh.mean()),  4),
        'alpha_beh_std':   round(float(alpha_beh.std()),   4),
        'n_samples':       int(len(y_te)),
    }

    per_class = {}
    for cls_idx, cls_name in enumerate(le.classes_):
        mask = (y_te == cls_idx)
        n = int(mask.sum())
        if n == 0:
            continue
        af = alpha_flow[mask]
        ab = alpha_beh[mask]
        per_class[cls_name] = {
            'alpha_flow_mean': round(float(af.mean()), 4),
            'alpha_flow_std':  round(float(af.std()),  4),
            'alpha_beh_mean':  round(float(ab.mean()),  4),
            'alpha_beh_std':   round(float(ab.std()),   4),
            'n_samples': n,
        }

    return overall, per_class


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(dataset_name, overall, per_class, fig_dir):
    """Horizontal stacked bar chart: alpha_flow vs alpha_beh per class."""
    os.makedirs(fig_dir, exist_ok=True)

    classes = list(per_class.keys())
    af_means = [per_class[c]['alpha_flow_mean'] for c in classes]
    ab_means = [per_class[c]['alpha_beh_mean']  for c in classes]

    # Sort by alpha_flow descending (flow-dominant classes first)
    order = sorted(range(len(classes)), key=lambda i: af_means[i], reverse=True)
    classes  = [classes[i]  for i in order]
    af_means = [af_means[i] for i in order]
    ab_means = [ab_means[i] for i in order]

    # Pretty-print class names
    labels = [c.replace('web attack \ufffd', 'Web').replace('_', ' ').title()
              for c in classes]

    fig_h = max(4, len(classes) * 0.45)
    fig, ax = plt.subplots(figsize=(8, fig_h))

    y_pos = np.arange(len(classes))
    ax.barh(y_pos, af_means, color='#2196F3', label='α_flow (flow stream)')
    ax.barh(y_pos, ab_means, left=af_means, color='#FF5722', label='α_beh (behavioral stream)')

    # Overall mean line
    ax.axvline(overall['alpha_flow_mean'], color='#2196F3', linestyle='--',
               linewidth=1.2, alpha=0.7, label=f'Overall α_flow={overall["alpha_flow_mean"]:.3f}')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Attention weight (α_flow + α_beh = 1)')
    ax.set_xlim(0, 1)
    ax.set_title(f'{dataset_name}: Per-Class Modality Attention Weights (seed=42)', fontsize=10)
    ax.legend(loc='lower right', fontsize=8)
    ax.invert_yaxis()
    plt.tight_layout()

    out_path = os.path.join(fig_dir, 'fig_attention_weights.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved → {out_path}")


# ── Main runner ───────────────────────────────────────────────────────────────

def run_dataset(dataset_name, seed=None):
    if seed is None:
        seed = SEED
    print(f"\n{'='*60}")
    print(f"  Attention Weight Extraction — {dataset_name}  (seed={seed})")
    print(f"{'='*60}")

    out_dir = os.path.join(RESULTS_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    # seed=42 keeps the original filename for backward compatibility
    out_file = os.path.join(out_dir, 'attention_weights_perclass.json' if seed == 42
                            else f'attention_weights_perclass_seed{seed}.json')
    fig_dir  = os.path.join(FIGURES_DIR, dataset_name)

    # Load data
    data = load_data(dataset_name)

    # Set seed and train
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.backend.clear_session()

    print(f"\n  Training attention_fusion (seed={seed})...")
    model = build_attention_fusion(data['n_flow'], data['n_beh'], data['n_classes'])
    t0 = time.time()
    compile_and_train(
        model,
        [data['X_tr_flow'], data['X_tr_beh']], data['y_tr'],
        [data['X_val_flow'], data['X_val_beh']], data['y_v'],
        data['n_classes'], name='attention_fusion',
    )
    print(f"  Training done in {(time.time()-t0)/60:.1f} min")

    # Extract per-class attention weights
    print("  Extracting per-class attention weights...")
    overall, per_class = extract_perclass_alphas(
        model,
        data['X_te_flow'], data['X_te_beh'],
        data['y_te'], data['le'],
    )

    # Print summary
    print(f"\n  Overall: alpha_flow={overall['alpha_flow_mean']:.4f} ± {overall['alpha_flow_std']:.4f}  "
          f"alpha_beh={overall['alpha_beh_mean']:.4f} ± {overall['alpha_beh_std']:.4f}")
    print(f"  Per-class breakdown:")
    for cls, v in sorted(per_class.items(), key=lambda x: x[1]['alpha_flow_mean'], reverse=True):
        print(f"    {cls:30s}  α_flow={v['alpha_flow_mean']:.4f}  α_beh={v['alpha_beh_mean']:.4f}  n={v['n_samples']}")

    # Save JSON
    result = {
        'dataset': dataset_name,
        'seed': seed,
        'overall': overall,
        'per_class': per_class,
    }
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved → {out_file}")

    # Generate figure (only for seed=42 to avoid overwriting canonical figures)
    if seed == 42:
        make_figure(dataset_name, overall, per_class, fig_dir)

    print(f"  [Done — {dataset_name}]")
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['TON_IoT', 'UNSW_NB15', 'CICIDS2017'])
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.all:
        for ds in ['TON_IoT', 'UNSW_NB15', 'CICIDS2017']:
            run_dataset(ds, seed=args.seed)
    elif args.dataset:
        run_dataset(args.dataset, seed=args.seed)
    else:
        parser.print_help()
