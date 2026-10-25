"""
experiments/paper6_multimodal/run_multiseed.py

Multi-seed confidence interval runner for Paper 6 multi-modal IDS.

Runs seeds [42, 43, 44] for all 5 fusion architectures on each dataset.
- Data split is fixed (loaded from saved pipeline.pkl with SEED=42).
- Only NN initialisation and training stochasticity vary across seeds.
- Extracts actual attention gate weights (alpha_flow, alpha_beh) from the
  modality_attention Dense layer on the test set.

Outputs:
  results/{dataset}/seed_{s}/multimodal_results.json  — per-seed raw metrics
  results/{dataset}/multiseed_results.json             — mean ± std aggregated
  results/{dataset}/attention_weights.json             — actual alpha values

Usage:
  python experiments/paper6_multimodal/run_multiseed.py --dataset TON_IoT
  python experiments/paper6_multimodal/run_multiseed.py --dataset UNSW_NB15
  python experiments/paper6_multimodal/run_multiseed.py --dataset CICIDS2017
  python experiments/paper6_multimodal/run_multiseed.py --all
"""

import argparse, json, os, sys, time
import numpy as np
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
from experiments.baseline.config import DATASETS, SEED
from src.data.loader import load_dataset

# Import model builders and feature partitions from main experiment script
from experiments.paper6_multimodal.run_multimodal_experiments import (
    FEATURE_PARTITIONS, TRAIN_SUBSAMPLE,
    build_cnn_single, build_early_fusion, build_late_fusion, build_attention_fusion,
    compile_and_train, evaluate_model, count_params, get_feature_indices,
)

from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

SEEDS = [42, 43, 44]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


# ── Attention weight extractor ────────────────────────────────────────────────

def extract_attention_weights(attention_model, X_te_flow_3d, X_te_beh_3d):
    """
    Extract actual alpha_flow and alpha_beh from the modality_attention
    Dense(2, softmax) layer on the full test set.
    Returns dict with mean, std, and per-class-averaged values.
    """
    attn_extractor = Model(
        inputs=attention_model.inputs,
        outputs=attention_model.get_layer('modality_attention').output,
    )
    alpha = attn_extractor.predict(
        [X_te_flow_3d, X_te_beh_3d], verbose=0, batch_size=1024
    )
    # alpha shape: (n_test, 2) — columns are [alpha_flow, alpha_beh]
    return {
        'alpha_flow_mean': round(float(alpha[:, 0].mean()), 4),
        'alpha_flow_std':  round(float(alpha[:, 0].std()),  4),
        'alpha_beh_mean':  round(float(alpha[:, 1].mean()), 4),
        'alpha_beh_std':   round(float(alpha[:, 1].std()),  4),
    }


# ── Per-dataset data loader (called once) ─────────────────────────────────────

def load_data(dataset_name):
    """Load and preprocess data. Returns fixed arrays (data split is seed-invariant)."""
    csv_path, label_col = DATASETS[dataset_name]
    pipe_path = os.path.join(ROOT, 'experiments', 'baseline', 'results',
                             dataset_name, 'pipeline.pkl')
    print(f"  Loading pipeline: {pipe_path}")
    pipe = joblib.load(pipe_path)

    print(f"  Loading raw dataset: {csv_path}")
    raw = load_dataset(csv_path, label_col=label_col, seed=SEED)

    X_train_raw = raw['X_train']
    X_val_raw   = raw['X_val']
    X_test_raw  = raw['X_test']
    y_train_raw = raw['y_train']
    y_val_raw   = raw['y_val']
    y_test_raw  = raw['y_test']
    le          = raw['label_encoder']

    print(f"  Applying preprocessing pipeline...")
    X_train, y_tr_smote = pipe.transform(X_train_raw, apply_smote=True,  y=y_train_raw)
    X_val,   _          = pipe.transform(X_val_raw,   apply_smote=False)
    X_test,  _          = pipe.transform(X_test_raw,  apply_smote=False)
    feature_names       = pipe.selected_features_

    y_tr = y_tr_smote.astype(int)
    y_v  = y_val_raw.astype(int)
    y_te = y_test_raw.astype(int)
    n_classes = len(le.classes_)

    # Optional post-SMOTE subsample (fixed seed for reproducibility)
    subsample_cap = TRAIN_SUBSAMPLE.get(dataset_name)
    if subsample_cap is not None and X_train.shape[0] > subsample_cap:
        rng_ss = np.random.default_rng(SEED)
        keep = rng_ss.choice(X_train.shape[0], subsample_cap, replace=False)
        keep.sort()
        X_train = X_train[keep]
        y_tr    = y_tr[keep]
        print(f"  Post-SMOTE subsample: {X_train.shape[0]:,} (cap={subsample_cap:,})")

    # Feature partition
    partition = FEATURE_PARTITIONS[dataset_name]
    flow_idx  = get_feature_indices(feature_names, partition['flow'])
    beh_idx   = get_feature_indices(feature_names, partition['behavioral'])

    missing = set(range(X_train.shape[1])) - set(flow_idx + beh_idx)
    if missing:
        beh_idx = beh_idx + sorted(missing)
    all_idx = sorted(flow_idx + beh_idx)

    n_flow = len(flow_idx)
    n_beh  = len(beh_idx)
    n_all  = len(all_idx)
    print(f"  Features: {X_train.shape[1]} total ({n_flow} flow + {n_beh} behavioral)")
    print(f"  Train: {X_train.shape[0]:,}  Val: {X_val.shape[0]:,}  Test: {X_test.shape[0]:,}")
    print(f"  Classes ({n_classes}): {list(le.classes_)}")

    def s(X, idx): return X[:, idx][..., np.newaxis]  # slice + expand dims

    return {
        'X_tr_flow':  s(X_train, flow_idx),
        'X_val_flow': s(X_val,   flow_idx),
        'X_te_flow':  s(X_test,  flow_idx),
        'X_tr_beh':   s(X_train, beh_idx),
        'X_val_beh':  s(X_val,   beh_idx),
        'X_te_beh':   s(X_test,  beh_idx),
        'X_tr_all':   s(X_train, all_idx),
        'X_val_all':  s(X_val,   all_idx),
        'X_te_all':   s(X_test,  all_idx),
        'y_tr': y_tr, 'y_v': y_v, 'y_te': y_te,
        'le': le, 'n_classes': n_classes,
        'n_flow': n_flow, 'n_beh': n_beh, 'n_all': n_all,
        'feature_names': feature_names,
        'flow_idx': flow_idx, 'beh_idx': beh_idx,
    }


# ── Single-seed runner ────────────────────────────────────────────────────────

def run_seed(dataset_name, seed, data, out_dir):
    """Run all 5 experiments for one seed. Returns metrics dict and attn weights."""
    out_file = os.path.join(out_dir, f'multimodal_results.json')

    # Resume if already done
    if os.path.exists(out_file):
        with open(out_file) as f:
            cached = json.load(f)
        if len(cached.get('experiments', {})) == 5:
            print(f"    [seed={seed}] All 5 experiments cached — skipping.")
            attn_file = os.path.join(out_dir, 'attention_weights.json')
            attn_weights = json.load(open(attn_file)) if os.path.exists(attn_file) else None
            return cached, attn_weights

    np.random.seed(seed)
    tf.random.set_seed(seed)

    d = data
    nc = d['n_classes']
    nf = d['n_flow']
    nb = d['n_beh']
    le = d['le']

    if os.path.exists(out_file):
        with open(out_file) as f:
            results = json.load(f)
        print(f"    [seed={seed}] Resuming: {list(results.get('experiments', {}).keys())}")
    else:
        results = {
            'dataset': dataset_name, 'seed': seed,
            'n_classes': nc, 'classes': list(le.classes_),
            'n_flow_features': nf, 'n_beh_features': nb,
            'experiments': {},
        }

    def save():
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2)

    def run_exp(name, build_fn, train_X, val_X, test_X):
        if name in results.get('experiments', {}):
            print(f"      [{name}] cached F1={results['experiments'][name]['f1']:.4f}")
            return None  # return None means no model to extract attn from
        tf.keras.backend.clear_session()
        np.random.seed(seed)
        tf.random.set_seed(seed)
        model = build_fn()
        compile_and_train(model, train_X, d['y_tr'], val_X, d['y_v'], nc, name)
        m = evaluate_model(model, test_X, d['y_te'], le, nc)
        m['n_params'] = count_params(model)
        results['experiments'][name] = m
        print(f"      [{name}] F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}  FPR={m['fpr']:.4f}")
        save()
        return model

    run_exp('flow_only',
            lambda: build_cnn_single(nf, nc, 'flow_cnn'),
            d['X_tr_flow'], d['X_val_flow'], d['X_te_flow'])

    run_exp('behavioral_only',
            lambda: build_cnn_single(nb, nc, 'beh_cnn'),
            d['X_tr_beh'], d['X_val_beh'], d['X_te_beh'])

    run_exp('early_fusion',
            lambda: build_early_fusion(nf, nb, nc),
            d['X_tr_all'], d['X_val_all'], d['X_te_all'])

    run_exp('late_fusion',
            lambda: build_late_fusion(nf, nb, nc),
            [d['X_tr_flow'], d['X_tr_beh']],
            [d['X_val_flow'], d['X_val_beh']],
            [d['X_te_flow'],  d['X_te_beh']])

    # Attention fusion — also extract gate weights
    tf.keras.backend.clear_session()
    np.random.seed(seed)
    tf.random.set_seed(seed)

    attn_weights = None
    attn_file = os.path.join(out_dir, 'attention_weights.json')

    if 'attention_fusion' not in results.get('experiments', {}):
        attn_model = build_attention_fusion(nf, nb, nc)
        compile_and_train(attn_model,
                          [d['X_tr_flow'], d['X_tr_beh']],
                          d['y_tr'],
                          [d['X_val_flow'], d['X_val_beh']],
                          d['y_v'], nc, 'attention_fusion')
        m = evaluate_model(attn_model,
                           [d['X_te_flow'], d['X_te_beh']],
                           d['y_te'], le, nc)
        m['n_params'] = count_params(attn_model)
        results['experiments']['attention_fusion'] = m
        print(f"      [attention_fusion] F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}")
        save()

        # Extract actual gate weights from test set
        attn_weights = extract_attention_weights(
            attn_model, d['X_te_flow'], d['X_te_beh']
        )
        attn_weights['seed'] = seed
        with open(attn_file, 'w') as f:
            json.dump(attn_weights, f, indent=2)
        print(f"      [attention_fusion] alpha_flow={attn_weights['alpha_flow_mean']:.4f} "
              f"alpha_beh={attn_weights['alpha_beh_mean']:.4f}")
    else:
        print(f"      [attention_fusion] cached F1={results['experiments']['attention_fusion']['f1']:.4f}")
        if os.path.exists(attn_file):
            with open(attn_file) as f:
                attn_weights = json.load(f)

    return results, attn_weights


# ── Aggregation ───────────────────────────────────────────────────────────────

METRICS = ['f1', 'precision', 'recall', 'accuracy', 'fpr', 'latency_ms']
EXP_NAMES = ['flow_only', 'behavioral_only', 'early_fusion', 'late_fusion', 'attention_fusion']


def aggregate_seeds(dataset_name, seed_dirs):
    """Aggregate per-seed results into mean ± std."""
    all_results = []
    for sd in seed_dirs:
        rf = os.path.join(sd, 'multimodal_results.json')
        if os.path.exists(rf):
            with open(rf) as f:
                all_results.append(json.load(f))

    if not all_results:
        print("  No seed results to aggregate.")
        return None

    n_seeds = len(all_results)
    agg = {
        'dataset': dataset_name,
        'n_seeds': n_seeds,
        'seeds': SEEDS[:n_seeds],
        'experiments': {},
    }

    for exp in EXP_NAMES:
        exp_metrics = {}
        for metric in METRICS:
            vals = [r['experiments'][exp][metric]
                    for r in all_results
                    if exp in r.get('experiments', {})]
            if vals:
                exp_metrics[metric + '_mean'] = round(float(np.mean(vals)), 4)
                exp_metrics[metric + '_std']  = round(float(np.std(vals, ddof=0)), 4)
        # n_params is deterministic — take from first seed
        if exp in all_results[0].get('experiments', {}):
            exp_metrics['n_params'] = all_results[0]['experiments'][exp]['n_params']
        agg['experiments'][exp] = exp_metrics

    return agg


def aggregate_attention_weights(seed_dirs):
    """Aggregate attention weights across seeds."""
    all_alpha_flow = []
    all_alpha_beh  = []
    for sd in seed_dirs:
        af = os.path.join(sd, 'attention_weights.json')
        if os.path.exists(af):
            with open(af) as f:
                w = json.load(f)
            all_alpha_flow.append(w['alpha_flow_mean'])
            all_alpha_beh.append(w['alpha_beh_mean'])

    if not all_alpha_flow:
        return None

    return {
        'alpha_flow_mean': round(float(np.mean(all_alpha_flow)), 4),
        'alpha_flow_std':  round(float(np.std(all_alpha_flow, ddof=0)), 4),
        'alpha_beh_mean':  round(float(np.mean(all_alpha_beh)), 4),
        'alpha_beh_std':   round(float(np.std(all_alpha_beh, ddof=0)), 4),
        'n_seeds': len(all_alpha_flow),
        'per_seed': [
            {'seed': SEEDS[i], 'alpha_flow': all_alpha_flow[i], 'alpha_beh': all_alpha_beh[i]}
            for i in range(len(all_alpha_flow))
        ],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_dataset(dataset_name):
    print(f"\n{'='*60}")
    print(f"  Multi-Seed Run — {dataset_name}  (seeds={SEEDS})")
    print(f"{'='*60}")

    ds_dir = os.path.join(RESULTS_DIR, dataset_name)

    print("\nLoading data (once for all seeds)...")
    data = load_data(dataset_name)

    seed_dirs = []
    for seed in SEEDS:
        sd = os.path.join(ds_dir, f'seed_{seed}')
        os.makedirs(sd, exist_ok=True)
        seed_dirs.append(sd)

        print(f"\n  --- Seed {seed} ---")
        run_seed(dataset_name, seed, data, sd)

    # Aggregate
    print(f"\n  Aggregating {len(SEEDS)} seeds...")
    agg = aggregate_seeds(dataset_name, seed_dirs)
    if agg:
        agg_file = os.path.join(ds_dir, 'multiseed_results.json')
        with open(agg_file, 'w') as f:
            json.dump(agg, f, indent=2)
        print(f"  Aggregated results saved → {agg_file}")
        for exp in EXP_NAMES:
            m = agg['experiments'].get(exp, {})
            print(f"    {exp:<22} F1={m.get('f1_mean', 0):.4f}±{m.get('f1_std', 0):.4f}")

    # Aggregate attention weights
    attn_agg = aggregate_attention_weights(seed_dirs)
    if attn_agg:
        attn_agg_file = os.path.join(ds_dir, 'attention_weights_agg.json')
        with open(attn_agg_file, 'w') as f:
            json.dump(attn_agg, f, indent=2)
        print(f"\n  Attention weights (aggregated):")
        print(f"    alpha_flow = {attn_agg['alpha_flow_mean']:.4f} ± {attn_agg['alpha_flow_std']:.4f}")
        print(f"    alpha_beh  = {attn_agg['alpha_beh_mean']:.4f} ± {attn_agg['alpha_beh_std']:.4f}")
        print(f"  Saved → {attn_agg_file}")

    return agg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None,
                        choices=['TON_IoT', 'UNSW_NB15', 'CICIDS2017'])
    parser.add_argument('--all', action='store_true', help='Run all 3 datasets in sequence')
    args = parser.parse_args()

    datasets = ['TON_IoT', 'UNSW_NB15', 'CICIDS2017'] if args.all else [args.dataset]
    if not datasets[0]:
        parser.error("Specify --dataset or --all")

    for ds in datasets:
        run_dataset(ds)
        print(f"\n  [{ds}] Done.\n")

    print("All done.")
