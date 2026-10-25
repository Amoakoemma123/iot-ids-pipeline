"""
experiments/paper7_integrated/run_integrated.py

Paper 7 — Integrated Framework Experiment Runner.

Runs 5 ablation conditions (C0–C4) building progressively toward the
full IREAP (Intelligent, Robust, Explainable, Adaptive, Privacy-preserving)
framework:

  C0: Baseline CNN           — single-stream, centralized (Paper 1 arch)
  C1: +Multimodal            — dual-stream attention_fusion, centralized (Paper 6)
  C2: +Adversarial           — C1 + CCAT adversarial training (Paper 2)
  C3: +Federated             — C1 + FedAvg-IID 4 clients (Paper 4)
  C4: Full IREAP             — C1 + CCAT + Replay adaptation + SHAP (Papers 2,3,5)

For C4, the evaluation sequence is:
  1. Train with CCAT on Task 0 (seen attack types)
  2. Evaluate on clean test set
  3. Evaluate under PGD adversarial attack (ε=0.1, 20 steps)
  4. Adapt with Replay to Task 1 (new/unseen attack types)
  5. Evaluate on Task 0 (forgetting) and Task 1 (new task)
  6. Compute SHAP top-10 feature importance

Datasets: TON_IoT, UNSW_NB15, CICIDS2017

Usage:
  python experiments/paper7_integrated/run_integrated.py --dataset TON_IoT
  python experiments/paper7_integrated/run_integrated.py --dataset UNSW_NB15
  python experiments/paper7_integrated/run_integrated.py --all
"""

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

from experiments.baseline.config import DATASETS, SEED
from experiments.paper6_multimodal.run_multimodal_experiments import (
    FEATURE_PARTITIONS, TRAIN_SUBSAMPLE, get_feature_indices,
)
from experiments.paper7_integrated.unified_pipeline import (
    build_baseline_cnn,
    build_attention_fusion,
    train_centralized,
    train_baseline_cnn,
    train_ccat,
    train_fedavg,
    adapt_replay,
    evaluate_model,
    evaluate_baseline_cnn,
    evaluate_under_pgd,
    compute_shap,
    DualStreamReplayBuffer,
)
from src.data.loader import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')

# ── CCAT config (Paper 2 epsilon/ratio; reduced steps+epochs for integration) ─
# Paper 2 used pgd_steps=7, epochs=50 on full data (~180 min/dataset).
# For Paper 7 integration we subsample to 50K and use pgd_steps=3, epochs=30
# to keep C2/C4 under ~20 min while preserving the adversarial hardening signal.
CCAT_CONFIG = {
    'TON_IoT':    {'epsilon': 0.1, 'adv_ratio': 0.5, 'pgd_steps': 3,
                   'epochs': 30, 'patience': 8, 'rare_threshold': 500,
                   'subsample': 50_000},
    'UNSW_NB15':  {'epsilon': 0.1, 'adv_ratio': 0.5, 'pgd_steps': 3,
                   'epochs': 30, 'patience': 8, 'rare_threshold': 500,
                   'subsample': 50_000},
    # CICIDS2017: 15 classes, severe imbalance (Heartbleed=7 samples in test).
    # rare_threshold=100 excludes only the very rarest classes from adv examples.
    'CICIDS2017': {'epsilon': 0.1, 'adv_ratio': 0.5, 'pgd_steps': 3,
                   'epochs': 30, 'patience': 8, 'rare_threshold': 100,
                   'subsample': 50_000},
}

# ── FedAvg config ─────────────────────────────────────────────────────────────
# lr reduced (1e-3→2e-4) and local_epochs halved (2→1) to cut client drift
# with frozen-BN local training; rounds increased (15→25) to compensate.
FEDAVG_CONFIG = {
    'n_clients': 4, 'n_rounds': 25,
    'local_epochs': 1, 'lr': 2e-4, 'batch_size': 256,
}

# ── Replay config (matching Paper 3 optimal buffer size) ─────────────────────
REPLAY_CONFIG = {
    'buffer_size': 2000, 'replay_ratio': 0.3,
    'epochs': 15, 'lr': 5e-4, 'batch_size': 256, 'patience': 5,
}


# ── Data loader ───────────────────────────────────────────────────────────────

def load_data(dataset_name):
    """
    Load and preprocess data using the saved baseline pipeline.
    Returns flow/behavioral split arrays + task split for continual learning.
    """
    csv_path, label_col = DATASETS[dataset_name]
    pipe_path = os.path.join(ROOT, 'experiments', 'baseline', 'results',
                             dataset_name, 'pipeline.pkl')
    pipe = joblib.load(pipe_path)
    print(f"  Loading dataset: {csv_path}")
    raw = load_dataset(csv_path, label_col=label_col, seed=SEED)

    X_train, y_tr = pipe.transform(raw['X_train'], apply_smote=True, y=raw['y_train'])
    X_val,   _    = pipe.transform(raw['X_val'],   apply_smote=False)
    X_test,  _    = pipe.transform(raw['X_test'],  apply_smote=False)
    feature_names = pipe.selected_features_

    y_tr = y_tr.astype(int)
    y_v  = raw['y_val'].astype(int)
    y_te = raw['y_test'].astype(int)
    le   = raw['label_encoder']
    n_classes = len(le.classes_)

    # Subsample if needed (CICIDS2017 cap)
    cap = TRAIN_SUBSAMPLE.get(dataset_name)
    if cap and X_train.shape[0] > cap:
        rng = np.random.default_rng(SEED)
        keep = rng.choice(X_train.shape[0], cap, replace=False)
        keep.sort()
        X_train = X_train[keep]
        y_tr    = y_tr[keep]
        print(f"  Post-SMOTE subsample: {X_train.shape[0]:,}")

    # Feature partition
    partition = FEATURE_PARTITIONS[dataset_name]
    flow_idx  = get_feature_indices(feature_names, partition['flow'])
    beh_idx   = get_feature_indices(feature_names, partition['behavioral'])
    missing   = set(range(X_train.shape[1])) - set(flow_idx + beh_idx)
    if missing:
        beh_idx = beh_idx + sorted(missing)

    n_flow = len(flow_idx)
    n_beh  = len(beh_idx)
    print(f"  {n_flow} flow + {n_beh} beh features | {n_classes} classes")
    print(f"  Train: {X_train.shape[0]:,}  Val: {X_val.shape[0]:,}  "
          f"Test: {X_test.shape[0]:,}")

    def s(X, idx): return X[:, idx][..., np.newaxis]

    # Continual learning task split:
    # Task 0 = first ceil(n_classes/2) classes, Task 1 = remaining
    task0_classes = sorted(set(y_tr))[:max(1, n_classes // 2)]
    task1_classes = sorted(set(y_tr))[max(1, n_classes // 2):]

    def task_mask(y, classes): return np.isin(y, classes)

    m0_tr = task_mask(y_tr, task0_classes)
    m1_tr = task_mask(y_tr, task1_classes)
    m0_te = task_mask(y_te, task0_classes)
    m1_te = task_mask(y_te, task1_classes)

    return {
        # Full arrays (3D for dual-stream)
        'X_tr_flow':  s(X_train, flow_idx),
        'X_val_flow': s(X_val,   flow_idx),
        'X_te_flow':  s(X_test,  flow_idx),
        'X_tr_beh':   s(X_train, beh_idx),
        'X_val_beh':  s(X_val,   beh_idx),
        'X_te_beh':   s(X_test,  beh_idx),
        # Full arrays (2D for baseline CNN)
        'X_tr_all':   X_train,
        'X_val_all':  X_val,
        'X_te_all':   X_test,
        # Labels
        'y_tr': y_tr, 'y_v': y_v, 'y_te': y_te,
        # Metadata
        'le': le, 'n_classes': n_classes,
        'n_flow': n_flow, 'n_beh': n_beh,
        'n_features': X_train.shape[1],
        'feature_names': feature_names,
        'flow_features': [feature_names[i] for i in flow_idx],
        'beh_features':  [feature_names[i] for i in beh_idx],
        # Task splits for continual learning
        'task0_classes': task0_classes,
        'task1_classes': task1_classes,
        'X_tr0_flow': s(X_train[m0_tr], flow_idx),
        'X_tr0_beh':  s(X_train[m0_tr], beh_idx),
        'y_tr0':      y_tr[m0_tr],
        'X_tr1_flow': s(X_train[m1_tr], flow_idx),
        'X_tr1_beh':  s(X_train[m1_tr], beh_idx),
        'y_tr1':      y_tr[m1_tr],
        'X_te0_flow': s(X_test[m0_te], flow_idx),
        'X_te0_beh':  s(X_test[m0_te], beh_idx),
        'y_te0':      y_te[m0_te],
        'X_te1_flow': s(X_test[m1_te], flow_idx),
        'X_te1_beh':  s(X_test[m1_te], beh_idx),
        'y_te1':      y_te[m1_te],
    }


# ── Per-condition runners ─────────────────────────────────────────────────────

def run_c0_baseline(data, seed=SEED):
    """C0: Single-stream CNN trained centrally (Paper 1 architecture)."""
    print("\n  [C0] Baseline CNN (single-stream, centralized)")
    tf.keras.backend.clear_session()
    np.random.seed(seed); tf.random.set_seed(seed)

    # Stack flow+beh into one flat array
    X_tr  = data['X_tr_all'].astype(np.float32)[..., np.newaxis]
    X_val = data['X_val_all'].astype(np.float32)[..., np.newaxis]
    X_te  = data['X_te_all'].astype(np.float32)[..., np.newaxis]

    model = build_baseline_cnn(data['n_features'], data['n_classes'])
    t0 = time.time()
    train_baseline_cnn(model, X_tr, data['y_tr'], X_val, data['y_v'],
                       data['n_classes'], seed=seed)
    train_time = time.time() - t0

    metrics = evaluate_baseline_cnn(model, X_te, data['y_te'], data['n_classes'])
    metrics['train_time_min'] = round(train_time / 60, 2)
    metrics['condition'] = 'C0_baseline'
    return metrics


def run_c1_multimodal(data, seed=SEED):
    """C1: Dual-stream attention_fusion trained centrally (Paper 6)."""
    print("\n  [C1] +Multimodal (attention_fusion, centralized)")
    tf.keras.backend.clear_session()
    np.random.seed(seed); tf.random.set_seed(seed)

    model = build_attention_fusion(data['n_flow'], data['n_beh'], data['n_classes'])
    t0 = time.time()
    train_centralized(model,
                      data['X_tr_flow'], data['X_tr_beh'], data['y_tr'],
                      data['X_val_flow'], data['X_val_beh'], data['y_v'],
                      data['n_classes'], seed=seed)
    train_time = time.time() - t0

    metrics = evaluate_model(model,
                             data['X_te_flow'], data['X_te_beh'],
                             data['y_te'], data['n_classes'])
    metrics['train_time_min'] = round(train_time / 60, 2)
    metrics['condition'] = 'C1_multimodal'
    return metrics


def _ccat_subsample(data, cfg, seed):
    """Subsample training data for CCAT to keep runtime manageable."""
    cap = cfg.get('subsample')
    Xf, Xb, y = data['X_tr_flow'], data['X_tr_beh'], data['y_tr']
    if cap and len(y) > cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), cap, replace=False)
        idx.sort()
        Xf, Xb, y = Xf[idx], Xb[idx], y[idx]
        print(f"    CCAT subsample: {cap:,} / {len(data['y_tr']):,} training samples")
    return Xf, Xb, y


def run_c2_adversarial(data, dataset_name, seed=SEED):
    """C2: Dual-stream attention_fusion + CCAT adversarial training (Paper 2)."""
    print("\n  [C2] +Adversarial (attention_fusion + CCAT)")
    tf.keras.backend.clear_session()
    np.random.seed(seed); tf.random.set_seed(seed)

    cfg = CCAT_CONFIG[dataset_name]
    Xf_tr, Xb_tr, y_tr = _ccat_subsample(data, cfg, seed)
    model = build_attention_fusion(data['n_flow'], data['n_beh'], data['n_classes'])
    t0 = time.time()
    train_ccat(model,
               Xf_tr, Xb_tr, y_tr,
               data['X_val_flow'], data['X_val_beh'], data['y_v'],
               data['n_classes'],
               epsilon=cfg['epsilon'], adv_ratio=cfg['adv_ratio'],
               pgd_steps=cfg['pgd_steps'], epochs=cfg['epochs'],
               patience=cfg['patience'],
               rare_threshold=cfg['rare_threshold'], seed=seed)
    train_time = time.time() - t0

    metrics = evaluate_model(model,
                             data['X_te_flow'], data['X_te_beh'],
                             data['y_te'], data['n_classes'])

    # Also evaluate under PGD attack
    print("    Evaluating under PGD attack...")
    adv_metrics = evaluate_under_pgd(
        model, data['X_te_flow'], data['X_te_beh'], data['y_te'],
        data['n_classes'], epsilon=cfg['epsilon'], pgd_steps=20, seed=seed)

    metrics['train_time_min'] = round(train_time / 60, 2)
    metrics['adversarial'] = adv_metrics
    metrics['condition'] = 'C2_adversarial'
    return metrics


def run_c3_federated(data, seed=SEED):
    """C3: Dual-stream attention_fusion + FedAvg-IID (Paper 4)."""
    print("\n  [C3] +Federated (attention_fusion + FedAvg-IID, 4 clients)")
    tf.keras.backend.clear_session()
    np.random.seed(seed); tf.random.set_seed(seed)

    cfg = FEDAVG_CONFIG
    model = build_attention_fusion(data['n_flow'], data['n_beh'], data['n_classes'])
    t0 = time.time()
    train_fedavg(model,
                 data['X_tr_flow'], data['X_tr_beh'], data['y_tr'],
                 data['X_val_flow'], data['X_val_beh'], data['y_v'],
                 data['n_classes'],
                 n_clients=cfg['n_clients'], n_rounds=cfg['n_rounds'],
                 local_epochs=cfg['local_epochs'], lr=cfg['lr'],
                 batch_size=cfg['batch_size'], seed=seed)
    train_time = time.time() - t0

    metrics = evaluate_model(model,
                             data['X_te_flow'], data['X_te_beh'],
                             data['y_te'], data['n_classes'])
    metrics['train_time_min'] = round(train_time / 60, 2)
    metrics['n_clients'] = cfg['n_clients']
    metrics['n_rounds']  = cfg['n_rounds']
    metrics['condition'] = 'C3_federated'
    return metrics


def run_c4_full(data, dataset_name, seed=SEED):
    """
    C4: Full IREAP — dual-stream + CCAT + Replay adaptation + SHAP.

    Sequence:
      1. Train with CCAT on Task 0 data
      2. Evaluate clean and under PGD on full test set
      3. Adapt with Replay to Task 1 data (new attack types)
      4. Evaluate on Task 0 (forgetting) and Task 1 (new task)
      5. Compute SHAP on adapted model
    """
    print("\n  [C4] Full IREAP (CCAT + Replay + SHAP)")
    tf.keras.backend.clear_session()
    np.random.seed(seed); tf.random.set_seed(seed)

    cfg_ccat   = CCAT_CONFIG[dataset_name]
    cfg_replay = REPLAY_CONFIG

    # ── Step 1: CCAT training on Task 0 ──────────────────────────────────────
    print("    Step 1: CCAT training on Task 0 "
          f"({len(data['task0_classes'])} classes)...")
    # Use task-0 slice of the subsampled training set
    task0_data = {'X_tr_flow': data['X_tr0_flow'], 'X_tr_beh': data['X_tr0_beh'],
                  'y_tr': data['y_tr0']}
    Xf_t0, Xb_t0, y_t0 = _ccat_subsample(task0_data, cfg_ccat, seed)
    model = build_attention_fusion(data['n_flow'], data['n_beh'], data['n_classes'])
    t0 = time.time()
    train_ccat(model,
               Xf_t0, Xb_t0, y_t0,
               data['X_val_flow'], data['X_val_beh'], data['y_v'],
               data['n_classes'],
               epsilon=cfg_ccat['epsilon'], adv_ratio=cfg_ccat['adv_ratio'],
               pgd_steps=cfg_ccat['pgd_steps'], epochs=cfg_ccat['epochs'],
               patience=cfg_ccat['patience'],
               rare_threshold=cfg_ccat['rare_threshold'], seed=seed)
    ccat_time = time.time() - t0

    # ── Step 2: Clean + adversarial evaluation (full test set) ───────────────
    print("    Step 2: Clean + PGD evaluation...")
    clean_metrics = evaluate_model(model,
                                   data['X_te_flow'], data['X_te_beh'],
                                   data['y_te'], data['n_classes'])
    adv_metrics = evaluate_under_pgd(
        model, data['X_te_flow'], data['X_te_beh'], data['y_te'],
        data['n_classes'], epsilon=cfg_ccat['epsilon'], pgd_steps=20, seed=seed)

    # Task-0-only evaluation (before adaptation)
    metrics_task0_before = evaluate_model(model,
                                          data['X_te0_flow'], data['X_te0_beh'],
                                          data['y_te0'], data['n_classes'])

    # ── Step 3: Replay adaptation to Task 1 ──────────────────────────────────
    print(f"    Step 3: Replay adaptation to Task 1 "
          f"({len(data['task1_classes'])} new classes, "
          f"n={len(data['y_tr1']):,})...")

    buffer = DualStreamReplayBuffer(size=cfg_replay['buffer_size'], seed=seed)
    # Seed buffer with Task 0 samples
    buf_idx = np.random.default_rng(seed).choice(
        len(data['y_tr0']), min(cfg_replay['buffer_size'], len(data['y_tr0'])),
        replace=False)
    buffer.update(data['X_tr0_flow'][buf_idx],
                  data['X_tr0_beh'] [buf_idx],
                  data['y_tr0'][buf_idx])

    t1 = time.time()
    model, buffer = adapt_replay(
        model, buffer,
        data['X_tr1_flow'], data['X_tr1_beh'], data['y_tr1'],
        data['n_classes'],
        replay_ratio=cfg_replay['replay_ratio'],
        epochs=cfg_replay['epochs'], lr=cfg_replay['lr'],
        batch_size=cfg_replay['batch_size'],
        patience=cfg_replay['patience'], seed=seed)
    adapt_time = time.time() - t1

    # ── Step 4: Post-adaptation evaluation ───────────────────────────────────
    print("    Step 4: Post-adaptation evaluation...")
    metrics_task0_after = evaluate_model(model,
                                         data['X_te0_flow'], data['X_te0_beh'],
                                         data['y_te0'], data['n_classes'])
    metrics_task1 = evaluate_model(model,
                                   data['X_te1_flow'], data['X_te1_beh'],
                                   data['y_te1'], data['n_classes'])
    metrics_full_after = evaluate_model(model,
                                        data['X_te_flow'], data['X_te_beh'],
                                        data['y_te'], data['n_classes'])

    forgetting = round(
        metrics_task0_before['f1'] - metrics_task0_after['f1'], 4)

    # ── Step 5: SHAP ──────────────────────────────────────────────────────────
    print("    Step 5: SHAP explainability...")
    shap_results = compute_shap(
        model,
        data['X_tr_flow'][:500], data['X_tr_beh'][:500],
        data['X_te_flow'],        data['X_te_beh'],
        data['flow_features'], data['beh_features'],
        n_explain=100, n_background=50, seed=seed,
    )

    return {
        'condition':           'C4_full_IREAP',
        'clean_metrics':       clean_metrics,
        'adversarial':         adv_metrics,
        'task0_before_adapt':  metrics_task0_before,
        'task0_after_adapt':   metrics_task0_after,
        'task1_new':           metrics_task1,
        'full_after_adapt':    metrics_full_after,
        'forgetting_measure':  forgetting,
        'task0_classes':       [int(c) for c in data['task0_classes']],
        'task1_classes':       [int(c) for c in data['task1_classes']],
        'ccat_time_min':       round(ccat_time / 60, 2),
        'adapt_time_min':      round(adapt_time / 60, 2),
        'shap':                shap_results,
    }


# ── Main experiment orchestrator ──────────────────────────────────────────────

def run_dataset(dataset_name, resume=False):
    print(f"\n{'='*65}")
    print(f"  Paper 7 — Integrated Framework: {dataset_name}")
    print(f"{'='*65}")

    out_dir      = os.path.join(RESULTS_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file     = os.path.join(out_dir, 'integrated_results.json')
    partial_file = os.path.join(out_dir, 'partial_results.json')

    # Load partial results if resuming
    if resume and os.path.exists(partial_file):
        with open(partial_file) as f:
            results = json.load(f)
        done = list(results['conditions'].keys())
        print(f"  Resuming — conditions already done: {done}")
        data = load_data(dataset_name)
    else:
        data = load_data(dataset_name)
        results = {
            'dataset':       dataset_name,
            'seed':          SEED,
            'n_classes':     int(data['n_classes']),
            'classes':       [str(c) for c in data['le'].classes_],
            'n_flow':        int(data['n_flow']),
            'n_beh':         int(data['n_beh']),
            'n_features':    int(data['n_features']),
            'task0_classes': [int(c) for c in data['task0_classes']],
            'task1_classes': [int(c) for c in data['task1_classes']],
            'conditions':    {},
        }

    def _save_partial():
        with open(partial_file, 'w') as f:
            json.dump(results, f, indent=2)

    t_total = time.time()

    # C0: Baseline CNN
    if 'C0' not in results['conditions']:
        results['conditions']['C0'] = run_c0_baseline(data)
        print(f"    → F1={results['conditions']['C0']['f1']:.4f}")
        _save_partial()
    else:
        print(f"  [C0] Skipping (done, F1={results['conditions']['C0']['f1']:.4f})")

    # C1: +Multimodal
    if 'C1' not in results['conditions']:
        results['conditions']['C1'] = run_c1_multimodal(data)
        print(f"    → F1={results['conditions']['C1']['f1']:.4f}")
        _save_partial()
    else:
        print(f"  [C1] Skipping (done, F1={results['conditions']['C1']['f1']:.4f})")

    # C2: +Adversarial
    if 'C2' not in results['conditions']:
        results['conditions']['C2'] = run_c2_adversarial(data, dataset_name)
        print(f"    → F1={results['conditions']['C2']['f1']:.4f}  "
              f"PGD-ASR={results['conditions']['C2']['adversarial']['asr']:.4f}")
        _save_partial()
    else:
        print(f"  [C2] Skipping (done, F1={results['conditions']['C2']['f1']:.4f})")

    # C3: +Federated
    if 'C3' not in results['conditions']:
        results['conditions']['C3'] = run_c3_federated(data)
        print(f"    → F1={results['conditions']['C3']['f1']:.4f}")
        _save_partial()
    else:
        print(f"  [C3] Skipping (done, F1={results['conditions']['C3']['f1']:.4f})")

    # C4: Full IREAP
    if 'C4' not in results['conditions']:
        results['conditions']['C4'] = run_c4_full(data, dataset_name)
        c4 = results['conditions']['C4']
        print(f"    → Clean F1={c4['clean_metrics']['f1']:.4f}  "
              f"PGD-ASR={c4['adversarial']['asr']:.4f}  "
              f"FM={c4['forgetting_measure']:.4f}  "
              f"Task1 F1={c4['task1_new']['f1']:.4f}")
        _save_partial()
    else:
        c4 = results['conditions']['C4']
        print(f"  [C4] Skipping (done, Clean F1={c4['clean_metrics']['f1']:.4f})")

    results['total_time_min'] = round((time.time() - t_total) / 60, 2)

    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {out_file}")
    print(f"  Total time: {results['total_time_min']:.1f} min")

    _print_summary(dataset_name, results)
    return results


def _print_summary(dataset_name, results):
    print(f"\n{'─'*65}")
    print(f"  SUMMARY — {dataset_name}")
    print(f"{'─'*65}")
    print(f"  {'Condition':<30} {'F1':>7}  {'FPR':>7}  {'Latency':>9}  {'Params':>8}")
    print(f"  {'─'*30} {'─'*7}  {'─'*7}  {'─'*9}  {'─'*8}")

    labels = {
        'C0': 'C0: Baseline CNN',
        'C1': 'C1: +Multimodal',
        'C2': 'C2: +Adversarial',
        'C3': 'C3: +Federated',
    }
    for k, label in labels.items():
        m = results['conditions'][k]
        print(f"  {label:<30} {m['f1']:>7.4f}  {m['fpr']:>7.4f}  "
              f"{m['latency_ms']:>7.3f}ms  {m['n_params']:>8,}")

    c4 = results['conditions']['C4']
    cm = c4['clean_metrics']
    print(f"  {'C4: Full IREAP':<30} {cm['f1']:>7.4f}  {cm['fpr']:>7.4f}  "
          f"{cm['latency_ms']:>7.3f}ms  {cm['n_params']:>8,}")
    print(f"\n  C4 adversarial ASR (PGD ε=0.1): {c4['adversarial']['asr']:.4f}")
    print(f"  C4 forgetting after adaptation:  {c4['forgetting_measure']:+.4f}")
    print(f"  C4 Task-1 (new attacks) F1:      {c4['task1_new']['f1']:.4f}")
    if c4['shap']:
        top3 = list(c4['shap']['top10_features'].items())[:3]
        print(f"  C4 SHAP top-3: {top3}")
    print(f"{'─'*65}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['TON_IoT', 'UNSW_NB15', 'CICIDS2017'])
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from partial_results.json if it exists')
    args = parser.parse_args()

    if args.all:
        for ds in ['TON_IoT', 'UNSW_NB15', 'CICIDS2017']:
            run_dataset(ds, resume=args.resume)
    elif args.dataset:
        run_dataset(args.dataset, resume=args.resume)
    else:
        parser.print_help()
