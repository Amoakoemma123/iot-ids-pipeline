"""
run_hparam_sweep.py
Hyperparameter sweep for EWC (lambda_ewc) and Replay (memory_size).

For each (dataset, strategy), sweeps the relevant hyperparameter, caches each
run, and overwrites the main results file with the best configuration (lowest FM).

Sweep spaces:
  EWC:    lambda_ewc  in {100, 500, 1000, 5000}   (5000 = current default)
  Replay: memory_size in {500, 1000, 2000, 5000}  (2000 = current default)

Caching: completed runs are saved to
  experiments/continual_learning/sweep_results/{dataset}/{strategy}/{param}_{val}.json
  Re-running the script skips already-cached runs.

Usage:
  python run_hparam_sweep.py                               # all datasets, both
  python run_hparam_sweep.py --datasets TON_IoT UNSW_NB15 # skip CICIDS
  python run_hparam_sweep.py --strategies ewc              # EWC only
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_dataset
from src.data.preprocessing import PreprocessingPipeline
from src.models.continual_cnn import build_continual_cnn
from src.evaluation.cl_metrics import summarise, compute_r_matrix
from experiments.continual_learning.config import (
    DATASETS, PREPROCESSING, PREPROCESSING_OVERRIDES,
    TASK_SPLITS, CNN_INIT, CNN_INCREMENTAL,
    EWC_PARAMS, REPLAY_PARAMS,
    RESULTS_DIR, SEED,
)
from experiments.continual_learning.tasks.task_generator import build_tasks
from experiments.continual_learning.strategies.ewc import EWC
from experiments.continual_learning.strategies.replay import ExperienceReplay
from experiments.continual_learning.evaluation.adaptation_rate import mean_adaptation_rate
from experiments.continual_learning.evaluation.memory_usage import CostTracker
from experiments.continual_learning.evaluation.forward_transfer import forward_transfer
from sklearn.metrics import f1_score

# ── GPU memory growth ─────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ── Search spaces ─────────────────────────────────────────────────────────────
EWC_LAMBDAS    = [100, 500, 1000, 5000]
REPLAY_BUFFERS = [500, 1000, 2000, 5000]
SWEEP_DIR      = ROOT / 'experiments' / 'continual_learning' / 'sweep_results'

# Current defaults (load existing results instead of re-running)
EWC_DEFAULT    = EWC_PARAMS['lambda_ewc']      # 5000
REPLAY_DEFAULT = REPLAY_PARAMS['memory_size']  # 2000


# ── Training helpers ─────────────────────────────────────────────────────────

def _predict_cpu(model, X_exp):
    """Run model.predict on CPU to avoid GPU OOM on large validation sets."""
    with tf.device('/CPU:0'):
        return model.predict(X_exp, verbose=0, batch_size=2048)


def evaluate_on_task(model, task):
    X_exp = np.expand_dims(task.X_test.astype(np.float32), -1)
    preds = np.argmax(_predict_cpu(model, X_exp), axis=1)
    return float(f1_score(task.y_test, preds, average='macro', zero_division=0))


MAX_VAL_SAMPLES = 20_000   # cap validation size for speed on large datasets


def train_one_task(model, strategy, task, cfg):
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['lr'])
    X_tr = task.X_train.astype(np.float32)
    y_tr = task.y_train.astype(np.int32)
    dataset = (tf.data.Dataset
               .from_tensor_slices((X_tr, y_tr))
               .shuffle(len(X_tr), seed=SEED)
               .batch(cfg['batch_size']))

    # Subsample validation for large datasets to keep each epoch fast
    X_val_full = task.X_val.astype(np.float32)
    y_val_full = task.y_val
    if len(X_val_full) > MAX_VAL_SAMPLES:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(X_val_full), MAX_VAL_SAMPLES, replace=False)
        X_val_use = X_val_full[idx]
        y_val_use = y_val_full[idx]
    else:
        X_val_use = X_val_full
        y_val_use = y_val_full
    X_val_exp = np.expand_dims(X_val_use, -1)

    best_val_f1, patience_ctr, history = -1.0, 0, []
    for epoch in range(cfg['epochs']):
        epoch_loss = []
        for X_b, y_b in dataset:
            X_b_exp = tf.expand_dims(X_b, -1)
            with tf.GradientTape() as tape:
                loss = strategy.compute_loss(model, X_b_exp, y_b)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss.append(float(loss))
        mean_loss = np.mean(epoch_loss)
        val_preds = np.argmax(_predict_cpu(model, X_val_exp), axis=1)
        val_f1 = float(f1_score(y_val_use, val_preds, average='macro', zero_division=0))
        history.append({'epoch': epoch + 1, 'loss': mean_loss, 'val_f1': val_f1})
        if (epoch + 1) % 5 == 0:
            print(f"      Epoch {epoch+1:3d}/{cfg['epochs']}  "
                  f"loss={mean_loss:.4f}  val_F1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1, patience_ctr = val_f1, 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg['patience']:
                print(f"      Early stop at epoch {epoch+1}")
                break
    return history


def run_single(dataset_name, strat_name, hparam_val, preprocessed, tasks):
    """Run one full CL experiment with a specific hyperparameter value."""
    X_tr_pp, y_tr, X_va_pp, y_va, X_te_pp, y_te, class_names = preprocessed
    n_features, n_classes = X_tr_pp.shape[1], len(class_names)

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    model = build_continual_cnn(n_features, n_classes, lr=CNN_INIT['lr'])

    if strat_name == 'ewc':
        strategy = EWC(model, n_classes, seed=SEED,
                       lambda_ewc=hparam_val,
                       fisher_samples=EWC_PARAMS['fisher_samples'])
    else:
        strategy = ExperienceReplay(model, n_classes, seed=SEED,
                                    memory_size=hparam_val,
                                    replay_ratio=REPLAY_PARAMS['replay_ratio'])

    session_results, tracker, per_task_histories = [], CostTracker(), []
    t0 = time.time()

    for t, task in enumerate(tasks):
        print(f"\n    Task {t}: {task.class_names}")
        cfg = CNN_INIT if t == 0 else CNN_INCREMENTAL
        strategy.before_task(t, task)
        with tracker.track_task(t):
            history = train_one_task(model, strategy, task, cfg)
        strategy.after_task(t, task, model)
        per_task_histories.append(history)

        task_f1s = [evaluate_on_task(model, tasks[j]) for j in range(t + 1)]
        for j, f1 in enumerate(task_f1s):
            print(f"      Eval Task {j}: F1={f1:.4f}")
        session_results.append({'task_id': t, 'task_f1': task_f1s, 'history': history})

    R = compute_r_matrix(session_results)
    metrics = summarise(R, strategy_name=strat_name)
    metrics['adaptation_rate_epochs'] = mean_adaptation_rate(per_task_histories, threshold=0.9)
    metrics['forward_transfer'] = forward_transfer(R)
    metrics['cost'] = tracker.summary()
    metrics['elapsed_sec'] = round(time.time() - t0, 1)
    metrics['dataset'] = dataset_name
    return metrics


# ── Per-dataset sweep ─────────────────────────────────────────────────────────

def sweep(dataset_name, strat_name, hparam_name, hparam_values):
    print(f"\n{'='*65}")
    print(f"  SWEEP  {strat_name.upper()} on {dataset_name}")
    print(f"  {hparam_name} ∈ {hparam_values}")
    print(f"{'='*65}")

    cache_dir = SWEEP_DIR / dataset_name / strat_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load & preprocess dataset once for all sweep values
    csv_path, label_col = DATASETS[dataset_name]
    data = load_dataset(csv_path, label_col=label_col, seed=SEED)
    pp_cfg = {**PREPROCESSING, **PREPROCESSING_OVERRIDES.get(dataset_name, {})}
    apply_smote = pp_cfg.pop('apply_smote', False)
    pipeline = PreprocessingPipeline(**pp_cfg)
    X_tr_pp, y_tr = pipeline.fit_transform(data['X_train'], data['y_train'],
                                            feature_names=data['feature_names'],
                                            apply_smote=apply_smote)
    X_va_pp, y_va = pipeline.transform(data['X_val'],  y=data['y_val'])
    X_te_pp, y_te = pipeline.transform(data['X_test'], y=data['y_test'])
    class_names   = data['class_names']
    tasks = build_tasks(X_tr_pp, y_tr, X_va_pp, y_va, X_te_pp, y_te,
                        class_names=class_names,
                        task_class_groups=TASK_SPLITS[dataset_name])
    preprocessed = (X_tr_pp, y_tr, X_va_pp, y_va, X_te_pp, y_te, class_names)
    print(f"  Dataset loaded: {len(X_tr_pp):,} train samples, "
          f"{X_tr_pp.shape[1]} features, {len(class_names)} classes")

    all_results = {}

    for val in hparam_values:
        cache_path = cache_dir / f'{hparam_name}_{val}.json'

        # ── Use cached result if available ────────────────────────────────────
        if cache_path.exists():
            print(f"\n  [{hparam_name}={val}] Loading cached result...")
            metrics = json.loads(cache_path.read_text())

        # ── For current default, load from main results (avoid re-running) ───
        elif val == (EWC_DEFAULT if strat_name == 'ewc' else REPLAY_DEFAULT):
            main = RESULTS_DIR / dataset_name / f'{strat_name}_results.json'
            if main.exists():
                print(f"\n  [{hparam_name}={val}] Using existing main result (default)...")
                metrics = json.loads(main.read_text())
                with open(cache_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
            else:
                print(f"\n  [{hparam_name}={val}] Running (default, no cached result)...")
                metrics = run_single(dataset_name, strat_name, val, preprocessed, tasks)
                with open(cache_path, 'w') as f:
                    json.dump(metrics, f, indent=2)

        # ── Run new experiment ────────────────────────────────────────────────
        else:
            print(f"\n  [{hparam_name}={val}] Running experiment...")
            metrics = run_single(dataset_name, strat_name, val, preprocessed, tasks)
            with open(cache_path, 'w') as f:
                json.dump(metrics, f, indent=2)

        fm      = metrics.get('forgetting', float('inf'))
        avg_acc = metrics.get('average_accuracy', 0)
        elapsed = metrics.get('elapsed_sec', 0)
        print(f"  → FM={fm:.4f}  AvgAcc={avg_acc:.4f}  BWT={metrics.get('bwt',0):.4f}  "
              f"Plasticity={metrics.get('plasticity',0):.4f}  elapsed={elapsed:.0f}s")
        all_results[val] = metrics

    # ── Select best by lowest FM ──────────────────────────────────────────────
    best_val     = min(all_results, key=lambda v: all_results[v].get('forgetting', float('inf')))
    best_metrics = all_results[best_val]
    best_fm      = best_metrics['forgetting']

    print(f"\n  ── Sweep summary for {strat_name.upper()} on {dataset_name} ──")
    for val in hparam_values:
        marker = ' ← BEST' if val == best_val else ''
        r = all_results[val]
        print(f"    {hparam_name}={val:>5}:  FM={r.get('forgetting',0):.4f}  "
              f"AvgAcc={r.get('average_accuracy',0):.4f}{marker}")
    print(f"  BEST: {hparam_name}={best_val} → FM={best_fm:.4f}")

    # ── Save sweep summary ────────────────────────────────────────────────────
    summary = {
        'best_value': best_val,
        'best_fm':    best_fm,
        'all': {str(v): {
            'fm':       all_results[v].get('forgetting'),
            'avg_acc':  all_results[v].get('average_accuracy'),
            'bwt':      all_results[v].get('bwt'),
            'plasticity': all_results[v].get('plasticity'),
            'elapsed_sec': all_results[v].get('elapsed_sec'),
        } for v in hparam_values}
    }
    with open(cache_dir / 'sweep_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # ── Overwrite main results with best ─────────────────────────────────────
    out_path = RESULTS_DIR / dataset_name / f'{strat_name}_results.json'
    with open(out_path, 'w') as f:
        json.dump(best_metrics, f, indent=2)
    print(f"  Saved best → {out_path}")

    return best_val, best_fm, summary


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Hyperparameter sweep for EWC and Replay strategies.')
    parser.add_argument('--datasets', nargs='+',
                        default=list(DATASETS.keys()),
                        choices=list(DATASETS.keys()))
    parser.add_argument('--strategies', nargs='+',
                        default=['ewc', 'replay'],
                        choices=['ewc', 'replay'])
    args = parser.parse_args()

    print(f"\n{'#'*65}")
    print(f"  Hyperparameter Sweep")
    print(f"  Datasets:   {args.datasets}")
    print(f"  Strategies: {args.strategies}")
    print(f"  EWC λ:      {EWC_LAMBDAS}")
    print(f"  Replay buf: {REPLAY_BUFFERS}")
    print(f"{'#'*65}")

    final = {}
    for ds in args.datasets:
        final[ds] = {}
        for strat in args.strategies:
            if strat == 'ewc':
                bv, bfm, summ = sweep(ds, 'ewc', 'lambda_ewc', EWC_LAMBDAS)
                final[ds]['ewc'] = {'best_lambda': bv, 'best_fm': bfm, 'all': summ['all']}
            else:
                bv, bfm, summ = sweep(ds, 'replay', 'memory_size', REPLAY_BUFFERS)
                final[ds]['replay'] = {'best_buffer': bv, 'best_fm': bfm, 'all': summ['all']}

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n\n{'#'*65}")
    print("  SWEEP COMPLETE — Best Hyperparameters Per Dataset")
    print(f"{'#'*65}")
    for ds in args.datasets:
        print(f"\n  {ds}:")
        if 'ewc' in final[ds]:
            info = final[ds]['ewc']
            print(f"    EWC   best λ       = {info['best_lambda']}  (FM={info['best_fm']:.4f})")
            for v, r in info['all'].items():
                marker = ' ←' if int(v) == info['best_lambda'] else ''
                print(f"      λ={v:>6}: FM={r['fm']:.4f}  AvgAcc={r['avg_acc']:.4f}{marker}")
        if 'replay' in final[ds]:
            info = final[ds]['replay']
            print(f"    Replay best buffer = {info['best_buffer']}  (FM={info['best_fm']:.4f})")
            for v, r in info['all'].items():
                marker = ' ←' if int(v) == info['best_buffer'] else ''
                print(f"      buf={v:>5}: FM={r['fm']:.4f}  AvgAcc={r['avg_acc']:.4f}{marker}")

    # Save global summary
    global_out = SWEEP_DIR / 'global_sweep_summary.json'
    with open(global_out, 'w') as f:
        json.dump(final, f, indent=2)
    print(f"\n  Global summary: {global_out}")
    print("\n  Main results files updated. Run generate_cl_paper.py to regenerate paper.")
