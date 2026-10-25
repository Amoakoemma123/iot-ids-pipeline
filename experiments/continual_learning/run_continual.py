"""
experiments/continual_learning/run_continual.py
Main runner for the continual learning experiment.

Usage
-----
    python experiments/continual_learning/run_continual.py \
        --dataset TON_IoT \
        --strategies naive ewc replay lwf

Experiment flow per strategy
----------------------------
1. Load dataset → preprocess (MI + RFECV + MinMax, no SMOTE)
2. Apply task splits (task_generator.py)
3. Task 0: initial training (with SMOTE on Task 0 training data only)
4. Tasks 1..T: incremental training using chosen strategy
5. After each task: evaluate on ALL tasks seen so far → build R matrix
6. Compute FM, BWT, Plasticity, Average Accuracy
7. Save results JSON to results/<dataset>/
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_dataset
from src.data.preprocessing import PreprocessingPipeline
from src.models.continual_cnn import build_continual_cnn
from src.evaluation.cl_metrics import summarise, compute_r_matrix

from experiments.continual_learning.config import (
    DATASETS, PREPROCESSING, PREPROCESSING_OVERRIDES,
    TASK_SPLITS, CNN_INIT, CNN_INCREMENTAL,
    EWC_PARAMS, REPLAY_PARAMS, LWF_PARAMS,
    RESULTS_DIR, SEED,
)
from experiments.continual_learning.tasks.task_generator import build_tasks
from experiments.continual_learning.strategies.naive import NaiveFineTuning
from experiments.continual_learning.strategies.ewc import EWC
from experiments.continual_learning.strategies.replay import ExperienceReplay
from experiments.continual_learning.strategies.lwf import LwF
from experiments.continual_learning.strategies.gem import GEM
from experiments.continual_learning.evaluation.adaptation_rate import mean_adaptation_rate
from experiments.continual_learning.evaluation.memory_usage import CostTracker
from experiments.continual_learning.evaluation.forward_transfer import forward_transfer

from sklearn.metrics import f1_score

# ── GPU memory growth ─────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_strategy(name: str, model: tf.keras.Model, n_classes: int) -> object:
    if name == 'naive':
        return NaiveFineTuning(model, n_classes, seed=SEED)
    elif name == 'ewc':
        return EWC(model, n_classes, seed=SEED, **EWC_PARAMS)
    elif name == 'replay':
        return ExperienceReplay(model, n_classes, seed=SEED, **REPLAY_PARAMS)
    elif name == 'lwf':
        return LwF(model, n_classes, seed=SEED, **LWF_PARAMS)
    elif name == 'gem':
        return GEM(model, n_classes, seed=SEED,
                   memory_per_task=REPLAY_PARAMS['memory_size'] // 3)
    else:
        raise ValueError(f"Unknown strategy: {name}")


def evaluate_on_task(model: tf.keras.Model, task) -> float:
    """Return macro F1 on task's test set."""
    X = task.X_test.astype(np.float32)
    X_exp = np.expand_dims(X, -1)
    preds = np.argmax(model.predict(X_exp, verbose=0), axis=1)
    return float(f1_score(task.y_test, preds, average='macro', zero_division=0))


def train_one_task(model: tf.keras.Model,
                   strategy,
                   task,
                   cfg: dict) -> list:
    """
    Custom training loop for one task.
    GEM uses gem_gradient_step() for gradient projection;
    all other strategies use the standard compute_loss() path.
    Returns list of per-epoch dicts for logging.
    """
    use_gem = hasattr(strategy, 'gem_gradient_step')
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['lr'])
    X_tr = task.X_train.astype(np.float32)
    y_tr = task.y_train.astype(np.int32)

    dataset = (tf.data.Dataset
               .from_tensor_slices((X_tr, y_tr))
               .shuffle(len(X_tr), seed=SEED)
               .batch(cfg['batch_size']))

    best_val_f1 = -1.0
    patience_ctr = 0
    history = []

    for epoch in range(cfg['epochs']):
        epoch_loss = []
        for X_b, y_b in dataset:
            if use_gem:
                # GEM projects gradients to avoid increasing loss on past tasks
                X_b_exp = tf.expand_dims(X_b, -1)
                loss_val = strategy.gem_gradient_step(model, optimizer, X_b_exp, y_b)
                epoch_loss.append(loss_val)
            else:
                X_b_exp = tf.expand_dims(X_b, -1)
                with tf.GradientTape() as tape:
                    loss = strategy.compute_loss(model, X_b_exp, y_b)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                epoch_loss.append(float(loss))

        mean_loss = np.mean(epoch_loss)

        # Validation F1
        X_val_exp = np.expand_dims(task.X_val.astype(np.float32), -1)
        val_preds = np.argmax(model.predict(X_val_exp, verbose=0), axis=1)
        val_f1 = float(f1_score(task.y_val, val_preds,
                                 average='macro', zero_division=0))

        history.append({'epoch': epoch + 1, 'loss': mean_loss, 'val_f1': val_f1})

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:3d}/{cfg['epochs']}  "
                  f"loss={mean_loss:.4f}  val_F1={val_f1:.4f}")

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg['patience']:
                print(f"    Early stop at epoch {epoch+1}")
                break

    return history


# ── Main experiment ───────────────────────────────────────────────────────────

def run_experiment(dataset_name: str, strategy_names: list):
    print(f"\n{'#'*60}")
    print(f"  Continual Learning — {dataset_name}")
    print(f"  Strategies: {strategy_names}")
    print(f"{'#'*60}\n")

    # ── Load dataset ──────────────────────────────────────────────────────────
    csv_path, label_col = DATASETS[dataset_name]
    data = load_dataset(csv_path, label_col=label_col, seed=SEED)

    # ── Preprocessing ─────────────────────────────────────────────────────────
    pp_cfg = {**PREPROCESSING, **PREPROCESSING_OVERRIDES.get(dataset_name, {})}
    apply_smote = pp_cfg.pop('apply_smote', False)
    pipeline = PreprocessingPipeline(**pp_cfg)

    X_tr_pp, y_tr = pipeline.fit_transform(
        data['X_train'], data['y_train'],
        feature_names=data['feature_names'],
        apply_smote=apply_smote,
    )
    X_va_pp, y_va = pipeline.transform(data['X_val'],  y=data['y_val'])
    X_te_pp, y_te = pipeline.transform(data['X_test'], y=data['y_test'])

    n_features = X_tr_pp.shape[1]
    class_names = data['class_names']
    n_classes   = len(class_names)
    print(f"Features: {n_features}   Classes: {n_classes}   "
          f"Train: {len(X_tr_pp):,}   Val: {len(X_va_pp):,}   Test: {len(X_te_pp):,}")

    # ── Build tasks ───────────────────────────────────────────────────────────
    task_groups = TASK_SPLITS[dataset_name]
    print(f"\nBuilding {len(task_groups)} tasks…")
    tasks = build_tasks(
        X_tr_pp, y_tr, X_va_pp, y_va, X_te_pp, y_te,
        class_names=class_names,
        task_class_groups=task_groups,
    )
    if not tasks:
        print("ERROR: No tasks built — check TASK_SPLITS class names vs dataset classes.")
        return

    # ── Run each strategy ─────────────────────────────────────────────────────
    out_dir = RESULTS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for strat_name in strategy_names:
        print(f"\n{'='*55}")
        print(f"  Strategy: {strat_name.upper()}")
        print(f"{'='*55}")
        t0 = time.time()

        # Fresh model per strategy
        tf.keras.backend.clear_session()
        model = build_continual_cnn(n_features, n_classes, lr=CNN_INIT['lr'])

        strategy = make_strategy(strat_name, model, n_classes)
        session_results = []
        tracker = CostTracker()
        per_task_histories = []

        for t, task in enumerate(tasks):
            print(f"\n  --- Task {t}: {task.class_names} ---")
            cfg = CNN_INIT if t == 0 else CNN_INCREMENTAL
            strategy.before_task(t, task)

            with tracker.track_task(t):
                history = train_one_task(model, strategy, task, cfg)

            strategy.after_task(t, task, model)
            per_task_histories.append(history)

            # Save checkpoint after each task
            ckpt_dir = Path(__file__).parent / 'checkpoints' / dataset_name / strat_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_weights(str(ckpt_dir / f'task{t}.weights.h5'))

            # Evaluate on all tasks seen so far
            task_f1s = []
            for j in range(t + 1):
                f1 = evaluate_on_task(model, tasks[j])
                task_f1s.append(f1)
                print(f"    Eval Task {j} ({tasks[j].class_names[0]}…): F1={f1:.4f}")

            session_results.append({
                'task_id':  t,
                'task_f1':  task_f1s,
                'history':  history,
            })

        # ── Metrics ───────────────────────────────────────────────────────────
        R = compute_r_matrix(session_results)
        metrics = summarise(R, strategy_name=strat_name)

        # Adaptation Rate
        adapt_rate = mean_adaptation_rate(per_task_histories, threshold=0.9)
        metrics['adaptation_rate_epochs'] = adapt_rate

        # Forward Transfer
        fwt = forward_transfer(R)
        metrics['forward_transfer'] = fwt

        # Computational cost
        cost = tracker.summary()
        tracker.print_summary(strat_name)
        metrics['cost'] = cost

        metrics['elapsed_sec'] = round(time.time() - t0, 1)
        metrics['dataset'] = dataset_name
        all_results[strat_name] = metrics

        # Save per-strategy results
        out_path = out_dir / f'{strat_name}_results.json'
        with open(out_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved: {out_path}")

    # ── Save combined summary ──────────────────────────────────────────────────
    summary_path = out_dir / 'cl_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[DONE] Summary: {summary_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='TON_IoT',
                        choices=list(DATASETS.keys()))
    parser.add_argument('--strategies', nargs='+',
                        default=['naive', 'ewc', 'replay', 'lwf', 'gem'],
                        choices=['naive', 'ewc', 'replay', 'lwf', 'gem'])
    args = parser.parse_args()

    run_experiment(args.dataset, args.strategies)
