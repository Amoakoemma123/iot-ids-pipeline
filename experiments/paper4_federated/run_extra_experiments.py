"""
run_extra_experiments.py — Three additional experiments for Paper 4

1. FedAdam on UNSW-NB15        → results_unsw/fedadam_alpha0.5.json
2. DP-FedAvg (σ=1.0) on UNSW   → results_unsw/dp_fedavg_eps13.json
3. FedAvg-SGD on TON-IoT α=0.5 → results/fedavg_sgd_alpha0.5.json

Run all:  python experiments/paper4_federated/run_extra_experiments.py
Run one:  python experiments/paper4_federated/run_extra_experiments.py --exp fedadam_unsw
"""

import sys, os, json, time, argparse
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Import everything from run_federated (safe — has __name__ == '__main__' guard)
from experiments.paper4_federated.run_federated import (
    build_model, get_weights, set_weights, model_size_bytes,
    dirichlet_partition,
    local_train_fedavg, local_train_dp,
    fedavg_aggregate, fedadam_aggregate,
    run_fl,
    load_ton_iot, load_unsw_nb15,
    RESULTS_DIR, RESULTS_DIR_UNSW,
    SEED, N_ROUNDS, LOCAL_EPOCHS, BATCH_SIZE, LR, CLIP_NORM,
)

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── FedAvg-SGD local training ──────────────────────────────────────────────────
def local_train_fedavg_sgd(model, X_local, y_local, local_epochs, lr, batch_size):
    """FedAvg with SGD local optimiser — mirrors SCAFFOLD's local optimiser for fair comparison."""
    opt = tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    X_exp = np.expand_dims(X_local.astype(np.float32), -1)
    model.fit(X_exp, y_local, epochs=local_epochs, batch_size=batch_size, verbose=0)


def run_fedavg_sgd(name, clients, X_test, y_test, n_features, n_classes,
                   n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
                   lr=LR, batch_size=BATCH_SIZE, results_dir=None):
    """FedAvg with SGD local optimiser."""
    if results_dir is None:
        results_dir = RESULTS_DIR
    cache_path = results_dir / f'{name}.json'
    if cache_path.exists():
        print(f'  [{name}] Loading cached result...')
        return json.loads(cache_path.read_text())

    print(f'\n{"="*60}')
    print(f'  Experiment: {name}')
    print(f'  Algorithm:  fedavg_sgd  |  Rounds: {n_rounds}  |  Local epochs: {local_epochs}')
    print(f'  Clients:    {len(clients)}  |  Sizes: {[len(c[1]) for c in clients]}')
    print(f'{"="*60}')

    global_model = build_model(n_features, n_classes)
    global_weights = get_weights(global_model)
    m_bytes = model_size_bytes(global_model)
    bytes_per_round = len(clients) * m_bytes * 2
    history = []
    t0 = time.time()

    for rnd in range(1, n_rounds + 1):
        client_results = []
        for X_c, y_c in clients:
            client_model = build_model(n_features, n_classes)
            set_weights(client_model, global_weights)
            local_train_fedavg_sgd(client_model, X_c, y_c, local_epochs, lr, batch_size)
            client_results.append((get_weights(client_model), len(y_c)))
        global_weights = fedavg_aggregate(client_results)

        if rnd % 2 == 0 or rnd == n_rounds:
            set_weights(global_model, global_weights)
            X_exp = np.expand_dims(X_test.astype(np.float32), -1)
            with tf.device('/CPU:0'):
                preds = np.argmax(global_model.predict(X_exp, verbose=0, batch_size=2048), axis=1)
            f1   = float(f1_score(y_test, preds, average='macro', zero_division=0))
            acc  = float(accuracy_score(y_test, preds))
            prec = float(precision_score(y_test, preds, average='macro', zero_division=0))
            rec  = float(recall_score(y_test, preds, average='macro', zero_division=0))
            elapsed = time.time() - t0
            entry = {'round': rnd, 'f1': f1, 'accuracy': acc,
                     'precision': prec, 'recall': rec, 'elapsed_sec': elapsed}
            history.append(entry)
            print(f'  Round {rnd:3d}/{n_rounds}  F1={f1:.4f}  Acc={acc:.4f}  ({elapsed:.0f}s)')

    best_f1 = max(h['f1'] for h in history)
    best_round = history[[h['f1'] for h in history].index(best_f1)]['round']
    conv_round = next((h['round'] for h in history if h['f1'] >= 0.9 * best_f1), best_round)

    result = {
        'name': name, 'algorithm': 'fedavg_sgd',
        'n_clients': len(clients), 'n_rounds': n_rounds, 'local_epochs': local_epochs,
        'client_sizes': [len(c[1]) for c in clients],
        'final_metrics': history[-1],
        'best_f1': best_f1, 'best_round': best_round,
        'history': history, 'convergence_round': conv_round,
        'communication': {
            'model_size_bytes': int(m_bytes),
            'bytes_per_round': int(bytes_per_round),
            'total_bytes': int(bytes_per_round * n_rounds),
            'total_mb': round(bytes_per_round * n_rounds / 1e6, 3),
        },
        'elapsed_sec': history[-1]['elapsed_sec'],
    }
    cache_path.write_text(json.dumps(result, indent=2))
    print(f'  → Saved: {cache_path.name}  F1={best_f1:.4f}  Conv@round={conv_round}')
    return result


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='all',
                        help='fedadam_unsw | dp_unsw | fedavg_sgd | fedavg_sgd_alpha01 | all')
    args = parser.parse_args()

    def should_run(name):
        return args.exp in ('all', name)

    N_CLIENTS = 4
    ALPHA = 0.5

    # ── Exp 1: FedAdam on UNSW-NB15 ───────────────────────────────────────────
    if should_run('fedadam_unsw'):
        print('\n[EXP 1] FedAdam on UNSW-NB15 (α=0.5, N=4)')
        X_train, y_train, X_test, y_test, n_feat, n_cls = load_unsw_nb15()
        clients = dirichlet_partition(X_train, y_train, N_CLIENTS, ALPHA)
        run_fl('fedadam_alpha0.5', clients, X_test, y_test, n_feat, n_cls,
               algorithm='fedadam', server_lr=1e-3, results_dir=RESULTS_DIR_UNSW)

    # ── Exp 2: DP-FedAvg (σ=1.0) on UNSW-NB15 ────────────────────────────────
    if should_run('dp_unsw'):
        print('\n[EXP 2] DP-FedAvg (σ=1.0, ε≈13) on UNSW-NB15 (α=0.5, N=4)')
        X_train, y_train, X_test, y_test, n_feat, n_cls = load_unsw_nb15()
        clients = dirichlet_partition(X_train, y_train, N_CLIENTS, ALPHA)
        run_fl('dp_fedavg_eps13', clients, X_test, y_test, n_feat, n_cls,
               algorithm='dp_fedavg', noise_multiplier=1.0, clip_norm=CLIP_NORM,
               results_dir=RESULTS_DIR_UNSW)

    # ── Exp 3: FedAvg-SGD baseline on TON-IoT (α=0.5) ────────────────────────
    if should_run('fedavg_sgd'):
        print('\n[EXP 3] FedAvg-SGD baseline on TON-IoT (α=0.5, N=4)')
        X_train, y_train, X_test, y_test, n_feat, n_cls = load_ton_iot()
        clients = dirichlet_partition(X_train, y_train, N_CLIENTS, ALPHA)
        run_fedavg_sgd('fedavg_sgd_alpha0.5', clients, X_test, y_test, n_feat, n_cls,
                       results_dir=RESULTS_DIR)

    # ── Exp 4: FedAvg-SGD baseline on TON-IoT (α=0.1) — 50 rounds ────────────
    if should_run('fedavg_sgd_alpha01'):
        print('\n[EXP 4] FedAvg-SGD baseline on TON-IoT (α=0.1, N=4, 50 rounds)')
        X_train, y_train, X_test, y_test, n_feat, n_cls = load_ton_iot()
        clients = dirichlet_partition(X_train, y_train, N_CLIENTS, alpha=0.1)
        run_fedavg_sgd('fedavg_sgd_alpha0.1', clients, X_test, y_test, n_feat, n_cls,
                       n_rounds=50, results_dir=RESULTS_DIR)

    print('\n[DONE] All extra experiments complete.')
