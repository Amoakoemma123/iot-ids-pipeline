"""
experiments/paper6_multimodal/run_multimodal_experiments.py

Dual-stream multi-modal IoT IDS experiments.

Feature partitioning:
  Stream 1 (Flow):       packet/byte/timing statistics — network-level
  Stream 2 (Behavioral): port/protocol/state/DNS flags — host-level proxy

Five experiments per dataset:
  network_only      — CNN on flow features only
  behavioral_only   — CNN on behavioral features only
  early_fusion      — all features concatenated → single CNN
  late_fusion       — two separate CNNs → concatenate embeddings → softmax
  attention_fusion  — two CNNs + cross-modal attention → softmax

Usage:
  python experiments/paper6_multimodal/run_multimodal_experiments.py --dataset TON_IoT
  python experiments/paper6_multimodal/run_multimodal_experiments.py --dataset UNSW_NB15
"""

import argparse, json, os, sys, time
import numpy as np
import tensorflow as tf

# ── suppress TF warnings ──────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

# ── project root on path ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import joblib
from experiments.baseline.config import DATASETS, SEED
from src.data.loader import load_dataset

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense,
    Concatenate, Multiply, Add, Dropout, BatchNormalization, Reshape, Lambda
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

# ── Feature partition definitions ─────────────────────────────────────────────
# Keys are dataset names; values are (flow_features, behavioral_features)

FEATURE_PARTITIONS = {
    'TON_IoT': {
        'flow': [
            'duration', 'src_bytes', 'dst_bytes',
            'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes',
        ],
        'behavioral': [
            'src_port', 'dst_port',
            'dns_qclass', 'dns_qtype', 'dns_rcode',
            'proto_tcp', 'proto_udp',
            'service_dns', 'service_http',
            'conn_state_rej', 'conn_state_s0', 'conn_state_s1',
            'conn_state_sf', 'conn_state_sh',
            'dns_RD_t', 'dns_RA_f', 'dns_rejected_f', 'dns_rejected_t',
        ],
    },
    'UNSW_NB15': {
        'flow': [
            'id', 'dur', 'sbytes', 'dbytes', 'rate',
            'sttl', 'sload', 'dload', 'smean', 'dmean',
        ],
        'behavioral': [
            'ct_srv_src', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
            'ct_dst_src_ltm', 'ct_srv_dst', 'service_dns',
        ],
    },
    # CICIDS-2017: 67 features → 44 flow (volumetric/timing) + 23 behavioral (port/flag/state)
    'CICIDS2017': {
        'flow': [
            'Flow Duration',
            'Total Fwd Packets', 'Total Backward Packets',
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
            'Flow Bytes/s', 'Flow Packets/s',
            'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
            'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
            'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
            'Fwd Packets/s', 'Bwd Packets/s',
            'Min Packet Length', 'Max Packet Length',
            'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
            'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
            'Subflow Fwd Packets', 'Subflow Fwd Bytes',
            'Subflow Bwd Packets', 'Subflow Bwd Bytes',
            'Active Mean', 'Active Std', 'Active Max', 'Active Min',
            'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
            'Down/Up Ratio',
        ],
        'behavioral': [
            'Destination Port',
            'Fwd Packet Length Max', 'Fwd Packet Length Min',
            'Fwd Packet Length Mean', 'Fwd Packet Length Std',
            'Bwd Packet Length Max', 'Bwd Packet Length Min',
            'Bwd Packet Length Mean', 'Bwd Packet Length Std',
            'Fwd PSH Flags',
            'Fwd Header Length', 'Bwd Header Length', 'Fwd Header Length_duplicated_0',
            'FIN Flag Count', 'SYN Flag Count', 'PSH Flag Count',
            'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count',
            'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
            'act_data_pkt_fwd', 'min_seg_size_forward',
        ],
    },
}

# Post-SMOTE training set cap per dataset (None = no cap).
# CICIDS2017's SMOTE output is ~2.5M samples; cap to 200K for training speed.
TRAIN_SUBSAMPLE = {
    'TON_IoT':    None,
    'UNSW_NB15':  None,
    'CICIDS2017': 200_000,
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')

EPOCHS    = 80
BATCH     = 256
PATIENCE  = 12

# ── Model builders ────────────────────────────────────────────────────────────

def build_cnn_single(n_features, n_classes, name='cnn'):
    """1D-CNN for a single feature stream."""
    inp = Input(shape=(n_features, 1), name=f'{name}_input')
    x = Conv1D(64, 3, activation='relu', padding='same', name=f'{name}_conv1')(inp)
    x = BatchNormalization(name=f'{name}_bn1')(x)
    x = MaxPooling1D(2, name=f'{name}_pool1')(x)
    x = Conv1D(128, 3, activation='relu', padding='same', name=f'{name}_conv2')(x)
    x = BatchNormalization(name=f'{name}_bn2')(x)
    x = GlobalAveragePooling1D(name=f'{name}_gap')(x)
    x = Dense(64, activation='relu', name=f'{name}_dense1')(x)
    x = Dropout(0.3, name=f'{name}_dropout')(x)
    out = Dense(n_classes, activation='softmax', name=f'{name}_out')(x)
    return Model(inp, out, name=name)


def _build_encoder(inp, n_classes, prefix):
    """Shared CNN encoder block returning a 64-dim embedding."""
    x = Conv1D(64, 3, activation='relu', padding='same', name=f'{prefix}_conv1')(inp)
    x = BatchNormalization(name=f'{prefix}_bn1')(x)
    x = MaxPooling1D(2, name=f'{prefix}_pool1')(x)
    x = Conv1D(128, 3, activation='relu', padding='same', name=f'{prefix}_conv2')(x)
    x = BatchNormalization(name=f'{prefix}_bn2')(x)
    x = GlobalAveragePooling1D(name=f'{prefix}_gap')(x)
    x = Dense(64, activation='relu', name=f'{prefix}_dense')(x)
    return x


def build_early_fusion(n_flow, n_beh, n_classes):
    """Concatenate both streams before the CNN."""
    n_total = n_flow + n_beh
    inp = Input(shape=(n_total, 1), name='early_input')
    x = Conv1D(64, 3, activation='relu', padding='same', name='ef_conv1')(inp)
    x = BatchNormalization(name='ef_bn1')(x)
    x = MaxPooling1D(2, name='ef_pool1')(x)
    x = Conv1D(128, 3, activation='relu', padding='same', name='ef_conv2')(x)
    x = BatchNormalization(name='ef_bn2')(x)
    x = GlobalAveragePooling1D(name='ef_gap')(x)
    x = Dense(64, activation='relu', name='ef_dense1')(x)
    x = Dropout(0.3, name='ef_dropout')(x)
    out = Dense(n_classes, activation='softmax', name='ef_out')(x)
    return Model(inp, out, name='early_fusion')


def build_late_fusion(n_flow, n_beh, n_classes):
    """Two separate CNNs → concatenate 64-dim embeddings → classifier."""
    inp_flow = Input(shape=(n_flow, 1), name='lf_flow_input')
    inp_beh  = Input(shape=(n_beh,  1), name='lf_beh_input')

    enc_flow = _build_encoder(inp_flow, n_classes, 'lf_flow')
    enc_beh  = _build_encoder(inp_beh,  n_classes, 'lf_beh')

    merged = Concatenate(name='lf_concat')([enc_flow, enc_beh])
    merged = Dense(128, activation='relu', name='lf_merge_dense')(merged)
    merged = Dropout(0.3, name='lf_merge_dropout')(merged)
    out = Dense(n_classes, activation='softmax', name='lf_out')(merged)
    return Model([inp_flow, inp_beh], out, name='late_fusion')


def build_attention_fusion(n_flow, n_beh, n_classes):
    """Two CNNs + cross-modal attention → weighted sum of 64-dim embeddings."""
    inp_flow = Input(shape=(n_flow, 1), name='af_flow_input')
    inp_beh  = Input(shape=(n_beh,  1), name='af_beh_input')

    enc_flow = _build_encoder(inp_flow, n_classes, 'af_flow')  # (batch, 64)
    enc_beh  = _build_encoder(inp_beh,  n_classes, 'af_beh')   # (batch, 64)

    # Cross-modal attention: softmax over 2 modality weights
    combined = Concatenate(name='af_combined')([enc_flow, enc_beh])
    attn = Dense(2, activation='softmax', name='modality_attention')(combined)  # (batch, 2)

    # Extract scalar attention weight per modality, expand to (batch, 1) for broadcasting
    attn_flow = Lambda(lambda x: tf.expand_dims(x[:, 0], -1), name='attn_flow')(attn)
    attn_beh  = Lambda(lambda x: tf.expand_dims(x[:, 1], -1), name='attn_beh')(attn)

    # Weighted sum of embeddings
    weighted_flow = Multiply(name='w_flow')([enc_flow, attn_flow])
    weighted_beh  = Multiply(name='w_beh') ([enc_beh,  attn_beh])
    fused = Add(name='fused')([weighted_flow, weighted_beh])

    fused = Dense(64, activation='relu', name='af_final_dense')(fused)
    fused = Dropout(0.3, name='af_final_dropout')(fused)
    out   = Dense(n_classes, activation='softmax', name='af_out')(fused)
    return Model([inp_flow, inp_beh], out, name='attention_fusion')


# ── Training & evaluation ─────────────────────────────────────────────────────

def compile_and_train(model, X_train, y_train, X_val, y_val, n_classes, name='model'):
    """Compile, train, return history."""
    y_tr  = to_categorical(y_train, n_classes)
    y_val_ = to_categorical(y_val,   n_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    cbs = [
        EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5, monitor='val_loss'),
    ]
    t0 = time.time()
    history = model.fit(
        X_train, y_tr,
        validation_data=(X_val, y_val_),
        epochs=EPOCHS, batch_size=BATCH,
        callbacks=cbs, verbose=1,
    )
    train_time = round(time.time() - t0, 1)
    return history, train_time


def evaluate_model(model, X_test, y_test, le, n_classes, n_latency=1000, n_runs=5):
    """Return metrics dict."""
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    f1   = round(float(f1_score(y_test, y_pred, average='macro', zero_division=0)), 4)
    prec = round(float(precision_score(y_test, y_pred, average='macro', zero_division=0)), 4)
    rec  = round(float(recall_score(y_test, y_pred, average='macro', zero_division=0)), 4)
    acc  = round(float(accuracy_score(y_test, y_pred)), 4)

    # FPR: (FP) / (FP + TN) averaged over classes
    cm   = confusion_matrix(y_test, y_pred)
    fprs = []
    for i in range(n_classes):
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fprs.append(fp / (fp + tn + 1e-9))
    fpr = round(float(np.mean(fprs)), 4)

    # Latency — handle single or multi-input models
    n_samples = X_test[0].shape[0] if isinstance(X_test, list) else len(X_test)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(n_samples, min(n_latency, n_samples), replace=False)
    if isinstance(X_test, list):
        lat_in = [x[idx] for x in X_test]
    else:
        lat_in = X_test[idx]
    n_lat_samples = min(n_latency, n_samples)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(lat_in, verbose=0, batch_size=1000)
        times.append((time.perf_counter() - t0) * 1000 / n_lat_samples)
    latency_ms = round(float(np.mean(times)), 3)

    # Per-class F1
    per_class_f1 = {}
    f1s = f1_score(y_test, y_pred, average=None, zero_division=0)
    for i, cls in enumerate(le.classes_):
        per_class_f1[str(cls)] = round(float(f1s[i]), 4)

    return {
        'f1': f1, 'precision': prec, 'recall': rec,
        'accuracy': acc, 'fpr': fpr,
        'latency_ms': latency_ms,
        'per_class_f1': per_class_f1,
    }


def count_params(model):
    return int(model.count_params())


# ── Feature index helpers ──────────────────────────────────────────────────────

def get_feature_indices(all_features, subset):
    """Return indices of `subset` that exist in `all_features`."""
    idx = []
    for f in subset:
        try:
            idx.append(list(all_features).index(f))
        except ValueError:
            pass
    return idx


# ── Main ──────────────────────────────────────────────────────────────────────

def run_dataset(dataset_name):
    print(f"\n{'='*60}")
    print(f"  Paper 6 Multimodal — {dataset_name}")
    print(f"{'='*60}")

    out_dir = os.path.join(RESULTS_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'multimodal_results.json')

    # ── Load data via saved baseline pipeline (matches Paper 1 exactly) ───────
    csv_path, label_col = DATASETS[dataset_name]
    pipe_path = os.path.join(ROOT, 'experiments', 'baseline', 'results',
                             dataset_name, 'pipeline.pkl')

    print("Loading saved preprocessing pipeline...")
    pipe = joblib.load(pipe_path)

    print("Loading raw dataset...")
    raw = load_dataset(csv_path, label_col=label_col, seed=SEED)

    X_train_raw = raw['X_train']
    X_val_raw   = raw['X_val']
    X_test_raw  = raw['X_test']
    y_train_raw = raw['y_train']
    y_val_raw   = raw['y_val']
    y_test_raw  = raw['y_test']
    le          = raw['label_encoder']

    print("Applying saved preprocessing pipeline...")
    # Use already-fitted pipeline: MI filter + RFECV + scaler are already fit
    # apply_smote=True on train to match Paper 1 balanced training set
    X_train, y_tr_smote = pipe.transform(X_train_raw, apply_smote=True,  y=y_train_raw)
    X_val,   _          = pipe.transform(X_val_raw,   apply_smote=False)
    X_test,  _          = pipe.transform(X_test_raw,  apply_smote=False)
    feature_names       = pipe.selected_features_

    # Labels: SMOTE returns integer-encoded; val/test from raw are already int
    y_tr = y_tr_smote.astype(int)
    y_v  = y_val_raw.astype(int)
    y_te = y_test_raw.astype(int)
    n_classes = len(le.classes_)

    # Optionally subsample post-SMOTE training set (for large datasets like CICIDS2017)
    subsample_cap = TRAIN_SUBSAMPLE.get(dataset_name)
    if subsample_cap is not None and X_train.shape[0] > subsample_cap:
        from sklearn.model_selection import train_test_split
        rng_ss = np.random.default_rng(SEED)
        keep = rng_ss.choice(X_train.shape[0], subsample_cap, replace=False)
        keep.sort()
        X_train = X_train[keep]
        y_tr    = y_tr[keep]
        print(f"  Post-SMOTE subsample: {X_train.shape[0]:,} samples (cap={subsample_cap:,})")

    print(f"  Features after preprocessing: {X_train.shape[1]}")
    print(f"  Train (post-SMOTE): {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")
    print(f"  Feature names: {list(feature_names)}")
    print(f"  Classes ({n_classes}): {list(le.classes_)}")

    # ── Feature partition ─────────────────────────────────────────────────────
    partition = FEATURE_PARTITIONS[dataset_name]
    flow_idx  = get_feature_indices(feature_names, partition['flow'])
    beh_idx   = get_feature_indices(feature_names, partition['behavioral'])
    all_idx   = sorted(flow_idx + beh_idx)

    # Sanity: all_idx should cover all features (or close)
    missing = set(range(X_train.shape[1])) - set(all_idx)
    if missing:
        # Assign unassigned features to behavioral stream
        beh_idx = beh_idx + sorted(missing)
    all_idx = sorted(flow_idx + beh_idx)

    n_flow = len(flow_idx)
    n_beh  = len(beh_idx)
    n_all  = len(all_idx)
    print(f"  Flow features ({n_flow}): {[feature_names[i] for i in flow_idx]}")
    print(f"  Behavioral features ({n_beh}): {[feature_names[i] for i in beh_idx]}")

    flow_names = [feature_names[i] for i in flow_idx]
    beh_names  = [feature_names[i] for i in beh_idx]

    # Slice feature matrices
    def slice_features(X, idx):
        return X[:, idx]

    X_tr_flow  = slice_features(X_train, flow_idx)
    X_val_flow = slice_features(X_val,   flow_idx)
    X_te_flow  = slice_features(X_test,  flow_idx)

    X_tr_beh   = slice_features(X_train, beh_idx)
    X_val_beh  = slice_features(X_val,   beh_idx)
    X_te_beh   = slice_features(X_test,  beh_idx)

    X_tr_all   = slice_features(X_train, all_idx)
    X_val_all  = slice_features(X_val,   all_idx)
    X_te_all   = slice_features(X_test,  all_idx)

    # Reshape for 1D-CNN: (samples, features, 1)
    def expand(X): return X[..., np.newaxis]

    # Load existing results if available (for crash recovery)
    if os.path.exists(out_file):
        with open(out_file) as f:
            results = json.load(f)
        print(f"  Resuming from cached results: {list(results.get('experiments', {}).keys())}")
    else:
        results = {
            'dataset': dataset_name,
            'n_classes': n_classes,
            'classes': list(le.classes_),
            'n_flow_features': n_flow,
            'n_beh_features': n_beh,
            'n_total_features': n_all,
            'flow_features': flow_names,
            'behavioral_features': beh_names,
            'experiments': {},
        }

    def save():
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved → {out_file}")

    def run_exp(name, build_fn, train_X, train_y_one_hot=None,
                val_Xv=None, test_Xv=None, eval_X=None):
        """Run a single experiment, skipping if already cached."""
        if name in results.get('experiments', {}):
            print(f"\n--- {name} [CACHED F1={results['experiments'][name]['f1']:.4f}] ---")
            return
        print(f"\n--- {name} ---")
        tf.keras.backend.clear_session()
        model = build_fn()
        hist, t = compile_and_train(
            model, train_X, y_tr,
            val_Xv if val_Xv is not None else train_X, y_v,
            n_classes, name,
        )
        ev_input = eval_X if eval_X is not None else test_Xv
        m = evaluate_model(model, ev_input, y_te, le, n_classes)
        m['train_time_sec'] = t
        m['n_params'] = count_params(model)
        results['experiments'][name] = m
        print(f"  F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}  "
              f"FPR={m['fpr']:.4f}  Latency={m['latency_ms']:.3f}ms")
        save()

    # ── Experiment 1: Flow-only ───────────────────────────────────────────────
    run_exp('flow_only',
            lambda: build_cnn_single(n_flow, n_classes, 'flow_cnn'),
            expand(X_tr_flow), val_Xv=expand(X_val_flow), eval_X=expand(X_te_flow))

    # ── Experiment 2: Behavioral-only ─────────────────────────────────────────
    run_exp('behavioral_only',
            lambda: build_cnn_single(n_beh, n_classes, 'beh_cnn'),
            expand(X_tr_beh), val_Xv=expand(X_val_beh), eval_X=expand(X_te_beh))

    # ── Experiment 3: Early Fusion ────────────────────────────────────────────
    run_exp('early_fusion',
            lambda: build_early_fusion(n_flow, n_beh, n_classes),
            expand(X_tr_all), val_Xv=expand(X_val_all), eval_X=expand(X_te_all))

    # ── Experiment 4: Late Fusion ─────────────────────────────────────────────
    run_exp('late_fusion',
            lambda: build_late_fusion(n_flow, n_beh, n_classes),
            [expand(X_tr_flow), expand(X_tr_beh)],
            val_Xv=[expand(X_val_flow), expand(X_val_beh)],
            eval_X=[expand(X_te_flow),  expand(X_te_beh)])

    # ── Experiment 5: Attention Fusion ────────────────────────────────────────
    run_exp('attention_fusion',
            lambda: build_attention_fusion(n_flow, n_beh, n_classes),
            [expand(X_tr_flow), expand(X_tr_beh)],
            val_Xv=[expand(X_val_flow), expand(X_val_beh)],
            eval_X=[expand(X_te_flow),  expand(X_te_beh)])

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Summary — {dataset_name}")
    print(f"{'='*60}")
    for exp_name, m in results['experiments'].items():
        print(f"  {exp_name:<22} F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}  "
              f"FPR={m['fpr']:.4f}  Lat={m['latency_ms']:.3f}ms  "
              f"Params={m['n_params']:,}")

    # ── Generate figures ───────────────────────────────────────────────────────
    try:
        generate_figures(results, dataset_name)
    except Exception as e:
        print(f"  [WARN] Figure generation failed: {e}")

    return results


def generate_figures(results, dataset_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(FIGURES_DIR, dataset_name)
    os.makedirs(fig_dir, exist_ok=True)

    exps = results['experiments']
    exp_names = list(exps.keys())
    labels = {
        'flow_only': 'Flow-Only',
        'behavioral_only': 'Behavioral-Only',
        'early_fusion': 'Early Fusion',
        'late_fusion': 'Late Fusion',
        'attention_fusion': 'Attention Fusion',
    }

    # ── Fig 1: F1 comparison bar chart ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    f1s = [exps[e]['f1'] for e in exp_names]
    bars = ax.bar([labels[e] for e in exp_names], f1s,
                  color=['#4472C4', '#ED7D31', '#A9D18E', '#5B9BD5', '#FF0000'],
                  width=0.6, edgecolor='black', linewidth=0.5)
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel('Macro F1-Score')
    ax.set_title(f'Multi-Modal Fusion Comparison — {dataset_name}')
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', labelsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_f1_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Fig 2: Precision / Recall / F1 grouped bar chart ─────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(exp_names))
    w = 0.25
    p_vals = [exps[e]['precision'] for e in exp_names]
    r_vals = [exps[e]['recall']    for e in exp_names]
    f_vals = [exps[e]['f1']        for e in exp_names]
    ax.bar(x - w,   p_vals, w, label='Precision', color='#4472C4', edgecolor='black', linewidth=0.4)
    ax.bar(x,       r_vals, w, label='Recall',    color='#ED7D31', edgecolor='black', linewidth=0.4)
    ax.bar(x + w,   f_vals, w, label='F1',        color='#A9D18E', edgecolor='black', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[e] for e in exp_names], fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title(f'Precision / Recall / F1 — {dataset_name}')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_precision_recall_f1.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Fig 3: FPR comparison ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    fprs = [exps[e]['fpr'] for e in exp_names]
    bars = ax.bar([labels[e] for e in exp_names], fprs,
                  color=['#4472C4', '#ED7D31', '#A9D18E', '#5B9BD5', '#FF0000'],
                  width=0.6, edgecolor='black', linewidth=0.5)
    for bar, v in zip(bars, fprs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Mean False Positive Rate')
    ax.set_title(f'False Positive Rate — {dataset_name}')
    ax.tick_params(axis='x', labelsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_fpr.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Fig 4: Latency vs F1 scatter ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    colours = ['#4472C4', '#ED7D31', '#A9D18E', '#5B9BD5', '#C00000']
    for i, e in enumerate(exp_names):
        ax.scatter(exps[e]['latency_ms'], exps[e]['f1'],
                   s=120, color=colours[i], edgecolors='black', linewidths=0.5,
                   label=labels[e], zorder=5)
    ax.set_xlabel('Mean Inference Latency (ms/sample)')
    ax.set_ylabel('Macro F1-Score')
    ax.set_title(f'Latency–Accuracy Trade-off — {dataset_name}')
    ax.legend(fontsize=8)
    ax.grid(linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_latency_f1.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Fig 5: Per-class F1 heatmap (attention vs flow-only vs beh-only) ─────
    try:
        classes = results['classes']
        compare_exps = ['flow_only', 'behavioral_only', 'early_fusion', 'late_fusion', 'attention_fusion']
        compare_exps = [e for e in compare_exps if e in exps]
        matrix = np.array([
            [exps[e]['per_class_f1'].get(c, 0.0) for c in classes]
            for e in compare_exps
        ])
        fig, ax = plt.subplots(figsize=(max(8, len(classes)*0.8), 4))
        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(compare_exps)))
        ax.set_yticklabels([labels[e] for e in compare_exps], fontsize=9)
        ax.set_title(f'Per-Class F1 Heatmap — {dataset_name}')
        plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(len(compare_exps)):
            for j in range(len(classes)):
                v = matrix[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=6, color='black' if 0.3 < v < 0.8 else 'white')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'fig_per_class_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  [WARN] Per-class heatmap failed: {e}")

    print(f"  Figures saved → {fig_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='TON_IoT',
                        choices=['TON_IoT', 'UNSW_NB15', 'CICIDS2017'],
                        help='Dataset to run experiments on')
    args = parser.parse_args()

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    run_dataset(args.dataset)
    print("\nDone.")
