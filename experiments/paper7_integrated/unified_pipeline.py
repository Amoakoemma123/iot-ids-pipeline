"""
experiments/paper7_integrated/unified_pipeline.py

Unified IoT IDS Framework — Paper 7.

Integrates five capabilities from Papers 1–6:
  - Dual-stream multimodal CNN with attention fusion  (Paper 6)
  - CCAT adversarial training                        (Paper 2)
  - FedAvg federated learning                        (Paper 4)
  - Experience Replay continual learning             (Paper 3)
  - SHAP explainability at inference time            (Paper 5)

Each capability is a self-contained module that can be enabled independently.
The run_integrated.py experiment runner uses this class to evaluate 5 ablation
conditions (C0–C4) that progressively add each module.

Architecture
------------
Core model: dual-stream attention_fusion CNN (two 1D-CNN encoders + soft gate)
  - Flow encoder:      Conv1D(64,3) → Pool → Conv1D(128,3) → GAP → Dense(64)
  - Behavioral encoder: same architecture, separate weights
  - Gate:             Dense(2, softmax) on concatenated embeddings
  - Fusion:           weighted sum of 64-dim embeddings → Dense(64) → softmax

Training options (can be combined):
  - Centralized: standard cross-entropy with Adam + early stopping
  - CCAT:        class-conditional PGD adversarial training on joint feature space
  - FedAvg:      IID partition across N clients, weighted averaging over R rounds
  - Replay:      experience replay buffer for continual adaptation to new tasks

Inference options:
  - Standard predict: argmax of softmax output
  - SHAP:        KernelExplainer on background sample, concatenated feature space
"""

import os
import sys
import time
import json
import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense,
    Concatenate, Multiply, Add, Dropout, BatchNormalization, Lambda,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass


# ── Model builders ────────────────────────────────────────────────────────────

def _cnn_encoder(inp, prefix):
    """Shared 1D-CNN encoder block → 64-dim embedding."""
    x = Conv1D(64, 3, activation='relu', padding='same', name=f'{prefix}_c1')(inp)
    x = BatchNormalization(name=f'{prefix}_bn1')(x)
    x = MaxPooling1D(2, padding='same', name=f'{prefix}_mp1')(x)
    x = Conv1D(128, 3, activation='relu', padding='same', name=f'{prefix}_c2')(x)
    x = BatchNormalization(name=f'{prefix}_bn2')(x)
    x = GlobalAveragePooling1D(name=f'{prefix}_gap')(x)
    x = Dense(64, activation='relu', name=f'{prefix}_emb')(x)
    x = Dropout(0.3, name=f'{prefix}_drop')(x)
    return x


def build_baseline_cnn(n_features, n_classes):
    """Single-stream CNN (Paper 1 architecture)."""
    inp = Input(shape=(n_features, 1), name='base_input')
    x = _cnn_encoder(inp, 'base')
    out = Dense(n_classes, activation='softmax', name='base_out')(x)
    return Model(inp, out, name='baseline_cnn')


def build_attention_fusion(n_flow, n_beh, n_classes):
    """Dual-stream attention_fusion CNN (Paper 6 best architecture)."""
    inp_flow = Input(shape=(n_flow, 1), name='flow_input')
    inp_beh  = Input(shape=(n_beh,  1), name='beh_input')

    enc_flow = _cnn_encoder(inp_flow, 'flow')
    enc_beh  = _cnn_encoder(inp_beh,  'beh')

    combined = Concatenate(name='combined')([enc_flow, enc_beh])
    attn = Dense(2, activation='softmax', name='modality_attention')(combined)

    attn_flow = Lambda(lambda x: tf.expand_dims(x[:, 0], -1), name='attn_flow')(attn)
    attn_beh  = Lambda(lambda x: tf.expand_dims(x[:, 1], -1), name='attn_beh')(attn)

    fused = Add(name='fused')([
        Multiply(name='w_flow')([enc_flow, attn_flow]),
        Multiply(name='w_beh') ([enc_beh,  attn_beh]),
    ])
    fused = Dense(64, activation='relu', name='final_dense')(fused)
    fused = Dropout(0.3, name='final_drop')(fused)
    out = Dense(n_classes, activation='softmax', name='out')(fused)
    return Model([inp_flow, inp_beh], out, name='attention_fusion')


# ── CCAT adversarial training (dual-stream) ───────────────────────────────────

def _pgd_dual(model, X_flow, X_beh, y_oh, epsilon, alpha, n_steps, loss_fn):
    """
    PGD on concatenated dual-stream input.
    Perturbs both flow and behavioral inputs jointly, then splits.
    """
    X_cat = tf.concat([X_flow, X_beh], axis=1)           # (B, n_flow+n_beh, 1)
    delta = tf.random.uniform(tf.shape(X_cat), -epsilon, epsilon)
    X_adv = tf.clip_by_value(X_cat + delta, 0.0, 1.0)

    n_flow = X_flow.shape[1]

    for _ in range(n_steps):
        with tf.GradientTape() as tape:
            tape.watch(X_adv)
            Xf = X_adv[:, :n_flow, :]
            Xb = X_adv[:, n_flow:, :]
            loss = loss_fn(y_oh, model([Xf, Xb], training=False))
        grad  = tape.gradient(loss, X_adv)
        X_adv = X_adv + alpha * tf.sign(grad)
        X_adv = tf.clip_by_value(X_adv, X_cat - epsilon, X_cat + epsilon)
        X_adv = tf.clip_by_value(X_adv, 0.0, 1.0)
        X_adv = tf.stop_gradient(X_adv)

    return X_adv[:, :n_flow, :], X_adv[:, n_flow:, :]


def train_ccat(model, X_tr_flow, X_tr_beh, y_tr,
               X_val_flow, X_val_beh, y_v, n_classes,
               epsilon=0.1, adv_ratio=0.5, pgd_steps=7,
               epochs=50, patience=10, batch_size=256,
               rare_threshold=500, seed=42):
    """
    Class-Conditional Adversarial Training on dual-stream model.

    Rare classes (n_train < rare_threshold) are trained on clean examples only;
    common classes use PGD-perturbed examples for adv_ratio of each batch.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    opt = tf.keras.optimizers.Adam(1e-3)

    # Identify rare classes
    class_counts = np.bincount(y_tr, minlength=n_classes)
    rare_mask = np.array([class_counts[c] < rare_threshold for c in range(n_classes)])

    # Class weights (inverse frequency)
    total = len(y_tr)
    class_weights = {c: total / (n_classes * max(class_counts[c], 1))
                     for c in range(n_classes)}
    sample_weights = np.array([class_weights[c] for c in y_tr], dtype=np.float32)

    pgd_alpha = epsilon / max(pgd_steps, 1) * 2.5

    # Validation data
    y_v_oh = tf.keras.utils.to_categorical(y_v, n_classes)
    Xvf = X_val_flow.astype(np.float32)
    Xvb = X_val_beh.astype(np.float32)

    best_val_loss = np.inf
    patience_cnt = 0
    best_weights = model.get_weights()

    n = len(y_tr)
    idx = np.arange(n)
    alpha_pgd = pgd_alpha

    print(f"    CCAT: ε={epsilon}, adv_ratio={adv_ratio}, pgd_steps={pgd_steps}, "
          f"rare_classes={int(rare_mask.sum())}/{n_classes}")

    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(idx)
        ep_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            Xf_b = tf.constant(X_tr_flow[batch_idx], dtype=tf.float32)
            Xb_b = tf.constant(X_tr_beh [batch_idx], dtype=tf.float32)
            y_b  = y_tr[batch_idx]
            y_oh = tf.keras.utils.to_categorical(y_b, n_classes)
            sw   = sample_weights[batch_idx]

            # Split into rare and common
            is_rare = rare_mask[y_b]
            adv_eligible = ~is_rare
            n_adv = int(adv_eligible.sum() * adv_ratio)

            Xf_train, Xb_train = Xf_b, Xb_b

            if n_adv > 0:
                adv_idx = np.where(adv_eligible)[0]
                adv_chosen = adv_idx[:n_adv]
                Xf_adv, Xb_adv = _pgd_dual(
                    model,
                    tf.gather(Xf_b, adv_chosen),
                    tf.gather(Xb_b, adv_chosen),
                    tf.gather(tf.constant(y_oh, dtype=tf.float32), adv_chosen),
                    epsilon, alpha_pgd, pgd_steps, loss_fn,
                )
                # Replace chosen samples with adversarial versions
                Xf_np = Xf_b.numpy().copy()
                Xb_np = Xb_b.numpy().copy()
                Xf_np[adv_chosen] = Xf_adv.numpy()
                Xb_np[adv_chosen] = Xb_adv.numpy()
                Xf_train = tf.constant(Xf_np, dtype=tf.float32)
                Xb_train = tf.constant(Xb_np, dtype=tf.float32)

            with tf.GradientTape() as tape:
                preds = model([Xf_train, Xb_train], training=True)
                per_sample = tf.keras.losses.categorical_crossentropy(
                    tf.constant(y_oh, dtype=tf.float32), preds)
                loss = tf.reduce_mean(per_sample * tf.constant(sw, dtype=tf.float32))

            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            ep_loss += float(loss)
            n_batches += 1

        # Validation (batched to avoid GPU OOM on large val sets like CICIDS2017)
        _vloss_sum = 0.0
        _vloss_n   = 0
        for _vi in range(0, len(y_v_oh), batch_size * 4):
            _vpf = tf.constant(Xvf[_vi:_vi+batch_size*4], dtype=tf.float32)
            _vpb = tf.constant(Xvb[_vi:_vi+batch_size*4], dtype=tf.float32)
            _yoh = tf.constant(y_v_oh[_vi:_vi+batch_size*4], dtype=tf.float32)
            _vp  = model([_vpf, _vpb], training=False)
            _vloss_sum += float(tf.reduce_sum(
                tf.keras.losses.categorical_crossentropy(_yoh, _vp)))
            _vloss_n   += len(_yoh)
        val_loss = _vloss_sum / max(_vloss_n, 1)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_weights = model.get_weights()
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"    Early stop at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: train_loss={ep_loss/n_batches:.4f}  "
                  f"val_loss={val_loss:.4f}")

    model.set_weights(best_weights)
    return model


# ── FedAvg federated training (dual-stream) ───────────────────────────────────

def _local_train_dual(model, X_flow, X_beh, y, local_epochs, lr, batch_size, n_classes):
    """
    Local FedAvg training step for dual-stream model.

    BatchNorm running stats (mean/variance) are frozen during local training
    by using training=False in the forward pass. This prevents each client
    from accumulating divergent BN statistics that destabilise aggregation.
    BN affine parameters (gamma/beta) are still gradient-updated normally.
    Dropout is also disabled, which improves federated stability.
    """
    opt = tf.keras.optimizers.Adam(lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    n = len(y)
    idx = np.arange(n)

    for _ in range(local_epochs):
        np.random.shuffle(idx)
        for start in range(0, n, batch_size):
            bi = idx[start:start + batch_size]
            Xf = tf.constant(X_flow[bi], dtype=tf.float32)
            Xb = tf.constant(X_beh [bi], dtype=tf.float32)
            yb = tf.constant(y[bi], dtype=tf.int32)
            with tf.GradientTape() as tape:
                # training=False: BN uses global running stats (not updated
                # per-client) — standard fix for BN in federated learning.
                loss = loss_fn(yb, model([Xf, Xb], training=False))
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

    return model.get_weights(), n


def train_fedavg(model, X_tr_flow, X_tr_beh, y_tr,
                 X_val_flow, X_val_beh, y_v, n_classes,
                 n_clients=4, n_rounds=10,
                 local_epochs=2, lr=1e-3, batch_size=256,
                 seed=42):
    """
    FedAvg-IID: partition training data equally across n_clients,
    aggregate weights by sample count after each round.
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # IID partition
    n = len(y_tr)
    perm = rng.permutation(n)
    splits = np.array_split(perm, n_clients)
    clients = [(X_tr_flow[s], X_tr_beh[s], y_tr[s]) for s in splits]

    print(f"    FedAvg: {n_clients} clients, {n_rounds} rounds, "
          f"local_epochs={local_epochs}")

    global_weights = model.get_weights()
    best_val_f1 = -1.0
    best_weights = global_weights

    for rnd in range(n_rounds):
        client_updates = []
        for Xf_c, Xb_c, y_c in clients:
            model.set_weights(global_weights)
            w, n_c = _local_train_dual(
                model, Xf_c, Xb_c, y_c,
                local_epochs, lr, batch_size, n_classes,
            )
            client_updates.append((w, n_c))

        # Weighted average
        total_samples = sum(nc for _, nc in client_updates)
        new_weights = []
        for layer_idx in range(len(global_weights)):
            layer_avg = sum(
                w[layer_idx] * (nc / total_samples)
                for w, nc in client_updates
            )
            new_weights.append(layer_avg)
        global_weights = new_weights
        model.set_weights(global_weights)

        # Validation F1 (batched to avoid GPU OOM on large val sets like CICIDS2017)
        Xvf = X_val_flow.astype(np.float32)
        Xvb = X_val_beh.astype(np.float32)
        val_pred = np.argmax(_predict_batched_dual(model, Xvf, Xvb, batch_size=1024), axis=1)
        val_f1 = f1_score(y_v, val_pred, average='macro', zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = global_weights

        if (rnd + 1) % 5 == 0:
            print(f"    Round {rnd+1}/{n_rounds}: val_F1={val_f1:.4f}")

    model.set_weights(best_weights)
    return model


# ── Experience Replay continual adaptation (dual-stream) ──────────────────────

class DualStreamReplayBuffer:
    """
    Fixed-size ring buffer with reservoir sampling for dual-stream inputs.
    Stores (X_flow, X_beh, y) tuples.
    """

    def __init__(self, size=2000, seed=42):
        self.size = size
        self._rng = np.random.default_rng(seed)
        self._Xf = None
        self._Xb = None
        self._y  = None
        self._n  = 0      # items currently stored
        self._ptr = 0     # next write position

    def update(self, X_flow, X_beh, y):
        """Add new samples via reservoir sampling."""
        n_new = len(y)
        if self._Xf is None:
            cap = min(self.size, n_new)
            idx = self._rng.choice(n_new, cap, replace=False)
            self._Xf  = X_flow[idx].copy()
            self._Xb  = X_beh [idx].copy()
            self._y   = y[idx].copy()
            self._n   = cap
            self._ptr = cap % self.size
        else:
            for i in range(n_new):
                if self._n < self.size:
                    # Buffer not full yet — append
                    if self._n == len(self._y):
                        # Extend arrays
                        self._Xf = np.concatenate([self._Xf, X_flow[i:i+1]], axis=0)
                        self._Xb = np.concatenate([self._Xb, X_beh [i:i+1]], axis=0)
                        self._y  = np.concatenate([self._y,  y[i:i+1]],       axis=0)
                    else:
                        self._Xf[self._ptr] = X_flow[i]
                        self._Xb[self._ptr] = X_beh [i]
                        self._y [self._ptr] = y[i]
                    self._n  += 1
                    self._ptr = self._n % self.size
                else:
                    # Reservoir replace
                    j = int(self._rng.integers(0, self._n + i + 1))
                    if j < self.size:
                        self._Xf[j] = X_flow[i]
                        self._Xb[j] = X_beh [i]
                        self._y [j] = y[i]

    def sample(self, n):
        """Draw n random samples from the buffer."""
        if self._n == 0:
            return None, None, None
        idx = self._rng.choice(self._n, min(n, self._n), replace=False)
        return self._Xf[idx], self._Xb[idx], self._y[idx]

    @property
    def filled(self):
        return self._n > 0


def adapt_replay(model, buffer, X_new_flow, X_new_beh, y_new, n_classes,
                 replay_ratio=0.3, epochs=10, lr=5e-4, batch_size=256,
                 patience=5, seed=42):
    """
    Continual adaptation: train on new task data mixed with replay samples.
    Updates the buffer with new task data after training.
    """
    np.random.seed(seed)
    opt = tf.keras.optimizers.Adam(lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    n = len(y_new)
    n_replay_per_batch = max(1, int(batch_size * replay_ratio))
    n_new_per_batch = batch_size - n_replay_per_batch

    best_loss = np.inf
    patience_cnt = 0
    best_weights = model.get_weights()

    for epoch in range(epochs):
        idx = np.random.permutation(n)
        ep_loss = 0.0
        n_batches = 0

        for start in range(0, n, n_new_per_batch):
            bi = idx[start:start + n_new_per_batch]
            Xf_new = tf.constant(X_new_flow[bi], dtype=tf.float32)
            Xb_new = tf.constant(X_new_beh [bi], dtype=tf.float32)
            y_new_ = tf.constant(y_new[bi], dtype=tf.int32)

            if buffer.filled:
                Xf_r, Xb_r, y_r = buffer.sample(n_replay_per_batch)
                Xf_b = tf.concat([Xf_new, tf.constant(Xf_r, dtype=tf.float32)], axis=0)
                Xb_b = tf.concat([Xb_new, tf.constant(Xb_r, dtype=tf.float32)], axis=0)
                y_b  = tf.concat([y_new_, tf.constant(y_r,  dtype=tf.int32)],   axis=0)
            else:
                Xf_b, Xb_b, y_b = Xf_new, Xb_new, y_new_

            with tf.GradientTape() as tape:
                loss = loss_fn(y_b, model([Xf_b, Xb_b], training=True))
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            ep_loss += float(loss)
            n_batches += 1

        avg_loss = ep_loss / max(n_batches, 1)
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            best_weights = model.get_weights()
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    model.set_weights(best_weights)
    buffer.update(X_new_flow, X_new_beh, y_new)
    return model, buffer


# ── Evaluation ────────────────────────────────────────────────────────────────

def _predict_batched_dual(model, Xf, Xb, batch_size=2048):
    """Batched prediction for dual-stream to avoid GPU OOM on large test sets."""
    n = Xf.shape[0]
    chunks = []
    for i in range(0, n, batch_size):
        chunks.append(model([Xf[i:i+batch_size], Xb[i:i+batch_size]], training=False).numpy())
    return np.concatenate(chunks, axis=0)


def evaluate_model(model, X_te_flow, X_te_beh, y_te, n_classes,
                   n_latency_runs=200, batch_size=1024):
    """Standard evaluation: F1, precision, recall, accuracy, FPR, latency."""
    Xf = X_te_flow.astype(np.float32)
    Xb = X_te_beh.astype(np.float32)

    probs = _predict_batched_dual(model, Xf, Xb, batch_size=batch_size)
    y_pred = np.argmax(probs, axis=1)

    f1   = f1_score(y_te, y_pred, average='macro', zero_division=0)
    prec = precision_score(y_te, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_te, y_pred, average='macro', zero_division=0)
    acc  = accuracy_score(y_te, y_pred)

    # FPR: mean over classes (FP / (FP + TN))
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_te, y_pred, labels=list(range(n_classes)))
    fprs = []
    for c in range(n_classes):
        fp = cm[:, c].sum() - cm[c, c]
        tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        fprs.append(fp / max(fp + tn, 1))
    fpr = float(np.mean(fprs))

    # Latency: single-sample inference (CPU)
    sample_f = Xf[:1]
    sample_b = Xb[:1]
    with tf.device('/CPU:0'):
        _ = model([sample_f, sample_b], training=False)  # warm-up
        t0 = time.perf_counter()
        for _ in range(n_latency_runs):
            model([sample_f, sample_b], training=False)
        latency_ms = (time.perf_counter() - t0) / n_latency_runs * 1000

    return {
        'f1':          round(float(f1),   4),
        'precision':   round(float(prec), 4),
        'recall':      round(float(rec),  4),
        'accuracy':    round(float(acc),  4),
        'fpr':         round(float(fpr),  4),
        'latency_ms':  round(float(latency_ms), 3),
        'n_params':    int(model.count_params()),
    }


def evaluate_under_pgd(model, X_te_flow, X_te_beh, y_te, n_classes,
                        epsilon=0.1, pgd_steps=20, n_eval=1000, seed=42):
    """
    Evaluate model under PGD adversarial attack.
    Subsample n_eval test examples for speed.
    Returns attack success rate (ASR) and adversarial F1.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_te), min(n_eval, len(y_te)), replace=False)
    Xf = tf.constant(X_te_flow[idx], dtype=tf.float32)
    Xb = tf.constant(X_te_beh [idx], dtype=tf.float32)
    y  = y_te[idx]

    alpha = epsilon / pgd_steps * 2.5
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    y_oh = tf.keras.utils.to_categorical(y, n_classes)

    Xf_adv, Xb_adv = _pgd_dual(model, Xf, Xb,
                                 tf.constant(y_oh, dtype=tf.float32),
                                 epsilon, alpha, pgd_steps, loss_fn)

    y_clean = np.argmax(model([Xf, Xb], training=False).numpy(), axis=1)
    y_adv   = np.argmax(model([Xf_adv, Xb_adv], training=False).numpy(), axis=1)

    # Only count samples correctly classified on clean as "attackable"
    correct_clean = (y_clean == y)
    asr = float((y_adv[correct_clean] != y[correct_clean]).mean()) if correct_clean.any() else 0.0
    adv_f1 = f1_score(y, y_adv, average='macro', zero_division=0)

    return {
        'asr':    round(asr,          4),
        'adv_f1': round(float(adv_f1), 4),
        'epsilon': epsilon,
        'pgd_steps': pgd_steps,
        'n_eval': int(correct_clean.sum()),
    }


# ── SHAP explainability ───────────────────────────────────────────────────────

def compute_shap(model, X_bg_flow, X_bg_beh, X_te_flow, X_te_beh,
                 feature_names_flow, feature_names_beh,
                 n_explain=100, n_background=50, seed=42):
    """
    SHAP KernelExplainer on concatenated feature space.
    Returns mean |SHAP| per feature for the test subset.
    """
    try:
        import shap
    except ImportError:
        print("    SHAP not available — skipping explainability")
        return None

    rng = np.random.default_rng(seed)

    # Background: subsample
    bg_idx = rng.choice(len(X_bg_flow), min(n_background, len(X_bg_flow)), replace=False)
    bg = np.concatenate([X_bg_flow[bg_idx, :, 0],
                         X_bg_beh [bg_idx, :, 0]], axis=1)  # (n_bg, n_flow+n_beh)

    # Test: subsample
    te_idx = rng.choice(len(X_te_flow), min(n_explain, len(X_te_flow)), replace=False)
    te = np.concatenate([X_te_flow[te_idx, :, 0],
                         X_te_beh [te_idx, :, 0]], axis=1)  # (n_te, n_flow+n_beh)

    n_flow = X_bg_flow.shape[1]

    def predict_fn(X_concat):
        Xf = X_concat[:, :n_flow, np.newaxis].astype(np.float32)
        Xb = X_concat[:, n_flow:, np.newaxis].astype(np.float32)
        return model([Xf, Xb], training=False).numpy()

    explainer = shap.KernelExplainer(predict_fn, bg)
    shap_vals = explainer.shap_values(te, nsamples=100, silent=True)

    # Mean |SHAP| across classes and samples
    all_names = feature_names_flow + feature_names_beh
    mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)

    top10_idx = np.argsort(mean_abs)[::-1][:10]
    top10 = {all_names[i]: round(float(mean_abs[i]), 5) for i in top10_idx}

    return {
        'top10_features': top10,
        'mean_abs_shap':  [round(float(v), 5) for v in mean_abs],
        'feature_names':  all_names,
    }


# ── Centralized training ──────────────────────────────────────────────────────

def train_centralized(model, X_tr_flow, X_tr_beh, y_tr,
                      X_val_flow, X_val_beh, y_v, n_classes,
                      epochs=80, batch_size=256, patience=10, seed=42):
    """Standard centralized training with Adam + early stopping."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                          min_lr=1e-5, verbose=0),
    ]
    model.fit(
        [X_tr_flow, X_tr_beh], y_tr,
        validation_data=([X_val_flow, X_val_beh], y_v),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=0,
    )
    return model


def train_baseline_cnn(model, X_tr, y_tr, X_val, y_v, n_classes,
                        epochs=80, batch_size=256, patience=10, seed=42):
    """Centralized training for single-stream baseline CNN."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                          min_lr=1e-5, verbose=0),
    ]
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_v),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=0,
    )
    return model


def evaluate_baseline_cnn(model, X_te, y_te, n_classes,
                            n_latency_runs=200, batch_size=2048):
    """Evaluate single-stream CNN."""
    X = X_te.astype(np.float32)
    n = X.shape[0]
    chunks = [model(X[i:i+batch_size], training=False).numpy() for i in range(0, n, batch_size)]
    probs = np.concatenate(chunks, axis=0)
    y_pred = np.argmax(probs, axis=1)

    f1   = f1_score(y_te, y_pred, average='macro', zero_division=0)
    prec = precision_score(y_te, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_te, y_pred, average='macro', zero_division=0)
    acc  = accuracy_score(y_te, y_pred)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_te, y_pred, labels=list(range(n_classes)))
    fprs = []
    for c in range(n_classes):
        fp = cm[:, c].sum() - cm[c, c]
        tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        fprs.append(fp / max(fp + tn, 1))
    fpr = float(np.mean(fprs))

    sample = X[:1]
    with tf.device('/CPU:0'):
        _ = model(sample, training=False)
        t0 = time.perf_counter()
        for _ in range(n_latency_runs):
            model(sample, training=False)
        latency_ms = (time.perf_counter() - t0) / n_latency_runs * 1000

    return {
        'f1':         round(float(f1),   4),
        'precision':  round(float(prec), 4),
        'recall':     round(float(rec),  4),
        'accuracy':   round(float(acc),  4),
        'fpr':        round(float(fpr),  4),
        'latency_ms': round(float(latency_ms), 3),
        'n_params':   int(model.count_params()),
    }
