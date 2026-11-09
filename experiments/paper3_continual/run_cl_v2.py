"""Paper 3 v2: class-incremental learning of new attack families for a compact IoT intrusion detector.

Replaces experiments/continual_learning (v1: preprocessing fitted on the full training split, which
contains future-task classes; benign traffic only in Task 0; no upper bound; one seed; unseeded,
non-deterministic training without best-weight restore; Fisher computed from softmax outputs
passed through log_softmax and from batch-mean gradients).

Protocol
  * TON-IoT and UNSW-NB15 through the corrected loader (v2.1: dedup after column drops, cross-split
    guard); N-BaIoT as prepared for Paper 4. Three tasks of new attack classes; benign training and
    validation records are split into three disjoint shares, one per task, so every task has benign
    traffic. The test set is never split.
  * Feature selection (MI + RFECV) and MinMax scaling are fitted on the Task-0 training rows only
    (N-BaIoT: MinMax refitted on Task 0; it is affine in the stored [0, 1] values). Later-task values
    are clipped to [-1, 2]. 'oracle' fits the same pipeline on all training rows (ablation only).
  * Single head over all classes; the loss and the prediction are restricted to the classes seen so
    far. Class-weighted cross-entropy on each training pool. Early stopping on the validation data
    available at that stage (current task; buffer-free), best weights restored.
  * Strategies: naive fine-tuning, joint retraining on all data so far (upper bound), online EWC
    (per-sample empirical Fisher), LwF, experience replay with a class-balanced buffer (500, 2,000),
    A-GEM (buffer 2,000). Tuning on a separate development seed.
  * BatchNorm momentum 0.9 (Keras default 0.99 leaves the running statistics far from the data on
    these small-valued features for hundreds of steps, so eval-mode predictions collapse to one class
    and early stopping fires before they recover; a likely contributor to the v1 Task-0 anomaly).
  * Deterministic ops + fixed seed: all strategies share the same Task-0 model (checked).
  * After every task the full test set is predicted (argmax over seen classes) and the confusion
    matrix stored, so all metrics, including detection of not-yet-learned attacks, are derived later.

  * v3 (2026-11-01): after MinMax scaling every feature is compressed by the fixed, monotone map
    x -> asinh(1000 x) / asinh(1000) (0 and 1 unchanged, no fitted parameters, extrapolates smoothly for
    later-task values). The MinMax-scaled flow counts are heavy-tailed (median ~0); without the map the
    batch-normalised CNN failed silently on TON-IoT (joint retraining macro-F1 0.77 against 0.965 for an RF
    on the same features and data; results_v2, superseded). Same failure as Paper 6 v2.
  * final review round (2026-11-07), results in results_p3final/<dataset>/<group>/ (results_v3 untouched):
      - 'agem2' (group agemfix: semantic order x 5 seeds, then the 4 random orders): A-GEM with a corrected
        reference loss. In 'agem' the reference loss on the memory reused the current task's class weights, which
        are 0 for every earlier attack class, so the projection protected only the benign memory records. 'agem2'
        weights the memory loss with class weights computed over the memory's own labels (every stored class counts
        equally) and evaluates it in inference mode (training=False), so the memory batch neither updates the
        batch-normalisation statistics nor uses dropout.
      - 'er2000p' (group replayabl, semantic order x 5 seeds): replay-weighting ablation. ER-2000 draws 512 memory
        records (with replacement) for every 512 current records AND weights the loss with class weights computed
        over task + memory, so earlier classes are up-weighted twice. 'er2000p' adds the same class-balanced memory
        once to the training pool (no 1:1 mixing, no resampling); the class weights over pool + memory are unchanged,
        so every seen class gets equal total weight per epoch. Same memory contents as ER-2000 for a given seed.
      - group rfmem (semantic order x 5 seeds): random forest retrained after every task on the ER-2000 memory plus
        the current task ('rf_er2000'; identical memory contents to the CNN ER-2000 run of the same seed), and the
        all-data random forest after every task ('rf_joint'; its last stage equals the 'rf' reference).
      - group rangeshare (no training): share of values outside the Task-0 min-max range [0, 1] before the
        compression map, per split and task, and the overlap of the Task-0 and all-data feature selections.
Results: results_v3/<dataset>/<group>/<strategy>_o<order>_s<seed>.json (resumable).
Run from `Papers pipeline/`:
  python -m experiments.paper3_continual.run_cl_v2 --dataset TON_IoT --group tune|main|orders|oracle|rf [--smoke]
  python -m experiments.paper3_continual.run_cl_v2 --dataset TON_IoT --group agemfix|replayabl|rfmem|rangeshare
"""
import argparse, contextlib, io, json, os, sys, time
from pathlib import Path

import numpy as np

# CPU only: op determinism has no deterministic GPU kernel for the sparse cross-entropy gradient
# v3: GPU allowed; dense (one-hot) cross-entropy and one-hot log-likelihood have deterministic GPU kernels,
# unlike the gradients of sparse_softmax_cross_entropy and tf.gather used in v2. Set P3_CPU=1 to force the CPU.
if os.environ.get('P3_CPU') == '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_v3'
OUT_FINAL = HERE / 'results_p3final'   # final review round (agemfix, replayabl, rfmem, rangeshare)
FINAL_GROUPS = ('agemfix', 'replayabl', 'rfmem', 'rangeshare')
CACHE = HERE / 'cache_v2'
DATASETS = ('TON_IoT', 'UNSW_NB15', 'N_BaIoT')

# semantic task order (v1 grouping of attack families); random orders keep the task sizes
SEMANTIC = {
    'TON_IoT': [['scanning', 'dos'], ['ddos', 'password', 'mitm'], ['backdoor', 'ransomware', 'injection', 'xss']],
    'UNSW_NB15': [['generic', 'exploits'], ['fuzzers', 'reconnaissance', 'dos'], ['analysis', 'backdoor', 'shellcode', 'worms']],
    'N_BaIoT': [['gafgyt_combo', 'gafgyt_junk', 'gafgyt_scan'], ['mirai_scan', 'mirai_ack', 'mirai_syn'], ['mirai_udp', 'mirai_udpplain']],
}
N_RANDOM_ORDERS = 4
SEEDS = (0, 1, 2, 3, 4)
DEV_SEED = 100
STRATEGIES = ('naive', 'joint', 'ewc', 'lwf', 'er500', 'er2000', 'agem')
GRID = {'ewc': [('lam', v) for v in (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7)],
        'lwf': [('alpha', v) for v in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)]}
# (2026-11-02: LwF grid extended below 0.5 on every dataset because N-BaIoT selected the old lower edge)
# v3 budget: v2 (40/5 and 20/5) stopped the CNN long before convergence (TON joint 0.885 at 40/5 vs 0.925 at 80/15)
CFG0 = dict(epochs=100, patience=15, lr=1e-3)      # Task 0 and joint retraining
CFGI = dict(epochs=50, patience=10, lr=5e-4)       # incremental tasks
BATCH, CLIP, FISHER_N, LWF_T, WMAX = 512, (-1.0, 2.0), 1000, 2.0, 50.0
ASINH_C = 1000.0     # input compression x -> asinh(c x) / asinh(c), applied after the Task-0 MinMax scaling
USE_ASINH = True     # False only in the 'factorial' group (reconstruction of the v2 failure)
BUDGETS = {'v3': (CFG0, CFGI), 'v2': (dict(epochs=40, patience=5, lr=1e-3), dict(epochs=20, patience=5, lr=5e-4))}
BENIGN_SHARE = 0.5  # 'er2000b': share of the replay memory reserved for benign records
MIN_EPOCHS = 5       # early stopping counts only after this (BN running statistics must settle first)


# ── data ──────────────────────────────────────────────────────────────────────

def load(ds):
    if ds == 'N_BaIoT':
        z = np.load(ROOT / 'dataset4' / 'nbaiot_prepared.npz')
        info = json.loads((ROOT / 'dataset4' / 'nbaiot_prepared.json').read_text())
        return dict(ds=ds, raw=False, feat=list(info['features']), classes=list(info['classes']),
                    Xtr=z['Xtr'], ytr=z['ytr'].astype(int), Xva=z['Xva'], yva=z['yva'].astype(int),
                    Xte=z['Xte'], yte=z['yte'].astype(int))
    from experiments.baseline.config import DATASETS as BASE
    from src.data.loader import load_dataset
    csv, label = BASE[ds]
    with contextlib.redirect_stdout(io.StringIO()):
        r = load_dataset(csv, label_col=label, seed=42)
    return dict(ds=ds, raw=True, feat=list(r['feature_names']), classes=[c.lower() for c in r['class_names']],
                Xtr=r['X_train'], ytr=np.asarray(r['y_train'], int), Xva=r['X_val'], yva=np.asarray(r['y_val'], int),
                Xte=r['X_test'], yte=np.asarray(r['y_test'], int))


def task_order(D, order):
    """order 0 = semantic; order k >= 1 = random permutation of the attack classes, same task sizes."""
    sem = SEMANTIC[D['ds']]
    if order == 0:
        names = sem
    else:
        flat = [c for t in sem for c in t]
        perm = list(np.random.default_rng(500 + order).permutation(flat))
        names, i = [], 0
        for t in sem:
            names.append(sorted(perm[i:i + len(t)])); i += len(t)
    idx = {c: k for k, c in enumerate(D['classes'])}
    missing = {c for t in names for c in t} - set(idx)
    assert not missing, f'unknown classes {missing}'
    return [[idx[c] for c in t] for t in names]


def build_tasks(D, tasks, order):
    """Training/validation row indices per task; benign split into len(tasks) disjoint shares
    (fixed per order, independent of the training seed)."""
    nrm = D['classes'].index('normal')
    rng = np.random.default_rng(700 + order)
    out = []
    shares = {}
    for split in ('tr', 'va'):
        b = rng.permutation(np.where(D['y' + split] == nrm)[0])
        shares[split] = np.array_split(b, len(tasks))
    for t, cls in enumerate(tasks):
        T = {}
        for split in ('tr', 'va'):
            att = np.where(np.isin(D['y' + split], cls))[0]
            T[split] = np.sort(np.concatenate([shares[split][t], att]))
        T['classes'] = [nrm] + list(cls)
        out.append(T)
    return out


def preprocess(D, T, order, oracle=False, raw_minmax=False):
    """Fit feature selection + scaling on Task-0 training rows (or all rows for the oracle ablation);
    cached per dataset and order. Returns scaled float32 splits and the selected feature names.
    raw_minmax=True (diagnostics only): min-max values without the compression map and without clipping."""
    import joblib
    CACHE.mkdir(parents=True, exist_ok=True)
    key = CACHE / f"{D['ds']}_{'oracle' if oracle else f'o{order}'}.pkl"
    fit_idx = np.arange(len(D['ytr'])) if oracle else T[0]['tr']
    if D['raw']:
        if key.exists():
            pipe = joblib.load(key)
        else:
            from src.data.preprocessing import PreprocessingPipeline
            pipe = PreprocessingPipeline(mi_threshold=0.05, rfecv_subsample=50_000, seed=42)
            with contextlib.redirect_stdout(io.StringIO()):
                pipe.fit(D['Xtr'][fit_idx], D['ytr'][fit_idx], D['feat'])
            joblib.dump(pipe, key)
        tf_ = lambda X: pipe.transform(X)[0]
        feat = list(pipe.selected_features_)
    else:
        lo, hi = D['Xtr'][fit_idx].min(0), D['Xtr'][fit_idx].max(0)
        rng_ = np.where(hi > lo, hi - lo, 1.0)
        tf_ = lambda X: (X - lo) / rng_
        feat = D['feat']
    S = {}
    for split in ('tr', 'va', 'te'):
        X = tf_(D['X' + split]).astype(np.float32)
        if raw_minmax:
            S[split] = X
            continue
        if USE_ASINH:
            X = (np.arcsinh(ASINH_C * X) / np.arcsinh(ASINH_C)).astype(np.float32)
        S['clip_share_' + split] = float(((X < CLIP[0]) | (X > CLIP[1])).mean())
        S[split] = np.clip(X, *CLIP)
    S['feat'] = feat
    return S


# ── model and training ────────────────────────────────────────────────────────

def build_model(n_feat, n_cls):
    """Paper 1 CNN (Conv32-BN-pool, Conv64-BN-pool, GAP, Dense128, Dense64), logits output."""
    from tensorflow.keras import layers as L, Model
    inp = L.Input((n_feat, 1))
    x = L.MaxPooling1D(2, padding='same')(L.BatchNormalization(momentum=0.9)(L.Conv1D(32, 3, activation='relu', padding='same')(inp)))
    x = L.MaxPooling1D(2, padding='same')(L.BatchNormalization(momentum=0.9)(L.Conv1D(64, 3, activation='relu', padding='same')(x)))
    x = L.GlobalAveragePooling1D()(x)
    x = L.Dropout(0.3)(L.Dense(128, activation='relu')(x))
    x = L.Dropout(0.3)(L.Dense(64, activation='relu')(x))
    return Model(inp, L.Dense(n_cls)(x))


def class_weights(y, n_cls):
    cnt = np.bincount(y, minlength=n_cls).astype(float)
    w = np.zeros(n_cls)
    m = cnt > 0
    w[m] = cnt[m].sum() / (m.sum() * cnt[m])
    return np.minimum(w, WMAX).astype(np.float32)


class Buffer:
    """Class-balanced episodic memory: equal slots per seen class, random records within a class."""

    def __init__(self, size, rng, benign=None, benign_share=None):
        self.size, self.rng, self.X, self.y = size, rng, None, None
        self.benign, self.benign_share = benign, benign_share

    def add(self, X, y):
        X = X if self.X is None else np.vstack([self.X, X])
        y = y if self.y is None else np.concatenate([self.y, y])
        cls = np.unique(y)
        if self.benign_share and self.benign in cls and len(cls) > 1:
            # benign gets a fixed share of the memory, the attack classes split the rest equally
            nb = int(self.size * self.benign_share)
            per = {c: (nb if c == self.benign else (self.size - nb) // (len(cls) - 1)) for c in cls}
        else:
            per = {c: self.size // len(cls) for c in cls}
        keep = np.concatenate([self.rng.permutation(np.where(y == c)[0])[:per[c]] for c in cls])
        self.X, self.y = X[keep], y[keep]


def predict_logits(model, X, bs=8192):
    return np.vstack([model(X[i:i + bs, :, None], training=False).numpy() for i in range(0, len(X), bs)])


def masked_pred(logits, seen):
    m = np.full(logits.shape[1], -np.inf); m[seen] = 0
    return np.argmax(logits + m, 1)


def run(D, S, T, strategy, seed, hp, smoke=False, cache_key=None):
    import tensorflow as tf
    from sklearn.metrics import f1_score, confusion_matrix
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    n_cls, n_feat = len(D['classes']), S['tr'].shape[1]
    rng = np.random.default_rng(seed)

    def fresh():
        tf.keras.utils.set_random_seed(seed)
        return build_model(n_feat, n_cls)

    model = fresh()
    is_agem = strategy in ('agem', 'agem2')
    pooled = strategy.startswith('er') and strategy.endswith('p')   # 'er2000p': memory added once to the pool
    bsize = int(''.join(ch for ch in strategy[2:] if ch.isdigit())) if strategy.startswith('er') else 2000
    bnb = strategy.startswith('er') and strategy.endswith('b')
    buf = Buffer(bsize, np.random.default_rng(seed + 1), benign=D['classes'].index('normal') if bnb else None,
                 benign_share=BENIGN_SHARE if bnb else None) if strategy.startswith('er') or is_agem else None
    ewc_F, ewc_star, teacher, old_seen = None, None, None, []
    stages, seen = [], []
    for t, task in enumerate(T):
        seen = sorted(set(seen) | set(task['classes']))
        if strategy == 'joint':
            tr = np.concatenate([T[k]['tr'] for k in range(t + 1)]); va = np.concatenate([T[k]['va'] for k in range(t + 1)])
            model = fresh(); cfg = CFG0
        else:
            tr, va = task['tr'], task['va']; cfg = CFG0 if t == 0 else CFGI
        if smoke:
            cfg = dict(cfg, epochs=10)
        Xt, yt = S['tr'][tr], D['ytr'][tr]
        Xv, yv = S['va'][va], D['yva'][va]
        pool_y = yt if buf is None or buf.y is None or is_agem else np.concatenate([yt, buf.y])
        w = tf.constant(class_weights(pool_y, n_cls))
        # 'agem2': the reference loss on the memory is weighted over the memory's own labels (each stored class equally)
        w_ref = tf.constant(class_weights(buf.y, n_cls)) if strategy == 'agem2' and buf.y is not None else w
        # training pool: the current task, plus (er...p only) the memory, each memory record once per epoch
        if pooled and buf.y is not None and t > 0:
            Xfit, yfit = np.vstack([Xt, buf.X]), np.concatenate([yt, buf.y])
        else:
            Xfit, yfit = Xt, yt
        mask = np.full(n_cls, -1e9, np.float32); mask[seen] = 0; mask = tf.constant(mask)
        old_mask = np.zeros(n_cls, np.float32); old_mask[old_seen] = 1; old_mask = tf.constant(old_mask)
        opt = tf.keras.optimizers.Adam(cfg['lr'])
        use_mem = buf is not None and buf.y is not None and t > 0 and not pooled
        use_ewc = strategy == 'ewc' and ewc_F is not None
        use_lwf = strategy == 'lwf' and teacher is not None
        lam, alpha = hp.get('lam', 0.0), hp.get('alpha', 0.0)
        Fv = [tf.constant(f) for f in ewc_F] if use_ewc else None
        Sv = [tf.constant(s) for s in ewc_star] if use_ewc else None

        def ce(x, y, training=True):
            lg = model(x, training=training) + mask
            l = tf.nn.softmax_cross_entropy_with_logits(tf.one_hot(y, n_cls), lg)
            return tf.reduce_sum(l * tf.gather(w, y)) / tf.reduce_sum(tf.gather(w, y)), lg

        def ce_ref(x, y):
            """'agem2' reference loss: memory-label class weights, inference mode (no BN-statistic update, no dropout)."""
            lg = model(x, training=False) + mask
            l = tf.nn.softmax_cross_entropy_with_logits(tf.one_hot(y, n_cls), lg)
            return tf.reduce_sum(l * tf.gather(w_ref, y)) / tf.reduce_sum(tf.gather(w_ref, y))

        @tf.function(reduce_retracing=True)
        def step(x, y, xm, ym):
            with tf.GradientTape(persistent=True) as tape:
                if use_mem and strategy.startswith('er'):
                    loss, _ = ce(tf.concat([x, xm], 0), tf.concat([y, ym], 0))
                else:
                    loss, lg = ce(x, y)
                if use_ewc:
                    loss += lam / 2 * tf.add_n([tf.reduce_sum(f * tf.square(v - s)) for f, v, s in zip(Fv, model.trainable_variables, Sv)])
                if use_lwf:
                    om = (1 - old_mask) * -1e9
                    pt = tf.nn.softmax((teacher(x, training=False) + om) / LWF_T)
                    ls = tf.nn.log_softmax((lg + om) / LWF_T)
                    loss += alpha * LWF_T ** 2 * tf.reduce_mean(-tf.reduce_sum(pt * ls * old_mask, 1))
                if use_mem and strategy == 'agem':
                    lref, _ = ce(xm, ym)   # flawed (kept for the record): current-task weights, 0 for old attacks
                if use_mem and strategy == 'agem2':
                    lref = ce_ref(xm, ym)
            g = tape.gradient(loss, model.trainable_variables)
            if use_mem and is_agem:
                gr = tape.gradient(lref, model.trainable_variables)
                dot = tf.add_n([tf.reduce_sum(a * b) for a, b in zip(g, gr)])
                nn = tf.add_n([tf.reduce_sum(b * b) for b in gr]) + 1e-12
                g = tf.cond(dot < 0, lambda: [a - dot / nn * b for a, b in zip(g, gr)], lambda: list(g))
            del tape
            opt.apply_gradients(zip(g, model.trainable_variables))
            return loss

        best, best_w, bad, ep_run, t0 = -1.0, None, 0, 0, time.time()
        nb = max(len(yfit) // BATCH, 1)
        # Task 0 is identical for every strategy (same seed, data, model and budget): train it once per
        # (dataset, order, seed, preprocessing) and reuse the stored weights (exact, deterministic ops).
        t0_file = CACHE / 'task0' / f'{cache_key}.pkl' if (t == 0 and cache_key) else None
        cached = t0_file is not None and t0_file.exists()
        for ep in range(0 if cached else cfg['epochs']):
            perm = rng.permutation(len(yfit))
            for b in range(nb):
                ii = perm[b * BATCH:(b + 1) * BATCH]
                if use_mem:
                    jj = rng.integers(0, len(buf.y), len(ii))
                    xm, ym = buf.X[jj][:, :, None], buf.y[jj]
                else:
                    xm, ym = Xfit[ii[:1]][:, :, None], yfit[ii[:1]]
                step(tf.constant(Xfit[ii][:, :, None]), tf.constant(yfit[ii]), tf.constant(xm), tf.constant(ym))
            ep_run += 1
            f1v = f1_score(yv, masked_pred(predict_logits(model, Xv), seen), average='macro', labels=sorted(set(yv)))
            if f1v > best + 1e-4:
                best, best_w, bad = f1v, [v.numpy() for v in model.variables], 0
            else:
                bad += 1
                if bad >= cfg['patience'] and ep + 1 >= MIN_EPOCHS:
                    break
        if cached:
            import pickle
            c = pickle.loads(t0_file.read_bytes())
            best, best_w, ep_run, c_train_s = c['best'], c['w'], c['epochs'], c['train_s']
        elif t0_file is not None:
            import pickle
            t0_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = t0_file.with_suffix('.tmp')
            tmp.write_bytes(pickle.dumps(dict(best=best, w=best_w, epochs=ep_run, train_s=time.time() - t0)))
            tmp.replace(t0_file)
        for v, val in zip(model.variables, best_w):
            v.assign(val)
        train_s = c_train_s if cached else time.time() - t0
        if t == 0:   # re-seed so that later tasks do not depend on whether Task 0 was trained or loaded
            rng = np.random.default_rng(seed + 1000)
            tf.keras.utils.set_random_seed(seed + 1000)

        # strategy state after the task
        if strategy == 'ewc':
            n = min(FISHER_N, len(yt)); ii = rng.choice(len(yt), n, replace=False)
            F = [np.zeros(v.shape, np.float32) for v in model.trainable_variables]

            @tf.function
            def sgrad(x, y):
                with tf.GradientTape() as tp:
                    lp = tf.nn.log_softmax(model(x, training=False) + mask)
                    ll = tf.reduce_sum(lp[0] * tf.one_hot(y[0], n_cls))
                return tp.gradient(ll, model.trainable_variables)
            for i in ii:
                for k, gk in enumerate(sgrad(tf.constant(Xt[i:i + 1, :, None]), tf.constant(yt[i:i + 1]))):
                    F[k] += gk.numpy() ** 2 / n
            ewc_F = F if ewc_F is None else [a + b for a, b in zip(ewc_F, F)]
            ewc_star = [v.numpy() for v in model.trainable_variables]
        if strategy == 'lwf':
            teacher = tf.keras.models.clone_model(model); teacher.set_weights(model.get_weights())
        if buf is not None:
            buf.add(Xt, yt)
        old_seen = list(seen)

        lg = predict_logits(model, S['te'])
        pred = masked_pred(lg, seen)
        cm = confusion_matrix(D['yte'], pred, labels=range(n_cls))
        seen_te = np.isin(D['yte'], seen)
        # validation F1 over ALL classes seen so far (validation data of every task up to t); used only for
        # hyperparameter selection on the development seed, never for checkpoint selection
        va_all = np.concatenate([T[k]['va'] for k in range(t + 1)])
        val_seen_f1 = float(f1_score(D['yva'][va_all], masked_pred(predict_logits(model, S['va'][va_all]), seen),
                                     average='macro', labels=seen))
        stages.append(dict(task=t, seen=[int(c) for c in seen], epochs=ep_run, train_s=round(train_s, 2), n_train=int(len(yt)),
                           **({'n_fit': int(len(yfit))} if pooled else {}),
                           best_val_f1=float(best), val_seen_f1=val_seen_f1, cm=cm.tolist(),
                           macro_f1_seen=float(f1_score(D['yte'][seen_te], pred[seen_te], average='macro', labels=seen)),
                           buffer=int(len(buf.y)) if buf is not None else 0))
    return dict(stages=stages, hp=hp, params=int(model.count_params()))


def rf_reference(D, S, T, seed):
    """Random forest retrained on all task data (final stage), as in Paper 5."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    tr = np.concatenate([t['tr'] for t in T])
    t0 = time.time()
    rf = RandomForestClassifier(200, max_depth=20, min_samples_leaf=2, n_jobs=4, random_state=seed)
    rf.fit(S['tr'][tr], D['ytr'][tr])
    return dict(cm=confusion_matrix(D['yte'], rf.predict(S['te']), labels=range(len(D['classes']))).tolist(),
                train_s=round(time.time() - t0, 1))


def rf_stagewise(D, S, T, seed, mode, mem_size=2000, n_jobs=-1):
    """Random forest after every task (final review round, referee M4), same hyperparameters as rf_reference.
    mode 'memory': retrained on the current task plus the class-balanced memory; the memory uses the same generator
    (seed + 1) and the same update as the CNN ER-2000 run, so its contents are identical for a given seed.
    mode 'joint': retrained on all task data so far (the last stage equals rf_reference).
    Output has the layout of run() (stages with confusion matrices), so analyze_cl_v2.metrics applies."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, f1_score
    buf = Buffer(mem_size, np.random.default_rng(seed + 1)) if mode == 'memory' else None
    stages, seen = [], []
    for t, task in enumerate(T):
        seen = sorted(set(seen) | set(task['classes']))
        if mode == 'joint':
            tr = np.concatenate([T[k]['tr'] for k in range(t + 1)])
            Xf, yf = S['tr'][tr], D['ytr'][tr]
        else:
            Xt, yt = S['tr'][task['tr']], D['ytr'][task['tr']]
            Xf, yf = (Xt, yt) if buf.y is None else (np.vstack([Xt, buf.X]), np.concatenate([yt, buf.y]))
        t0 = time.time()
        rf = RandomForestClassifier(200, max_depth=20, min_samples_leaf=2, n_jobs=n_jobs, random_state=seed)
        rf.fit(Xf, yf)
        train_s = time.time() - t0
        pred = rf.predict(S['te'])
        if buf is not None:
            buf.add(Xt, yt)
        cm = confusion_matrix(D['yte'], pred, labels=range(len(D['classes'])))
        seen_te = np.isin(D['yte'], seen)
        stages.append(dict(task=t, seen=[int(c) for c in seen], epochs=0, train_s=round(train_s, 2), n_train=int(len(yf)),
                           cm=cm.tolist(), buffer=int(len(buf.y)) if buf is not None else 0,
                           macro_f1_seen=float(f1_score(D['yte'][seen_te], pred[seen_te], average='macro', labels=seen))))
    return dict(stages=stages, hp={}, params=0)


def range_share(D, T, order):
    """No training (final review round, M6/C4): share of min-max-scaled values outside the Task-0 range [0, 1], before
    the compression map and before clipping, per split; for the training and validation splits also per task; and
    the share beyond the effective clip of the compressed inputs. TON/UNSW: also the Task-0 and all-data feature sets."""
    S = preprocess(D, T, order, raw_minmax=True)   # cached Task-0 pipeline; min-max values, no map, no clip
    out = {}
    hi_eff = float(np.sinh(CLIP[1] * np.arcsinh(ASINH_C)) / ASINH_C)   # min-max value mapped to the upper clip (~2,000)
    for split in ('tr', 'va', 'te'):
        X = S[split]
        out[split] = dict(out01=float(((X < 0) | (X > 1)).mean()), below0=float((X < 0).mean()), above1=float((X > 1).mean()),
                          beyond_clip=float(((X < CLIP[0]) | (X > hi_eff)).mean()),
                          rows_any_out01=float(((X < 0) | (X > 1)).any(1).mean()))
        if split in ('tr', 'va'):
            out[split]['per_task'] = [float(((X[t_[split]] < 0) | (X[t_[split]] > 1)).mean()) for t_ in T]
    out['upper_clip_minmax_units'] = hi_eff
    out['n_features'] = len(S['feat'])
    out['features_task0'] = list(map(str, S['feat']))
    if D['raw']:
        So = preprocess(D, T, order, oracle=True)
        f0, fo = set(map(str, S['feat'])), set(map(str, So['feat']))
        out['features_oracle'] = sorted(fo)
        out['n_features_oracle'] = len(fo)
        out['n_features_common'] = len(f0 & fo)
    return out


# ── driver ────────────────────────────────────────────────────────────────────

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj)); tmp.replace(path)


def tuned(ds):
    p = OUT / ds / 'tuned.json'
    return json.loads(p.read_text()) if p.exists() else None


def main():
    global USE_ASINH, CFG0, CFGI
    import tensorflow as tf
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=DATASETS, required=True)
    ap.add_argument('--group', choices=('tune', 'main', 'orders', 'oracle', 'rf', 'factorial', 'benignbuf', 'budget') + FINAL_GROUPS,
                    required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    tf.config.experimental.enable_op_determinism()
    D = load(a.dataset)
    if a.group in FINAL_GROUPS:
        out = (OUT_FINAL if not a.smoke else HERE / 'results_p3final_smoke') / a.dataset
    else:
        out = OUT / a.dataset if not a.smoke else HERE / 'results_v3_smoke' / a.dataset
    print(f"{a.dataset} {a.group}: devices {[d.name for d in tf.config.list_logical_devices()]}, output {out}", flush=True)

    def job(order, seed, strategy, hp, group, oracle=False, tag=None, variant=None):
        p = out / group / f"{tag or strategy}_o{order}_s{seed}.json"
        if p.exists():
            return json.loads(p.read_text())
        tasks = task_order(D, order); T = build_tasks(D, tasks, order)
        S = preprocess(D, T, order, oracle)
        t0 = time.time()
        r = run(D, S, T, strategy, seed, hp, smoke=a.smoke,
                cache_key=f"{a.dataset}_o{order}_s{seed}_{'oracle' if oracle else 'std'}{'_' + variant if variant else ''}{'_smoke' if a.smoke else ''}")
        r.update(dataset=a.dataset, order=order, seed=seed, strategy=strategy, oracle=oracle,
                 tasks=[[D['classes'][c] for c in t['classes']] for t in T], features=S['feat'],
                 classes=D['classes'], clip_share_te=S['clip_share_te'], wall_s=round(time.time() - t0, 1), variant=variant)
        save(p, r)
        print(f"{a.dataset} {group} {tag or strategy} o{order} s{seed}: final macro-F1(seen) "
              f"{r['stages'][-1]['macro_f1_seen']:.4f}  ({r['wall_s']:.0f} s)", flush=True)
        return r

    seeds = SEEDS[:1] if a.smoke else SEEDS
    if a.group == 'tune':
        best = {}
        for strat, grid in GRID.items():
            scores = {}
            for k, v in (grid[:2] if a.smoke else grid):
                r = job(0, DEV_SEED, strat, {k: v}, 'tune', tag=f'{strat}_{k}{v:g}')
                # selection: validation macro-F1 over all seen classes after the last task (development seed only).
                # (v3 first used the mean current-task validation F1, which rewards plasticity and always picked the
                # weakest regulariser; replaced before any results were analysed.)
                scores[v] = float(r['stages'][-1]['val_seen_f1'])
            vbest = max(scores, key=scores.get)
            best[strat] = dict(param=grid[0][0], value=vbest, scores={f'{k:g}': s for k, s in scores.items()},
                               edge=vbest in (grid[0][1], grid[-1][1]))
        save(out / 'tuned.json', best)
        print('tuned:', best)
        return
    tn = tuned(a.dataset) if not a.smoke else {'ewc': {'param': 'lam', 'value': 100.0}, 'lwf': {'param': 'alpha', 'value': 1.0}}
    assert tn, 'run --group tune first'
    hp = lambda s: {tn[s]['param']: tn[s]['value']} if s in tn else {}
    if a.group == 'main':
        for seed in seeds:
            for s in STRATEGIES:
                job(0, seed, s, hp(s), 'main')
    elif a.group == 'orders':
        for o in range(1, (1 if a.smoke else N_RANDOM_ORDERS) + 1):
            for s in STRATEGIES:
                job(o, o, s, hp(s), 'orders')
    elif a.group == 'oracle':
        for seed in seeds:
            for s in ('naive', 'joint', 'er2000'):
                job(0, seed, s, hp(s), 'oracle', oracle=True)
    elif a.group == 'factorial':
        # separate the two v2 -> v3 changes (input compression map, training budget) for joint retraining;
        # the asinh + v3-budget cell is the main group
        for seed in seeds:
            for asinh, budget in ((False, 'v3'), (True, 'v2'), (False, 'v2')):
                USE_ASINH = asinh
                CFG0, CFGI = BUDGETS[budget]
                v = f"{'asinh' if asinh else 'minmax'}_{budget}"
                job(0, seed, 'joint', {}, 'factorial', tag=f'joint_{v}', variant=v)
        USE_ASINH, (CFG0, CFGI) = True, BUDGETS['v3']
    elif a.group == 'benignbuf':
        # is replay's higher benign FPR caused by the class-balanced memory under-representing benign traffic?
        for seed in seeds:
            job(0, seed, 'er2000b', {}, 'benignbuf')
    elif a.group == 'budget':
        # reviewer check (2026-11-02): is replay's newest-task shortfall caused by the smaller incremental budget?
        # ER-2000 with the Task-0 / joint budget for every task (same shared Task-0 model)
        for seed in seeds:
            CFGI = dict(CFG0)
            job(0, seed, 'er2000', {}, 'budget', tag='er2000_fullbudget')
        CFG0, CFGI = BUDGETS['v3']
    elif a.group == 'rf':
        T = build_tasks(D, task_order(D, 0), 0)
        S = preprocess(D, T, 0)
        for seed in seeds:
            p = out / 'rf' / f'rf_o0_s{seed}.json'
            if not p.exists():
                save(p, dict(rf_reference(D, S, T, seed), classes=D['classes'], seed=seed))
                print('rf', seed, flush=True)
    # ── final review round (2026-11-07): results_p3final/<dataset>/<group>/ ──
    elif a.group == 'agemfix':
        # corrected A-GEM: semantic order x 5 seeds (main/), then one seed on each random order (orders/);
        # same shared Task-0 models as results_v3 (cache keys unchanged)
        for seed in seeds:
            job(0, seed, 'agem2', {}, 'main')
        for o in range(1, (1 if a.smoke else N_RANDOM_ORDERS) + 1):
            job(o, o, 'agem2', {}, 'orders')
    elif a.group == 'replayabl':
        # replay-weighting ablation: memory added once to the training pool, no 1:1 mixing (same memory contents)
        for seed in seeds:
            job(0, seed, 'er2000p', {}, 'replayabl')
    elif a.group == 'rfmem':
        T = build_tasks(D, task_order(D, 0), 0)
        S = preprocess(D, T, 0)
        tasks_named = [[D['classes'][c] for c in t['classes']] for t in T]
        for seed in seeds:
            for mode, name in (('memory', 'rf_er2000'), ('joint', 'rf_joint')):
                p = out / 'rfmem' / f'{name}_o0_s{seed}.json'
                if p.exists():
                    continue
                t0 = time.time()
                r = rf_stagewise(D, S, T, seed, mode)
                r.update(dataset=a.dataset, order=0, seed=seed, strategy=name, oracle=False, tasks=tasks_named,
                         features=S['feat'], classes=D['classes'], wall_s=round(time.time() - t0, 1), variant=None)
                save(p, r)
                print(f"{a.dataset} rfmem {name} s{seed}: final macro-F1(seen) {r['stages'][-1]['macro_f1_seen']:.4f} "
                      f"({r['wall_s']:.0f} s)", flush=True)
    elif a.group == 'rangeshare':
        p = out / 'rangeshare.json'
        if not p.exists():
            T = build_tasks(D, task_order(D, 0), 0)
            save(p, dict(range_share(D, T, 0), dataset=a.dataset, order=0))
            print(f'{a.dataset} rangeshare saved', flush=True)


if __name__ == '__main__':
    main()
