"""
run_fl_v2.py — Paper 4 v2: federated learning for IoT intrusion detection, rebuilt after review.

Fixes relative to run_federated.py (see Revision_v2/Paper4/review_r1_*.md):
  * Round selection on a held-out VALIDATION set (the validation split is no longer merged into
    training); test F1 is reported at the validation-selected round and at the final round.
  * Every repetition draws a new client partition, model initialisation and batch order
    (the old code fixed the partition at seed 42).
  * Matched references: a centralised model trained with the same recipe (no SMOTE, same
    optimiser, same number of passes, same validation selection) and local-only models.
  * Hyperparameters tuned on validation: FedProx mu, FedAdam server lr (tau = 1e-3, trainable
    weights only, Reddi et al.), SCAFFOLD and FedAvg-SGD learning rates, FedAvg (Adam) lr.
  * GroupNorm in every model (v2.1): BatchNorm's running statistics made evaluation swing between
    checkpoints (centralised validation F1 0.51-0.82) although batch-statistics evaluation was stable;
    BatchNorm is kept as an ablation.
  * Record-level DP-SGD on each client (Abadi et al. 2016) that actually trains: per-example
    gradients, per-example clipping of the full gradient, Gaussian noise sigma*C, Poisson
    sampling with rate q = B/n_k, a GroupNorm model (no batch statistics), and an RDP accountant
    (dp_accounting) matching the sampler; epsilon reported per client (max over clients).
    Sanity runs: no noise and no clipping; clipping only.
  * Compiled training steps, one reused client model, data kept on the GPU.

Usage:  python -m experiments.paper4_federated.run_fl_v2 --dataset TON_IoT --group tune
        python -m experiments.paper4_federated.run_fl_v2 --dataset TON_IoT --group core --seeds 0,1,2,3,4
Results: experiments/paper4_federated/results_v2/<dataset>/<experiment>/seed<r>.json
"""
import argparse, contextlib, gc, io, json, os, sys, time, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
if '--cpu' in sys.argv:            # DP-SGD runs: per-example gradients are faster on this CPU than on the GTX 1650
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
if '--cpu' in sys.argv:
    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(2)

for _g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_g, True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sklearn.metrics import f1_score
from experiments.baseline.config import DATASETS

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_v2'
BASE_RES = ROOT / 'experiments' / 'baseline' / 'results'

ROUNDS = 50
LOCAL_EPOCHS = 2
BATCH = 256
LR = 1e-3
DELTA = 1e-5
CLIP = 1.0
FEDADAM_TAU = 1e-3
DP_SIGMAS = (0.6, 1.0, 1.5, 2.5)
DP_ROUNDS, DP_LOCAL_EPOCHS = 20, 1      # shorter schedule for the (CPU-bound) DP-SGD runs and their sanity runs

# round-3 tuning: two tuning repetitions on partitions disjoint from the reported ones (seeds 100, 101),
# wider grids, FedProx/FedAdam tuned at the tuned FedAvg client lr (stage b after stage a)
# round 5: a third tuning repetition (seed 102) and grids widened where round 4 selected an edge
TUNE_SEEDS = (100, 101, 102)
TUNE3_A = {
    'fedavg':    ('lr', (1e-3, 3e-3, 6e-3, 1e-2)),
    'scaffold':  ('lr', (0.01, 0.03, 0.1, 0.3, 1.0)),
    'fedavgsgd': ('lr', (0.01, 0.03, 0.1, 0.3, 1.0)),
}
TUNE3_B = {
    'fedprox':   ('mu', (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)),
    'fedadam':   ('server_lr', (0.003, 0.01, 0.03, 0.1)),
}
PORTS = ('src_port', 'dst_port')
TUNE3_LC = {'fedlc': ('tau', (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0))}
CENTRAL_LR = (1e-3, 3e-3, 6e-3, 1e-2)
# N-BaIoT (115 features, slower runs): two tuning repetitions and grids without the values that diverged
# on both other datasets (SGD lr 1.0) or were never close (FedAdam server lr 0.003)
NB_TUNE_SEEDS = (100, 101)
NB_GRIDS = {
    'a': {'fedavg': ('lr', (3e-4, 1e-3, 3e-3, 6e-3, 1e-2)), 'scaffold': ('lr', (0.01, 0.03, 0.1, 0.3)),
          'fedavgsgd': ('lr', (0.01, 0.03, 0.1, 0.3))},
    'b': {'fedprox': ('mu', (1e-5, 1e-4, 1e-3, 1e-2)), 'fedadam': ('server_lr', (0.01, 0.03, 0.1))},
    'lc': {'fedlc': ('tau', (0.3, 1.0, 3.0, 10.0, 30.0, 100.0))},
}


def grids3(ds, stage):
    if ds == 'N_BaIoT':
        return NB_GRIDS[stage]
    return TUNE3_A if stage == 'a' else TUNE3_B if stage == 'b' else TUNE3_LC     # round 5: the centralised reference's own lr check
DP_FULL_SIGMA = 1.8       # epsilon at 50 rounds x 2 epochs about equal to sigma = 1 at 20 x 1

TUNE_GRID = {
    'fedavg':    ('lr', (3e-4, 1e-3, 3e-3)),
    'fedprox':   ('mu', (0.001, 0.01, 0.1, 1.0)),
    'fedadam':   ('server_lr', (0.01, 0.03, 0.1, 0.3, 1.0)),
    'scaffold':  ('lr', (0.01, 0.03, 0.1, 0.3)),
    'fedavgsgd': ('lr', (0.01, 0.03, 0.1, 0.3)),
}


# ── data ──────────────────────────────────────────────────────────────────────

def load_nbaiot():
    """N-BaIoT prepared by prepare_nbaiot.py (3% sample, deduplicated, 70/15/15, min-max on train)."""
    z = np.load(ROOT / 'dataset4' / 'nbaiot_prepared.npz')
    info = json.loads((ROOT / 'dataset4' / 'nbaiot_prepared.json').read_text())
    classes = info['classes']
    return dict(ds='N_BaIoT', Xtr=z['Xtr'], ytr=z['ytr'], Xva=z['Xva'], yva=z['yva'], Xte=z['Xte'], yte=z['yte'],
                classes=classes, normal=classes.index('normal'), features=info['features'], dropped=[],
                dev_tr=z['dtr'], dev_te=z['dte'], devices=info['devices'], dst_tr=None, dst_te=None)


def load_data(ds, drop=(), meta=False):
    if ds == 'N_BaIoT':
        return load_nbaiot()
    import joblib
    from src.data.loader import load_dataset
    pipe = joblib.load(BASE_RES / ds / 'pipeline.pkl')
    csv, label = DATASETS[ds]
    with contextlib.redirect_stdout(io.StringIO()):
        raw = load_dataset(csv, label_col=label, seed=42, return_meta=meta)
    f = lambda X: pipe.transform(X)[0].astype(np.float32)
    classes = list(raw['class_names'])
    feats = list(pipe.selected_features_)
    keep = [i for i, n in enumerate(feats) if n not in drop]
    g = lambda X: f(X)[:, keep]
    return dict(ds=ds, Xtr=g(raw['X_train']), ytr=raw['y_train'].astype(np.int32),
                Xva=g(raw['X_val']), yva=raw['y_val'].astype(np.int32),
                Xte=g(raw['X_test']), yte=raw['y_test'].astype(np.int32),
                classes=classes, normal=classes.index('normal'), features=[feats[i] for i in keep],
                dropped=[n for n in feats if n in drop],
                dst_tr=raw['meta_train']['dst_ip'].astype(str).values if meta else None,
                dst_te=raw['meta_test']['dst_ip'].astype(str).values if meta else None)


def iid_partition(n, n_clients, rng):
    return np.array_split(rng.permutation(n), n_clients)


def natural_partition(dst, n_clients, rng, randomized=False):
    """Device-level partition: every destination host (IP) belongs to exactly one client, so a
    client sees all traffic to the devices it monitors. Each host is assigned to the currently
    smallest client. randomized=False (round 4, 'nat'): large hosts first, only ties in random order,
    so the partition is almost the same in every repetition. randomized=True (round 5, 'natr'):
    hosts in a fresh random order per repetition, so the partition is redrawn.
    Returns the client index arrays and the host -> client map."""
    hosts, inv = np.unique(dst, return_inverse=True)
    sizes = np.bincount(inv)
    load, owner = np.zeros(n_clients), np.empty(len(hosts), int)
    order = rng.permutation(len(hosts))
    if not randomized:
        order = order[np.argsort(-sizes[order], kind='stable')]   # large hosts first, ties in random order
    for h in order:
        k = int(np.argmin(load)); owner[h] = k; load[k] += sizes[h]
    clients = [rng.permutation(np.where(owner[inv] == k)[0]) for k in range(n_clients)]
    return clients, dict(zip(hosts.tolist(), owner.tolist()))


def make_partition(D, cfg, prng):
    """Client index arrays, and (natural partitions) each client's own test records: test traffic to
    the hosts it owns (test hosts never seen in training belong to no client)."""
    if cfg.get('partition') == 'device':
        # N-BaIoT: every physical device is one client (fixed by the data; repetitions vary the rest)
        nd = len(D['devices'])
        return ([prng.permutation(np.where(D['dev_tr'] == k)[0]) for k in range(nd)],
                [np.where(D['dev_te'] == k)[0] for k in range(nd)])
    if cfg.get('partition') == 'natural':
        clients, own = natural_partition(D['dst_tr'], cfg['n_clients'], prng, cfg.get('randomized', False))
        te_owner = np.array([own.get(h, -1) for h in D['dst_te']])
        return clients, [np.where(te_owner == k)[0] for k in range(cfg['n_clients'])]
    if cfg['alpha'] is None:
        return iid_partition(len(D['ytr']), cfg['n_clients'], prng), None
    return dirichlet_partition(D['ytr'], cfg['n_clients'], cfg['alpha'], prng), None


def dirichlet_partition(y, n_clients, alpha, rng):
    """Per class, split the records over clients with Dirichlet(alpha) proportions."""
    idx = [[] for _ in range(n_clients)]
    for c in np.unique(y):
        ci = rng.permutation(np.where(y == c)[0])
        cuts = (np.cumsum(rng.dirichlet(alpha * np.ones(n_clients))) * len(ci)).astype(int)[:-1]
        for k, part in enumerate(np.split(ci, cuts)):
            idx[k].extend(part.tolist())
    out = [rng.permutation(np.array(i, dtype=np.int64)) for i in idx]
    # a client must hold data: move 1% of the largest client's records to any empty client
    for k in range(n_clients):
        if len(out[k]) == 0:
            big = int(np.argmax([len(o) for o in out]))
            m = max(1, len(out[big]) // 100)
            out[k], out[big] = out[big][:m], out[big][m:]
    return out


# ── model ─────────────────────────────────────────────────────────────────────

class GroupNorm(tf.keras.layers.Layer):
    """Group normalisation (Wu & He 2018) over (length, channels-in-group); no batch statistics."""

    def __init__(self, groups=8, eps=1e-5, **kw):
        super().__init__(**kw)
        self.groups, self.eps = groups, eps

    def build(self, shape):
        c = int(shape[-1])
        self.gamma = self.add_weight(name='gamma', shape=(c,), initializer='ones')
        self.beta = self.add_weight(name='beta', shape=(c,), initializer='zeros')

    def call(self, x):
        b, l, c, g = tf.shape(x)[0], tf.shape(x)[1], x.shape[-1], self.groups
        xr = tf.reshape(x, tf.stack([b, l, g, c // g]))
        m, v = tf.nn.moments(xr, axes=[1, 3], keepdims=True)
        xr = (xr - m) * tf.math.rsqrt(v + self.eps)
        return tf.reshape(xr, tf.stack([b, l, c])) * self.gamma + self.beta


def build_model(nf, nc, norm='bn', dropout=0.3):
    Norm = (lambda: tf.keras.layers.BatchNormalization()) if norm == 'bn' else (lambda: GroupNorm(8))
    inp = tf.keras.Input(shape=(nf, 1))
    x = tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu')(inp)
    x = Norm()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = Norm()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    if dropout:
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    if dropout:
        x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(nc, activation='softmax')(x)
    return tf.keras.Model(inp, out)


def weights(model):
    return [w.numpy() for w in model.weights]


def assign(model, ws):
    for v, w in zip(model.weights, ws):
        v.assign(w)


def trainable_mask(model):
    tr = {v.ref() for v in model.trainable_variables}
    return [w.ref() in tr for w in model.weights]


# ── local training ────────────────────────────────────────────────────────────

class LocalTrainer:
    """Compiled local steps for one model; data live on the GPU and batches are index tensors."""

    def __init__(self, model, X, Y, algo, lr, mu=0.0, clip=0.0, sigma=0.0, expected_b=BATCH):
        self.model, self.algo, self.lr = model, algo, lr
        self.X, self.Y = tf.constant(X), tf.constant(Y)
        self.tv = model.trainable_variables
        sgd = algo in ('scaffold', 'fedavgsgd')
        self.opt = tf.keras.optimizers.SGD(lr) if sgd else tf.keras.optimizers.Adam(lr)
        self.mu = tf.Variable(float(mu), trainable=False)
        self.gw = [tf.Variable(tf.zeros_like(v), trainable=False) for v in self.tv]    # FedProx anchor
        self.c = [tf.Variable(tf.zeros_like(v), trainable=False) for v in self.tv]     # SCAFFOLD global
        self.ci = [tf.Variable(tf.zeros_like(v), trainable=False) for v in self.tv]    # SCAFFOLD client
        self.delta = tf.Variable(tf.zeros(model.output_shape[-1]), trainable=False)   # FedLC calibration
        ce = tf.keras.losses.sparse_categorical_crossentropy
        prox, scaf, lc = algo == 'fedprox', algo == 'scaffold', algo == 'fedlc'
        clip_, sigma_, eb = float(clip), float(sigma), float(expected_b)

        def step(idx):
            x = tf.expand_dims(tf.gather(self.X, idx), -1); y = tf.gather(self.Y, idx)
            with tf.GradientTape() as t:
                if lc:   # calibrated logits z_c - delta_c, delta_c = tau * n_{k,c}^(-1/4)
                    z = tf.math.log(self.model(x, training=True) + 1e-8) - self.delta
                    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, z))
                else:
                    loss = tf.reduce_mean(ce(y, self.model(x, training=True)))
                if prox:
                    loss += self.mu / 2.0 * tf.add_n([tf.reduce_sum(tf.square(w - g)) for w, g in zip(self.tv, self.gw)])
            g = t.gradient(loss, self.tv)
            if scaf:
                g = [gi - ci + c for gi, ci, c in zip(g, self.ci, self.c)]
            self.opt.apply_gradients(zip(g, self.tv))
            return loss

        def dp_step(idx):
            x = tf.expand_dims(tf.gather(self.X, idx), -1); y = tf.gather(self.Y, idx)

            def one(a):
                xi, yi = a
                with tf.GradientTape() as t:
                    li = ce(yi[None], self.model(xi[None], training=False))[0]
                gi = t.gradient(li, self.tv)
                if clip_ > 0:
                    f = tf.minimum(1.0, clip_ / (tf.linalg.global_norm(gi) + 1e-12))
                    gi = [v * f for v in gi]
                return gi

            G = tf.vectorized_map(one, (x, y))
            s = [tf.reduce_sum(gi, 0) for gi in G]
            if sigma_ > 0:
                s = [si + tf.random.normal(tf.shape(si), stddev=sigma_ * clip_) for si in s]
            self.opt.apply_gradients(zip([si / eb for si in s], self.tv))
            return tf.constant(0.0)

        body = dp_step if algo == 'dpsgd' else step

        @tf.function(input_signature=[tf.TensorSpec([None, None], tf.int32)], jit_compile=True)
        def run_fixed(B):
            # fixed-size batches (rows of B) in one XLA-compiled loop (non-DP training on the GPU)
            for i in tf.range(tf.shape(B)[0]):
                step(B[i])
            return tf.constant(0)

        self.run_fixed = run_fixed

        @tf.function(input_signature=[tf.TensorSpec([None], tf.int32), tf.TensorSpec([None], tf.int32)])
        def run_batches(flat, starts):
            # all batches of one or more epochs in one compiled loop: batch i = flat[starts[i]:starts[i+1]]
            for i in tf.range(tf.shape(starts)[0] - 1):
                body(flat[starts[i]:starts[i + 1]])
            return tf.constant(0)

        self.run_batches = run_batches

    def reset_optimizer(self):
        for v in self.opt.variables():
            v.assign(tf.zeros_like(v))

    def run(self, idx, epochs, rng, poisson_q=None):
        """Local epochs over the index array `idx` (into X). Poisson sampling when poisson_q is set.
        Returns the number of local steps."""
        if poisson_q is None:
            # full batches of BATCH records per epoch (the last partial batch of each epoch is dropped)
            rows = []
            for _ in range(epochs):
                perm = rng.permutation(idx)
                nb = len(perm) // BATCH
                rows.append(perm[:nb * BATCH].reshape(nb, BATCH) if nb else perm[None, :])
            if len({r.shape[1] for r in rows}) == 1:
                B = np.concatenate(rows).astype(np.int32)
                self.run_fixed(tf.constant(B))
                return int(B.shape[0])
        batches = []
        for _ in range(epochs):
            if poisson_q is None:
                perm = rng.permutation(idx)
                batches += [perm[i:i + BATCH] for i in range(0, len(perm), BATCH)]
            else:
                for _ in range(int(np.ceil(1.0 / poisson_q))):
                    b = idx[rng.random(len(idx)) < poisson_q]
                    if len(b):
                        batches.append(b)
        flat = np.concatenate(batches).astype(np.int32)
        starts = np.concatenate([[0], np.cumsum([len(b) for b in batches])]).astype(np.int32)
        self.run_batches(tf.constant(flat), tf.constant(starts))
        return len(batches) if poisson_q is None else epochs * int(np.ceil(1.0 / poisson_q))


# ── evaluation ────────────────────────────────────────────────────────────────

class Evaluator:
    def __init__(self, model, D):
        self.model, self.D = model, D

        @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32)])
        def pred(x):
            return tf.argmax(self.model(tf.expand_dims(x, -1), training=False), axis=1)
        self.pred = pred
        self.labels = list(range(len(D['classes'])))

    def predict(self, X):
        return np.concatenate([self.pred(tf.constant(X[i:i + 8192])).numpy() for i in range(0, len(X), 8192)])

    def scores(self, X, y, full=False):
        p = self.predict(X)
        r = dict(f1=float(f1_score(y, p, average='macro', labels=self.labels, zero_division=0)), acc=float((p == y).mean()))
        if full:
            nrm = self.D['normal']
            r.update(per_class_f1=dict(zip(self.D['classes'], f1_score(y, p, average=None, labels=self.labels, zero_division=0).tolist())),
                     benign_fpr=float((p[y == nrm] != nrm).mean()), attack_detection=float((p[y != nrm] != nrm).mean()))
        return r

    def site_scores(self, site_te):
        """Each client's own test traffic: macro-F1 over the classes present in that traffic,
        accuracy, benign false-positive rate and attack detection."""
        out = []
        for ix in site_te:
            y = self.D['yte'][ix]
            if len(ix) == 0:
                out.append(dict(n=0)); continue
            p = self.predict(self.D['Xte'][ix])
            nrm = self.D['normal']
            out.append(dict(n=int(len(ix)), classes=np.unique(y).tolist(),
                            f1=float(f1_score(y, p, average='macro', labels=np.unique(y), zero_division=0)),
                            acc=float((p == y).mean()),
                            benign_fpr=float((p[y == nrm] != nrm).mean()) if (y == nrm).any() else None,
                            attack_detection=float((p[y != nrm] != nrm).mean()) if (y != nrm).any() else None))
        return out


def fed_recal(model, D, clients, rng, per_client=8192):
    """Federated re-estimation of BatchNorm statistics (deployable): layer by layer, every client
    sends the mean and mean square of the layer input over (up to) per_client of its own records,
    computed with the current global model; the server averages them weighted by n_k and sets the
    running mean and variance. Returns the saved statistics so the caller can restore them."""
    bns = [l for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    saved = [(l.moving_mean.numpy(), l.moving_variance.numpy()) for l in bns]
    probe = tf.keras.Model(model.input, [l.input for l in bns])
    samples = [rng.choice(c, min(per_client, len(c)), replace=False) for c in clients]
    w = np.array([len(c) for c in clients], float); w /= w.sum()
    for j, l in enumerate(bns):
        m1, m2 = 0.0, 0.0
        for wk, s in zip(w, samples):
            a = probe(tf.constant(D['Xtr'][s][..., None]), training=False)[j].numpy()
            a = a.reshape(-1, a.shape[-1])
            m1 = m1 + wk * a.mean(0); m2 = m2 + wk * (a * a).mean(0)
        l.moving_mean.assign(m1.astype(np.float32))
        l.moving_variance.assign(np.maximum(m2 - m1 * m1, 0).astype(np.float32))
    del probe
    return bns, saved


def fed_recal_scores(model, ev, D, clients, rng):
    """Diagnostic per round: score validation and test after federated re-estimation, then restore."""
    bns, saved = fed_recal(model, D, clients, rng)
    v, t = ev.scores(D['Xva'], D['yva']), ev.scores(D['Xte'], D['yte'])
    for l, (m, var) in zip(bns, saved):
        l.moving_mean.assign(m); l.moving_variance.assign(var)
    return dict(val_f1_fedrecal=v['f1'], test_f1_fedrecal=t['f1'])


def recal_scores(model, ev, D, rng):
    """Diagnostic only: replace BatchNorm running statistics by the statistics of 8,192 random
    training records (momentum 0, one forward pass in training mode; dropout follows the BN layers
    and does not affect them), score validation and test, then restore the running statistics."""
    bns = [l for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    if not bns:
        return {}
    saved = [(l.moving_mean.numpy(), l.moving_variance.numpy(), l.momentum) for l in bns]
    for l in bns:
        l.momentum = 0.0
    idx = rng.choice(len(D['Xtr']), min(8192, len(D['Xtr'])), replace=False)
    model(tf.constant(D['Xtr'][idx][..., None]), training=True)
    v, t = ev.scores(D['Xva'], D['yva']), ev.scores(D['Xte'], D['yte'])
    for l, (m, var, mom) in zip(bns, saved):
        l.moving_mean.assign(m); l.moving_variance.assign(var); l.momentum = mom
    return dict(val_f1_recal=v['f1'], test_f1_recal=t['f1'])


def select(history):
    """Validation-selected round (earliest maximum of validation macro-F1)."""
    best = max(range(len(history)), key=lambda i: (history[i]['val_f1'], -i))
    return best


# ── federated / centralised / local-only runs ─────────────────────────────────

def run_federated(D, cfg, rep):
    rng = np.random.default_rng(2000 + rep)
    prng = np.random.default_rng(1000 + rep)
    tf.keras.utils.set_random_seed(rep)
    n = len(D['ytr'])
    clients, site_te = make_partition(D, cfg, prng)
    nf, nc = D['Xtr'].shape[1], len(D['classes'])
    dp = cfg['algo'] == 'dpsgd'
    norm = 'gn' if dp else cfg.get('norm', 'gn')
    gmodel = build_model(nf, nc, norm, dropout=0 if dp else 0.3)
    cmodel = build_model(nf, nc, norm, dropout=0 if dp else 0.3)
    assign(cmodel, weights(gmodel))
    mask = trainable_mask(gmodel)
    lt = LocalTrainer(cmodel, D['Xtr'], D['ytr'], cfg['algo'], cfg['lr'], mu=cfg.get('mu', 0.0),
                      clip=cfg.get('clip', 0.0), sigma=cfg.get('sigma', 0.0))
    ev = Evaluator(gmodel, D)
    gw = weights(gmodel)
    m_t = [np.zeros_like(w) for w, t in zip(gw, mask) if t]; v_t = [np.zeros_like(w) for w, t in zip(gw, mask) if t]
    c_glob = [np.zeros_like(v.numpy()) for v in cmodel.trainable_variables]
    c_cli = [[np.zeros_like(v.numpy()) for v in cmodel.trainable_variables] for _ in clients]
    hist, t0 = [], time.time()
    rrng = np.random.default_rng(3000 + rep)
    for rnd in range(1, cfg['rounds'] + 1):
        upd, sizes, dcs = [], [], []
        for k, idx in enumerate(clients):
            assign(cmodel, gw)
            lt.reset_optimizer()
            if cfg['algo'] == 'fedprox':
                for a, v in zip(lt.gw, cmodel.trainable_variables):
                    a.assign(v)
            if cfg['algo'] == 'fedlc':
                cnt = np.maximum(np.bincount(D['ytr'][idx], minlength=nc), 1).astype(np.float32)
                lt.delta.assign(cfg['tau'] * cnt ** -0.25)
            if cfg['algo'] == 'scaffold':
                x0 = [v.numpy() for v in cmodel.trainable_variables]
                for a, b in zip(lt.c, c_glob): a.assign(b)
                for a, b in zip(lt.ci, c_cli[k]): a.assign(b)
            q = BATCH / len(idx) if dp else None
            K = lt.run(idx, cfg['local_epochs'], rng, poisson_q=q)
            if cfg['algo'] == 'scaffold':
                yk = [v.numpy() for v in cmodel.trainable_variables]
                new = [ci - c + (a - b) / (K * cfg['lr']) for ci, c, a, b in zip(c_cli[k], c_glob, x0, yk)]
                dcs.append([a - b for a, b in zip(new, c_cli[k])]); c_cli[k] = new
            upd.append(weights(cmodel)); sizes.append(len(idx))
        tot = float(sum(sizes))
        avg = [sum(u[i] * (s / tot) for u, s in zip(upd, sizes)) for i in range(len(gw))]
        if cfg['algo'] == 'fedadam':
            j, new = 0, []
            for i, (g, a, t) in enumerate(zip(gw, avg, mask)):
                if t:
                    d = a - g
                    m_t[j] = 0.9 * m_t[j] + 0.1 * d
                    v_t[j] = 0.99 * v_t[j] + 0.01 * d * d
                    new.append(g + cfg['server_lr'] * m_t[j] / (np.sqrt(v_t[j]) + FEDADAM_TAU)); j += 1
                else:
                    new.append(a)
            gw = new
        else:
            gw = avg
        if cfg['algo'] == 'scaffold':
            c_glob = [c + sum(d[j] for d in dcs) / len(clients) for j, c in enumerate(c_glob)]
        assign(gmodel, gw)
        h = dict(round=rnd, **{f'val_{k}': v for k, v in ev.scores(D['Xva'], D['yva']).items()},
                 **{f'test_{k}': v for k, v in ev.scores(D['Xte'], D['yte']).items()}, sec=time.time() - t0)
        if cfg.get('recal') == 'fed':
            h.update(fed_recal_scores(gmodel, ev, D, clients, rrng))
        elif cfg.get('recal'):
            h.update(recal_scores(gmodel, ev, D, rrng))
        if not np.isfinite(sum(float(np.abs(w).sum()) for w in gw)):
            h['nonfinite'] = True
        hist.append(h)
        if rnd == cfg['rounds'] or rnd % 10 == 0:
            print(f"    r{rnd:3d} val {h['val_f1']:.4f} test {h['test_f1']:.4f} ({h['sec']:.0f}s)", flush=True)
        if cfg.get('keep_best'):
            if select(hist) == len(hist) - 1:
                best_w = [w.copy() for w in gw]
    s = select(hist)
    if cfg.get('keep_best'):
        assign(gmodel, best_w)
        sel_full = ev.scores(D['Xte'], D['yte'], full=True)
        sel_site = ev.site_scores(site_te) if site_te is not None else None
        ft = (local_finetune2(cmodel, best_w, D, clients, site_te, cfg, rep) if cfg.get('ft_modes') else
              local_finetune(cmodel, lt, best_w, D, clients, site_te, cfg['finetune'], rep)) if cfg.get('finetune') and site_te is not None else None
    else:
        sel_full = sel_site = ft = None
    assign(gmodel, gw)
    fin_full = ev.scores(D['Xte'], D['yte'], full=True)
    size_b = int(sum(w.size * 4 for w in gw))
    comm = len(clients) * size_b * 2 * (2 if cfg['algo'] == 'scaffold' else 1)
    out = dict(cfg=cfg, rep=rep, norm=norm, client_sizes=[int(len(c)) for c in clients],
               client_class_counts=[np.bincount(D['ytr'][c], minlength=nc).tolist() for c in clients],
               selected_round=hist[s]['round'], val_f1_selected=hist[s]['val_f1'],
               test_f1_selected=hist[s]['test_f1'], test_acc_selected=hist[s]['test_acc'],
               test_f1_final=hist[-1]['test_f1'], test_acc_final=hist[-1]['test_acc'],
               test_selected_full=sel_full, test_final_full=fin_full, site_selected=sel_site,
               model_bytes=size_b, bytes_per_round=comm, total_mb=comm * cfg['rounds'] / 1e6,
               raw_train_mb=float(D['Xtr'].nbytes / 1e6), history=hist, seconds=time.time() - t0,
               any_nonfinite=any(h.get('nonfinite') for h in hist))
    if ft is not None:
        out['finetune'] = ft
    if dp:
        out['dp'] = dp_epsilon(out['client_sizes'], cfg)
    del gmodel, cmodel, lt, ev; tf.keras.backend.clear_session(); gc.collect()
    return out


def local_finetune(cmodel, lt, w0, D, clients, site_te, epochs, rep):
    """Round 5b: each site adapts the validation-selected global model to its own data with a few
    local epochs (Adam, same learning rate, optimiser state reset) and is scored after each number of
    epochs in `epochs` (cumulative) on the common test set and on its own test traffic. No selection
    on test data: the epoch counts are fixed in advance."""
    evc = Evaluator(cmodel, D)
    frng = np.random.default_rng(4000 + rep)
    out = []
    for k, idx in enumerate(clients):
        assign(cmodel, w0); lt.reset_optimizer(); done, rec = 0, {}
        for E in epochs:
            lt.run(idx, E - done, frng); done = E
            rec[str(E)] = dict(test_f1=evc.scores(D['Xte'], D['yte'])['f1'], site=evc.site_scores([site_te[k]])[0])
        out.append(rec)
    del evc
    return out


FT_MU = 0.1   # proximal weight for proximal fine-tuning, fixed in advance (no selection)


def binary(pred, y, normal):
    """Attack detection rate and benign false-positive rate of an attack/benign decision."""
    att = pred != normal
    return dict(attack_detection=float(att[y != normal].mean()) if (y != normal).any() else None,
                benign_fpr=float(att[y == normal].mean()) if (y == normal).any() else None)


def local_finetune2(cmodel, w0, D, clients, site_te, cfg, rep):
    """Round 6: personalisation baselines. Each site adapts the validation-selected global model w0 with
    'full' (Adam, same lr), 'low' (Adam, lr / 10) or 'prox' (Adam, same lr, proximal term FT_MU/2 ||w - w0||^2)
    for the cumulative epoch counts in cfg['finetune'] (fixed in advance, no selection). Every variant is scored on the
    common test set and on the site's own test traffic, and so is the two-model combination 'either' (alert if the
    global or the adapted model predicts an attack: attack detection and benign FPR)."""
    evc = Evaluator(cmodel, D)
    nrm = D['normal']
    trainers = {'full': LocalTrainer(cmodel, D['Xtr'], D['ytr'], 'fedavg', cfg['lr']),
                'low': LocalTrainer(cmodel, D['Xtr'], D['ytr'], 'fedavg', cfg['lr'] / 10),
                'prox': LocalTrainer(cmodel, D['Xtr'], D['ytr'], 'fedprox', cfg['lr'], mu=FT_MU)}
    assign(cmodel, w0)
    g_test = evc.predict(D['Xte'])
    out = []
    for k, idx in enumerate(clients):
        sk = site_te[k]; ys, yt = D['yte'][sk], D['yte']
        g_site = g_test[sk]
        rec = dict(global_site=binary(g_site, ys, nrm), global_test=binary(g_test, yt, nrm))
        for mode, tr in [(m, trainers[m]) for m in cfg['ft_modes']]:
            frng = np.random.default_rng(4000 + rep)
            assign(cmodel, w0); tr.reset_optimizer()
            if mode == 'prox':
                for a, v in zip(tr.gw, cmodel.trainable_variables):
                    a.assign(v)
            done, r = 0, {}
            for E in cfg['finetune']:
                tr.run(idx, E - done, frng); done = E
                p_test = evc.predict(D['Xte']); p_site = p_test[sk]
                r[str(E)] = dict(test_f1=evc.scores(D['Xte'], D['yte'])['f1'], site=evc.site_scores([sk])[0],
                                 either_site=binary(np.where(g_site != nrm, g_site, p_site), ys, nrm),
                                 either_test=binary(np.where(g_test != nrm, g_test, p_test), yt, nrm))
            rec[mode] = r
        out.append(rec)
    del evc, trainers
    return out


def dp_epsilon(sizes, cfg):
    """Record-level (eps, delta) per client: Poisson-subsampled Gaussian, RDP accountant."""
    if cfg.get('sigma', 0) <= 0:
        return dict(epsilon=None, note='no noise: no privacy guarantee')
    import dp_accounting
    eps = []
    for n in sizes:
        q = BATCH / n
        steps = cfg['rounds'] * cfg['local_epochs'] * int(np.ceil(1.0 / q))
        acc = dp_accounting.rdp.RdpAccountant()
        acc.compose(dp_accounting.SelfComposedDpEvent(
            dp_accounting.PoissonSampledDpEvent(q, dp_accounting.GaussianDpEvent(cfg['sigma'])), steps))
        eps.append(float(acc.get_epsilon(DELTA)))
    return dict(epsilon_max=max(eps), epsilon_per_client=eps, delta=DELTA, sigma=cfg['sigma'], clip=cfg['clip'],
                sampling='Poisson, q = B / n_k per client', unit='record-level, per client')


def run_centralised(D, cfg, rep, idx=None, site_te=None):
    """Same model, optimiser and number of passes as one federated client sees in total."""
    rng = np.random.default_rng(2000 + rep)
    tf.keras.utils.set_random_seed(rep)
    nf, nc = D['Xtr'].shape[1], len(D['classes'])
    model = build_model(nf, nc, cfg.get('norm', 'gn'))
    lt = LocalTrainer(model, D['Xtr'], D['ytr'], 'fedavg', cfg['lr'])
    ev = Evaluator(model, D)
    idx = np.arange(len(D['ytr'])) if idx is None else idx
    hist, t0, best_w = [], time.time(), None
    rrng = np.random.default_rng(3000 + rep)
    for rnd in range(1, cfg['rounds'] + 1):
        lt.run(idx, cfg['local_epochs'], rng)
        h = dict(round=rnd, **{f'val_{k}': v for k, v in ev.scores(D['Xva'], D['yva']).items()},
                 **{f'test_{k}': v for k, v in ev.scores(D['Xte'], D['yte']).items()})
        if cfg.get('recal'):
            h.update(recal_scores(model, ev, D, rrng))
        hist.append(h)
        if select(hist) == len(hist) - 1:
            best_w = weights(model)
    s = select(hist)
    fin = ev.scores(D['Xte'], D['yte'], full=True)
    assign(model, best_w)
    sel = ev.scores(D['Xte'], D['yte'], full=True)
    site = ev.site_scores(site_te) if site_te is not None else None
    out = dict(cfg=cfg, rep=rep, n_train=int(len(idx)), site_selected=site, selected_round=hist[s]['round'], val_f1_selected=hist[s]['val_f1'],
               test_f1_selected=hist[s]['test_f1'], test_acc_selected=hist[s]['test_acc'],
               test_f1_final=hist[-1]['test_f1'], test_selected_full=sel, test_final_full=fin,
               history=hist, seconds=time.time() - t0)
    del model, lt, ev; tf.keras.backend.clear_session(); gc.collect()
    return out


def run_local_only(D, cfg, rep):
    prng = np.random.default_rng(1000 + rep)
    clients, site_te = make_partition(D, cfg, prng)
    res = [run_centralised(D, cfg, rep, idx=c, site_te=None if site_te is None else [site_te[k]])
           for k, c in enumerate(clients)]
    f = [r['test_f1_selected'] for r in res]
    return dict(cfg=cfg, rep=rep, client_sizes=[int(len(c)) for c in clients], per_client=res,
                test_f1_selected_mean=float(np.mean(f)), test_f1_selected_min=float(np.min(f)),
                test_f1_selected_max=float(np.max(f)),
                test_f1_final_mean=float(np.mean([r['test_f1_final'] for r in res])))


# ── experiments ───────────────────────────────────────────────────────────────

def tuned(ds):
    for name in ('tuned_v3.json', 'tuned.json'):
        p = OUT / ds / name
        if p.exists():
            return json.loads(p.read_text())
    return {}


def base_cfg(**kw):
    c = dict(algo='fedavg', alpha=0.5, n_clients=4, rounds=ROUNDS, local_epochs=LOCAL_EPOCHS, lr=LR, norm='gn')
    c.update(kw)
    return c


def experiments(ds, group):
    T = tuned(ds)
    hp = lambda algo: {TUNE_GRID[algo][0]: T[algo]} if algo in T else {}
    lr_sgd = lambda algo: T.get(algo, 0.1)
    E = {}
    if group in ('tune3a', 'tune3b', 'tunelc'):
        grids = grids3(ds, 'a' if group == 'tune3a' else 'b' if group == 'tune3b' else 'lc')
        for algo, (key, grid) in grids.items():
            for v in grid:
                c = base_cfg(algo=algo, lr=T.get('fedavg', LR) if algo in ('fedprox', 'fedadam', 'fedlc') else LR)
                if ds == 'N_BaIoT':   # round 6: N-BaIoT is tuned on its device partition (nine clients)
                    c.update(partition='device', alpha=None, n_clients=9)
                c[key] = v
                E[f'tune3_{algo}_{key}{v:g}'] = c
        return E
    if group == 'tune':
        for algo, (key, grid) in TUNE_GRID.items():
            for v in grid:
                c = base_cfg(algo=algo, **({'lr': lr_sgd(algo)} if algo in ('scaffold', 'fedavgsgd') else {}))
                c[key] = v
                if algo == 'fedprox':
                    c['lr'] = T.get('fedavg', LR)
                if algo == 'fedadam':
                    c['lr'] = T.get('fedavg', LR)
                E[f'tune_{algo}_{key}{v:g}'] = c
        return E
    lr_adam = T.get('fedavg', LR)
    if group == 'core':
        E['central'] = base_cfg(kind='central', lr=lr_adam)
        for a in (0.5, 0.1):
            E[f'local_a{a:g}'] = base_cfg(kind='local', alpha=a, lr=lr_adam)
        E['fedavg_iid'] = base_cfg(alpha=None, lr=lr_adam)
        for a in (1.0, 0.5, 0.1):
            E[f'fedavg_a{a:g}'] = base_cfg(alpha=a, lr=lr_adam)
        for a in (0.5, 0.1):
            E[f'fedprox_a{a:g}'] = base_cfg(algo='fedprox', alpha=a, lr=lr_adam, mu=T.get('fedprox', 0.01))
            E[f'fedadam_a{a:g}'] = base_cfg(algo='fedadam', alpha=a, lr=lr_adam, server_lr=T.get('fedadam', 0.1))
            E[f'scaffold_a{a:g}'] = base_cfg(algo='scaffold', alpha=a, lr=lr_sgd('scaffold'))
            E[f'fedavgsgd_a{a:g}'] = base_cfg(algo='fedavgsgd', alpha=a, lr=lr_sgd('fedavgsgd'))
    if group == 'extra':
        for n in (2, 8, 16):
            E[f'scale_N{n}'] = base_cfg(n_clients=n, lr=lr_adam)
        # normalisation ablation: BatchNorm (running statistics averaged by the server)
        for a in (0.5, 0.1):
            E[f'fedavg_bn_a{a:g}'] = base_cfg(alpha=a, lr=lr_adam, norm='bn')
        E['central_bn'] = base_cfg(kind='central', lr=lr_adam, norm='bn')
    if group == 'natural':
        nat = dict(partition='natural', alpha=None)
        E['local_nat'] = base_cfg(kind='local', lr=lr_adam, **nat)
        E['fedavg_nat'] = base_cfg(lr=lr_adam, **nat)
        E['fedprox_nat'] = base_cfg(algo='fedprox', lr=lr_adam, mu=T.get('fedprox', 0.01), **nat)
        E['fedadam_nat'] = base_cfg(algo='fedadam', lr=lr_adam, server_lr=T.get('fedadam', 0.1), **nat)
        E['scaffold_nat'] = base_cfg(algo='scaffold', lr=lr_sgd('scaffold'), **nat)
        E['fedavgsgd_nat'] = base_cfg(algo='fedavgsgd', lr=lr_sgd('fedavgsgd'), **nat)
        E['fedlc_nat'] = base_cfg(algo='fedlc', lr=lr_adam, tau=T.get('fedlc', 1.0), **nat)
    if group == 'natural_r':
        # round 5: host order redrawn in every repetition; every site also scored on its own hosts' test traffic
        nat = dict(partition='natural', alpha=None, randomized=True)
        E['local_natr'] = base_cfg(kind='local', lr=lr_adam, **nat)
        E['fedavg_natr'] = base_cfg(lr=lr_adam, **nat)
        E['fedprox_natr'] = base_cfg(algo='fedprox', lr=lr_adam, mu=T.get('fedprox', 0.01), **nat)
        E['fedadam_natr'] = base_cfg(algo='fedadam', lr=lr_adam, server_lr=T.get('fedadam', 0.1), **nat)
        E['scaffold_natr'] = base_cfg(algo='scaffold', lr=lr_sgd('scaffold'), **nat)
        E['fedavgsgd_natr'] = base_cfg(algo='fedavgsgd', lr=lr_sgd('fedavgsgd'), **nat)
        E['fedlc_natr'] = base_cfg(algo='fedlc', lr=lr_adam, tau=T.get('fedlc', 1.0), **nat)
    if group == 'device':
        # N-BaIoT: one client per device (9 clients); IID and Dirichlet references with the same N
        dev = dict(partition='device', alpha=None, n_clients=9)
        E['local_dev'] = base_cfg(kind='local', lr=lr_adam, **dev)
        E['fedavg_dev'] = base_cfg(lr=lr_adam, **dev)
        E['fedprox_dev'] = base_cfg(algo='fedprox', lr=lr_adam, mu=T.get('fedprox', 0.01), **dev)
        E['fedadam_dev'] = base_cfg(algo='fedadam', lr=lr_adam, server_lr=T.get('fedadam', 0.1), **dev)
        E['scaffold_dev'] = base_cfg(algo='scaffold', lr=lr_sgd('scaffold'), **dev)
        E['fedavgsgd_dev'] = base_cfg(algo='fedavgsgd', lr=lr_sgd('fedavgsgd'), **dev)
        E['fedlc_dev'] = base_cfg(algo='fedlc', lr=lr_adam, tau=T.get('fedlc', 1.0), **dev)
        E['fedavg_iid9'] = base_cfg(alpha=None, n_clients=9, lr=lr_adam)
        for a in (0.5, 0.1):
            E[f'fedavg_a{a:g}_N9'] = base_cfg(alpha=a, n_clients=9, lr=lr_adam)
    if group == 'personal':
        # round 6: personalisation baselines (full / low-lr / proximal fine-tuning) and the two-model combination
        modes = dict(finetune=[1, 2, 5], ft_modes=['full', 'low', 'prox'])
        if ds == 'N_BaIoT':
            E['fedavg_dev_pers'] = base_cfg(lr=lr_adam, partition='device', alpha=None, n_clients=9, **modes)
        else:
            E['fedavg_natr_pers'] = base_cfg(lr=lr_adam, partition='natural', alpha=None, randomized=True, **modes)
    if group == 'dpsmall':
        # round 6: smaller noise levels on the short DP schedule (TON-IoT, CPU)
        for sg in (0.2, 0.3):
            E[f'dp_sigma{sg:g}'] = base_cfg(algo='dpsgd', rounds=DP_ROUNDS, local_epochs=DP_LOCAL_EPOCHS, lr=lr_adam, clip=CLIP, sigma=sg)
    if group == 'natr_noports':
        # round 6: own-traffic comparison on the host partition without the port features (shortcut check)
        nat = dict(partition='natural', alpha=None, randomized=True)
        E['local_natr_noports'] = base_cfg(kind='local', lr=lr_adam, **nat)
        E['fedavg_natr_noports_ft'] = base_cfg(lr=lr_adam, finetune=[1, 2, 5], **nat)
    if group == 'finetune':
        # round 5b: does local adaptation of the federated model recover each site's own-traffic accuracy?
        if ds == 'N_BaIoT':
            E['fedavg_dev_ft'] = base_cfg(lr=lr_adam, partition='device', alpha=None, n_clients=9, finetune=[1, 2, 5])
        else:
            E['fedavg_natr_ft'] = base_cfg(lr=lr_adam, partition='natural', alpha=None, randomized=True, finetune=[1, 2, 5])
    if group == 'tunecentral':
        for v in CENTRAL_LR:
            E[f'tunec_central_lr{v:g}'] = base_cfg(kind='central', lr=v)
    if group == 'portsalg':
        # round 5: algorithm comparison without the port features (TON-IoT, alpha = 0.5)
        E['fedprox_a0.5_noports'] = base_cfg(algo='fedprox', lr=lr_adam, mu=T.get('fedprox', 0.01))
        E['fedadam_a0.5_noports'] = base_cfg(algo='fedadam', lr=lr_adam, server_lr=T.get('fedadam', 0.1))
        E['fedlc_a0.5_noports'] = base_cfg(algo='fedlc', lr=lr_adam, tau=T.get('fedlc', 1.0))
        E['scaffold_a0.5_noports'] = base_cfg(algo='scaffold', lr=lr_sgd('scaffold'))
        E['fedavgsgd_a0.5_noports'] = base_cfg(algo='fedavgsgd', lr=lr_sgd('fedavgsgd'))
    if group == 'dptune2':
        d = dict(algo='dpsgd', rounds=DP_ROUNDS, local_epochs=DP_LOCAL_EPOCHS, clip=4.0, sigma=1.0, lr=3e-3)
        E['dpt_C4'] = base_cfg(**d)
    if group == 'bnfed':
        # round 5: BatchNorm with federated re-estimation of its statistics (deployable recalibration)
        for a in (0.5, 0.1):
            E[f'fedavg_bn_a{a:g}_fedrecal'] = base_cfg(alpha=a, lr=lr_adam, norm='bn', recal='fed')
        E['central_bn_recal'] = base_cfg(kind='central', lr=lr_adam, norm='bn', recal=True)
    if group == 'fedlc':
        for a in (0.5, 0.1):
            E[f'fedlc_a{a:g}'] = base_cfg(algo='fedlc', alpha=a, lr=lr_adam, tau=T.get('fedlc', 1.0))
    if group == 'long':
        E['central_r100'] = base_cfg(kind='central', lr=lr_adam, rounds=100)
        E['fedavg_iid_r100'] = base_cfg(alpha=None, lr=lr_adam, rounds=100)
        for a in (0.5, 0.1):
            E[f'fedavg_a{a:g}_r100'] = base_cfg(alpha=a, lr=lr_adam, rounds=100)
    if group == 'followup2':
        for a in (0.5, 0.1):
            E[f'scaffold_a{a:g}_lr0.01'] = base_cfg(algo='scaffold', alpha=a, lr=0.01)
    if group == 'dpfull':
        E['dp_full_sigma1.8'] = base_cfg(algo='dpsgd', lr=lr_adam, clip=CLIP, sigma=DP_FULL_SIGMA)
        E['dp_full_nonoise'] = base_cfg(algo='dpsgd', lr=lr_adam, clip=0.0, sigma=0.0)
    if group == 'ports':
        E['central_noports'] = base_cfg(kind='central', lr=lr_adam)
        E['fedavg_iid_noports'] = base_cfg(alpha=None, lr=lr_adam)
        for a in (0.5, 0.1):
            E[f'fedavg_a{a:g}_noports'] = base_cfg(alpha=a, lr=lr_adam)
    if group == 'bnrecal':
        E['central_bn_recal'] = base_cfg(kind='central', lr=lr_adam, norm='bn', recal=True)
        E['fedavg_bn_a0.5_recal'] = base_cfg(alpha=0.5, lr=lr_adam, norm='bn', recal=True)
    if group == 'dptune':
        # small DP grid at sigma = 1 (TON-IoT; CPU): clip norm and lr, and a 40-round schedule
        d = dict(algo='dpsgd', rounds=DP_ROUNDS, local_epochs=DP_LOCAL_EPOCHS, clip=CLIP, sigma=1.0, lr=3e-3)
        E['dpt_C0.5'] = base_cfg(**{**d, 'clip': 0.5})
        E['dpt_C2'] = base_cfg(**{**d, 'clip': 2.0})
        E['dpt_lr0.001'] = base_cfg(**{**d, 'lr': 1e-3})
        E['dpt_lr0.006'] = base_cfg(**{**d, 'lr': 6e-3})
        E['dpt_r40_sigma1'] = base_cfg(**{**d, 'rounds': 40})
        E['dpt_r40_nonoise'] = base_cfg(**{**d, 'rounds': 40, 'clip': 0.0, 'sigma': 0.0})
    if group == 'followup':
        # SCAFFOLD at alpha = 0.1 with a smaller lr (its lr was tuned at alpha = 0.5)
        E['scaffold_a0.1_lr0.1'] = base_cfg(algo='scaffold', alpha=0.1, lr=0.1)
    if group == 'dp':
        dpc = dict(algo='dpsgd', lr=lr_adam, rounds=DP_ROUNDS, local_epochs=DP_LOCAL_EPOCHS)
        E['dp_noclip_nonoise'] = base_cfg(**dpc, clip=0.0, sigma=0.0)
        E['dp_clip_nonoise'] = base_cfg(**dpc, clip=CLIP, sigma=0.0)
        for s in DP_SIGMAS:
            E[f'dp_sigma{s:g}'] = base_cfg(**dpc, clip=CLIP, sigma=s)
        # same short schedule without DP machinery (BatchNorm model, minibatch Adam) for reference
        E['fedavg_short'] = base_cfg(lr=lr_adam, rounds=DP_ROUNDS, local_epochs=DP_LOCAL_EPOCHS, norm='bn')
        E['fedavg_gn_short'] = base_cfg(lr=lr_adam, rounds=DP_ROUNDS, local_epochs=DP_LOCAL_EPOCHS, norm='gn')
    return E


def finalize_tuning(ds):
    """Pick each algorithm's hyperparameter by mean validation F1 (seed 0, alpha = 0.5)."""
    res = {}
    for algo, (key, grid) in TUNE_GRID.items():
        sc = {}
        for v in grid:
            p = OUT / ds / f'tune_{algo}_{key}{v:g}' / 'seed0.json'
            if p.exists():
                sc[v] = json.loads(p.read_text())['val_f1_selected']
        if sc:
            res[algo] = max(sc, key=sc.get)
            res[f'{algo}_scores'] = {f'{k:g}': s for k, s in sc.items()}
    (OUT / ds / 'tuned.json').write_text(json.dumps(res, indent=2))
    print('tuned', res)
    return res


def finalize_tuning3(ds, stage):
    """Pick hyperparameters by mean validation F1 over the two tuning repetitions; write tuned_v3.json."""
    p = OUT / ds / 'tuned_v3.json'
    res = json.loads(p.read_text()) if p.exists() else {}
    grids = grids3(ds, stage)
    seeds = NB_TUNE_SEEDS if ds == 'N_BaIoT' else TUNE_SEEDS
    for algo, (key, grid) in grids.items():
        sc = {}
        for v in grid:
            vals = [json.loads((OUT / ds / f'tune3_{algo}_{key}{v:g}' / f'seed{r}.json').read_text())['val_f1_selected']
                    for r in seeds if (OUT / ds / f'tune3_{algo}_{key}{v:g}' / f'seed{r}.json').exists()]
            if vals:
                sc[v] = float(np.mean(vals))
        if sc:
            res[algo] = max(sc, key=sc.get)
            res[f'{algo}_scores'] = {f'{k:g}': s for k, s in sc.items()}
            res[f'{algo}_edge'] = res[algo] in (min(grid), max(grid))
    p.write_text(json.dumps(res, indent=2))
    print('tuned_v3', {k: v for k, v in res.items() if not k.endswith('_scores')})
    return res


def prepare_rerun(ds):
    """Move core/extra/followup results whose hyperparameters changed under round-3 tuning aside."""
    import shutil
    old = json.loads((OUT / ds / 'tuned.json').read_text())
    new = json.loads((OUT / ds / 'tuned_v3.json').read_text())
    changed = {a for a in ('fedavg', 'fedprox', 'fedadam', 'scaffold', 'fedavgsgd') if old.get(a) != new.get(a)}
    changed |= {'fedprox', 'fedadam'}          # tuned at the wrong client lr in round 2
    if 'fedavg' in changed:                    # client lr used by every Adam-based run
        changed |= {'central', 'local', 'scale', 'fedavg_bn', 'central_bn', 'fedprox', 'fedadam'}
    dest = OUT / '_superseded_r2' / ds
    moved = []
    for d in sorted((OUT / ds).iterdir()):
        n = d.name
        if not d.is_dir() or n.startswith(('tune', 'dp', 'fedavg_short', 'fedavg_gn_short')):
            continue
        hit = (any(n.startswith(c + '_') or n == c for c in changed) or
               ('fedavg' in changed and n.startswith('fedavg_') and not n.startswith('fedavgsgd')))
        if hit:
            dest.mkdir(parents=True, exist_ok=True); shutil.move(str(d), str(dest / n)); moved.append(n)
    print('changed', sorted(changed), 'moved', moved)


def code_version():
    """git commit of the code that produced a result (with '-dirty' if run_fl_v2.py has uncommitted changes)."""
    import subprocess
    try:
        h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=HERE, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--', str(Path(__file__).name)], cwd=HERE,
                               capture_output=True, text=True).stdout.strip()
        return h + ('-dirty' if dirty else '')
    except Exception:
        return 'unknown'


CODE_VERSION = code_version()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['TON_IoT', 'UNSW_NB15', 'N_BaIoT'])
    ap.add_argument('--group', required=True, choices=['tune', 'core', 'extra', 'dp', 'followup', 'finalize_tuning', 'tune3a', 'tune3b',
                                         'finalize3a', 'finalize3b', 'prepare_rerun', 'ports', 'bnrecal', 'dptune',
                                         'natural', 'natural_r', 'device', 'bnfed', 'tunecentral', 'portsalg', 'dptune2', 'fedlc', 'tunelc', 'finalizelc', 'long', 'followup2', 'dpfull', 'finetune', 'natr_noports', 'personal', 'dpsmall'])
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--only', default=None, help='comma-separated experiment names')
    ap.add_argument('--smoke', action='store_true', help='2 rounds, results_v2_smoke/')
    ap.add_argument('--cpu', action='store_true', help='hide the GPU (used for the DP-SGD group)')
    a = ap.parse_args()
    global OUT
    if a.smoke:
        OUT = HERE / 'results_v2_smoke'
    if a.group in ('tune', 'finalize_tuning'):
        # the round-2 single-stage tuning tuned FedProx/FedAdam at a mismatched client lr; superseded by tune3a/tune3b
        print('superseded: use tune3a/finalize3a, tune3b/finalize3b and tunelc/finalizelc'); return
    if a.group in ('finalize3a', 'finalize3b'):
        finalize_tuning3(a.dataset, a.group[-1]); return
    if a.group == 'finalizelc':
        finalize_tuning3(a.dataset, 'lc'); return
    if a.group in ('natural', 'natural_r') and a.dataset != 'TON_IoT':
        print('natural partition needs IP addresses (TON-IoT only)'); return
    if a.group == 'prepare_rerun':
        prepare_rerun(a.dataset); return
    D = load_data(a.dataset, drop=PORTS if a.group in ('ports', 'portsalg', 'natr_noports') else (), meta=a.group in ('natural', 'natural_r', 'finetune', 'natr_noports', 'personal'))
    print(f"===== {a.dataset}: train {len(D['ytr'])} val {len(D['yva'])} test {len(D['yte'])} features {D['Xtr'].shape[1]}", flush=True)
    E = experiments(a.dataset, a.group)
    if a.only:
        E = {k: v for k, v in E.items() if k in a.only.split(',')}
    if a.smoke:
        E = {k: {**v, 'rounds': 2} for k, v in E.items()}
    for rep in [int(s) for s in a.seeds.split(',')]:
        for name, cfg in E.items():
            path = OUT / a.dataset / name / f'seed{rep}.json'
            if path.exists():
                old = json.loads(path.read_text())['cfg']
                if {k: v for k, v in old.items() if k != 'keep_best'} == cfg:
                    print(f'  [{name} seed{rep}] cached'); continue
                # configuration changed since the cached run (e.g. a re-tuned value): set it aside and re-run
                stale = OUT / '_stale' / a.dataset / name; stale.mkdir(parents=True, exist_ok=True)
                path.replace(stale / path.name)
                print(f'  [{name} seed{rep}] cached with a different cfg {old} -> re-run', flush=True)
            print(f'  [{name} seed{rep}] running {cfg}', flush=True)
            t0 = time.time()
            kind = cfg.get('kind')
            r = run_centralised(D, cfg, rep) if kind == 'central' else run_local_only(D, cfg, rep) if kind == 'local' \
                else run_federated(D, {**cfg, 'keep_best': True}, rep)
            r['code_version'] = CODE_VERSION
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(r, indent=1))
            sel = r.get('test_f1_selected', r.get('test_f1_selected_mean'))
            print(f'  [{name} seed{rep}] done in {time.time() - t0:.0f}s  test F1 (val-selected) {sel:.4f}', flush=True)


if __name__ == '__main__':
    main()
