"""Paper 2 v2: evasion of compact IoT intrusion detectors under a realistic (problem-space) threat model.

Design: Revision_v2/Paper2/DESIGN.md. Data: the corrected pipeline of Papers 1 and 4-6 (TON-IoT, UNSW-NB15;
experiments.paper6_multimodal.run_p6_v2.load, split 0, keep_raw=True).

Attacker = source of a malicious flow; goal = the flow is classified 'normal'. It can only ADD source bytes and
packets, delay (duration, ACK) and, on UNSW-NB15, choose its initial TTL. Derived features are recomputed
(feature_map), then the deployed pre-processing (min-max; quantile transform for networks) is applied.

Groups:
  base    : undefended CNN, MLP, RF, GBDT (5 seeds) + all attacks
  defend  : CNN with AT-all / CCAT / AT-threat / TRADES, MLP with AT-threat, trees with adversarial
            augmentation (5 seeds) + all attacks
Results: results_v2/<ds>/<model>_<defence>/seed<k>.json (resumable).
Run from `Papers pipeline/`: python -m experiments.paper2_adv_v2.run_adv_v2 --dataset D --group G
"""
import argparse, gc, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.paper6_multimodal import run_p6_v2 as P6  # noqa: E402
from experiments.baseline import run_p1_v2 as P1          # noqa: E402

OUT = ROOT / 'experiments' / 'paper2_adv_v2' / 'results_v2'
DATASETS = ('TON_IoT', 'UNSW_NB15')
SEEDS = (0, 1, 2, 3, 4)
DEF_SEEDS = (0, 1, 2)     # defended models: 3 seeds (each adversarially trained CNN takes ~45 min on the laptop GPU)
BETAS = (0.1, 0.25, 0.5, 1.0, 2.0)
EPS_UNC = (0.01, 0.05, 0.1)
N_EVAL = 5000                 # attack test records attacked (stratified, fixed across models)
N_TTL = 4                     # TTL candidates: most frequent initial TTLs of benign training flows (UNSW)
BETA_TRAIN, EPS_TRAIN = 0.5, 0.05
RARE_SHARE = 0.02             # CCAT: classes with < 2% of the (pre-SMOTE) training records are trained clean
TREE_CFG = {'rf': ('rf', dict(n_estimators=100, max_depth=16, min_samples_leaf=1)),
            'gb': ('gb', dict(max_leaf_nodes=31, learning_rate=0.05))}   # 0.1 collapsed on some seeds (Paper 1); TON base gb rerun in p1fix


# ── problem-space feature map ─────────────────────────────────────────────────

class Space:
    """Holds what the feature map needs for one dataset: raw column indices, selected columns, scaler, budgets."""

    def __init__(self, D):
        r = D['raw']
        self.ds, self.names, self.sel = D['ds'], r['names'], D['feat']
        self.sel_idx = np.array([self.names.index(n) for n in self.sel])
        self.mm_min = r['scaler'].data_min_.astype(np.float64)
        self.mm_rng = np.where(r['scaler'].data_range_ > 0, r['scaler'].data_range_, 1.0).astype(np.float64)
        self.qt = D['qt']
        c = lambda n: self.names.index(n)
        s = lambda n: self.sel.index(n) if n in self.sel else None
        if self.ds == 'TON_IoT':
            self.k = 3      # bytes, packets, delay
            self.raw = dict(b=c('src_bytes'), p=c('src_pkts'), d=c('duration'), ipb=c('src_ip_bytes'))
            self.out = dict(b=s('src_bytes'), p=s('src_pkts'), d=s('duration'), ipb=s('src_ip_bytes'))
            self.floors = np.array([1000.0, 10.0, 1.0])
        else:
            self.k = 4      # bytes, packets, delay, ACK delay
            self.raw = dict(b=c('sbytes'), p=c('spkts'), d=c('dur'), load=c('sload'), mean=c('smean'),
                            ack=c('ackdat'), ttl=c('sttl'), dttl=c('dttl'), cst=c('ct_state_ttl'))
            self.out = dict(b=s('sbytes'), load=s('sload'), mean=s('smean'), ack=s('ackdat'), ttl=s('sttl'), cst=s('ct_state_ttl'))
            self.floors = np.array([1000.0, 10.0, 1.0, 0.1])
            self.state_cols = [i for i, n in enumerate(self.names) if n.startswith('state_')]
            R, y = r['Rtr'], D['ytr']
            # ct_state_ttl as a function of (sttl, dttl, state) in the training data (mode)
            key = self._key(R[:, c('sttl')], R)
            from collections import Counter, defaultdict
            tab = defaultdict(Counter)
            for k_, v in zip(key, R[:, c('ct_state_ttl')]):
                tab[k_][v] += 1
            self.cst_tab = {k_: cnt.most_common(1)[0][0] for k_, cnt in tab.items()}
            ben = R[y == D['normal'], c('sttl')]
            vals, cnt = np.unique(ben, return_counts=True)
            self.ttl_cands = vals[np.argsort(-cnt)][:N_TTL].astype(np.float64)
            m = (R[:, c('dur')] > 0) & (R[:, c('sbytes')] > 0)
            self.load_ratio = float(np.median(R[m, c('sload')] / (8 * R[m, c('sbytes')] / R[m, c('dur')])))
        assert all(v is not None for v in self.out.values()), f'touched feature not selected: {self.out}'

    def _key(self, sttl, R):
        st_ = R[:, self.state_cols].argmax(1) if self.state_cols else np.zeros(len(R), int)
        return list(zip(np.round(sttl, 3), np.round(R[:, self.names.index('dttl')], 3), st_))

    def caps(self, R, beta):
        rr = self.raw
        base = [R[:, rr['b']], R[:, rr['p']], R[:, rr['d']]] + ([R[:, rr['ack']]] if self.k == 4 else [])
        return beta * (np.stack(base, 1) + self.floors)

    def ttl_columns(self, R, ttl):
        """(sttl', ct_state_ttl') for TTL choice `ttl` (None = unchanged)"""
        rr = self.raw
        if ttl is None:
            return R[:, rr['ttl']].copy(), R[:, rr['cst']].copy()
        t = np.full(len(R), ttl)
        cst = np.array([self.cst_tab.get(k_, v) for k_, v in zip(self._key(t, R), R[:, rr['cst']])])
        return t, cst

    def selected(self, R, A, ttlc=None, xp=np):
        """raw selected-feature matrix after actions A (n x k, absolute units). xp = numpy or tensorflow."""
        rr, oo = self.raw, self.out
        tf_ = xp is not np
        if tf_:
            import tensorflow as tf
            R = tf.convert_to_tensor(R, A.dtype)
            col = lambda j: R[:, j]
            mx, wh = tf.maximum, tf.where
        else:
            col = lambda j: R[:, j]
            mx, wh = np.maximum, np.where
        S = {i: col(int(j)) for i, j in enumerate(self.sel_idx)}
        b, p, d = col(rr['b']), col(rr['p']), col(rr['d'])
        ab, ap, ad = A[:, 0], A[:, 1], A[:, 2]
        if self.ds == 'TON_IoT':
            ipb = col(rr['ipb'])
            h = wh(p > 0, (ipb - b) / mx(p, 1.0), 40.0)
            h = mx(20.0 + 0 * h, h); h = -mx(-60.0 + 0 * h, -h)          # clip to [20, 60]
            S[oo['b']], S[oo['p']], S[oo['d']] = b + ab, p + ap, d + ad
            S[oo['ipb']] = ipb + ab + h * ap
        else:
            ak = A[:, 3]
            load, ack = col(rr['load']), col(rr['ack'])
            b2, p2, d2 = b + ab, p + ap, d + ad + ak
            S[oo['b']], S[oo['ack']] = b2, ack + ak
            S[oo['mean']] = b2 / mx(p2, 1.0)
            ratio_b = wh(b > 0, b2 / mx(b, 1e-9), 1.0 + 0 * b)
            ratio_d = wh(d2 > 0, d / mx(d2, 1e-9), 1.0 + 0 * d)
            scaled = load * ratio_b * ratio_d
            fresh = load + 8.0 * ab * self.load_ratio / mx(d2, 1e-3)
            S[oo['load']] = wh((b > 0) & (d > 0), scaled, fresh)
            if ttlc is not None:
                t, cst = ttlc
                if tf_:
                    t, cst = tf.convert_to_tensor(t, A.dtype), tf.convert_to_tensor(cst, A.dtype)
                S[oo['ttl']], S[oo['cst']] = t, cst
        cols = [S[i] for i in range(len(self.sel))]
        if tf_:
            return tf.stack(cols, 1)
        return np.stack(cols, 1)

    def minmax(self, S, xp=np):
        return (S - self.mm_min) / self.mm_rng

    # quantile transform in TensorFlow (same formula as sklearn: average of forward and backward interpolation)
    def tf_quantile(self, X):
        import tensorflow as tf
        q = tf.constant(self.qt.quantiles_, X.dtype); ref = tf.constant(self.qt.references_, X.dtype)
        outs = []
        for j in range(X.shape[1]):
            x, xp_ = X[:, j], q[:, j]
            f = 0.5 * (_tf_interp(x, xp_, ref) - _tf_interp(-x, -tf.reverse(xp_, [0]), -tf.reverse(ref, [0])))
            f = tf.where(tf.equal(x, xp_[0]), tf.zeros_like(f), f)      # sklearn 1.7: exact bounds map to 0 / 1
            f = tf.where(tf.equal(x, xp_[-1]), tf.ones_like(f), f)
            outs.append(f)
        return tf.stack(outs, 1)


def _tf_interp(x, xp, fp):
    import tensorflow as tf
    m = tf.shape(xp)[0]
    i = tf.clip_by_value(tf.searchsorted(xp, x, side='right'), 1, m - 1)
    x0, x1 = tf.gather(xp, i - 1), tf.gather(xp, i)
    f0, f1 = tf.gather(fp, i - 1), tf.gather(fp, i)
    w = tf.math.divide_no_nan(x - x0, x1 - x0)
    y = f0 + tf.clip_by_value(w, 0.0, 1.0) * (f1 - f0)
    y = tf.where(x < xp[0], fp[0] + 0 * y, y)
    return tf.where(x > xp[-1], fp[-1] + 0 * y, y)


# ── models ────────────────────────────────────────────────────────────────────

class Model:
    """uniform predict_proba on min-max scaled selected features"""

    def __init__(self, kind, obj, space):
        self.kind, self.obj, self.sp = kind, obj, space

    def proba(self, MM):
        if self.kind in ('cnn', 'mlp'):
            Q = self.sp.qt.transform(MM).astype(np.float32)
            return self.obj.predict(Q[..., None], batch_size=8192, verbose=0)
        P = self.obj.predict_proba(MM)
        out = np.zeros((len(MM), self.n_classes)); out[:, self.obj.classes_] = P
        return out


def train_net(D, kind, seed, defence='none', sp=None, rare=()):
    import tensorflow as tf
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    nc, nf = len(D['classes']), len(D['feat'])
    m = P1.build_net(kind, 1, nf, nc)
    opt = tf.keras.optimizers.Adam(1e-3)
    Xv, yv = D['Qva'][..., None], D['yva']
    if defence == 'none':
        m.compile(opt, 'sparse_categorical_crossentropy')
        bw = P6.BestWeights()
        h = m.fit(D['Qfit'][..., None], D['yfit'], validation_data=(Xv, yv), epochs=P1.EPOCHS, batch_size=P1.BATCH,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=P1.PATIENCE, monitor='val_loss'), bw,
                             tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5)], verbose=0)
        return m, dict(epochs=len(h.history['loss']), best_epoch=bw.epoch)
    return train_robust(D, m, opt, defence, sp, rare, seed)


ROBUST = dict(epochs=30, steps=200, patience=5, bs=256)


def train_robust(D, m, opt, defence, sp, rare, seed):
    """custom loop: clean SMOTE batch + adversarial part (see DESIGN.md)"""
    import tensorflow as tf
    rng = np.random.default_rng(seed)
    ce = tf.keras.losses.SparseCategoricalCrossentropy()
    normal = D['normal']
    Xfit_mm, Qfit, yfit = D['Xfit'], D['Qfit'], D['yfit']
    # real (pre-SMOTE) training attack records for problem-space adversarial examples
    R, ytr = D['raw']['Rtr'], D['ytr']
    att = np.where(ytr != normal)[0]
    if defence == 'ccat':
        keep = ~np.isin(yfit, list(rare))
    else:
        keep = np.ones(len(yfit), bool)
    adv_rows = np.where(keep)[0]
    best, best_w, wait = np.inf, None, 0
    qtf = sp.tf_quantile
    epochs, steps, patience, bs = ROBUST['epochs'], ROBUST['steps'], ROBUST['patience'], ROBUST['bs']

    @tf.function
    def step_ce(xq, yb, xa_q, ya):
        with tf.GradientTape() as tape:
            loss = 0.5 * ce(yb, m(xq, training=True)) + 0.5 * ce(ya, m(xa_q, training=True))
        opt.apply_gradients(zip(tape.gradient(loss, m.trainable_variables), m.trainable_variables))

    @tf.function
    def step_trades(xq, yb, cq, aq):
        with tf.GradientTape() as tape:
            pc, pa = m(cq, training=True), m(aq, training=True)
            loss = ce(yb, m(xq, training=True)) + 6.0 * tf.reduce_mean(
                tf.reduce_sum(pc * (tf.math.log(pc + 1e-12) - tf.math.log(pa + 1e-12)), 1))
        opt.apply_gradients(zip(tape.gradient(loss, m.trainable_variables), m.trainable_variables))

    for ep in range(epochs):
        for _ in range(steps):
            i = rng.choice(len(yfit), bs, replace=False)
            xq, yb = tf.constant(Qfit[i][..., None]), tf.constant(yfit[i])
            if defence in ('at_all', 'ccat'):
                j = rng.choice(adv_rows, bs // 2, replace=False)
                xa = _pgd_feature(m, sp, Xfit_mm[j], yfit[j], EPS_TRAIN, 7, targeted=False)
                xa_q, ya = qtf(xa)[..., None], tf.constant(yfit[j])
            else:     # at_threat / trades: action-space attack on real training attack records
                j = rng.choice(att, bs // 2, replace=False)
                Rj, cap = R[j].astype(np.float32), sp.caps(R[j], BETA_TRAIN).astype(np.float32)
                ttlc = None
                if sp.ds == 'UNSW_NB15':
                    t = rng.choice(list(sp.ttl_cands) + [None]) if rng.random() < 0.5 else None
                    ttlc = sp.ttl_columns(R[j], t)
                ya = tf.constant(ytr[j])
                if defence == 'trades':
                    clean_q = qtf(tf.constant(sp.minmax(sp.selected(Rj, np.zeros((len(j), sp.k), np.float32), ttlc=None)), tf.float32))
                    A = _pgd_action(m, sp, Rj, cap, ttlc, target=None, steps=7, ref_probs=m(clean_q[..., None], training=False).numpy(),
                                    seed=int(rng.integers(1 << 30)))
                else:
                    A = _pgd_action(m, sp, Rj, cap, ttlc, target=normal, steps=7, seed=int(rng.integers(1 << 30)))
                xa_q = qtf(tf.constant(sp.minmax(sp.selected(Rj, A, ttlc)), tf.float32))[..., None]
            if defence == 'trades':
                step_trades(xq, yb, clean_q[..., None], xa_q)
            else:
                step_ce(xq, yb, xa_q, ya)
        vl = float(ce(D['yva'], m.predict(D['Qva'][..., None], batch_size=8192, verbose=0)))
        if vl < best:
            best, best_w, wait, best_ep = vl, m.get_weights(), 0, ep + 1
        else:
            wait += 1
            if wait >= patience:
                break
    m.set_weights(best_w)
    return m, dict(epochs=ep + 1, best_epoch=best_ep, steps_per_epoch=steps, val_loss_best=best)


def train_tree(D, name, seed, augment=None):
    kind, params = TREE_CFG[name]
    X, y = D['Xfit'], D['yfit']
    if augment is not None:
        X, y = np.vstack([X, augment[0]]), np.r_[y, augment[1]]
    t = P1.make_tree(kind, params, seed).fit(X, y)
    return t


# ── attacks ───────────────────────────────────────────────────────────────────

_GRADS = {}


def _grad_fns(m, sp):
    """compiled gradient functions for one model (feature-space and action-space)"""
    key = id(m)
    if key in _GRADS:
        return _GRADS[key]
    import tensorflow as tf
    mn, rg = tf.constant(sp.mm_min, tf.float32), tf.constant(sp.mm_rng, tf.float32)
    unsw = sp.ds == 'UNSW_NB15'

    @tf.function(reduce_retracing=True)
    def feat(x, aux, targeted):
        # aux: one-hot of the true class (untargeted: ascend CE) or of the target class (targeted: ascend log p_t)
        with tf.GradientTape() as tape:
            tape.watch(x)
            p = m(sp.tf_quantile(x)[..., None], training=False)
            s = tf.reduce_sum(aux * tf.math.log(p + 1e-12), 1)
            loss = tf.reduce_mean(s) if targeted else -tf.reduce_mean(s)
        return tape.gradient(loss, x)

    @tf.function(reduce_retracing=True)
    def action(u, Rt, capt, t, cst, aux, kl):
        # aux: one-hot target (ascend log p_target) or reference probabilities (ascend KL(ref || p))
        with tf.GradientTape() as tape:
            tape.watch(u)
            S = sp.selected(Rt, u * capt, (t, cst) if unsw else None, xp=tf)
            p = m(sp.tf_quantile((S - mn) / rg)[..., None], training=False)
            if kl:
                loss = tf.reduce_mean(tf.reduce_sum(aux * (tf.math.log(aux + 1e-12) - tf.math.log(p + 1e-12)), 1))
            else:
                loss = tf.reduce_mean(tf.reduce_sum(aux * tf.math.log(p + 1e-12), 1))
        return tape.gradient(loss, u)

    _GRADS.clear()
    _GRADS[key] = (feat, action)
    return _GRADS[key]


def _pgd_feature(m, sp, x0, y, eps, steps, targeted, normal=None, nc=None):
    """L-inf PGD in min-max feature space (all selected features), clipped to [0, 1]"""
    import tensorflow as tf
    feat, _ = _grad_fns(m, sp)
    nc = nc or m.output_shape[-1]
    lab = np.full(len(y), normal) if targeted else np.asarray(y)
    aux = tf.constant(np.eye(nc, dtype=np.float32)[lab])
    alpha = eps / 4 if steps <= 10 else eps / 10
    x0 = tf.constant(x0, tf.float32)
    x = tf.clip_by_value(x0 + tf.random.uniform(tf.shape(x0), -eps, eps), 0.0, 1.0)
    for _ in range(steps):
        g = feat(x, aux, targeted)
        x = tf.clip_by_value(tf.clip_by_value(x + alpha * tf.sign(g), x0 - eps, x0 + eps), 0.0, 1.0)
    return x


def _pgd_action(m, sp, R, cap, ttlc, target, steps=50, ref_probs=None, seed=0):
    """sign-gradient ascent on u in [0,1]^k, actions A = u * cap. target = class to reach (normal) or None:
    maximise KL to ref_probs (TRADES inner problem). ttlc: (sttl', ct_state_ttl') for UNSW (None = unchanged)."""
    import tensorflow as tf
    _, action = _grad_fns(m, sp)
    n = len(R)
    if sp.ds == 'UNSW_NB15':
        t, cst = ttlc if ttlc is not None else sp.ttl_columns(R, None)
    else:
        t = cst = np.zeros(n)
    t, cst = tf.constant(np.asarray(t, np.float32)), tf.constant(np.asarray(cst, np.float32))
    if target is not None:
        aux, kl = tf.constant(np.eye(m.output_shape[-1], dtype=np.float32)[np.full(n, target)]), False
    else:
        aux, kl = tf.constant(np.asarray(ref_probs, np.float32)), True
    rng = np.random.default_rng(seed)
    u = tf.constant(rng.uniform(0, 1, cap.shape).astype(np.float32))
    capt, Rt = tf.constant(cap, tf.float32), tf.constant(R, tf.float32)
    for _ in range(steps):
        g = action(u, Rt, capt, t, cst, aux, kl)
        u = tf.clip_by_value(u + 0.05 * tf.sign(g), 0.0, 1.0)
    return (u * capt).numpy()


def attack_set(D, sp, rng_seed=0):
    """fixed stratified sample of attack test records (<= N_EVAL), same for every model"""
    y = D['yte']; att = np.where(y != D['normal'])[0]
    rng = np.random.default_rng(rng_seed)
    per = max(1, N_EVAL // (len(D['classes']) - 1))
    idx = []
    for c in np.unique(y[att]):
        ii = att[y[att] == c]
        idx.append(rng.choice(ii, min(per, len(ii)), replace=False))
    idx = np.concatenate(idx)
    if len(idx) < N_EVAL:
        rest = np.setdiff1d(att, idx)
        idx = np.r_[idx, rng.choice(rest, min(N_EVAL - len(idx), len(rest)), replace=False)]
    return np.sort(idx)


def candidates(sp, R, beta, n_rand, seed):
    """action candidates per record: box corners + uniform random points (n x K x k)"""
    import itertools
    cap = sp.caps(R, beta)
    corners = np.array(list(itertools.product((0.0, 1.0), repeat=sp.k)))
    U = np.random.default_rng(seed).uniform(0, 1, (n_rand, sp.k))
    G = np.vstack([corners, U])                        # K x k
    return G[None, :, :] * cap[:, None, :]              # n x K x k


def evaluate_attacks(D, sp, model, idx, seed, white_box=None, surrogate=None):
    """ASR of black-box search (volume-only, TTL-only, both), white-box action PGD and unconstrained PGD."""
    normal = D['normal']
    R = D['raw']['Rte'][idx].astype(np.float64)
    y = D['yte'][idx]
    MM0 = sp.minmax(sp.selected(R, np.zeros((len(R), sp.k))))
    det = model.proba(MM0).argmax(1) != normal
    out = dict(n_attacked=int(len(idx)), n_detected=int(det.sum()))
    ttl_opts = [None] + (list(sp.ttl_cands) if sp.ds == 'UNSW_NB15' else [])
    per_class = lambda ev: {D['classes'][c]: float(ev[det & (y == c)].mean()) for c in np.unique(y[det])}
    for beta in BETAS:
        A = candidates(sp, R, beta, 120 if sp.ds == 'TON_IoT' else 48, seed)
        ev_vol = np.zeros(len(R), bool); ev_both = np.zeros(len(R), bool); ev_ttl = np.zeros(len(R), bool)
        for t in ttl_opts:
            ttlc = sp.ttl_columns(R, t) if sp.ds == 'UNSW_NB15' else None
            n, K, k = A.shape
            Rk = np.repeat(R, K, 0); Ak = A.reshape(-1, k)
            tc = None if ttlc is None else (np.repeat(ttlc[0], K), np.repeat(ttlc[1], K))
            pred = model.proba(sp.minmax(sp.selected(Rk, Ak, tc))).argmax(1).reshape(n, K)
            ok = (pred == normal).any(1)
            ev_both |= ok
            if t is None:
                ev_vol |= ok
            else:   # TTL only: zero actions with this TTL
                ev_ttl |= pred[:, 0] == normal
        out[f'bb_both_b{beta}'] = float(ev_both[det].mean())
        out[f'bb_volume_b{beta}'] = float(ev_vol[det].mean())
        if sp.ds == 'UNSW_NB15':
            out['bb_ttl_only'] = float(ev_ttl[det].mean())
        if beta == 0.5:
            out['bb_both_b0.5_per_class'] = per_class(ev_both)
        if white_box is not None:
            ev = np.zeros(len(R), bool)
            for t in ttl_opts:
                ttlc = sp.ttl_columns(R, t) if sp.ds == 'UNSW_NB15' else None
                Aw = np.vstack([_pgd_action(white_box, sp, R[s:s + 1024].astype(np.float32), sp.caps(R[s:s + 1024], beta).astype(np.float32),
                                            None if ttlc is None else (ttlc[0][s:s + 1024], ttlc[1][s:s + 1024]), normal, 50, seed=seed)
                                for s in range(0, len(R), 1024)])
                ev |= model.proba(sp.minmax(sp.selected(R, Aw, ttlc))).argmax(1) == normal
            out[f'wb_action_b{beta}'] = float(ev[det].mean())
    if surrogate is not None:      # transfer: action PGD on the undefended CNN of the same seed
        for beta in (0.5, 1.0):
            if beta not in BETAS:
                continue
            ev = np.zeros(len(R), bool)
            for t in ttl_opts:
                ttlc = sp.ttl_columns(R, t) if sp.ds == 'UNSW_NB15' else None
                Aw = np.vstack([_pgd_action(surrogate, sp, R[s:s + 1024].astype(np.float32), sp.caps(R[s:s + 1024], beta).astype(np.float32),
                                            None if ttlc is None else (ttlc[0][s:s + 1024], ttlc[1][s:s + 1024]), normal, 50, seed=seed)
                                for s in range(0, len(R), 1024)])
                ev |= model.proba(sp.minmax(sp.selected(R, Aw, ttlc))).argmax(1) == normal
            out[f'transfer_action_b{beta}'] = float(ev[det].mean())
    if white_box is not None:
        import tensorflow as tf
        for eps in EPS_UNC:
            xs = []
            for s in range(0, len(R), 1024):
                xs.append(_pgd_feature(white_box, sp, np.clip(MM0[s:s + 1024], 0, 1), y[s:s + 1024], eps, 50, True, normal).numpy())
            ev = model.proba(np.vstack(xs)).argmax(1) == normal
            out[f'wb_unconstrained_e{eps}'] = float(ev[det].mean())
    return out


# ── driver ────────────────────────────────────────────────────────────────────

def rare_classes(D):
    cnt = np.bincount(D['ytr'], minlength=len(D['classes']))
    return {c for c in range(len(cnt)) if c != D['normal'] and cnt[c] < RARE_SHARE * cnt.sum()}


AUG_N = 20000


def augmentation(D, sp, model, seed, n=None):
    n = n or AUG_N
    """black-box-search adversarials of real training attack records (beta_train), true labels"""
    R, y = D['raw']['Rtr'], D['ytr']
    att = np.where(y != D['normal'])[0]
    rng = np.random.default_rng(seed)
    j = rng.choice(att, min(n, len(att)), replace=False)
    Rj = R[j].astype(np.float64)
    A = candidates(sp, Rj, BETA_TRAIN, 24, seed)
    n_, K, k = A.shape
    ttlc = None
    if sp.ds == 'UNSW_NB15':
        # per record, as in AT-threat: TTL unchanged with probability 0.5, otherwise one of the benign TTL candidates
        # (2026-11-02 fix: one TTL for all records never produced volume-only examples)
        tch = np.where(rng.random(n_) < 0.5, -1, rng.integers(0, len(sp.ttl_cands), n_))
        t_col, c_col = sp.ttl_columns(Rj, None)
        for ci in np.unique(tch[tch >= 0]):
            msk = tch == ci
            t_col[msk], c_col[msk] = sp.ttl_columns(Rj[msk], float(sp.ttl_cands[ci]))
        ttlc = (t_col, c_col)
    tc = None if ttlc is None else (np.repeat(ttlc[0], K), np.repeat(ttlc[1], K))
    MM = sp.minmax(sp.selected(np.repeat(Rj, K, 0), A.reshape(-1, k), tc))
    P = model.proba(MM)[:, D['normal']].reshape(n_, K)
    best = P.argmax(1)                                  # the most 'normal-looking' candidate per record
    return MM.reshape(n_, K, -1)[np.arange(n_), best].astype(np.float32), y[j]


def run(D, sp, name, defence, seed, idx, surrogate=None):
    import tensorflow as tf
    t0 = time.time()
    rare = rare_classes(D)
    if name in ('cnn', 'mlp'):
        net, info = train_net(D, name, seed, defence, sp, rare)
        model = Model(name, net, sp); wb = net
    else:
        aug = None
        if defence == 'augment':
            base = Model(name, train_tree(D, name, seed), sp); base.n_classes = len(D['classes'])
            aug = augmentation(D, sp, base, seed)
        t = train_tree(D, name, seed, aug)
        model = Model(name, t, sp); model.n_classes = len(D['classes']); wb, info = None, {}
    P = model.proba(np.clip(D['Xte'], -1e9, 1e9))
    res = dict(train_s=time.time() - t0, **info, **P6.metrics(D, P.argmax(1), P), rare_classes=[D['classes'][c] for c in rare])
    res.update(evaluate_attacks(D, sp, model, idx, seed, white_box=wb, surrogate=surrogate))
    return res, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='all', choices=('all',) + DATASETS)
    ap.add_argument('--group', default='base', choices=('base', 'defend'))
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    global OUT, SEEDS, BETAS, EPS_UNC, N_EVAL, AUG_N
    if a.smoke:
        OUT, SEEDS, BETAS, EPS_UNC, N_EVAL = OUT.parent / 'results_v2_smoke', (0,), (0.5,), (0.05,), 300
        P1.EPOCHS = 2
        ROBUST.update(epochs=1, steps=5)
        AUG_N = 500
    import tensorflow as tf
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    plan = {'base': [(m, 'none') for m in ('cnn', 'mlp', 'rf', 'gb')],
            'defend': [('cnn', d) for d in ('at_all', 'ccat', 'at_threat', 'trades')] + [('mlp', 'at_threat'), ('rf', 'augment'), ('gb', 'augment')]}[a.group]
    for ds in DATASETS if a.dataset == 'all' else (a.dataset,):
        D = P6.load(ds, keep_raw=True)
        sp = Space(D)
        idx = attack_set(D, sp)
        # sanity: the feature map with zero actions reproduces the pipeline's min-max features exactly
        MM0 = sp.minmax(sp.selected(D['raw']['Rte'][idx].astype(np.float64), np.zeros((len(idx), sp.k))))
        err = np.abs(MM0 - D['Xte'][idx]).max()
        assert err < 1e-3, f'feature map mismatch {err}'   # float32 rounding of large raw values
        (OUT / ds).mkdir(parents=True, exist_ok=True)
        (OUT / ds / 'attack_set.json').write_text(json.dumps(dict(idx=idx.tolist(), ttl_cands=getattr(sp, 'ttl_cands', np.array([])).tolist(),
                                                                  rare=[D['classes'][c] for c in rare_classes(D)], map_err=float(err),
                                                                  n_features=len(D['feat']))))
        print(f'== {ds}: {len(D["feat"])} features, attack set {len(idx)}, map error {err:.1e}', flush=True)
        for seed in (SEEDS if a.group == 'base' else tuple(x for x in DEF_SEEDS if x in SEEDS)):
            surrogate = None
            for name, defence in plan:
                p = OUT / ds / f'{name}_{defence}' / f'seed{seed}.json'
                if p.exists():
                    continue
                if name != 'cnn' and defence == 'none' and surrogate is None:
                    # cnn_none of this seed was saved in an earlier process: retrain it (same seed) as the transfer surrogate
                    surrogate, _ = train_net(D, 'cnn', seed, 'none', sp, rare_classes(D))
                    print(f'  (retrained cnn_none seed{seed} as transfer surrogate)', flush=True)
                res, model = run(D, sp, name, defence, seed, idx, surrogate if name != 'cnn' else None)
                if name == 'cnn' and defence == 'none':
                    surrogate = model.obj
                res['cfg'] = dict(model=name, defence=defence, seed=seed, beta_train=BETA_TRAIN, eps_train=EPS_TRAIN)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(res, indent=1))
                print(f'  {name}_{defence} seed{seed}: F1 {res["macro_f1"]:.4f}  bb@0.5 {res.get("bb_both_b0.5", float("nan")):.3f}  '
                      f'({res["train_s"]:.0f}s)', flush=True)
                if model.obj is not surrogate:
                    del model
                gc.collect()
        del D; gc.collect()


if __name__ == '__main__':
    main()
