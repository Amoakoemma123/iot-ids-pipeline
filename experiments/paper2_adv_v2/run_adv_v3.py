"""Paper 2 v3: evasion of compact IoT intrusion detectors under a realistic (problem-space) threat model.

v3 = v2 (run_adv_v2.py) revised after review round 1 (Revision_v2/Paper2/review_r1_*.md):
  feature map   - UNSW: added packets add header bytes (sbytes counts headers; min 28 B/packet), dload recomputed
                  when the flow is lengthened, ACK delay only for flows with an observed handshake (ackdat > 0);
                  both datasets: a delay needs at least two packets after the actions; TTL 0 is not a candidate;
                  ct_state_ttl for an unseen TTL = value of the nearest training TTL with the same (dttl, state).
  evaluation    - candidate grids drawn with seeds disjoint from the augmentation grids; dense grids (saturation);
                  score-based adaptive search (fixed TTL candidates and any TTL 1-255); TTL sweep 1-255;
                  strong white-box PGD (margin loss, 4 restarts incl. box corners, 100 steps, decaying step);
                  per-action ablation; budget-floor sensitivity; queries to first success; minimal successful action;
                  black-box feature-space search on every model; feature-space PGD restricted to mutable features;
                  per-flow outcomes saved (bootstrap CIs).
  training      - models saved; AT-threat/TRADES choose the TTL per flow (p = 0.5 unchanged); clean-loop control
                  (same custom loop, adversarial half replaced by clean attack records); second augmentation round;
                  augmentation without delay actions (held-out action); 5 seeds for the key defences;
                  data resplits 1-2; UNSW detectors without the TTL features.
round-2 additions (review_r2_*.md; existing groups and results unchanged):
  BPDA quantile transform (Space.tf_quantile_bpda: exact forward, smoothed-slope backward) for feature-space PGD;
  defence variants at_all_e<eps>, at_threat_b<beta> (one-config check), at_all_bpda (eps-ball training with a
  BPDA inner maximiser), augment_nobytes (held-out bytes); groups more_seeds, r2_nets, r2_trees; TTL-free MLP.
  Evaluation of saved models for round 2: run_adv_r2.py.
Results: results_v3/<ds>/<tag>/seed<k>.json; models: models_v3/<ds>/<tag>/seed<k>.(keras|joblib).
Run from `Papers pipeline/`: python -m experiments.paper2_adv_v2.run_adv_v3 --dataset D --group G
"""
import argparse, base64, gc, itertools, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.paper6_multimodal import run_p6_v2 as P6  # noqa: E402
from experiments.baseline import run_p1_v2 as P1          # noqa: E402

HERE = ROOT / 'experiments' / 'paper2_adv_v2'
OUT, MOD = HERE / 'results_v3', HERE / 'models_v3'
DATASETS = ('TON_IoT', 'UNSW_NB15')
SEEDS = (0, 1, 2, 3, 4)
DEF_SEEDS = (0, 1, 2)
BETAS = (0.1, 0.25, 0.5, 1.0, 2.0)
EPS_UNC = (0.01, 0.05, 0.1)
N_EVAL = 5000
N_TTL = 4
BETA_TRAIN, EPS_TRAIN = 0.5, 0.05
RARE_SHARE = 0.02
AUG_N = 20000
TREE_CFG = {'rf': ('rf', dict(n_estimators=100, max_depth=16, min_samples_leaf=1)),
            'gb': ('gb', dict(max_leaf_nodes=31, learning_rate=0.05))}
# seed offsets: evaluation grids and augmentation grids never share random directions
EVAL_OFF, DENSE_OFF, AUG_OFF, AUG2_OFF, ADA_OFF = 1000, 2000, 500, 600, 3000
TTL_FEATS = ('sttl', 'ct_state_ttl')


# ── problem-space feature map ─────────────────────────────────────────────────

class Space:
    def __init__(self, D):
        r = D['raw']
        self.ds, self.names, self.sel = D['ds'], r['names'], D['feat']
        self.sel_idx = np.array([self.names.index(n) for n in self.sel])
        full_sel = r.get('full_sel', self.sel)
        fi = [full_sel.index(n) for n in self.sel]
        self.mm_min = r['scaler'].data_min_.astype(np.float64)[fi]
        rng_ = r['scaler'].data_range_[fi]
        self.mm_rng = np.where(rng_ > 0, rng_, 1.0).astype(np.float64)
        self.qt = D['qt']
        self.floor_scale = 1.0
        self.reply = 0.0          # final review round 2: header bytes of one destination reply per added packet (0 = off)
        c = lambda n: self.names.index(n)
        s = lambda n: self.sel.index(n) if n in self.sel else None
        if self.ds == 'TON_IoT':
            self.k = 3
            self.raw = dict(b=c('src_bytes'), p=c('src_pkts'), d=c('duration'), ipb=c('src_ip_bytes'))
            self.out = dict(b=s('src_bytes'), p=s('src_pkts'), d=s('duration'), ipb=s('src_ip_bytes'))
            self.floors = np.array([1000.0, 10.0, 1.0])
            self.has_ttl = False
            self.mutable = [self.out[x] for x in ('b', 'p', 'd', 'ipb')]
            self.reply_raw = dict(dp=c('dst_pkts'), dib=c('dst_ip_bytes'))
            self.reply_out = dict(dp=s('dst_pkts'), dib=s('dst_ip_bytes'))
        else:
            self.k = 4
            self.raw = dict(b=c('sbytes'), p=c('spkts'), d=c('dur'), load=c('sload'), mean=c('smean'), ack=c('ackdat'),
                            ttl=c('sttl'), dttl=c('dttl'), cst=c('ct_state_ttl'), dload=c('dload'))
            self.out = dict(b=s('sbytes'), load=s('sload'), mean=s('smean'), ack=s('ackdat'), ttl=s('sttl'),
                            cst=s('ct_state_ttl'), dload=s('dload'))
            self.floors = np.array([1000.0, 10.0, 1.0, 0.1])
            self.has_ttl = self.out['ttl'] is not None
            self.mutable = [self.out[x] for x in ('b', 'load', 'mean', 'ack', 'dload') if self.out[x] is not None]
            self.reply_raw = dict(db=c('dbytes'))
            self.reply_out = dict(db=s('dbytes'))
            self.state_cols = [i for i, n in enumerate(self.names) if n.startswith('state_')]
            R, y = r['Rtr'], D['ytr']
            tab = defaultdict(Counter)
            for k_, v in zip(self._key(R[:, c('sttl')], R), R[:, c('ct_state_ttl')]):
                tab[k_][v] += 1
            self.cst_tab = {k_: cnt.most_common(1)[0][0] for k_, cnt in tab.items()}
            self.cst_near = defaultdict(list)
            for (t, dt, st), v in self.cst_tab.items():
                self.cst_near[(dt, st)].append((t, v))
            self._cache = {}
            ben = R[y == D['normal'], c('sttl')]
            vals, cnt = np.unique(ben, return_counts=True)
            ok = (vals >= 1) & (vals <= 255)                # TTL 0 cannot be sent
            vals, cnt = vals[ok], cnt[ok]
            self.ttl_cands = vals[np.argsort(-cnt)][:N_TTL].astype(np.float64)
            m = (R[:, c('dur')] > 0) & (R[:, c('sbytes')] > 0)
            self.load_ratio = float(np.median(R[m, c('sload')] / (8 * R[m, c('sbytes')] / R[m, c('dur')])))
        need = ('b', 'load', 'mean', 'ack') if self.ds == 'UNSW_NB15' else ('b', 'p', 'd', 'ipb')
        assert all(self.out[x] is not None for x in need), f'touched feature not selected: {self.out}'

    def _key(self, sttl, R):
        st_ = R[:, self.state_cols].argmax(1) if self.state_cols else np.zeros(len(R), int)
        return list(zip(np.round(sttl, 3), np.round(R[:, self.names.index('dttl')], 3), st_))

    def _cst_one(self, t, dt, st):
        k_ = (t, dt, st)
        if k_ in self._cache:
            return self._cache[k_]
        v = self.cst_tab.get(k_)
        if v is None:
            near = self.cst_near.get((dt, st))
            v = min(near, key=lambda e: abs(e[0] - t))[1] if near else np.nan
        self._cache[k_] = v
        return v

    def caps(self, R, beta):
        rr = self.raw
        base = [R[:, rr['b']], R[:, rr['p']], R[:, rr['d']]] + ([R[:, rr['ack']]] if self.k == 4 else [])
        return beta * (np.stack(base, 1) + self.floor_scale * self.floors)

    def ttl_columns(self, R, ttl):
        """(sttl', ct_state_ttl') for TTL choice `ttl`: None (unchanged), a scalar, or a per-row array (nan = unchanged)"""
        rr = self.raw
        t0, c0 = R[:, rr['ttl']].copy(), R[:, rr['cst']].copy()
        if ttl is None:
            return t0, c0
        t = np.broadcast_to(np.asarray(ttl, np.float64), (len(R),)).copy()
        keep = np.isnan(t)
        t[keep] = t0[keep]
        st_ = R[:, self.state_cols].argmax(1) if self.state_cols else np.zeros(len(R), int)
        keys = np.stack([np.round(t, 3), np.round(R[:, rr['dttl']], 3), st_], 1)
        uk, inv = np.unique(keys, axis=0, return_inverse=True)
        vals = np.array([self._cst_one(float(a), float(b), int(c_)) for a, b, c_ in uk], np.float64)
        cst = vals[np.asarray(inv).reshape(-1)]
        cst = np.where(np.isnan(cst) | keep, c0, cst)
        return t, cst

    def selected(self, R, A, ttlc=None, xp=np):
        rr, oo = self.raw, self.out
        tf_ = xp is not np
        if tf_:
            import tensorflow as tf
            R = tf.convert_to_tensor(R, A.dtype)
            mx, wh, mn_ = tf.maximum, tf.where, tf.minimum
        else:
            mx, wh, mn_ = np.maximum, np.where, np.minimum
        col = lambda j: R[:, j]
        S = {i: col(int(j)) for i, j in enumerate(self.sel_idx)}
        b, p, d = col(rr['b']), col(rr['p']), col(rr['d'])
        ab, ap, ad = A[:, 0], A[:, 1], A[:, 2]
        zero = 0 * ad
        ad = wh(p + ap >= 2, ad, zero)                    # a delay needs >= 2 packets
        if self.ds == 'TON_IoT':
            ipb = col(rr['ipb'])
            h = wh(p > 0, (ipb - b) / mx(p, 1.0), 40.0 + zero)
            h = mn_(mx(h, 20.0 + zero), 60.0 + zero)
            S[oo['b']], S[oo['p']], S[oo['d']] = b + ab, p + ap, d + ad
            S[oo['ipb']] = ipb + ab + h * ap
            if self.reply:
                ro, rw = self.reply_out, self.reply_raw
                if ro['dp'] is not None:
                    S[ro['dp']] = col(rw['dp']) + ap
                if ro['dib'] is not None:
                    S[ro['dib']] = col(rw['dib']) + self.reply * ap
        else:
            ack = col(rr['ack'])
            ak = wh(ack > 0, A[:, 3], zero)               # ACK delay only with an observed handshake
            h = wh(p > 0, mn_(mx(b / mx(p, 1.0), 28.0 + zero), 40.0 + zero), 40.0 + zero)   # header bytes per packet
            load = col(rr['load'])
            b2, p2, d2 = b + ab + h * ap, p + ap, d + ad + ak
            S[oo['b']], S[oo['ack']] = b2, ack + ak
            S[oo['mean']] = b2 / mx(p2, 1.0)
            ratio_b = wh(b > 0, b2 / mx(b, 1e-9), 1.0 + zero)
            ratio_d = wh(d2 > 0, d / mx(d2, 1e-9), 1.0 + zero)
            fresh = load + 8.0 * (b2 - b) * self.load_ratio / mx(d2, 1e-3)
            S[oo['load']] = wh((b > 0) & (d > 0), load * ratio_b * ratio_d, fresh)
            if oo['dload'] is not None:
                dl = col(rr['dload'])
                S[oo['dload']] = wh((d > 0) & (d2 > 0), dl * d / mx(d2, 1e-9), dl)
            if self.reply:
                db = col(self.reply_raw['db'])
                db2 = db + self.reply * ap
                if self.reply_out['db'] is not None:
                    S[self.reply_out['db']] = db2
                if oo['dload'] is not None:      # dload scales with destination bytes as well as duration
                    S[oo['dload']] = wh(db > 0, S[oo['dload']] * db2 / mx(db, 1e-9),
                                        wh(d2 > 0, 8.0 * db2 * self.load_ratio / mx(d2, 1e-3), S[oo['dload']]))
            if ttlc is not None and self.has_ttl:
                t, cst = ttlc
                if tf_:
                    t, cst = tf.convert_to_tensor(t, A.dtype), tf.convert_to_tensor(cst, A.dtype)
                S[oo['ttl']], S[oo['cst']] = t, cst
        cols = [S[i] for i in range(len(self.sel))]
        return tf.stack(cols, 1) if tf_ else np.stack(cols, 1)

    def minmax(self, S):
        return (S - self.mm_min) / self.mm_rng

    def tf_quantile(self, X):
        import tensorflow as tf
        q = tf.constant(self.qt.quantiles_, X.dtype); ref = tf.constant(self.qt.references_, X.dtype)
        outs = []
        for j in range(X.shape[1]):
            x, xp_ = X[:, j], q[:, j]
            f = 0.5 * (_tf_interp(x, xp_, ref) - _tf_interp(-x, -tf.reverse(xp_, [0]), -tf.reverse(ref, [0])))
            f = tf.where(tf.equal(x, xp_[0]), tf.zeros_like(f), f)
            f = tf.where(tf.equal(x, xp_[-1]), tf.ones_like(f), f)
            outs.append(f)
        return tf.stack(outs, 1)

    def tf_quantile_bpda(self, X):
        """quantile transform with a BPDA backward pass: exact forward; backward = slope of the transform over a window
        of +-25 quantile knots at the nearest knot (non-zero outside the reference range and on flat segments)"""
        import tensorflow as tf
        if not hasattr(self, '_bpda_slope'):
            q, ref = self.qt.quantiles_.astype(np.float64), self.qt.references_.astype(np.float64)
            nq, nf = q.shape
            S = np.zeros((nf, nq))
            for j in range(nf):
                rngj = q[-1, j] - q[0, j]
                for k in range(nq):
                    w = 25
                    while True:
                        lo, hi = max(k - w, 0), min(k + w, nq - 1)
                        if q[hi, j] > q[lo, j] or (lo == 0 and hi == nq - 1):
                            break
                        w *= 2
                    S[j, k] = (ref[hi] - ref[lo]) / (q[hi, j] - q[lo, j]) if q[hi, j] > q[lo, j] else (1.0 if rngj <= 0 else 1.0 / rngj)
            self._bpda_slope, self._bpda_q = S, q.T.copy()
        slope = tf.constant(self._bpda_slope, X.dtype)
        qT = tf.constant(self._bpda_q, X.dtype)
        nq = self._bpda_q.shape[1]

        @tf.custom_gradient
        def f(x):
            y = self.tf_quantile(x)

            def grad(dy):
                i = tf.clip_by_value(tf.searchsorted(qT, tf.transpose(x)), 0, nq - 1)
                return dy * tf.transpose(tf.gather(slope, i, batch_dims=1))
            return y, grad
        return f(X)


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


def drop_ttl(D):
    """UNSW detectors without sttl and ct_state_ttl (same split, scaling and SMOTE; quantile transform refitted)"""
    keep = [i for i, n in enumerate(D['feat']) if n not in TTL_FEATS]
    E = dict(D)
    E['raw'] = dict(D['raw']); E['raw']['full_sel'] = list(D['feat'])
    E['feat'] = [D['feat'][i] for i in keep]
    for k in ('Xtr', 'Xfit', 'Xva', 'Xte'):
        if k in D:
            E[k] = D[k][:, keep]
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(n_quantiles=1000, subsample=200000, random_state=0).fit(E['Xfit'])
    for k in ('Xfit', 'Xva', 'Xte'):
        E['Q' + k[1:]] = qt.transform(E[k]).astype(np.float32)
    E['qt'] = qt
    return E


# ── models ────────────────────────────────────────────────────────────────────

class Model:
    def __init__(self, kind, obj, space, n_classes):
        self.kind, self.obj, self.sp, self.n_classes = kind, obj, space, n_classes

    def proba(self, MM):
        outs = []
        for s in range(0, len(MM), 400000):
            X = MM[s:s + 400000]
            if self.kind in ('cnn', 'mlp'):
                Q = self.sp.qt.transform(X).astype(np.float32)
                outs.append(self.obj.predict(Q[..., None], batch_size=16384, verbose=0))
            else:
                P = self.obj.predict_proba(X)
                o = np.zeros((len(X), self.n_classes)); o[:, self.obj.classes_] = P
                outs.append(o)
        return np.vstack(outs) if outs else np.zeros((0, self.n_classes))


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


def per_flow_ttl(sp, rng, n):
    """per flow: TTL unchanged with probability 0.5 (nan), otherwise one of the TTL candidates"""
    t = np.full(n, np.nan)
    m = rng.random(n) >= 0.5
    t[m] = rng.choice(sp.ttl_cands, m.sum())
    return t


def train_robust(D, m, opt, defence, sp, rare, seed):
    """custom loop: clean SMOTE batch + a second part (adversarial, or clean real attack records for 'clean_loop')"""
    import tensorflow as tf
    rng = np.random.default_rng(seed)
    ce = tf.keras.losses.SparseCategoricalCrossentropy()
    normal = D['normal']
    Xfit_mm, Qfit, yfit = D['Xfit'], D['Qfit'], D['yfit']
    R, ytr = D['raw']['Rtr'], D['ytr']
    att = np.where(ytr != normal)[0]
    eps, beta_tr, bpda = EPS_TRAIN, BETA_TRAIN, False     # round-2 variants: at_all_e<eps>, at_threat_b<beta>, at_all_bpda
    if defence.startswith('at_all_e'):
        defence, eps = 'at_all', float(defence[len('at_all_e'):])
    elif defence.startswith('at_threat_b'):
        defence, beta_tr = 'at_threat', float(defence[len('at_threat_b'):])
    elif defence == 'at_all_bpda':
        defence, bpda = 'at_all', True
    corner_start = False                     # final review round 2: at_threat_cs = AT-threat whose inner attack starts
    if defence == 'at_threat_cs':            # each flow at the most benign-scoring corner of the budget box
        defence, corner_start = 'at_threat', True
    keep =~np.isin(yfit, list(rare)) if defence == 'ccat' else np.ones(len(yfit), bool)
    adv_rows = np.where(keep)[0]
    best, best_w, wait, best_ep = np.inf, None, 0, 0
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

    ep = 0
    for ep in range(epochs):
        for _ in range(steps):
            i = rng.choice(len(yfit), bs, replace=False)
            xq, yb = tf.constant(Qfit[i][..., None]), tf.constant(yfit[i])
            if defence in ('at_all', 'ccat'):
                j = rng.choice(adv_rows, bs // 2, replace=False)
                xa = _pgd_feature(m, sp, Xfit_mm[j], yfit[j], eps, 7, targeted=False, bpda=bpda)
                xa_q, ya = qtf(xa)[..., None], tf.constant(yfit[j])
            else:
                j = rng.choice(att, bs // 2, replace=False)
                Rj = R[j].astype(np.float32)
                ya = tf.constant(ytr[j])
                if defence == 'clean_loop':
                    Z = np.zeros((len(j), sp.k), np.float32)
                    xa_q = qtf(tf.constant(sp.minmax(sp.selected(Rj, Z)), tf.float32))[..., None]
                else:
                    cap = sp.caps(R[j], beta_tr).astype(np.float32)
                    ttlc = sp.ttl_columns(R[j], per_flow_ttl(sp, rng, len(j))) if sp.has_ttl else None
                    if defence == 'trades':
                        Z = np.zeros((len(j), sp.k), np.float32)
                        clean_q = qtf(tf.constant(sp.minmax(sp.selected(Rj, Z)), tf.float32))
                        A = _pgd_action(m, sp, Rj, cap, ttlc, target=None, steps=7,
                                        ref_probs=m(clean_q[..., None], training=False).numpy(), seed=int(rng.integers(1 << 30)))
                    else:
                        u0 = _corner_start(m, sp, Rj, cap, ttlc, normal, qtf, rng) if corner_start else None
                        A = _pgd_action(m, sp, Rj, cap, ttlc, target=normal, steps=7, seed=int(rng.integers(1 << 30)), u0=u0)
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
    return m, dict(epochs=ep + 1, best_epoch=best_ep, steps_per_epoch=steps, val_loss_best=best,
                   eps_train_used=eps, beta_train_used=beta_tr, bpda=bpda, corner_start=corner_start)


def _corner_start(m, sp, R, cap, ttlc, normal, qtf, rng, n_rand=8):
    """inner-attack start for at_threat_cs: every corner of the unit budget box plus n_rand random points (the
    candidates of the black-box grid search and of tree augmentation), scored by the current network; each flow
    starts at the candidate with the highest benign probability"""
    import tensorflow as tf
    n, k = cap.shape
    G = np.vstack([np.array(list(itertools.product((0.0, 1.0), repeat=k))), rng.uniform(0, 1, (n_rand, k))]).astype(np.float32)
    # all candidates scored in one batched call (rows candidate-major); every step is row-wise, so this equals the
    # earlier one-call-per-candidate loop, and argmax keeps the first maximum as the loop's strict '>' did
    nG = len(G)
    Rt = np.tile(R, (nG, 1))
    A = (np.repeat(G, n, axis=0) * np.tile(cap, (nG, 1))).astype(np.float32)
    tt = None if ttlc is None else tuple(np.tile(np.asarray(c), nG) for c in ttlc)
    p = m(qtf(tf.constant(sp.minmax(sp.selected(Rt, A, tt)), tf.float32))[..., None], training=False).numpy()[:, normal]
    return G[p.reshape(nG, n).argmax(0)].astype(np.float32)


def train_tree(D, name, seed, augment=()):
    kind, params = TREE_CFG[name]
    X, y = D['Xfit'], D['yfit']
    for Xa, ya in augment:
        X, y = np.vstack([X, Xa]), np.r_[y, ya]
    return P1.make_tree(kind, params, seed).fit(X, y)


# ── gradient attacks ──────────────────────────────────────────────────────────

_GRADS = {}


def _grad_fns(m, sp):
    key = id(m)
    if key in _GRADS:
        return _GRADS[key]
    import tensorflow as tf
    mn, rg = tf.constant(sp.mm_min, tf.float32), tf.constant(sp.mm_rng, tf.float32)
    unsw = sp.has_ttl

    @tf.function(reduce_retracing=True)
    def feat(x, aux, targeted):
        with tf.GradientTape() as tape:
            tape.watch(x)
            p = m(sp.tf_quantile(x)[..., None], training=False)
            s = tf.reduce_sum(aux * tf.math.log(p + 1e-12), 1)
            loss = tf.reduce_mean(s) if targeted else -tf.reduce_mean(s)
        return tape.gradient(loss, x)

    @tf.function(reduce_retracing=True)
    def action(u, Rt, capt, t, cst, aux, mode):
        # mode 0: ascend log p_target; 1: ascend KL(ref || p); 2: margin log p_target - max_other log p
        with tf.GradientTape() as tape:
            tape.watch(u)
            S = sp.selected(Rt, u * capt, (t, cst) if unsw else None, xp=tf)
            p = m(sp.tf_quantile((S - mn) / rg)[..., None], training=False)
            lp = tf.math.log(p + 1e-12)
            if mode == 1:
                loss = tf.reduce_mean(tf.reduce_sum(aux * (tf.math.log(aux + 1e-12) - lp), 1))
            elif mode == 2:
                tgt = tf.reduce_sum(aux * lp, 1)
                other = tf.reduce_max(lp - 1e9 * aux, 1)
                loss = tf.reduce_mean(tgt - other)
            else:
                loss = tf.reduce_mean(tf.reduce_sum(aux * lp, 1))
        return tape.gradient(loss, u)

    _GRADS.clear()
    _GRADS[key] = (feat, action)
    return _GRADS[key]


_GRADS_B = {}


def _bpda_fns(m, sp):
    """feature-space gradients through the BPDA quantile transform: cross-entropy (training) and margin (evaluation)"""
    key = id(m)
    if key in _GRADS_B:
        return _GRADS_B[key]
    import tensorflow as tf

    @tf.function(reduce_retracing=True)
    def ce(x, aux, targeted):
        with tf.GradientTape() as tape:
            tape.watch(x)
            p = m(sp.tf_quantile_bpda(x)[..., None], training=False)
            s = tf.reduce_sum(aux * tf.math.log(p + 1e-12), 1)
            loss = tf.reduce_mean(s) if targeted else -tf.reduce_mean(s)
        return tape.gradient(loss, x)

    @tf.function(reduce_retracing=True)
    def margin(x, aux):
        with tf.GradientTape() as tape:
            tape.watch(x)
            lp = tf.math.log(m(sp.tf_quantile_bpda(x)[..., None], training=False) + 1e-12)
            loss = tf.reduce_mean(tf.reduce_sum(aux * lp, 1) - tf.reduce_max(lp - 1e9 * aux, 1))
        return tape.gradient(loss, x)

    _GRADS_B.clear()
    _GRADS_B[key] = (ce, margin)
    return _GRADS_B[key]


def _pgd_feature_strong(m, model, sp, x0, eps, normal, mask, seed, steps=50):
    """strong L-inf PGD in min-max space: margin loss through the BPDA transform, restarts from x0, a random point and
    a random corner of the ball, step 0.25 eps decaying to 0.025 eps; success checked with the exact model every
    25 steps. Returns per-flow success."""
    import tensorflow as tf
    _, margin = _bpda_fns(m, sp)
    nc = m.output_shape[-1]
    aux = tf.constant(np.eye(nc, dtype=np.float32)[np.full(len(x0), normal)])
    mk = np.ones_like(x0) if mask is None else np.broadcast_to(mask, x0.shape).astype(np.float64)
    rng = np.random.default_rng(seed)
    lo, hi = np.clip(x0 - eps * mk, 0, 1), np.clip(x0 + eps * mk, 0, 1)
    starts = [x0, np.clip(x0 + mk * rng.uniform(-eps, eps, x0.shape), 0, 1),
              np.clip(x0 + mk * eps * rng.choice((-1.0, 1.0), x0.shape), 0, 1)]
    ok = np.zeros(len(x0), bool)
    lo_t, hi_t, mk_t = (tf.constant(v.astype(np.float32)) for v in (lo, hi, mk))
    for xs in starts:
        x = tf.constant(xs.astype(np.float32))
        for s in range(steps):
            a = eps * 0.25 * 0.1 ** (s / (steps - 1))
            x = tf.clip_by_value(x + mk_t * a * tf.sign(margin(x, aux)), lo_t, hi_t)
            if (s + 1) % 25 == 0:
                ok |= model.proba(x.numpy().astype(np.float64)).argmax(1) == normal
    return ok


def _pgd_feature(m, sp, x0, y, eps, steps, targeted, normal=None, mask=None, bpda=False):
    """L-inf PGD in min-max feature space, clipped to [0, 1]; mask = features allowed to change (None = all)"""
    import tensorflow as tf
    feat = _bpda_fns(m, sp)[0] if bpda else _grad_fns(m, sp)[0]
    nc = m.output_shape[-1]
    lab = np.full(len(y), normal) if targeted else np.asarray(y)
    aux = tf.constant(np.eye(nc, dtype=np.float32)[lab])
    alpha = eps / 4 if steps <= 10 else eps / 10
    x0 = tf.constant(x0, tf.float32)
    mk = tf.ones_like(x0) if mask is None else tf.constant(np.broadcast_to(mask, x0.shape).astype(np.float32))
    x = tf.clip_by_value(x0 + mk * tf.random.uniform(tf.shape(x0), -eps, eps), 0.0, 1.0)
    for _ in range(steps):
        g = feat(x, aux, targeted)
        x = tf.clip_by_value(tf.clip_by_value(x + mk * alpha * tf.sign(g), x0 - eps, x0 + eps), 0.0, 1.0)
    return x


def _ttl_tensors(sp, R, ttlc):
    import tensorflow as tf
    if sp.has_ttl:
        t, cst = ttlc if ttlc is not None else sp.ttl_columns(R, None)
    else:
        t = cst = np.zeros(len(R))
    return tf.constant(np.asarray(t, np.float32)), tf.constant(np.asarray(cst, np.float32))


def _pgd_action(m, sp, R, cap, ttlc, target, steps=50, ref_probs=None, seed=0, u0=None):
    """v2 action PGD: one random start (or the given unit start u0), sign steps of 0.05 of the box (kept for
    continuity and for training)"""
    import tensorflow as tf
    _, action = _grad_fns(m, sp)
    n = len(R)
    t, cst = _ttl_tensors(sp, R, ttlc)
    if target is not None:
        aux, mode = tf.constant(np.eye(m.output_shape[-1], dtype=np.float32)[np.full(n, target)]), 0
    else:
        aux, mode = tf.constant(np.asarray(ref_probs, np.float32)), 1
    u = tf.constant(np.random.default_rng(seed).uniform(0, 1, cap.shape).astype(np.float32) if u0 is None
                    else np.asarray(u0, np.float32))
    capt, Rt = tf.constant(cap, tf.float32), tf.constant(R, tf.float32)
    for _ in range(steps):
        u = tf.clip_by_value(u + 0.05 * tf.sign(action(u, Rt, capt, t, cst, aux, mode)), 0.0, 1.0)
    return (u * capt).numpy()


def _pgd_action_strong(m, model, sp, R, cap, ttlc, normal, seed, steps=100):
    """margin loss, restarts from u = 0, u = 1 and two random points, step 0.1 decaying to 0.01;
    returns per-flow success (any restart, checked every 25 steps)"""
    import tensorflow as tf
    _, action = _grad_fns(m, sp)
    n = len(R)
    t, cst = _ttl_tensors(sp, R, ttlc)
    aux = tf.constant(np.eye(m.output_shape[-1], dtype=np.float32)[np.full(n, normal)])
    capt, Rt = tf.constant(cap, tf.float32), tf.constant(R, tf.float32)
    rng = np.random.default_rng(seed)
    starts = [np.zeros(cap.shape), np.ones(cap.shape), rng.uniform(0, 1, cap.shape), rng.uniform(0, 1, cap.shape)]
    ok = np.zeros(n, bool)
    for u0 in starts:
        u = tf.constant(u0.astype(np.float32))
        for s in range(steps):
            a = 0.1 * 0.1 ** (s / (steps - 1))
            u = tf.clip_by_value(u + a * tf.sign(action(u, Rt, capt, t, cst, aux, 2)), 0.0, 1.0)
            if (s + 1) % 25 == 0:
                ok |= model.proba(sp.minmax(sp.selected(R, (u * capt).numpy(), ttlc))).argmax(1) == normal
    return ok


# ── black-box searches ────────────────────────────────────────────────────────

def unit_grid(k, n_rand, seed, drop=()):
    G = np.vstack([np.array(list(itertools.product((0.0, 1.0), repeat=k))), np.random.default_rng(seed).uniform(0, 1, (n_rand, k))])
    if drop:
        G[:, list(drop)] = 0.0
        G = np.unique(G, axis=0)
    return G


def search(sp, model, R, G, beta, ttl_list, normal, chunk=None):
    """exhaustive evaluation of the candidate set G (K x k unit actions, scaled per flow) x TTL options.
    Returns ok[n, T] (any candidate with that TTL option evades), first query index (T-major order, inf if none)
    and the cheapest successful volume-and-timing candidate (min over max(u), TTL unchanged)."""
    n, K, T = len(R), len(G), len(ttl_list)
    chunk = chunk or max(1, 600000 // max(1, K))
    ok = np.zeros((n, T), bool); first = np.full(n, np.inf); best_cost = np.full(n, np.inf); best_k = np.full(n, -1)
    cost = G.max(1)
    for s in range(0, n, chunk):
        Rc = R[s:s + chunk]; nc = len(Rc)
        cap = sp.caps(Rc, beta)
        A = (G[None] * cap[:, None]).reshape(-1, sp.k)
        Rk = np.repeat(Rc, K, 0)
        for ti, t in enumerate(ttl_list):
            # TTL options after the first are evaluated only on flows no earlier option evaded: the callers use
            # ok[:, 0], ok.any(1), the first success (T-major order) and the cost (ti == 0 only), all unchanged by this
            act = np.ones(nc, bool) if ti == 0 else ~ok[s:s + nc, :ti].any(1)
            if not act.any():
                continue
            rows = np.repeat(act, K)
            tc = None
            if sp.has_ttl:
                tt, cc = sp.ttl_columns(Rc[act], t)
                tc = (np.repeat(tt, K), np.repeat(cc, K))
            hit = np.zeros((nc, K), bool)
            hit[act] = (model.proba(sp.minmax(sp.selected(Rk[rows], A[rows], tc))).argmax(1) == normal).reshape(act.sum(), K)
            ok[s:s + nc, ti] = hit.any(1)
            fi = np.where(hit.any(1), hit.argmax(1) + ti * K, np.inf)
            first[s:s + nc] = np.minimum(first[s:s + nc], fi)
            if ti > 0:      # cheapest action: volume and timing successes only (a TTL choice costs nothing)
                continue
            c = np.where(hit, cost[None], np.inf)
            kk = c.argmin(1); cv = c[np.arange(nc), kk]
            better = cv < best_cost[s:s + nc]
            best_cost[s:s + nc] = np.where(better, cv, best_cost[s:s + nc])
            best_k[s:s + nc] = np.where(better, kk, best_k[s:s + nc])
    return ok, first, best_cost, best_k


def adaptive(sp, model, R, beta, ttl_mode, normal, seed, iters=40, pop=16):
    """score-based adaptive search (the attacker sees class probabilities): evolutionary hill climbing on the
    unit action box (Gaussian steps, decaying sigma, coordinate snaps to 0/1) and, if allowed, on the TTL.
    ttl_mode: 'none' | 'cands' | 'free' (any TTL 1-255). Returns per-flow success."""
    rng = np.random.default_rng(seed)
    n, k = len(R), sp.k
    cap = sp.caps(R, beta)
    choices = {'none': None, 'cands': sp.ttl_cands if sp.has_ttl else None,
               'free': np.arange(1, 256, dtype=np.float64) if sp.has_ttl else None}[ttl_mode]

    def score(U, T, idx, live=None):
        # flows that already evaded are not queried again (their success is final; random draws are unchanged)
        live = np.ones(len(idx), bool) if live is None else live
        s_, hit = np.full(len(idx), -np.inf), np.zeros(len(idx), bool)
        if live.any():
            Rr = R[idx[live]]
            tc = sp.ttl_columns(Rr, T[live]) if (sp.has_ttl and choices is not None) else None
            P = model.proba(sp.minmax(sp.selected(Rr, U[live] * cap[idx[live]], tc)))
            s_[live], hit[live] = P[:, normal], P.argmax(1) == normal
        return s_, hit

    cur_u = rng.uniform(0, 1, (n, k)); cur_t = np.full(n, np.nan)
    cur_s, ok = score(cur_u, cur_t, np.arange(n))
    for it in range(iters):
        sig = 0.3 * (0.02 / 0.3) ** (it / max(1, iters - 1))
        idx = np.repeat(np.arange(n), pop)
        U = np.clip(cur_u[idx] + sig * rng.standard_normal((n * pop, k)), 0, 1)
        snap = rng.random(n * pop) < 0.2
        cj = rng.integers(0, k, n * pop)
        U[snap, cj[snap]] = rng.integers(0, 2, snap.sum())
        T = cur_t[idx].copy()
        if choices is not None:
            r_ = rng.random(n * pop) < 0.25
            T[r_] = rng.choice(np.r_[np.nan, choices], r_.sum())
        s_, hit = score(U, T, idx, ~ok[idx])
        ok |= hit.reshape(n, pop).any(1)
        S2 = s_.reshape(n, pop); j = S2.argmax(1); bs = S2[np.arange(n), j]
        up = bs > cur_s
        cur_s = np.where(up, bs, cur_s)
        cur_u[up] = U.reshape(n, pop, k)[np.arange(n), j][up]
        cur_t[up] = T.reshape(n, pop)[np.arange(n), j][up]
    return ok


def pack(b):
    return base64.b64encode(np.packbits(np.asarray(b, bool)).tobytes()).decode()


def attack_set(D, rng_seed=0):
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


def evaluate_attacks(D, sp, model, idx, seed, white_box=None, surrogate=None, light=False):
    normal = D['normal']
    R = D['raw']['Rte'][idx].astype(np.float64)
    y = D['yte'][idx]
    MM0 = sp.minmax(sp.selected(R, np.zeros((len(R), sp.k))))
    det = model.proba(MM0).argmax(1) != normal
    Rd, yd = R[det], y[det]
    out = dict(n_attacked=int(len(idx)), n_detected=int(det.sum()), det_flows=pack(det))
    cls = np.unique(yd)
    out['n_detected_per_class'] = {D['classes'][c]: int((yd == c).sum()) for c in cls}
    if len(Rd) == 0:        # degenerate model that detects no attack flow: nothing to attack
        print('    no detected attack flows - attacks skipped', flush=True)
        return out
    per_class = lambda ev: {D['classes'][c]: float(ev[yd == c].mean()) for c in cls}
    ttl_list = [None] + (list(sp.ttl_cands) if sp.has_ttl else [])
    nr = 120 if sp.ds == 'TON_IoT' else 48
    frac = lambda ev: float(np.mean(ev)) if len(ev) else float('nan')
    t0 = time.time()
    GRID = {}
    # 1. fixed grid (fresh seed), all budgets
    for beta in BETAS:
        G = unit_grid(sp.k, nr, seed + EVAL_OFF)
        ok, first, bcost, bk = search(sp, model, Rd, G, beta, ttl_list, normal)
        GRID[beta] = ok
        both, vol = ok.any(1), ok[:, 0]
        out[f'bb_both_b{beta}'], out[f'bb_volume_b{beta}'] = frac(both), frac(vol)
        if sp.has_ttl:
            ttl0 = np.zeros((1, sp.k))
            okt, _, _, _ = search(sp, model, Rd, ttl0, beta, ttl_list[1:], normal)
            out['bb_ttl_only'] = frac(okt.any(1))
        if beta == 0.5:
            out['bb_both_b0.5_per_class'] = per_class(both)
            out['bb_volume_b0.5_per_class'] = per_class(vol)
            out['ev_both_b0.5'], out['ev_volume_b0.5'] = pack(both), pack(vol)
            out['queries_first_success_median'] = float(np.median(first[both])) if both.any() else None
            out['queries_total'] = int(len(G) * len(ttl_list))
            # cheapest successful action (absolute units) per class
            cap = sp.caps(Rd, beta)
            A = np.where(bk[:, None] >= 0, G[np.maximum(bk, 0)] * cap, np.nan)
            out['min_action_per_class'] = {D['classes'][c]: dict(
                budget_frac=float(np.nanmedian(bcost[vol & (yd == c)])) if (vol & (yd == c)).any() else None,
                bytes=float(np.nanmedian(A[vol & (yd == c), 0])) if (vol & (yd == c)).any() else None,
                pkts=float(np.nanmedian(A[vol & (yd == c), 1])) if (vol & (yd == c)).any() else None,
                delay_s=float(np.nanmedian(A[vol & (yd == c), 2])) if (vol & (yd == c)).any() else None) for c in cls}
            E05 = {'grid': both, 'grid_vol': vol}
            # per-action ablation (TTL unchanged)
            for nm_, keep in (('bytes', (0,)), ('pkts', (1,)), ('delay', (2,) if sp.k == 3 else (2, 3)), ('bytes_pkts', (0, 1))):
                Gs = unit_grid(sp.k, nr, seed + EVAL_OFF, drop=tuple(i for i in range(sp.k) if i not in keep))
                oks, _, _, _ = search(sp, model, Rd, Gs, beta, [None], normal)
                out[f'bb_only_{nm_}_b0.5'] = frac(oks[:, 0])
            # budget floors
            for fs in (0.0, 0.5, 2.0):
                sp.floor_scale = fs
                okf, _, _, _ = search(sp, model, Rd, G, beta, ttl_list, normal)
                out[f'bb_both_b0.5_floor{fs}'] = frac(okf.any(1))
                out[f'bb_volume_b0.5_floor{fs}'] = frac(okf[:, 0])
            sp.floor_scale = 1.0
    print(f'    grid {time.time() - t0:.0f}s', flush=True)
    if sp.has_ttl:      # TTL sweep 1-255, TTL only
        sweep = {}
        anyu = np.zeros(len(Rd), bool); anyall = np.zeros(len(Rd), bool)
        P0 = np.zeros((len(Rd), sp.k))
        for t in range(1, 256):
            tc = sp.ttl_columns(Rd, float(t))
            e = model.proba(sp.minmax(sp.selected(Rd, P0, tc))).argmax(1) == normal
            sweep[t] = frac(e); anyall |= e
            if float(t) not in set(sp.ttl_cands):
                anyu |= e
        out['ttl_sweep'] = sweep
        out['bb_ttl_only_unseen'], out['bb_ttl_only_all'] = frac(anyu), frac(anyall)
    if not light:
        # 2. dense grids (saturation), beta 0.5
        # small grid alone (24 random points), then the standard grid plus 1,000 further random points per flow
        # (searched only on flows the standard grid did not evade; reported as the union)
        G = unit_grid(sp.k, 24, seed + DENSE_OFF)
        ok, _, _, _ = search(sp, model, Rd, G, 0.5, ttl_list, normal)
        out['bb_dense24_both_b0.5'], out['bb_dense24_volume_b0.5'] = frac(ok.any(1)), frac(ok[:, 0])
        G = unit_grid(sp.k, 1000, seed + DENSE_OFF)[2 ** sp.k:]
        rest = ~E05['grid_vol']
        okd = np.zeros((len(Rd), len(ttl_list)), bool)
        if rest.any():
            okd[rest], _, _, _ = search(sp, model, Rd[rest], G, 0.5, ttl_list, normal)
        E05['dense1000'] = E05['grid'] | okd.any(1); E05['dense1000_vol'] = E05['grid_vol'] | okd[:, 0]
        out['bb_grid_plus1000_both_b0.5'], out['bb_grid_plus1000_volume_b0.5'] = frac(E05['dense1000']), frac(E05['dense1000_vol'])
        print(f'    dense {time.time() - t0:.0f}s', flush=True)
    # 3. adaptive score-based search
    # (run only on flows the standard grid did not evade at that budget; reported as grid + adaptive)
    for beta in ((0.5, 2.0) if not light else (0.5,)):
        gvol, gboth = GRID[beta][:, 0], GRID[beta].any(1)
        modes = (('none', 'cands', 'free') if beta == 0.5 else ('none', 'free')) if sp.has_ttl else ('none',)
        for md in modes:
            base = gvol if md == 'none' else gboth
            ev = base.copy()
            if (~base).any():
                ev[~base] = adaptive(sp, model, Rd[~base], beta, md, normal, seed + ADA_OFF)
            out[f'ad_{md}_b{beta}'] = frac(ev)
            if beta == 0.5:
                E05[f'ad_{md}'] = ev
    print(f'    adaptive {time.time() - t0:.0f}s', flush=True)
    # 4. black-box feature-space search (all selected features, eps-ball corners), every model
    rng = np.random.default_rng(seed + 4000)
    MMd = np.clip(MM0[det], 0, 1)
    for eps in EPS_UNC:
        ev = np.zeros(len(Rd), bool)
        for _ in range(128):
            s = rng.choice((-1.0, 1.0), MMd.shape[1])
            ev |= model.proba(np.clip(MMd + eps * s, 0, 1)).argmax(1) == normal
        out[f'bb_unconstrained_e{eps}'] = frac(ev)
    # 5. gradient attacks (networks)
    if white_box is not None:
        for beta in BETAS:
            ev = np.zeros(len(Rd), bool)
            for t in ttl_list:
                tc = sp.ttl_columns(Rd, t) if sp.has_ttl else None
                Aw = np.vstack([_pgd_action(white_box, sp, Rd[s:s + 1024].astype(np.float32), sp.caps(Rd[s:s + 1024], beta).astype(np.float32),
                                            None if tc is None else (tc[0][s:s + 1024], tc[1][s:s + 1024]), normal, 50, seed=seed)
                                for s in range(0, len(Rd), 1024)])
                ev |= model.proba(sp.minmax(sp.selected(Rd, Aw, tc))).argmax(1) == normal
            out[f'wb_action_b{beta}'] = frac(ev)
        for beta in (0.5, 2.0):
            ev = np.zeros(len(Rd), bool)
            for t in ttl_list:
                tc = sp.ttl_columns(Rd, t) if sp.has_ttl else None
                for s in range(0, len(Rd), 1024):
                    ev[s:s + 1024] |= _pgd_action_strong(white_box, model, sp, Rd[s:s + 1024].astype(np.float32),
                                                         sp.caps(Rd[s:s + 1024], beta).astype(np.float32),
                                                         None if tc is None else (tc[0][s:s + 1024], tc[1][s:s + 1024]), normal, seed)
            out[f'wbs_action_b{beta}'] = frac(ev)
            if beta == 0.5:
                E05['wbs'] = ev
        mask = np.zeros(len(sp.sel)); mask[sp.mutable] = 1
        for eps in EPS_UNC:
            for tag, mk in (('', None), ('_mutable', mask)):
                xs = [_pgd_feature(white_box, sp, MMd[s:s + 1024], yd[s:s + 1024], eps, 50, True, normal, mk).numpy()
                      for s in range(0, len(Rd), 1024)]
                out[f'wb_unconstrained{tag}_e{eps}'] = frac(model.proba(np.vstack(xs)).argmax(1) == normal)
        print(f'    white-box {time.time() - t0:.0f}s', flush=True)
    # strongest attack per flow at beta 0.5: union of every attack run (volume-and-timing only, and with TTL)
    vol_keys = [k for k in E05 if k.endswith('_vol') or k == 'ad_none']
    allk = list(E05)
    sv = np.logical_or.reduce([E05[k] for k in vol_keys]); sa = np.logical_or.reduce([E05[k] for k in allk])
    out['strongest_volume_b0.5'], out['strongest_b0.5'] = frac(sv), frac(sa)
    out['strongest_b0.5_per_class'], out['strongest_volume_b0.5_per_class'] = per_class(sa), per_class(sv)
    out['ev_strongest_b0.5'], out['ev_strongest_volume_b0.5'] = pack(sa), pack(sv)
    out['strongest_components'] = allk
    if surrogate is not None:
        for beta in (0.5, 1.0):
            ev = np.zeros(len(Rd), bool)
            for t in ttl_list:
                tc = sp.ttl_columns(Rd, t) if sp.has_ttl else None
                Aw = np.vstack([_pgd_action(surrogate, sp, Rd[s:s + 1024].astype(np.float32), sp.caps(Rd[s:s + 1024], beta).astype(np.float32),
                                            None if tc is None else (tc[0][s:s + 1024], tc[1][s:s + 1024]), normal, 50, seed=seed)
                                for s in range(0, len(Rd), 1024)])
                ev |= model.proba(sp.minmax(sp.selected(Rd, Aw, tc))).argmax(1) == normal
            out[f'transfer_action_b{beta}'] = frac(ev)
    return out


# ── augmentation ──────────────────────────────────────────────────────────────

def augmentation(D, sp, model, seed, grid_seed, row_seed, drop=()):
    """most normal-looking candidate (beta_train; corners + 24 random points, grid seed disjoint from evaluation)
    for each of up to AUG_N real training attack records, with its true label. drop = action dims not used."""
    R, y = D['raw']['Rtr'], D['ytr']
    att = np.where(y != D['normal'])[0]
    rng = np.random.default_rng(row_seed)
    j = rng.choice(att, min(AUG_N, len(att)), replace=False)
    Rj = R[j].astype(np.float64)
    G = unit_grid(sp.k, 24, grid_seed, drop)
    K = len(G)
    cap = sp.caps(Rj, BETA_TRAIN)
    A = (G[None] * cap[:, None]).reshape(-1, sp.k)
    tc = None
    if sp.has_ttl:
        tt, cc = sp.ttl_columns(Rj, per_flow_ttl(sp, rng, len(j)))
        tc = (np.repeat(tt, K), np.repeat(cc, K))
    MM = sp.minmax(sp.selected(np.repeat(Rj, K, 0), A, tc))
    P = model.proba(MM)[:, D['normal']].reshape(len(j), K)
    return MM.reshape(len(j), K, -1)[np.arange(len(j)), P.argmax(1)].astype(np.float32), y[j]


# ── driver ────────────────────────────────────────────────────────────────────

def rare_classes(D):
    cnt = np.bincount(D['ytr'], minlength=len(D['classes']))
    return {c for c in range(len(cnt)) if c != D['normal'] and cnt[c] < RARE_SHARE * cnt.sum()}


def save_model(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if model.kind in ('cnn', 'mlp'):
        model.obj.save(str(path) + '.keras')
    else:
        import joblib
        joblib.dump(model.obj, str(path) + '.joblib', compress=3)


def load_model(kind, path, sp, nc):
    if kind in ('cnn', 'mlp'):
        import tensorflow as tf
        return Model(kind, tf.keras.models.load_model(str(path) + '.keras'), sp, nc)
    import joblib
    return Model(kind, joblib.load(str(path) + '.joblib'), sp, nc)


def run(D, sp, name, defence, seed, idx, surrogate=None, light=False):
    t0 = time.time()
    rare = rare_classes(D)
    nc = len(D['classes'])
    if name in ('cnn', 'mlp'):
        net, info = train_net(D, name, seed, defence, sp, rare)
        model = Model(name, net, sp, nc); wb = net
    else:
        aug, info = [], {}
        if defence.startswith('augment'):
            base = Model(name, train_tree(D, name, seed), sp, nc)
            drop = (2, 3) if defence == 'augment_nodelay' and sp.k == 4 else ((2,) if defence == 'augment_nodelay' else ())
            if defence == 'augment_nobytes':       # round 2: held-out bytes
                drop = (0,)
            aug.append(augmentation(D, sp, base, seed, seed + AUG_OFF, seed + 7, drop))
            if defence == 'augment2':
                m1 = Model(name, train_tree(D, name, seed, aug), sp, nc)
                aug.append(augmentation(D, sp, m1, seed, seed + AUG2_OFF, seed + 8))
            info = dict(n_aug=int(sum(len(a[1]) for a in aug)))
        model = Model(name, train_tree(D, name, seed, aug), sp, nc); wb = None
    P = model.proba(np.clip(D['Xte'], -1e9, 1e9))
    res = dict(train_s=time.time() - t0, **info, **P6.metrics(D, P.argmax(1), P), rare_classes=[D['classes'][c] for c in rare])
    res['benign_fpr_check'] = float(np.mean(P.argmax(1)[D['yte'] == D['normal']] != D['normal']))
    res.update(evaluate_attacks(D, sp, model, idx, seed, white_box=wb, surrogate=surrogate, light=light))
    res['eval_s'] = time.time() - t0 - res['train_s']
    return res, model


PLANS = {
    'base': [('cnn', 'none', SEEDS), ('mlp', 'none', SEEDS), ('rf', 'none', SEEDS), ('gb', 'none', SEEDS)],
    'defend_trees': [('rf', 'augment', SEEDS), ('gb', 'augment', SEEDS), ('rf', 'augment2', DEF_SEEDS), ('gb', 'augment2', DEF_SEEDS),
                     ('rf', 'augment_nodelay', DEF_SEEDS), ('gb', 'augment_nodelay', DEF_SEEDS)],
    'defend_nets': [('cnn', 'at_threat', SEEDS), ('cnn', 'at_all', DEF_SEEDS), ('cnn', 'ccat', DEF_SEEDS), ('cnn', 'trades', DEF_SEEDS),
                    ('cnn', 'clean_loop', DEF_SEEDS), ('mlp', 'at_threat', DEF_SEEDS)],
    'splits': [('cnn', 'none', (0,)), ('mlp', 'none', (0,)), ('rf', 'none', (0,)), ('gb', 'none', (0,)),
               ('rf', 'augment', (0,)), ('gb', 'augment', (0,)), ('cnn', 'at_threat', (0,))],
    'nottl': [('cnn', 'none', DEF_SEEDS), ('rf', 'none', DEF_SEEDS), ('gb', 'none', DEF_SEEDS), ('mlp', 'none', DEF_SEEDS)],
    # round 2
    'more_seeds': [('cnn', 'clean_loop', (3, 4)), ('cnn', 'at_all', (3, 4)), ('cnn', 'trades', (3, 4)), ('cnn', 'ccat', (3, 4))],
    'r2_nets': [('cnn', 'at_all_bpda', DEF_SEEDS), ('cnn', 'at_all_e0.02', (0,)), ('cnn', 'at_all_e0.1', (0,)),
                ('cnn', 'at_threat_b0.25', (0,)), ('cnn', 'at_threat_b1.0', (0,))],
    'r2_trees': [('rf', 'augment_nobytes', DEF_SEEDS), ('gb', 'augment_nobytes', DEF_SEEDS)],
    # round 3: AT-all at eps 0.1 on the remaining seeds (review_r3 point 6)
    'r3_sweep': [('cnn', 'at_all_e0.1', (1, 2, 3, 4))],
    # final review round 2: AT-threat with corner-start inner attack, five seeds (referee: inner search too weak)
    'fr2_atcs': [('cnn', 'at_threat_cs', SEEDS)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='all', choices=('all',) + DATASETS)
    ap.add_argument('--group', default='base', choices=tuple(PLANS))
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    global OUT, MOD, BETAS, EPS_UNC, N_EVAL, AUG_N
    if a.smoke:
        OUT, MOD, BETAS, EPS_UNC, N_EVAL, AUG_N = HERE / 'results_v3_smoke', HERE / 'models_v3_smoke', (0.5, 2.0), (0.05,), 200, 500
        P1.EPOCHS = 2
        ROBUST.update(epochs=1, steps=3)
    import tensorflow as tf
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    splits = (1, 2) if a.group == 'splits' else (0,)
    dss = DATASETS if a.dataset == 'all' else (a.dataset,)
    if a.group == 'nottl':
        dss = tuple(d for d in dss if d == 'UNSW_NB15')
    for ds in dss:
        for split in splits:
            D = P6.load(ds, split=split, keep_raw=True)
            if a.group == 'nottl':
                D = drop_ttl(D)
            sp = Space(D)
            idx = attack_set(D)
            MM0 = sp.minmax(sp.selected(D['raw']['Rte'][idx].astype(np.float64), np.zeros((len(idx), sp.k))))
            err = np.abs(MM0 - D['Xte'][idx]).max()
            assert err < 1e-3, f'feature map mismatch {err}'
            sfx = {0: ''}.get(split, f'_split{split}') + ('_nottl' if a.group == 'nottl' else '')
            (OUT / ds).mkdir(parents=True, exist_ok=True)
            meta = OUT / ds / f'attack_set{sfx}.json'
            if not meta.exists():
                meta.write_text(json.dumps(dict(idx=idx.tolist(), ttl_cands=getattr(sp, 'ttl_cands', np.array([])).tolist(),
                                                rare=[D['classes'][c] for c in rare_classes(D)], map_err=float(err), features=D['feat'],
                                                mm_rng={n: float(r) for n, r in zip(D['feat'], sp.mm_rng)},
                                                classes=D['classes'], yte_attack_set=D['yte'][idx].tolist(),
                                                test_class_counts={c: int((D['yte'] == i).sum()) for i, c in enumerate(D['classes'])})))
            print(f'== {ds} split {split} {a.group}: {len(D["feat"])} features, attack set {len(idx)}, map error {err:.1e}, '
                  f'TTL candidates {getattr(sp, "ttl_cands", None)}', flush=True)
            plan = PLANS[a.group]
            seeds = sorted({s for _, _, ss in plan for s in ss})
            if a.smoke:
                seeds = [0]
            for seed in seeds:
                for name, defence, ss in plan:
                    if seed not in ss:
                        continue
                    tag = f'{name}_{defence}{sfx}'
                    p = OUT / ds / tag / f'seed{seed}.json'
                    if p.exists():
                        continue
                    surrogate = None
                    if name != 'cnn' and defence == 'none':
                        sp_path = MOD / ds / f'cnn_none{sfx}' / f'seed{seed}'
                        if Path(str(sp_path) + '.keras').exists():
                            surrogate = load_model('cnn', sp_path, sp, len(D['classes'])).obj
                    res, model = run(D, sp, name, defence, seed, idx, surrogate, light=(a.group == 'splits'))
                    res['cfg'] = dict(model=name, defence=defence, seed=seed, split=split, variant=a.group,
                                      beta_train=BETA_TRAIN, eps_train=EPS_TRAIN, version='v3')
                    save_model(model, MOD / ds / tag / f'seed{seed}')
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(json.dumps(res, indent=1))
                    print(f'  {tag} seed{seed}: F1 {res["macro_f1"]:.4f}  bb@0.5 {res.get("bb_both_b0.5", float("nan")):.3f}  '
                          f'ad@0.5 {res.get("ad_" + ("free" if sp.has_ttl else "none") + "_b0.5", float("nan")):.3f}  '
                          f'({res["train_s"]:.0f}s + {res["eval_s"]:.0f}s)', flush=True)
                    del model, surrogate
                    gc.collect(); tf.keras.backend.clear_session()
            del D; gc.collect()


if __name__ == '__main__':
    main()
