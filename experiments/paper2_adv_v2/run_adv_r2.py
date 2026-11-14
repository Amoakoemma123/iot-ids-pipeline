"""Paper 2, review round 2: further evaluation of the saved v3 models (no retraining).

Parts (each stored once per model in results_r2/<ds>/<tag>/seed<k>.json; finished parts are skipped):
  union  strongest attack at beta 0.5 extended with the cheap attacks the v3 union left out (TTL sweep 1-255,
         24-point grid, per-action grids); on TON-IoT the volume-only union also counts the white-box attack
         (there is no TTL action, so it equals the all-actions union).
  fs     feature-space reference at eps 0.005-0.1: black-box corners of the full ball (128 random sign patterns),
         exhaustive {-1, 0, +1} patterns on the mutable features (and increase-only {0, +1}), and for the networks
         strong PGD through a BPDA quantile transform (full ball and mutable features only).
  op     operating point: ASR (grid + adaptive search, beta 0.5) and end-to-end evasion with the benign threshold
         set on validation benign flows to (a) the undefended model's FPR (same family and seed) and (b) 1% FPR,
         and at the model's own (argmax) operating point with the same attacks.
  feas   physical feasibility: beta 0.5 grid with integer packets and an MTU bound (added payload beyond what the
         flow's packets can carry needs extra packets, within the packet budget).
  sat    trees: saturation on 500 detected flows - regular grids of increasing resolution and a long adaptive
         search, against the v3 strongest attack on the same flows.
Round 3 (review_r3):
  opu    undefended models only: the same benign-threshold rule as 'op', with the threshold set to the model's own
         validation FPR (removes the argmax-vs-threshold confound from the matched-FPR comparison).
  bpa    networks: strong action-space PGD (margin loss, 4 restarts, 100 steps, beta 0.5, each TTL candidate)
         with the gradient taken through the BPDA quantile transform; checks gradient masking for every network.
  relax  consistency-relaxed realistic attack (beta 0.5, volume and timing): per flow, the range each mutable feature
         can reach under the realistic actions (min/max over the box corners and 2,000 random actions); then every
         mutable feature is set independently to its low end, its value or its high end (3^m patterns), plus 1,000
         independent uniform points in the box. Same magnitudes as the realistic attack, feature dependencies
         broken; relax_union adds the realistic evasions (every realistic point lies in the box).
Run from `Papers pipeline/`: python -m experiments.paper2_adv_v2.run_adv_r2 --dataset D --tags cnn_none,rf_none --parts all
"""
import argparse, base64, gc, itertools, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.paper2_adv_v2 import run_adv_v3 as V3   # noqa: E402

OUT = V3.HERE / 'results_r2'
EPS_R2 = (0.005, 0.01, 0.02, 0.05, 0.1)
PARTS = ('union', 'fs', 'op', 'feas', 'sat', 'opu', 'bpa', 'relax')
UNDEF = {'cnn': 'cnn_none', 'mlp': 'mlp_none', 'rf': 'rf_none', 'gb': 'gb_none'}


def unpack(s, n):
    return np.unpackbits(np.frombuffer(base64.b64decode(s), np.uint8))[:n].astype(bool)


class Thresh:
    """benign iff p_benign >= tau (otherwise the most likely attack class); scores stay monotone in p_benign"""
    def __init__(self, model, normal, tau):
        self.m, self.normal, self.tau = model, normal, tau
        self.kind = model.kind

    def proba(self, MM):
        P = self.m.proba(MM)
        pn = P[:, self.normal]
        Q = P.copy(); Q[:, self.normal] = -1
        other = Q.argmax(1)
        out = np.zeros_like(P)
        out[np.arange(len(P)), other] = 0.5 - (pn - self.tau)
        out[:, self.normal] = 0.5 + (pn - self.tau)
        return out


def val_fpr(model, D):
    yb = D['yva'] == D['normal']
    return float(np.mean(model.proba(D['Xva'][yb]).argmax(1) != D['normal']))


def tau_for(model, D, fpr):
    pn = model.proba(D['Xva'][D['yva'] == D['normal']])[:, D['normal']]
    return float(np.quantile(pn, fpr))


def feasible(sp, R, A, cap):
    """integer packets (rounded, within floor of the packet budget) and an MTU bound; returns (A', valid)"""
    rr = sp.raw
    b, p = R[:, rr['b']], R[:, rr['p']]
    A = A.copy()
    capp = np.floor(cap[:, 1] + 1e-9)
    ap = np.minimum(np.round(A[:, 1]), capp)
    if sp.ds == 'TON_IoT':          # payload bytes, MSS 1460
        C = np.maximum(1460.0 * p, b)
        need = np.ceil(np.maximum(0.0, b + A[:, 0] - C) / 1460.0 - 1e-9)
    else:                           # sbytes include headers; at most 1500 bytes per packet
        h = np.where(p > 0, np.clip(b / np.maximum(p, 1.0), 28.0, 40.0), 40.0)
        C = np.maximum(1500.0 * p, b)
        need = np.ceil(np.maximum(0.0, b + A[:, 0] - C) / (1500.0 - h) - 1e-9)
    ap = np.maximum(ap, need)
    valid = ap <= capp + 1e-9
    A[:, 1] = np.minimum(ap, capp)
    return A, valid


def feas_search(sp, model, R, G, beta, ttl_list, normal):
    n, K, T = len(R), len(G), len(ttl_list)
    ok = np.zeros((n, T), bool)
    chunk = max(1, 300000 // K)
    for s in range(0, n, chunk):
        Rc = R[s:s + chunk]; nc = len(Rc)
        cap = sp.caps(Rc, beta)
        A = (G[None] * cap[:, None]).reshape(-1, sp.k)
        Rk, capk = np.repeat(Rc, K, 0), np.repeat(cap, K, 0)
        A, valid = feasible(sp, Rk, A, capk)
        for ti, t in enumerate(ttl_list):
            tc = None
            if sp.has_ttl:
                tt, cc = sp.ttl_columns(Rc, t)
                tc = (np.repeat(tt, K), np.repeat(cc, K))
            hit = (model.proba(sp.minmax(sp.selected(Rk, A, tc))).argmax(1) == normal) & valid
            ok[s:s + nc, ti] = hit.reshape(nc, K).any(1)
    return ok


def light_attack(sp, model, Rd, seed, normal, ttl_list, ada=True):
    """grid (standard candidates) + adaptive search on the flows it misses, beta 0.5; returns per-flow success
    (volume and timing only, and with the TTL)"""
    nr = 120 if sp.ds == 'TON_IoT' else 48
    G = V3.unit_grid(sp.k, nr, seed + V3.EVAL_OFF)
    ok, _, _, _ = V3.search(sp, model, Rd, G, 0.5, ttl_list, normal)
    vol, both = ok[:, 0].copy(), ok.any(1).copy()
    if not ada:
        return vol, both
    if (~vol).any():
        vol[~vol] = V3.adaptive(sp, model, Rd[~vol], 0.5, 'none', normal, seed + V3.ADA_OFF)
    both |= vol
    if sp.has_ttl and (~both).any():
        both[~both] = V3.adaptive(sp, model, Rd[~both], 0.5, 'free', normal, seed + V3.ADA_OFF)
    return vol, both


def part_union(D, sp, model, res, Rd, yd, seed, normal, ttl_list):
    n = len(Rd)
    sa = unpack(res['ev_strongest_b0.5'], n); sv = unpack(res['ev_strongest_volume_b0.5'], n)
    out = {'strongest_b0.5': float(sa.mean()), 'strongest_volume_b0.5': float(sv.mean())}
    nr = 120 if sp.ds == 'TON_IoT' else 48
    add_a, add_v = np.zeros(n, bool), np.zeros(n, bool)
    G = V3.unit_grid(sp.k, 24, seed + V3.DENSE_OFF)
    ok, _, _, _ = V3.search(sp, model, Rd, G, 0.5, ttl_list, normal)
    add_a |= ok.any(1); add_v |= ok[:, 0]
    for keep in ((0,), (1,), (2,) if sp.k == 3 else (2, 3), (0, 1)):
        Gs = V3.unit_grid(sp.k, nr, seed + V3.EVAL_OFF, drop=tuple(i for i in range(sp.k) if i not in keep))
        oks, _, _, _ = V3.search(sp, model, Rd, Gs, 0.5, [None], normal)
        add_a |= oks[:, 0]; add_v |= oks[:, 0]
    if sp.has_ttl:
        P0 = np.zeros((n, sp.k))
        for t in range(1, 256):
            add_a |= model.proba(sp.minmax(sp.selected(Rd, P0, sp.ttl_columns(Rd, float(t))))).argmax(1) == normal
    if sp.ds == 'TON_IoT':
        sv = sv | sa          # no TTL action: the white-box attack counts as volume and timing
    pa, pv = sa | add_a, sv | add_v
    out.update({'strongest_plus_b0.5': float(pa.mean()), 'strongest_plus_volume_b0.5': float(pv.mean()),
                'strongest_plus_b0.5_per_class': {D['classes'][c]: float(pa[yd == c].mean()) for c in np.unique(yd)},
                'ev_strongest_plus_b0.5': V3.pack(pa), 'ev_strongest_plus_volume_b0.5': V3.pack(pv)})
    return out



def part_fs(D, sp, model, Rd, MMd, seed, normal, white_box):
    out = {}
    rng = np.random.default_rng(seed + 4100)
    mut = list(sp.mutable)
    pats = np.array(list(itertools.product((-1.0, 0.0, 1.0), repeat=len(mut))))
    inc = (pats >= 0).all(1)
    mask = np.zeros(MMd.shape[1]); mask[mut] = 1
    for eps in EPS_R2:
        evc = np.zeros(len(MMd), bool)
        for _ in range(128):
            s = rng.choice((-1.0, 1.0), MMd.shape[1])
            evc |= model.proba(np.clip(MMd + eps * s, 0, 1)).argmax(1) == normal
        evm, evi = np.zeros(len(MMd), bool), np.zeros(len(MMd), bool)
        for pi, pat in enumerate(pats):
            X = MMd.copy(); X[:, mut] = np.clip(X[:, mut] + eps * pat, 0, 1)
            e = model.proba(X).argmax(1) == normal
            evm |= e
            if inc[pi]:
                evi |= e
        out[f'bb_corners_e{eps}'] = float(evc.mean())
        out[f'bb_mut_e{eps}'] = float(evm.mean()); out[f'bb_mut_inc_e{eps}'] = float(evi.mean())
        fs_all, fs_mut = evc | evm, evm.copy()
        if white_box is not None:
            wf = np.zeros(len(MMd), bool); wm = np.zeros(len(MMd), bool)
            for s0 in range(0, len(MMd), 2048):
                x0 = MMd[s0:s0 + 2048]
                wf[s0:s0 + 2048] = V3._pgd_feature_strong(white_box, model, sp, x0, eps, normal, None, seed)
                wm[s0:s0 + 2048] = V3._pgd_feature_strong(white_box, model, sp, x0, eps, normal, mask, seed)
            out[f'wb_bpda_e{eps}'] = float(wf.mean()); out[f'wb_bpda_mutable_e{eps}'] = float(wm.mean())
            fs_all |= wf | wm; fs_mut |= wm
        out[f'fs_strongest_e{eps}'] = float(fs_all.mean()); out[f'fs_strongest_mutable_e{eps}'] = float(fs_mut.mean())
    return out


def part_op(D, sp, model, idx, seed, normal, ttl_list, tag, undef_model, ada=True):
    R = D['raw']['Rte'][idx].astype(np.float64)
    MM0 = sp.minmax(sp.selected(R, np.zeros((len(R), sp.k))))
    yb = D['yte'] == normal
    ops = {'native': model}
    ops['fpr1'] = Thresh(model, normal, tau_for(model, D, 0.01))
    if undef_model is not None:
        ops['fpr_match'] = Thresh(model, normal, tau_for(model, D, val_fpr(undef_model, D)))
    out = {}
    for nm, mdl in ops.items():
        det = mdl.proba(MM0).argmax(1) != normal
        test_fpr = float(np.mean(mdl.proba(D['Xte'][yb]).argmax(1) != normal))
        vol, both = light_attack(sp, mdl, R[det], seed, normal, ttl_list, ada) if det.any() else (np.zeros(0, bool),) * 2
        o = dict(detected=float(det.mean()), test_fpr=test_fpr,
                 asr_volume=float(vol.mean()) if len(vol) else None, asr=float(both.mean()) if len(both) else None,
                 e2e_volume=float(((~det).sum() + vol.sum()) / len(det)), e2e=float(((~det).sum() + both.sum()) / len(det)))
        if nm != 'native':
            o['tau'] = mdl.tau
        o['attack'] = 'grid + adaptive' if ada else 'grid only'
        out[nm] = o
    return out


def part_opu(D, sp, model, idx, seed, normal, ttl_list, ada=True):
    """undefended model scored with the threshold rule at its own validation FPR (same attack as part_op)"""
    R = D['raw']['Rte'][idx].astype(np.float64)
    MM0 = sp.minmax(sp.selected(R, np.zeros((len(R), sp.k))))
    yb = D['yte'] == normal
    vf = val_fpr(model, D)
    mdl = Thresh(model, normal, tau_for(model, D, vf))
    det = mdl.proba(MM0).argmax(1) != normal
    test_fpr = float(np.mean(mdl.proba(D['Xte'][yb]).argmax(1) != normal))
    vol, both = light_attack(sp, mdl, R[det], seed, normal, ttl_list, ada) if det.any() else (np.zeros(0, bool),) * 2
    return dict(val_fpr=vf, tau=mdl.tau, detected=float(det.mean()), test_fpr=test_fpr,
                asr_volume=float(vol.mean()) if len(vol) else None, asr=float(both.mean()) if len(both) else None,
                e2e_volume=float(((~det).sum() + vol.sum()) / len(det)), e2e=float(((~det).sum() + both.sum()) / len(det)),
                attack='grid + adaptive' if ada else 'grid only')


_BA = {}


def _bpda_action_fn(m, sp):
    key = id(m)
    if key in _BA:
        return _BA[key]
    import tensorflow as tf
    mn, rg = tf.constant(sp.mm_min, tf.float32), tf.constant(sp.mm_rng, tf.float32)
    unsw = sp.has_ttl

    @tf.function(reduce_retracing=True)
    def action(u, Rt, capt, t, cst, aux):
        with tf.GradientTape() as tape:
            tape.watch(u)
            S = sp.selected(Rt, u * capt, (t, cst) if unsw else None, xp=tf)
            lp = tf.math.log(m(sp.tf_quantile_bpda((S - mn) / rg)[..., None], training=False) + 1e-12)
            loss = tf.reduce_mean(tf.reduce_sum(aux * lp, 1) - tf.reduce_max(lp - 1e9 * aux, 1))
        return tape.gradient(loss, u)

    _BA.clear()
    _BA[key] = action
    return action


def pgd_action_bpda(m, model, sp, R, cap, ttlc, normal, seed, steps=100):
    """as V3._pgd_action_strong, but the gradient passes through the BPDA quantile transform"""
    import tensorflow as tf
    action = _bpda_action_fn(m, sp)
    n = len(R)
    t, cst = V3._ttl_tensors(sp, R, ttlc)
    aux = tf.constant(np.eye(m.output_shape[-1], dtype=np.float32)[np.full(n, normal)])
    capt, Rt = tf.constant(cap, tf.float32), tf.constant(R, tf.float32)
    rng = np.random.default_rng(seed)
    starts = [np.zeros(cap.shape), np.ones(cap.shape), rng.uniform(0, 1, cap.shape), rng.uniform(0, 1, cap.shape)]
    ok = np.zeros(n, bool)
    for u0 in starts:
        u = tf.constant(u0.astype(np.float32))
        for s in range(steps):
            a = 0.1 * 0.1 ** (s / (steps - 1))
            u = tf.clip_by_value(u + a * tf.sign(action(u, Rt, capt, t, cst, aux)), 0.0, 1.0)
            if (s + 1) % 25 == 0:
                ok |= model.proba(sp.minmax(sp.selected(R, (u * capt).numpy(), ttlc))).argmax(1) == normal
    return ok


def part_bpa(sp, model, white_box, res, Rd, seed, normal, ttl_list):
    n = len(Rd)
    sa = unpack(res['ev_strongest_b0.5'], n)
    ev, evv = np.zeros(n, bool), np.zeros(n, bool)
    for t in ttl_list:
        tc = sp.ttl_columns(Rd, t) if sp.has_ttl else None
        e = np.zeros(n, bool)
        for s in range(0, n, 1024):
            e[s:s + 1024] = pgd_action_bpda(white_box, model, sp, Rd[s:s + 1024].astype(np.float32),
                                            sp.caps(Rd[s:s + 1024], 0.5).astype(np.float32),
                                            None if tc is None else (tc[0][s:s + 1024], tc[1][s:s + 1024]), normal, seed)
        ev |= e
        if t is None:
            evv |= e
    return {
        'bpda_action_b0.5': float(ev.mean()), 'bpda_action_volume_b0.5': float(evv.mean()),
        'bpda_new_vs_strongest_b0.5': float((ev & ~sa).mean()), 'strongest_with_bpda_b0.5': float((ev | sa).mean()),
        'ev_bpda_action_b0.5': V3.pack(ev)}


def part_relax(sp, model, res, Rd, MMd, seed, normal):
    n = len(Rd)
    sv = unpack(res['ev_strongest_volume_b0.5'], n)
    mut = list(sp.mutable)
    cap = sp.caps(Rd, 0.5)
    G = V3.unit_grid(sp.k, 2000, seed + 7100)
    lo, hi = MMd[:, mut].copy(), MMd[:, mut].copy()
    for g in G:
        X = sp.minmax(sp.selected(Rd, g[None, :] * cap))[:, mut]
        lo, hi = np.minimum(lo, X), np.maximum(hi, X)
    lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
    ev, evc = np.zeros(n, bool), np.zeros(n, bool)
    for pat in itertools.product((0, 1, 2), repeat=len(mut)):
        X = MMd.copy()
        for j, c in enumerate(pat):
            if c != 1:
                X[:, mut[j]] = (lo if c == 0 else hi)[:, j]
        e = model.proba(X).argmax(1) == normal
        ev |= e
        if 1 not in pat:
            evc |= e
    rng = np.random.default_rng(seed + 7200)
    for _ in range(1000):
        X = MMd.copy()
        X[:, mut] = lo + rng.uniform(0, 1, lo.shape) * (hi - lo)
        ev |= model.proba(X).argmax(1) == normal
    width = (hi - lo)
    return {'relax_b0.5': float(ev.mean()), 'relax_union_b0.5': float((ev | sv).mean()), 'relax_corners_b0.5': float(evc.mean()),
            'realistic_volume_b0.5': float(sv.mean()), 'relax_new_vs_realistic': float((ev & ~sv).mean()),
            'realistic_not_relax': float((sv & ~ev).mean()),
            'median_width_mm': [float(x) for x in np.median(width, 0)], 'ev_relax_b0.5': V3.pack(ev)}


def part_feas(sp, model, Rd, seed, normal, ttl_list):
    nr = 120 if sp.ds == 'TON_IoT' else 48
    G = V3.unit_grid(sp.k, nr, seed + V3.EVAL_OFF)
    ok0, _, _, _ = V3.search(sp, model, Rd, G, 0.5, ttl_list, normal)
    ok1 = feas_search(sp, model, Rd, G, 0.5, ttl_list, normal)
    return dict(grid_both=float(ok0.any(1).mean()), grid_volume=float(ok0[:, 0].mean()),
                feas_both=float(ok1.any(1).mean()), feas_volume=float(ok1[:, 0].mean()),
                lost_both=float((ok0.any(1) & ~ok1.any(1)).mean()), gained_both=float((~ok0.any(1) & ok1.any(1)).mean()))


def part_sat(sp, model, res, Rd, seed, normal, ttl_list):
    n = len(Rd)
    sa = unpack(res['ev_strongest_b0.5'], n)
    rng = np.random.default_rng(seed + 5000)
    sub = np.sort(rng.choice(n, min(500, n), replace=False))
    Rs = Rd[sub]
    out = dict(n_sub=int(len(sub)), strongest_sub=float(sa[sub].mean()))
    union = sa[sub].copy()
    for r in ((5, 9, 17) if sp.k == 3 else (3, 5, 7)):
        G = np.array(list(itertools.product(np.linspace(0, 1, r), repeat=sp.k)))
        ok, _, _, _ = V3.search(sp, model, Rs, G, 0.5, ttl_list, normal)
        out[f'grid_r{r}'] = float(ok.any(1).mean()); union |= ok.any(1)
    left = ~union
    ad = np.zeros(len(sub), bool)
    if left.any():
        ad[left] = V3.adaptive(sp, model, Rs[left], 0.5, 'free' if sp.has_ttl else 'none', normal, seed + 6000, iters=150, pop=32)
    out['long_adaptive_new'] = float(ad.mean()); union |= ad
    out['union_sub'] = float(union.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=V3.DATASETS)
    ap.add_argument('--tags', required=True)
    ap.add_argument('--parts', default='all')
    ap.add_argument('--nottl', action='store_true')
    ap.add_argument('--seeds', default='0,1,2,3,4')
    ap.add_argument('--limit', type=int, default=0, help='test only: first N detected flows, results_r2_test')
    a = ap.parse_args()
    global OUT
    if a.limit:
        OUT = V3.HERE / 'results_r2_test'
    parts = PARTS if a.parts == 'all' else tuple(a.parts.split(','))
    import tensorflow as tf
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    ds = a.dataset
    D = V3.P6.load(ds, split=0, keep_raw=True)
    if a.nottl:
        D = V3.drop_ttl(D)
    sp = V3.Space(D)
    sfx = '_nottl' if a.nottl else ''
    meta = json.loads((V3.OUT / ds / f'attack_set{sfx}.json').read_text())
    idx = np.array(meta['idx'])
    assert (idx == V3.attack_set(D)).all()
    normal, nc = D['normal'], len(D['classes'])
    ttl_list = [None] + (list(sp.ttl_cands) if sp.has_ttl else [])
    for tag0 in a.tags.split(','):
        tag = tag0 + sfx
        kind = tag0.split('_')[0]
        for seed in [int(s) for s in a.seeds.split(',')]:
            pres = V3.OUT / ds / tag / f'seed{seed}.json'
            if not pres.exists():
                continue
            pout = OUT / ds / tag / f'seed{seed}.json'
            done = json.loads(pout.read_text()) if pout.exists() else {}
            todo = [p for p in parts if p not in done and not (p == 'sat' and kind not in ('rf', 'gb'))
                    and not (p == 'opu' and tag0 != UNDEF[kind]) and not (p == 'bpa' and kind not in ('cnn', 'mlp'))]
            if not todo:
                continue
            t0 = time.time()
            res = json.loads(pres.read_text())
            model = V3.load_model(kind, V3.MOD / ds / tag / f'seed{seed}', sp, nc)
            wb = model.obj if kind in ('cnn', 'mlp') else None
            R = D['raw']['Rte'][idx].astype(np.float64)
            MM0 = sp.minmax(sp.selected(R, np.zeros((len(R), sp.k))))
            det = model.proba(MM0).argmax(1) != normal
            assert (det == unpack(res['det_flows'], len(idx))).all(), f'{tag} seed{seed}: detected set differs'
            Rd, MMd = R[det], np.clip(MM0[det], 0, 1)
            yd = D['yte'][idx][det]
            idx_run = idx
            if a.limit:
                res = dict(res, **{k: V3.pack(unpack(res[k], len(Rd))[:a.limit]) for k in ('ev_strongest_b0.5', 'ev_strongest_volume_b0.5')})
                Rd, MMd, yd = Rd[:a.limit], MMd[:a.limit], yd[:a.limit]
                idx_run = idx[:3 * a.limit]
            undef = None
            if 'op' in todo and tag0 != UNDEF[kind]:
                pu = V3.MOD / ds / (UNDEF[kind] + sfx) / f'seed{seed}'
                if Path(str(pu) + ('.keras' if kind in ('cnn', 'mlp') else '.joblib')).exists():
                    undef = V3.load_model(kind, pu, sp, nc)
            for p in todo:
                t1 = time.time()
                if p == 'union':
                    done[p] = part_union(D, sp, model, res, Rd, yd, seed, normal, ttl_list)
                elif p == 'fs':
                    done[p] = part_fs(D, sp, model, Rd, MMd, seed, normal, wb)
                elif p == 'op':
                    done[p] = part_op(D, sp, model, idx_run, seed, normal, ttl_list, tag, undef, ada=kind in ('cnn', 'mlp'))
                elif p == 'feas':
                    done[p] = part_feas(sp, model, Rd, seed, normal, ttl_list)
                elif p == 'sat':
                    done[p] = part_sat(sp, model, res, Rd, seed, normal, ttl_list)
                elif p == 'opu':
                    done[p] = part_opu(D, sp, model, idx_run, seed, normal, ttl_list, ada=kind in ('cnn', 'mlp'))
                elif p == 'bpa':
                    done[p] = part_bpa(sp, model, wb, res, Rd, seed, normal, ttl_list)
                elif p == 'relax':
                    done[p] = part_relax(sp, model, res, Rd, MMd, seed, normal)
                pout.parent.mkdir(parents=True, exist_ok=True)
                cur = json.loads(pout.read_text()) if pout.exists() else {}   # merge: other queues may write other parts
                cur[p] = done[p]
                pout.write_text(json.dumps(cur, indent=1))
                print(f'  {tag} seed{seed} {p}: {time.time() - t1:.0f}s', flush=True)
            print(f'  {tag} seed{seed} done ({time.time() - t0:.0f}s)', flush=True)
            del model, wb, undef
            V3._GRADS.clear(); V3._GRADS_B.clear(); _BA.clear()
            gc.collect(); tf.keras.backend.clear_session()


if __name__ == '__main__':
    main()
