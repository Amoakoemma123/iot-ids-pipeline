"""
run_xai_v2.py — Paper 5 v2: explanation quality with adequate samples and CIs.

Changes from v1 (run_xai_experiments.py), each answering a review point:
  * Uses the v2 baseline pipeline/models (UNSW `id` removed, real dedup).
  * SHAP-vs-LIME agreement is measured PER INSTANCE on the same predicted
    class (v1 compared class-averaged global SHAP with LIME pooled over 12
    instances), over a class-stratified sample, with bootstrap 95% CIs.
  * LIME stability: 20 runs x N_STAB instances x every class (v1: 1 instance,
    2 classes).
  * Faithfulness: deletion test (replace top-k attributed features with the
    training mean) against a random-k baseline.
  * Global SHAP ranking uncertainty via bootstrap rank intervals.
  * Counterfactuals: sparse search over FEASIBLE values only — each changed
    field takes its value(s) from a real benign training flow, one-hot groups
    change jointly — for RF and CNN, every attack class, N_CF per class.
    (v1 DiCE moved one-hot flags and ports to fractional, impossible values.)
  * Hardening experiment: does feeding counterfactual variants back into
    training make the RF harder to evade at the same budget?
  * Port-shortcut check: RF retrained without port columns.

Usage:
  python -m experiments.paper5_xai.run_xai_v2 --dataset TON_IoT
  python -m experiments.paper5_xai.run_xai_v2 --dataset all --stage all
Stages are cached in results_v2/<dataset>/<stage>.json; delete a file to redo it.
"""
import argparse, json, os, sys, time, warnings
from pathlib import Path

import numpy as np
import joblib

warnings.filterwarnings('ignore')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.baseline.config import DATASETS as BASE_DATASETS

OUT_ROOT   = ROOT / 'experiments' / 'paper5_xai' / 'results_v2'
BASE_RES   = ROOT / 'experiments' / 'baseline' / 'results'
SEED       = 42

N_GLOBAL   = 1000   # random test rows for global SHAP
N_LOCAL    = 30     # per class, for SHAP-vs-LIME agreement + deletion test
N_STAB     = 10     # per class, LIME stability instances
N_RUNS     = 20     # LIME repeats per stability instance
N_CF       = 50     # per attack class, counterfactual queries
CF_MAX_K   = 4      # max number of changed fields
CF_TRIES   = 400    # random candidates per k
N_BOOT     = 2000
DEL_K      = (1, 3, 5)


# ── helpers ───────────────────────────────────────────────────────────────────

def save(obj, path):
    def _d(o):
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        raise TypeError(type(o))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_d))
    print(f'  saved {path.relative_to(ROOT)}')


def boot_ci(x, n=N_BOOT, seed=SEED):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return dict(mean=None, lo=None, hi=None, n=0)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return dict(mean=float(x.mean()), lo=float(np.percentile(means, 2.5)),
                hi=float(np.percentile(means, 97.5)), n=int(len(x)))


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def topk(attr, k):
    return list(np.argsort(-np.abs(attr))[:k])


def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(np.abs(a), np.abs(b)).correlation
    return float(r) if np.isfinite(r) else np.nan


# ── data / models ─────────────────────────────────────────────────────────────

def load_everything(ds):
    from src.data.loader import load_dataset
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    pipe = joblib.load(BASE_RES / ds / 'pipeline.pkl')
    csv, label_col = BASE_DATASETS[ds]
    raw = load_dataset(csv, label_col=label_col, seed=SEED)
    X_tr, _ = pipe.transform(raw['X_train'])
    X_te, _ = pipe.transform(raw['X_test'])
    X_tr_sm, y_tr_sm = pipe.transform(raw['X_train'], apply_smote=True, y=raw['y_train'])
    feat = list(pipe.selected_features_)
    classes = list(raw['class_names'])

    rf  = joblib.load(BASE_RES / 'models' / ds / 'model_rf.pkl')
    cnn = tf.keras.models.load_model(BASE_RES / 'models' / ds / 'model_cnn.h5', compile=False)

    def cnn_proba(X):
        return cnn.predict(np.asarray(X, np.float32)[..., None], verbose=0, batch_size=4096)

    return dict(pipe=pipe, feat=feat, classes=classes,
                X_tr=X_tr, y_tr=raw['y_train'], X_tr_sm=X_tr_sm, y_tr_sm=y_tr_sm,
                X_te=X_te, y_te=raw['y_test'],
                raw_feat=raw['feature_names'], X_tr_raw=raw['X_train'], X_te_raw=raw['X_test'],
                y_te_raw=raw['y_test'],
                models={'rf': rf.predict_proba, 'cnn': cnn_proba},
                rf=rf, cnn=cnn,
                normal=classes.index('normal'))


def field_groups(feat, X_tr):
    """One-hot dummies from the same source column form one 'field'."""
    groups = {}
    for i, f in enumerate(feat):
        vals = np.unique(X_tr[:, i])
        is_bin = len(vals) <= 2 and set(np.round(vals, 6)) <= {0.0, 1.0}
        key = f.rsplit('_', 1)[0] if (is_bin and '_' in f) else f
        groups.setdefault(key, []).append(i)
    return list(groups.items())


def stratified_idx(y, per_class, seed=SEED):
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        out.extend(rng.choice(idx, min(per_class, len(idx)), replace=False))
    return np.array(sorted(out))


# ── attributions ──────────────────────────────────────────────────────────────

def shap_rf(rf, X_bg, X):
    import shap
    ex = shap.TreeExplainer(rf, X_bg, model_output='probability',
                            feature_perturbation='interventional')
    sv = ex.shap_values(X, check_additivity=False)
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else sv   # (n, f, c)
    return sv


def shap_cnn(cnn, X_bg, X):
    import shap
    ex = shap.GradientExplainer(cnn, np.asarray(X_bg, np.float32)[..., None])
    sv = ex.shap_values(np.asarray(X, np.float32)[..., None])
    if isinstance(sv, list):
        sv = np.stack([s[..., 0] if s.ndim == 3 else s for s in sv], axis=-1)
    elif sv.ndim == 4:
        sv = sv[:, :, 0, :]
    return sv                                                       # (n, f, c)


def lime_explainer(X_tr, feat, classes, seed):
    from lime.lime_tabular import LimeTabularExplainer
    return LimeTabularExplainer(X_tr, feature_names=feat, class_names=classes,
                                discretize_continuous=False, random_state=seed)


def lime_attr(expl, x, fn, label, n_feat):
    e = expl.explain_instance(x, fn, labels=[label], num_features=n_feat, num_samples=1000)
    w = np.zeros(n_feat)
    for j, v in e.as_map()[label]:
        w[j] = v
    return w, float(e.score)


# ── stages ────────────────────────────────────────────────────────────────────

def stage_performance(D):
    from sklearn.metrics import f1_score
    out = {}
    for m, fn in D['models'].items():
        p = fn(D['X_te']).argmax(1)
        out[m] = dict(macro_f1=float(f1_score(D['y_te'], p, average='macro')),
                      accuracy=float((p == D['y_te']).mean()))
    out['n_test'] = int(len(D['y_te'])); out['n_features'] = len(D['feat'])
    out['features'] = D['feat']; out['classes'] = D['classes']
    out['class_counts_test'] = {c: int((D['y_te'] == i).sum()) for i, c in enumerate(D['classes'])}
    return out


def stage_global_shap(D):
    rng = np.random.default_rng(SEED)
    gi = rng.choice(len(D['X_te']), min(N_GLOBAL, len(D['X_te'])), replace=False)
    bg = D['X_tr'][rng.choice(len(D['X_tr']), 200, replace=False)]
    res = {}
    for m in ('rf', 'cnn'):
        t0 = time.time()
        sv = shap_rf(D['rf'], bg, D['X_te'][gi]) if m == 'rf' else shap_cnn(D['cnn'], bg, D['X_te'][gi])
        el = time.time() - t0
        per_inst = np.abs(sv).mean(axis=2)                 # (n, f) averaged over classes
        imp = per_inst.mean(0)
        # bootstrap rank intervals
        brng = np.random.default_rng(SEED)
        ranks = []
        for _ in range(500):
            b = per_inst[brng.integers(0, len(per_inst), len(per_inst))].mean(0)
            ranks.append(np.argsort(np.argsort(-b)) + 1)
        ranks = np.array(ranks)
        order = np.argsort(-imp)
        res[m] = dict(
            seconds=el, n=len(gi),
            ranking=[dict(feature=D['feat'][j], mean_abs_shap=float(imp[j]),
                          rank_lo=int(np.percentile(ranks[:, j], 2.5)),
                          rank_hi=int(np.percentile(ranks[:, j], 97.5)))
                     for j in order],
            per_class={c: dict(zip(D['feat'], np.abs(sv[:, :, ci]).mean(0).tolist()))
                       for ci, c in enumerate(D['classes'])})
    a = [r['feature'] for r in res['rf']['ranking']]
    b = [r['feature'] for r in res['cnn']['ranking']]
    res['cross_model'] = {f'jaccard@{k}': jaccard(a[:k], b[:k]) for k in (5, 10)}
    if len(D['feat']) <= 30:
        import shap
        t0 = time.time()
        iv = shap.TreeExplainer(D['rf'], feature_perturbation='tree_path_dependent') \
                 .shap_interaction_values(D['X_te'][gi[:100]])
        iv = np.stack(iv, 0) if isinstance(iv, list) else np.moveaxis(iv, -1, 0)  # (c,n,f,f)
        M = np.abs(iv).mean(axis=(0, 1)); np.fill_diagonal(M, 0)
        pairs = sorted(((M[i, j], D['feat'][i], D['feat'][j])
                        for i in range(len(M)) for j in range(i + 1, len(M))), reverse=True)[:10]
        res['rf_interactions'] = dict(n=100, seconds=time.time() - t0, matrix=M.tolist(),
                                      top_pairs=[dict(a=p[1], b=p[2], value=float(p[0])) for p in pairs])
    return res


def stage_local(D):
    """Per-instance SHAP vs LIME agreement, LIME fidelity, deletion faithfulness."""
    idx = stratified_idx(D['y_te'], N_LOCAL)
    X = D['X_te'][idx]; y = D['y_te'][idx]
    rng = np.random.default_rng(SEED)
    bg = D['X_tr'][rng.choice(len(D['X_tr']), 200, replace=False)]
    mean_vec = D['X_tr'].mean(0)
    nf = len(D['feat'])
    expl = lime_explainer(D['X_tr'], D['feat'], D['classes'], SEED)
    out = {}
    for m, fn in D['models'].items():
        t0 = time.time()
        sv = shap_rf(D['rf'], bg, X) if m == 'rf' else shap_cnn(D['cnn'], bg, X)
        t_shap = time.time() - t0
        pred = fn(X).argmax(1)
        rows = []
        t_lime = []
        for i in range(len(X)):
            s = sv[i, :, pred[i]]
            t1 = time.time()
            l, r2 = lime_attr(expl, X[i], fn, int(pred[i]), nf)
            t_lime.append(time.time() - t1)
            row = dict(test_idx=int(idx[i]), true=int(y[i]), pred=int(pred[i]),
                       j5=jaccard(topk(s, 5), topk(l, 5)), j3=jaccard(topk(s, 3), topk(l, 3)),
                       rho=spearman(s, l), lime_r2=r2)
            # deletion: replace top-k by train mean, measure drop in p(pred)
            p0 = fn(X[i:i + 1])[0, pred[i]]
            for k in DEL_K:
                for name, att in (('shap', s), ('lime', l)):
                    z = X[i].copy(); z[topk(att, k)] = mean_vec[topk(att, k)]
                    row[f'del_{name}_{k}'] = float(p0 - fn(z[None])[0, pred[i]])
                Z = np.repeat(X[i][None], 10, 0)
                for r in range(10):
                    js = rng.choice(nf, k, replace=False); Z[r, js] = mean_vec[js]
                row[f'del_rand_{k}'] = float(p0 - fn(Z)[:, pred[i]].mean())
            rows.append(row)
        summ = {key: boot_ci([r[key] for r in rows])
                for key in rows[0] if key not in ('test_idx', 'true', 'pred')}
        per_class = {}
        for ci, c in enumerate(D['classes']):
            rr = [r for r in rows if r['true'] == ci]
            if rr:
                per_class[c] = {k: boot_ci([r[k] for r in rr])
                                for k in ('j5', 'rho', 'lime_r2', 'del_shap_3', 'del_lime_3', 'del_rand_3')}
        out[m] = dict(summary=summ, per_class=per_class, rows=rows,
                      shap_ms_per_instance=1000 * t_shap / len(X),
                      lime_ms_per_instance=1000 * float(np.mean(t_lime)))
    return out


def stage_stability(D):
    idx = stratified_idx(D['y_te'], N_STAB, seed=SEED + 1)
    nf = len(D['feat'])
    out = {}
    for m, fn in D['models'].items():
        expl = lime_explainer(D['X_tr'], D['feat'], D['classes'], SEED)
        pred = fn(D['X_te'][idx]).argmax(1)
        rows = []
        for i, ti in enumerate(idx):
            sets = []
            for _ in range(N_RUNS):
                w, _r2 = lime_attr(expl, D['X_te'][ti], fn, int(pred[i]), nf)
                sets.append(topk(w, 5))
            js = [jaccard(sets[a], sets[b]) for a in range(N_RUNS) for b in range(a + 1, N_RUNS)]
            rows.append(dict(test_idx=int(ti), true=int(D['y_te'][ti]), stab=float(np.mean(js))))
        per_class = {c: boot_ci([r['stab'] for r in rows if r['true'] == ci])
                     for ci, c in enumerate(D['classes'])}
        out[m] = dict(overall=boot_ci([r['stab'] for r in rows]), per_class=per_class,
                      share_ge_075=float(np.mean([r['stab'] >= 0.75 for r in rows])), rows=rows)
    return out


def cf_search(fn, x, target, groups, donors, rng, max_k=CF_MAX_K, tries=CF_TRIES):
    """Smallest set of fields whose values, copied from real benign flows,
    flips the prediction to `target`. Returns (k, fields) or (None, None)."""
    G = len(groups)
    for k in range(1, max_k + 1):
        cand = np.repeat(x[None], tries, 0)
        chosen = []
        for t in range(tries):
            gs = rng.choice(G, min(k, G), replace=False)
            d = donors[rng.integers(len(donors))]
            for g in gs:
                cols = groups[g][1]; cand[t, cols] = d[cols]
            chosen.append(gs)
        ok = np.where(fn(cand).argmax(1) == target)[0]
        if len(ok):
            t = ok[0]
            return k, [groups[g][0] for g in chosen[t]], cand[t]
    return None, None, None


def stage_counterfactual(D):
    groups = field_groups(D['feat'], D['X_tr'])
    donors = D['X_tr'][D['y_tr'] == D['normal']]
    rng = np.random.default_rng(SEED)
    out = dict(n_fields=len(groups), fields=[g for g, _ in groups])
    for m, fn in D['models'].items():
        pred_te = fn(D['X_te']).argmax(1)
        per_class = {}
        all_k = []
        for ci, c in enumerate(D['classes']):
            if ci == D['normal']:
                continue
            # only correctly detected attacks: evasion presumes a current detection
            pool = np.where((D['y_te'] == ci) & (pred_te == ci))[0]
            if len(pool) == 0:
                continue
            q = rng.choice(pool, min(N_CF, len(pool)), replace=False)
            ks, fields = [], {}
            for ti in q:
                k, fs, _ = cf_search(fn, D['X_te'][ti], D['normal'], groups, donors, rng)
                ks.append(k if k is not None else np.nan)
                for f in fs or []:
                    fields[f] = fields.get(f, 0) + 1
            ks = np.array(ks, float)
            found = ks[~np.isnan(ks)]
            all_k.extend(found)
            per_class[c] = dict(n=len(q), success_rate=float((~np.isnan(ks)).mean()),
                                k_mean=float(found.mean()) if len(found) else None,
                                k_ci=boot_ci(found), k_hist={int(v): int((found == v).sum()) for v in np.unique(found)},
                                top_fields=sorted(fields.items(), key=lambda kv: -kv[1])[:5])
        out[m] = dict(per_class=per_class, overall_k=boot_ci(all_k),
                      overall_success=float(np.mean([v['success_rate'] for v in per_class.values()])))
    return out


def stage_hardening(D):
    """Train RF on train + CF variants of TRAIN attacks (labelled with their
    true attack class); compare evasion rate on held-out TEST attacks."""
    from sklearn.base import clone
    from sklearn.metrics import f1_score
    groups = field_groups(D['feat'], D['X_tr'])
    donors = D['X_tr'][D['y_tr'] == D['normal']]
    rng = np.random.default_rng(SEED + 7)
    rf = D['rf']
    fn = rf.predict_proba
    pred_tr = rf.predict(D['X_tr'])
    aug_X, aug_y = [], []
    t0 = time.time()
    for ci in range(len(D['classes'])):
        if ci == D['normal']:
            continue
        pool = np.where((D['y_tr'] == ci) & (pred_tr == ci))[0]
        if len(pool) == 0:
            continue
        for ti in rng.choice(pool, min(200, len(pool)), replace=False):
            k, _, xcf = cf_search(fn, D['X_tr'][ti], D['normal'], groups, donors, rng)
            if k is not None:
                aug_X.append(xcf); aug_y.append(ci)
    t_gen = time.time() - t0

    def evasion(model_fn, seed):
        r = np.random.default_rng(seed)
        p = model_fn(D['X_te']).argmax(1)
        res = {}
        for ci, c in enumerate(D['classes']):
            if ci == D['normal']:
                continue
            pool = np.where((D['y_te'] == ci) & (p == ci))[0]
            if len(pool) == 0:
                continue
            q = r.choice(pool, min(N_CF, len(pool)), replace=False)
            succ = [cf_search(model_fn, D['X_te'][ti], D['normal'], groups, donors, r)[0] is not None for ti in q]
            res[c] = float(np.mean(succ))
        return res

    # Evaluate the original RF first, then free it (and the CNN) before training the
    # hardened copy: holding two large forests at once exhausted 16 GB RAM on UNSW-NB15.
    ev0 = evasion(rf.predict_proba, SEED + 11)
    f0 = f1_score(D['y_te'], rf.predict(D['X_te']), average='macro')
    rf2 = clone(rf)
    del rf, fn
    D['rf'] = None; D['models'] = {}; D['cnn'] = None
    import gc; gc.collect()
    rf2.fit(np.vstack([D['X_tr_sm'], np.array(aug_X)]), np.concatenate([D['y_tr_sm'], aug_y]))
    ev1 = evasion(rf2.predict_proba, SEED + 11)
    f1 = f1_score(D['y_te'], rf2.predict(D['X_te']), average='macro')
    del rf2; gc.collect()
    return dict(n_augmented=len(aug_y), gen_seconds=t_gen,
                macro_f1_before=float(f0), macro_f1_after=float(f1),
                evasion_before=ev0, evasion_after=ev1,
                mean_evasion_before=float(np.mean(list(ev0.values()))),
                mean_evasion_after=float(np.mean(list(ev1.values()))))


def _shortcut_check(D, drop):
    """Retrain the RF without the `drop` features and compare macro-F1 (overall and per class).
    The original RF is evaluated first and then freed, so only one large forest is in memory."""
    from sklearn.base import clone
    from sklearn.metrics import f1_score
    idx = [i for i, f in enumerate(D['feat']) if f in drop]
    if not idx:
        return dict(note='none of these features selected', removed=[])
    keep = [i for i in range(len(D['feat'])) if i not in idx]
    p_full = D['rf'].predict(D['X_te'])
    f_full_c = f1_score(D['y_te'], p_full, average=None)
    rf2 = clone(D['rf'])
    D['rf'] = None; D['models'] = {}; D['cnn'] = None
    import gc; gc.collect()
    rf2.fit(D['X_tr_sm'][:, keep], D['y_tr_sm'])
    f_np = f1_score(D['y_te'], rf2.predict(D['X_te'][:, keep]), average=None)
    del rf2; gc.collect()
    return dict(removed=[D['feat'][i] for i in idx], ports=[D['feat'][i] for i in idx],
                macro_f1_with_ports=float(f_full_c.mean()), macro_f1_without_ports=float(f_np.mean()),
                macro_f1_with=float(f_full_c.mean()), macro_f1_without=float(f_np.mean()),
                per_class={c: dict(with_=float(a), without=float(b))
                           for c, a, b in zip(D['classes'], f_full_c, f_np)})


def stage_ports(D):
    """Port features: fixed testbed services can make ports a proxy for the label."""
    r = _shortcut_check(D, [f for f in D['feat'] if 'port' in f.lower()])
    if not r.get('removed'):
        r = dict(note='no port features selected', ports=[])
    return r


def stage_ttl(D):
    """TTL features: in UNSW-NB15 sttl takes few values, one of which (31) occurs only in benign
    traffic, so TTL can identify the sending host rather than the attack."""
    return _shortcut_check(D, [f for f in D['feat'] if 'ttl' in f.lower()])


STAGES = [('performance', stage_performance), ('global_shap', stage_global_shap),
          ('local', stage_local), ('stability', stage_stability),
          ('counterfactual', stage_counterfactual), ('hardening', stage_hardening),
          ('ports', stage_ports), ('ttl', stage_ttl)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='all', choices=['TON_IoT', 'UNSW_NB15', 'all'])
    ap.add_argument('--stage', default='all')
    a = ap.parse_args()
    for ds in (['TON_IoT', 'UNSW_NB15'] if a.dataset == 'all' else [a.dataset]):
        print(f'\n===== {ds} =====')
        D = load_everything(ds)
        print(f'  features={len(D["feat"])} test={len(D["y_te"])} classes={D["classes"]}')
        for name, fn in STAGES:
            if a.stage not in ('all', name):
                continue
            path = OUT_ROOT / ds / f'{name}.json'
            if path.exists():
                print(f'  [{name}] cached'); continue
            print(f'  [{name}] running…'); t0 = time.time()
            if D.get('rf') is None:
                D = load_everything(ds)
            res = fn(D); res['_seconds'] = time.time() - t0
            save(res, path)
            print(f'  [{name}] done in {res["_seconds"]:.0f}s')


if __name__ == '__main__':
    main()
