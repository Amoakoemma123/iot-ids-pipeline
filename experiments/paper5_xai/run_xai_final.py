"""
run_xai_final.py — Paper 5, final review round: small robustness runs (seed 42 unless stated).
Reuses the data, models, explainers and search code of run_xai_v3.py; v3 results are not touched.

Stages
  lime_sens         LIME configuration sensitivity (referee M5) and deletion with positive
                    attributions only (refs/code review M2). On N_SENS test records per class
                    (the first N_SENS per class of the agreement sample, same SHAP background and
                    donors as stage_deletion): LIME in continuous mode with 0.5x, 1x and 2x the
                    default kernel width (default = 0.75*sqrt(p)) and LIME's default discretised
                    mode (quartiles, default width). Per configuration: SHAP agreement, surrogate
                    R^2, deletion drops (k = 1, 3, 5; mean, record, gauss, condx). Positive-only
                    deletion: top-k by signed attribution towards the predicted class (SHAP and
                    continuous LIME at the default width).
  hardening_paired  stage_hardening(attacker_only=True), identical random streams (reproduces
                    hardening_attacker_r3), plus per-record evasion outcomes on the common query set
                    (paired tests, referee M1), benign FPR near the added records (referee M2),
                    and whether each query record's non-controllable features already lie within
                    the support radius of a benign training record (referee M2). Seeds 42-44.
  decomp_ci         UNSW only: the seed-42 fixed-feature id decomposition again, with per-record
                    predictions, for paired accuracy CIs, an interaction CI and a common test
                    population (referee M3).
  cf_monotone       TON only: attacker-controllable counterfactuals where source volume and
                    duration may only increase (referee minor 8), RF and CNN.

    python -m experiments.paper5_xai.run_xai_final --stage all [--dataset all] [--smoke]
Results: results_final/<ds>/[seed<s>/]<stage>.json (results_final_smoke/ with --smoke). Finished
stages are skipped, so the queue can be restarted.
"""
import argparse, gc, io, json, sys, time, contextlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import experiments.paper5_xai.run_xai_v3 as X3
from experiments.paper5_xai.run_xai_v3 import (load_data, load_rf, load_cnn, cnn_fn, shap_cnn, expected_jaccard,
                                               paired_ci, dependency_groups, attacker_groups, Support, cf_search,
                                               _replacements, DEL_K, N_LOCAL, N_HARD_GEN, N_CF, CF_MAX_K, CF_TRIES,
                                               ATTACKER_SPEC)
from experiments.paper5_xai.run_xai_v2 import boot_ci, jaccard, topk, shap_rf, stratified_idx, save

OUT_ROOT = ROOT / 'experiments' / 'paper5_xai' / 'results_final'
N_SENS = 15            # records per class for lime_sens
STRATS = ('mean', 'record', 'gauss', 'condx')
SMOKE = False


def topk_pos(att, k):
    """Top-k features by signed attribution (largest contributions towards the predicted class)."""
    return list(np.argsort(-np.asarray(att))[:k])


def lime_configs(nf):
    w0 = 0.75 * np.sqrt(nf)            # LIME's default kernel width
    return {'cont_w0.5': dict(discretize_continuous=False, kernel_width=0.5 * w0),
            'cont_w1': dict(discretize_continuous=False, kernel_width=w0),
            'cont_w2': dict(discretize_continuous=False, kernel_width=2 * w0),
            'disc': dict(discretize_continuous=True)}


def make_lime(D, seed, cfg):
    from lime.lime_tabular import LimeTabularExplainer
    return LimeTabularExplainer(D['X_tr'], feature_names=D['feat'], class_names=D['classes'], random_state=seed, **cfg)


def lime_attr_cfg(expl, x, fn, label, nf, continuous, ns=1000):
    """Continuous: contribution w*(x-mu)/sd (the paper's primary ranking). Discretised: the weight of
    the bin indicator the record falls into (already a contribution relative to the other bins)."""
    e = expl.explain_instance(x, fn, labels=[label], num_features=nf, num_samples=ns)
    w = np.zeros(nf)
    for j, v in e.as_map()[label]:
        w[j] = v
    if continuous:
        w = w * (x - expl.scaler.mean_) / expl.scaler.scale_
    return w, float(e.score)


# ── stage: LIME sensitivity + positive-only deletion ─────────────────────────

def stage_lime_sens(D, seed):
    idx_all = stratified_idx(D['y_te'], N_LOCAL, seed=42)
    per = 1 if SMOKE else N_SENS
    idx = np.array(sorted(i for c in np.unique(D['y_te']) for i in [j for j in idx_all if D['y_te'][j] == c][:per]))
    # same random streams as stage_deletion: background, then donors
    rng = np.random.default_rng(seed)
    bg = D['X_tr'][rng.choice(len(D['X_tr']), 200, replace=False)]
    mean_vec = D['X_tr'].mean(0); sd_vec = D['X_tr'].std(0)
    donors = D['X_tr'][rng.choice(len(D['X_tr']), 5000, replace=False)]
    ctx = (mean_vec, sd_vec, donors)
    nf = len(D['feat'])
    cfgs = lime_configs(nf)
    expls = {name: make_lime(D, seed, cfg) for name, cfg in cfgs.items()}
    X, y = D['X_te'][idx], D['y_te'][idx]
    out = dict(n_per_class=per, configs={k: {kk: (float(vv) if not isinstance(vv, bool) else vv) for kk, vv in v.items()} for k, v in cfgs.items()},
               default_kernel_width=float(0.75 * np.sqrt(nf)), chance_j5=expected_jaccard(nf, 5), strategies=STRATS)
    for m in ('rf', 'cnn'):
        if m == 'rf':
            model = load_rf(D['ds'], seed); fn = model.predict_proba; sv = shap_rf(model, bg, X)
        else:
            model = load_cnn(D['ds'], seed); fn = cnn_fn(model); sv = shap_cnn(model, bg, X, rseed=seed)
        pred = fn(X).argmax(1)
        drng = np.random.default_rng(seed + 700)
        rows = []
        for i in range(len(X)):
            s = sv[i, :, pred[i]]
            atts, row = {'shap': s}, dict(test_idx=int(idx[i]), true=int(y[i]), pred=int(pred[i]),
                                           shap_top5=[int(j) for j in topk(s, 5)])
            for name, expl in expls.items():
                l, r2 = lime_attr_cfg(expl, X[i], fn, int(pred[i]), nf, continuous=not name.startswith('disc'))
                atts[name] = l
                row[f'{name}_j5'] = jaccard(topk(s, 5), topk(l, 5)); row[f'{name}_r2'] = r2
                row[f'{name}_top5'] = [int(j) for j in topk(l, 5)]
                row[f'{name}_npos5'] = int(sum(l[j] > 0 for j in topk(l, 5)))
            row['shap_npos5'] = int(sum(s[j] > 0 for j in topk(s, 5)))
            batches, keys = [], []
            for k in DEL_K:
                for name, att in atts.items():
                    for st, Z in _replacements(X[i], topk(att, k), ctx, drng).items():
                        if st in STRATS:
                            batches.append(Z); keys.append((name, st, k, len(Z)))
                for name in ('shap', 'cont_w1'):                       # positive-only ranking
                    for st, Z in _replacements(X[i], topk_pos(atts[name], k), ctx, drng).items():
                        if st in STRATS:
                            batches.append(Z); keys.append((name + '_pos', st, k, len(Z)))
                for r in range(10):
                    js = drng.choice(nf, k, replace=False)
                    for st, Z in _replacements(X[i], js, ctx, drng, n=1).items():
                        if st in STRATS:
                            batches.append(Z); keys.append(('rand', st, k, len(Z)))
            P = fn(np.vstack([X[i][None]] + batches))[:, pred[i]]
            p0, pos, acc = P[0], 1, {}
            for (name, st, k, n) in keys:
                acc.setdefault((name, st, k), []).append(p0 - P[pos:pos + n].mean()); pos += n
            for (name, st, k), v in acc.items():
                row[f'{st}_{name}_{k}'] = float(np.mean(v))
            row['p0'] = float(p0)
            rows.append(row)
        skip = ('test_idx', 'true', 'pred') + tuple(k for k in rows[0] if k.endswith('_top5'))
        summ = {k: boot_ci([r[k] for r in rows]) for k in rows[0] if k not in skip}
        cc = lambda v: (v - out['chance_j5']) / (1 - out['chance_j5'])
        for name in cfgs:
            summ[f'{name}_j5_chance_corrected'] = boot_ci([cc(r[f'{name}_j5']) for r in rows])
        paired = {}
        for st in STRATS:
            for k in DEL_K:
                for name in cfgs:
                    paired[f'{st}_shap_minus_{name}_{k}'] = paired_ci([r[f'{st}_shap_{k}'] for r in rows], [r[f'{st}_{name}_{k}'] for r in rows])
                    paired[f'{st}_{name}_minus_rand_{k}'] = paired_ci([r[f'{st}_{name}_{k}'] for r in rows], [r[f'{st}_rand_{k}'] for r in rows])
                paired[f'{st}_shappos_minus_limepos_{k}'] = paired_ci([r[f'{st}_shap_pos_{k}'] for r in rows], [r[f'{st}_cont_w1_pos_{k}'] for r in rows])
                paired[f'{st}_shappos_minus_shap_{k}'] = paired_ci([r[f'{st}_shap_pos_{k}'] for r in rows], [r[f'{st}_shap_{k}'] for r in rows])
                paired[f'{st}_limepos_minus_lime_{k}'] = paired_ci([r[f'{st}_cont_w1_pos_{k}'] for r in rows], [r[f'{st}_cont_w1_{k}'] for r in rows])
        out[m] = dict(summary=summ, paired=paired, rows=rows, n=len(rows))
        del model, fn; gc.collect()
        if m == 'cnn':
            import tensorflow as tf; tf.keras.backend.clear_session()
    return out


# ── stage: hardening with per-record outcomes, local FPR and prior benign proximity ─

def stage_hardening_paired(D, seed):
    """Copy of run_xai_v3.stage_hardening(attacker_only=True) with the same random streams in the
    same order (so evasion rates reproduce hardening_attacker_r3), plus extra outputs computed
    without touching those streams."""
    from sklearn.base import clone
    from sklearn.metrics import f1_score
    from sklearn.neighbors import NearestNeighbors
    groups = attacker_groups(D['ds'], D['feat'])
    benign = D['X_tr'][D['y_tr'] == D['normal']]
    rng = np.random.default_rng(seed + 1000)
    perm = rng.permutation(len(benign)); poolA, poolB = benign[perm[::2]], benign[perm[1::2]]
    supA, supB = Support(poolA, seed), Support(poolB, seed + 1)
    attack_classes = [ci for ci in range(len(D['classes'])) if ci != D['normal']]
    n_gen = 3 if SMOKE else N_HARD_GEN

    rf = load_rf(D['ds'], seed)
    pred_tr = rf.predict(D['X_tr'])
    aug_X, aug_y, aug_k = [], [], []
    for ci in attack_classes:
        pool = np.where((D['y_tr'] == ci) & (pred_tr == ci))[0]
        for ti in rng.choice(pool, min(n_gen, len(pool)), replace=False) if len(pool) else []:
            r = cf_search(rf.predict_proba, D['X_tr'][ti], D['normal'], groups, poolA, rng, supA)
            if r['k'] is not None:
                aug_X.append(r['x_cf']); aug_y.append(ci); aug_k.append(r['k'])
    aug_X, aug_y = np.array(aug_X), np.array(aug_y)
    ctl_X = []
    for ci in aug_y:
        pool = np.where(D['y_tr'] == ci)[0]
        a, b = D['X_tr'][rng.choice(pool, 2)]
        lam = rng.random()
        ctl_X.append(lam * a + (1 - lam) * b)
    ctl_X = np.array(ctl_X)
    crng = np.random.default_rng(seed + 3000)
    cpy_X, n_cpy_unsupported = [], 0
    for ci, k in zip(aug_y, aug_k):
        pool = np.where(D['y_tr'] == ci)[0]
        for _ in range(50):
            z = D['X_tr'][crng.choice(pool)].copy()
            d = poolA[crng.integers(len(poolA))]
            for g in crng.choice(len(groups), min(k, len(groups)), replace=False):
                z[groups[g][1]] = d[groups[g][1]]
            if supA.ok(z[None])[0]:
                break
        else:
            n_cpy_unsupported += 1
        cpy_X.append(z)
    cpy_X = np.array(cpy_X)
    p_orig = rf.predict(D['X_te'])
    sup_idx = []
    n_q = 3 if SMOKE else 2 * N_CF
    for ci in attack_classes:
        pool = np.where((D['y_te'] == ci) & (p_orig == ci))[0]
        if len(pool):
            sup_idx.extend(rng.choice(pool, min(n_q, len(pool)), replace=False))
    sup_idx = np.array(sup_idx)

    # benign test records near the added records (no random stream involved)
    ben_te = np.where(D['y_te'] == D['normal'])[0]
    near = {}
    for name, A in (('cf', aug_X), ('copy', cpy_X)):
        if not len(A):
            near[name] = dict(tau=np.array([], int), knn5=np.array([], int)); continue
        dA, _ = NearestNeighbors(n_neighbors=1).fit(A).kneighbors(D['X_te'][ben_te])
        nnb = NearestNeighbors(n_neighbors=min(5, len(ben_te))).fit(D['X_te'][ben_te])
        _, kb = nnb.kneighbors(A)
        near[name] = dict(tau=ben_te[dA[:, 0] <= supA.tau], knn5=ben_te[np.unique(kb.ravel())])

    def evaluate(model):
        p = model.predict(D['X_te'])
        erng = np.random.default_rng(seed + 2000)
        succ = {}
        for ti in sup_idx:
            if p[ti] == D['y_te'][ti]:
                succ[int(ti)] = cf_search(model.predict_proba, D['X_te'][ti], D['normal'], groups, poolB, erng, supB)['k'] is not None
        loc = {f'{a}_{b}': dict(n=int(len(ix)), fpr=float((p[ix] != D['normal']).mean()) if len(ix) else None)
               for a, v in near.items() for b, ix in v.items()}
        return dict(macro_f1=float(f1_score(D['y_te'], p, average='macro')),
                    benign_fpr=float((p[D['y_te'] == D['normal']] != D['normal']).mean()),
                    attack_recall=float((p[D['y_te'] != D['normal']] != D['normal']).mean()),
                    detected=set(int(t) for t in sup_idx if p[t] == D['y_te'][t]), succ=succ, local_fpr=loc)

    res = {'original': evaluate(rf)}
    template = clone(rf)
    if SMOKE:
        template.set_params(n_estimators=5)
    del rf; gc.collect()
    for name, Xa in (('counterfactual', aug_X), ('control', ctl_X), ('control_copy', cpy_X)):
        m = clone(template)
        m.fit(np.vstack([D['X_tr_sm'], Xa]), np.concatenate([D['y_tr_sm'], aug_y]))
        res[name] = evaluate(m)
        del m; gc.collect()
    common = sorted(set.intersection(*(r['detected'] for r in res.values())))
    # non-controllable features of each query record already within the support radius of a benign
    # training record (distance on those features only, radius tau of evaluation pool B)
    ctrl = sorted({j for _, idx in groups for j in idx})
    nonc = [j for j in range(len(D['feat'])) if j not in ctrl]
    dn, _ = NearestNeighbors(n_neighbors=1).fit(benign[:, nonc]).kneighbors(D['X_te'][common][:, nonc]) if common else (np.zeros((0, 1)), None)
    out = dict(groups=[g[0] for g in groups], n_augmented=int(len(aug_y)), n_query=len(common),
               n_copy_unsupported=n_cpy_unsupported, tau_A=supA.tau, tau_B=supB.tau,
               query_per_class={D['classes'][ci]: int(sum(D['y_te'][t] == ci for t in common)) for ci in attack_classes},
               near_sets={f'{a}_{b}': int(len(ix)) for a, v in near.items() for b, ix in v.items()},
               records=dict(test_idx=common, cls=[int(D['y_te'][t]) for t in common],
                            noncontrollable_within_tau=[bool(x) for x in (dn[:, 0] <= supB.tau)],
                            noncontrollable_dist=[float(x) for x in dn[:, 0]],
                            **{f'evaded_{name}': [bool(r['succ'][t]) for t in common] for name, r in res.items()}))
    for name, r in res.items():
        ev = [r['succ'][t] for t in common]
        per_class = {}
        for ci in attack_classes:
            e = [r['succ'][t] for t in common if D['y_te'][t] == ci]
            if e:
                per_class[D['classes'][ci]] = float(np.mean(e))
        out[name] = dict(macro_f1=r['macro_f1'], benign_fpr=r['benign_fpr'], attack_recall=r['attack_recall'],
                         evasion=float(np.mean(ev)) if ev else None, evasion_ci=boot_ci(np.array(ev, float)),
                         evasion_per_class=per_class, local_fpr=r['local_fpr'])
    return out


# ── stage: fixed-feature id decomposition with per-record predictions (UNSW, seed 42) ─

def stage_decomp_ci(D, seed):
    from experiments.paper5_xai.leakage_decomposition_fixed import pipeline_for, CSV, LABEL
    from src.data import load_dataset
    from src.models import build_rf
    from sklearn.metrics import f1_score
    import joblib
    base = joblib.load(ROOT / 'experiments' / 'baseline' / 'results' / 'UNSW_NB15' / 'pipeline.pkl')
    feats = list(base.selected_features_)
    out = dict(features=feats, seed=seed)
    for pair, dedup in (('dups_kept', False), ('dups_removed', True)):
        with contextlib.redirect_stdout(io.StringIO()):
            raw = load_dataset(CSV, label_col=LABEL, seed=42, drop_id=False, deduplicate=dedup)
        names = raw['feature_names']
        Xtr_raw, ytr_raw = raw['X_train'], raw['y_train']
        if SMOKE:
            sub = np.random.default_rng(0).choice(len(ytr_raw), 5000, replace=False); Xtr_raw, ytr_raw = Xtr_raw[sub], ytr_raw[sub]
        preds = {}
        for cond, cols in (('with_id', feats + ['id']), ('without_id', feats)):
            p = pipeline_for(names, cols, Xtr_raw)
            with contextlib.redirect_stdout(io.StringIO()):
                Xtr, ytr = p.transform(Xtr_raw, apply_smote=True, y=ytr_raw)
            Xte, _ = p.transform(raw['X_test'])
            rf = build_rf(seed=seed)
            if SMOKE:
                rf.set_params(n_estimators=5)
            rf.fit(Xtr, ytr)
            preds[cond] = rf.predict(Xte).astype(int).tolist()
            del rf, Xtr; gc.collect()
        out[pair] = dict(test_id=raw['X_test'][:, names.index('id')].astype(np.int64).tolist(),
                         y=raw['y_test'].astype(int).tolist(), pred=preds,
                         macro_f1={c: float(f1_score(raw['y_test'], v, average='macro')) for c, v in preds.items()})
        del raw; gc.collect()
    return out


# ── stage: monotone attacker-controllable counterfactuals (TON) ──────────────

MONO_GROUPS = ('src_volume', 'duration')        # may only increase


def stage_cf_monotone(D, seed):
    """Attacker-controllable search as in stage_counterfactual, but a donor's values for a
    MONO_GROUPS group are accepted only if no feature of that group decreases."""
    agroups = attacker_groups(D['ds'], D['feat'])
    benign = D['X_tr'][D['y_tr'] == D['normal']]
    support = Support(benign, seed)
    rng = np.random.default_rng(seed + 4000)
    mono = [i for i, (g, _) in enumerate(agroups) if g in MONO_GROUPS]
    out = dict(groups={g: [D['feat'][j] for j in idx] for g, idx in agroups}, monotone=list(MONO_GROUPS))
    n_cf = 3 if SMOKE else N_CF
    for m in ('rf', 'cnn'):
        model = load_rf(D['ds'], seed) if m == 'rf' else load_cnn(D['ds'], seed)
        fn = model.predict_proba if m == 'rf' else cnn_fn(model)
        pred_te = fn(D['X_te']).argmax(1)
        rows = []
        for ci in range(len(D['classes'])):
            if ci == D['normal']:
                continue
            pool = np.where((D['y_te'] == ci) & (pred_te == ci))[0]
            for ti in (rng.choice(pool, min(n_cf, len(pool)), replace=False) if len(pool) else []):
                x = D['X_te'][ti]
                found = {}
                for mode in ('free', 'monotone'):
                    G = list(range(len(agroups))); hit_k = None; q = 0
                    for k in range(1, min(CF_MAX_K, len(G)) + 1):
                        cand = np.repeat(x[None], CF_TRIES, 0); ok_mono = np.ones(CF_TRIES, bool)
                        for t in range(CF_TRIES):
                            gs = list(rng.choice(G, k, replace=False)); d = benign[rng.integers(len(benign))]
                            for g in gs:
                                cols = agroups[g][1]; cand[t, cols] = d[cols]
                                if mode == 'monotone' and g in mono and np.any(d[cols] < x[cols]):
                                    ok_mono[t] = False
                        hit = (fn(cand).argmax(1) == D['normal']) & support.ok(cand) & ok_mono
                        if hit.any():
                            hit_k = k; q += int(np.argmax(hit)) + 1; break
                        q += CF_TRIES
                    found[mode] = dict(k=hit_k, queries=q)
                rows.append(dict(test_idx=int(ti), cls=ci, **{f'{md}_k': v['k'] for md, v in found.items()}))
        summ = {}
        for md in ('free', 'monotone'):
            succ = np.array([r[f'{md}_k'] is not None for r in rows], float)
            summ[md] = dict(success=boot_ci(succ),
                            per_class={D['classes'][ci]: dict(n=int(sum(r['cls'] == ci for r in rows)),
                                                              success=float(np.mean([r[f'{md}_k'] is not None for r in rows if r['cls'] == ci])))
                                       for ci in sorted({r['cls'] for r in rows})})
        summ['paired_monotone_minus_free'] = paired_ci([r['monotone_k'] is not None for r in rows], [r['free_k'] is not None for r in rows])
        out[m] = dict(summary=summ, rows=rows)
        del model, fn; gc.collect()
        if m == 'cnn':
            import tensorflow as tf; tf.keras.backend.clear_session()
    return out


JOBS = [('decomp_ci', stage_decomp_ci, ('UNSW_NB15',), (42,)),
        ('lime_sens', stage_lime_sens, ('TON_IoT', 'UNSW_NB15'), (42,)),
        ('cf_monotone', stage_cf_monotone, ('TON_IoT',), (42,)),
        ('hardening_paired', stage_hardening_paired, ('TON_IoT', 'UNSW_NB15'), (42, 43, 44))]


def main():
    global SMOKE, OUT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='all', choices=['TON_IoT', 'UNSW_NB15', 'all'])
    ap.add_argument('--stage', default='all')
    ap.add_argument('--seeds', default='42,43,44')
    ap.add_argument('--smoke', action='store_true', help='tiny subsamples, output to results_final_smoke/')
    a = ap.parse_args()
    SMOKE = a.smoke
    if SMOKE:
        OUT_ROOT = OUT_ROOT.parent / 'results_final_smoke'
    seeds = {int(s) for s in a.seeds.split(',')}
    for ds in ('TON_IoT', 'UNSW_NB15'):
        if a.dataset not in ('all', ds):
            continue
        todo = [(n, f, s) for n, f, dss, ss in JOBS for s in ss if ds in dss and s in seeds and a.stage in ('all', n)
                and not (OUT_ROOT / ds / f'seed{s}' / f'{n}.json').exists()]
        if not todo:
            print(f'{ds}: nothing to do'); continue
        needs_data = any(n != 'decomp_ci' for n, _, _ in todo)
        D = load_data(ds) if needs_data else dict(ds=ds)
        print(f'\n===== {ds}', flush=True)
        for name, fn, s in todo:
            path = OUT_ROOT / ds / f'seed{s}' / f'{name}.json'
            print(f'  [{name} seed={s}] running…', flush=True); t0 = time.time()
            res = fn(D, s)
            res['_seconds'] = time.time() - t0
            save(res, path)
            print(f'  [{name} seed={s}] done in {res["_seconds"]:.0f}s', flush=True)
            gc.collect()
        X3._RAW.clear(); del D; gc.collect()


if __name__ == '__main__':
    main()
