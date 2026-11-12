"""
run_xai_transfer.py - Paper 5, ESWA revision (2026-11-12): do the hardening results transfer to attacks
other than the one that generated the training data?

The novelty check found that generation and evaluation in stage_hardening_paired use the same search
(random dependency-group subsets copied from one benign donor). Here the four RF models of that stage
(original, counterfactual-augmented, interpolation control, copy control) are rebuilt with IDENTICAL random
streams (the original attack is re-run first and must reproduce the saved per-record outcomes exactly),
and then attacked on the same common query records with two different searches. Both keep the
attacker-controllable threat model (ATTACKER_SPEC) and the benign-support check of evaluation pool B:

  greedy   donors = benign TEST records (never used for generation or evaluation before); per donor,
           copy the attacker-controllable groups one at a time, each step taking the group that lowers the
           model's attack probability most; success if some donor reaches a benign prediction within support.
           Score-guided instead of random subsets, new donor population.
  interp   partial moves: x' = x + lam (d - x) on every non-empty subset of the groups, lam in
           {0.25, 0.5, 0.75}, donors = benign test records. Produces values that no benign record has, so a
           model that only memorised benign values in these fields is not protected by construction.

Output: results_transfer/<ds>/seed<s>/hardening_transfer.json with evasion rates, bootstrap CIs and
per-record outcomes per model x attack, and exact McNemar tests (counterfactual vs copy control,
counterfactual vs original, copy control vs original) per attack.

    python -m experiments.paper5_xai.run_xai_transfer [--dataset all] [--seeds 42,43,44] [--smoke]
Finished (dataset, seed) files are skipped, so the queue can be restarted.
"""
import argparse, gc, itertools, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import experiments.paper5_xai.run_xai_v3 as X3
from experiments.paper5_xai.run_xai_v3 import (load_data, load_rf, attacker_groups, Support, cf_search,
                                               N_HARD_GEN, N_CF)
from experiments.paper5_xai.run_xai_v2 import boot_ci, save

OUT_ROOT = ROOT / 'experiments' / 'paper5_xai' / 'results_transfer'
REF_ROOT = ROOT / 'experiments' / 'paper5_xai' / 'results_final'
N_DONOR = 100                 # benign test donors per query record
LAMS = (0.25, 0.5, 0.75)
SMOKE = False


def mcnemar(a, b):
    """exact two-sided McNemar test on paired booleans; returns (n a-only, n b-only, p)."""
    from scipy.stats import binomtest
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return n10, n01, float(p)


def attack_greedy(model, x, groups, donors, sup, normal):
    """greedy, score-guided copying of attacker-controllable groups from each donor (batched over donors)."""
    cur = np.repeat(x[None], len(donors), 0)
    left = [set(range(len(groups))) for _ in donors]
    alive = np.ones(len(donors), bool)
    for _ in range(len(groups)):
        idx = np.where(alive)[0]
        if not len(idx):
            break
        cand, owner, grp = [], [], []
        for i in idx:
            for g in left[i]:
                z = cur[i].copy(); z[groups[g][1]] = donors[i, groups[g][1]]
                cand.append(z); owner.append(i); grp.append(g)
        cand = np.array(cand); owner = np.array(owner); grp = np.array(grp)
        pr = model.predict_proba(cand)
        feas = sup.ok(cand)
        win = (pr.argmax(1) == normal) & feas
        if win.any():
            return True
        p_att = 1 - pr[:, normal]
        for i in idx:                      # take the group that lowers the attack probability most
            m = np.where(owner == i)[0]
            j = m[np.argmin(p_att[m])]
            cur[i] = cand[j]; left[i].discard(int(grp[j]))
            if not left[i]:
                alive[i] = False
    return False


def attack_interp(model, x, groups, donors, sup, normal):
    """partial moves towards each donor on every non-empty group subset, lam in LAMS (one batch)."""
    subsets = [s for r in range(1, len(groups) + 1) for s in itertools.combinations(range(len(groups)), r)]
    cand = []
    for d in donors:
        for s in subsets:
            cols = np.concatenate([groups[g][1] for g in s])
            for lam in LAMS:
                z = x.copy(); z[cols] = x[cols] + lam * (d[cols] - x[cols]); cand.append(z)
    cand = np.array(cand)
    win = (model.predict_proba(cand).argmax(1) == normal) & sup.ok(cand)
    return bool(win.any())


def stage_hardening_transfer(D, seed):
    """Rebuild the models of run_xai_final.stage_hardening_paired with the same random streams in the same
    order (copied line by line), then attack them with the original, greedy and interp searches."""
    from sklearn.base import clone
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
    cpy_X = []
    for ci, k in zip(aug_y, aug_k):
        pool = np.where(D['y_tr'] == ci)[0]
        for _ in range(50):
            z = D['X_tr'][crng.choice(pool)].copy()
            d = poolA[crng.integers(len(poolA))]
            for g in crng.choice(len(groups), min(k, len(groups)), replace=False):
                z[groups[g][1]] = d[groups[g][1]]
            if supA.ok(z[None])[0]:
                break
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

    models = {'original': rf}
    template = clone(rf)
    if SMOKE:
        template.set_params(n_estimators=5)
    for name, Xa in (('counterfactual', aug_X), ('control', ctl_X), ('control_copy', cpy_X)):
        m = clone(template)
        m.fit(np.vstack([D['X_tr_sm'], Xa]), np.concatenate([D['y_tr_sm'], aug_y]))
        models[name] = m
    preds = {n: m.predict(D['X_te']) for n, m in models.items()}
    detected = {n: set(int(t) for t in sup_idx if preds[n][t] == D['y_te'][t]) for n in models}
    common = sorted(set.intersection(*detected.values()))

    # original attack, same evaluation stream as before -> must reproduce the saved outcomes
    orig = {}
    for n, m in models.items():
        erng = np.random.default_rng(seed + 2000)
        succ = {}
        for ti in sup_idx:
            if preds[n][ti] == D['y_te'][ti]:
                succ[int(ti)] = cf_search(m.predict_proba, D['X_te'][ti], D['normal'], groups, poolB, erng, supB)['k'] is not None
        orig[n] = [succ[t] for t in common]
    reproduced = None
    ref = REF_ROOT / D['ds'] / f'seed{seed}' / 'hardening_paired.json'
    if not SMOKE and ref.exists():
        R = json.loads(ref.read_text())['records']
        reproduced = (R['test_idx'] == common and all(R[f'evaded_{n}'] == orig[n] for n in models))

    # new attacks: donors = benign test records, one fixed donor sample per query record (shared by all models)
    ben_te = D['X_te'][D['y_te'] == D['normal']]
    drng = np.random.default_rng(seed + 4000)
    n_don = 5 if SMOKE else N_DONOR
    donor_idx = {t: drng.choice(len(ben_te), n_don, replace=False) for t in common}
    out_att = {'original_search': orig}
    for aname, fn in (('greedy', attack_greedy), ('interp', attack_interp)):
        out_att[aname] = {}
        for n, m in models.items():
            t0 = time.time()
            out_att[aname][n] = [fn(m, D['X_te'][t], groups, ben_te[donor_idx[t]], supB, D['normal']) for t in common]
            print(f'    {aname:6s} {n:15s} evasion {np.mean(out_att[aname][n]):.3f}  ({time.time() - t0:.0f}s)', flush=True)

    res = dict(groups=[g[0] for g in groups], n_query=len(common), n_augmented=int(len(aug_y)),
               reproduces_hardening_paired=reproduced, n_donor=n_don, lams=list(LAMS),
               records=dict(test_idx=common, cls=[int(D['y_te'][t]) for t in common]))
    for aname, per in out_att.items():
        res[aname] = {n: dict(evasion=float(np.mean(v)), evasion_ci=boot_ci(np.array(v, float)), evaded=[bool(x) for x in v])
                      for n, v in per.items()}
        res[aname]['mcnemar'] = {f'{a}_vs_{b}': dict(zip(('a_only', 'b_only', 'p'), mcnemar(per[a], per[b])))
                                 for a, b in (('counterfactual', 'control_copy'), ('counterfactual', 'original'),
                                              ('control_copy', 'original'), ('counterfactual', 'control'))}
    del models; gc.collect()
    return res


def main():
    global SMOKE, OUT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='all', choices=['TON_IoT', 'UNSW_NB15', 'all'])
    ap.add_argument('--seeds', default='42,43,44')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    SMOKE = a.smoke
    if SMOKE:
        OUT_ROOT = OUT_ROOT.parent / 'results_transfer_smoke'
    for ds in ('TON_IoT', 'UNSW_NB15'):
        if a.dataset not in ('all', ds):
            continue
        todo = [int(s) for s in a.seeds.split(',') if not (OUT_ROOT / ds / f'seed{s}' / 'hardening_transfer.json').exists()]
        if not todo:
            print(f'{ds}: nothing to do'); continue
        D = load_data(ds)
        for s in todo:
            print(f'\n===== {ds} seed {s}', flush=True); t0 = time.time()
            res = stage_hardening_transfer(D, s)
            res['_seconds'] = time.time() - t0
            save(res, OUT_ROOT / ds / f'seed{s}' / 'hardening_transfer.json')
            print(f'  done in {res["_seconds"]:.0f}s, reproduces={res["reproduces_hardening_paired"]}', flush=True)
            gc.collect()
        X3._RAW.clear(); del D; gc.collect()


if __name__ == '__main__':
    main()
