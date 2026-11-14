"""Paper 6: re-run ONLY the GBDT runs with the model rolled back to its best internal-validation iteration.

Why: scikit-learn's HistGradientBoostingClassifier with early stopping keeps the n_iter_no_change (20) iterations
after the best validation score; on some seeds the boosting diverges inside them (see
experiments/baseline/gbdt_rollback.py). In results_v4 this shows as low per-stream random-split GBDTs on TON-IoT
(gb_s1_rand1/2/5/7: 0.761, 0.858, 0.600, 0.770 against about 0.92).

NEW script; run_p6_v2.py (load, make_tree, run_tree, ... used by other papers' scripts) is unchanged.
Writes results_p6gb/<dataset>/<experiment>/seed<k>.json (+ _pred.npy), with the same experiment names as
results_v4: gb, gb_s1, gb_s2 (5 seeds), gb_s1_rand<r>, gb_s2_rand<r> (r = 0..9, seed r mod 5), gb_split<k> (k = 1..5,
TON-IoT and UNSW-NB15, seed k). Each record holds the rolled-back test metrics (same keys as results_v4), the
pre-rollback ('raw_*') metrics, best_iter / n_iter_raw, validation-split macro-F1 and the results_v4 macro-F1.

GBDT hyperparameters: re-tuned on validation macro-F1 with rollback (same 4-point grid and protocol as
run_p6_v2.tune_trees: all features, semantic split, seed 0) -> results_p6gb/<ds>/tune_trees.json. --keep-v4-params
uses the results_v4 choice instead.

Run from `Papers pipeline/`:
  python -m experiments.paper6_multimodal.run_p6_gb_rollback --dataset D --group tune|main|random|resplit|all [--smoke]
"""
import argparse, gc, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.paper6_multimodal import run_p6_v2 as P6  # noqa: E402
from experiments.baseline.gbdt_rollback import rollback_best_iter  # noqa: E402

OUT = ROOT / 'experiments' / 'paper6_multimodal' / 'results_p6gb'
V4 = P6.OUT                                     # results_v4 (read only)
GB_ARCHS = ('gb', 'gb_s1', 'gb_s2')


def cols_of(arch, s1, s2):
    return {'gb': np.r_[s1, s2], 'gb_s1': s1, 'gb_s2': s2}[arch]


def fit_gb(D, params, seed, cols):
    t = P6.make_tree('gb', params, seed)        # identical constructor to results_v4
    t0 = time.time(); t.fit(D['Xfit'][:, cols], D['yfit']); fit_s = time.time() - t0
    return t, fit_s


def predict(t, X, K):
    Pc = t.predict_proba(X)
    P = np.zeros((len(Pc), K)); P[:, t.classes_] = Pc
    return P, t.classes_[Pc.argmax(1)]


def run_gb(D, arch, s1, s2, seed, params):
    from sklearn.metrics import f1_score
    K = len(D['classes'])
    cols = cols_of(arch, s1, s2)
    t, fit_s = fit_gb(D, params, seed, cols)
    P, pred = predict(t, D['Xte'][:, cols], K)
    raw = P6.metrics(D, pred, P)
    info = rollback_best_iter(t)
    P, pred = predict(t, D['Xte'][:, cols], K)
    out = dict(train_s=fit_s, **P6.metrics(D, pred, P), n_iter=int(t.n_iter_), **info,
               raw_macro_f1=raw['macro_f1'], raw_clean_macro_f1=raw['clean_macro_f1'], raw_benign_fpr=raw['benign_fpr'],
               val_macro_f1=float(f1_score(D['yva'], predict(t, D['Xva'][:, cols], K)[1], average='macro')),
               params=params)
    return out, pred


def tune(D, keep_v4):
    """the run_p6_v2.tune_trees GBDT grid, scored on validation macro-F1 after rollback"""
    from sklearn.metrics import f1_score
    v4 = json.loads((V4 / D['ds'] / 'tune_trees.json').read_text())['gb']
    if keep_v4:
        return dict(grid=v4['grid'], best=v4['best'], v4_best=v4['best'], source='results_v4 (--keep-v4-params)')
    scores, iters = [], []
    for g in P6.GB_GRID:
        t = P6.make_tree('gb', g, 0).fit(D['Xfit'], D['yfit'])
        info = rollback_best_iter(t)
        scores.append(float(f1_score(D['yva'], t.predict(D['Xva']), average='macro')))
        iters.append([info['best_iter'], info['n_iter_raw']])
        print(f'    tune gb {g}: val macro-F1 {scores[-1]:.4f} (iterations kept {iters[-1][0]} of {iters[-1][1]})', flush=True)
    best = P6.GB_GRID[int(np.argmax(scores))]
    return dict(grid=P6.GB_GRID, val_macro_f1=scores, iters_kept_fitted=iters, best=best,
                v4_val_macro_f1=v4['val_macro_f1'], v4_best=v4['best'], changed=best != v4['best'],
                source='re-tuned with rollback')


def jobs(D, group):
    s1, s2 = P6.semantic_split(D)
    if group == 'main':
        for seed in P6.SEEDS:
            for arch in GB_ARCHS:
                yield arch, arch, s1, s2, seed
    elif group == 'random':
        for r in range(P6.N_RANDOM):
            r1, r2 = P6.random_split(D, r)
            for arch in ('gb_s1', 'gb_s2'):
                yield arch, f'{arch}_rand{r}', r1, r2, r % 5
    elif group == 'resplit':
        k = D['split']
        yield 'gb', f'gb_split{k}', s1, s2, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='all', choices=('all',) + P6.DATASETS)
    ap.add_argument('--group', default='all', choices=('tune', 'main', 'random', 'resplit', 'all'))
    ap.add_argument('--keep-v4-params', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='seed 0, one random split, one resplit; results_p6gb_smoke/')
    a = ap.parse_args()
    global OUT
    if a.smoke:
        OUT = OUT.parent / 'results_p6gb_smoke'
        P6.SEEDS, P6.N_RANDOM, P6.RESPLITS = (0,), 1, (1,)
    groups = ('main', 'random', 'resplit') if a.group == 'all' else (a.group,)
    for ds in P6.DATASETS if a.dataset == 'all' else (a.dataset,):
        tt = OUT / ds / 'tune_trees.json'
        need_split0 = (not tt.exists()) or any(g in ('main', 'random') for g in groups)
        if need_split0:
            D = P6.load(ds, 0)
            print(f'== {ds} split 0: {len(D["feat"])} features, fit {len(D["yfit"]):,}', flush=True)
            if not tt.exists():
                tt.parent.mkdir(parents=True, exist_ok=True)
                tt.write_text(json.dumps(dict(gb=tune(D, a.keep_v4_params)), indent=1))
            params = json.loads(tt.read_text())['gb']['best']
            for g in groups:
                if g in ('main', 'random'):
                    run_group(D, g, params)
            del D; gc.collect()
        if 'resplit' in groups and ds != 'N_BaIoT':
            params = json.loads(tt.read_text())['gb']['best']
            for k in P6.RESPLITS:
                if (OUT / ds / f'gb_split{k}' / f'seed{k}.json').exists():
                    continue
                D = P6.load(ds, k)
                print(f'== {ds} split {k}', flush=True)
                run_group(D, 'resplit', params)
                del D; gc.collect()


def run_group(D, group, params):
    ds = D['ds']
    for arch, exp, i1, i2, seed in jobs(D, group):
        path = OUT / ds / exp / f'seed{seed}.json'
        if path.exists():
            continue
        t0 = time.time()
        res, pred = run_gb(D, arch, i1, i2, seed, params)
        v4 = V4 / ds / exp / f'seed{seed}.json'
        if v4.exists():
            res['v4_macro_f1'] = json.loads(v4.read_text())['macro_f1']
        res['cfg'] = dict(arch=arch, exp=exp, seed=seed, split=D['split'], s1=[D['feat'][i] for i in i1], rollback=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_name(f'seed{seed}_pred.npy'), pred.astype(np.int8))
        path.write_text(json.dumps(res, indent=1))
        print(f"  {exp} seed{seed}: macro-F1 {res['macro_f1']:.4f} (raw {res['raw_macro_f1']:.4f}, v4 "
              f"{res.get('v4_macro_f1', float('nan')):.4f}; iterations {res['best_iter']}/{res['n_iter_raw']}) "
              f"({time.time() - t0:.0f}s)", flush=True)
        gc.collect()


if __name__ == '__main__':
    main()
