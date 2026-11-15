"""Paper 6: latency and size of the GBDT the paper scores (final review round 2, 2026-11-14).

bench_latency.py (3 Nov) timed a GBDT built from the results_v4 tuned parameters without rollback. The paper's
GBDT rows come from results_p6gb: parameters re-tuned with rollback (results_p6gb/<ds>/tune_trees.json) and every
model evaluated at its best internal-validation iteration. This script refits that model (all features, seed 0),
rolls it back, checks that it reproduces results_p6gb/<ds>/gb/seed0.json, and times it with the protocol of
bench_latency.py (time_calls: single record, median of 5 x 200 calls; 1,000 records, mean of 20 calls), with
scikit-learn's default threads (as in bench_latency.py) and with one thread.
Output: results_p6gb/latency_gb.json.   Run (idle machine):  python -m experiments.paper6_multimodal.bench_p6_gb_rollback
"""
import json
import os
import pickle
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.paper6_multimodal import run_p6_v2 as R  # noqa: E402
from experiments.paper6_multimodal.bench_latency import cpu_name, time_calls  # noqa: E402
from experiments.paper6_multimodal.run_p6_gb_rollback import OUT, fit_gb, predict, cols_of  # noqa: E402
from experiments.baseline.gbdt_rollback import rollback_best_iter  # noqa: E402


def main():
    from threadpoolctl import threadpool_limits
    out = {'cpu': cpu_name(), 'threads': os.cpu_count()}
    for ds in R.DATASETS:
        D = R.load(ds)
        s1, s2 = R.semantic_split(D)
        params = json.loads((OUT / ds / 'tune_trees.json').read_text())['gb']['best']
        cols = cols_of('gb', s1, s2)
        t, _ = fit_gb(D, params, 0, cols)
        info = rollback_best_iter(t)
        ref = json.loads((OUT / ds / 'gb' / 'seed0.json').read_text())
        _, pred = predict(t, D['Xte'][:, cols], len(D['classes']))
        same = bool(np.array_equal(pred.astype(np.int8), np.load(OUT / ds / 'gb' / 'seed0_pred.npy')))
        rng = np.random.default_rng(0)
        X = D['Xte'][rng.choice(len(D['yte']), 1000, replace=False)][:, cols]
        res = dict(params=params, n_iter=int(t.n_iter_), best_iter=info['best_iter'], reproduces_seed0=same,
                   seed0_macro_f1=ref['macro_f1'], size_mb=len(pickle.dumps(t)) / 2 ** 20,
                   n_trees=int(sum(len(p) for p in t._predictors)))
        res['single_ms'] = time_calls(lambda x: t.predict(x), [X[j:j + 1] for j in range(200)])
        res['batch1000_ms'] = time_calls(lambda x: t.predict(x), [X] * 20, repeats=1)
        with threadpool_limits(1):
            res['single_ms_1thread'] = time_calls(lambda x: t.predict(x), [X[j:j + 1] for j in range(200)])
            res['batch1000_ms_1thread'] = time_calls(lambda x: t.predict(x), [X] * 20, repeats=1)
        out[ds] = res
        print(ds, res, flush=True)
        del D, t
    p = OUT / 'latency_gb.json'
    p.write_text(json.dumps(out, indent=1))
    print('saved', p)


if __name__ == '__main__':
    main()
