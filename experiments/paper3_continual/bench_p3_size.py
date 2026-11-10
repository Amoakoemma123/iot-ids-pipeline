"""Paper 3 pre-submission check: model size and prediction cost of the random forests against the 1D-CNN.

For each dataset (semantic order, seed 0) this rebuilds the final random forest of 'rf_er2000' (retrained after the
last task on the ER-2000 memory plus the last task; same memory update as rf_stagewise) and of the all-data forest
(last stage of 'rf_joint' = rf_reference), checks that their test macro-F1 equals the stored run, and reports:
total tree nodes, serialised size (pickle, KiB), and single-threaded prediction time per 10,000 test records
(median of 5 repeats). The CNN is the same architecture (build_model); its prediction time does not depend on the
weights, so a freshly initialised model is timed (single thread, batch 8,192). Timings are indicative: the laptop
was running another experiment at the same time.
Run:  python -m experiments.paper3_continual.bench_p3_size   ->  results_p3final/size_bench.json
"""
import json
import os
import pickle
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np                                                   # noqa: E402

from experiments.paper3_continual import run_cl_v2 as R              # noqa: E402

N_REP = 5


def time_predict(f, X):
    n = min(10000, len(X)); Xs = X[:n]
    f(Xs)                                          # warm-up
    ts = []
    for _ in range(N_REP):
        t0 = time.perf_counter(); f(Xs); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 10000 / n


def final_forest(D, S, T, seed, mode):
    from sklearn.ensemble import RandomForestClassifier
    buf = R.Buffer(2000, np.random.default_rng(seed + 1)) if mode == 'memory' else None
    rf = None
    for t, task in enumerate(T):
        if mode == 'joint':
            if t < len(T) - 1:
                continue
            tr = np.concatenate([T[k]['tr'] for k in range(t + 1)])
            Xf, yf = S['tr'][tr], D['ytr'][tr]
        else:
            Xt, yt = S['tr'][task['tr']], D['ytr'][task['tr']]
            Xf, yf = (Xt, yt) if buf.y is None else (np.vstack([Xt, buf.X]), np.concatenate([yt, buf.y]))
        if t == len(T) - 1:
            rf = RandomForestClassifier(200, max_depth=20, min_samples_leaf=2, n_jobs=4, random_state=seed).fit(Xf, yf)
        if buf is not None:
            buf.add(Xt, yt)
    return rf


def main():
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
    from sklearn.metrics import f1_score
    out = {}
    for ds in R.DATASETS:
        D = R.load(ds); T = R.build_tasks(D, R.task_order(D, 0), 0); S = R.preprocess(D, T, 0)
        seen = sorted(set().union(*[t['classes'] for t in T]))
        res = {}
        for mode, name in (('memory', 'rf_er2000'), ('joint', 'rf_joint')):
            rf = final_forest(D, S, T, 0, mode)
            stored = json.loads((R.OUT_FINAL / ds / 'rfmem' / f'{name}_o0_s0.json').read_text())['stages'][-1]['macro_f1_seen']
            rf.set_params(n_jobs=1)
            m_ = np.isin(D['yte'], seen)
            f1 = float(f1_score(D['yte'][m_], rf.predict(S['te'])[m_], average='macro', labels=seen))
            res[name] = dict(nodes=int(sum(e.tree_.node_count for e in rf.estimators_)),
                             kib=round(len(pickle.dumps(rf, protocol=4)) / 1024, 1),
                             pred_s_per_10k=round(time_predict(rf.predict, S['te']), 4),
                             f1_check=round(f1, 4), f1_stored=round(stored, 4), matches=abs(f1 - stored) < 1e-6)
        m = R.build_model(S['tr'].shape[1], len(D['classes']))
        res['cnn'] = dict(params=int(m.count_params()), kib=round(m.count_params() * 4 / 1024, 1),
                          pred_s_per_10k=round(time_predict(lambda X: m.predict(X, batch_size=8192, verbose=0), S['te']), 4))
        out[ds] = res
        print(ds, json.dumps(res), flush=True)
    (R.OUT_FINAL / 'size_bench.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
