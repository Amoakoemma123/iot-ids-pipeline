"""Paper 6: CPU inference latency of every architecture, measured in isolation (no other jobs running).

The latencies recorded during training (run_p6_v2.py) were taken on the GPU while other jobs shared the
machine, so they are not reported. Latency does not depend on the trained weights, so the CNNs are built
with the architecture of run_p6_v2.py and timed untrained; the RF is trained (seed 0, all features) because
its prediction time depends on the fitted trees.
  single_ms     : one record per call (tf.function / rf.predict on a 1-row array), median over 5 repeats of the mean time of 200 calls
  batch1000_ms  : 1,000 records in one call, mean of 20 calls (repeats=1)
Output: results_v4/latency.json.   Run:  python -m experiments.paper6_multimodal.bench_latency
"""
import json
import os
import platform
import statistics as st
import sys
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.paper6_multimodal import run_p6_v2 as R  # noqa: E402


def cpu_name():
    try:
        import subprocess
        out = subprocess.run(['powershell', '-c', '(Get-CimInstance Win32_Processor).Name'], capture_output=True, text=True).stdout
        return out.strip() or platform.processor()
    except Exception:
        return platform.processor()


def time_calls(f, items, repeats=5):
    for x in items[:5]:
        f(x)
    meds = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for x in items:
            f(x)
        meds.append(1000 * (time.perf_counter() - t0) / len(items))
    return st.median(meds)


def main():
    import tensorflow as tf
    out = {'cpu': cpu_name(), 'tensorflow': tf.__version__, 'threads': os.cpu_count()}
    for ds in R.DATASETS:
        D = R.load(ds)
        s1, s2 = R.semantic_split(D)
        nc = len(D['classes'])
        rng = np.random.default_rng(0)
        idx = rng.choice(len(D['yte']), 1000, replace=False)
        res = {}
        for arch in ('s1', 's2', 'early', 'late', 'attn', 'early_wide', 'early_head', 'late_all', 'm_early', 'm_late'):
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(0)
            n1, n2 = (len(s1) + len(s2),) * 2 if arch == 'late_all' else (len(s1), len(s2))
            m = R.build(arch, n1, n2, nc)
            Xin = R.inputs(arch, D['Qte'][idx], s1, s2)
            f = tf.function(m)
            one = [[x[j:j + 1] for x in Xin] if isinstance(Xin, list) else Xin[j:j + 1] for j in range(200)]
            single = time_calls(lambda x: f(x), one)
            batch = time_calls(lambda x: f(x), [Xin] * 20, repeats=1)
            res[arch] = dict(params=int(m.count_params()), single_ms=single, batch1000_ms=batch)
            print(ds, arch, res[arch], flush=True)
        import pickle
        cols = np.r_[s1, s2]
        X = D['Xte'][idx][:, cols]
        for kind in ('rf', 'gb'):          # tuned configurations (results_v4/<ds>/tune_trees.json)
            t = R.make_tree(kind, R.tuned(ds)[kind]['best'], 0).fit(D['Xfit'][:, cols], D['yfit'])
            if kind == 'rf':
                t.set_params(n_jobs=1)     # one record at a time: thread start-up would dominate
            single = time_calls(lambda x: t.predict(x), [X[j:j + 1] for j in range(200)])
            if kind == 'rf':
                t.set_params(n_jobs=-1)
            batch = time_calls(lambda x: t.predict(x), [X] * 20, repeats=1)
            res[kind] = dict(single_ms=single, batch1000_ms=batch, size_mb=len(pickle.dumps(t)) / 2 ** 20)
            if kind == 'rf':
                res[kind]['nodes'] = int(sum(e.tree_.node_count for e in t.estimators_))
            print(ds, kind, res[kind], flush=True)
        out[ds] = res
        del D
    p = R.OUT / 'latency.json'
    p.write_text(json.dumps(out, indent=1))
    print('saved', p)


if __name__ == '__main__':
    main()
