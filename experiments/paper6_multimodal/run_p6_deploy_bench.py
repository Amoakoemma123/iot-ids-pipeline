"""Paper 6: deployment cost of the tree ensembles in a compiled/serialised runtime format (ONNX + ONNX Runtime),
and sustainable flows per second per CPU core for trees and networks.

Why: the size/latency table (Revision_v2/Paper6 compute_values.py, from results_v4/latency.json and
results_p6gb/latency_gb.json) times the trees as pickled scikit-learn objects called through Python, which the
manuscript itself calls unfair to the trees. No compiler (MSVC/gcc) is installed on the benchmark laptop, so the
treelite + tl2cgen C route is not available; ONNX (skl2onnx 1.12 -> onnx 1.12, ONNX Runtime 1.23 CPU provider)
installs from PyPI without touching TensorFlow 2.10's protobuf 3.19 pin.

Models (no trained models were saved, so they are refitted exactly as the benchmarks that feed the table):
  rf : run_p6_v2.make_tree('rf', results_v4/<ds>/tune_trees.json best, seed 0), all features (bench_latency.py);
       checked against results_v4/<ds>/rf/seed0_pred.npy
  gb : run_p6_gb_rollback.fit_gb(results_p6gb/<ds>/tune_trees.json best, seed 0) + rollback_best_iter, all features
       (bench_p6_gb_rollback.py); checked against results_p6gb/<ds>/gb/seed0_pred.npy
Protocol (bench_latency.time_calls, the same 1,000 test records drawn with default_rng(0)):
  single_ms    : one record per call, median over 5 repeats of the mean time of 200 calls
  batch1000_ms : 1,000 records in one call, mean of 20 calls
  ONNX Runtime sessions use intra_op = inter_op = 1 thread (sequential execution); scikit-learn originals are
  re-timed in the same session with n_jobs=1 and threadpool_limits(1). Networks (architectures of run_p6_v2.build,
  untrained as in bench_latency.py) are re-timed with TensorFlow intra/inter-op threads = 1.
  flows_per_s_core_single = 1000 / single_ms        (one record at a time, one core)
  flows_per_s_core_batch  = 1e6 / batch1000_ms      (1,000-record batches, one core)
  For the existing results (latency.json / latency_gb.json) the same conversion is reported, but those timings
  were taken with default threads (TF: all 8 logical CPUs; RF batch n_jobs=-1; GBDT batch all cores), so their
  "per core" figure is a per-process figure.
Agreement: % identical predicted labels between the ONNX model and the scikit-learn model on the full test set.
Output: results_p6deploy/deploy_bench.json; ONNX artefacts in results_p6deploy/artefacts/ (git-ignored).
Run (idle machine):  python -m experiments.paper6_multimodal.run_p6_deploy_bench [--part trees|nets|all] [--ds ...]
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.paper6_multimodal import run_p6_v2 as R  # noqa: E402
from experiments.paper6_multimodal.bench_latency import cpu_name, time_calls  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_p6deploy'
ART = OUT / 'artefacts'
GB_RES = HERE / 'results_p6gb'
NET_ARCHS = ('s1', 's2', 'early', 'late', 'attn', 'm_early', 'm_late')


def fps(single_ms=None, batch_ms=None):
    d = {}
    if single_ms is not None:
        d['flows_per_s_core_single'] = 1000.0 / single_ms
    if batch_ms is not None:
        d['flows_per_s_core_batch'] = 1e6 / batch_ms
    return d


def bench_idx(D):
    return np.random.default_rng(0).choice(len(D['yte']), 1000, replace=False)


def fit_tree(kind, D, cols):
    if kind == 'rf':
        t = R.make_tree('rf', R.tuned(D['ds'])['rf']['best'], 0)
        t0 = time.time(); t.fit(D['Xfit'][:, cols], D['yfit'])
        ref = R.OUT / D['ds'] / 'rf' / 'seed0_pred.npy'
        info = dict(params=R.tuned(D['ds'])['rf']['best'], n_trees=len(t.estimators_),
                    nodes=int(sum(e.tree_.node_count for e in t.estimators_)), fit_s=time.time() - t0)
    else:
        from experiments.paper6_multimodal.run_p6_gb_rollback import fit_gb
        from experiments.baseline.gbdt_rollback import rollback_best_iter
        params = json.loads((GB_RES / D['ds'] / 'tune_trees.json').read_text())['gb']['best']
        t, fit_s = fit_gb(D, params, 0, cols)
        rb = rollback_best_iter(t)
        ref = GB_RES / D['ds'] / 'gb' / 'seed0_pred.npy'
        info = dict(params=params, best_iter=rb['best_iter'], n_trees=int(sum(len(p) for p in t._predictors)),
                    fit_s=fit_s)
    Pc = t.predict_proba(D['Xte'][:, cols])
    pred = t.classes_[Pc.argmax(1)]
    info['reproduces_seed0'] = bool(np.array_equal(pred.astype(np.int8), np.load(ref)))
    return t, pred, info


def ort_session(path, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = threads
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), so, providers=['CPUExecutionProvider'])


def trees(ds, res, save=lambda: None):
    import pickle
    import onnx
    from skl2onnx import to_onnx
    from threadpoolctl import threadpool_limits
    D = R.load(ds)
    s1, s2 = R.semantic_split(D)
    cols = np.r_[s1, s2]
    Xte = np.ascontiguousarray(D['Xte'][:, cols], dtype=np.float32)
    X = Xte[bench_idx(D)]
    (ART / ds).mkdir(parents=True, exist_ok=True)
    for kind in ('rf', 'gb'):
        t, pred, info = fit_tree(kind, D, cols)
        print(ds, kind, 'fitted', info, flush=True)
        r = dict(info, sklearn_pickle_mb=len(pickle.dumps(t)) / 1e6)
        # scikit-learn original, same session, one thread
        if kind == 'rf':
            t.set_params(n_jobs=1)
        with threadpool_limits(1):
            r['sklearn_1thread'] = dict(single_ms=time_calls(lambda x: t.predict(x), [X[j:j + 1] for j in range(200)]),
                                        batch1000_ms=time_calls(lambda x: t.predict(x), [X] * 20, repeats=1))
        r['sklearn_1thread'].update(fps(r['sklearn_1thread']['single_ms'], r['sklearn_1thread']['batch1000_ms']))
        # ONNX export
        t0 = time.time()
        m = to_onnx(t, Xte[:1], options={type(t): {'zipmap': False}}, target_opset={'': 15, 'ai.onnx.ml': 3})
        p = ART / ds / f'{kind}.onnx'
        onnx.save(m, str(p))
        del m; gc.collect()
        r['onnx_convert_s'] = time.time() - t0
        r['onnx_file_mb'] = p.stat().st_size / 1e6
        del t; gc.collect()
        sess = ort_session(p, 1)
        name = sess.get_inputs()[0].name
        run = lambda x: sess.run(None, {name: x})[0]
        lab = np.concatenate([run(Xte[i:i + 20000]) for i in range(0, len(Xte), 20000)])
        r['onnx_agreement_pct'] = float(100 * (lab.astype(np.int64) == pred.astype(np.int64)).mean())
        r['onnx_n_disagree'] = int((lab.astype(np.int64) != pred.astype(np.int64)).sum())
        r['n_test'] = int(len(Xte))
        r['onnx_1thread'] = dict(single_ms=time_calls(run, [X[j:j + 1] for j in range(200)]),
                                 batch1000_ms=time_calls(run, [X] * 20, repeats=1))
        r['onnx_1thread'].update(fps(r['onnx_1thread']['single_ms'], r['onnx_1thread']['batch1000_ms']))
        del sess; gc.collect()
        sa = ort_session(p, 0)          # 0 = ORT default (all physical cores), for reference only
        na = sa.get_inputs()[0].name
        r['onnx_allcores_batch1000_ms'] = time_calls(lambda x: sa.run(None, {na: x})[0], [X] * 20, repeats=1)
        del sa; gc.collect()
        res[kind] = r
        save()                          # persist after every model (partial runs are kept)
        print(ds, kind, {k: v for k, v in r.items() if k != 'params'}, flush=True)
    del D; gc.collect()


def nets(ds, res):
    import tensorflow as tf
    D = R.load(ds)
    s1, s2 = R.semantic_split(D)
    nc = len(D['classes'])
    idx = bench_idx(D)
    for arch in NET_ARCHS:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(0)
        m = R.build(arch, len(s1), len(s2), nc)
        Xin = R.inputs(arch, D['Qte'][idx], s1, s2)
        f = tf.function(m)
        one = [[x[j:j + 1] for x in Xin] if isinstance(Xin, list) else Xin[j:j + 1] for j in range(200)]
        r = dict(params=int(m.count_params()), float32_kb=m.count_params() * 4 / 1000,
                 single_ms=time_calls(lambda x: f(x), one), batch1000_ms=time_calls(lambda x: f(x), [Xin] * 20, repeats=1))
        r.update(fps(r['single_ms'], r['batch1000_ms']))
        res[arch] = r
        print(ds, arch, r, flush=True)
    del D; gc.collect()


def existing():
    """flows/s conversion of the timings already in the manuscript's table (default threads, see docstring)"""
    L = json.loads((R.OUT / 'latency.json').read_text())
    G = json.loads((GB_RES / 'latency_gb.json').read_text())
    out = dict(source='results_v4/latency.json (networks, RF) + results_p6gb/latency_gb.json (GBDT, as used by compute_values.py)')
    for ds in R.DATASETS:
        e = {}
        for a, v in L[ds].items():
            if a == 'gb':
                continue
            e[a] = dict(single_ms=v['single_ms'], batch1000_ms=v['batch1000_ms'], **fps(v['single_ms'], v['batch1000_ms']),
                        threads='TF default (8 logical CPUs)' if a != 'rf' else 'single n_jobs=1, batch n_jobs=-1')
        g = G[ds]
        e['gb'] = dict(single_ms=g['single_ms_1thread'], batch1000_ms=g['batch1000_ms'],
                       **fps(g['single_ms_1thread'], g['batch1000_ms']), threads='single 1 thread, batch all cores (as in the table)',
                       batch1000_ms_1thread=g['batch1000_ms_1thread'],
                       flows_per_s_core_batch_1thread=1e6 / g['batch1000_ms_1thread'])
        out[ds] = e
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=('trees', 'nets', 'all'), default='all')
    ap.add_argument('--ds', nargs='*', default=list(R.DATASETS))
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True); ART.mkdir(exist_ok=True)
    (ART / '.gitignore').write_text('*\n')
    p = OUT / 'deploy_bench.json'
    J = json.loads(p.read_text()) if p.exists() else {}
    import onnxruntime, skl2onnx, sklearn
    J.update(cpu=cpu_name(), logical_cpus=os.cpu_count(), onnxruntime=onnxruntime.__version__,
             skl2onnx=skl2onnx.__version__, sklearn=sklearn.__version__)
    if a.part in ('nets', 'all'):
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        J['tensorflow'] = tf.__version__
    J['existing_table_flows'] = existing()
    for ds in a.ds:
        J.setdefault(ds, {})
        if a.part in ('trees', 'all'):
            J[ds].setdefault('trees', {})
            trees(ds, J[ds]['trees'], lambda: p.write_text(json.dumps(J, indent=1)))
            J[ds]['trees_measured'] = time.strftime('%Y-%m-%d %H:%M')
            p.write_text(json.dumps(J, indent=1))
        if a.part in ('nets', 'all'):
            J[ds].setdefault('nets_1thread', {})
            nets(ds, J[ds]['nets_1thread'])
            J[ds]['nets_measured'] = time.strftime('%Y-%m-%d %H:%M')
            p.write_text(json.dumps(J, indent=1))
    print('saved', p)


if __name__ == '__main__':
    main()
