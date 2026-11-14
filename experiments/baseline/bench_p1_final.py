"""Paper 1 final round: CPU inference latency on ONE thread for every model family (refs/code review M1, referee M2).

Differences from bench_p1_latency.py (which is unchanged; its latency.json stays in results_v2):
  * really single-threaded: OMP/MKL/OpenBLAS/TF thread variables are set to 1 BEFORE numpy, scikit-learn and
    TensorFlow are imported, and every timed call runs inside threadpoolctl.threadpool_limits(1). (The v2 benchmark
    set n_jobs=1 only on models with an n_jobs attribute, so HistGradientBoosting predicted with all OpenMP threads.)
    The settings and threadpoolctl's view of the libraries are recorded in the output.
  * GBDTs: the rolled-back models of results_p1final (best internal-validation iteration), all five seeds
    (seed 0 with the full protocol, seeds 1-4 with 3 repeats of 100 calls), and the capped GBDTs; small trees too.
  * networks: float32 Keras (tf.function) as before, plus the float32 TensorFlow Lite file (size and latency); the v2
    int8 TFLite file timed with the default delegates (XNNPACK) and without them, its op list and tensor types, and
    batch-1000 times; and an extra FULL-INTEGER int8 model (TFLITE_BUILTINS_INT8 only, int8 input and output,
    same 500 calibration records as run_p1_v2) with its seed-0 test macro-F1 and latency (input quantisation
    included in the timed call).
  * optional --lgbm-proxy: a LightGBM model with the same leaves, learning rate and number of iterations as the
    seed-0 best GBDT per dataset, trained on the same data, timed single-threaded: an estimate of compiled
    (C++) tree evaluation cost per tree. It is a different model (not the scikit-learn trees), so only its cost
    per record and per tree is reported.
Timing protocol unchanged: per record = median over 5 repeats of 200 single-record calls after 20 warm-up calls;
per 1,000 records = median of 10 calls after one warm-up (the v2 script took the mean of 10).
Output: results_p1final/latency.json.   Run:  python -m experiments.baseline.bench_p1_final [--lgbm-proxy]
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS',
           'TF_NUM_INTEROP_THREADS'):
    os.environ[_v] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import argparse, json, pickle, platform, statistics as st, sys, time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from threadpoolctl import threadpool_info, threadpool_limits  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.baseline import run_p1_v2 as R  # noqa: E402
from experiments.baseline import run_p1_final as F  # noqa: E402
from experiments.baseline.bench_p1_latency import cpu_name  # noqa: E402
from experiments.paper6_multimodal import run_p6_v2 as P6  # noqa: E402

OUT = F.OUT


def timed(f, items, repeats=5, warm=20):
    with threadpool_limits(1):
        for x in items[:warm]:
            f(x)
        meds = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for x in items:
                f(x)
            meds.append(1000 * (time.perf_counter() - t0) / len(items))
    return st.median(meds)


def batch_ms(f, X):
    """median of 10 calls on 1,000 records (after one warm-up)"""
    with threadpool_limits(1):
        f(X)
        ts = []
        for _ in range(10):
            t0 = time.perf_counter(); f(X); ts.append(1000 * (time.perf_counter() - t0))
    return st.median(ts)


def tree_entry(t, X, n_classes, full=True):
    if hasattr(t, 'n_jobs'):
        t.set_params(n_jobs=1)
    e = dict(single_ms=timed(lambda x: t.predict(x), [X[j:j + 1] for j in range(200 if full else 100)],
                             repeats=5 if full else 3, warm=20 if full else 10))
    if full:
        e['batch1000_ms'] = batch_ms(lambda x: t.predict(x), X)
    e['compact_bytes'] = F.compact_bytes(t, n_classes, X.shape[1])
    if hasattr(t, '_predictors'):
        e['n_trees'] = int(t.n_iter_ * t.n_trees_per_iteration_)
        e['us_per_tree'] = 1000 * e['single_ms'] / e['n_trees']
    return e


def tflite_interp(blob, nodelegate=False, batch=None):
    import tensorflow as tf
    kw = {}
    if nodelegate:
        kw['experimental_op_resolver_type'] = tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    it = tf.lite.Interpreter(model_content=blob, num_threads=1, **kw)
    inp = it.get_input_details()[0]
    if batch:
        it.resize_tensor_input(inp['index'], [batch, *inp['shape'][1:]])
    it.allocate_tensors()
    return it, it.get_input_details()[0], it.get_output_details()[0]


def tflite_info(blob):
    import tensorflow as tf
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    info = {}
    try:
        info['ops'] = dict(Counter(o['op_name'] for o in it._get_ops_details()))
    except Exception as ex:          # private API; absent in some versions
        info['ops'] = f'unavailable: {ex}'
    info['tensor_dtypes'] = dict(Counter(np.dtype(t['dtype']).name for t in it.get_tensor_details()))
    info['input_dtype'] = np.dtype(it.get_input_details()[0]['dtype']).name
    info['output_dtype'] = np.dtype(it.get_output_details()[0]['dtype']).name
    return info


def tflite_timings(blob, X, quantise_input=False):
    """per-record ms with and without default delegates, batch-1000 ms; X float32 [n, f, 1]"""
    out = {}
    for tag, nd in (('', False), ('_nodelegate', True)):
        try:
            it, inp, o = tflite_interp(blob, nodelegate=nd)
        except Exception as ex:
            out[f'single_ms{tag}'] = None; out[f'error{tag}'] = str(ex); continue
        if quantise_input:
            sc, zp = inp['quantization']
            prep = lambda x: np.clip(np.round(x / sc + zp), -128, 127).astype(np.int8)
        else:
            prep = lambda x: x

        def one(x):
            it.set_tensor(inp['index'], prep(x)); it.invoke(); return it.get_tensor(o['index'])
        out[f'single_ms{tag}'] = timed(one, [X[j:j + 1] for j in range(200)])
    itb, inpb, ob = tflite_interp(blob, batch=len(X))
    if quantise_input:
        sc, zp = inpb['quantization']
        prepb = lambda x: np.clip(np.round(x / sc + zp), -128, 127).astype(np.int8)
    else:
        prepb = lambda x: x

    def many(x):
        itb.set_tensor(inpb['index'], prepb(x)); itb.invoke(); return itb.get_tensor(ob['index'])
    out['batch1000_ms'] = batch_ms(many, X)
    return out


def full_int8(m, Xcal):
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_keras_model(m)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([Xcal[i:i + 1]] for i in range(len(Xcal)))
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def tflite_predict_q(blob, X):
    it, inp, o = tflite_interp(blob, batch=len(X))
    sc, zp = inp['quantization']
    it.set_tensor(inp['index'], np.clip(np.round(X / sc + zp), -128, 127).astype(np.int8)); it.invoke()
    return it.get_tensor(o['index']).astype(np.float32)


def net_entry(D, cfg, idx):
    import tensorflow as tf
    d = R.OUT / D['ds'] / cfg                              # results_v2 seed-0 models (read only)
    ph, pq = d / 'model_seed0.h5', d / 'model_seed0.tflite'
    if not (ph.exists() and pq.exists()):
        return None
    m = tf.keras.models.load_model(ph, compile=False)
    f = tf.function(m)
    X = D['Qte'][idx][..., None].astype(np.float32)
    e = dict(single_ms=timed(lambda x: f(x), [X[j:j + 1] for j in range(200)]), batch1000_ms=batch_ms(lambda x: f(x), X),
             size_bytes=4 * m.count_params())
    # float32 TFLite (no optimisation): deployable float file size and latency
    conv = tf.lite.TFLiteConverter.from_keras_model(m)
    f32 = conv.convert()
    e['tflite_f32_size_bytes'] = len(f32)
    e.update({f'tflite_f32_{k}': v for k, v in tflite_timings(f32, X).items()})
    # the int8 model of the paper (Optimize.DEFAULT + representative data: float fallback allowed, float I/O)
    q = pq.read_bytes()
    e['int8_size_bytes'] = len(q)
    e['int8_info'] = tflite_info(q)
    tq = tflite_timings(q, X)
    e['int8_single_ms'] = tq.pop('single_ms')
    e.update({f'int8_{k}': v for k, v in tq.items()})
    # extra row: full-integer quantisation, int8 input/output, same calibration records as run_p1_v2 (seed 0)
    Xf = D['Qfit'][..., None]
    Xcal = Xf[np.random.default_rng(0).choice(len(Xf), 500, replace=False)]
    try:
        fq = full_int8(m, Xcal)
        e['fullint8_size_bytes'] = len(fq)
        e['fullint8_info'] = tflite_info(fq)
        tf_ = tflite_timings(fq, X, quantise_input=True)
        e.update({f'fullint8_{k}': v for k, v in tf_.items()})
        Pq = tflite_predict_q(fq, D['Qte'][..., None].astype(np.float32))
        e['fullint8_macro_f1_seed0'] = P6.metrics(D, Pq.argmax(1))['macro_f1']
        e['float32_macro_f1_seed0'] = json.loads((d / 'seed0.json').read_text())['macro_f1']
        e['int8_macro_f1_seed0'] = json.loads((d / 'seed0.json').read_text())['int8']['macro_f1']
    except Exception as ex:
        e['fullint8_error'] = repr(ex)[:500]
    return e


def lgbm_proxy(D, cfg, gb_model, X):
    import lightgbm as lgb
    kind, p = R.TREE_CFGS[cfg]
    n_it = int(gb_model.n_iter_)
    clf = lgb.LGBMClassifier(n_estimators=n_it, num_leaves=p['max_leaf_nodes'], learning_rate=p['learning_rate'],
                             min_child_samples=20, max_bin=255, num_threads=os.cpu_count(), verbose=-1, random_state=0)
    clf.fit(D['Xfit'], D['yfit'])
    bst = clf.booster_
    n_trees = bst.num_trees()
    f = lambda x: bst.predict(x, num_threads=1)
    e = dict(config=cfg, n_iter=n_it, n_trees=int(n_trees), single_ms=timed(f, [X[j:j + 1] for j in range(200)]),
             batch1000_ms=batch_ms(f, X))
    e['us_per_tree'] = 1000 * e['single_ms'] / n_trees
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lgbm-proxy', action='store_true')
    ap.add_argument('--dataset', default='all', choices=('all',) + R.DATASETS)
    a = ap.parse_args()
    import sklearn
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    p = OUT / 'latency.json'
    out = json.loads(p.read_text()) if p.exists() else {}
    out.update(cpu=cpu_name(), platform=platform.platform(), tensorflow=tf.__version__, sklearn=sklearn.__version__,
               threads=1, env={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                                              'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')},
               tf_threads=[tf.config.threading.get_intra_op_parallelism_threads(),
                           tf.config.threading.get_inter_op_parallelism_threads()],
               protocol='per record: median of 5 x 200 calls after 20 warm-up (GBDT seeds 1-4: 3 x 100 after 10); '
                        'per 1,000: median of 10 calls after 1 warm-up; all calls inside threadpool_limits(1)')
    with threadpool_limits(1):
        out['threadpool_info'] = [{k: v for k, v in d.items() if k in ('internal_api', 'num_threads', 'filepath')}
                                  for d in threadpool_info()]
    for ds in R.DATASETS if a.dataset == 'all' else (a.dataset,):
        D = P6.load(ds)
        nc = len(D['classes'])
        idx = np.random.default_rng(0).choice(len(D['yte']), 1000, replace=False)
        res = out.get(ds, {})
        # forests and LR of results_v2 (unchanged models), seed 0
        for cfg in F.RFLR_CFGS:
            pth = R.OUT / ds / cfg / 'model_seed0.pkl'
            if cfg in res or not pth.exists():
                continue
            t = pickle.loads(pth.read_bytes())
            X = (D['Qte'] if cfg == 'lr' else D['Xte'])[idx]
            res[cfg] = tree_entry(t, X, nc)
            print(ds, cfg, round(res[cfg]['single_ms'], 4), flush=True)
            del t
        # rolled-back GBDTs (all seeds), capped GBDTs and small trees of results_p1final
        X = D['Xte'][idx]
        for cfg in list(F.GB_CFGS) + list(F.CAPPED) + list(F.SMALL):
            if cfg in res:
                continue
            seeds = range(5) if cfg in F.GB_CFGS or cfg in F.CAPPED else (0,)
            paths = [OUT / ds / cfg / f'model_seed{s}.pkl' for s in seeds]
            if not paths[0].exists():
                continue
            e = tree_entry(pickle.loads(paths[0].read_bytes()), X, nc)
            if len(paths) > 1:
                per = [tree_entry(pickle.loads(q.read_bytes()), X, nc, full=False) for q in paths[1:] if q.exists()]
                e['single_ms_seeds'] = [e['single_ms']] + [x['single_ms'] for x in per]
                e['n_trees_seeds'] = [e.get('n_trees')] + [x.get('n_trees') for x in per]
            res[cfg] = e
            print(ds, cfg, round(e['single_ms'], 4), e.get('n_trees'), flush=True)
        # networks (results_v2 seed-0 models)
        for cfg in R.NET_CFGS:
            if cfg in res:
                continue
            e = net_entry(D, cfg, idx)
            if e is not None:
                res[cfg] = e
                print(ds, cfg, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items() if 'info' not in k}, flush=True)
        if a.lgbm_proxy and 'lgbm_proxy' not in res:
            gb = [c for c in F.GB_CFGS if (OUT / ds / c / 'seed0.json').exists()]
            if gb:
                best = max(gb, key=lambda c: np.mean([json.loads((OUT / ds / c / f'seed{s}.json').read_text())['macro_f1']
                                                      for s in range(5) if (OUT / ds / c / f'seed{s}.json').exists()]))
                res['lgbm_proxy'] = lgbm_proxy(D, best, pickle.loads((OUT / ds / best / 'model_seed0.pkl').read_bytes()), X)
                print(ds, 'lgbm_proxy', res['lgbm_proxy'], flush=True)
        out[ds] = res
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1))          # checkpoint per dataset (resumable)
        del D
    print('saved', p)


if __name__ == '__main__':
    main()
