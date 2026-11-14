"""Paper 1 v2: CPU inference latency of every seed-0 model saved by run_p1_v2.py, measured in isolation.

All measurements on ONE CPU thread (TF intra/inter-op threads = 1, sklearn n_jobs = 1, TFLite num_threads = 1),
to approximate a single-core gateway; GPU hidden.
  single_ms     : one record per call, median over 5 repeats of 200 calls (after 20 warm-up calls)
  batch1000_ms  : 1,000 records in one call, median of 10 calls
  size_bytes    : serialised model (pickle for scikit-learn, TFLite flatbuffer for int8, 4 bytes/parameter for float32)
Output: results_v2/latency.json.   Run:  python -m experiments.baseline.bench_p1_latency
"""
import json, os, pickle, platform, statistics as st, subprocess, sys, time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.baseline import run_p1_v2 as R  # noqa: E402
from experiments.paper6_multimodal import run_p6_v2 as P6  # noqa: E402


def cpu_name():
    try:
        out = subprocess.run(['powershell', '-c', '(Get-CimInstance Win32_Processor).Name'], capture_output=True, text=True).stdout
        return out.strip() or platform.processor()
    except Exception:
        return platform.processor()


def timed(f, items, repeats=5, warm=20):
    for x in items[:warm]:
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
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    out = {'cpu': cpu_name(), 'tensorflow': tf.__version__, 'threads': 1}
    for ds in R.DATASETS:
        D = P6.load(ds)
        idx = np.random.default_rng(0).choice(len(D['yte']), 1000, replace=False)
        res = {}
        for cfg in list(R.TREE_CFGS) + list(R.NET_CFGS):
            d = R.OUT / ds / cfg
            if cfg in R.TREE_CFGS:
                p = d / 'model_seed0.pkl'
                if not p.exists():
                    continue
                t = pickle.loads(p.read_bytes())
                if hasattr(t, 'n_jobs'):
                    t.set_params(n_jobs=1)
                X = (D['Qte'] if cfg == 'lr' else D['Xte'])[idx]
                single = timed(lambda x: t.predict(x), [X[j:j + 1] for j in range(200)])
                batch = timed(lambda x: t.predict(x), [X] * 10, repeats=1, warm=1)
                res[cfg] = dict(single_ms=single, batch1000_ms=batch, size_bytes=p.stat().st_size,
                                compact_bytes=R.compact_bytes(t, len(D['classes']), X.shape[1]))
            else:
                ph, pq = d / 'model_seed0.h5', d / 'model_seed0.tflite'
                if not (ph.exists() and pq.exists()):
                    continue
                m = tf.keras.models.load_model(ph, compile=False)
                f = tf.function(m)
                X = D['Qte'][idx][..., None].astype(np.float32)
                single = timed(lambda x: f(x), [X[j:j + 1] for j in range(200)])
                batch = timed(lambda x: f(x), [X] * 10, repeats=1, warm=1)
                it = tf.lite.Interpreter(model_path=str(pq), num_threads=1)
                inp, o = it.get_input_details()[0], it.get_output_details()[0]
                it.allocate_tensors()

                def one(x):
                    it.set_tensor(inp['index'], x); it.invoke(); return it.get_tensor(o['index'])
                q_single = timed(one, [X[j:j + 1] for j in range(200)])
                res[cfg] = dict(single_ms=single, batch1000_ms=batch, size_bytes=4 * m.count_params(),
                                int8_single_ms=q_single, int8_size_bytes=pq.stat().st_size)
            print(ds, cfg, {k: round(v, 4) if isinstance(v, float) else v for k, v in res[cfg].items()}, flush=True)
        out[ds] = res
        del D
    p = R.OUT / 'latency.json'
    p.write_text(json.dumps(out, indent=1))
    print('saved', p)


if __name__ == '__main__':
    main()
