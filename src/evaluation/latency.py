"""
src/evaluation/latency.py
Inference latency benchmarking for all model types (§4.3).
"""
import time
import numpy as np


def benchmark_latency(predict_fn, X: np.ndarray,
                      n_samples: int = 1000,
                      n_runs: int = 5,
                      seed: int = 42) -> dict:
    """
    Measure per-sample inference latency (ms) by timing n_samples predictions
    repeated n_runs times.

    Parameters
    ----------
    predict_fn : callable that accepts X and returns predictions (model.predict)
    X          : feature array to sample from
    n_samples  : number of samples per timing run
    n_runs     : number of timing repetitions

    Returns
    -------
    dict with: mean_ms, median_ms, std_ms, p95_ms, total_ms
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    X_bench = X[idx]

    timings = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        predict_fn(X_bench)
        elapsed = (time.perf_counter() - t0) * 1000  # ms total
        timings.append(elapsed / len(X_bench))         # ms per sample

    timings = np.array(timings)
    return dict(
        mean_ms   = float(timings.mean()),
        median_ms = float(np.median(timings)),
        std_ms    = float(timings.std()),
        p95_ms    = float(np.percentile(timings, 95)),
        total_ms  = float(timings.sum()),
    )


def benchmark_all(models: dict, X: np.ndarray,
                  n_samples: int = 1000, n_runs: int = 5, seed: int = 42) -> dict:
    """
    Benchmark a dictionary of {name: predict_fn} models.

    Returns
    -------
    dict of {model_name: latency_dict}
    """
    results = {}
    for name, fn in models.items():
        print(f"  Benchmarking latency: {name}…")
        results[name] = benchmark_latency(fn, X, n_samples=n_samples,
                                          n_runs=n_runs, seed=seed)
        print(f"    mean={results[name]['mean_ms']:.4f} ms/sample  "
              f"p95={results[name]['p95_ms']:.4f} ms/sample")
    return results
