"""
experiments/continual_learning/evaluation/memory_usage.py
Computational cost and memory overhead tracking for CL strategies.

Tracks:
  - Wall-clock training time per task (seconds)
  - Peak process memory during training (MB)
  - Replay buffer memory footprint (MB)
  - Fisher Information Matrix memory footprint for EWC (MB)
  - Total parameter count (unchanged — same CNN for all strategies)
  - Overhead ratio vs naive fine-tuning (time and memory)
"""
import time
import os
import numpy as np

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


def get_process_memory_mb() -> float:
    """Return current process RSS memory in MB."""
    if not _PSUTIL_AVAILABLE:
        return float('nan')
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


class CostTracker:
    """
    Context manager + accumulator for tracking per-task training cost.

    Usage
    -----
    tracker = CostTracker()
    with tracker.track_task(task_id=0):
        train_one_task(...)
    print(tracker.summary())
    """

    def __init__(self):
        self._records = []   # list of dicts, one per task
        self._start_time = None
        self._start_mem  = None
        self._current_task_id = None

    def track_task(self, task_id: int):
        """Return a context manager that times and memory-profiles one task."""
        self._current_task_id = task_id
        return self

    def __enter__(self):
        self._start_time = time.perf_counter()
        self._start_mem  = get_process_memory_mb()
        return self

    def __exit__(self, *_):
        elapsed = time.perf_counter() - self._start_time
        end_mem = get_process_memory_mb()
        peak_delta = end_mem - self._start_mem if not (
            np.isnan(self._start_mem) or np.isnan(end_mem)) else float('nan')
        self._records.append({
            'task_id':      self._current_task_id,
            'time_sec':     round(elapsed, 2),
            'mem_start_mb': round(self._start_mem, 1),
            'mem_end_mb':   round(end_mem, 1),
            'mem_delta_mb': round(peak_delta, 1),
        })

    def replay_buffer_mb(self, mem_X: np.ndarray | None,
                          mem_y: np.ndarray | None) -> float:
        """Compute memory footprint of Experience Replay buffer in MB."""
        if mem_X is None:
            return 0.0
        return (mem_X.nbytes + (mem_y.nbytes if mem_y is not None else 0)) / (1024 ** 2)

    def fisher_matrix_mb(self, fisher_diag: list | None) -> float:
        """Compute memory footprint of EWC Fisher diagonal in MB."""
        if fisher_diag is None:
            return 0.0
        total_bytes = sum(f.nbytes for f in fisher_diag)
        return total_bytes / (1024 ** 2)

    def total_time(self) -> float:
        """Total training time across all tasks (seconds)."""
        return sum(r['time_sec'] for r in self._records)

    def summary(self) -> dict:
        """Return full cost summary as a dict."""
        return {
            'per_task': self._records,
            'total_time_sec': round(self.total_time(), 2),
            'mean_time_per_task_sec': round(
                self.total_time() / max(len(self._records), 1), 2),
        }

    def print_summary(self, strategy_name: str = '') -> None:
        print(f"\n  [Cost] Strategy: {strategy_name}")
        for r in self._records:
            print(f"    Task {r['task_id']}: {r['time_sec']:.1f}s  "
                  f"mem_delta={r['mem_delta_mb']:.1f} MB")
        print(f"    Total: {self.total_time():.1f}s")
