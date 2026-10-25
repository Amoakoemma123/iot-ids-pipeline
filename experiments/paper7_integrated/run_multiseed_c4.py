"""
run_multiseed_c4.py

Runs C4 (Full IREAP) at seeds [42, 123, 456] for TON_IoT and UNSW_NB15.

CICIDS-2017 excluded: each C4 run takes ~90 min (CCAT + Replay + SHAP on
2.5M-sample dataset), making 3-seed × 1-condition (~4.5 h) infeasible
without dedicated compute; acknowledged in paper §8.1.

Output: experiments/paper7_integrated/results/{dataset}/multiseed_c4.json
Schema: { "<seed>": { "C4": <metrics> }, ... }
  where <metrics> mirrors the run_c4_full return dict.
"""

import json
import os
import sys
import time

import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

for g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

from experiments.paper7_integrated.run_integrated import (
    load_data, run_c4_full,
)

SEEDS       = [42, 123, 456]
DATASETS    = ['TON_IoT', 'UNSW_NB15']
RESULTS_DIR = os.path.join(ROOT, 'experiments', 'paper7_integrated', 'results')


for dataset_name in DATASETS:
    out_file = os.path.join(RESULTS_DIR, dataset_name, 'multiseed_c4.json')

    # Resume from partial file if it exists
    if os.path.exists(out_file):
        with open(out_file) as f:
            ms = json.load(f)
    else:
        ms = {}

    print(f'\n{"="*60}')
    print(f'  Multi-seed C4 (Full IREAP) — {dataset_name}')
    print(f'{"="*60}')

    data = load_data(dataset_name)

    for seed in SEEDS:
        sk = str(seed)
        if sk not in ms:
            ms[sk] = {}

        if 'C4' not in ms[sk]:
            print(f'\n  [seed={seed}] C4 (Full IREAP)...')
            t0 = time.time()
            ms[sk]['C4'] = run_c4_full(data, dataset_name, seed=seed)
            elapsed = (time.time() - t0) / 60
            full_f1 = ms[sk]['C4']['full_after_adapt']['f1']
            print(f'    → Full-adapt F1={full_f1:.4f}  ({elapsed:.1f} min)')
            with open(out_file, 'w') as f:
                json.dump(ms, f, indent=2)
        else:
            full_f1 = ms[sk]['C4']['full_after_adapt']['f1']
            print(f'  [seed={seed}] C4 done  (Full-adapt F1={full_f1:.4f})')

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n  Summary — {dataset_name}')
    vals = [ms[str(s)]['C4']['full_after_adapt']['f1']
            for s in SEEDS
            if str(s) in ms and 'C4' in ms[str(s)]]
    if vals:
        arr = np.array(vals)
        vals_str = ', '.join(f'{v:.4f}' for v in vals)
        print(f'    C4 full-adapt F1: mean={arr.mean():.4f}  std={arr.std():.4f}  ({vals_str})')

print('\nDone.')
