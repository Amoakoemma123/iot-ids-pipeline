"""
run_multiseed.py

Runs C2 (CCAT) and C3 (FedAvg) conditions at seeds [42, 123, 456]
for TON_IoT and UNSW_NB15.

CICIDS-2017 excluded: 69 min per full run makes 3-seed × 2-condition
infeasible without dedicated compute; acknowledged in paper §8.1.

Output: experiments/paper7_integrated/results/{dataset}/multiseed_c2c3.json
Schema: { "<seed>": { "C2": <metrics>, "C3": <metrics> }, ... }
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
    load_data, run_c2_adversarial, run_c3_federated,
)

SEEDS       = [42, 123, 456]
DATASETS    = ['TON_IoT', 'UNSW_NB15']
RESULTS_DIR = os.path.join(ROOT, 'experiments', 'paper7_integrated', 'results')


for dataset_name in DATASETS:
    out_file = os.path.join(RESULTS_DIR, dataset_name, 'multiseed_c2c3.json')

    # Resume from partial file if it exists
    if os.path.exists(out_file):
        with open(out_file) as f:
            ms = json.load(f)
    else:
        ms = {}

    print(f'\n{"="*60}')
    print(f'  Multi-seed C2/C3 — {dataset_name}')
    print(f'{"="*60}')

    data = load_data(dataset_name)

    for seed in SEEDS:
        sk = str(seed)
        if sk not in ms:
            ms[sk] = {}

        # ── C2 ────────────────────────────────────────────────────────────
        if 'C2' not in ms[sk]:
            print(f'\n  [seed={seed}] C2 (CCAT)...')
            t0 = time.time()
            ms[sk]['C2'] = run_c2_adversarial(data, dataset_name, seed=seed)
            elapsed = (time.time() - t0) / 60
            print(f'    → F1={ms[sk]["C2"]["f1"]:.4f}  ({elapsed:.1f} min)')
            with open(out_file, 'w') as f:
                json.dump(ms, f, indent=2)
        else:
            print(f'  [seed={seed}] C2 done  (F1={ms[sk]["C2"]["f1"]:.4f})')

        # ── C3 ────────────────────────────────────────────────────────────
        if 'C3' not in ms[sk]:
            print(f'\n  [seed={seed}] C3 (FedAvg)...')
            t0 = time.time()
            ms[sk]['C3'] = run_c3_federated(data, seed=seed)
            elapsed = (time.time() - t0) / 60
            print(f'    → F1={ms[sk]["C3"]["f1"]:.4f}  ({elapsed:.1f} min)')
            with open(out_file, 'w') as f:
                json.dump(ms, f, indent=2)
        else:
            print(f'  [seed={seed}] C3 done  (F1={ms[sk]["C3"]["f1"]:.4f})')

    # ── Summary ───────────────────────────────────────────────────────────
    print(f'\n  Summary — {dataset_name}')
    for cond in ['C2', 'C3']:
        vals = [ms[str(s)][cond]['f1']
                for s in SEEDS
                if str(s) in ms and cond in ms[str(s)]]
        if vals:
            arr = np.array(vals)
            print(f'    {cond}: mean={arr.mean():.4f}  std={arr.std():.4f}  '
                  f'({", ".join(f"{v:.4f}" for v in vals)})')

print('\nDone.')
