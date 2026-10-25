"""
run_multi_seed.py
Runs key FL experiments with seeds 43 and 44 (seed 42 = main results already done).
Then aggregates mean ± std across all 3 seeds and saves multi_seed_summary.json.

Key experiments (chosen to cover all main claims):
  TON-IoT:   fedavg_iid, fedavg_alpha0.5, fedavg_alpha0.1, fedprox_alpha0.5,
             scale_N2, scale_N8
  UNSW-NB15: fedavg_iid, fedavg_alpha0.5

Runtime estimate: ~4.5 hours total (run overnight).
"""
import subprocess, json, sys
import numpy as np
from pathlib import Path

ROOT   = Path(__file__).resolve().parent
PYTHON = ROOT / 'ids_env' / 'Scripts' / 'python.exe'
SCRIPT = ROOT / 'experiments' / 'paper4_federated' / 'run_federated.py'

TON_EXPS  = ['fedavg_iid', 'fedavg_alpha0.5', 'fedavg_alpha0.1',
              'fedprox_alpha0.5', 'scale_N2', 'scale_N8']
UNSW_EXPS = ['fedavg_iid', 'fedavg_alpha0.5']
EXTRA_SEEDS = [43, 44]

def run_exp(exp, dataset, seed):
    cmd = [str(PYTHON), str(SCRIPT),
           '--exp', exp, '--dataset', dataset, '--seed', str(seed)]
    print(f'\n>>> {exp}  dataset={dataset}  seed={seed}', flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        print(f'[ERROR] {exp} seed={seed} failed (exit {proc.returncode})', flush=True)
        return False
    return True

def result_path(exp, dataset, seed):
    base = ROOT / 'experiments' / 'paper4_federated'
    if dataset == 'unsw_nb15':
        sub = 'results_unsw'
    else:
        sub = 'results'
    if seed == 42:
        return base / sub / f'{exp}.json'
    else:
        return base / sub / f'seed_{seed}' / f'{exp}.json'

# ── Run additional seeds ───────────────────────────────────────────────────────
print('='*60)
print('Phase 1: Running TON-IoT extra seeds')
print('='*60)
for exp in TON_EXPS:
    for seed in EXTRA_SEEDS:
        run_exp(exp, 'ton_iot', seed)

print('='*60)
print('Phase 2: Running UNSW-NB15 extra seeds')
print('='*60)
for exp in UNSW_EXPS:
    for seed in EXTRA_SEEDS:
        run_exp(exp, 'unsw_nb15', seed)

# ── Aggregate mean ± std ───────────────────────────────────────────────────────
print('\n' + '='*60)
print('Phase 3: Aggregating across seeds [42, 43, 44]')
print('='*60)

ALL_SEEDS = [42] + EXTRA_SEEDS
summary = {}

for dataset, exps in [('ton_iot', TON_EXPS), ('unsw_nb15', UNSW_EXPS)]:
    summary[dataset] = {}
    for exp in exps:
        results = []
        for s in ALL_SEEDS:
            p = result_path(exp, dataset, s)
            if p.exists():
                try:
                    results.append(json.loads(p.read_text()))
                except Exception:
                    pass
        if not results:
            print(f'  [WARN] No results for {dataset}/{exp}')
            continue

        best_f1s  = [r['best_f1'] for r in results]
        final_f1s = [r['final_metrics']['f1'] for r in results]
        conv_rds  = [r['convergence_round'] for r in results if r.get('convergence_round')]

        summary[dataset][exp] = {
            'mean_best_f1':   float(np.mean(best_f1s)),
            'std_best_f1':    float(np.std(best_f1s)),
            'mean_final_f1':  float(np.mean(final_f1s)),
            'std_final_f1':   float(np.std(final_f1s)),
            'mean_conv_round': float(np.mean(conv_rds)) if conv_rds else None,
            'n_seeds':         len(results),
            'seeds':           ALL_SEEDS[:len(results)],
            'per_seed_best_f1': best_f1s,
        }
        print(f'  {dataset:<12} {exp:<25}  '
              f'F1={np.mean(best_f1s):.4f}±{np.std(best_f1s):.4f}  '
              f'(n={len(results)} seeds)', flush=True)

out = ROOT / 'experiments' / 'paper4_federated' / 'multi_seed_summary.json'
out.write_text(json.dumps(summary, indent=2))
print(f'\n[OK] Saved: {out}')
