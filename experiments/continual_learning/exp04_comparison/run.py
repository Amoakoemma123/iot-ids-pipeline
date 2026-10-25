"""
exp04_comparison/run.py
Full cross-strategy comparison experiment.

Runs all 5 strategies (naive, ewc, replay, lwf, gem) on all three datasets
and produces a unified comparison summary across strategies and datasets.
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.continual_learning.run_continual import run_experiment
from experiments.continual_learning.config import RESULTS_DIR

DATASETS   = ['TON_IoT', 'UNSW_NB15', 'CICIDS2017']
STRATEGIES = ['naive', 'ewc', 'replay', 'lwf', 'gem']


def build_comparison_table():
    """Read per-strategy result JSONs and build a comparison dict."""
    comparison = {}
    for dataset in DATASETS:
        comparison[dataset] = {}
        for strategy in STRATEGIES:
            result_path = RESULTS_DIR / dataset / f'{strategy}_results.json'
            if result_path.exists():
                with open(result_path) as f:
                    data = json.load(f)
                comparison[dataset][strategy] = {
                    'average_accuracy': data.get('average_accuracy'),
                    'plasticity':       data.get('plasticity'),
                    'forgetting':       data.get('forgetting'),
                    'bwt':              data.get('bwt'),
                    'elapsed_sec':      data.get('elapsed_sec'),
                }
            else:
                comparison[dataset][strategy] = None
    return comparison


def print_comparison_table(comparison: dict):
    header = f"{'Strategy':<10} {'Avg Acc':>8} {'Plasticity':>10} {'Forgetting':>10} {'BWT':>8} {'Time(s)':>8}"
    for dataset, strategies in comparison.items():
        print(f"\n{'='*60}")
        print(f"  Dataset: {dataset}")
        print(f"{'='*60}")
        print(header)
        print('-' * 60)
        for strategy, metrics in strategies.items():
            if metrics is None:
                print(f"{strategy:<10}  (no results)")
                continue
            print(f"{strategy:<10} "
                  f"{metrics['average_accuracy']:>8.4f} "
                  f"{metrics['plasticity']:>10.4f} "
                  f"{metrics['forgetting']:>10.4f} "
                  f"{metrics['bwt']:>8.4f} "
                  f"{metrics['elapsed_sec']:>8.1f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true',
                        help='Re-run all experiments before comparing')
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--strategies', nargs='+', default=STRATEGIES)
    args = parser.parse_args()

    if args.run:
        for dataset in args.datasets:
            run_experiment(dataset, strategies=args.strategies)

    comparison = build_comparison_table()
    print_comparison_table(comparison)

    out_path = RESULTS_DIR / 'comparison_summary.json'
    with open(out_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved: {out_path}")
