"""
exp03_lwf/run.py
Run the Learning without Forgetting continual learning experiment on all three datasets.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.continual_learning.run_continual import run_experiment

if __name__ == '__main__':
    for dataset in ['TON_IoT', 'UNSW_NB15', 'CICIDS2017']:
        run_experiment(dataset, strategies=['lwf'])
