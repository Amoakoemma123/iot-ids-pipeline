"""Re-tuning after the selection criterion was corrected (current-task -> all-seen-class validation F1).
--archive: move the old tune results of a dataset aside (they lack val_seen_f1).
--prune:   delete EWC/LwF main/orders results whose hyperparameter differs from the new tuned.json,
           so that the resumable runner recomputes exactly those."""
import argparse, json, shutil
from pathlib import Path
OUT = Path(__file__).resolve().parent / 'results_v3'
ap = argparse.ArgumentParser(); ap.add_argument('--dataset', required=True)
ap.add_argument('--archive', action='store_true'); ap.add_argument('--prune', action='store_true')
a = ap.parse_args(); d = OUT / a.dataset
if a.archive:
    if (d / 'tune').exists():
        shutil.move(str(d / 'tune'), str(d / 'tune_v3a_currenttask'))
    if (d / 'tuned.json').exists():
        (d / 'tuned.json').replace(d / 'tuned_v3a_currenttask.json')
    print('archived old tuning for', a.dataset)
if a.prune:
    tn = json.loads((d / 'tuned.json').read_text())
    n = 0
    for g in ('main', 'orders'):
        for s in ('ewc', 'lwf'):
            want = {tn[s]['param']: tn[s]['value']}
            for p in (d / g).glob(f'{s}_o*_s*.json'):
                if json.loads(p.read_text())['hp'] != want:
                    p.unlink(); n += 1
    print(f'pruned {n} EWC/LwF results with outdated hyperparameters; tuned = {tn}')
