"""Move EWC/LwF main/orders results whose hyperparameter differs from tuned.json into <group>_superseded/
(kept for the record; the resumable runner then recomputes exactly those). Prints the number moved."""
import json, shutil, sys
from pathlib import Path
d = Path(__file__).resolve().parent / 'results_v3' / sys.argv[1]
tn = json.loads((d / 'tuned.json').read_text())
n = 0
for g in ('main', 'orders'):
    for s in ('ewc', 'lwf'):
        want = {tn[s]['param']: tn[s]['value']}
        for p in (d / g).glob(f'{s}_o*_s*.json'):
            if json.loads(p.read_text())['hp'] != want:
                (d / f'{g}_superseded').mkdir(exist_ok=True)
                shutil.move(str(p), str(d / f'{g}_superseded' / p.name)); n += 1
# after a restart the files are already moved: still report them so that the queue reruns main/orders (runs skip finished files)
print(n or sum(len(list((d / f"{g}_superseded").glob("*.json"))) for g in ("main", "orders") if (d / f"{g}_superseded").exists()))
