"""P2 final review round 2, minor 11: capacity of the saved tree models (GBDT iterations after early stopping; RF
leaves and depth). No training. Writes results_r2/<ds>/tree_capacity.json.
Run:  python -m experiments.paper2_adv_v2.tree_capacity"""
import json
import joblib
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
for ds in ('TON_IoT', 'UNSW_NB15'):
    out = {}
    for tag in ('rf_none', 'gb_none', 'rf_augment', 'gb_augment'):
        rows = []
        for s in range(5):
            f = HERE / 'models_v3' / ds / tag / f'seed{s}.joblib'
            if not f.exists():
                continue
            m = joblib.load(f)
            m = getattr(m, 'model', m)
            if hasattr(m, 'n_iter_'):
                rows.append(dict(seed=s, n_iter=int(m.n_iter_)))
            elif hasattr(m, 'estimators_'):
                rows.append(dict(seed=s, leaves=int(np.mean([e.get_n_leaves() for e in m.estimators_])),
                                 depth=int(np.max([e.get_depth() for e in m.estimators_]))))
            else:
                rows.append(dict(seed=s, type=type(m).__name__))
        out[tag] = rows
    print(ds, json.dumps(out))
    (HERE / 'results_r2' / ds / 'tree_capacity.json').write_text(json.dumps(out, indent=1))
