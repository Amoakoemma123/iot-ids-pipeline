"""
twin_mask.py — Paper 5 (final round, referee M6): which test records have a feature-space twin in
training, i.e. are identical to a training record in the selected, scaled features (exact float32
equality, as in feature_space_copies.py). Saves the test indices so that saved per-record outputs
(local_c, deletion_c, hardening_paired) can be re-summarised without these records. Data load only,
no model is trained or loaded.

    python -m experiments.paper5_xai.twin_mask
Output: results_final/<ds>/twin_mask.json
"""
import io, json, sys, contextlib
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.baseline.config import DATASETS as BASE_DATASETS
from experiments.paper5_xai.feature_space_copies import keys

OUT = ROOT / 'experiments' / 'paper5_xai' / 'results_final'
BASE_RES = ROOT / 'experiments' / 'baseline' / 'results'


def main():
    from src.data.loader import load_dataset
    for ds in ('TON_IoT', 'UNSW_NB15'):
        pipe = joblib.load(BASE_RES / ds / 'pipeline.pkl')
        csv, label_col = BASE_DATASETS[ds]
        with contextlib.redirect_stdout(io.StringIO()):
            raw = load_dataset(csv, label_col=label_col, seed=42)
            X_tr, _ = pipe.transform(raw['X_train'])
            X_te, _ = pipe.transform(raw['X_test'])
        y_tr, y_te = raw['y_train'], raw['y_test']
        tr_l, tr_f = set(keys(X_tr, y_tr)), set(keys(X_tr))
        copy_l = np.array([k in tr_l for k in keys(X_te, y_te)])
        copy_f = np.array([k in tr_f for k in keys(X_te)])
        classes = list(raw['class_names'])
        res = dict(definition='exact float32 equality of the selected, scaled features (as feature_space_copies.py)',
                   n_test=int(len(y_te)), share_twin_with_label=float(copy_l.mean()), share_twin_features=float(copy_f.mean()),
                   twin_features_idx=np.where(copy_f)[0].tolist(), twin_with_label_idx=np.where(copy_l)[0].tolist(),
                   per_class_twin_features={c: float(copy_f[y_te == i].mean()) for i, c in enumerate(classes)})
        (OUT / ds).mkdir(parents=True, exist_ok=True)
        (OUT / ds / 'twin_mask.json').write_text(json.dumps(res))
        print(ds, f"twins (features) {res['share_twin_features']:.4f}, with label {res['share_twin_with_label']:.4f}", flush=True)


if __name__ == '__main__':
    main()
