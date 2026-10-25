"""
feature_space_copies.py — Paper 5 (v3.1): test records that are distinct in the cleaned data but
identical to a training record in the models' selected (scaled) features, and test performance
with and without them, for the RF and CNN of every seed.

    python -m experiments.paper5_xai.feature_space_copies
"""
import gc, io, json, sys, contextlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.paper5_xai.run_xai_v3 import load_data, load_rf, load_cnn, cnn_fn, SEEDS
from sklearn.metrics import f1_score

OUT = ROOT / 'experiments' / 'paper5_xai' / 'results_v3'


def keys(X, y=None, nd=None):
    """Row hashes; nd=None: exact float32 equality, nd=8: equal to 8 decimals."""
    X = np.asarray(X, np.float32)
    df = pd.DataFrame(X if nd is None else np.round(X.astype(float), nd))
    if y is not None:
        df['__y'] = y
    return pd.util.hash_pandas_object(df, index=False).values


def main():
    for ds in ('TON_IoT', 'UNSW_NB15'):
        with contextlib.redirect_stdout(io.StringIO()):
            D = load_data(ds)
        y = D['y_te']
        near = {}
        for nd in (8, 6):
            t = set(keys(D['X_tr'], D['y_tr'], nd))
            near[f'share_copy_with_label_{nd}dp'] = float(np.mean([k in t for k in keys(D['X_te'], y, nd)]))
        tr_l, tr_f = set(keys(D['X_tr'], D['y_tr'])), set(keys(D['X_tr']))
        copy_l = np.array([k in tr_l for k in keys(D['X_te'], y)])
        copy_f = np.array([k in tr_f for k in keys(D['X_te'])])
        res = dict(definition='exact float32 equality of the selected, scaled features', **near,
                   n_test=int(len(y)), share_copy_with_label=float(copy_l.mean()), share_copy_features_only=float(copy_f.mean()),
                   per_class_copy_with_label={c: float(copy_l[y == i].mean()) for i, c in enumerate(D['classes'])},
                   n_test_without_copy=int((~copy_f).sum()), seeds={})
        for s in SEEDS:
            r = {}
            for m in ('rf', 'cnn'):
                model = load_rf(ds, s) if m == 'rf' else load_cnn(ds, s)
                p = (model.predict(D['X_te']) if m == 'rf' else cnn_fn(model)(D['X_te']).argmax(1))
                lab = list(range(len(D['classes'])))
                r[m] = dict(macro_f1_all=float(f1_score(y, p, average='macro', labels=lab)),
                            macro_f1_without_copies=float(f1_score(y[~copy_f], p[~copy_f], average='macro', labels=lab)),
                            accuracy_all=float((p == y).mean()), accuracy_without_copies=float((p[~copy_f] == y[~copy_f]).mean()),
                            accuracy_on_copies=float((p[copy_f] == y[copy_f]).mean()) if copy_f.any() else None)
                del model; gc.collect()
            res['seeds'][str(s)] = r
            print(ds, s, r, flush=True)
        (OUT / ds / 'feature_space_copies.json').write_text(json.dumps(res, indent=2))
        print('saved', ds, res['share_copy_with_label'], res['share_copy_features_only'])


if __name__ == '__main__':
    main()
