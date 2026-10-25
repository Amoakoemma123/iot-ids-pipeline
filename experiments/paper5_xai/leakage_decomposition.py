"""
leakage_decomposition.py — Paper 5, Table 1: separate the effects of the `id` column, of
duplicate records and of the changed test set on the UNSW-NB15 Random Forest.

Conditions (same RF settings and pipeline configuration as the baseline, seed 42):
  A  original v1 model (id kept, no effective dedup), evaluated on (i) the full v1 test set
     and (ii) the v1 test records that have no exact copy (all columns except id) in v1 training
  B  id removed, duplicates kept
  C  id kept, duplicates removed (judged on all columns except id)
  D  id removed, duplicates removed (= corrected pipeline, v2.1)
B–D are trained here and evaluated on their own test sets; B also on its unseen subset.

    python -m experiments.paper5_xai.leakage_decomposition
"""
import gc, json, sys, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.baseline.config import DATASETS, PREPROCESSING, PREPROCESSING_OVERRIDES
from src.data import load_dataset, PreprocessingPipeline
from src.models import build_rf
from sklearn.metrics import f1_score

OUT = ROOT / 'experiments' / 'paper5_xai' / 'results_v3' / 'UNSW_NB15' / 'leakage_decomposition.json'
V1 = ROOT.parent / '_v1_backup_2026-10-24' / 'experiments' / 'baseline' / 'results'
CSV, LABEL = DATASETS['UNSW_NB15']


def keys(X, y, drop_cols=()):
    df = pd.DataFrame(X)
    if drop_cols:
        df = df.drop(columns=list(drop_cols))
    df['__y'] = y
    return pd.util.hash_pandas_object(df, index=False).values


def scores(y, p):
    return dict(macro_f1=float(f1_score(y, p, average='macro')), accuracy=float((p == y).mean()), n=int(len(y)))


def unseen_mask(raw, id_col):
    drop = (id_col,) if id_col is not None else ()
    tr = set(keys(raw['X_train'], raw['y_train'], drop))
    return np.array([k not in tr for k in keys(raw['X_test'], raw['y_test'], drop)])


def train_eval(raw):
    cfg = {**PREPROCESSING, **PREPROCESSING_OVERRIDES['UNSW_NB15']}
    pipe = PreprocessingPipeline(mi_threshold=cfg['mi_threshold'], rfecv_subsample=cfg['rfecv_subsample'],
                                 skip_rfecv=cfg.get('skip_rfecv', False), smote_k=cfg['smote_k'],
                                 smote_max_per_class=cfg.get('smote_max_per_class'), mi_subsample=cfg.get('mi_subsample'),
                                 seed=cfg['seed'])
    Xtr, ytr = pipe.fit_transform(raw['X_train'], raw['y_train'], feature_names=raw['feature_names'], apply_smote=True)
    Xte, _ = pipe.transform(raw['X_test'])
    rf = build_rf(seed=42); rf.fit(Xtr, ytr)
    p = rf.predict(Xte)
    del rf; gc.collect()
    return p, list(pipe.selected_features_)


def main():
    res = {}
    t0 = time.time()
    # A: original v1 model on the v1 split (the v1 split is reproduced by drop_id=False, deduplicate=False)
    rawA = load_dataset(CSV, label_col=LABEL, seed=42, drop_id=False, deduplicate=False)
    id_col = rawA['feature_names'].index('id')
    pipeA = joblib.load(V1 / 'UNSW_NB15' / 'pipeline.pkl')
    XteA, _ = pipeA.transform(rawA['X_test'])
    rfA = joblib.load(V1 / 'models' / 'UNSW_NB15' / 'model_rf.pkl')
    pA = rfA.predict(XteA); del rfA; gc.collect()
    mA = unseen_mask(rawA, id_col)
    res['A_original'] = dict(features=list(pipeA.selected_features_), full_test=scores(rawA['y_test'], pA),
                             unseen_test=scores(rawA['y_test'][mA], pA[mA]), seen_test=scores(rawA['y_test'][~mA], pA[~mA]),
                             share_test_seen=float((~mA).mean()))
    print('A', res['A_original']['full_test'], res['A_original']['unseen_test'], flush=True)

    for name, kw in (('B_id_removed_duplicates_kept', dict(drop_id=True, deduplicate=False)),
                     ('C_id_kept_duplicates_removed', dict(drop_id=False, deduplicate=True)),
                     ('D_corrected', dict(drop_id=True, deduplicate=True))):
        raw = load_dataset(CSV, label_col=LABEL, seed=42, **kw)
        p, feats = train_eval(raw)
        r = dict(features=feats, full_test=scores(raw['y_test'], p))
        if not kw['deduplicate']:
            m = unseen_mask(raw, None)
            r.update(unseen_test=scores(raw['y_test'][m], p[m]), share_test_seen=float((~m).mean()))
        res[name] = r
        print(name, r['full_test'], r.get('unseen_test'), flush=True)
        del raw; gc.collect()
    res['_seconds'] = time.time() - t0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print('saved', OUT)


if __name__ == '__main__':
    main()
