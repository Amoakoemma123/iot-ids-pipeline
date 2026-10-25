"""
leakage_decomposition_fixed.py — Paper 5, Table 1 (v3.1): the effect of `id` with the feature set
held fixed. Every condition uses the 10 features selected by the corrected pipeline, with or
without `id` added, so `id` is the only feature that differs within a pair.

  A' id + 10 features, duplicates kept      B' 10 features, duplicates kept    (same split)
  C' id + 10 features, duplicates removed   D' 10 features, duplicates removed (same split)

Each pair is built from ONE load of the data (the no-id condition drops the `id` column), so the
two conditions of a pair share their training and test records exactly. RF seeds 42, 43, 44;
paired bootstrap CI of the per-record macro-F1 difference is approximated by resampling test
records (2,000 resamples, seed 42) for each seed.

    python -m experiments.paper5_xai.leakage_decomposition_fixed
"""
import gc, io, json, sys, time, contextlib
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.baseline.config import DATASETS, PREPROCESSING, PREPROCESSING_OVERRIDES
from src.data import load_dataset, PreprocessingPipeline
from src.models import build_rf
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler

OUT = ROOT / 'experiments' / 'paper5_xai' / 'results_v3' / 'UNSW_NB15' / 'leakage_decomposition_fixed.json'
CSV, LABEL = DATASETS['UNSW_NB15']
SEEDS = (42, 43, 44)
N_BOOT = 2000


def pipeline_for(names, cols, X_tr):
    """A fitted PreprocessingPipeline that keeps exactly `cols` (no MI/RFECV), then scales."""
    cfg = {**PREPROCESSING, **PREPROCESSING_OVERRIDES['UNSW_NB15']}
    p = PreprocessingPipeline(mi_threshold=cfg['mi_threshold'], rfecv_subsample=cfg['rfecv_subsample'], skip_rfecv=True,
                              smote_k=cfg['smote_k'], smote_max_per_class=cfg.get('smote_max_per_class'),
                              mi_subsample=cfg.get('mi_subsample'), seed=cfg['seed'])
    p.mi_mask_ = np.array([n in cols for n in names])
    p.selected_features_ = [n for n in names if n in cols]
    p.n_features_ = len(p.selected_features_)
    p.rfecv_ = None
    p.scaler_ = MinMaxScaler().fit(X_tr[:, p.mi_mask_])
    return p


def boot_diff(y, pa, pb, seed=42):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(N_BOOT):
        i = rng.integers(len(y), size=len(y))
        d.append(f1_score(y[i], pa[i], average='macro') - f1_score(y[i], pb[i], average='macro'))
    return dict(mean=float(f1_score(y, pa, average='macro') - f1_score(y, pb, average='macro')),
                lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))


def main():
    base = joblib.load(ROOT / 'experiments' / 'baseline' / 'results' / 'UNSW_NB15' / 'pipeline.pkl')
    feats = list(base.selected_features_)
    res = dict(features=feats, seeds=SEEDS)
    t0 = time.time()
    for pair, dedup in (('dups_kept', False), ('dups_removed', True)):
        with contextlib.redirect_stdout(io.StringIO()):
            raw = load_dataset(CSV, label_col=LABEL, seed=42, drop_id=False, deduplicate=dedup)
        names = raw['feature_names']
        r = dict(n_train=int(len(raw['y_train'])), n_test=int(len(raw['y_test'])), seeds={})
        for s in SEEDS:
            preds = {}
            for cond, cols in (('with_id', feats + ['id']), ('without_id', feats)):
                p = pipeline_for(names, cols, raw['X_train'])
                with contextlib.redirect_stdout(io.StringIO()):
                    Xtr, ytr = p.transform(raw['X_train'], apply_smote=True, y=raw['y_train'])
                Xte, _ = p.transform(raw['X_test'])
                rf = build_rf(seed=s); rf.fit(Xtr, ytr)
                preds[cond] = rf.predict(Xte)
                del rf, Xtr; gc.collect()
            y = raw['y_test']
            r['seeds'][str(s)] = dict(
                with_id=dict(macro_f1=float(f1_score(y, preds['with_id'], average='macro')), accuracy=float((preds['with_id'] == y).mean())),
                without_id=dict(macro_f1=float(f1_score(y, preds['without_id'], average='macro')), accuracy=float((preds['without_id'] == y).mean())),
                id_effect=boot_diff(y, preds['with_id'], preds['without_id']))
            print(pair, s, r['seeds'][str(s)]['id_effect'], flush=True)
        res[pair] = r
        del raw; gc.collect()
    res['_seconds'] = time.time() - t0
    OUT.write_text(json.dumps(res, indent=2))
    print('saved', OUT)


if __name__ == '__main__':
    main()
