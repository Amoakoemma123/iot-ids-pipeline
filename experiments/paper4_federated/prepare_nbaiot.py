"""
prepare_nbaiot.py — N-BaIoT (Meidan et al. 2018, UCI 442) for the Paper 4 device-level study.

Nine commercial IoT devices, each infected with BASHLITE (gafgyt) and, for seven of them, Mirai;
115 Kitsune stream statistics per packet; 11 classes (benign + 5 gafgyt + 5 mirai attacks).

Steps (all random choices seeded):
  1. uniform Bernoulli sample of SAMPLE_FRAC of the rows of every (device, class) file, which keeps
     each device's natural class mix (the full data have 7.06 M rows);
  2. remove exact duplicates (features + label + device); identical feature vectors with different
     labels are counted and dropped as a whole;
  3. stratified 70/15/15 split by (device, class);
  4. zero-variance features (on the training split) dropped, min-max scaling fitted on training only.
No feature selection: the CNN takes all remaining features.
Output: dataset4/nbaiot_prepared.npz and nbaiot_prepared.json (counts).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'dataset4' / 'raw'
OUT = ROOT / 'dataset4'
SAMPLE_FRAC = 0.03
SEED = 42


def files():
    for dev in sorted(p for p in RAW.iterdir() if p.is_dir()):
        yield dev.name, 'normal', dev / 'benign_traffic.csv'
        for fam in ('gafgyt', 'mirai'):
            d = dev / f'{fam}_attacks'
            if d.is_dir():
                for f in sorted(d.glob('*.csv')):
                    yield dev.name, f'{fam}_{f.stem}', f


def main():
    rng = np.random.default_rng(SEED)
    parts, counts = [], {}
    for dev, cls, f in files():
        df = pd.read_csv(f, dtype=np.float64)
        keep = rng.random(len(df)) < SAMPLE_FRAC
        s = df[keep].copy()
        s['label'], s['device'] = cls, dev
        parts.append(s)
        counts[f'{dev}/{cls}'] = dict(rows=int(len(df)), sampled=int(keep.sum()))
        print(f'{dev:42s} {cls:18s} {len(df):8d} -> {keep.sum():6d}', flush=True)
    df = pd.concat(parts, ignore_index=True)
    feats = [c for c in df.columns if c not in ('label', 'device')]
    n0 = len(df)
    df = df.drop_duplicates(subset=feats + ['label', 'device'])
    n_dup = n0 - len(df)
    conflict = df.duplicated(subset=feats, keep=False) & (df.groupby(feats, sort=False)['label'].transform('nunique') > 1)
    n_conf = int(conflict.sum())
    df = df[~conflict].reset_index(drop=True)
    print(f'sampled {n0}, exact duplicates removed {n_dup}, conflicting-label rows removed {n_conf}, kept {len(df)}')

    classes = ['normal'] + sorted(c for c in df['label'].unique() if c != 'normal')
    devices = sorted(df['device'].unique())
    y = df['label'].map({c: i for i, c in enumerate(classes)}).values.astype(np.int32)
    dvc = df['device'].map({d: i for i, d in enumerate(devices)}).values.astype(np.int32)
    strat = dvc * 100 + y
    idx = np.arange(len(df))
    i_tr, i_tmp = train_test_split(idx, test_size=0.30, stratify=strat, random_state=SEED)
    i_va, i_te = train_test_split(i_tmp, test_size=0.50, stratify=strat[i_tmp], random_state=SEED)

    X = df[feats].values
    nzv = X[i_tr].std(0) > 0
    X = X[:, nzv]
    lo, hi = X[i_tr].min(0), X[i_tr].max(0)
    X = np.clip((X - lo) / (hi - lo), 0, None).astype(np.float32)   # test values above the train max are kept (> 1)
    fnames = [f for f, k in zip(feats, nzv) if k]

    np.savez_compressed(OUT / 'nbaiot_prepared.npz',
                        Xtr=X[i_tr], ytr=y[i_tr], dtr=dvc[i_tr], Xva=X[i_va], yva=y[i_va], dva=dvc[i_va],
                        Xte=X[i_te], yte=y[i_te], dte=dvc[i_te])
    info = dict(sample_frac=SAMPLE_FRAC, seed=SEED, n_sampled=n0, n_exact_dup=int(n_dup), n_conflict=n_conf,
                n_kept=int(len(df)), n_train=int(len(i_tr)), n_val=int(len(i_va)), n_test=int(len(i_te)),
                classes=classes, devices=devices, features=fnames, n_features=len(fnames),
                train_counts={d: np.bincount(y[i_tr][dvc[i_tr] == k], minlength=len(classes)).tolist()
                              for k, d in enumerate(devices)},
                files=counts)
    (OUT / 'nbaiot_prepared.json').write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k not in ('files', 'features')}, indent=1))


if __name__ == '__main__':
    main()
