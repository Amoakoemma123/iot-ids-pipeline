"""
run_p4final.py — Paper 4, final review round (review_final_referee.md M1-M3): small diagnostic runs.

1. Own traffic under a BLOCKED per-site split (M1). Within every (site unit, class) group the records are taken in
   file order (N-BaIoT: capture order of the device-class file; TON-IoT: row order of train_test_network.csv, which
   the dataset does not document as temporal) and cut into contiguous blocks: first 65% training, 5% gap, 12.5%
   validation, 5% gap, last 12.5% test. Site unit = destination host (TON-IoT) or device (N-BaIoT). The RANDOM
   control cuts the same groups with the same fractions after a random permutation, so the two arms differ only in
   the order. FedAvg (host / device partition, tuned lr) and local-only models, as in run_fl_v2.py; every site is
   scored on its own held-out test traffic.
2. Label-prior correction of the selected federated model at each site (M2), at no training cost: add
   log(pi_site,c) - log(pi_pooled,c) to the log-probabilities and mask classes absent from the site's training data
   ('prior'), or mask only ('mask'). Computed for every FedAvg run above and, for the paper's original random split,
   in FedAvg re-runs (fedavg_natr_prior, fedavg_dev_prior; the local models are those of results_v2, same partitions).
3. DP optimiser check at matched epsilon (M3), TON-IoT, alpha = 0.5, short schedule (20 rounds x 1 epoch), the
   epsilon of the paper's sigma = 1 runs (about 3): DP-SGD with SGD + momentum 0.9 (lr 0.03 and 0.1, choice by
   validation) and Adam with a four times larger expected batch (B = 1024) and sigma recalibrated per repetition to
   the epsilon of the B = 256 run with the same seed. Optional full budget (50 x 2, epsilon of the sigma = 1.8 runs):
   the better SGD-momentum lr and B = 1024.

Usage (from the Papers pipeline root):
  python -m experiments.paper4_federated.run_p4final --group prep_nbaiot
  python -m experiments.paper4_federated.run_p4final --group own --dataset TON_IoT --split blocked --seeds 0,1,2
  python -m experiments.paper4_federated.run_p4final --group prior --dataset N_BaIoT --seeds 0,1,2
  python -m experiments.paper4_federated.run_p4final --group dp_short --seeds 0,1,2 --cpu-dp
  python -m experiments.paper4_federated.run_p4final --group check            (import and data check only)
Results: experiments/paper4_federated/results_p4final/<dataset>/<experiment>/seed<r>.json (skipped if present).
"""
import argparse, contextlib, gc, io, json, os, sys, time, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
if '--cpu-dp' in sys.argv:          # DP-SGD on the CPU (per-example gradients), thread count from P4_THREADS
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
if '--cpu-dp' in sys.argv:
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get('P4_THREADS', '2')))
    tf.config.threading.set_inter_op_parallelism_threads(1)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.paper4_federated import run_fl_v2 as rf          # noqa: E402  (reuses data, model, training, scoring)
from sklearn.metrics import f1_score                              # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_p4final'
FRACS = (0.65, 0.05, 0.125, 0.05, 0.125)    # train, gap, validation, gap, test (per site unit and class, in file order)
SGDM_LRS = (0.03, 0.1)
BIG_B = 1024


# ── blocked / random per-site splits ─────────────────────────────────────────

def group_split(unit, y, order, mode, rng):
    """Cut every (unit, class) group into train / gap / val / gap / test blocks. mode 'blocked': in `order`
    (file order); 'random': after a random permutation (same sizes)."""
    key = unit.astype(np.int64) * 1000 + y
    tr, va, te = [], [], []
    for k in np.unique(key):
        ix = np.where(key == k)[0]
        ix = ix[np.argsort(order[ix], kind='stable')] if mode == 'blocked' else rng.permutation(ix)
        n = len(ix)
        nva, nte, ng = int(FRACS[2] * n), int(FRACS[4] * n), int(FRACS[1] * n)   # small groups stay in training
        ntr = n - nva - nte - 2 * ng
        tr.append(ix[:ntr]); va.append(ix[ntr + ng:ntr + ng + nva]); te.append(ix[n - nte:])
    return [np.sort(np.concatenate(a)) for a in (tr, va, te)]


def guard(X, y, tr, others):
    """Remove validation/test records identical (features and label) to a training record."""
    import pandas as pd
    key = lambda ix: pd.util.hash_pandas_object(pd.DataFrame(np.column_stack([X[ix], y[ix]])), index=False).values
    seen = set(key(tr))
    out, removed = [], []
    for ix in others:
        keep = np.array([k not in seen for k in key(ix)], bool) if len(ix) else np.zeros(0, bool)
        out.append(ix[keep]); removed.append(int((~keep).sum()))
    return out, removed


def load_ton_split(mode):
    """TON-IoT records of the paper's pipeline (same cleaning, feature selection and scaling as run_fl_v2.load_data),
    pooled over its train/val/test splits and re-split per destination host and class."""
    import joblib
    from src.data.loader import load_dataset
    from experiments.baseline.config import DATASETS
    pipe = joblib.load(rf.BASE_RES / 'TON_IoT' / 'pipeline.pkl')
    csv, label = DATASETS['TON_IoT']
    with contextlib.redirect_stdout(io.StringIO()):
        raw = load_dataset(csv, label_col=label, seed=42, return_meta=True, meta_row=True)
    tfm = lambda X: pipe.transform(X)[0].astype(np.float32)
    X = np.concatenate([tfm(raw[f'X_{s}']) for s in ('train', 'val', 'test')])
    y = np.concatenate([raw[f'y_{s}'] for s in ('train', 'val', 'test')]).astype(np.int32)
    dst = np.concatenate([raw[f'meta_{s}']['dst_ip'].astype(str).values for s in ('train', 'val', 'test')])
    row = np.concatenate([raw[f'meta_{s}']['row'].values for s in ('train', 'val', 'test')])
    hosts, unit = np.unique(dst, return_inverse=True)
    tr, va, te = group_split(unit, y, row, mode, np.random.default_rng(7))
    (va, te), removed = guard(X, y, tr, [va, te])
    classes = list(raw['class_names'])
    return dict(ds='TON_IoT', Xtr=X[tr], ytr=y[tr], Xva=X[va], yva=y[va], Xte=X[te], yte=y[te], classes=classes,
                normal=classes.index('normal'), features=list(pipe.selected_features_), dropped=[],
                dst_tr=dst[tr], dst_te=dst[te], dev_tr=None, dev_te=None, guard_removed=removed, split=mode)


def prep_nbaiot():
    """N-BaIoT as prepare_nbaiot.py (same 3% sample, float32 deduplication, conflicting labels and the two BASHLITE
    floods dropped), keeping each record's row in its device-class file, then cut per device and class into blocks
    in capture order ('blocked') or after a random permutation ('random'); zero-variance features and min-max
    scaling on the new training split; cross-split guard. Writes dataset4/nbaiot_{blocked,random}.npz/.json."""
    import pandas as pd
    from experiments.paper4_federated import prepare_nbaiot as pn
    rng = np.random.default_rng(pn.SEED)
    parts = []
    for dev, cls, f in pn.files():                    # same file order and random draws as prepare_nbaiot.py
        df = pd.read_csv(f, dtype=np.float64)
        keep = rng.random(len(df)) < pn.SAMPLE_FRAC
        s = df[keep].copy()
        s['label'], s['device'], s['row'] = cls, dev, np.where(keep)[0]
        parts.append(s)
        print(f'{dev:42s} {cls:18s} {len(df):8d} -> {keep.sum():6d}', flush=True)
        del df
    df = pd.concat(parts, ignore_index=True); del parts
    feats = [c for c in df.columns if c not in ('label', 'device', 'row')]
    df[feats] = df[feats].astype(np.float32)
    df = df.drop_duplicates(subset=feats + ['label', 'device'])
    conflict = df.duplicated(subset=feats, keep=False) & (df.groupby(feats, sort=False)['label'].transform('nunique') > 1)
    df = df[~conflict].reset_index(drop=True)
    n_distinct = df.groupby('label').size()
    df = df[~df['label'].isin(sorted(n_distinct[n_distinct < pn.MIN_DISTINCT].index))].reset_index(drop=True)
    classes = ['normal'] + sorted(c for c in df['label'].unique() if c != 'normal')
    devices = sorted(df['device'].unique())
    y = df['label'].map({c: i for i, c in enumerate(classes)}).values.astype(np.int32)
    dvc = df['device'].map({d: i for i, d in enumerate(devices)}).values.astype(np.int32)
    row = df['row'].values
    X0 = df[feats].values
    for mode in ('blocked', 'random'):
        tr, va, te = group_split(dvc, y, row, mode, np.random.default_rng(7))
        nzv = X0[tr].std(0) > 0
        X = X0[:, nzv]
        lo, hi = X[tr].min(0), X[tr].max(0)
        X = np.clip((X.astype(np.float64) - lo) / (hi - lo), 0, None).astype(np.float32)
        (va, te), removed = guard(X, y, tr, [va, te])
        np.savez_compressed(ROOT / 'dataset4' / f'nbaiot_{mode}.npz', Xtr=X[tr], ytr=y[tr], dtr=dvc[tr],
                            Xva=X[va], yva=y[va], dva=dvc[va], Xte=X[te], yte=y[te], dte=dvc[te])
        info = dict(split=mode, fracs=FRACS, classes=classes, devices=devices, n_features=int(nzv.sum()),
                    features=[f for f, k in zip(feats, nzv) if k], n_train=int(len(tr)), n_val=int(len(va)),
                    n_test=int(len(te)), guard_removed=removed)
        (ROOT / 'dataset4' / f'nbaiot_{mode}.json').write_text(json.dumps(info, indent=1))
        print(mode, {k: v for k, v in info.items() if k != 'features'}, flush=True)


def load_nbaiot_split(mode):
    z = np.load(ROOT / 'dataset4' / f'nbaiot_{mode}.npz')
    info = json.loads((ROOT / 'dataset4' / f'nbaiot_{mode}.json').read_text())
    c = info['classes']
    return dict(ds='N_BaIoT', Xtr=z['Xtr'], ytr=z['ytr'], Xva=z['Xva'], yva=z['yva'], Xte=z['Xte'], yte=z['yte'],
                classes=c, normal=c.index('normal'), features=info['features'], dropped=[], dev_tr=z['dtr'],
                dev_te=z['dte'], devices=info['devices'], dst_tr=None, dst_te=None, split=mode)


# ── label-prior correction (M2), hooked into run_fl_v2.Evaluator.site_scores ──

PRIOR_CTX = None          # set to {'counts': [per-client training class counts]} while a FedAvg run is scored
_orig_site_scores = rf.Evaluator.site_scores


def _site_scores_with_prior(self, site_te):
    out = _orig_site_scores(self, site_te)
    if PRIOR_CTX is None:
        return out
    cnt = np.asarray(PRIOR_CTX['counts'], float)
    pool = cnt.sum(0) / cnt.sum()
    nrm = self.D['normal']
    for k, (ix, rec) in enumerate(zip(site_te, out)):
        if len(ix) == 0:
            continue
        X, y = self.D['Xte'][ix], self.D['yte'][ix]
        P = np.concatenate([self.model(tf.expand_dims(tf.constant(X[i:i + 8192]), -1), training=False).numpy()
                            for i in range(0, len(X), 8192)])
        pi = cnt[k] / cnt[k].sum()
        for mode in ('mask', 'prior'):
            lp = np.log(P + 1e-12)
            if mode == 'prior':
                lp = lp + np.log(np.maximum(pi, 1e-12)) - np.log(np.maximum(pool, 1e-12))
            lp[:, cnt[k] == 0] = -np.inf
            p = lp.argmax(1)
            rec[mode] = dict(f1=float(f1_score(y, p, average='macro', labels=np.unique(y), zero_division=0)),
                             acc=float((p == y).mean()),
                             benign_fpr=float((p[y == nrm] != nrm).mean()) if (y == nrm).any() else None,
                             attack_detection=float((p[y != nrm] != nrm).mean()) if (y != nrm).any() else None)
    return out


rf.Evaluator.site_scores = _site_scores_with_prior


def run_fed_prior(D, cfg, rep):
    """FedAvg as run_fl_v2.run_federated, with the label-prior correction added to the own-traffic scores."""
    global PRIOR_CTX
    clients, _ = rf.make_partition(D, cfg, np.random.default_rng(1000 + rep))     # the partition run_federated draws
    PRIOR_CTX = dict(counts=[np.bincount(D['ytr'][c], minlength=len(D['classes'])) for c in clients])
    try:
        return rf.run_federated(D, {**cfg, 'keep_best': True}, rep)
    finally:
        PRIOR_CTX = None


# ── DP-SGD with other optimisers / batch sizes (M3) ──────────────────────────

def eps_max(sizes, B, rounds, epochs, sigma):
    import dp_accounting
    out = []
    for n in sizes:
        q = min(1.0, B / n)
        acc = dp_accounting.rdp.RdpAccountant()
        acc.compose(dp_accounting.SelfComposedDpEvent(
            dp_accounting.PoissonSampledDpEvent(q, dp_accounting.GaussianDpEvent(sigma)),
            rounds * epochs * int(np.ceil(1.0 / q))))
        out.append(float(acc.get_epsilon(rf.DELTA)))
    return max(out), out


def calibrate_sigma(sizes, B, rounds, epochs, target):
    lo, hi = 0.2, 40.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if eps_max(sizes, B, rounds, epochs, mid)[0] > target:
            lo = mid
        else:
            hi = mid
    return hi


def run_dp(D, cfg, rep):
    """FedAvg with record-level DP-SGD on every client (as run_fl_v2.run_federated for algo 'dpsgd'), with the
    expected batch size cfg['batch'] and the local optimiser cfg['opt'] ('adam', or 'sgdm': SGD with momentum 0.9,
    state reset every round like Adam). sigma is calibrated to cfg['target_eps'] when given."""
    B = cfg['batch']
    rng = np.random.default_rng(2000 + rep); prng = np.random.default_rng(1000 + rep)
    tf.keras.utils.set_random_seed(rep)
    clients, _ = rf.make_partition(D, cfg, prng)
    sizes = [len(c) for c in clients]
    cfg = dict(cfg)
    if cfg.get('target_eps'):
        cfg['sigma'] = calibrate_sigma(sizes, B, cfg['rounds'], cfg['local_epochs'], cfg['target_eps'])
    nf, nc = D['Xtr'].shape[1], len(D['classes'])
    gmodel = rf.build_model(nf, nc, 'gn', dropout=0); cmodel = rf.build_model(nf, nc, 'gn', dropout=0)
    rf.assign(cmodel, rf.weights(gmodel))
    lt = rf.LocalTrainer(cmodel, D['Xtr'], D['ytr'], 'dpsgd', cfg['lr'], clip=cfg['clip'], sigma=cfg['sigma'], expected_b=B)
    if cfg['opt'] == 'sgdm':
        lt.opt = tf.keras.optimizers.SGD(cfg['lr'], momentum=0.9)   # replaced before the first (tracing) call
    ev = rf.Evaluator(gmodel, D)
    gw = rf.weights(gmodel)
    hist, t0, best_w = [], time.time(), None
    for rnd in range(1, cfg['rounds'] + 1):
        upd = []
        for idx in clients:
            rf.assign(cmodel, gw); lt.reset_optimizer()
            lt.run(idx, cfg['local_epochs'], rng, poisson_q=min(1.0, B / len(idx)))
            upd.append(rf.weights(cmodel))
        tot = float(sum(sizes))
        gw = [sum(u[i] * (s / tot) for u, s in zip(upd, sizes)) for i in range(len(gw))]
        rf.assign(gmodel, gw)
        h = dict(round=rnd, **{f'val_{k}': v for k, v in ev.scores(D['Xva'], D['yva']).items()},
                 **{f'test_{k}': v for k, v in ev.scores(D['Xte'], D['yte']).items()}, sec=time.time() - t0)
        if not np.isfinite(sum(float(np.abs(w).sum()) for w in gw)):
            h['nonfinite'] = True
        hist.append(h)
        if rf.select(hist) == len(hist) - 1:
            best_w = [w.copy() for w in gw]
        if rnd % 5 == 0 or rnd == cfg['rounds']:
            print(f"    r{rnd:3d} val {h['val_f1']:.4f} test {h['test_f1']:.4f} ({h['sec']:.0f}s)", flush=True)
    s = rf.select(hist)
    fin_full = ev.scores(D['Xte'], D['yte'], full=True)
    rf.assign(gmodel, best_w)
    sel_full = ev.scores(D['Xte'], D['yte'], full=True)
    e_max, e_all = eps_max(sizes, B, cfg['rounds'], cfg['local_epochs'], cfg['sigma'])
    out = dict(cfg=cfg, rep=rep, client_sizes=sizes, selected_round=hist[s]['round'], val_f1_selected=hist[s]['val_f1'],
               test_f1_selected=hist[s]['test_f1'], test_f1_final=hist[-1]['test_f1'], test_selected_full=sel_full,
               test_final_full=fin_full, history=hist, seconds=time.time() - t0,
               any_nonfinite=any(h.get('nonfinite') for h in hist),
               dp=dict(epsilon_max=e_max, epsilon_per_client=e_all, delta=rf.DELTA, sigma=cfg['sigma'], clip=cfg['clip'],
                       batch=B, sampling='Poisson, q = B / n_k per client', unit='record-level, per client'))
    del gmodel, cmodel, lt, ev; tf.keras.backend.clear_session(); gc.collect()
    return out


def ref_eps(name, rep):
    return json.loads((rf.OUT / 'TON_IoT' / name / f'seed{rep}.json').read_text())['dp']['epsilon_max']


def best_sgdm_lr():
    sc = {}
    for lr in SGDM_LRS:
        fs = sorted((OUT / 'TON_IoT' / f'dp_sgdm_lr{lr:g}').glob('seed*.json'))
        if fs:
            sc[lr] = float(np.mean([json.loads(p.read_text())['val_f1_selected'] for p in fs]))
    if not sc:
        raise SystemExit('dp_full needs the dp_short SGD-momentum results first')
    return max(sc, key=sc.get)


# ── experiment definitions ──────────────────────────────────────────────────

def experiments(group, ds, split, rep):
    T = rf.tuned(ds)
    lr = T.get('fedavg', rf.LR)
    if ds == 'N_BaIoT':
        part = dict(partition='device', alpha=None, n_clients=9); tag = 'dev'
    else:
        part = dict(partition='natural', alpha=None, randomized=True); tag = 'hosts'
    if group == 'own':
        return {f'fedavg_{tag}_{split}': ('fedprior', rf.base_cfg(lr=lr, split=split, **part)),
                f'local_{tag}_{split}': ('local', rf.base_cfg(kind='local', lr=lr, split=split, **part))}
    if group == 'prior':
        name = 'fedavg_dev_prior' if ds == 'N_BaIoT' else 'fedavg_natr_prior'
        return {name: ('fedprior', rf.base_cfg(lr=lr, split='orig', **part))}
    dp = dict(algo='dpsgd', alpha=0.5, clip=rf.CLIP)
    if group == 'dp_short':
        sh = dict(rounds=rf.DP_ROUNDS, local_epochs=rf.DP_LOCAL_EPOCHS)
        E = {f'dp_sgdm_lr{v:g}': ('dp', rf.base_cfg(**dp, **sh, lr=v, opt='sgdm', batch=rf.BATCH, sigma=1.0)) for v in SGDM_LRS}
        E[f'dp_b{BIG_B}'] = ('dp', rf.base_cfg(**dp, **sh, lr=lr, opt='adam', batch=BIG_B, sigma=None,
                                               target_eps=ref_eps('dp_sigma1', rep), eps_ref='dp_sigma1'))
        return E
    if group == 'dp_full':
        v = best_sgdm_lr()
        return {'dpfull_sgdm': ('dp', rf.base_cfg(**dp, lr=v, opt='sgdm', batch=rf.BATCH, sigma=rf.DP_FULL_SIGMA)),
                f'dpfull_b{BIG_B}': ('dp', rf.base_cfg(**dp, lr=lr, opt='adam', batch=BIG_B, sigma=None,
                                                       target_eps=ref_eps('dp_full_sigma1.8', rep), eps_ref='dp_full_sigma1.8'))}
    raise ValueError(group)


def load(ds, split):
    if split == 'orig':
        return rf.load_data(ds, meta=ds == 'TON_IoT')
    return load_ton_split(split) if ds == 'TON_IoT' else load_nbaiot_split(split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', required=True, choices=['check', 'prep_nbaiot', 'own', 'prior', 'dp_short', 'dp_full'])
    ap.add_argument('--dataset', default='TON_IoT', choices=['TON_IoT', 'N_BaIoT'])
    ap.add_argument('--split', default='blocked', choices=['blocked', 'random'])
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--cpu-dp', action='store_true')
    a = ap.parse_args()
    if a.group == 'check':
        print('import ok; tf', tf.__version__, '; GPUs', tf.config.list_physical_devices('GPU'))
        print('tuned lr', {d: rf.tuned(d).get('fedavg') for d in ('TON_IoT', 'N_BaIoT')})
        print('dp ref eps', [round(ref_eps('dp_sigma1', r), 3) for r in (0, 1, 2)],
              [round(ref_eps('dp_full_sigma1.8', r), 3) for r in (0, 1, 2)])
        print('raw N-BaIoT files', sum(1 for _ in __import__('experiments.paper4_federated.prepare_nbaiot', fromlist=['files']).files()))
        return
    if a.group == 'prep_nbaiot':
        if all((ROOT / 'dataset4' / f'nbaiot_{m}.npz').exists() for m in ('blocked', 'random')):
            print('N-BaIoT blocked/random splits exist'); return
        prep_nbaiot(); return
    ds = 'TON_IoT' if a.group.startswith('dp') else a.dataset
    split = 'orig' if a.group in ('prior', 'dp_short', 'dp_full') else a.split
    D = None
    for rep in [int(s) for s in a.seeds.split(',')]:
        for name, (kind, cfg) in experiments(a.group, ds, split, rep).items():
            path = OUT / ds / name / f'seed{rep}.json'
            if path.exists():
                print(f'  [{name} seed{rep}] cached'); continue
            if D is None:
                D = load(ds, split)
                print(f"===== {ds} ({split}): train {len(D['ytr'])} val {len(D['yva'])} test {len(D['yte'])} "
                      f"features {D['Xtr'].shape[1]}", flush=True)
            print(f'  [{name} seed{rep}] running {cfg}', flush=True)
            t0 = time.time()
            r = run_fed_prior(D, cfg, rep) if kind == 'fedprior' else rf.run_local_only(D, cfg, rep) if kind == 'local' \
                else run_dp(D, cfg, rep)
            r['code_version'] = rf.CODE_VERSION
            r['data'] = dict(split=split, n_train=int(len(D['ytr'])), n_val=int(len(D['yva'])), n_test=int(len(D['yte'])),
                             guard_removed=D.get('guard_removed'))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(r, indent=1))
            sel = r.get('test_f1_selected', r.get('test_f1_selected_mean'))
            print(f'  [{name} seed{rep}] done in {time.time() - t0:.0f}s  test F1 (val-selected) {sel:.4f}', flush=True)


if __name__ == '__main__':
    main()
