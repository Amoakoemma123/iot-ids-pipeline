"""
run_xai_v3.py — Paper 5 v3: answers the independent review of the v2 draft.

New relative to run_xai_v2.py:
  * Three training seeds (42 = baseline models; 43, 44 trained here) for the RF and CNN.
  * Agreement: shared-feature counts, chance-corrected Jaccard (hypergeometric null),
    and field-level rankings (one-hot columns of one source variable summed).
  * Stability (seed 42): LIME at 1,000 vs 5,000 samples, consensus LIME (mean of 20 runs)
    vs SHAP, and SHAP self-agreement (RF: resampled backgrounds; CNN: GradientExplainer
    re-run with new random seeds and backgrounds).
  * Deletion: two replacement strategies (training mean; values of random training
    records), paired SHAP−LIME bootstrap CIs, and a normalised deletion score.
  * Counterfactuals: dependent features are copied jointly from one benign donor
    (DEPENDENCY_GROUPS), candidates far from all benign training traffic are rejected
    (nearest-neighbour support check), and three search modes are compared: random
    groups, SHAP-guided groups, and attacker-controllable groups only. Queries to the
    first feasible evasion are recorded. Evasion = predicted 'normal'.
  * Hardening: counterfactual augmentation vs a control of the same size (interpolated
    training attacks), separate benign donor pools for generation and evaluation, one
    fixed query set (attacks detected by all three models), benign false-positive rate.
  * Shortcut checks: TTL features removed BEFORE feature selection, RF and CNN, plus the
    train→test sttl majority rule.
  * Model size / latency for the 'lightweight' claim.

Usage:  python -m experiments.paper5_xai.run_xai_v3 --dataset UNSW_NB15 --stage train_seeds
        python -m experiments.paper5_xai.run_xai_v3 --dataset all --stage all
Results: results_v3/<ds>/<stage>.json (seed-independent) and results_v3/<ds>/seed<s>/<stage>.json.
Every stage frees its large models before training another one (16 GB RAM).
"""
import argparse, gc, json, os, sys, time, warnings
from math import comb
from pathlib import Path

import numpy as np
import joblib

warnings.filterwarnings('ignore')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.baseline.config import DATASETS as BASE_DATASETS, CNN_TRAIN, PREPROCESSING, PREPROCESSING_OVERRIDES
from experiments.paper5_xai.run_xai_v2 import (boot_ci, jaccard, topk, spearman, shap_rf,
                                               lime_explainer, lime_attr, stratified_idx, save)

OUT_ROOT = ROOT / 'experiments' / 'paper5_xai' / 'results_v3'
SEED_MODELS = ROOT / 'experiments' / 'paper5_xai' / 'models_seeds'
BASE_RES = ROOT / 'experiments' / 'baseline' / 'results'
SEEDS = (42, 43, 44)

N_LOCAL = 30        # per class: agreement + deletion
N_STAB = 10         # per class: stability (seed 42)
N_RUNS = 20         # LIME repeats
N_SHAP_RUNS = 5     # SHAP repeats for SHAP self-agreement
N_CF = 30           # per attack class: counterfactual queries
CF_MAX_K = 4        # max number of dependency groups changed
CF_TRIES = 200      # candidates per k
N_HARD_GEN = 200    # per attack class: training attacks used to generate augmentation
N_BOOT = 2000
DEL_K = (1, 3, 5)

# Features that must change together (copied from the same benign donor).
DEPENDENCY_GROUPS = {
    'TON_IoT': {
        'protocol_state': ('proto_', 'service_', 'conn_state_', 'dns_'),
        'src_volume': ('src_bytes', 'src_pkts', 'src_ip_bytes'),
        'dst_volume': ('dst_bytes', 'dst_pkts', 'dst_ip_bytes'),
    },
    'UNSW_NB15': {
        'ttl': ('sttl', 'dttl', 'ct_state_ttl'),
        'src_volume': ('sbytes', 'spkts', 'smean', 'sload', 'sinpkt'),
        'dst_volume': ('dbytes', 'dpkts', 'dmean', 'dload', 'dinpkt'),
        'tcp_timing': ('ackdat', 'synack', 'tcprtt'),
        'conn_history': ('ct_',),
    },
}
# Fields an attacker can set from the attacking host (others depend on the victim or
# on traffic history), copied jointly per group from one benign donor. UNSW: only the
# sender's own TTL (sttl); ct_state_ttl and dttl depend on the responder and are left
# unchanged (v3.1; v3.0 used the whole 'ttl' dependency group, a generous reading).
ATTACKER_SPEC = {'TON_IoT': {'src_port': ('src_port',),
                             'src_volume': ('src_bytes', 'src_pkts', 'src_ip_bytes'),
                             'duration': ('duration',)},
                 'UNSW_NB15': {'sttl': ('sttl',),
                               'src_volume': ('sbytes', 'spkts', 'smean', 'sload', 'sinpkt')}}
ATTACKER_GROUPS = {ds: tuple(spec) for ds, spec in ATTACKER_SPEC.items()}
TTL_RAW = ('sttl', 'dttl', 'ct_state_ttl')


# ── helpers ───────────────────────────────────────────────────────────────────

def expected_jaccard(p, k):
    """E[|A∩B|/|A∪B|] for two independent uniform random k-subsets of p items."""
    tot = 0.0
    for i in range(max(0, 2 * k - p), k + 1):
        prob = comb(k, i) * comb(p - k, k - i) / comb(p, k)
        tot += prob * i / (2 * k - i)
    return tot


def paired_ci(a, b, n=N_BOOT, seed=42):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return dict(mean=float(d.mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)), n=int(len(d)))


def onehot_fields(feat, X_tr):
    """One-hot columns of one source variable form one field (for field-level rankings)."""
    groups = {}
    for i, f in enumerate(feat):
        vals = np.unique(X_tr[:, i])
        is_bin = len(vals) <= 2 and set(np.round(vals, 6)) <= {0.0, 1.0}
        key = f.rsplit('_', 1)[0] if (is_bin and '_' in f) else f
        groups.setdefault(key, []).append(i)
    return list(groups.items())


def dependency_groups(ds, feat, X_tr):
    """Dependency groups for counterfactuals: DEPENDENCY_GROUPS patterns first, then
    one-hot fields, then singletons."""
    spec = DEPENDENCY_GROUPS[ds]
    taken, groups = set(), {}
    for g, pats in spec.items():
        idx = [i for i, f in enumerate(feat) if i not in taken and
               any(f == p or (p.endswith('_') and f.startswith(p)) for p in pats)]
        if idx:
            groups[g] = idx; taken |= set(idx)
    for name, idx in onehot_fields(feat, X_tr):
        idx = [i for i in idx if i not in taken]
        if idx:
            groups[name] = idx; taken |= set(idx)
    return list(groups.items())


def attacker_groups(ds, feat):
    """Groups for the attacker-controllable threat model (ATTACKER_SPEC)."""
    out = []
    for g, pats in ATTACKER_SPEC[ds].items():
        idx = [i for i, f in enumerate(feat) if f in pats]
        if idx:
            out.append((g, idx))
    return out


def field_attr(att, fields):
    return np.array([np.abs(att[idx]).sum() for _, idx in fields])


def shap_cnn(cnn, X_bg, X, rseed, nsamples=200):
    import shap
    ex = shap.GradientExplainer(cnn, np.asarray(X_bg, np.float32)[..., None])
    sv = ex.shap_values(np.asarray(X, np.float32)[..., None], rseed=rseed, nsamples=nsamples)
    if isinstance(sv, list):
        sv = np.stack([s[..., 0] if s.ndim == 3 else s for s in sv], axis=-1)
    elif sv.ndim == 4:
        sv = sv[:, :, 0, :]
    return sv


# ── data / models ─────────────────────────────────────────────────────────────

_RAW = {}


def load_data(ds):
    if ds in _RAW:
        return _RAW[ds]
    from src.data.loader import load_dataset
    pipe = joblib.load(BASE_RES / ds / 'pipeline.pkl')
    csv, label_col = BASE_DATASETS[ds]
    raw = load_dataset(csv, label_col=label_col, seed=42)
    X_tr, _ = pipe.transform(raw['X_train'])
    X_va, _ = pipe.transform(raw['X_val'])
    X_te, _ = pipe.transform(raw['X_test'])
    X_tr_sm, y_tr_sm = pipe.transform(raw['X_train'], apply_smote=True, y=raw['y_train'])
    classes = list(raw['class_names'])
    D = dict(ds=ds, pipe=pipe, feat=list(pipe.selected_features_), classes=classes, raw=raw,
             X_tr=X_tr, y_tr=raw['y_train'], X_va=X_va, y_va=raw['y_val'], X_te=X_te, y_te=raw['y_test'],
             X_tr_sm=X_tr_sm, y_tr_sm=y_tr_sm, normal=classes.index('normal'))
    _RAW[ds] = D
    return D


def model_paths(ds, seed):
    if seed == 42:
        return BASE_RES / 'models' / ds / 'model_rf.pkl', BASE_RES / 'models' / ds / 'model_cnn.h5'
    return SEED_MODELS / ds / f'rf_s{seed}.pkl', SEED_MODELS / ds / f'cnn_s{seed}.h5'


def load_rf(ds, seed):
    return joblib.load(model_paths(ds, seed)[0])


def load_cnn(ds, seed):
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], 'GPU')
    except RuntimeError:   # TF already initialised in this process (e.g. by train_seeds)
        pass
    return tf.keras.models.load_model(model_paths(ds, seed)[1], compile=False)


def cnn_fn(cnn):
    return lambda X: cnn.predict(np.asarray(X, np.float32)[..., None], verbose=0, batch_size=4096)


# ── stage: extra training seeds ───────────────────────────────────────────────

def stage_train_seeds(D):
    """RF and CNN with seeds 43 and 44 on the same pipeline/data as the seed-42 baseline."""
    from src.models import build_rf, build_cnn
    from src.training import Trainer
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import f1_score
    import tensorflow as tf
    out = {}
    d = SEED_MODELS / D['ds']; d.mkdir(parents=True, exist_ok=True)
    for s in SEEDS[1:]:
        rfp, cnnp = model_paths(D['ds'], s)
        if not rfp.exists():
            rf = build_rf(seed=s); rf.fit(D['X_tr_sm'], D['y_tr_sm']); joblib.dump(rf, rfp)
            f_rf = f1_score(D['y_te'], rf.predict(D['X_te']), average='macro'); del rf; gc.collect()
        else:
            f_rf = None
        if not cnnp.exists():
            tf.keras.utils.set_random_seed(s)
            cw = compute_class_weight('balanced', classes=np.unique(D['y_tr_sm']), y=D['y_tr_sm'])
            cnn = build_cnn(len(D['feat']), len(D['classes']))
            Trainer(output_dir=d / f'_tmp_s{s}').fit_cnn(cnn, D['X_tr_sm'], D['y_tr_sm'], D['X_va'], D['y_va'],
                                                         name='cnn', class_weights=dict(enumerate(cw)), **CNN_TRAIN)
            cnn.save(str(cnnp))
            f_cnn = f1_score(D['y_te'], cnn.predict(D['X_te'][..., None], verbose=0).argmax(1), average='macro')
            del cnn; tf.keras.backend.clear_session(); gc.collect()
        else:
            f_cnn = None
        out[f'seed{s}'] = dict(rf_macro_f1=f_rf, cnn_macro_f1=f_cnn)
    return out


# ── stage: performance and model size ─────────────────────────────────────────

def stage_performance(D, seed):
    from sklearn.metrics import f1_score, recall_score
    out = dict(n_test=int(len(D['y_te'])), n_features=len(D['feat']), features=D['feat'], classes=D['classes'],
               class_counts_test={c: int((D['y_te'] == i).sum()) for i, c in enumerate(D['classes'])})
    rf = load_rf(D['ds'], seed)
    p = rf.predict(D['X_te'])
    out['rf'] = dict(macro_f1=float(f1_score(D['y_te'], p, average='macro')), accuracy=float((p == D['y_te']).mean()),
                     per_class_f1=dict(zip(D['classes'], f1_score(D['y_te'], p, average=None).tolist())),
                     benign_fpr=float((p[D['y_te'] == D['normal']] != D['normal']).mean()),
                     n_nodes=int(sum(t.tree_.node_count for t in rf.estimators_)),
                     pickle_mb=model_paths(D['ds'], seed)[0].stat().st_size / 2**20)
    t0 = time.time(); rf.predict(D['X_te'][:1000]); out['rf']['ms_per_1000'] = 1000 * (time.time() - t0)
    del rf; gc.collect()
    cnn = load_cnn(D['ds'], seed); fn = cnn_fn(cnn)
    p = fn(D['X_te']).argmax(1)
    out['cnn'] = dict(macro_f1=float(f1_score(D['y_te'], p, average='macro')), accuracy=float((p == D['y_te']).mean()),
                      per_class_f1=dict(zip(D['classes'], f1_score(D['y_te'], p, average=None).tolist())),
                      benign_fpr=float((p[D['y_te'] == D['normal']] != D['normal']).mean()),
                      n_params=int(cnn.count_params()), file_mb=model_paths(D['ds'], seed)[1].stat().st_size / 2**20)
    t0 = time.time(); fn(D['X_te'][:1000]); out['cnn']['ms_per_1000'] = 1000 * (time.time() - t0)
    return out


# ── stage: agreement + deletion (per seed) ────────────────────────────────────

def _deletion(fn, x, att, pred, mean_vec, donors, rng, k):
    p0 = fn(x[None])[0, pred]
    js = topk(att, k)
    z = x.copy(); z[js] = mean_vec[js]
    d_mean = p0 - fn(z[None])[0, pred]
    Z = np.repeat(x[None], 10, 0)
    for r in range(10):
        Z[r, js] = donors[rng.integers(len(donors))][js]
    d_samp = p0 - fn(Z)[:, pred].mean()
    return float(d_mean), float(d_samp), float(p0)


def _deletion_random(fn, x, pred, mean_vec, donors, rng, k, nf):
    p0 = fn(x[None])[0, pred]
    Zm = np.repeat(x[None], 10, 0); Zs = np.repeat(x[None], 10, 0)
    for r in range(10):
        js = rng.choice(nf, k, replace=False)
        Zm[r, js] = mean_vec[js]; Zs[r, js] = donors[rng.integers(len(donors))][js]
    return float(p0 - fn(Zm)[:, pred].mean()), float(p0 - fn(Zs)[:, pred].mean())


def stage_local(D, seed):
    idx = stratified_idx(D['y_te'], N_LOCAL, seed=42)
    X, y = D['X_te'][idx], D['y_te'][idx]
    rng = np.random.default_rng(seed)
    bg = D['X_tr'][rng.choice(len(D['X_tr']), 200, replace=False)]
    mean_vec = D['X_tr'].mean(0)
    donors = D['X_tr'][rng.choice(len(D['X_tr']), 5000, replace=False)]
    nf = len(D['feat'])
    fields = onehot_fields(D['feat'], D['X_tr'])
    expl = lime_explainer(D['X_tr'], D['feat'], D['classes'], seed)
    out = dict(chance_j5=expected_jaccard(nf, 5), chance_j3=expected_jaccard(nf, 3),
               n_fields=len(fields), chance_field_j5=expected_jaccard(len(fields), min(5, len(fields))))
    for m in ('rf', 'cnn'):
        if m == 'rf':
            model = load_rf(D['ds'], seed); fn = model.predict_proba
            t0 = time.time(); sv = shap_rf(model, bg, X); t_shap = time.time() - t0
        else:
            model = load_cnn(D['ds'], seed); fn = cnn_fn(model)
            t0 = time.time(); sv = shap_cnn(model, bg, X, rseed=seed); t_shap = time.time() - t0
        pred = fn(X).argmax(1)
        rows = []
        for i in range(len(X)):
            s = sv[i, :, pred[i]]
            l, r2 = lime_attr(expl, X[i], fn, int(pred[i]), nf)
            fs, fl = field_attr(s, fields), field_attr(l, fields)
            row = dict(test_idx=int(idx[i]), true=int(y[i]), pred=int(pred[i]), correct=bool(y[i] == pred[i]),
                       j5=jaccard(topk(s, 5), topk(l, 5)), shared5=len(set(topk(s, 5)) & set(topk(l, 5))),
                       j3=jaccard(topk(s, 3), topk(l, 3)), rho=spearman(s, l), lime_r2=r2,
                       fj5=jaccard(topk(fs, 5), topk(fl, 5)), fj3=jaccard(topk(fs, 3), topk(fl, 3)),
                       shap_abs=np.abs(s).tolist())
            for k in DEL_K:
                for name, att in (('shap', s), ('lime', l)):
                    dm, ds_, p0 = _deletion(fn, X[i], att, int(pred[i]), mean_vec, donors, rng, k)
                    row[f'del_{name}_{k}'] = dm; row[f'dels_{name}_{k}'] = ds_
                rm, rs = _deletion_random(fn, X[i], int(pred[i]), mean_vec, donors, rng, k, nf)
                row[f'del_rand_{k}'] = rm; row[f'dels_rand_{k}'] = rs
            row['p0'] = float(p0)
            for name in ('shap', 'lime', 'rand'):
                row[f'auc_{name}'] = float(np.mean([row[f'del_{name}_{k}'] for k in DEL_K]) / max(p0, 1e-9))
                row[f'aucs_{name}'] = float(np.mean([row[f'dels_{name}_{k}'] for k in DEL_K]) / max(p0, 1e-9))
            rows.append(row)
        keys = [k for k in rows[0] if k not in ('test_idx', 'true', 'pred', 'correct', 'shap_abs')]
        summ = {k: boot_ci([r[k] for r in rows]) for k in keys}
        paired = {}
        for pre in ('del', 'dels'):
            for k in DEL_K:
                paired[f'{pre}_shap_minus_lime_{k}'] = paired_ci([r[f'{pre}_shap_{k}'] for r in rows], [r[f'{pre}_lime_{k}'] for r in rows])
                paired[f'{pre}_lime_minus_rand_{k}'] = paired_ci([r[f'{pre}_lime_{k}'] for r in rows], [r[f'{pre}_rand_{k}'] for r in rows])
                paired[f'{pre}_shap_minus_rand_{k}'] = paired_ci([r[f'{pre}_shap_{k}'] for r in rows], [r[f'{pre}_rand_{k}'] for r in rows])
        cc = lambda v, e: (v - e) / (1 - e)
        summ['j5_chance_corrected'] = boot_ci([cc(r['j5'], out['chance_j5']) for r in rows])
        summ['fj5_chance_corrected'] = boot_ci([cc(r['fj5'], out['chance_field_j5']) for r in rows])
        summ['share_shared_ge3'] = dict(mean=float(np.mean([r['shared5'] >= 3 for r in rows])))
        summ_correct = {k: boot_ci([r[k] for r in rows if r['correct']]) for k in ('j5', 'del_shap_3', 'del_lime_3', 'del_rand_3')}
        mean_abs = np.mean([r['shap_abs'] for r in rows], axis=0)
        out[m] = dict(summary=summ, paired=paired, summary_correct_only=summ_correct,
                      n_correct=int(sum(r['correct'] for r in rows)),
                      shap_ranking=[D['feat'][j] for j in np.argsort(-mean_abs)],
                      shap_ms_per_instance=1000 * t_shap / len(X), rows=rows)
        del model, fn; gc.collect()
        if m == 'cnn':
            import tensorflow as tf; tf.keras.backend.clear_session()
    return out


# ── stage: deletion with four replacement strategies (v3.1, per seed) ─────────
# mean   = training mean; record = values of random training records (interventional,
# SHAP's own background distribution); gauss = independent Gaussian with training
# mean and s.d. (LIME's sampling distribution); cond = values of the 10 training
# records nearest to x on the features that are NOT replaced (conditional, neither
# explainer's distribution).
DEL_STRATS = ('mean', 'record', 'gauss', 'cond')
N_COND = 10


def _replacements(x, js, ctx, rng, n=10):
    """{strategy: array of replaced copies of x} for one feature set js."""
    mean_vec, sd_vec, donors = ctx
    rest = np.setdiff1d(np.arange(len(x)), js)
    out = {}
    z = x.copy(); z[js] = mean_vec[js]; out['mean'] = z[None]
    Z = np.repeat(x[None], n, 0); Z[:, js] = donors[rng.integers(len(donors), size=n)][:, js]; out['record'] = Z
    Z = np.repeat(x[None], n, 0); Z[:, js] = mean_vec[js] + sd_vec[js] * rng.standard_normal((n, len(js))); out['gauss'] = Z
    d = ((donors[:, rest] - x[rest]) ** 2).sum(1) if len(rest) else rng.random(len(donors))
    nn = np.argpartition(d, N_COND)[:N_COND]
    Z = np.repeat(x[None], N_COND, 0); Z[:, js] = donors[nn][:, js]; out['cond'] = Z
    return out


def stage_deletion(D, seed):
    """Same records, backgrounds and explainers as stage_local (same random streams), so the
    SHAP and LIME attributions are those of stage_local; four replacement strategies."""
    idx = stratified_idx(D['y_te'], N_LOCAL, seed=42)
    X, y = D['X_te'][idx], D['y_te'][idx]
    rng = np.random.default_rng(seed)
    bg = D['X_tr'][rng.choice(len(D['X_tr']), 200, replace=False)]
    mean_vec = D['X_tr'].mean(0); sd_vec = D['X_tr'].std(0)
    donors = D['X_tr'][rng.choice(len(D['X_tr']), 5000, replace=False)]
    ctx = (mean_vec, sd_vec, donors)
    nf = len(D['feat'])
    expl = lime_explainer(D['X_tr'], D['feat'], D['classes'], seed)
    drng = np.random.default_rng(seed + 500)
    out = dict(strategies=DEL_STRATS, n_cond=N_COND)
    for m in ('rf', 'cnn'):
        if m == 'rf':
            model = load_rf(D['ds'], seed); fn = model.predict_proba
            sv = shap_rf(model, bg, X)
        else:
            model = load_cnn(D['ds'], seed); fn = cnn_fn(model)
            sv = shap_cnn(model, bg, X, rseed=seed)
        pred = fn(X).argmax(1)
        rows = []
        for i in range(len(X)):
            s = sv[i, :, pred[i]]
            l, _ = lime_attr(expl, X[i], fn, int(pred[i]), nf)
            row = dict(test_idx=int(idx[i]), true=int(y[i]), pred=int(pred[i]), j5=jaccard(topk(s, 5), topk(l, 5)))
            batches, keys = [], []
            for k in DEL_K:
                for name, att in (('shap', s), ('lime', l)):
                    for st, Z in _replacements(X[i], topk(att, k), ctx, drng).items():
                        batches.append(Z); keys.append((name, st, k, len(Z)))
                for r in range(10):
                    js = drng.choice(nf, k, replace=False)
                    for st, Z in _replacements(X[i], js, ctx, drng, n=1).items():
                        batches.append(Z); keys.append(('rand', st, k, len(Z)))
            P = fn(np.vstack([X[i][None]] + batches))[:, pred[i]]
            p0, pos, acc = P[0], 1, {}
            for (name, st, k, n) in keys:
                acc.setdefault((name, st, k), []).append(p0 - P[pos:pos + n].mean()); pos += n
            for (name, st, k), v in acc.items():
                row[f'{st}_{name}_{k}'] = float(np.mean(v))
            row['p0'] = float(p0)
            rows.append(row)
        summ = {k: boot_ci([r[k] for r in rows]) for k in rows[0] if k not in ('test_idx', 'true', 'pred')}
        paired = {}
        for st in DEL_STRATS:
            for k in DEL_K:
                for a, b in (('shap', 'lime'), ('shap', 'rand'), ('lime', 'rand')):
                    paired[f'{st}_{a}_minus_{b}_{k}'] = paired_ci([r[f'{st}_{a}_{k}'] for r in rows], [r[f'{st}_{b}_{k}'] for r in rows])
        out[m] = dict(summary=summ, paired=paired, rows=rows)
        del model, fn; gc.collect()
        if m == 'cnn':
            import tensorflow as tf; tf.keras.backend.clear_session()
    return out


# ── stage: SHAP sampling budget and background (v3.1, seed 42) ────────────────

def stage_shap_budget(D, seed):
    """CNN: GradientExplainer self-agreement with a fixed background and new random seeds at
    nsamples 200 (default) and 1,000, and with new backgrounds at 1,000. RF: TreeSHAP with three
    backgrounds of 200 and of 1,000 records."""
    idx = stratified_idx(D['y_te'], N_STAB, seed=43)
    X = D['X_te'][idx]

    def pair(S):
        return [float(np.mean([jaccard(topk(S[a][i], 5), topk(S[b][i], 5))
                               for a in range(len(S)) for b in range(a + 1, len(S))])) for i in range(len(X))]

    def bgs(size, n):
        return [D['X_tr'][np.random.default_rng(100 + r).choice(len(D['X_tr']), size, replace=False)] for r in range(n)]

    out = {}
    cnn = load_cnn(D['ds'], seed); fn = cnn_fn(cnn); pred = fn(X).argmax(1)
    pick = lambda sv: np.stack([sv[i, :, pred[i]] for i in range(len(X))])
    fixed = bgs(200, 1)[0]
    res = {}
    for name, runs in (('fixed_bg_ns200', [(fixed, r, 200) for r in range(N_SHAP_RUNS)]),
                       ('fixed_bg_ns1000', [(fixed, r, 1000) for r in range(N_SHAP_RUNS)]),
                       ('new_bg_ns1000', [(b, 100 + r, 1000) for r, b in enumerate(bgs(200, N_SHAP_RUNS))])):
        t0 = time.time()
        S = [pick(shap_cnn(cnn, b, X, rseed=r, nsamples=ns)) for b, r, ns in runs]
        v = pair(S)
        res[name] = dict(self_agreement=boot_ci(v), share_ge_075=float(np.mean(np.array(v) >= 0.75)), seconds=time.time() - t0)
    out['cnn'] = res
    del cnn, fn; gc.collect()
    import tensorflow as tf; tf.keras.backend.clear_session()
    rf = load_rf(D['ds'], seed); pred = rf.predict_proba(X).argmax(1)
    res = {}
    for size in (200, 1000):
        t0 = time.time()
        S = [pick(shap_rf(rf, b, X)) for b in bgs(size, 3)]
        v = pair(S)
        res[f'bg{size}'] = dict(self_agreement=boot_ci(v), share_ge_075=float(np.mean(np.array(v) >= 0.75)), seconds=time.time() - t0)
    out['rf'] = res
    del rf; gc.collect()
    return out


# ── stage: stability, LIME sensitivity, consensus LIME, SHAP stability (seed 42) ─

def stage_stability(D, seed):
    idx = stratified_idx(D['y_te'], N_STAB, seed=43)
    X = D['X_te'][idx]; y = D['y_te'][idx]
    nf = len(D['feat'])
    out = dict(chance_j5=expected_jaccard(nf, 5))
    pair = lambda S: float(np.mean([jaccard(S[a], S[b]) for a in range(len(S)) for b in range(a + 1, len(S))]))
    for m in ('rf', 'cnn'):
        if m == 'rf':
            model = load_rf(D['ds'], seed); fn = model.predict_proba
        else:
            model = load_cnn(D['ds'], seed); fn = cnn_fn(model)
        pred = fn(X).argmax(1)
        # SHAP self-agreement: resampled backgrounds (RF) / new rseed + background (CNN)
        shap_runs = []
        for r in range(N_SHAP_RUNS if m == 'cnn' else 3):
            brng = np.random.default_rng(100 + r)
            bg = D['X_tr'][brng.choice(len(D['X_tr']), 200, replace=False)]
            sv = shap_rf(model, bg, X) if m == 'rf' else shap_cnn(model, bg, X, rseed=100 + r)
            shap_runs.append(np.stack([sv[i, :, pred[i]] for i in range(len(X))]))
        shap_runs = np.stack(shap_runs)             # (runs, n, f)
        rows = []
        for ns, nruns in ((1000, N_RUNS), (5000, 5)):
            expl = lime_explainer(D['X_tr'], D['feat'], D['classes'], seed)
            for i in range(len(X)):
                W = np.stack([lime_attr(expl, X[i], fn, int(pred[i]), nf)[0] if ns == 1000 else
                              _lime_ns(expl, X[i], fn, int(pred[i]), nf, ns) for _ in range(nruns)])
                if ns == 1000:
                    s_ref = shap_runs[0, i]
                    rows.append(dict(test_idx=int(idx[i]), true=int(y[i]),
                                     stab_1000=pair([topk(w, 5) for w in W]),
                                     single_lime_vs_shap=jaccard(topk(W[0], 5), topk(s_ref, 5)),
                                     consensus_lime_vs_shap=jaccard(topk(np.abs(W).mean(0), 5), topk(s_ref, 5)),
                                     shap_self=pair([topk(sr[i], 5) for sr in shap_runs])))
                else:
                    rows[i]['stab_5000'] = pair([topk(w, 5) for w in W])
        keys = ('stab_1000', 'stab_5000', 'single_lime_vs_shap', 'consensus_lime_vs_shap', 'shap_self')
        out[m] = dict(summary={k: boot_ci([r[k] for r in rows]) for k in keys},
                      per_class={c: {k: boot_ci([r[k] for r in rows if r['true'] == ci]) for k in ('stab_1000', 'stab_5000', 'shap_self')}
                                 for ci, c in enumerate(D['classes'])},
                      share_ge_075={k: float(np.mean([r[k] >= 0.75 for r in rows])) for k in ('stab_1000', 'stab_5000', 'shap_self')},
                      paired_consensus_minus_single=paired_ci([r['consensus_lime_vs_shap'] for r in rows], [r['single_lime_vs_shap'] for r in rows]),
                      rows=rows)
        del model, fn; gc.collect()
    return out


def _lime_ns(expl, x, fn, label, nf, ns):
    e = expl.explain_instance(x, fn, labels=[label], num_features=nf, num_samples=ns)
    w = np.zeros(nf)
    for j, v in e.as_map()[label]:
        w[j] = v
    return w


# ── counterfactuals ───────────────────────────────────────────────────────────

class Support:
    """Nearest-neighbour support check against benign training traffic: a candidate is
    feasible if its distance to the nearest benign record is within the 95th percentile
    of benign-to-benign nearest-neighbour distances."""
    def __init__(self, benign, seed=42, n=20000):
        from sklearn.neighbors import NearestNeighbors
        rng = np.random.default_rng(seed)
        B = benign[rng.choice(len(benign), min(n, len(benign)), replace=False)]
        self.nn = NearestNeighbors(n_neighbors=2).fit(B)
        d, _ = self.nn.kneighbors(B[:5000])
        self.tau = float(np.percentile(d[:, 1], 95))

    def ok(self, Z):
        d, _ = self.nn.kneighbors(Z, n_neighbors=1)
        return d[:, 0] <= self.tau


def cf_search(fn, x, target, groups, donors, rng, support, order=None, allowed=None,
              max_k=CF_MAX_K, tries=CF_TRIES):
    """Smallest number of dependency groups whose values, copied jointly from one benign
    donor, make the record benign while staying within benign support.
    order: group indices ranked by explanation (SHAP-guided: always the top-k groups);
    allowed: restrict to these group indices. Returns dict with k, groups, queries, x_cf."""
    G = list(range(len(groups))) if allowed is None else list(allowed)
    queries = 0
    infeasible = 0
    tried = 0
    for k in range(1, min(max_k, len(G)) + 1):
        cand = np.repeat(x[None], tries, 0); chosen = []
        for t in range(tries):
            gs = list(order[:k]) if order is not None else list(rng.choice(G, k, replace=False))
            d = donors[rng.integers(len(donors))]
            for g in gs:
                cols = groups[g][1]; cand[t, cols] = d[cols]
            chosen.append(gs)
        feas = support.ok(cand)
        hit = (fn(cand).argmax(1) == target) & feas
        infeasible += int((~feas).sum()); tried += tries
        if hit.any():
            t = int(np.argmax(hit))
            return dict(k=k, groups=[groups[g][0] for g in chosen[t]], queries=queries + t + 1,
                        x_cf=cand[t], infeasible=infeasible, tried=tried)
        queries += tries
    return dict(k=None, groups=None, queries=queries, x_cf=None, infeasible=infeasible, tried=tried)


def stage_counterfactual(D, seed):
    groups = dependency_groups(D['ds'], D['feat'], D['X_tr'])
    gname = [g for g, _ in groups]
    agroups = attacker_groups(D['ds'], D['feat'])
    benign = D['X_tr'][D['y_tr'] == D['normal']]
    support = Support(benign, seed)
    rng = np.random.default_rng(seed)
    out = dict(groups={g: [D['feat'][j] for j in idx] for g, idx in groups},
               attacker_groups={g: [D['feat'][j] for j in idx] for g, idx in agroups},
               support_tau=support.tau)
    for m in ('rf', 'cnn'):
        model = load_rf(D['ds'], seed) if m == 'rf' else load_cnn(D['ds'], seed)
        fn = model.predict_proba if m == 'rf' else cnn_fn(model)
        pred_te = fn(D['X_te']).argmax(1)
        q_all, q_cls = [], []
        for ci in range(len(D['classes'])):
            if ci == D['normal']:
                continue
            pool = np.where((D['y_te'] == ci) & (pred_te == ci))[0]
            if len(pool):
                q = rng.choice(pool, min(N_CF, len(pool)), replace=False); q_all.extend(q); q_cls.extend([ci] * len(q))
        q_all = np.array(q_all); Xq = D['X_te'][q_all]
        bg = D['X_tr'][rng.choice(len(D['X_tr']), 200, replace=False)]
        sv = shap_rf(model, bg, Xq) if m == 'rf' else shap_cnn(model, bg, Xq, rseed=seed)
        res = {mode: [] for mode in ('random', 'shap', 'attacker')}
        for i, ti in enumerate(q_all):
            ga = np.array([np.abs(sv[i, idx, pred_te[ti]]).sum() for _, idx in groups])
            order = list(np.argsort(-ga))
            for mode in res:
                r = cf_search(fn, D['X_te'][ti], D['normal'], agroups if mode == 'attacker' else groups, benign, rng, support,
                              order=order if mode == 'shap' else None)
                res[mode].append(dict(cls=int(q_cls[i]), k=r['k'], groups=r['groups'], queries=r['queries'], infeasible=r['infeasible'], tried=r['tried']))
        summ = {}
        for mode, rows in res.items():
            ks = np.array([r['k'] if r['k'] is not None else np.nan for r in rows], float)
            qs = np.array([r['queries'] for r in rows if r['k'] is not None], float)
            cnt = {}
            for r in rows:
                for g in r['groups'] or []:
                    cnt[g] = cnt.get(g, 0) + 1
            per_class = {}
            for ci, c in enumerate(D['classes']):
                rr = [r for r in rows if r['cls'] == ci]
                if rr:
                    kk = np.array([r['k'] if r['k'] is not None else np.nan for r in rr], float)
                    per_class[c] = dict(n=len(rr), success=float((~np.isnan(kk)).mean()),
                                        k=boot_ci(kk[~np.isnan(kk)]),
                                        queries_median=float(np.median([r['queries'] for r in rr if r['k'] is not None])) if (~np.isnan(kk)).any() else None)
            summ[mode] = dict(success=float((~np.isnan(ks)).mean()), k=boot_ci(ks[~np.isnan(ks)]),
                              queries_median=float(np.median(qs)) if len(qs) else None,
                              queries_mean=float(qs.mean()) if len(qs) else None,
                              success_within_1=float(np.mean([r['k'] == 1 for r in rows])),
                              group_counts=dict(sorted(cnt.items(), key=lambda kv: -kv[1])),
                              infeasible_share=float(np.sum([r['infeasible'] for r in rows]) /
                                                     max(1, np.sum([r['tried'] for r in rows]))),
                              per_class=per_class, n=len(rows))
        out[m] = summ
        del model, fn; gc.collect()
    return out


def stage_hardening(D, seed, attacker_only=False):
    """RF only. Original vs counterfactual-augmented vs control (same number of
    interpolated training attacks). Generation donors = pool A, evaluation donors = pool B.
    attacker_only: counterfactuals (generation and evaluation) may change only ATTACKER_GROUPS."""
    from sklearn.base import clone
    from sklearn.metrics import f1_score
    groups = dependency_groups(D['ds'], D['feat'], D['X_tr'])
    if attacker_only:
        groups = attacker_groups(D['ds'], D['feat'])
    benign = D['X_tr'][D['y_tr'] == D['normal']]
    rng = np.random.default_rng(seed + 1000)
    perm = rng.permutation(len(benign)); poolA, poolB = benign[perm[::2]], benign[perm[1::2]]
    supA, supB = Support(poolA, seed), Support(poolB, seed + 1)
    attack_classes = [ci for ci in range(len(D['classes'])) if ci != D['normal']]

    rf = load_rf(D['ds'], seed)
    pred_tr = rf.predict(D['X_tr'])
    aug_X, aug_y, aug_k = [], [], []
    for ci in attack_classes:
        pool = np.where((D['y_tr'] == ci) & (pred_tr == ci))[0]
        for ti in rng.choice(pool, min(N_HARD_GEN, len(pool)), replace=False) if len(pool) else []:
            r = cf_search(rf.predict_proba, D['X_tr'][ti], D['normal'], groups, poolA, rng, supA)
            if r['k'] is not None:
                aug_X.append(r['x_cf']); aug_y.append(ci); aug_k.append(r['k'])
    aug_X, aug_y = np.array(aug_X), np.array(aug_y)
    # control: same per-class counts, convex combinations of two training attacks of that class
    ctl_X = []
    for ci in aug_y:
        pool = np.where(D['y_tr'] == ci)[0]
        a, b = D['X_tr'][rng.choice(pool, 2)]
        lam = rng.random()
        ctl_X.append(lam * a + (1 - lam) * b)
    ctl_X = np.array(ctl_X)
    # copy control (v3.1): the same benign-donor perturbation (same groups, same k
    # distribution, same support check) applied to random training attacks of each class,
    # WITHOUT requiring the model to call the result benign
    crng = np.random.default_rng(seed + 3000)
    cpy_X = []
    for ci, k in zip(aug_y, aug_k):
        pool = np.where(D['y_tr'] == ci)[0]
        for _ in range(50):
            z = D['X_tr'][crng.choice(pool)].copy()
            d = poolA[crng.integers(len(poolA))]
            for g in crng.choice(len(groups), min(k, len(groups)), replace=False):
                z[groups[g][1]] = d[groups[g][1]]
            if supA.ok(z[None])[0]:
                break
        cpy_X.append(z)
    cpy_X = np.array(cpy_X)
    # candidate query superset: test attacks detected by the original model (up to 2*N_CF per class)
    p_orig = rf.predict(D['X_te'])
    sup_idx = []
    for ci in attack_classes:
        pool = np.where((D['y_te'] == ci) & (p_orig == ci))[0]
        if len(pool):
            sup_idx.extend(rng.choice(pool, min(2 * N_CF, len(pool)), replace=False))
    sup_idx = np.array(sup_idx)

    def evaluate(model):
        p = model.predict(D['X_te'])
        erng = np.random.default_rng(seed + 2000)       # same evaluation randomness for every model
        succ = {}
        for ti in sup_idx:
            if p[ti] == D['y_te'][ti]:
                succ[int(ti)] = cf_search(model.predict_proba, D['X_te'][ti], D['normal'], groups, poolB, erng, supB)['k'] is not None
        return dict(macro_f1=float(f1_score(D['y_te'], p, average='macro')),
                    benign_fpr=float((p[D['y_te'] == D['normal']] != D['normal']).mean()),
                    attack_recall=float((p[D['y_te'] != D['normal']] != D['normal']).mean()),
                    detected=set(int(t) for t in sup_idx if p[t] == D['y_te'][t]), succ=succ)

    res = {'original': evaluate(rf)}
    template = clone(rf)
    del rf; gc.collect()
    for name, Xa in (('counterfactual', aug_X), ('control', ctl_X), ('control_copy', cpy_X)):
        m = clone(template)
        m.fit(np.vstack([D['X_tr_sm'], Xa]), np.concatenate([D['y_tr_sm'], aug_y]))
        res[name] = evaluate(m)
        del m; gc.collect()
    common = sorted(set.intersection(*(r['detected'] for r in res.values())))
    out = dict(groups=[g[0] for g in groups], n_augmented=int(len(aug_y)), n_query=len(common),
               query_per_class={D['classes'][ci]: int(sum(D['y_te'][t] == ci for t in common)) for ci in attack_classes})
    for name, r in res.items():
        ev = [r['succ'][t] for t in common]
        per_class = {}
        for ci in attack_classes:
            e = [r['succ'][t] for t in common if D['y_te'][t] == ci]
            if e:
                per_class[D['classes'][ci]] = float(np.mean(e))
        out[name] = dict(macro_f1=r['macro_f1'], benign_fpr=r['benign_fpr'], attack_recall=r['attack_recall'],
                         evasion=float(np.mean(ev)) if ev else None, evasion_ci=boot_ci(np.array(ev, float)),
                         evasion_per_class=per_class)
    return out


# ── shortcut check with re-selection (UNSW TTL) ───────────────────────────────

def stage_ttl_reselect(D, seed):
    """Remove TTL features from the raw data BEFORE feature selection, rebuild the
    pipeline, train RF + CNN; also the train→test sttl majority rule."""
    from src.data import PreprocessingPipeline
    from src.models import build_rf, build_cnn
    from src.training import Trainer
    from sklearn.metrics import f1_score
    from sklearn.utils.class_weight import compute_class_weight
    import tensorflow as tf
    raw = D['raw']; rf_names = raw['feature_names']
    ttl_idx = [i for i, f in enumerate(rf_names) if f in TTL_RAW]
    if not ttl_idx:
        return dict(note='no TTL features in raw data')
    out = dict(removed_raw=[rf_names[i] for i in ttl_idx])
    s_i = rf_names.index('sttl')
    tr_s, te_s = raw['X_train'][:, s_i], raw['X_test'][:, s_i]
    yb_tr, yb_te = (raw['y_train'] != D['normal']).astype(int), (raw['y_test'] != D['normal']).astype(int)
    rule_b, rule_c = {}, {}
    for v in np.unique(tr_s):
        msk = tr_s == v
        rule_b[v] = int(yb_tr[msk].mean() >= 0.5)
        rule_c[v] = int(np.bincount(raw['y_train'][msk]).argmax())
    maj_b, maj_c = int(yb_tr.mean() >= 0.5), int(np.bincount(raw['y_train']).argmax())
    pb = np.array([rule_b.get(v, maj_b) for v in te_s]); pc = np.array([rule_c.get(v, maj_c) for v in te_s])
    out['sttl_rule_test'] = dict(binary_accuracy=float((pb == yb_te).mean()), class_accuracy=float((pc == raw['y_test']).mean()),
                                 class_macro_f1=float(f1_score(raw['y_test'], pc, average='macro')),
                                 majority_class_accuracy=float((raw['y_test'] == maj_c).mean()))
    keep = [i for i in range(len(rf_names)) if i not in ttl_idx]
    cfg = {**PREPROCESSING, **PREPROCESSING_OVERRIDES.get(D['ds'], {})}
    pipe = PreprocessingPipeline(mi_threshold=cfg['mi_threshold'], rfecv_subsample=cfg['rfecv_subsample'],
                                 skip_rfecv=cfg.get('skip_rfecv', False), smote_k=cfg['smote_k'],
                                 smote_max_per_class=cfg.get('smote_max_per_class'), mi_subsample=cfg.get('mi_subsample'),
                                 seed=cfg['seed'])
    Xtr, ytr = pipe.fit_transform(raw['X_train'][:, keep], raw['y_train'], feature_names=[rf_names[i] for i in keep], apply_smote=True)
    Xva, _ = pipe.transform(raw['X_val'][:, keep]); Xte, _ = pipe.transform(raw['X_test'][:, keep])
    out['features_selected'] = list(pipe.selected_features_)
    rf = build_rf(seed=seed); rf.fit(Xtr, ytr)
    p = rf.predict(Xte); out['rf_macro_f1'] = float(f1_score(raw['y_test'], p, average='macro'))
    out['rf_per_class'] = dict(zip(D['classes'], f1_score(raw['y_test'], p, average=None).tolist()))
    del rf; gc.collect()
    tf.keras.utils.set_random_seed(seed)
    cw = compute_class_weight('balanced', classes=np.unique(ytr), y=ytr)
    cnn = build_cnn(Xtr.shape[1], len(D['classes']))
    tmp = SEED_MODELS / D['ds'] / '_tmp_ttl'; tmp.mkdir(parents=True, exist_ok=True)
    Trainer(output_dir=tmp).fit_cnn(cnn, Xtr, ytr, Xva, raw['y_val'], name='cnn', class_weights=dict(enumerate(cw)), **CNN_TRAIN)
    p = cnn.predict(Xte[..., None], verbose=0).argmax(1)
    out['cnn_macro_f1'] = float(f1_score(raw['y_test'], p, average='macro'))
    out['cnn_per_class'] = dict(zip(D['classes'], f1_score(raw['y_test'], p, average=None).tolist()))
    return out


SEEDLESS = [('train_seeds', stage_train_seeds)]
PER_SEED = [('performance', stage_performance), ('local', stage_local), ('counterfactual', stage_counterfactual),
            ('hardening', stage_hardening),
            ('hardening_attacker', lambda D, seed: stage_hardening(D, seed, attacker_only=True)),
            ('deletion', stage_deletion)]
SEED42_ONLY = [('stability', stage_stability), ('ttl_reselect', stage_ttl_reselect), ('shap_budget', stage_shap_budget)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='all', choices=['TON_IoT', 'UNSW_NB15', 'all'])
    ap.add_argument('--stage', default='all')
    ap.add_argument('--seeds', default='42,43,44')
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    for ds in (['TON_IoT', 'UNSW_NB15'] if a.dataset == 'all' else [a.dataset]):
        D = load_data(ds)
        print(f'\n===== {ds}: features={len(D["feat"])} test={len(D["y_te"])}')
        jobs = [(n, f, None) for n, f in SEEDLESS] + \
               [(n, f, s) for s in seeds for n, f in PER_SEED] + \
               [(n, f, 42) for n, f in SEED42_ONLY]
        for name, fn, s in jobs:
            if a.stage not in ('all', name):
                continue
            if name == 'ttl_reselect' and ds != 'UNSW_NB15':
                continue
            path = OUT_ROOT / ds / (f'{name}.json' if s is None else f'seed{s}/{name}.json')
            if path.exists():
                print(f'  [{name} {s}] cached'); continue
            print(f'  [{name} seed={s}] running…', flush=True); t0 = time.time()
            res = fn(D) if s is None else fn(D, s)
            res['_seconds'] = time.time() - t0
            save(res, path)
            print(f'  [{name} seed={s}] done in {res["_seconds"]:.0f}s', flush=True)
            gc.collect()


if __name__ == '__main__':
    main()
