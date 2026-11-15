"""Paper 1: every number quoted in Manuscript.md. Writes values.json ({{key}} scalars and {{table:key}} markdown
tables) and points.json (for make_figures.py). Missing values stay visible as ⟦key⟧ placeholders after build.py.

Sources (Papers pipeline/experiments/baseline):
  results_v2       run_p1_v2.py: the v2 grid (5 seeds), networks, port ablation, min–max control; bench_p1_latency.py
  results_p1final  run_p1_final.py / bench_p1_final.py (final review round, queue logs/p1final_queue.sh):
                   GBDTs rolled back to their best internal-validation iteration (gb_*, gb05_*), capped-iteration
                   GBDTs (gbc*), single decision trees and shallow forests, RF/LR refits with validation scores,
                   validation F1 of the seed-0 networks (val_seed0.json; or all seeds after the optional nets_val
                   stage), GBDT port ablation with rollback, GBDTs early-stopped on the validation split (*_ev), and
                   the single-thread latency benchmark (latency.json).
Two kinds of values:
  independent  numbers that do not involve a GBDT, the whole tree grid, model selection or latency (data, networks,
               int8, forests, LR, min–max control, port ablation of forests and networks, the diverged seed-0 model
               of v2): computed now from results_v2.
  final        everything that involves a GBDT, the best tree, size budgets, selection or latency: emitted ONLY when
               every final group is complete (FINAL below), so that no number of the buggy v2 GBDTs (evaluated 20
               iterations past their best point) or of the multi-threaded v2 GBDT latency can reach the text.
Model selection (referee M3): the best configuration and the smallest one within 0.01 of it are chosen on
VALIDATION macro-F1 (mean over seeds when every configuration has validation scores for all seeds; otherwise the
seed-0 validation macro-F1 of every configuration), and their TEST macro-F1 (mean over five seeds) is reported.
Env: P1_FIN=<dir> reads another final-results folder and P1_VALUES_SUFFIX=<s> writes values<s>.json /
points<s>.json (code checks with mock data).
"""
import json
import os
import pickle
import statistics as st
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / 'experiments' / 'baseline'
RES = BASE / 'results_v2'
FIN = Path(os.environ['P1_FIN']) if os.environ.get('P1_FIN') else BASE / 'results_p1final'
SUFFIX = os.environ.get('P1_VALUES_SUFFIX', '')
DATA = json.loads((HERE.parent / 'Paper6' / 'data_stats.json').read_text())      # same data and split as Paper 6
DSS = ('TON_IoT', 'UNSW_NB15', 'N_BaIoT')
HARD = ('TON_IoT', 'UNSW_NB15')
DSN = {'TON_IoT': 'TON-IoT', 'UNSW_NB15': 'UNSW-NB15', 'N_BaIoT': 'N-BaIoT'}
SEEDS = range(5)
RFS = ['rf_t25_d8', 'rf_t25_d12', 'rf_t25_d16', 'rf_t100_d8', 'rf_t100_d12', 'rf_t100_d16', 'rf_t200_dNone', 'rf_v1']
RFLR = RFS + ['lr']
GB_LEAVES = (7, 15, 31, 63)
GB = [f'gb_l{l}' for l in GB_LEAVES] + [f'gb05_l{l}' for l in GB_LEAVES]
CAPPED = [f'gbc{it}_l{l}' for it in (50, 100, 200, 400) for l in (7, 15, 31)]
SMALL = ['dt_d4', 'dt_d6', 'dt_d8', 'dt_d12', 'rf_t10_d4', 'rf_t10_d6', 'rf_t25_d4', 'rf_t25_d6']
NETS = [f'{k}_w{w}' for k in ('cnn', 'mlp') for w in (0.5, 1, 2)]
PORT_CFGS = ('rf_t100_d16', 'rf_v1', 'gb_l31', 'gb05_l31', 'cnn_w1', 'mlp_w1')
BUDGET = 110 * 1024      # superseded (final review round 2): see BUDGET_DS below
BUDGET_DS = {}           # per dataset: size of the largest int8 network including its input table (set with the points)
V, T, VF, TF = {}, {}, {}, {}


def have(base, ds, cfg, n=5):
    return all((base / ds / cfg / f'seed{s}.json').exists() for s in range(n))


def complete(cfgs, dss=DSS):
    return all(have(FIN, ds, c) for ds in dss for c in cfgs)


LATF = FIN / 'latency.json'
_lat = json.loads(LATF.read_text()) if LATF.exists() else {}
ST = dict(gb_rollback=complete(GB), trees_val=complete(RFLR), gb_capped=complete(CAPPED), small_trees=complete(SMALL),
          gb_ports=complete(['gb_l31_noports', 'gb05_l31_noports'], ('TON_IoT',)),
          latency=all(ds in _lat and any(c.startswith('gb') for c in _lat[ds]) for ds in DSS),
          nets_val0=all((FIN / ds / c / 'val_seed0.json').exists() for ds in DSS for c in NETS),
          nets_val=complete(NETS), gb_extval=complete([c + '_ev' for c in GB], HARD))
FINAL = all(ST[k] for k in ('gb_rollback', 'trees_val', 'gb_capped', 'small_trees', 'gb_ports', 'latency')) \
    and (ST['nets_val0'] or ST['nets_val'])
NETS_FIN = FINAL and ST['nets_val']


def pf(p):
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def neg(x, nd=3):
    return f'{x:+.{nd}f}'.replace('-', '−')


def kb(b):
    return f'{b / 1024:,.0f}' if b >= 10 * 1024 else f'{b / 1024:.1f}'


def rng(xs, f='.2f'):
    a, b = f'{min(xs):{f}}', f'{max(xs):{f}}'
    return a if a == b else f'{a}–{b}'


YTE = BASE.parent / 'paper6_multimodal' / 'results_v4'     # the shared split-0 test labels (yte_split0.npy)


def _pred(ds, cfg, seed):
    for root in (FIN, RES):
        f = root / ds / cfg / f'seed{seed}_pred.npy'
        if f.exists():
            return np.load(f).astype(np.int64)
    raise FileNotFoundError(f'{ds}/{cfg}/seed{seed}_pred.npy')


def _f1cm(c, K):
    cm = c.reshape(K, K); tp = np.diag(cm); d = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return float(np.mean(np.where(d > 0, 2 * tp / np.where(d > 0, d, 1), 0.0)))


def boot_diff(ds, a, b, B=2000):
    """95% percentile interval of mean macro-F1(a) - macro-F1(b): seeds resampled, then test records (as weights)"""
    y = np.load(YTE / ds / 'yte_split0.npy').astype(np.int64)
    K = int(y.max()) + 1
    ca = [y * K + _pred(ds, a, s) for s in range(5)]
    cb = [y * K + _pred(ds, b, s) for s in range(5)]
    rng = np.random.default_rng(0)
    n, out = len(y), []
    for _ in range(B):
        ss = rng.integers(0, 5, 5); w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        out.append(np.mean([_f1cm(np.bincount(ca[j], weights=w, minlength=K * K), K)
                            - _f1cm(np.bincount(cb[j], weights=w, minlength=K * K), K) for j in ss]))
    return tuple(np.percentile(out, [2.5, 97.5]))


def welch(a, b):
    return stats.ttest_ind(a, b, equal_var=False).pvalue


def md(rows):
    out = ['| ' + ' | '.join(rows[0]) + ' |', '|' + '|'.join('---' for _ in rows[0]) + '|']
    return '\n'.join(out + ['| ' + ' | '.join(r) + ' |' for r in rows[1:]])


def label(cfg, q=False):
    if cfg == 'lr':
        return 'LR'
    if cfg == 'rf_v1':
        return 'RF, earlier configuration (200 trees, depth 20)'
    if cfg.startswith('rf_'):
        _, t, d = cfg.split('_')
        return f'RF, {t[1:]} trees, depth {"unlimited" if d == "dNone" else d[1:]}'
    if cfg.startswith('dt_'):
        return f'Decision tree, depth {cfg[4:]}'
    if cfg.startswith('gbc'):
        it, l = cfg[3:].split('_l')
        return f'GBDT, {l} leaves, ≤ {it} iterations'
    if cfg.startswith('gb'):
        return f'GBDT, {cfg.split("_l")[1]} leaves, lr {"0.05" if cfg.startswith("gb05") else "0.1"}'
    k, w = cfg.split('_w')
    return f'{k.upper()} width {w}' + (' (int8)' if q else '')


def plabel(cfg, q=False):
    """configuration name for running text (no inner parentheses, so that '(size, macro-F1)' can follow)"""
    if cfg == 'lr':
        return 'LR'
    if cfg == 'rf_v1':
        return 'the earlier forest configuration'
    if cfg.startswith('rf_'):
        _, t, d = cfg.split('_')
        return f'RF with {t[1:]} trees of ' + ('unlimited depth' if d == 'dNone' else f'depth {d[1:]}')
    if cfg.startswith('dt_'):
        return f'a decision tree of depth {cfg[4:]}'
    if cfg.startswith('gbc'):
        it, l = cfg[3:].split('_l')
        return f'GBDT with {l} leaves and at most {it} iterations'
    if cfg.startswith('gb'):
        return f'GBDT with {cfg.split("_l")[1]} leaves at lr {"0.05" if cfg.startswith("gb05") else "0.1"}'
    k, w = cfg.split('_w')
    return ('int8 ' if q else '') + f'{k.upper()} of width {w}'


def desc(p, size_key='size', f_key='f1', extra=''):
    return f"{plabel(p['cfg'], p['q'])} ({kb(p[size_key])} KB{extra}, {p[f_key]:.3f})"


def src(cfg):
    base = cfg.replace('_noports', '')
    if FINAL and (base in GB + CAPPED + SMALL + RFLR and (cfg == base or base in GB)):
        return FIN
    if NETS_FIN and cfg in NETS:
        return FIN
    return RES


def runs(ds, cfg, base=None):
    base = base or src(cfg)
    if not have(base, ds, cfg):
        return None
    return [json.loads((base / ds / cfg / f'seed{s}.json').read_text()) for s in SEEDS]


def mean(R, k):
    return st.mean(r[k] for r in R)


# ── data (independent) ───────────────────────────────────────────────────────
for ds in DSS:
    V[f'{ds}_n_feat'] = str(len(DATA[ds]['features']))
    c = RES / ds / 'clean_split0.npy'
    if c.exists():
        V[f'{ds}_dup_sel_pct'] = f'{100 * (1 - np.load(c).mean()):.1f}%'
V['TON_mitm_test'] = str(DATA['TON_IoT']['test']['mitm'])
V['UNSW_worms_test'] = str(DATA['UNSW_NB15']['test']['worms'])
rows = [['Dataset', 'Classes', 'Selected features', 'Training (after oversampling)', 'Validation', 'Test']]
for ds in DSS:
    d = DATA[ds]
    rows.append([DSN[ds], str(len(d['classes'])), str(len(d['features'])),
                 f"{d['n_train']:,} ({d['n_fit']:,})" if d['n_fit'] != d['n_train'] else f"{d['n_train']:,} (none)",
                 f"{d['n_val']:,}", f"{d['n_test']:,}"])
T['data'] = md(rows)

# ── points on the accuracy-footprint plane ────────────────────────────────────
# Deployable size includes the input transform the model needs. The networks and LR read quantile-transformed features:
# the transform is a table of N_QUANTILES quantiles per feature (float32), which is counted with them. The trees read
# min-max scaled features, whose scaling folds into the split thresholds, so they need no table. size_w = model only.
N_QUANTILES = 1000
QT_BYTES = {}
for _ds in DSS:
    _r = runs(_ds, 'cnn_w1')
    if _r:
        QT_BYTES[_ds] = N_QUANTILES * len(_r[0]['cfg']['features']) * 4
        V[f'{_ds}_qt_kb'] = kb(QT_BYTES[_ds])
        V[f'{_ds}_n_features'] = str(len(_r[0]['cfg']['features']))
V['n_quantiles'] = f'{N_QUANTILES:,}'
TREE_CFGS = RFS + GB + ((CAPPED + SMALL) if FINAL else []) + ['lr']
points = {}
for ds in DSS:
    pts = []
    for cfg in TREE_CFGS:
        R = runs(ds, cfg)
        if not R:
            continue
        f = [r['macro_f1'] for r in R]
        p = dict(cfg=cfg, q=False, name=label(cfg), f1=st.mean(f), sd=st.stdev(f), f=f, size=mean(R, 'compact_bytes'),
                 pickle=mean(R, 'size_bytes'), fpr=mean(R, 'benign_fpr'), det=mean(R, 'attack_detection'),
                 prauc=st.mean(r.get('macro_pr_auc', np.nan) for r in R), src='final' if src(cfg) == FIN else 'v2',
                 clean=[r.get('clean_macro_f1') for r in R])
        if all('val_macro_f1' in r for r in R):
            p['vf'] = [r['val_macro_f1'] for r in R]; p['vf1'] = st.mean(p['vf']); p['vf1_0'] = p['vf'][0]
        if cfg.startswith('gb') and all('compact_q_bytes' in r for r in R):
            p['qsize'] = mean(R, 'compact_q_bytes'); p['q16'] = mean(R, 'q16_macro_f1')
            p['q16_drop'] = st.mean(r['macro_f1'] - r['q16_macro_f1'] for r in R)
            p['ntrees'] = [r['n_trees'] for r in R]; p['iters'] = [r['n_iter'] for r in R]
            p['iters_raw'] = [r.get('n_iter_raw', r['n_iter']) for r in R]
        p['size_w'] = p['size']
        if cfg == 'lr':
            p['size'] = p['size_w'] + QT_BYTES[ds]
        pts.append(p)
    for cfg in NETS:
        R = runs(ds, cfg)
        if not R:
            continue
        f = [r['macro_f1'] for r in R]
        fq = [r['int8']['macro_f1'] for r in R]
        pf_ = dict(cfg=cfg, q=False, name=label(cfg), f1=st.mean(f), sd=st.stdev(f), f=f, size=R[0]['size_bytes'],
                   fpr=mean(R, 'benign_fpr'), det=mean(R, 'attack_detection'),
                   prauc=st.mean(r.get('macro_pr_auc', np.nan) for r in R), params=R[0]['params'],
                   src='final' if src(cfg) == FIN else 'v2', clean=[r.get('clean_macro_f1') for r in R])
        pq = dict(cfg=cfg, q=True, name=label(cfg, True), f1=st.mean(fq), sd=st.stdev(fq), f=fq,
                  size=st.mean(r['int8']['size_bytes'] for r in R), fpr=st.mean(r['int8']['benign_fpr'] for r in R),
                  det=st.mean(r['int8']['attack_detection'] for r in R),
                  prauc=st.mean(r['int8'].get('macro_pr_auc') or np.nan for r in R),
                  agree=st.mean(r['int8']['agreement'] for r in R), agree_min=min(r['int8']['agreement'] for r in R),
                  drop=st.mean(a - b for a, b in zip(f, fq)), drops=[a - b for a, b in zip(f, fq)], src=pf_['src'])
        if all('val_macro_f1' in r for r in R):
            pf_['vf'] = [r['val_macro_f1'] for r in R]; pf_['vf1'] = st.mean(pf_['vf']); pf_['vf1_0'] = pf_['vf'][0]
            pq['vf'] = [r['int8']['val_macro_f1'] for r in R]; pq['vf1'] = st.mean(pq['vf']); pq['vf1_0'] = pq['vf'][0]
        else:
            v0 = FIN / ds / cfg / 'val_seed0.json'
            if v0.exists():
                j = json.loads(v0.read_text())
                pf_['vf1_0'], pq['vf1_0'] = j['val_macro_f1'], j['int8']['val_macro_f1']
        for z in (pf_, pq):
            z['size_w'] = z['size']; z['size'] = z['size_w'] + QT_BYTES[ds]
        pts += [pf_, pq]
        if cfg in ('cnn_w1', 'mlp_w1'):
            V[f'{ds}_{cfg}_params'] = f"{R[0]['params']:,}"
    points[ds] = pts
    _q = [z['size'] for z in pts if z.get('q')]
    if _q:
        BUDGET_DS[ds] = max(_q)
        V[f'{ds}_budget_kb'] = kb(BUDGET_DS[ds])

is_net = lambda p: p['cfg'] in NETS
is_tree = lambda p: p['cfg'] not in NETS and p['cfg'] != 'lr'
is_gb = lambda p: p['cfg'].startswith('gb')


def put(k, v, final):
    (VF if final else V)[k] = v


# per-configuration values (GBDT ones are final-only)
for ds in DSS:
    for p in points[ds]:
        k = f"{ds}_{p['cfg']}{'_int8' if p['q'] else ''}"
        fin = is_gb(p) or p['cfg'] in CAPPED + SMALL
        put(k + '_f1', f"{p['f1']:.3f}", fin); put(k + '_kb', kb(p['size']), fin); put(k + '_kb_w', kb(p['size_w']), fin)
        if 'pickle' in p:
            put(k + '_pickle_kb', kb(p['pickle']), fin)

# ── networks and quantisation (independent) ──────────────────────────────────
alld = {'cnn': [], 'mlp': []}
for ds in DSS:
    pts = points[ds]
    nets = [p for p in pts if is_net(p)]
    if not nets:
        continue
    bn = max(nets, key=lambda p: p['f1'])
    V[f'{ds}_best_net'], V[f'{ds}_best_net_f1'], V[f'{ds}_best_net_kb'] = plabel(bn['cfg'], bn['q']), f"{bn['f1']:.3f}", kb(bn['size'])
    nb = max((p for p in nets if p['size'] <= BUDGET_DS[ds] + 1), key=lambda p: p['f1'])
    V[f'{ds}_budget_net'], V[f'{ds}_budget_net_f1'], V[f'{ds}_budget_net_kb'] = plabel(nb['cfg'], nb['q']), f"{nb['f1']:.3f}", kb(nb['size'])
    cnn = next(p for p in pts if p['cfg'] == 'cnn_w1' and not p['q'])
    V[f'{ds}_cnn_w1_f1'], V[f'{ds}_cnn_w1_kb'] = f"{cnn['f1']:.3f}", kb(cnn['size'])
    q = [p for p in pts if p['q']]
    for p in q:
        V[f"{ds}_{p['cfg']}_int8_drop"] = neg(p['drop'])
        V[f"{ds}_{p['cfg']}_int8_agree"] = f"{100 * p['agree']:.1f}%"
    V[f'{ds}_int8_drop_max'] = f"{max(p['drop'] for p in q):.3f}"
    V[f'{ds}_int8_agree_min'] = f"{100 * min(p['agree'] for p in q):.1f}%"
    V[f'{ds}_int8_agree_seed_min'] = f"{100 * min(p['agree_min'] for p in q):.1f}%"
    for fam in ('cnn', 'mlp'):
        qf = [p for p in q if p['cfg'].startswith(fam)]
        V[f'{ds}_{fam}_int8_drop_max'] = f"{max(p['drop'] for p in qf):.3f}"
        V[f'{ds}_{fam}_int8_seed_drop_max'] = f"{max(max(p['drops']) for p in qf):.3f}"
        alld[fam].append((max(p['drop'] for p in qf), max(max(p['drops']) for p in qf)))
    V[f'{ds}_mlp_int8_sd_max'] = f"{max(p['sd'] for p in q if p['cfg'].startswith('mlp')):.3f}"
for fam in ('cnn', 'mlp'):
    if alld[fam]:
        V[f'{fam}_int8_drop_max_all'] = f"{max(a for a, _ in alld[fam]):.3f}"
        V[f'{fam}_int8_seed_drop_max_all'] = f"{max(b for _, b in alld[fam]):.3f}"
red = [pf_['size_w'] / pq['size_w'] for ds in DSS for pf_ in points[ds] if is_net(pf_) and not pf_['q']
       for pq in points[ds] if pq['cfg'] == pf_['cfg'] and pq['q']]
if red:
    V['int8_size_ratio'] = f'{min(red):.1f}–{max(red):.1f}'
for ds in DSS:          # the int8 CNN of width 2 vs its float32 version (noise level; referee minor 2)
    a = next((p for p in points[ds] if p['cfg'] == 'cnn_w2' and p['q']), None)
    if a:
        V[f'{ds}_cnn_w2_int8_gain'] = neg(-a['drop'])

# forests and LR (independent: deterministic refits give identical test results)
rf_ratios = [p['pickle'] / p['size'] for ds in DSS for p in points[ds] if p['cfg'] in RFS]
if rf_ratios:
    V['rf_pickle_ratio'] = f'{min(rf_ratios):.1f}–{max(rf_ratios):.1f}'
_t = {p['cfg']: p for p in points.get('TON_IoT', []) if not p['q']}
if 'rf_v1' in _t and 'rf_t25_d16' in _t:
    V['TON_rfv1_minus_t25d16'] = f"{_t['rf_v1']['f1'] - _t['rf_t25_d16']['f1']:.3f}"

# min–max control (independent)
for ds in HARD:
    A, B = runs(ds, 'cnn_w1_minmax', RES), runs(ds, 'cnn_w1', RES)
    if A and B:
        a, b = [r['macro_f1'] for r in A], [r['macro_f1'] for r in B]
        V[f'{ds}_cnn_minmax_f1'], V[f'{ds}_cnn_minmax_sd'] = f'{st.mean(a):.3f}', f'{st.stdev(a):.3f}'
        V[f'{ds}_cnn_q_minus_minmax'] = neg(st.mean(b) - st.mean(a))
        V[f'{ds}_cnn_minmax_welch_p'] = pf(welch(b, a))

# GBDT before rollback (v2 files as run; describes the bug, so independent of the re-runs)
for ds in DSS:
    lows, n01, n05 = [], 0, 0
    for l in GB_LEAVES:
        A, B = runs(ds, f'gb_l{l}', RES), runs(ds, f'gb05_l{l}', RES)
        if not (A and B):
            continue
        a, b = [r['macro_f1'] for r in A], [r['macro_f1'] for r in B]
        lows.append((min(a), min(b)))
        n01 += sum(x < st.mean(b) - 0.05 for x in a)          # lr 0.1 seed > 0.05 below the lr 0.05 mean
        n05 += sum(x < st.median(b) - 0.05 for x in b)        # lr 0.05 seed > 0.05 below its own median
    if lows:
        V[f'{ds}_gb01_min'] = f'{min(x for x, _ in lows):.3f}'
        V[f'{ds}_gb05_min'] = f'{min(y for _, y in lows):.3f}'
        V[f'{ds}_gb01_collapsed'] = f'{n01} of {5 * len(lows)}'
        V[f'{ds}_gb05_collapsed'] = f'{n05} of {5 * len(lows)}'
_m = RES / 'TON_IoT' / 'gb_l7' / 'model_seed0.pkl'          # the diverged v2 model (refs/code review C1)
if _m.exists():
    g = pickle.loads(_m.read_bytes())
    vs = -np.asarray(g.validation_score_)
    V['TON_gbl7_s0_best_loss'], V['TON_gbl7_s0_best_iter'] = f'{vs.min():.3f}', str(int(vs.argmin()))
    V['TON_gbl7_s0_last_loss'], V['TON_gbl7_s0_last_iter'] = f'{vs[-1]:,.0f}', str(len(vs) - 1)
    V['TON_gbl7_s0_f1'] = f"{json.loads((RES / 'TON_IoT' / 'gb_l7' / 'seed0.json').read_text())['macro_f1']:.3f}"

# port ablation (forests and networks independent; GBDT rows final)
rows = [['Model', 'With ports', 'Without ports', 'Difference', 'Welch p']]
for cfg in PORT_CFGS:
    A, B = runs('TON_IoT', cfg), runs('TON_IoT', f'{cfg}_noports')
    fin = cfg.startswith('gb')
    if not (A and B):
        continue
    a, b = [r['macro_f1'] for r in A], [r['macro_f1'] for r in B]
    d = st.mean(b) - st.mean(a)
    put(f'TON_noports_{cfg}', neg(d), fin)
    put(f'TON_noports_{cfg}_drop', f'{-d:.3f}', fin)
    rows.append([label(cfg), f'{st.mean(a):.3f}', f'{st.mean(b):.3f}', neg(d), pf(welch(a, b))])
    cls = DATA['TON_IoT']['classes']
    pc = {c: st.mean(r['per_class_f1'][c] for r in B) - st.mean(r['per_class_f1'][c] for r in A) for c in cls}
    worst = min(pc, key=pc.get)
    put(f'TON_noports_{cfg}_worst_cls', worst, fin)
    put(f'TON_noports_{cfg}_worst_drop', f'{-pc[worst]:.2f}', fin)
TF['ports'] = md(rows)

# ── FINAL values: GBDTs, the tree grid, budgets, selection ───────────────────
for ds in DSS:
    pts = points[ds]
    nets = [p for p in pts if is_net(p)]
    trees = [p for p in pts if is_tree(p)]
    if not (nets and trees):
        continue
    bn, bt = max(nets, key=lambda p: p['f1']), max(trees, key=lambda p: p['f1'])
    best = max(pts, key=lambda p: p['f1'])
    VF[f'{ds}_best'], VF[f'{ds}_best_f1'], VF[f'{ds}_best_kb'] = plabel(best['cfg'], best['q']), f"{best['f1']:.3f}", kb(best['size'])
    VF[f'{ds}_best_tree'], VF[f'{ds}_best_tree_f1'], VF[f'{ds}_best_tree_kb'] = plabel(bt['cfg']), f"{bt['f1']:.3f}", kb(bt['size'])
    VF[f'{ds}_best_tree_cfg'] = bt['cfg']
    VF[f'{ds}_tree_minus_net'] = neg(bt['f1'] - bn['f1'])
    VF[f'{ds}_tree_net_gap'] = f"{abs(bt['f1'] - bn['f1']):.3f}"
    VF[f'{ds}_tree_net_welch_p'] = pf(welch(bt['f'], bn['f']))
    # tree configurations statistically tied with the best tree (Welch p >= 0.05)
    tied = [p for p in trees if p is bt or welch(p['f'], bt['f']) >= 0.05]
    VF[f'{ds}_top_tied'] = '; '.join(desc(p) for p in sorted(tied, key=lambda p: p['size']))
    VF[f'{ds}_top_tied_n'] = str(len(tied))
    # budget of about 100 KB
    nb = max((p for p in nets if p['size'] <= BUDGET_DS[ds] + 1), key=lambda p: p['f1'])
    tb = [p for p in trees if p['size'] <= BUDGET_DS[ds] + 1]
    if tb:
        t = max(tb, key=lambda p: p['f1'])
        VF[f'{ds}_budget_tree'] = desc(t)
        VF[f'{ds}_budget_tree_name'], VF[f'{ds}_budget_tree_f1'], VF[f'{ds}_budget_tree_kb'] = plabel(t['cfg']), f"{t['f1']:.3f}", kb(t['size'])
        VF[f'{ds}_budget_tree_cfg'] = t['cfg']
        VF[f'{ds}_budget_gap'] = f"{abs(nb['f1'] - t['f1']):.3f}"
        VF[f'{ds}_budget_welch_p'] = pf(welch(nb['f'], t['f']))
        VF[f'{ds}_budget_winner'] = ('tie' if welch(nb['f'], t['f']) >= 0.05      # Welch p >= 0.05: statistically tied
                                     else 'network' if nb['f1'] > t['f1'] else 'tree')
        VF[f'{ds}_budget_net_f1'] = f"{nb['f1']:.3f}"
        VF[f'{ds}_budget_net_minus_tree'] = neg(nb['f1'] - t['f1'])
        win, lose = (nb, t) if nb['f1'] > t['f1'] else (t, nb)
        VF[f'{ds}_budget_best'], VF[f'{ds}_budget_other'] = desc(win), desc(lose)
    gq = [p for p in trees if 'qsize' in p and p['qsize'] <= BUDGET_DS[ds] + 1]
    if gq:
        g = max(gq, key=lambda p: p['q16'])
        VF[f'{ds}_budget_tree_q'] = desc(g, 'qsize', 'q16', ' quantised')
    # smallest tree above the best network
    over = [p for p in trees if p['f1'] > bn['f1']]
    if over:
        o = min(over, key=lambda p: p['size'])
        VF[f'{ds}_tree_over_net'] = desc(o)
        VF[f'{ds}_tree_over_net_kb'] = kb(o['size'])
        VF[f'{ds}_tree_over_net_welch_p'] = pf(welch(o['f'], bn['f']))
    over_sig = [p for p in over if welch(p['f'], bn['f']) < 0.05]
    if over_sig:
        o = min(over_sig, key=lambda p: p['size'])
        VF[f'{ds}_tree_over_net_sig'] = desc(o)
        VF[f'{ds}_tree_over_net_sig_kb'] = kb(o['size'])
        VF[f'{ds}_tree_over_net_sig_welch_p'] = pf(welch(o['f'], bn['f']))
    # final review round 2 (referee major 1): seeds x test-records bootstrap of the tree-minus-network differences
    # behind the budget claims (network side float32: int8 test predictions were not stored)
    if ds in HARD:
        netf_ = next(p for p in nets if p['cfg'] == bn['cfg'] and not p['q'])
        for tag, tp in (('best', bt), ('over', over and min(over, key=lambda p: p['size'])),
                        ('oversig', over_sig and min(over_sig, key=lambda p: p['size']))):
            if tp:
                lo, hi = boot_diff(ds, tp['cfg'], netf_['cfg'])
                VF[f'{ds}_boot_{tag}_lo'], VF[f'{ds}_boot_{tag}_hi'] = neg(lo), neg(hi)
                VF[f'{ds}_boot_{tag}_excl0'] = 'yes' if lo > 0 or hi < 0 else 'no'
    # clean test records (no selected-feature match in training); network side: float32 (int8 not stored)
    cl = lambda p: st.mean(p['clean']) if p.get('clean') and None not in p['clean'] else None
    netf = next(p for p in nets if p['cfg'] == bn['cfg'] and not p['q'])
    a, b = cl(bt), cl(netf)
    if a is not None and b is not None:
        VF[f'{ds}_clean_tree_f1'], VF[f'{ds}_clean_net_f1'] = f'{a:.3f}', f'{b:.3f}'
        VF[f'{ds}_clean_gap'] = f'{a - b:.3f}'
        VF[f'{ds}_clean_tree_minus_net'] = neg(a - b)
        ct = max((p for p in trees if cl(p) is not None), key=cl)
        VF[f'{ds}_clean_top_tree'] = f"{plabel(ct['cfg'])} ({cl(ct):.3f})"
        VF[f'{ds}_clean_top_is_best'] = str(ct is bt)
    # model selection on validation macro-F1
    basis = 'mean' if all('vf1' in p for p in pts) else ('seed0' if all('vf1_0' in p for p in pts) else None)
    if basis:
        vk = 'vf1' if basis == 'mean' else 'vf1_0'
        sb = max(pts, key=lambda p: p[vk])
        within = [p for p in pts if p[vk] >= sb[vk] - 0.01]
        ss = min(within, key=lambda p: p['size'])
        VF[f'{ds}_sel_best'] = f"{plabel(sb['cfg'], sb['q'])} ({kb(sb['size'])} KB; validation {sb[vk]:.3f}, test {sb['f1']:.3f})"
        VF[f'{ds}_sel_best_name'], VF[f'{ds}_sel_best_f1'] = plabel(sb['cfg'], sb['q']), f"{sb['f1']:.3f}"
        VF[f'{ds}_sel_smallest'] = f"{plabel(ss['cfg'], ss['q'])} ({kb(ss['size'])} KB; validation {ss[vk]:.3f}, test {ss['f1']:.3f})"
        VF[f'{ds}_sel_smallest_name'], VF[f'{ds}_sel_smallest_kb'] = plabel(ss['cfg'], ss['q']), kb(ss['size'])
        VF[f'{ds}_sel_smallest_cfg'], VF[f'{ds}_sel_best_cfg'] = ss['cfg'], sb['cfg']
        VF[f'{ds}_sel_smallest_f1'] = f"{ss['f1']:.3f}"
        VF[f'{ds}_sel_smallest_family'] = 'network' if is_net(ss) else ('LR' if ss['cfg'] == 'lr' else 'tree model')
        VF[f'{ds}_sel_margin'] = f"{ss[vk] - (sb[vk] - 0.01):.3f}"
        VF[f'{ds}_sel_share'] = f"{sum(x >= y - 0.01 for x, y in zip(ss['f'], sb['f']))} of 5"
        VF['sel_basis'] = ('the mean validation macro-F1 over the five seeds' if basis == 'mean' else
                           'the validation macro-F1 of the seed-0 model of every configuration (validation scores of the other network seeds were not recorded)')
# GBDTs after rollback
nrb = ntot = ndiv = 0
gb_rows = [['Dataset', 'Leaves', 'lr 0.1 as run: mean (min seed)', 'lr 0.1 rolled back: mean (min seed)',
            'lr 0.05 rolled back: mean (min seed)', 'Iterations kept (lr 0.1; lr 0.05)']]
q_ratio, q_drop = [], []
for ds in DSS:
    lo1, lo5, ncoll, cap = [], [], 0, 0
    for l in GB_LEAVES:
        A0, A, B = runs(ds, f'gb_l{l}', RES), runs(ds, f'gb_l{l}'), runs(ds, f'gb05_l{l}')
        if not (A0 and A and B and 'best_iter' in A[0]):
            continue
        a0, a, b = ([r['macro_f1'] for r in X] for X in (A0, A, B))
        lo1.append(min(a)); lo5.append(min(b))
        ncoll += sum(x < st.mean(b) - 0.05 for x in a)
        cap += sum(r['n_iter_raw'] >= 1000 for r in B)
        for r in A + B:
            ntot += 1; nrb += bool(r.get('rolled_back'))
            if r.get('val_score_last') is not None and -r['val_score_last'] > 10 * -r['val_score_best']:
                ndiv += 1
        gb_rows.append([DSN[ds], str(l), f'{st.mean(a0):.3f} ({min(a0):.3f})', f'{st.mean(a):.3f} ({min(a):.3f})',
                        f'{st.mean(b):.3f} ({min(b):.3f})',
                        f"{rng([r['n_iter'] for r in A], 'd')}; {rng([r['n_iter'] for r in B], 'd')}"])
    if lo1:
        VF[f'{ds}_gb_rb_min01'], VF[f'{ds}_gb_rb_min05'] = f'{min(lo1):.3f}', f'{min(lo5):.3f}'
        VF[f'{ds}_gb_rb_collapsed'] = f'{ncoll} of {5 * len(lo1)}'
        VF[f'{ds}_gb05_cap_n'] = f'{cap} of {5 * len(lo5)}'
    for p in points[ds]:
        if 'qsize' in p and p['cfg'] in GB + CAPPED:
            q_ratio.append((ds, p['size'] / p['qsize'])); q_drop.append(p['q16_drop'])
if ntot:
    VF['gb_rolled_n'], VF['gb_diverged_n'], VF['gb_runs_n'] = str(nrb), str(ndiv), str(ntot)
TF['gb_lr'] = md(gb_rows)
if q_ratio:
    VF['gb_q_ratio'] = rng([r for d, r in q_ratio if d in HARD], '.1f')
    _nb = [r for d, r in q_ratio if d not in HARD]
    VF['gb_q_nb_larger'] = f"{sum(r < 1 for r in _nb)} of {len(_nb)}"
    VF['gb_q_nb_grow_max'] = f"{max(1 / r for r in _nb):.1f}" if _nb else ''
    VF['gb_q16_drop_max'] = f'{max(abs(x) for x in q_drop):.3f}'
for ds in DSS:
    bt = max((p for p in points[ds] if is_tree(p)), key=lambda p: p['f1'], default=None)
    if bt and 'ntrees' in bt:
        VF[f'{ds}_best_tree_ntrees'] = f"{bt['ntrees'][0]:,}"
        VF[f'{ds}_best_tree_ntrees_range'] = rng([float(x) for x in bt['ntrees']], ',.0f')
        VF[f'{ds}_best_tree_iters_range'] = rng([float(x) for x in bt['iters']], ',.0f')
        VF[f'{ds}_best_tree_cap'] = str(sum(x >= 1000 for x in bt['iters_raw']))
    elif bt:
        VF[f'{ds}_best_tree_ntrees'] = '–'
_t = {p['cfg']: p for p in points.get('TON_IoT', []) if not p['q']}
_bt = max((p for p in points.get('TON_IoT', []) if is_tree(p)), key=lambda p: p['f1'], default=None)
if _bt and 'rf_v1' in _t:
    VF['TON_rfv1_over_best_tree_size'] = f"{_t['rf_v1']['size'] / _bt['size']:.0f}"
    VF['TON_besttree_rfv1_welch_p'] = pf(welch(_bt['f'], _t['rf_v1']['f']))
VF['n_configs'] = str(len(points.get('TON_IoT', [])))
VF['n_tree_cfgs'] = str(sum(not is_net(p) for p in points.get('TON_IoT', [])))
_grid = [p for p in points.get('N_BaIoT', []) if p['cfg'] not in CAPPED + SMALL]       # the main (v2) grid of 29 points
VF['N_BaIoT_n_998'] = str(sum(round(p['f1'], 3) >= 0.998 for p in _grid))
VF['n_grid_configs'] = str(len(_grid))
gr = [p['pickle'] / p['size'] for ds in DSS for p in points[ds] if is_gb(p)]
if gr:
    VF['gb_pickle_ratio'] = rng(gr, '.0f')
# early stopping on the validation split (sensitivity, referee M5)
if ST['gb_extval']:
    for ds in HARD:
        bt = max((p for p in points[ds] if p['cfg'] in GB), key=lambda p: p['f1'])
        E = runs(ds, bt['cfg'] + '_ev', FIN)
        e = [r['macro_f1'] for r in E]
        VF[f'{ds}_ev_best_f1'] = f'{st.mean(e):.3f}'
        VF[f'{ds}_ev_minus_internal'] = neg(st.mean(e) - bt['f1'])
        VF[f'{ds}_ev_welch_p'] = pf(welch(e, bt['f']))
        allev = [r['macro_f1'] for c in GB for r in runs(ds, c + '_ev', FIN)]
        VF[f'{ds}_ev_min'] = f'{min(allev):.3f}'

# main table (Table 2): a readable subset plus the configurations named in the text
KEY = ['lr', 'rf_v1', 'rf_t200_dNone', 'rf_t25_d12', 'rf_t100_d16', 'gb_l15', 'gb05_l15', 'gb_l63', 'gb05_l63', 'cnn_w1', 'mlp_w1']
for ds in DSS:
    for k in (f'{ds}_budget_tree_cfg', f'{ds}_sel_smallest_cfg', f'{ds}_sel_best_cfg'):
        c = VF.get(k)
        if c and c not in KEY:
            KEY.insert(-2, c)
rows = [['Model', 'Encoding'] + [f'{DSN[ds]} {m}' for ds in DSS for m in ('macro-F1', 'size (KB)')]]
for cfg in KEY:
    for q in ((False, True) if cfg in NETS else (False,)):
        cells = []
        for ds in DSS:
            p = next((x for x in points.get(ds, []) if x['cfg'] == cfg and x['q'] == q), None)
            cells += ['–', '–'] if p is None else [f"{p['f1']:.3f} ± {p['sd']:.3f}", kb(p['size'])]
        if any(c != '–' for c in cells):
            rows.append([label(cfg, q), 'compact' if cfg not in NETS else ('int8' if q else 'float32')] + cells)
TF['main'] = md(rows)
full = [['Dataset', 'Model', 'Macro-F1 (mean ± SD)', 'Validation macro-F1', 'Benign FPR', 'Attack detection', 'Macro PR-AUC',
         'Compact size (KB)', 'Pickled size (KB)']]
for ds in DSS:
    for p in sorted(points.get(ds, []), key=lambda x: x['size']):
        vtxt = f"{p['vf1']:.3f}" if 'vf1' in p else (f"{p['vf1_0']:.3f}*" if 'vf1_0' in p else '–')
        full.append([DSN[ds], p['name'], f"{p['f1']:.3f} ± {p['sd']:.3f}", vtxt, f"{p['fpr']:.3f}", f"{p['det']:.3f}",
                     f"{p['prauc']:.3f}", kb(p['size']), kb(p['pickle']) if 'pickle' in p else '–'])
TF['grid'] = md(full)

# ── latency (final benchmark, one thread) ─────────────────────────────────────
if ST['latency']:
    L = _lat
    VF['lat_cpu'] = L.get('cpu', '')
    ms = lambda ds, p: L[ds].get(p['cfg'], {}).get('int8_single_ms' if p['q'] else 'single_ms')
    TAB = RFLR[:-1] + GB + ['lr'] + NETS
    rows = [['Model'] + [f'{DSN[ds]} {m}' for ds in DSS for m in ('ms/record', 'ms/1,000')]]
    for cfg in TAB:
        if not all(cfg in L.get(ds, {}) for ds in DSS):
            continue
        rows.append([label(cfg)] + [x for ds in DSS for x in (f"{L[ds][cfg]['single_ms']:.3f}", f"{L[ds][cfg]['batch1000_ms']:.1f}")])
        if cfg in NETS:
            rows.append([label(cfg, True)] + [x for ds in DSS for x in (f"{L[ds][cfg]['int8_single_ms']:.3f}",
                                                                      f"{L[ds][cfg]['int8_batch1000_ms']:.1f}")])
    TF['latency'] = md(rows)
    rows = [['Network', 'Dataset', 'TFLite float32 (ms)', 'int8, float fallback (ms)', 'same, no XNNPACK (ms)',
             'Full-integer int8 (ms)', 'File sizes f32 / int8 / full-int8 (KB)', 'Seed-0 macro-F1 f32 / int8 / full-int8']]
    for cfg in NETS:
        for ds in DSS:
            e = L[ds].get(cfg)
            if not e:
                continue
            fi = lambda k, f='.3f': f"{e[k]:{f}}" if e.get(k) is not None else '–'
            rows.append([label(cfg), DSN[ds], fi('tflite_f32_single_ms'), fi('int8_single_ms'), fi('int8_single_ms_nodelegate'),
                         fi('fullint8_single_ms'),
                         f"{kb(e['tflite_f32_size_bytes'])} / {kb(e['int8_size_bytes'])} / {kb(e['fullint8_size_bytes']) if 'fullint8_size_bytes' in e else '–'}",
                         f"{e.get('float32_macro_f1_seed0', float('nan')):.3f} / {e.get('int8_macro_f1_seed0', float('nan')):.3f} / {fi('fullint8_macro_f1_seed0')}"])
    TF['tflite'] = md(rows)
    for ds in DSS:
        for cfg, e in L[ds].items():
            if isinstance(e, dict) and 'single_ms' in e:
                VF[f'{ds}_{cfg}_ms'] = f"{e['single_ms']:.3f}"
                if 'int8_single_ms' in e:
                    VF[f'{ds}_{cfg}_int8_ms'] = f"{e['int8_single_ms']:.3f}"
    lgbm_sp = []
    FPS = {}
    for ds in DSS:
        pts = points[ds]
        nets = [p for p in pts if is_net(p)]
        trees = [p for p in pts if is_tree(p)]
        bn, bt = max(nets, key=lambda p: p['f1']), max(trees, key=lambda p: p['f1'])
        a, b = ms(ds, bt), ms(ds, bn)
        VF[f'{ds}_lat_best_tree_ms'] = f'{a:.0f}' if a >= 10 else f'{a:.1f}'
        VF[f'{ds}_lat_best_net_ms'] = f'{b:.2f}'
        VF[f'{ds}_lat_tree_net_ratio'] = f'{a / b:.0f}'
        e = L[ds].get(bt['cfg'], {})
        if e.get('single_ms_seeds'):
            VF[f'{ds}_lat_best_tree_ms_seeds'] = rng(e['single_ms_seeds'], '.1f')
        tied = [p for p in trees if p is bt or welch(p['f'], bt['f']) >= 0.05]
        tl = [ms(ds, p) for p in tied if ms(ds, p) is not None]
        VF[f'{ds}_lat_tied_range'] = rng(tl, '.1f')
        near = [p for p in trees if p['f1'] >= bt['f1'] - 0.005 and ms(ds, p) is not None]
        fast = min(near, key=lambda p: ms(ds, p))
        VF[f'{ds}_lat_near_best_fast'] = f"{plabel(fast['cfg'])} ({fast['f1']:.3f}, {ms(ds, fast):.2f} ms)"
        nb = max((p for p in nets if p['size'] <= BUDGET_DS[ds] + 1), key=lambda p: p['f1'])
        tb = [p for p in trees if p['size'] <= BUDGET_DS[ds] + 1 and ms(ds, p) is not None]
        VF[f'{ds}_lat_budget_net_ms'] = f'{ms(ds, nb):.2f}'
        if tb:
            VF[f'{ds}_lat_budget_tree_ms'] = f"{ms(ds, max(tb, key=lambda p: p['f1'])):.2f}"
        # sustained flows per second on one core (final review round 2, journal switch): 1 / per-record time, and
        # 1,000 / time of one 1,000-record call; the compiled-tree value is the LightGBM per-tree estimate
        bms = lambda p: L[ds].get(p['cfg'], {}).get('int8_batch1000_ms' if p['q'] else 'batch1000_ms')
        fps = lambda t: f"{1000 / t:,.0f}" if t else '–'
        bfps = lambda t: f"{1e6 / t:,.0f}" if t else '–'
        FPS.setdefault(ds, {})
        FPS[ds]['Most accurate tree model: ' + plabel(bt['cfg'])] = (fps(a), bfps(bms(bt)))
        FPS[ds]['Fastest tree model within 0.005 of it: ' + plabel(fast['cfg'])] = (fps(ms(ds, fast)), bfps(bms(fast)))
        FPS[ds]['Most accurate network: ' + label(bn['cfg'], bn['q'])] = (fps(b), bfps(bms(bn)))
        VF[f'{ds}_fps_best_tree'], VF[f'{ds}_fps_best_tree_batch'] = fps(a), bfps(bms(bt))
        VF[f'{ds}_fps_fast_near'] = fps(ms(ds, fast))
        VF[f'{ds}_fps_fast_near_batch'] = bfps(bms(fast))
        VF[f'{ds}_fps_best_net'], VF[f'{ds}_fps_best_net_batch'] = fps(b), bfps(bms(bn))
        VF[f'{ds}_cores_10k_best_tree'] = f"{10000 * a / 1000:.0f}"
        px = L[ds].get('lgbm_proxy')
        if px and px.get('config') in L[ds]:
            VF[f'{ds}_lgbm_us_per_tree'] = f"{px['us_per_tree']:.2f}"
            lgbm_sp.append(L[ds][px['config']]['us_per_tree'] / px['us_per_tree'])
    sel = lambda pred: [e for ds in DSS for c, e in L[ds].items() if isinstance(e, dict) and pred(c)]
    VF['lat_net_f32_range'] = rng([e['single_ms'] for e in sel(lambda c: c in NETS)])
    _ov = [(e['tflite_f32_size_bytes'] - e['size_bytes']) / 1024 for e in sel(lambda c: c in NETS)]
    VF['tflite_f32_overhead_kb'] = f"{min(_ov):.1f} to {max(_ov):+.1f}".replace('-', '−')
    VF['lat_tflite_f32_range'] = rng([e['tflite_f32_single_ms'] for e in sel(lambda c: c in NETS)], '.3f')
    VF['lat_mlp_int8_range'] = rng([e['int8_single_ms'] for e in sel(lambda c: c in ('mlp_w0.5', 'mlp_w1'))])
    VF['lat_cnn_int8_range'] = rng([e['int8_single_ms'] for e in sel(lambda c: c in ('cnn_w1', 'cnn_w2'))], '.1f')
    VF['lat_int8_all_range'] = rng([e['int8_single_ms'] for e in sel(lambda c: c in NETS)], '.2f')
    nd = [abs(e['int8_single_ms_nodelegate'] / e['int8_single_ms'] - 1) for e in sel(lambda c: c in NETS)
          if e.get('int8_single_ms_nodelegate')]
    if nd:
        VF['lat_int8_delegate_maxdiff'] = f'{100 * max(nd):.0f}%'
    fi8 = sel(lambda c: c in NETS)
    if all('fullint8_single_ms' in e for e in fi8):
        VF['lat_fullint8_range'] = rng([e['fullint8_single_ms'] for e in fi8], '.2f')
        VF['fullint8_drop_max'] = f"{max(e['float32_macro_f1_seed0'] - e['fullint8_macro_f1_seed0'] for e in fi8):.3f}"
        VF['fullint8_vs_int8_maxdiff'] = f"{max(abs(e['int8_macro_f1_seed0'] - e['fullint8_macro_f1_seed0']) for e in fi8):.3f}"
        VF['fullint8_faster_n'] = f"{sum(e['fullint8_single_ms'] < e['int8_single_ms'] for e in fi8)} of {len(fi8)}"
    VF['lat_int8_batch1000_range'] = rng([e['int8_batch1000_ms'] for e in sel(lambda c: c in NETS)], '.1f')
    VF['lat_rf25_range'] = rng([e['single_ms'] for e in sel(lambda c: c.startswith('rf_t25') and c in RFS)], '.1f')
    VF['lat_rf100_range'] = rng([e['single_ms'] for e in sel(lambda c: c.startswith('rf_t100'))], '.1f')
    VF['lat_gb_range'] = rng([e['single_ms'] for e in sel(lambda c: c in GB)], '.1f')
    VF['lat_gbc_range'] = rng([e['single_ms'] for e in sel(lambda c: c in CAPPED)], '.1f')
    VF['lat_gb_us_per_tree'] = rng([e['us_per_tree'] for e in sel(lambda c: c in GB + CAPPED)], '.0f')
    VF['lat_lr_max'] = f"{max(e['single_ms'] for e in sel(lambda c: c == 'lr')):.2f}"
    VF['lat_gb_batch_speedup'] = rng([1000 * e['single_ms'] / e['batch1000_ms'] for e in sel(lambda c: c in GB)], '.0f')
    VF['lat_gb_batch1000_range'] = rng([e['batch1000_ms'] for e in sel(lambda c: c in GB)], '.0f')
    VF['lat_net_batch1000_range'] = rng([e['batch1000_ms'] for e in sel(lambda c: c in NETS)], '.1f')
    _keys = list(dict.fromkeys(k.split(':')[0] for ds in HARD for k in FPS.get(ds, {})))
    _rows = [['Model (TON-IoT; UNSW-NB15)'] + [f'{DSN[ds]} {m}' for ds in HARD for m in ('single', 'batched')]]
    for k in _keys:
        names, cells = [], []
        for ds in HARD:
            kk = next(x for x in FPS[ds] if x.startswith(k + ':'))
            names.append(kk.split(': ', 1)[1]); cells += list(FPS[ds][kk])
        _rows.append([f"{k} ({'; '.join(dict.fromkeys(names))})"] + cells)
    FPS_ROWS = _rows
    if lgbm_sp:
        VF['lgbm_tree_speedup'] = rng(lgbm_sp, '.0f')
        VF['lgbm_us_per_tree_range'] = rng([L[ds]['lgbm_proxy']['us_per_tree'] for ds in DSS if 'lgbm_proxy' in L[ds]], '.2f')

# ── computed sentences whose wording depends on the outcome (final only) ─────
if all(f'{ds}_budget_winner' in VF for ds in HARD):
    # wording follows the Welch test at 0.05: 'tie' = no significant difference between the best int8/float network
    # and the best tree model no larger than the largest int8 network (with its input table)
    def _ph(ds):
        w, d = VF[f'{ds}_budget_winner'], DSN[ds]
        if w == 'tie':
            nf, tf = VF[f'{ds}_budget_net_f1'], VF[f'{ds}_budget_tree_f1']
            return f"the best network and the best tree model of that size tied on {d} ({'both ' + nf if nf == tf else nf + ' and ' + tf})"
        fam = 'a network' if w == 'network' else 'a tree model'
        return f"{fam} was more accurate on {d} (by {VF[f'{ds}_budget_gap']})"
    def _hl(ds):
        w, d = VF[f'{ds}_budget_winner'], DSN[ds]
        return {'tie': f'networks and trees tied on {d}', 'network': f'networks won on {d}',
                'tree': f'trees won on {d}'}[w]
    def _sum(ds):
        w, d = VF[f'{ds}_budget_winner'], DSN[ds]
        return {'tie': f'the compact networks and size-budgeted trees were statistically tied on {d}',
                'network': f'the compact networks were more accurate on {d}',
                'tree': f'size-budgeted trees were more accurate on {d}'}[w]
    _w = {VF[f'{ds}_budget_winner'] for ds in HARD}
    _kbs = ' and '.join(f"{V[f'{ds}_budget_kb']} KB on {DSN[ds]}" for ds in HARD)
    if _w == {'tree'}:
        VF['abs_budget'] = (f'At the size of the largest int8 network with its input table ({_kbs}), the best tree model '
                            'was more accurate than the best network, by ' + ' and '.join(
                                f"{VF[f'{ds}_budget_gap']}" for ds in HARD) + '.')
        VF['hl_budget'] = "At the largest int8 network's size, trees were more accurate on both hard datasets"
    else:
        VF['abs_budget'] = (f'At the size of the largest int8 network with its input table ({_kbs}), '
                            + ' and '.join(_ph(ds) for ds in HARD) + '.')
        VF['hl_budget'] = 'At the largest int8 network size, ' + '; '.join(_hl(ds) for ds in HARD)
    VF['budget_summary'] = 'At the size of the largest int8 network, ' + ' and '.join(_sum(ds) for ds in HARD)
if all(f'{ds}_gb_rb_collapsed' in VF for ds in HARD):
    c = [VF[f'{ds}_gb_rb_collapsed'] for ds in HARD]
    n_ = sum(int(x.split()[0]) for x in c); N_ = sum(int(x.split()[-1]) for x in c)
    VF['rb_outcome'] = ('rolling back to the best iteration removed every collapse' if n_ == 0 else
                        f'after rollback {n_} of {N_} default-rate runs still collapsed')
for ds in DSS:
    fp = [p['fpr'] for p in points[ds] if p['cfg'] != 'lr']
    if fp:
        VF[f'{ds}_fpr_range'] = f'{100 * min(fp):.0f}–{100 * max(fp):.0f}%'
        _lr = [p['fpr'] for p in points[ds] if p['cfg'] == 'lr']
        if _lr:
            VF[f'{ds}_lr_fpr'] = f'{100 * _lr[0]:.0f}%'
    bt = max((p for p in points[ds] if is_tree(p)), key=lambda p: p['f1'], default=None)
    px = _lat.get(ds, {}).get('lgbm_proxy') if ST['latency'] else None
    if bt and px and 'ntrees' in bt:
        VF[f'{ds}_lgbm_est_ms'] = f"{bt['ntrees'][0] * px['us_per_tree'] / 1000:.2f}"
        VF[f'{ds}_fps_lgbm_est'] = f"{1e6 / (bt['ntrees'][0] * px['us_per_tree']):,.0f}"
if 'FPS_ROWS' in globals():
    if all(f'{ds}_fps_lgbm_est' in VF for ds in HARD):
        FPS_ROWS.append(['Most accurate tree model, compiled-library estimate'] +
                        [x for ds in HARD for x in (VF[f'{ds}_fps_lgbm_est'], '–')])
    VF['fps_min_batch'] = f"{min(int(r[i].replace(',', '')) for r in FPS_ROWS[1:] for i in (2, 4) if r[i] != '–'):,}"
    TF['fps'] = md(FPS_ROWS)
_R = runs('UNSW_NB15', 'rf_t200_dNone', RES)
if _R:
    _pc = {c: st.mean(r['per_class_f1'][c] for r in _R) for c in _R[0]['per_class_f1']}
    V['UNSW_NB15_worst_classes'] = ', '.join(f'{c} {v:.2f}' for c, v in sorted(_pc.items(), key=lambda x: x[1])[:3])
for ds in DSS:
    V[f'{ds}_feature_list'] = ', '.join(DATA[ds]['features']) if ds != 'N_BaIoT' else ''

# budget without the input table (final review round 2): what the comparison would show if the table were not counted
for ds in HARD:
    _n = [p for p in points[ds] if p['cfg'].startswith(('cnn', 'mlp'))]
    if not _n or 'size_w' not in _n[0]:
        continue
    _B = max(p['size_w'] for p in _n if p['q'])
    _bn = max((p for p in _n if p['size_w'] <= _B), key=lambda p: p['f1'])
    _bt = max((p for p in points[ds] if is_tree(p) and p['size'] <= _B), key=lambda p: p['f1'])
    VF[f'{ds}_nott_kb'] = kb(_B)
    VF[f'{ds}_nott_net_f1'] = f"{_bn['f1']:.3f}"
    VF[f'{ds}_nott_tree_f1'] = f"{_bt['f1']:.3f}"
    VF[f'{ds}_nott_tree'] = _bt['name']
    VF[f'{ds}_nott_winner'] = 'network' if _bn['f1'] > _bt['f1'] else 'tree'
# ── assemble ─────────────────────────────────────────────────────────────────
V['p1final_complete'] = FINAL
V['p1final_status'] = ST
if FINAL:
    V.update(VF); T.update(TF)
V['p1final_pending_keys'] = [] if FINAL else sorted(VF)
V['tables'] = T

# final review round 2: five further stratified data splits (run_p1_final --group resplit; one seed per split, seed k on
# split k): the best tree model, the smallest tree significantly above every network and the best tree of at most
# 110 KB against the best network (TON-IoT: float32 CNN of width 2; UNSW-NB15: its int8 version)
RS_TREES = {'TON_IoT': (('best', 'gb_l63'), ('oversig', 'gbc50_l31'), ('small', 'gb_l15')),
            'UNSW_NB15': (('best', 'gb05_l31'), ('oversig', 'gbc50_l15'), ('small', 'gbc100_l7'))}
for ds, trs in RS_TREES.items():
    def _rs(cfg, k):
        f = FIN / ds / f'{cfg}_split{k}' / f'seed{k}.json'
        return json.loads(f.read_text()) if f.exists() else None
    nets = [_rs('cnn_w2', k) for k in range(1, 6)]
    if not all(nets):
        continue
    net = [(n['int8']['macro_f1'] if ds == 'UNSW_NB15' else n['macro_f1']) for n in nets]
    V[f'{ds}_rs_net_f1'] = f'{np.mean(net):.3f}'
    for lab, cfg in trs:
        t = [_rs(cfg, k) for k in range(1, 6)]
        if not all(t):
            break
        d = [a['macro_f1'] - b for a, b in zip(t, net)]
        V[f'{ds}_rs_{lab}_pos'] = str(sum(x > 0 for x in d))
        V[f'{ds}_rs_{lab}_mean'] = f'{np.mean(d):+.3f}'.replace('-', '−')
        V[f'{ds}_rs_{lab}_min'] = f'{min(d):+.4f}'.replace('-', '−')
        V[f'{ds}_rs_{lab}_f1'] = f'{np.mean([a["macro_f1"] for a in t]):.3f}'
    else:
        V[f'{ds}_rs_done'] = 'yes'

(HERE / f'values{SUFFIX}.json').write_text(json.dumps(V, indent=1, ensure_ascii=False), encoding='utf-8')
(HERE / f'points{SUFFIX}.json').write_text(json.dumps(dict(
    _final=FINAL, **{ds: [{k: v for k, v in p.items() if k not in ('f', 'vf', 'clean')} for p in pts] for ds, pts in points.items()}),
    indent=1), encoding='utf-8')
print(f'FINAL={FINAL} status={ST}')
print(f'{len(V) - 4} values ({len(VF)} final-only {"included" if FINAL else "withheld"}), {len(T)} tables; points:',
      {ds: len(p) for ds, p in points.items()})
