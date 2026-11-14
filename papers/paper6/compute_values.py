"""Paper 6: every number quoted in Manuscript.md, from the result files of run_p6_v2.py (round 2: results_v4).

Writes values.json: scalar placeholders {{key}} and markdown tables {{table:key}}.
Missing result groups are skipped, so their placeholders stay visible as ⟦key⟧ in a partial build.
Set P6_RES=<dir name> to analyse another results directory (e.g. results_v3 for a code check).

Statistics (Manuscript §4.3):
  * networks vs networks: paired over seeds (same seed = same initialisation stream); |d| < 1e-6 is a tie;
  * trees vs networks: Welch's t-test (no shared seeds);
  * confirmatory family (Holm): on TON-IoT and UNSW-NB15, late−early, attn−early, late−early_wide, late−late_all;
  * semantic vs random splits: within seed s, semantic run minus the mean of the two random splits with seed s;
  * data splits: original split (seed mean) + 5 resplits = 6 paired observations.
"""
import json
import os
import statistics as st
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
PIPE = HERE.parents[1]
BASE = PIPE / 'experiments' / 'paper6_multimodal'
RES = BASE / os.environ.get('P6_RES', 'results_v4')
RES_MM, RES_Q = BASE / 'results_v2', BASE / 'results_v3'      # scaling comparison: identical except input scaling
DATA = json.loads((HERE / 'data_stats.json').read_text())
DSS = ('TON_IoT', 'UNSW_NB15', 'N_BaIoT')
CONF = ('TON_IoT', 'UNSW_NB15')
DSN = {'TON_IoT': 'TON-IoT', 'UNSW_NB15': 'UNSW-NB15', 'N_BaIoT': 'N-BaIoT'}
SEEDS = range(5)
N_RANDOM, RESPLITS = 10, range(1, 6)
BUDGET = 120
CNN = ('s1', 's2', 'early', 'late', 'attn')
ARCH_NAME = {'s1': 'Stream 1 only (CNN)', 's2': 'Stream 2 only (CNN)', 'early': 'Early fusion (CNN, all features)',
             'late': 'Late fusion', 'attn': 'Attention fusion', 'early_wide': 'Wide early fusion (×1.6)',
             'early_head': 'Early fusion with head', 'late_all': 'Late fusion on all features',
             'm_s1': 'MLP stream 1', 'm_s2': 'MLP stream 2', 'm_early': 'MLP early fusion', 'm_late': 'MLP late fusion',
             'm_attn': 'MLP attention fusion',
             'rf': 'RF, all features', 'rf_s1': 'RF, stream 1', 'rf_s2': 'RF, stream 2',
             'gb': 'GBDT, all features', 'gb_s1': 'GBDT, stream 1', 'gb_s2': 'GBDT, stream 2'}
TREES = ('rf', 'rf_s1', 'rf_s2', 'gb', 'gb_s1', 'gb_s2')
V, T = {}, {}


def f3(x):
    return f'{x:.3f}'


def pf(p):
    """p-value: '< 0.001' below 0.0005 (build.py turns 'p = < 0.001' into 'p < 0.001')."""
    return '< 0.001' if p < 0.0005 else f'{p:.3f}'


def sgn(x, nd=3):
    if abs(x) < 0.5 * 10 ** -nd:
        return f'{0:.{nd}f}'
    return f'{x:+.{nd}f}'.replace('-', '−')


# GBDT runs rolled back to the best internal-validation iteration (run_p6_gb_rollback.py -> results_p6gb): used in
# place of the results_v4 GBDT runs once every GBDT run exists there (P6_GB_ROLLBACK=0 disables the switch).
# scikit-learn's early stopping keeps the 20 iterations after the best validation score, in which boosting diverged on
# some seeds (e.g. TON-IoT gb_s1_rand1/2/5/7 in results_v4).
RES_GB = Path(os.environ['P6_GB_RES']) if os.environ.get('P6_GB_RES') else BASE / 'results_p6gb'   # override: code checks
_GB_EXPS = [(ds, a, s) for ds in DSS for a in ('gb', 'gb_s1', 'gb_s2') for s in SEEDS] + \
           [(ds, f'{a}_rand{r}', r % 5) for ds in DSS for a in ('gb_s1', 'gb_s2') for r in range(N_RANDOM)] + \
           [(ds, f'gb_split{k}', k) for ds in CONF for k in RESPLITS]
GB_ROLLBACK = (os.environ.get('P6_GB_ROLLBACK', '1') != '0' and RES.name == 'results_v4'
               and all((RES_GB / ds / e / f'seed{s}.json').exists() for ds, e, s in _GB_EXPS))


def is_gb(exp):
    return exp == 'gb' or exp.startswith('gb_')


def res_for(exp, base=None):
    """results folder of an experiment: results_p6gb for GBDT runs once complete, else base or RES"""
    if base is None and GB_ROLLBACK and is_gb(exp):
        return RES_GB
    return base or RES


def load(ds, exp, seed, base=None):
    p = res_for(exp, base) / ds / exp / f'seed{seed}.json'
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    if 'clean_macro_f1' not in r and 'reused_from' in r:
        # N-BaIoT main runs reused from v3 (saturated, stopped early) predate the clean metric; their clean
        # mask covers every test record (no test record duplicates a training record), so clean = all
        # Random-split runs reused from v3 on TON-IoT/UNSW-NB15 have no clean score and get none (the random-split
        # analysis uses macro_f1 only; main/capacity use clean only when every run has it)
        cm = (base or RES) / ds / 'clean_split0.npy'
        if cm.exists() and bool(np.load(cm).all()):
            r['clean_macro_f1'] = r['macro_f1']
        else:
            assert '_rand' in exp, (ds, exp, seed)
    return r


def runs(ds, exp, base=None):
    R = [load(ds, exp, s, base) for s in SEEDS]
    return R if all(R) else None


def rand_runs(ds, arch):
    R = [load(ds, f'{arch}_rand{r}', r % 5) for r in range(N_RANDOM)]
    return R if all(R) else None


def mean(R, k='macro_f1'):
    return st.mean(r[k] for r in R)


def sd(R, k='macro_f1'):
    return st.stdev(r[k] for r in R)


def tci(d):
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    p = stats.ttest_1samp(d, 0).pvalue if d.std() > 0 else 1.0
    return dict(mean=d.mean(), lo=d.mean() - h, hi=d.mean() + h, min=d.min(), max=d.max(),
                pos=int((d > 1e-6).sum()), ties=int((np.abs(d) <= 1e-6).sum()), n=len(d), p=p)


def put(key, P, n=5):
    V[key] = sgn(P['mean'])
    V[key + '_4'] = sgn(P['mean'], 4)
    V[key + '_ci'] = f"[{sgn(P['lo'])}, {sgn(P['hi'])}]"
    V[key + '_pos'] = str(P['pos'])
    V[key + '_range'] = f"{sgn(P['min'])} to {sgn(P['max'])}"
    V[key + '_p'] = pf(P['p'])
    ties = (', ' + str(P['ties']) + (' tie' if P['ties'] == 1 else ' ties')) if P['ties'] else ''
    return f"{sgn(P['mean'])} [{sgn(P['lo'])}, {sgn(P['hi'])}] ({P['pos']}/{n}{ties})"


def paired(ds, a, b, k='macro_f1'):
    A, B = runs(ds, a), runs(ds, b)
    if not (A and B):
        return None
    return tci([x[k] - y[k] for x, y in zip(A, B)])


def md(rows):
    out = ['| ' + ' | '.join(rows[0]) + ' |', '|' + '|'.join('---' for _ in rows[0]) + '|']
    return '\n'.join(out + ['| ' + ' | '.join(r) + ' |' for r in rows[1:]])


def best_stream(ds, prefix=''):
    return max((prefix + 's1', prefix + 's2'), key=lambda x: mean(runs(ds, x)))


# ── data ──────────────────────────────────────────────────────────────────────
for ds in DSS:
    d = DATA[ds]
    V[f'{ds}_n_feat'] = str(len(d['features']))
    V[f'{ds}_n_s1'], V[f'{ds}_n_s2'] = str(len(d['s1'])), str(len(d['s2']))
    V[f'{ds}_n_classes'] = str(len(d['classes']))
    for k in ('n_train', 'n_fit', 'n_val', 'n_test'):
        V[f'{ds}_{k}'] = f'{d[k]:,}'
    c = RES / ds / 'clean_split0.npy'
    if c.exists():
        m = np.load(c)
        V[f'{ds}_dup_sel_pct'] = f'{100 * (1 - m.mean()):.1f}%'
        V[f'{ds}_clean_n'] = f'{int(m.sum()):,}'
V['TON_mitm_test'] = str(DATA['TON_IoT']['test']['mitm'])
V['UNSW_worms_test'] = str(DATA['UNSW_NB15']['test']['worms'])
V['TON_s1_median_max'] = f"{max(DATA['TON_IoT']['s1_median']):.4f}"
V['TON_s1_sd_max'] = f"{max(DATA['TON_IoT']['s1_sd']):.3f}"
V['TON_s1_sd_min'] = f"{min(DATA['TON_IoT']['s1_sd']):.4f}"
for ds in DSS:     # macro-F1 of always predicting the majority (benign) class
    te = DATA[ds]['test']; n = sum(te.values()); K = len(DATA[ds]['classes'])
    p = max(te.values()) / n
    V[f'{ds}_majority_f1'] = f3(2 * p / (1 + p) / K)

rows = [['Dataset', 'Classes', 'Features (stream 1 + stream 2)', 'Training (after SMOTE)', 'Validation', 'Test', 'Stream 1', 'Stream 2']]
DESC = {'TON_IoT': ('volume and timing: duration, bytes and packets per direction',
                    'ports, protocol, service, connection state, DNS fields'),
        'UNSW_NB15': ('volume and timing: bytes and load per direction, mean source packet size, ACK delay',
                      'TTL and connection-count fields (sttl, ct_state_ttl, ct_dst_src_ltm, ct_srv_dst)'),
        'N_BaIoT': ('host-level statistics (MI_dir_*, H_*)', 'channel and socket statistics (HH_*, HH_jit_*, HpHp_*)')}
for ds in DSS:
    d = DATA[ds]
    rows.append([DSN[ds], str(len(d['classes'])), f"{len(d['features'])} ({len(d['s1'])} + {len(d['s2'])})",
                 f"{d['n_train']:,} ({d['n_fit']:,})" if d['n_fit'] != d['n_train'] else f"{d['n_train']:,} (no SMOTE)",
                 f"{d['n_val']:,}", f"{d['n_test']:,}", *DESC[ds]])
T['data'] = md(rows)

# tuned trees
for ds in DSS:
    p = RES / ds / 'tune_trees.json'
    if p.exists():
        tt = json.loads(p.read_text())
        V[f'{ds}_rf_cfg'] = ', '.join(f'{k} {v}' for k, v in tt['rf']['best'].items())
        V[f'{ds}_gb_cfg'] = ', '.join(f'{k} {v}' for k, v in tt['gb']['best'].items())
    pg = RES_GB / ds / 'tune_trees.json'
    if GB_ROLLBACK and pg.exists():          # GBDT re-tuned with rollback (same grid and protocol)
        V[f'{ds}_gb_cfg'] = ', '.join(f'{k} {v}' for k, v in json.loads(pg.read_text())['gb']['best'].items())
V['gb_rollback_used'] = str(GB_ROLLBACK)
if GB_ROLLBACK:     # emitted only with the rolled-back runs, so a build before the re-run shows ⟦gb_rollback_note⟧
    V['gb_rollback_note'] = ('Every GBDT is evaluated at its best internal-validation iteration: scikit-learn stops 20 '
                             'iterations after it but keeps those iterations, in which boosting can diverge.')

# ── main results ──────────────────────────────────────────────────────────────
ORDER = CNN + ('early_wide', 'early_head', 'late_all') + tuple('m_' + a for a in CNN) + TREES
rows = [['Model', 'Params (TON-IoT)'] + [f'{DSN[ds]} {m}' for ds in DSS for m in ('macro-F1', 'FPR')]]
for a in ORDER:
    cells, params = [], None
    for ds in DSS:
        R = runs(ds, a)
        if R is None:
            cells += ['–', '–']; continue
        params = params or (R[0].get('params') if isinstance(R[0].get('params'), int) else None)   # GBDT rollback runs store hyper-parameters there
        cells += [f'{mean(R):.3f} ± {sd(R):.3f}', f"{mean(R, 'benign_fpr'):.3f}"]
        V[f'{ds}_{a}_f1'], V[f'{ds}_{a}_sd'] = f3(mean(R)), f3(sd(R))
        V[f'{ds}_{a}_f1_4'] = f'{mean(R):.4f}'
        V[f'{ds}_{a}_fpr'] = f3(mean(R, 'benign_fpr'))
        V[f'{ds}_{a}_det'] = f3(mean(R, 'attack_detection'))
        if all('macro_pr_auc' in r for r in R):
            V[f'{ds}_{a}_prauc'] = f3(mean(R, 'macro_pr_auc'))
        if all('clean_macro_f1' in r for r in R):
            V[f'{ds}_{a}_clean_f1'] = f3(mean(R, 'clean_macro_f1'))
        if isinstance(R[0].get('params'), int):
            V[f'{ds}_{a}_params'] = f"{R[0]['params']:,}"
    if all(c == '–' for c in cells):
        continue
    rows.append([ARCH_NAME[a], '–' if a in TREES else (f'{params:,}' if params else '–')] + cells)
T['main'] = md(rows)
for ds in DSS:   # spread of the main table: largest seed SD among the networks, FPR range over all models
    nets = [a for a in CNN if runs(ds, a)]     # the five main CNNs (text says so)
    if nets:
        a_sd = max(nets, key=lambda a: sd(runs(ds, a)))
        V[f'{ds}_sd_max'], V[f'{ds}_sd_max_model'] = f3(sd(runs(ds, a_sd))), ARCH_NAME[a_sd].split(' (')[0].lower()
    allm = [a for a in ORDER if runs(ds, a)]
    if allm:
        fp = [mean(runs(ds, a), 'benign_fpr') for a in allm]
        V[f'{ds}_fpr_min_all'], V[f'{ds}_fpr_max_all'] = f3(min(fp)), f3(max(fp))
for ds in DSS:
    for a, k in (('early', 'single'), ('late', 'late'), ('attn', 'attn')):
        if f'{ds}_{a}_params' in V:
            V[f'{ds}_params_{k}'] = V[f'{ds}_{a}_params']
dev = []
for a in ORDER:
    if a in TREES or not runs('TON_IoT', a):
        continue
    p0 = runs('TON_IoT', a)[0]['params']
    dev += [abs(runs(ds, a)[0]['params'] - p0) / p0 for ds in DSS if runs(ds, a) and 'params' in runs(ds, a)[0]]
V['params_dev_max_pct'] = f'{100 * max(dev):.0f}%' if max(dev) >= 0.01 else f'{100 * max(dev):.1f}%'
if 'TON_IoT_early_wide_params' not in V and runs('TON_IoT', 'early_wide'):
    V['TON_IoT_early_wide_params'] = f"{runs('TON_IoT', 'early_wide')[0]['params']:,}"

# epoch budget: share of network runs that used all epochs
net_runs = [r for ds in DSS for a in CNN + ('early_wide', 'early_head', 'late_all') if runs(ds, a) for r in runs(ds, a)
            if 'reused_from' not in r]
if net_runs:
    V['n_cnn_runs_main'] = str(len(net_runs))
    V['n_capped'] = str(sum(r['epochs'] >= BUDGET for r in net_runs))
    V['best_epoch_max'] = str(max(r['best_epoch'] for r in net_runs))

# ── paired comparisons and the confirmatory family ───────────────────────────
COMP = [('early', 'best1', 'Early fusion − better single stream'),
        ('late', 'early', 'Late − early fusion'),
        ('attn', 'early', 'Attention − early fusion'),
        ('late', 'attn', 'Late − attention fusion'),
        ('early_wide', 'early', 'Wide − standard early fusion'),
        ('early_head', 'early', 'Early fusion with head − early fusion'),
        ('late', 'early_head', 'Late fusion − early fusion with head'),
        ('late', 'early_wide', 'Late fusion − wide early fusion'),
        ('late', 'late_all', 'Late fusion − late fusion on all features'),
        ('m_late', 'm_early', 'MLP: late − early fusion'),
        ('m_attn', 'm_early', 'MLP: attention − early fusion')]
FAMILY = {('late', 'early'), ('attn', 'early'), ('late', 'early_wide'), ('late', 'late_all')}
rows = [['Comparison'] + [DSN[ds] for ds in DSS]]
pvals, famP = {}, {}
for a, b, name in COMP:
    cells = []
    for ds in DSS:
        bb = best_stream(ds) if b == 'best1' and runs(ds, 's1') else b
        P = paired(ds, a, bb)
        if P is None:
            cells.append('–'); continue
        key = f'{ds}_d_{a}_{b}'
        cells.append(put(key, P))
        if (a, b) in FAMILY and ds in CONF:
            pvals[key] = P['p']
            famP[key] = (ds, name, P)
        Pc = paired(ds, a, bb, 'clean_macro_f1') if runs(ds, a) and 'clean_macro_f1' in runs(ds, a)[0] else None
        if Pc is not None:
            put(key + '_clean', Pc)
    if any(c != '–' for c in cells):
        rows.append([name] + cells)
# trees vs networks (Welch) and RF gain from all features
for ds in DSS:
    for tr in ('rf', 'gb'):
        for net in ('late', 'early'):
            A, B = runs(ds, tr), runs(ds, net)
            if A and B:
                x, y = [r['macro_f1'] for r in A], [r['macro_f1'] for r in B]
                V[f'{ds}_d_{tr}_{net}'] = sgn(st.mean(x) - st.mean(y))
                V[f'{ds}_{tr}_{net}_welch_p'] = pf(stats.ttest_ind(x, y, equal_var=False).pvalue)
        if runs(ds, tr) and runs(ds, f'{tr}_s1') and runs(ds, f'{tr}_s2'):
            bs = max((f'{tr}_s1', f'{tr}_s2'), key=lambda z: mean(runs(ds, z)))
            P = paired(ds, tr, bs)
            put(f'{ds}_d_{tr}_{tr}_best1', P)
rows.append(['RF (all features) − late fusion (Welch)'] + [V.get(f'{ds}_d_rf_late', '–') + (f" (p = {V[f'{ds}_rf_late_welch_p']})" if f'{ds}_rf_late_welch_p' in V else '') for ds in DSS])
rows.append(['GBDT (all features) − late fusion (Welch)'] + [V.get(f'{ds}_d_gb_late', '–') + (f" (p = {V[f'{ds}_gb_late_welch_p']})" if f'{ds}_gb_late_welch_p' in V else '') for ds in DSS])
T['paired'] = md(rows)
ks = sorted(pvals, key=pvals.get); m_ = len(ks); run_max = 0.0
for i, k in enumerate(ks):
    run_max = max(run_max, min(1.0, (m_ - i) * pvals[k]))
    V[k + '_pholm'] = pf(run_max)
    V[k + '_pholm_num'] = run_max
assert m_ == 8, f'Holm family has {m_} tests, expected 8'      # a missing result file must not shrink the family
V['holm_family_size'] = str(m_)
V['holm_sig_list'] = ', '.join(k for k in ks if V[k + '_pholm_num'] < 0.05) or 'none'


def margin_verdict(P, mg=0.01):
    if P['lo'] > -mg and P['hi'] < mg:
        return 'negligible (CI inside ±0.01)'
    if P['lo'] > mg or P['hi'] < -mg:
        return 'beyond the margin'
    return 'inconclusive (CI crosses ±0.01)'


rows = [['Dataset', 'Comparison', 'Mean difference [95% CI] (seeds > 0)', 'Unadjusted p', 'Holm-adjusted p', 'Margin reading']]
for ds in CONF:
    for k in [k for k in famP if famP[k][0] == ds]:
        _, name, P = famP[k]
        rows.append([DSN[ds], name, f"{sgn(P['mean'])} [{sgn(P['lo'])}, {sgn(P['hi'])}] ({P['pos']}/5)",
                     pf(P['p']), pf(V[k + '_pholm_num']), margin_verdict(P)])
        V[k + '_margin'] = margin_verdict(P).split(' (')[0]
T['holm'] = md(rows)
for k in ks:
    del V[k + '_pholm_num']

# share of the TON-IoT late-minus-early gain reproduced by each control (means over seeds)
_g = mean(runs('TON_IoT', 'late')) - mean(runs('TON_IoT', 'early'))
for a in ('late_all', 'early_head', 'early_wide'):
    V[f'TON_IoT_{a}_share'] = f"{100 * (mean(runs('TON_IoT', a)) - mean(runs('TON_IoT', 'early'))) / _g:.0f}%"

# ── data splits: original split (seed mean) + 5 resplits ─────────────────────
for ds in CONF:
    def split_vals(a):
        R0 = runs(ds, a)
        Rk = [load(ds, f'{a}_split{k}', k) for k in RESPLITS]
        if not R0 or not all(Rk):
            return None
        return [mean(R0)] + [r['macro_f1'] for r in Rk]
    have = {a: split_vals(a) for a in ('s1', 's2', 'early', 'late', 'attn', 'rf', 'gb', 'rf_s1', 'rf_s2')}
    for a, v in have.items():
        if v:
            V[f'{ds}_split_{a}_mean'] = f3(st.mean(v))
            V[f'{ds}_split_{a}_range'] = f'{f3(min(v))}–{f3(max(v))}'
    for a, b in (('late', 'early'), ('attn', 'early'), ('rf', 'late'), ('gb', 'late')):
        if have.get(a) and have.get(b):
            put(f'{ds}_splitd_{a}_{b}', tci(np.subtract(have[a], have[b])), 6)
    if all(have.get(a) for a in ('early', 's1', 's2')):
        put(f'{ds}_splitd_early_best1', tci(np.subtract(have['early'], np.maximum(have['s1'], have['s2']))), 6)

# ── attention: weights, contributions, in-model ablation ─────────────────────
rng = np.random.default_rng(0)


def perm_p(x, y, n=20000):
    x, y = np.asarray(x), np.asarray(y)
    r0 = stats.spearmanr(x, y).correlation
    rs = np.array([stats.spearmanr(x, rng.permutation(y)).correlation for _ in range(n)])
    return r0, float((np.abs(rs) >= abs(r0) - 1e-12).mean())


for ds in DSS:
    A = runs(ds, 'attn')
    if not A:
        continue
    w = [r['attn_s1_mean'] for r in A]
    V[f'{ds}_attn_w_mean'], V[f'{ds}_attn_w_range'] = f'{st.mean(w):.2f}', f'{min(w):.2f}–{max(w):.2f}'
    V[f'{ds}_attn_sd_within'] = f"{st.mean(r['attn_s1_sd_within'] for r in A):.2f}"
    cls = list(A[0]['attn_s1_per_class'])
    S1, S2 = runs(ds, 's1'), runs(ds, 's2')
    adv = [st.mean(S1[s]['per_class_f1'][c] - S2[s]['per_class_f1'][c] for s in SEEDS) for c in cls]
    rhos = [stats.spearmanr([r['attn_s1_per_class'][c] for c in cls], adv).correlation for r in A]
    V[f'{ds}_attn_rho_adv'] = f'{st.mean(rhos):.2f}'
    V[f'{ds}_attn_rho_adv_range'] = f'{sgn(min(rhos), 2)} to {sgn(max(rhos), 2)}'
    wbar = [st.mean(r['attn_s1_per_class'][c] for r in A) for c in cls]
    _, p = perm_p(wbar, adv)
    V[f'{ds}_attn_rho_adv_p'] = f'{p:.2f}'
    pr = [stats.spearmanr([A[i]['attn_s1_per_class'][c] for c in cls], [A[j]['attn_s1_per_class'][c] for c in cls]).correlation
          for i, j in combinations(range(5), 2)]
    V[f'{ds}_attn_rho_seeds'], V[f'{ds}_attn_rho_seeds_min'] = f'{st.mean(pr):.2f}', f'{min(pr):.2f}'
    pc = dict(zip(cls, wbar))
    V[f'{ds}_attn_class_max'], V[f'{ds}_attn_class_max_w'] = max(pc, key=pc.get), f'{max(pc.values()):.2f}'
    V[f'{ds}_attn_class_min'], V[f'{ds}_attn_class_min_w'] = min(pc, key=pc.get), f'{min(pc.values()):.2f}'
    if all('attn_ablation' in r for r in A):
        # in-model reliance on stream 1 per class: F1 lost when stream 1 is mean-replaced minus F1 lost for stream 2
        rel = {c: st.mean(r['attn_ablation']['s2']['per_class_f1'][c] - r['attn_ablation']['s1']['per_class_f1'][c] for r in A) for c in cls}
        drop = {k: st.mean(r['macro_f1'] - r['attn_ablation'][k]['macro_f1'] for r in A) for k in ('s1', 's2')}
        V[f'{ds}_abl_drop_s1'], V[f'{ds}_abl_drop_s2'] = f3(drop['s1']), f3(drop['s2'])
        r_w, p_w = perm_p(wbar, [rel[c] for c in cls])
        V[f'{ds}_attn_rho_rel'], V[f'{ds}_attn_rho_rel_p'] = f'{r_w:.2f}', f'{p_w:.2f}'
        cbar = [st.mean(r['contrib_s1_per_class'][c] for r in A) for c in cls]
        r_c, p_c = perm_p(cbar, [rel[c] for c in cls])
        V[f'{ds}_contrib_rho_rel'], V[f'{ds}_contrib_rho_rel_p'] = f'{r_c:.2f}', f'{p_c:.2f}'
        r_a, p_a = perm_p([rel[c] for c in cls], adv)
        V[f'{ds}_rel_rho_adv'], V[f'{ds}_rel_rho_adv_p'] = f'{r_a:.2f}', f'{p_a:.2f}'
        V[f'{ds}_attn_check_min'] = f"{min(r['attn_base_check'] for r in A):.4f}"

# per-class tables
for ds in CONF:
    cls = DATA[ds]['classes']
    archs = [a for a in ('s1', 's2', 'early', 'late', 'attn', 'rf', 'gb') if runs(ds, a)]
    if not archs:
        continue
    rows = [['Class', 'Test n'] + [ARCH_NAME[a].replace(' (CNN)', '').replace(' (CNN, all features)', '') for a in archs]]
    for c in cls:
        rows.append([c, f"{DATA[ds]['test'][c]:,}"] + [f"{st.mean(r['per_class_f1'][c] for r in runs(ds, a)):.2f}" for a in archs])
        for a in archs:
            V[f'{ds}_{a}_pc_{c}'] = f"{st.mean(r['per_class_f1'][c] for r in runs(ds, a)):.2f}"
    T[f'perclass_{ds}'] = md(rows)

# ── scaling lesson (v2 min–max vs v3 quantile: identical code otherwise) ─────
rows = [['Dataset', 'Input scaling'] + [ARCH_NAME[a].replace(' (CNN)', '').replace(' (CNN, all features)', '') for a in CNN] + ['RF, stream 1']]
for ds in CONF:
    for tag, base, sfx in (('min–max, first run', RES_MM, 'mm'), ('quantile, first-run protocol', RES_Q, 'q'),
                           ('quantile, final run', RES, 'fin')):
        cells = []
        for a in CNN:
            R = runs(ds, a, base)
            key = f'{ds}_{a}_{sfx}'
            V[key], V[key + '_fpr'] = f3(mean(R)), f3(mean(R, 'benign_fpr'))
            V[key + '_best_ep_max'] = str(max(r['best_epoch'] for r in R))
            cells.append(f3(mean(R)))
        rows.append([DSN[ds], tag] + cells + [f3(mean(runs(ds, 'rf_s1', base)))])
        V[f'{ds}_rf_s1_{sfx}'] = f3(mean(runs(ds, 'rf_s1', base)))
T['scaling'] = md(rows)

# ── semantic vs random splits ────────────────────────────────────────────────
rows = [['Dataset', 'Model', 'Semantic split (mean of 5 seeds)', 'Random splits: mean', 'Random splits: min–max',
         'Within-seed difference, semantic − random [95% CI]']]
have_random = []
for ds in DSS:
    ok = True
    for a in ('s1', 's2', 'late', 'attn', 'rf_s1', 'rf_s2', 'gb_s1', 'gb_s2', 'm_s1', 'm_s2', 'm_late', 'm_attn'):
        R, S = rand_runs(ds, a), runs(ds, a)
        if R is None or S is None:
            if a in ('s1', 's2', 'late', 'attn'):
                ok = False
            continue
        x = [r['macro_f1'] for r in R]
        V[f'{ds}_rand_{a}_mean'], V[f'{ds}_rand_{a}_min'], V[f'{ds}_rand_{a}_max'] = f3(st.mean(x)), f3(min(x)), f3(max(x))
        V[f'{ds}_rand_{a}_below'] = str(sum(v < mean(S) for v in x))
        d = [S[s]['macro_f1'] - st.mean(R[r]['macro_f1'] for r in range(N_RANDOM) if r % 5 == s) for s in SEEDS]
        P = tci(d)
        cell = put(f'{ds}_semrand_{a}', P)
        rows.append([DSN[ds], ARCH_NAME[a].replace(' (CNN)', ''), f3(mean(S)), f3(st.mean(x)), f'{f3(min(x))}–{f3(max(x))}', cell])
    if ok:
        have_random.append(ds)
        sem_gain = [load(ds, 'late', s)['macro_f1'] - max(load(ds, 's1', s)['macro_f1'], load(ds, 's2', s)['macro_f1']) for s in SEEDS]
        rnd_gain = [load(ds, f'late_rand{r}', r % 5)['macro_f1'] - max(load(ds, f's1_rand{r}', r % 5)['macro_f1'],
                                                                     load(ds, f's2_rand{r}', r % 5)['macro_f1']) for r in range(N_RANDOM)]
        V[f'{ds}_gain_sem'], V[f'{ds}_gain_rand'] = sgn(st.mean(sem_gain)), sgn(st.mean(rnd_gain))
        lr = [load(ds, f'late_rand{r}', r % 5)['macro_f1'] for r in range(N_RANDOM)]
        V[f'{ds}_late_sem_minus_rand'] = sgn(mean(runs(ds, 'late')) - st.mean(lr))
        if runs(ds, 'early'):
            V[f'{ds}_rand_late_minus_early'] = sgn(st.mean(lr) - mean(runs(ds, 'early')))
            V[f'{ds}_rand_late_above_early'] = str(sum(v > mean(runs(ds, 'early')) for v in lr))
T['random'] = md(rows)
V['random_done'] = ', '.join(DSN[d] for d in have_random) or 'none'

# ── column order (early fusion on 5 random feature orders) ───────────────────
for ds in CONF:
    P = [load(ds, f'early_perm{r}', r) for r in range(5)]
    if all(P) and runs(ds, 'early'):
        x = [r['macro_f1'] for r in P]
        V[f'{ds}_perm_mean'], V[f'{ds}_perm_range'] = f3(st.mean(x)), f'{f3(min(x))}–{f3(max(x))}'
        V[f'{ds}_perm_minus_early'] = sgn(st.mean(x) - mean(runs(ds, 'early')))
        V[f'{ds}_perm_sd'], V[f'{ds}_early_sd_seeds'] = f3(st.stdev(x)), f3(sd(runs(ds, 'early')))

# ── latency (bench_latency.py, CPU, isolated) ────────────────────────────────
lat = RES / 'latency.json'
if lat.exists():
    L = json.loads(lat.read_text())
    rows = [['Model'] + [f'{DSN[ds]} (ms/record)' for ds in DSS] + [f'{DSN[ds]} (ms / 1,000 records)' for ds in DSS]]
    for a in ('s1', 's2', 'early', 'late', 'attn', 'early_wide', 'early_head', 'late_all', 'm_early', 'm_late', 'rf', 'gb'):
        if all(a in L.get(ds, {}) for ds in DSS):
            rows.append([ARCH_NAME[a]] + [f"{L[ds][a]['single_ms']:.2f}" for ds in DSS] + [f"{L[ds][a]['batch1000_ms']:.1f}" for ds in DSS])
            for ds in DSS:
                V[f'{ds}_{a}_lat'], V[f'{ds}_{a}_lat_batch'] = f"{L[ds][a]['single_ms']:.2f}", f"{L[ds][a]['batch1000_ms']:.1f}"
    T['latency'] = md(rows)
    V['lat_cpu'] = L.get('cpu', '')
    for ds in DSS:
        for a in ('rf', 'gb'):
            mb = L[ds][a]['size_mb'] * 2 ** 20 / 1e6      # bench_latency stores MiB; the text uses MB = 10^6 bytes
            V[f'{ds}_{a}_size_mb'] = f"{mb:.0f}" if mb >= 10 else f"{mb:.1f}"
        for a in ('early', 'late'):
            V[f'{ds}_{a}_size_kb'] = f"{L[ds][a]['params'] * 4 / 1000:.0f}"    # float32 weights, kB = 1000 bytes
        V[f'{ds}_rf_nodes'] = f"{L[ds]['rf']['nodes']:,}"
        V[f'{ds}_rf_over_late_single'] = f"{L[ds]['rf']['single_ms'] / L[ds]['late']['single_ms']:.1f}"

# ── runs reused from results_v3 (stopped early before the old 60-epoch cap) ─────
import re as _re
REUSE = {}
for ds in DSS:
    for p in RES.glob(f'{ds}/*/seed*.json'):
        exp = p.parent.name
        kind = 'main' if exp in CNN else 'rand' if _re.fullmatch(r'(s1|s2|late|attn)_rand\d', exp) else None
        if kind:
            n = REUSE.setdefault((ds, kind), [0, 0])
            n[0] += 'reused_from' in json.loads(p.read_text()); n[1] += 1
V['n_reused_rand'] = str(sum(v[0] for (ds, k), v in REUSE.items() if k == 'rand'))
V['n_rand_net'] = str(sum(v[1] for (ds, k), v in REUSE.items() if k == 'rand'))
V['n_reused_main'] = str(sum(v[0] for (ds, k), v in REUSE.items() if k == 'main'))
V['n_reused_main_ds'] = ', '.join(DSN[ds] for (ds, k), v in sorted(REUSE.items()) if k == 'main' and v[0])
# reuse per random-split architecture (TON-IoT and UNSW-NB15, the datasets of the semantic-vs-random inference)
for arch in ('s1', 's2', 'late', 'attn'):
    n_re = n_all = 0
    for ds in CONF:
        for r in range(N_RANDOM):
            rr = load(ds, f'{arch}_rand{r}', r % 5)
            if rr is not None:
                n_all += 1; n_re += 'reused_from' in rr
    V[f'n_reused_rand_{arch}_conf'], V[f'n_rand_{arch}_conf'] = str(n_re), str(n_all)
_lr = [load(ds, f'late_rand{r}', r % 5) for ds in DSS for r in range(N_RANDOM)]
V['n_reused_rand_late_all'], V['n_rand_late_all'] = str(sum('reused_from' in r for r in _lr)), str(len(_lr))
V['late_main_new_conf'] = str(all('reused_from' not in load(ds, 'late', s) for ds in CONF for s in SEEDS))

# ── round-2 review additions ─────────────────────────────────────────────────
# (a) how much do same-seed reruns differ? v3 runs that stopped early before the 60-epoch cap, rerun in v4
same_ep, dF, dF_ds = 0, [], {}
for ds in DSS:
    for p in RES.glob(f'{ds}/*/seed*.json'):
        r4 = json.loads(p.read_text())
        q = RES_Q / ds / p.parent.name / p.name
        if 'reused_from' in r4 or not q.exists() or 'epochs' not in r4:
            continue
        r3 = json.loads(q.read_text())
        if r3.get('epochs', 99) >= 60 or r3.get('best_epoch') is None:
            continue
        same_ep += r3['best_epoch'] == r4['best_epoch']; dF.append(abs(r3['macro_f1'] - r4['macro_f1']))
        dF_ds.setdefault(ds, []).append(abs(r3['macro_f1'] - r4['macro_f1']))
V['rerun_n'], V['rerun_same_epoch'] = str(len(dF)), str(same_ep)
V['rerun_dF_mean'], V['rerun_dF_max'] = f3(st.mean(dF)), f3(max(dF))
for ds, x in dF_ds.items():      # per dataset (round-3 m8)
    V[f'{ds}_rerun_n'], V[f'{ds}_rerun_dF_mean'], V[f'{ds}_rerun_dF_max'] = str(len(x)), f3(st.mean(x)), f3(max(x))
# (b) UNSW-NB15 main experiment under the first-run protocol (all of its runs stopped early): an independent replicate
for a in ('late', 'attn'):
    A, B = runs('UNSW_NB15', a, RES_Q), runs('UNSW_NB15', 'early', RES_Q)
    if A and B:
        put(f'UNSW_NB15_v3_d_{a}_early', tci([x['macro_f1'] - y['macro_f1'] for x, y in zip(A, B)]))
# (c) resolution: half-width of the 95% CI of the paired family comparisons
for ds in CONF:
    hw = []
    for a, b in FAMILY:
        P = paired(ds, a, b)
        if P:
            hw.append((P['hi'] - P['lo']) / 2)
    V[f'{ds}_ci_halfwidth'] = f3(st.median(hw))
# (d) RF minus late fusion on the test records without a training match
for ds in CONF:
    if all('clean_macro_f1' in r for r in runs(ds, 'rf') + runs(ds, 'late')):
        V[f'{ds}_d_rf_late_clean'] = sgn(mean(runs(ds, 'rf'), 'clean_macro_f1') - mean(runs(ds, 'late'), 'clean_macro_f1'))
# (e) where UNSW-NB15 benign false positives go (pooled over seeds)
cls = DATA['UNSW_NB15']['classes']; y = np.load(RES / 'UNSW_NB15' / 'yte_split0.npy'); b = cls.index('normal')
for a in ('late', 'rf'):
    C = np.zeros(len(cls))
    for s_ in SEEDS:
        C += np.bincount(np.load(res_for(a) / 'UNSW_NB15' / a / f'seed{s_}_pred.npy')[y == b], minlength=len(cls))
    C[b] = 0
    V[f'UNSW_NB15_{a}_fp_fuzzers'] = f"{100 * C[cls.index('fuzzers')] / C.sum():.0f}%"
# (f) attention: in how many seeds does the class with the lowest mean weight have the lowest weight
for ds in DSS:
    A = runs(ds, 'attn')
    if A:
        c0 = min(A[0]['attn_s1_per_class'], key=lambda c: st.mean(r['attn_s1_per_class'][c] for r in A))
        V[f'{ds}_attn_class_min_seeds'] = str(sum(min(r['attn_s1_per_class'], key=r['attn_s1_per_class'].get) == c0 for r in A))
# (g) compact forests on the same features and split (companion accuracy-footprint study, experiments/baseline/results_v2)
P1R = PIPE / 'experiments' / 'baseline' / 'results_v2'
for ds in CONF:
    for cfg in ('rf_t25_d12', 'rf_t100_d16'):
        R = [json.loads(f.read_text()) for f in sorted((P1R / ds / cfg).glob('seed*.json'))]
        if len(R) == 5:
            V[f'{ds}_{cfg}_f1'] = f3(mean(R)); V[f'{ds}_{cfg}_mb'] = f"{st.mean(r['size_bytes'] for r in R) / 1e6:.1f}"
            if cfg == 'rf_t25_d12' and lat.exists():     # size ratio to the float32 late-fusion weights
                V[f'{ds}_rf_t25_over_late'] = f"{st.mean(r['size_bytes'] for r in R) / (L[ds]['late']['params'] * 4):.0f}"
# (h) RFECV cross-validated macro-F1 from the fitted pipelines (Section 3.1)
import sys as _sys, joblib
_sys.path.insert(0, str(PIPE))
# The fitted pipelines (13 MB each) are not in the public repository: their RFECV scores are cached in rfecv_scores.json.
RFECV_CACHE = HERE / 'rfecv_scores.json'
_rc = json.loads(RFECV_CACHE.read_text()) if RFECV_CACHE.exists() else {}
for ds in CONF:
    pk = PIPE / 'experiments' / 'baseline' / 'results' / ds / 'pipeline.pkl'
    if pk.exists():
        rf_ = joblib.load(pk).rfecv_
        _rc[ds] = dict(scores=[float(x) for x in rf_.cv_results_['mean_test_score']],
                       n_in=int(rf_.n_features_in_), n_sel=int(rf_.n_features_))
    sc = np.asarray(_rc[ds]['scores'])
    V[f'{ds}_rfecv_best'], V[f'{ds}_rfecv_all'] = f3(sc.max()), f3(sc[-1])
    V[f'{ds}_rfecv_n_in'], V[f'{ds}_rfecv_n_sel'] = str(_rc[ds]['n_in']), str(_rc[ds]['n_sel'])
RFECV_CACHE.write_text(json.dumps(_rc, indent=1))
# (i) attack detection rate: minimum over the models of Table 2; GBDT iterations on UNSW-NB15
for ds in CONF:     # rounded DOWN, because the text says "at least"
    _dm = min(mean(runs(ds, a), 'attack_detection') for a in ORDER if runs(ds, a))
    V[f'{ds}_det_min'] = f'{np.floor(_dm * 1000) / 1000:.3f}'
    V[f'{ds}_det_min_cnn'] = f'{np.floor(min(mean(runs(ds, a), "attack_detection") for a in CNN) * 1000) / 1000:.3f}'
it = [r.get('n_iter') for r in runs('UNSW_NB15', 'gb')]
if all(it):
    V['UNSW_NB15_gb_n_iter_range'] = f'{min(it):,}–{max(it):,}' if min(it) != max(it) else f'{it[0]:,}'
# ── round-3 review additions ─────────────────────────────────────────────────
# (j) test records that coincide with a training record on the selected features (referee M6), from the saved
#     per-record predictions and the clean mask of run_p6_v2.py. The label of the matching TRAINING record was not
#     saved, so the share of matches with a different label cannot be computed from the result files.
from sklearn.metrics import f1_score
DUPM = ('early', 'late', 'attn', 'early_head', 'late_all', 'rf', 'gb')
rows = [['Dataset', 'Model', 'Macro-F1, all test records', 'Macro-F1, without a training match', 'Accuracy on records with a training match',
         'Accuracy on records without a match']]
for ds in CONF:
    cm = np.load(RES / ds / 'clean_split0.npy'); y = np.load(RES / ds / 'yte_split0.npy'); dup = ~cm
    cls = DATA[ds]['classes']
    V[f'{ds}_dup_n'] = f'{int(dup.sum()):,}'
    comp = np.bincount(y[dup], minlength=len(cls))
    V[f'{ds}_dup_benign_share'] = f"{100 * comp[cls.index('normal')] / dup.sum():.0f}%"
    order_c = np.argsort(-comp)
    V[f'{ds}_dup_top_classes'] = ', '.join(f"{cls[i]} {100 * comp[i] / dup.sum():.0f}%" for i in order_c[:3])
    rate = {c: comp[i] / max(1, np.bincount(y, minlength=len(cls))[i]) for i, c in enumerate(cls)}
    for c in cls:
        V[f'{ds}_dup_rate_{c}'] = f'{100 * rate[c]:.0f}%'
    acc_dup, acc_cl = {}, {}
    for a in DUPM:
        P_ = [np.load(res_for(a) / ds / a / f'seed{s}_pred.npy') for s in SEEDS]
        acc_dup[a] = [float((p[dup] == y[dup]).mean()) for p in P_]
        acc_cl[a] = [float((p[cm] == y[cm]).mean()) for p in P_]
        f1d = [f1_score(y[dup], p[dup], average='macro', labels=sorted(set(y[dup]))) for p in P_]
        V[f'{ds}_{a}_dup_acc'], V[f'{ds}_{a}_clean_acc'] = f3(st.mean(acc_dup[a])), f3(st.mean(acc_cl[a]))
        V[f'{ds}_{a}_dup_f1'] = f3(st.mean(f1d))
        rows.append([DSN[ds], ARCH_NAME[a].replace(' (CNN, all features)', ''), f3(mean(runs(ds, a))),
                     f3(mean(runs(ds, a), 'clean_macro_f1')), f3(st.mean(acc_dup[a])), f3(st.mean(acc_cl[a]))])
    _ad, _ac = [st.mean(v) for v in acc_dup.values()], [st.mean(v) for v in acc_cl.values()]
    V[f'{ds}_dup_acc_range'], V[f'{ds}_clean_acc_range'] = f'{f3(min(_ad))}–{f3(max(_ad))}', f'{f3(min(_ac))}–{f3(max(_ac))}'
    V[f'{ds}_d_rf_late_dup_acc'] = sgn(st.mean(acc_dup['rf']) - st.mean(acc_dup['late']))
    V[f'{ds}_d_rf_late_clean_acc'] = sgn(st.mean(acc_cl['rf']) - st.mean(acc_cl['late']))
T['dup'] = md(rows)
# (k) PR-AUC and attack detection rate for every model of Table 2 (referee m3)
rows = [['Model'] + [f'{DSN[ds]} {m}' for ds in CONF for m in ('macro-F1', 'PR-AUC', 'detection rate', 'FPR')]]
for a in ORDER:
    if not all(runs(ds, a) for ds in CONF):
        continue
    cells = []
    for ds in CONF:
        R = runs(ds, a)
        cells += [f3(mean(R)), f3(mean(R, 'macro_pr_auc')) if all('macro_pr_auc' in r for r in R) else '–',
                  f3(mean(R, 'attack_detection')), f3(mean(R, 'benign_fpr'))]
    rows.append([ARCH_NAME[a]] + cells)
T['metrics'] = md(rows)
# (l) per-split attention − early fusion differences (text gives the mean and this range)
for ds in CONF:
    a0 = [mean(runs(ds, 'attn')) - mean(runs(ds, 'early'))] + [load(ds, f'attn_split{k}', k)['macro_f1'] - load(ds, f'early_split{k}', k)['macro_f1'] for k in RESPLITS]
    V[f'{ds}_splitd_attn_early_minmax'] = f'{sgn(min(a0))} to {sgn(max(a0))}'

# negative numbers get a true minus sign everywhere
for k, v in list(V.items()):
    if isinstance(v, str) and _re.match(r'^-\d', v):
        V[k] = '−' + v[1:]

V['tables'] = T
out = HERE / ('values.json' if RES.name == 'results_v4' else f'values_{RES.name}.json')
out.write_text(json.dumps(V, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'{RES.name}: {len(V) - 1} values, {len(T)} tables -> {out.name}; random splits: {V["random_done"]}')
print('Holm family', m_, 'significant:', V['holm_sig_list'])
