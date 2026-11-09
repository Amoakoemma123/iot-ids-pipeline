"""Paper 3 v2: every number quoted in the manuscript, computed from the stored run files.

Source: Papers pipeline/experiments/paper3_continual/results_v3 (per-run JSONs with per-stage confusion matrices)
and results_v2 (the invalid earlier protocol, quoted only in Section 5.6).
Output: values.json (scalars and markdown tables, keys used as {{key}} / {{table:key}} in Manuscript.md).
Run analyze_cl_v2.py first only if you want summary.json; this script recomputes everything itself.
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1] / 'experiments' / 'paper3_continual'   # repository root (authors' tree: <project>/Papers pipeline)
sys.path.insert(0, str(EXP.parents[1]))
from experiments.paper3_continual.analyze_cl_v2 import holm, metrics  # noqa: E402

RES, RES2 = EXP / 'results_v3', EXP / 'results_v2'
# final review round (2026-11-07): corrected A-GEM ('agem2'), replay-weighting ablation ('er2000p'), random forest on
# the replay memory ('rf_er2000', 'rf_joint') and the range-share diagnostic, all in results_p3final (queue
# logs/p3final_queue.sh). Every number that depends on these runs gets a key with the prefix P3F_, created only when
# the runs are complete; until then build.py leaves ⟦P3F_...⟧ visible and verify.py reports it as pending.
# The flawed A-GEM runs in results_v3 ('agem': reference loss with zero weight on earlier attack classes) are never
# reported: until the corrected runs exist, A-GEM is excluded from every aggregate and its table cells are pending.
RESF = Path(os.environ.get('P3F_RESULTS', EXP / 'results_p3final'))   # override only for plumbing tests
OUT_JSON = Path(os.environ.get('P3_VALUES_OUT', HERE / 'values.json'))
DSETS = ('TON_IoT', 'UNSW_NB15', 'N_BaIoT')
DNAME = {'TON_IoT': 'TON-IoT', 'UNSW_NB15': 'UNSW-NB15', 'N_BaIoT': 'N-BaIoT'}
DSHORT = {'TON_IoT': 'TON', 'UNSW_NB15': 'UNSW', 'N_BaIoT': 'NB'}
SHORTNAME = {'TON': 'TON-IoT', 'UNSW': 'UNSW-NB15', 'NB': 'N-BaIoT'}
STRATS_ALL = ('naive', 'ewc', 'lwf', 'agem', 'er500', 'er2000', 'joint')
PENDING_CELL = '⟦P3F_agem⟧'
SNAME = {'naive': 'Fine-tuning', 'ewc': 'Online EWC', 'lwf': 'LwF', 'agem': 'A-GEM', 'er500': 'ER-500',
         'er2000': 'ER-2000', 'joint': 'Joint retraining'}
CNAME = {'normal': 'benign', 'ddos': 'DDoS', 'dos': 'DoS', 'mitm': 'MITM', 'xss': 'XSS', 'dos': 'DoS'}
V = {}


def f3(x):
    return f'{x:.3f}'


def load(ds, group, res=RES):
    out = {}
    for p in sorted((res / ds / group).glob('*.json')):
        r = json.loads(p.read_text())
        out.setdefault(r['strategy'], {})[(r['order'], r['seed'])] = (r, metrics(r))
    return out


def ms(vals):
    vals = np.asarray(vals, float)
    return vals.mean(), (vals.std(ddof=1) if len(vals) > 1 else 0.0)


def rf_f1(ds):
    from sklearn.metrics import f1_score
    vals = []
    for p in sorted((RES / ds / 'rf').glob('*.json')):
        cm = np.array(json.loads(p.read_text())['cm'])
        y = np.repeat(np.arange(len(cm)), cm.sum(1))
        yp = np.concatenate([np.repeat(np.arange(len(cm)), row) for row in cm])
        vals.append(f1_score(y, yp, average='macro'))
    return np.array(vals)


def cname(c):
    return CNAME.get(c, c.replace('_', ' '))


def names(ds_short):
    """['TON', 'NB'] -> 'TON-IoT and N-BaIoT'."""
    n = [SHORTNAME[x] for x in ds_short]
    return n[0] if len(n) == 1 else ', '.join(n[:-1]) + ' and ' + n[-1]


def load_final(ds, group, strategy, n):
    """Runs of one strategy from results_p3final, or None if fewer than n exist."""
    G = load(ds, group, RESF) if (RESF / ds / group).exists() else {}
    runs = G.get(strategy, {})
    return runs if len(runs) >= n else None


def pct(x):
    """Whole-number percentage without a '-0'."""
    return f'{int(round(x))}'


def pfmt(p):
    return 'n/a' if not np.isfinite(p) else (f'{p:.2g}' if p >= 1e-3 else '< 0.001')


def ci95(x):
    x = np.asarray(x, float)
    lo, hi = stats.t.interval(0.95, len(x) - 1, x.mean(), stats.sem(x))
    return f'[{lo:+.3f}, {hi:+.3f}]'


AGEM_FIX = {ds: (load_final(ds, 'main', 'agem2', 5), load_final(ds, 'orders', 'agem2', 4)) for ds in DSETS}
AG_DONE = all(m is not None and o is not None for m, o in AGEM_FIX.values())
STRATS = STRATS_ALL if AG_DONE else tuple(s for s in STRATS_ALL if s != 'agem')
NONREPLAY = tuple(s for s in ('naive', 'ewc', 'lwf', 'agem') if s in STRATS)
if AG_DONE:
    V['P3F_agem_done'] = 'yes'

main_rows, task_rows, order_rows, oracle_rows, zd_rows = [], [], [], [], []
GALL, agem_beats = {}, []
for ds in DSETS:
    d = ds.split('_')[0] if ds != 'N_BaIoT' else 'NB'
    G = load(ds, 'main')
    # flawed A-GEM (results_v3) kept only under {d}_agemv1_* for the response letter; corrected runs replace it
    Gv1 = G.pop('agem')
    if AG_DONE:
        G['agem'] = AGEM_FIX[ds][0]
    assert all(len(G[s]) == 5 for s in STRATS), {s: len(G.get(s, {})) for s in STRATS}
    seeds = sorted(G['naive'])
    GALL[d] = (G, seeds)
    # Task-0 identity across strategies
    ident = all(len({round(G[s][k][1]['task0_macro_f1'], 6) for s in STRATS}) == 1 for k in seeds)
    V[f'{d}_task0_identical'] = 'yes' if ident else 'NO'
    V[f'{d}_task0_f1'] = f3(np.mean([G['naive'][k][1]['task0_macro_f1'] for k in seeds]))
    get = lambda s, key: np.array([G[s][k][1][key] for k in seeds])  # noqa: E731

    for s in STRATS:
        m, sd = ms(get(s, 'final_macro_f1'))
        V[f'{d}_{s}_f1'] = f3(m)
        V[f'{d}_{s}_f1_sd'] = f3(sd)
        for key in ('A', 'FM', 'BWT', 'plasticity', 'det_A', 'det_FM', 'det_BWT', 'zero_day_mean'):
            V[f'{d}_{s}_{key}'] = f3(get(s, key).mean())
        fpr = np.array([G[s][k][1]['benign_fpr'] for k in seeds])  # seeds x stages
        V[f'{d}_{s}_fpr_final'] = f3(fpr[:, -1].mean())
        V[f'{d}_{s}_fpr_max'] = f3(fpr.mean(0).max())
        V[f'{d}_{s}_fpr_t1'] = f3(fpr[:, 1].mean())
        # retention of the earlier tasks only (j < T), excluding the task just learned
        V[f'{d}_{s}_old_cls'] = f3(np.mean([np.mean(G[s][k][1]['R'][-1][:-1]) for k in seeds]))
        V[f'{d}_{s}_old_det'] = f3(np.mean([np.mean(G[s][k][1]['Rd'][-1][:-1]) for k in seeds]))
        main_rows.append((ds, s, get(s, 'final_macro_f1'), {k: get(s, k).mean() for k in ('A', 'FM', 'BWT', 'plasticity', 'det_A', 'det_FM')},
                          fpr[:, -1].mean()))
    # retention of each task after the last task, and on the task just learned (plasticity per task)
    for s in STRATS:
        Rm = np.nanmean([G[s][k][1]['R'] for k in seeds], 0)
        for j in range(Rm.shape[0]):
            V[f'{d}_{s}_Rlast_t{j}'] = f3(Rm[-1, j])
    # final per-class recall (semantic order), ER-2000 and joint
    for s in ('er2000', 'joint'):
        rec = []
        for k in seeds:
            cmk = np.array(G[s][k][0]['stages'][-1]['cm'], float); rec.append(np.diag(cmk) / np.maximum(cmk.sum(1), 1))
        for c, v in zip(G[s][seeds[0]][0]['classes'], np.mean(rec, 0)):
            V[f'{d}_{s}_rec_{c}'] = f'{v:.2f}'
    # benign recall of the RF reference (its benign FPR), and its class- and detection-level A at the final stage
    fr, rfA, rfdA = [], [], []
    tasks_idx = [[G['naive'][seeds[0]][0]['classes'].index(c) for c in t if c != 'normal'] for t in G['naive'][seeds[0]][0]['tasks']]
    for p in sorted((RES / ds / 'rf').glob('*.json')):
        r = json.loads(p.read_text()); cmr = np.array(r['cm'], float); n = r['classes'].index('normal')
        fr.append(1 - cmr[n, n] / cmr[n].sum())
        rec = np.diag(cmr) / np.maximum(cmr.sum(1), 1); det = 1 - cmr[:, n] / np.maximum(cmr.sum(1), 1)
        rfA.append(np.mean([np.mean(rec[c]) for c in tasks_idx]))
        rfdA.append(np.mean([np.average(det[c], weights=cmr[c].sum(1)) for c in tasks_idx]))
    V[f'{d}_rf_fpr'] = f3(np.mean(fr))
    V[f'{d}_rf_A'], V[f'{d}_rf_det_A'] = f3(np.mean(rfA)), f3(np.mean(rfdA))
    rf = rf_f1(ds)
    V[f'{d}_rf_f1'] = f3(rf.mean())
    V[f'{d}_rf_f1_sd'] = f3(rf.std(ddof=1))
    main_rows.append((ds, 'rf', rf, dict(A=np.mean(rfA), det_A=np.mean(rfdA)), np.mean(fr)))
    if not AG_DONE:   # pending row for A-GEM, placed where it belongs in the table
        main_rows.insert(len(main_rows) - 1 - (len(STRATS) - STRATS.index('er500')), (ds, 'agem', None, None, None))
    # final replay memory (at most 2,000; fewer where rare classes have fewer records than their quota)
    V[f'{d}_er2000_mem_final'] = f"{G['er2000'][seeds[0]][0]['stages'][-1]['buffer']:,}"
    V[f'{d}_er500_mem_final'] = f"{G['er500'][seeds[0]][0]['stages'][-1]['buffer']:,}"
    # storage: all training data of the sequence (what joint retraining keeps) vs the 2,000-record memory, KiB
    r0_ = G['naive'][seeds[0]][0]
    n_all = sum(s_['n_train'] for s_ in r0_['stages'])
    V[f'{d}_store_all_kib'] = f"{n_all * len(r0_['features']) * 4 / 1024:,.0f}"
    V[f'{d}_store_all_ratio'] = f"{n_all / 2000:.0f}"
    # indicative update cost (shared laptop): median over seeds of the summed training time of Tasks 1-2
    for s in ('er2000', 'joint', 'naive'):
        V[f'{d}_{s}_upd_s'] = f"{np.median([sum(st['train_s'] for st in G[s][k][0]['stages'][1:]) for k in seeds]):.0f}"
    V[f'{d}_joint_rf_gap'] = f3(rf.mean() - get('joint', 'final_macro_f1').mean())

    # confirmatory family: the six strategies vs fine-tuning, paired over seeds, Holm within dataset. Until the corrected
    # A-GEM runs exist, the flawed A-GEM p-value stands in so that the family keeps its size of six (the adjusted
    # p-values of the other five are provisional then; verify.py reports this as pending).
    fam = [s for s in STRATS_ALL if s != 'naive']
    nvf = get('naive', 'final_macro_f1')
    fam_f1 = {s: (get(s, 'final_macro_f1') if s in G else np.array([Gv1[k][1]['final_macro_f1'] for k in seeds])) for s in fam}
    ps = [stats.ttest_rel(fam_f1[s], nvf).pvalue for s in fam]
    for s, p, ph in zip(fam, ps, holm(ps)):
        if s not in G:
            continue
        V[f'{d}_{s}_vs_naive_d'] = f'{np.mean(fam_f1[s] - nvf):+.3f}'
        V[f'{d}_{s}_vs_naive_abs'] = f3(abs(np.mean(fam_f1[s] - nvf)))
        V[f'{d}_{s}_vs_naive_ci'] = ci95(fam_f1[s] - nvf)
        V[f'{d}_{s}_vs_naive_p_raw'] = f'{p:.2g}'
        V[f'{d}_{s}_vs_naive_pholm'] = f'{ph:.2g}' if ph >= 1e-3 else '< 0.001'
        V[f'{d}_{s}_vs_naive_sig'] = 'yes' if ph < 0.05 else 'no'
    if not AG_DONE:
        V['holm_provisional'] = 'yes'
    V[f'{d}_agemv1_f1'] = f3(np.mean([Gv1[k][1]['final_macro_f1'] for k in seeds]))
    V[f'{d}_agemv1_vs_naive_d'] = f"{np.mean(np.array([Gv1[k][1]['final_macro_f1'] for k in seeds]) - nvf):+.3f}"
    # share of the naive-to-joint gap closed by replay
    nv, jt = get('naive', 'final_macro_f1').mean(), get('joint', 'final_macro_f1').mean()
    for s in ('er500', 'er2000', 'ewc', 'lwf', 'agem'):
        if s in G:
            V[f'{d}_{s}_gap_closed'] = pct(100 * (get(s, "final_macro_f1").mean() - nv) / (jt - nv))
    V[f'{d}_er2000_joint_gap'] = f3(jt - get('er2000', 'final_macro_f1').mean())
    V[f'{d}_fpr_all_max'] = f3(max(np.mean([G[s][k][1]['benign_fpr'][t] for k in seeds]) for s in STRATS for t in range(3)))
    V[f'{d}_er2000_vs_er500'] = f'{get("er2000", "final_macro_f1").mean() - get("er500", "final_macro_f1").mean():+.3f}'
    p_buf = stats.ttest_rel(get('er2000', 'final_macro_f1'), get('er500', 'final_macro_f1')).pvalue
    V[f'{d}_er2000_vs_er500_p'] = f'{p_buf:.2g}' if p_buf >= 1e-3 else '< 0.001'
    # gap between class-level and detection-level average retention, fine-tuning
    V[f'{d}_naive_det_minus_cls'] = f3(get('naive', 'det_A').mean() - get('naive', 'A').mean())

    # tasks table: classes, training records (task's own records), test records
    r0 = G['naive'][seeds[0]][0]
    cls = r0['classes']
    cm = np.array(r0['stages'][0]['cm'])
    nte = cm.sum(1)
    for t, (tk, st) in enumerate(zip(r0['tasks'], r0['stages'])):
        atk = [c for c in tk if c != 'normal']
        task_rows.append((ds, t, ', '.join(cname(c) for c in atk), st['n_train'],
                          int(sum(nte[cls.index(c)] for c in atk))))
    V[f'{d}_n_benign_test'] = f"{int(nte[cls.index('normal')]):,}"
    V[f'{d}_n_features'] = str(len(r0['features']))
    V[f'{d}_params'] = f"{r0['params']:,}"
    V[f'{d}_clip_share'] = f"{100 * np.mean([G[s][k][0]['clip_share_te'] for s in STRATS for k in seeds]):.3f}"
    V[f'{d}_buffer_kb_2000'] = f"{2000 * len(r0['features']) * 4 / 1024:.0f}"

    # zero-day: stage 0 (identical across strategies: same Task-0 model) and stage 1 on task 2
    z0 = np.array([G['naive'][k][1]['zero_day'][0][1:] for k in seeds])  # stage 0 on tasks 1, 2
    z1 = {s: np.array([G[s][k][1]['zero_day'][1][2] for k in seeds]) for s in STRATS}
    V[f'{d}_zd_stage0_t1'] = f3(z0[:, 0].mean())
    V[f'{d}_zd_stage0_t2'] = f3(z0[:, 1].mean())
    V[f'{d}_zd_stage0'] = f3(z0.mean())
    V[f'{d}_fpr_t0'] = f3(np.mean([G['naive'][k][1]['benign_fpr'][0] for k in seeds]))
    for s in STRATS:
        V[f'{d}_{s}_zd_stage1'] = f3(z1[s].mean())
    zr = [f'{z1[s].mean():.3f}' for s in STRATS]
    V[f'{d}_zd_stage1_min'] = min(zr)
    V[f'{d}_zd_stage1_max'] = max(zr)
    zd_rows.append((ds, z0[:, 0].mean(), z0[:, 1].mean(), {s: z1[s].mean() for s in STRATS},
                    {s: np.mean([G[s][k][1]['benign_fpr'] for k in seeds], 0) for s in STRATS}))

    # task orders: one seed on each of four random orders
    O = load(ds, 'orders')
    O.pop('agem')
    if AG_DONE:
        O['agem'] = AGEM_FIX[ds][1]
    orders = sorted({o for s in O for o, _ in O[s]})
    V[f'{d}_n_orders'] = str(len(orders))
    best_reg, best_reg_na = [], []
    for o in orders:
        f = {s: np.mean([m['final_macro_f1'] for (oo, _), (_, m) in O[s].items() if oo == o]) for s in STRATS}
        best_reg.append(min(f['er500'], f['er2000']) - max(f[s] for s in NONREPLAY))
        best_reg_na.append(min(f['er500'], f['er2000']) - max(f[s] for s in ('naive', 'ewc', 'lwf')))
        if AG_DONE:
            agem_beats.append(f['agem'] >= min(f['er500'], f['er2000']))
    V[f'{d}_orders_replay_margin_min'] = f3(min(best_reg))
    V[f'{d}_orders_replay_wins'] = f'{sum(b > 0 for b in best_reg)}/{len(orders)}'
    V[f'{d}_orders_replay_wins_noagem'] = f'{sum(b > 0 for b in best_reg_na)}/{len(orders)}'
    V[f'{d}_orders_replay_margin_min_noagem'] = f3(min(best_reg_na))
    for s in STRATS:
        v = [m['final_macro_f1'] for _, (_, m) in O[s].items()]
        V[f'{d}_{s}_orders_f1'] = f3(np.mean(v))
        V[f'{d}_{s}_orders_f1_min'] = f3(np.min(v))
        V[f'{d}_{s}_orders_f1_max'] = f3(np.max(v))
    on, oj = float(V[f'{d}_naive_orders_f1']), float(V[f'{d}_joint_orders_f1'])
    for s in ('er500', 'er2000'):
        V[f'{d}_{s}_orders_gap_closed'] = f"{100 * (float(V[f'{d}_{s}_orders_f1']) - on) / (oj - on):.0f}"
    order_rows.append((ds, {s: [m['final_macro_f1'] for _, (_, m) in O[s].items()] for s in STRATS}))

    # oracle pre-processing ablation (fitted on all training data), paired with main by seed
    OR = load(ds, 'oracle')
    if OR:
        for s in sorted(OR):
            keys = sorted(set(OR[s]) & set(G[s]))
            a = np.array([OR[s][k][1]['final_macro_f1'] for k in keys])
            b = np.array([G[s][k][1]['final_macro_f1'] for k in keys])
            zo = np.array([OR[s][k][1]['zero_day_mean'] for k in keys])
            zm = np.array([G[s][k][1]['zero_day_mean'] for k in keys])
            V[f'{d}_{s}_oracle_d'] = f'{np.mean(a - b):+.3f}'
            V[f'{d}_{s}_oracle_zd_d'] = f'{np.mean(zo - zm):+.3f}'
            V[f'{d}_{s}_oracle_p'] = f'{stats.ttest_rel(a, b).pvalue:.2g}'
            V[f'{d}_{s}_f1_sd'] = f3(np.std(b, ddof=1))
            lo, hi = stats.t.interval(0.95, len(keys) - 1, np.mean(a - b), stats.sem(a - b))
            V[f'{d}_{s}_oracle_ci'] = f'[{lo:+.3f}, {hi:+.3f}]'
            V[f'{d}_{s}_oracle_ci_hw'] = f3((hi - lo) / 2)
            oracle_rows.append((ds, s, len(keys), b.mean(), a.mean(), np.mean(a - b), np.mean(zo - zm)))
        # Holm over the three oracle tests of this dataset (descriptive comparisons; reported next to the raw p)
        o_s = sorted(OR)
        o_p = [float(V[f'{d}_{s}_oracle_p']) for s in o_s]
        for s, ph in zip(o_s, holm([stats.ttest_rel(np.array([OR[s][k][1]['final_macro_f1'] for k in sorted(set(OR[s]) & set(G[s]))]),
                                                    np.array([G[s][k][1]['final_macro_f1'] for k in sorted(set(OR[s]) & set(G[s]))])).pvalue
                                    for s in o_s])):
            V[f'{d}_{s}_oracle_pholm3'] = f'{ph:.2g}'
        V[f'{d}_oracle_n_seeds'] = str(len(keys))
        V[f'{d}_oracle_strats'] = ', '.join(SNAME[s] for s in STRATS if s in OR)

    # ER-2000 with the full (Task-0/joint) budget on every task, paired with main ER-2000 by seed
    BU = load(ds, 'budget')
    if 'er2000' in BU and len(BU['er2000']) == 5:
        fb = BU['er2000']; ks = sorted(set(fb) & set(G['er2000']))
        a = np.array([fb[k][1]['final_macro_f1'] for k in ks]); b = np.array([G['er2000'][k][1]['final_macro_f1'] for k in ks])
        V[f'{d}_er2000fb_f1'] = f3(a.mean())
        V[f'{d}_er2000fb_d'] = f'{np.mean(a - b):+.3f}'
        V[f'{d}_er2000fb_p'] = f'{stats.ttest_rel(a, b).pvalue:.2g}'
        V[f'{d}_er2000fb_Rlast_t2'] = f3(np.mean([fb[k][1]['R'][-1][-1] for k in ks]))
        V[f'{d}_er2000fb_old_cls'] = f3(np.mean([np.mean(fb[k][1]['R'][-1][:-1]) for k in ks]))
        V[f'{d}_er2000_old_cls'] = f3(np.mean([np.mean(G['er2000'][k][1]['R'][-1][:-1]) for k in ks]))
        V[f'{d}_er2000fb_fpr'] = f3(np.mean([fb[k][1]['benign_fpr'][-1] for k in ks]))
        V[f'{d}_er2000fb_gap_closed'] = f'{100 * (a.mean() - nv) / (jt - nv):.0f}'
    # TON-only ablations (2026-11-02): benign-share replay memory and the v2 -> v3 factorial for joint retraining
    BB = load(ds, 'benignbuf')
    if 'er2000b' in BB and len(BB['er2000b']) == 5:
        b = BB['er2000b']; keys = sorted(b)
        for key in ('final_macro_f1', 'A', 'det_A'):
            V[f'{d}_er2000b_{key}'] = f3(np.mean([b[k][1][key] for k in keys]))
        fb = np.array([b[k][1]['benign_fpr'][-1] for k in keys]); fa = np.array([G['er2000'][k][1]['benign_fpr'][-1] for k in keys])
        V[f'{d}_er2000b_fpr_final'] = f3(fb.mean())
        V[f'{d}_er2000b_fpr_d'] = f'{np.mean(fb - fa):+.3f}'
        V[f'{d}_er2000b_fpr_p'] = f'{stats.ttest_rel(fb, fa).pvalue:.2g}'
        fj = np.array([G['joint'][k][1]['benign_fpr'][-1] for k in keys])
        V[f'{d}_er2000b_fpr_excess_removed'] = f'{100 * np.mean(fa - fb) / np.mean(fa - fj):.0f}'
        f1b = np.array([b[k][1]['final_macro_f1'] for k in keys]); f1a = get('er2000', 'final_macro_f1')
        V[f'{d}_er2000b_f1_d'] = f'{np.mean(f1b - f1a):+.3f}'
        V[f'{d}_er2000b_f1_p'] = f'{stats.ttest_rel(f1b, f1a).pvalue:.2g}'
    FA = {}
    for p in sorted((RES / ds / 'factorial').glob('*.json')):
        r = json.loads(p.read_text()); FA.setdefault(r['variant'], []).append(metrics(r)['final_macro_f1'])
    if FA and all(len(v) == 5 for v in FA.values()) and len(FA) == 3:
        FA['asinh_v3'] = list(get('joint', 'final_macro_f1'))
        for v, x in FA.items():
            V[f'{d}_fact_{v}'] = f3(np.mean(x))
            V[f'{d}_fact_{v}_sd'] = f3(np.std(x, ddof=1))
        V[f'{d}_fact_map_effect_v2budget'] = f"{np.mean(FA['asinh_v2']) - np.mean(FA['minmax_v2']):+.3f}"
        V[f'{d}_fact_map_effect_v3budget'] = f"{np.mean(FA['asinh_v3']) - np.mean(FA['minmax_v3']):+.3f}"
        V[f'{d}_fact_budget_effect_asinh'] = f"{np.mean(FA['asinh_v3']) - np.mean(FA['asinh_v2']):+.3f}"
        V[f'{d}_fact_budget_effect_minmax'] = f"{np.mean(FA['minmax_v3']) - np.mean(FA['minmax_v2']):+.3f}"
        # main effects averaged over the other factor
        V[f'{d}_fact_map_effect_avg'] = f"{(np.mean(FA['asinh_v2']) + np.mean(FA['asinh_v3']) - np.mean(FA['minmax_v2']) - np.mean(FA['minmax_v3'])) / 2:+.3f}"
        V[f'{d}_fact_budget_effect_avg'] = f"{(np.mean(FA['asinh_v3']) + np.mean(FA['minmax_v3']) - np.mean(FA['asinh_v2']) - np.mean(FA['minmax_v2'])) / 2:+.3f}"
        # reconstruction check: minmax_v2 against the archived v2 joint runs (same seeds; Task 0 is bit-identical)
        v2 = {json.loads(p.read_text())['seed']: metrics(json.loads(p.read_text()))['final_macro_f1']
              for p in (RES2 / ds / 'main').glob('joint_o0_s*.json')}
        rec = {json.loads(p.read_text())['seed']: metrics(json.loads(p.read_text()))['final_macro_f1']
               for p in (RES / ds / 'factorial').glob('joint_minmax_v2_o0_s*.json')}
        sd = sorted(set(v2) & set(rec))
        if len(sd) == 5:
            a, b = np.array([rec[s] for s in sd]), np.array([v2[s] for s in sd])
            V[f'{d}_fact_vs_v2_d'] = f'{np.mean(a - b):+.3f}'
            V[f'{d}_fact_vs_v2_p'] = f'{stats.ttest_rel(a, b).pvalue:.2g}'
            V[f'{d}_v2_joint_sd'] = f3(np.std(b, ddof=1))
    pre = RES / ds / 'tuned_pre_ext.json'
    if pre.exists():
        V[f'{d}_lwf_alpha_pre_ext'] = f"{json.loads(pre.read_text())['lwf']['value']:g}"

    # tuned regularisation strengths
    tn = json.loads((RES / ds / 'tuned.json').read_text())
    V[f'{d}_ewc_lam'] = f"{tn['ewc']['value']:g}"
    V[f'{d}_lwf_alpha'] = f"{tn['lwf']['value']:g}"
    for a_, sc in tn['lwf']['scores'].items():
        V[f"{d}_lwf_tune_{a_.replace('.', 'p')}"] = f3(sc)
    V[f'{d}_lwf_edge'] = 'yes' if tn['lwf']['edge'] else 'no'
    V[f'{d}_ewc_edge'] = 'yes' if tn['ewc']['edge'] else 'no'

    # earlier invalid protocol (results_v2): min-max inputs without the asinh map, budget 40/5 and 20/5 epochs
    G2 = load(ds, 'main', RES2)
    if G2:
        for s in ('naive', 'joint', 'er2000', 'ewc'):
            if s in G2:
                V[f'{d}_v2_{s}_f1'] = f3(np.mean([m['final_macro_f1'] for _, m in G2[s].values()]))
        if 'joint' in G2:
            V[f'{d}_v2_task0_f1'] = f3(np.mean([m['task0_macro_f1'] for _, m in G2['naive'].values()]))

# ---- final review round: keys waiting for results_p3final (prefix P3F_) ----
D3 = ('TON', 'UNSW', 'NB')
SH2DS = {v: k for k, v in DSHORT.items()}


def fl(k):
    return float(V[k].replace('+', '').replace('%', '').replace(',', ''))


def old_cls(m):
    return float(np.mean(m['R'][-1][:-1]))


def old_det(m):
    return float(np.mean(m['Rd'][-1][:-1]))


if AG_DONE:
    for k in [k for k in V if re.match(r'^(TON|UNSW|NB)_agem_', k)]:
        V['P3F_' + k] = V[k]
    pos = [d for d in D3 if V[f'{d}_agem_vs_naive_sig'] == 'yes' and fl(f'{d}_agem_vs_naive_d') > 0]
    neg = [d for d in D3 if V[f'{d}_agem_vs_naive_sig'] == 'yes' and fl(f'{d}_agem_vs_naive_d') < 0]
    gaps = {d: fl(f'{d}_agem_gap_closed') for d in D3}
    er_gaps = {d: fl(f'{d}_er2000_gap_closed') for d in D3}
    if not pos and not neg:
        V['P3F_agem_verdict_short'] = 'did not significantly improve on it either'
    elif pos and not neg:
        V['P3F_agem_verdict_short'] = (f'improved on it significantly on {names(pos)}, closing at most '
                                      f'{max(gaps[d] for d in pos):.0f}% of the gap')
    else:
        V['P3F_agem_verdict_short'] = (f'improved on it significantly on {names(pos)} and was worse on {names(neg)}' if pos
                                      else f'was significantly worse than it on {names(neg)}')
    V['P3F_agem_mechanism'] = (
        'With the same memory, using it as training data closed far more of the gap than using it only to constrain '
        'the gradient direction.' if max(gaps.values()) < 0.5 * min(er_gaps.values()) else
        'Used only to constrain the gradient direction, the memory recovered a sizeable part of what replay recovers.')
    V['P3F_fig1_agem'] = ' A-GEM is the corrected version (Section 4.2).'
    V['P3F_ranges_note'] = 'Ranges cover all seven strategies.'
    n_beat = sum(agem_beats)
    V['P3F_orders_nonreplay'] = 'every non-replay strategy' if n_beat == 0 else 'fine-tuning, online EWC and LwF'
    V['P3F_orders_agem_sentence'] = ('Replay also beat A-GEM in every order.' if n_beat == 0 else
                                     f'A-GEM matched or beat the weaker replay variant in {n_beat} of the {len(agem_beats)} '
                                     'dataset–order combinations.')

# replay-weighting ablation: ER-2000 with the memory added once to the training pool (no 1:1 mixing)
RP = {d: load_final(SH2DS[d], 'replayabl', 'er2000p', 5) for d in D3}
if all(v is not None for v in RP.values()):
    shrunk, moved, persists = [], [], []
    for d in D3:
        G, seeds = GALL[d]; P = RP[d]; ks = sorted(set(P) & set(G['er2000']))
        a = np.array([P[k][1]['final_macro_f1'] for k in ks]); b = np.array([G['er2000'][k][1]['final_macro_f1'] for k in ks])
        V[f'P3F_{d}_er2000p_f1'], V[f'P3F_{d}_er2000p_f1_sd'] = f3(a.mean()), f3(a.std(ddof=1))
        V[f'P3F_{d}_er2000p_d'] = f'{np.mean(a - b):+.3f}'
        V[f'P3F_{d}_er2000p_ci'] = ci95(a - b)
        V[f'P3F_{d}_er2000p_p'] = pfmt(stats.ttest_rel(a, b).pvalue)
        Rm = np.nanmean([P[k][1]['R'] for k in ks], 0)
        for j in range(3):
            V[f'P3F_{d}_er2000p_Rlast_t{j}'] = f3(Rm[-1, j])
        V[f'P3F_{d}_er2000p_old_cls'] = f3(np.mean([old_cls(P[k][1]) for k in ks]))
        V[f'P3F_{d}_er2000p_old_det'] = f3(np.mean([old_det(P[k][1]) for k in ks]))
        V[f'P3F_{d}_er2000p_plasticity'] = f3(np.mean([P[k][1]['plasticity'] for k in ks]))
        V[f'P3F_{d}_er2000p_fpr_final'] = f3(np.mean([P[k][1]['benign_fpr'][-1] for k in ks]))
        fa = np.array([P[k][1]['benign_fpr'][-1] for k in ks]); fb = np.array([G['er2000'][k][1]['benign_fpr'][-1] for k in ks])
        V[f'P3F_{d}_er2000p_fpr_d'] = f'{np.mean(fa - fb):+.3f}'
        V[f'P3F_{d}_er2000p_fpr_p'] = pfmt(stats.ttest_rel(fa, fb).pvalue)
        nv = np.mean([G['naive'][k][1]['final_macro_f1'] for k in seeds]); jt = np.mean([G['joint'][k][1]['final_macro_f1'] for k in seeds])
        V[f'P3F_{d}_er2000p_gap_closed'] = pct(100 * (a.mean() - nv) / (jt - nv))
        sf_er = fl(f'{d}_joint_Rlast_t2') - fl(f'{d}_er2000_Rlast_t2')
        sf_p = fl(f'{d}_joint_Rlast_t2') - Rm[-1, 2]
        early_p = max(fl(f'{d}_joint_Rlast_t{j}') - Rm[-1, j] for j in (0, 1))
        V[f'P3F_{d}_er2000p_shortfall_t2'] = f3(sf_p)
        V[f'{d}_er2000_shortfall_t2'] = f3(sf_er)
        if sf_p < 0.5 * sf_er:
            shrunk.append(d)
        if early_p >= sf_p:
            moved.append(d)
        if sf_p > early_p and sf_p >= 0.5 * sf_er:
            persists.append(d)
        fp, fe = fl(f'P3F_{d}_er2000p_fpr_final'), fl(f'{d}_er2000_fpr_final')
        V[f'P3F_{d}_er2000p_fpr_dir'] = 'below' if fp < fe - 0.01 else ('above' if fp > fe + 0.01 else 'close to')
    if len(persists) == 3:
        V['P3F_er2000p_shortfall_verdict'] = ('The shortfall stayed on the newest task on all three datasets, so it does not '
                                              'depend on the double weighting.')
    else:
        parts = []
        if shrunk:
            parts.append(f'the newest-task shortfall shrank by more than half on {names(shrunk)}')
        if moved:
            parts.append(f'at least one earlier task now fell as far below the joint level as the newest task, or further, on {names(moved)}')
        s_ = '; '.join(parts) if parts else 'the pattern changed'
        V['P3F_er2000p_shortfall_verdict'] = (s_[0].upper() + s_[1:] + ', so the location of the shortfall depends partly on '
                                              'the double weighting.')
    V['P3F_replayabl_done'] = 'yes'

# random forest retrained after every task on the ER-2000 memory plus the new task, and the all-data forest per stage
RM = {d: load_final(SH2DS[d], 'rfmem', 'rf_er2000', 5) for d in D3}
RJ = {d: load_final(SH2DS[d], 'rfmem', 'rf_joint', 5) for d in D3}
if all(v is not None for v in list(RM.values()) + list(RJ.values())):
    higher, lower = [], []
    for d in D3:
        G, seeds = GALL[d]; M = RM[d]; ks = sorted(set(M) & set(G['er2000']))
        a = np.array([M[k][1]['final_macro_f1'] for k in ks]); b = np.array([G['er2000'][k][1]['final_macro_f1'] for k in ks])
        V[f'P3F_{d}_rfmem_f1'], V[f'P3F_{d}_rfmem_f1_sd'] = f3(a.mean()), f3(a.std(ddof=1))
        V[f'P3F_{d}_rfmem_vs_er2000_d'] = f'{np.mean(a - b):+.3f}'
        V[f'P3F_{d}_rfmem_vs_er2000_ci'] = ci95(a - b)
        p_ = stats.ttest_rel(a, b).pvalue
        V[f'P3F_{d}_rfmem_vs_er2000_p'] = pfmt(p_)
        (higher if np.mean(a - b) > 0 and p_ < 0.05 else lower if np.mean(a - b) < 0 and p_ < 0.05 else []).append(d)
        V[f'P3F_{d}_rfmem_old_cls'] = f3(np.mean([old_cls(M[k][1]) for k in ks]))
        V[f'P3F_{d}_rfmem_old_det'] = f3(np.mean([old_det(M[k][1]) for k in ks]))
        V[f'P3F_{d}_rfmem_Rlast_t2'] = f3(np.mean([M[k][1]['R'][-1][-1] for k in ks]))
        V[f'P3F_{d}_rfmem_fpr_final'] = f3(np.mean([M[k][1]['benign_fpr'][-1] for k in ks]))
        V[f'P3F_{d}_rfmem_train_s'] = f"{np.median([M[k][1]['train_s'] for k in ks]):.0f}"
        J = RJ[d]; kj = sorted(J)
        for t in range(3):
            V[f'P3F_{d}_rfjoint_f1_t{t}'] = f3(np.mean([J[k][1]['macro_f1_stage'][t] for k in kj]))
        V[f'P3F_{d}_rfjoint_FM'] = f3(np.mean([J[k][1]['FM'] for k in kj]))
        V[f'P3F_{d}_rfjoint_last_equals_rf'] = 'yes' if abs(np.mean([J[k][1]['final_macro_f1'] for k in kj]) - fl(f'{d}_rf_f1')) < 6e-4 else 'NO'
    # phrase completing "A random forest retrained on the same memory was ..." (paired over seeds, uncorrected p < 0.05)
    lab = []
    if higher:
        lab.append(f'better than the CNN with replay on {names(higher)}')
    if lower:
        lab.append(f'worse on {names(lower)}' if higher else f'worse than the CNN with replay on {names(lower)}')
    rest = [d for d in D3 if d not in higher + lower]
    if rest:
        lab.append(f'not significantly different on {names(rest)}' if lab else
                   'not significantly different from the CNN with replay on any dataset')
    V['P3F_rfmem_verdict'] = lab[0] if len(lab) == 1 else ', '.join(lab[:-1]) + ' and ' + lab[-1]
    if len(higher) == 3:
        V['P3F_rfmem_advice'] = ('A random forest retrained on the same memory beat the CNN with replay on all three datasets, '
                                 'so where a tree model fits the deployment it is the simpler and stronger choice.')
    elif not higher:
        V['P3F_rfmem_advice'] = ('A random forest retrained on the same memory did not beat the CNN with replay on any dataset, '
                                 'so here the memory is best spent on the network.')
    else:
        V['P3F_rfmem_advice'] = (f'A random forest retrained on the same memory beat the CNN with replay on {names(higher)}, '
                                 'so a tree model on the memory is a serious alternative where it fits the deployment.')
    V['P3F_rfmem_done'] = 'yes'

# range share: min-max values outside the Task-0 range [0, 1], before the compression map
RS = {d: RESF / SH2DS[d] / 'rangeshare' / 'rangeshare.json' for d in D3}
RS = {d: (p if p.exists() else RESF / SH2DS[d] / 'rangeshare.json') for d, p in RS.items()}
if all(p.exists() for p in RS.values()):
    for d, p in RS.items():
        r = json.loads(p.read_text())
        V[f'P3F_{d}_out01_te'] = f"{100 * r['te']['out01']:.2f}"
        V[f'P3F_{d}_out01_rows_te'] = f"{100 * r['te']['rows_any_out01']:.1f}"
        V[f'P3F_{d}_out01_later_tr'] = f"{100 * np.mean(r['tr']['per_task'][1:]):.2f}"
        V[f'P3F_{d}_beyond_clip_te'] = f"{100 * r['te']['beyond_clip']:.3f}"
        if 'n_features_oracle' in r:
            V[f'P3F_{d}_feat_oracle'] = str(r['n_features_oracle'])
            V[f'P3F_{d}_feat_common'] = str(r['n_features_common'])
    V['P3F_rangeshare_done'] = 'yes'


# ---- tables ----
def table_main():
    L = ['| Dataset | Strategy | Final macro-F1 | A (class) | FM (class) | Plasticity (class) | A (detection) | FM (detection) | Benign FPR |',
         '|---|---|---|---|---|---|---|---|---|']
    for ds, s, f, m, fpr in main_rows:
        if f is None:   # A-GEM pending (corrected runs not yet available)
            L.append(f'| {DNAME[ds]} | {SNAME[s]} | ' + ' | '.join([PENDING_CELL] * 7) + ' |')
            continue
        mu, sd = ms(f)
        if s == 'rf':
            L.append(f"| {DNAME[ds]} | Random forest (all data) | {mu:.3f} ± {sd:.3f} | {m['A']:.3f} | | | {m['det_A']:.3f} | | {fpr:.3f} |")
        else:
            L.append(f"| {DNAME[ds]} | {SNAME[s]} | {mu:.3f} ± {sd:.3f} | {m['A']:.3f} | {m['FM']:.3f} | {m['plasticity']:.3f} | "
                     f"{m['det_A']:.3f} | {m['det_FM']:.3f} | {fpr:.3f} |")
    return '\n'.join(L)


def table_tasks():
    L = ['| Dataset | Task | Attack classes | Training records | Attack test records |', '|---|---|---|---|---|']
    for ds, t, c, ntr, nte in task_rows:
        L.append(f'| {DNAME[ds]} | {t} | {c} | {ntr:,} | {nte:,} |')
    return '\n'.join(L)


def table_orders():
    L = ['| Dataset | ' + ' | '.join(SNAME[s] for s in STRATS_ALL) + ' |', '|---' * (len(STRATS_ALL) + 1) + '|']
    for ds, O in order_rows:
        L.append(f'| {DNAME[ds]} | ' + ' | '.join(f'{np.mean(O[s]):.3f} ({np.min(O[s]):.3f}–{np.max(O[s]):.3f})' if s in O
                                                  else PENDING_CELL for s in STRATS_ALL) + ' |')
    return '\n'.join(L)


def table_oracle():
    L = ['| Dataset | Strategy | Task-0 pre-processing | All-data pre-processing | Difference | Not-yet-learned flagging difference |',
         '|---|---|---|---|---|---|']
    for ds, s, n, b, a, dd, dz in oracle_rows:
        L.append(f'| {DNAME[ds]} | {SNAME[s]} | {b:.3f} | {a:.3f} | {dd:+.3f} | {dz:+.3f} |')
    return '\n'.join(L)


def table_zeroday():
    L = ['| Dataset | Task 1 before learning it (after Task 0) | Task 2 after Task 0 | Task 2 after Task 1, range over strategies | '
         'Benign FPR after Task 0 | Benign FPR after Task 2, range over strategies |', '|---|---|---|---|---|---|']
    for ds, a, b, z1, fpr in zd_rows:
        f_last = [fpr[s][-1] for s in STRATS]
        L.append(f'| {DNAME[ds]} | {a:.3f} | {b:.3f} | {min(z1.values()):.3f}–{max(z1.values()):.3f} | '
                 f"{fpr['naive'][0]:.3f} | {min(f_last):.3f}–{max(f_last):.3f} |")
    return '\n'.join(L)


for d in ('TON', 'UNSW', 'NB'):
    f1 = [float(V[f'{d}_{s}_fpr_t1']) for s in STRATS if s != 'agem']
    V[f'{d}_fpr_t1_min_noagem'], V[f'{d}_fpr_t1_max_noagem'] = f3(min(f1)), f3(max(f1))
V['oracle_ci_bound_TU'] = f3(max(abs(float(b)) for d in ('TON', 'UNSW') for s in ('naive', 'er2000', 'joint')
                                 for b in V[f'{d}_{s}_oracle_ci'].strip('[]').split(', ')))
V['oracle_max_abs'] = f3(max(abs(r[5]) for r in oracle_rows))
V['oracle_zd_max_abs'] = f3(max(abs(r[6]) for r in oracle_rows))
V['table:main'] = table_main()
V['table:tasks'] = table_tasks()
V['table:orders'] = table_orders()
V['table:oracle'] = table_oracle()
V['table:zeroday'] = table_zeroday()

if __name__ == '__main__':
    OUT_JSON.write_text(json.dumps(V, indent=1, ensure_ascii=False), encoding='utf-8')
    for k, v in V.items():
        if k.startswith('table:'):
            print(f'\n{k}\n{v}')
    print()
    print(json.dumps({k: v for k, v in V.items() if not k.startswith('table:')}, indent=0, ensure_ascii=False))
