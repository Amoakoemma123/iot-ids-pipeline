"""Paper 2 v3: every number quoted in Manuscript.md, from experiments/paper2_adv_v2/results_v3 (run_adv_v3.py).
Writes values.json ({{key}} scalars, {{table:key}} markdown tables) and asr.json (for make_figures.py).
Missing groups are skipped, so their placeholders stay visible as ⟦key⟧ in a partial build.
Set P2_RES=<path> to analyse another results directory (e.g. results_v3_smoke for a code check).

Result files: results_v3/<ds>/<model>_<defence>[_split<k>|_nottl]/seed<k>.json; attack_set*.json per dataset.
ASR = share of detected attack flows (of the fixed, class-stratified attack set) classified benign after the attack.
Prevalence-weighted ASR re-weights the per-class ASR by the test-set class shares of attack traffic.
Overall evasion = share of ALL attack flows of the attack set that end up classified benign (missed or evaded).
The v2 version of this script is compute_values_v2.py; the v3 version (before review round 2) compute_values_v3_before_r2.py.
Round 2 (run_adv_r2.py, results_r2/): extended union, strong feature-space reference (fs), operating point (op),
physical feasibility (feas), tree saturation (sat). Primary Holm family: see PRIMARY below.
"""
import base64
import json
import os
import statistics as st
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get('P2_RES', HERE.parents[1] / 'experiments' / 'paper2_adv_v2' / 'results_v3'))
SMOKE = 'smoke' in RES.name
DSS = ('TON_IoT', 'UNSW_NB15')
DSN = {'TON_IoT': 'TON-IoT', 'UNSW_NB15': 'UNSW-NB15'}
SEEDS5, SEEDS3 = (0, 1, 2, 3, 4), (0, 1, 2)
if SMOKE:
    SEEDS5 = SEEDS3 = (0,)
BETAS = (0.1, 0.25, 0.5, 1.0, 2.0)
EPS = (0.01, 0.05, 0.1)
BASE = ('cnn_none', 'mlp_none', 'rf_none', 'gb_none')
NETDEF = ('cnn_clean_loop', 'cnn_at_all', 'cnn_ccat', 'cnn_at_threat', 'cnn_trades', 'mlp_at_threat', 'cnn_at_all_bpda')
TREEDEF = ('rf_augment', 'gb_augment', 'rf_augment2', 'gb_augment2', 'rf_augment_nodelay', 'gb_augment_nodelay',
           'rf_augment_nobytes', 'gb_augment_nobytes')
SWEEP = ('cnn_at_all_e0.02', 'cnn_at_all_e0.1', 'cnn_at_threat_b0.25', 'cnn_at_threat_b1.0')   # seed 0 only
DEF = NETDEF + TREEDEF
FIVE = {'cnn_at_threat', 'rf_augment', 'gb_augment', 'cnn_clean_loop', 'cnn_at_all', 'cnn_ccat', 'cnn_trades'} | set(BASE)
NAME = {'cnn_none': 'CNN', 'mlp_none': 'MLP', 'rf_none': 'RF', 'gb_none': 'GBDT',
        'cnn_clean_loop': 'CNN, clean loop', 'cnn_at_all': 'CNN, AT-all', 'cnn_ccat': 'CNN, CCAT',
        'cnn_at_threat': 'CNN, AT-threat', 'cnn_trades': 'CNN, TRADES', 'mlp_at_threat': 'MLP, AT-threat',
        'rf_augment': 'RF, augmented', 'gb_augment': 'GBDT, augmented', 'rf_augment2': 'RF, augmented ×2',
        'gb_augment2': 'GBDT, augmented ×2', 'rf_augment_nodelay': 'RF, augmented w/o delay',
        'gb_augment_nodelay': 'GBDT, augmented w/o delay', 'rf_augment_nobytes': 'RF, augmented w/o bytes',
        'gb_augment_nobytes': 'GBDT, augmented w/o bytes', 'cnn_at_all_bpda': 'CNN, AT-all (BPDA)',
        'cnn_at_all_e0.02': 'CNN, AT-all, ε = 0.02', 'cnn_at_all_e0.1': 'CNN, AT-all, ε = 0.1',
        'cnn_at_threat_b0.25': 'CNN, AT-threat, β = 0.25', 'cnn_at_threat_b1.0': 'CNN, AT-threat, β = 1',
        'cnn_at_threat_cs': 'CNN, AT-threat, corner start'}
UNDEF = {m: m.split('_')[0] + '_none' for m in DEF}
V, T, A = {}, {}, {}
NBOOT = 10000   # review round 3, numbers N1: stable CI bounds


def f3(x):
    return f'{x:.3f}'


def sgn(x, nd=3):
    return f'{0:.{nd}f}' if abs(x) < 0.5 * 10 ** -nd else f'{x:+.{nd}f}'.replace('-', '−')


def pct(x):
    return '–' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{100 * x:.1f}%'


def pp(x):
    return sgn(100 * x, 1) + ' pp'


def pfmt(p):
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def md(rows):
    out = ['| ' + ' | '.join(rows[0]) + ' |', '|' + '|'.join('---' for _ in rows[0]) + '|']
    return '\n'.join(out + ['| ' + ' | '.join(r) + ' |' for r in rows[1:]])


def holm(ps):
    """Holm-adjusted p-values, same order as ps"""
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


def unpack(s, n):
    return np.unpackbits(np.frombuffer(base64.b64decode(s), np.uint8))[:n].astype(bool)


_cache = {}


def runs(ds, m, seeds=None):
    seeds = seeds if seeds is not None else (SEEDS5 if (m.split('_split')[0].split('_nottl')[0] in FIVE and '_split' not in m and '_nottl' not in m) else SEEDS3)
    if '_split' in m:
        seeds = (0,)
    key = (ds, m, tuple(seeds))
    if key not in _cache:
        P = [RES / ds / m / f'seed{s}.json' for s in seeds]
        R = [json.loads(p.read_text()) for p in P] if all(p.exists() for p in P) else None
        if R and ds == 'TON_IoT':   # no TTL action: volume-only = all actions (review round 3, numbers E4)
            for r in R:
                for a in ('strongest_b0.5', 'ev_strongest_b0.5', 'strongest_b0.5_per_class'):
                    if a in r:
                        r[a.replace('strongest_', 'strongest_volume_')] = r[a]
        _cache[key] = R
    return _cache[key]


def mean(R, k):
    v = [r[k] for r in R if r.get(k) is not None]
    return st.mean(v) if len(v) == len(R) and v else None


def sd(R, k):
    v = [r[k] for r in R if r.get(k) is not None]
    return st.stdev(v) if len(v) == len(R) and len(v) > 1 else 0.0


def boot_ci(R, ev_key):
    """95% two-level bootstrap CI of the seed-mean ASR: resample seeds with replacement, then the attacked flows
    within each drawn seed (so the interval includes seed-to-seed variation, review round 2 M7)"""
    if not R or any(ev_key not in r for r in R):
        return None
    E = [unpack(r[ev_key], r['n_detected']) for r in R]
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(NBOOT):
        draw = rng.integers(0, len(E), len(E))
        bs.append(np.mean([E[i][rng.integers(0, len(E[i]), len(E[i]))].mean() for i in draw if len(E[i])]))
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def ci_txt(ci):
    return '–' if ci is None else f'{100 * ci[0]:.1f}–{100 * ci[1]:.1f}'


def overall(r, att):
    """share of all attack flows of the attack set classified benign: missed before the attack, or evaded"""
    return 1 - r['n_detected'] / r['n_attacked'] * (1 - r[att])


def prevalence(r, att_pc, shares):
    """per-class ASR re-weighted by the test-set shares of the attack classes (detected classes only)"""
    pc = r.get(att_pc)
    if not pc:
        return None
    w = {c: shares[c] for c in pc if c in shares}
    tot = sum(w.values())
    return sum(pc[c] * w[c] for c in w) / tot if tot else None


# ── threat model: action table (fixed by the code's feature map) ────────────────
T['actions'] = md([
    ['Dataset', 'Action (attacker)', 'Budget at β', 'Applies to', 'Selected features recomputed'],
    ['TON-IoT', 'add payload bytes', 'β(bytes + 1,000)', 'all flows', 'src_bytes; src_ip_bytes'],
    ['TON-IoT', 'add packets', 'β(packets + 10)', 'all flows', 'src_pkts; src_ip_bytes (+ header bytes per packet)'],
    ['TON-IoT', 'delay packets', 'β(duration + 1 s)', 'flows with ≥ 2 packets', 'duration'],
    ['UNSW-NB15', 'add payload bytes', 'β(bytes + 1,000)', 'all flows', 'sbytes; smean; sload'],
    ['UNSW-NB15', 'add packets', 'β(packets + 10)', 'all flows', 'sbytes (+ header bytes per packet); smean; sload'],
    ['UNSW-NB15', 'delay packets', 'β(duration + 1 s)', 'flows with ≥ 2 packets', 'sload; dload'],
    ['UNSW-NB15', 'delay acknowledgements', 'β(ACK delay + 0.1 s)', 'flows with a handshake (ackdat > 0)', 'ackdat'],
    ['UNSW-NB15', 'choose initial TTL', 'none (any value 1–255)', 'all flows', 'sttl; ct_state_ttl'],
])

# ── attack sets ──────────────────────────────────────────────────────────────
SHARES = {}
for ds in DSS:
    p = RES / ds / 'attack_set.json'
    if not p.exists():
        continue
    a = json.loads(p.read_text())
    V[f'{ds}_n_eval'] = f"{len(a['idx']):,}"
    V[f'{ds}_rare'] = ', '.join(a['rare']) or 'none'
    V[f'{ds}_map_err'] = f"{a['map_err']:.1e}"
    V[f'{ds}_n_feat'] = str(len(a['features']))
    tc = a['test_class_counts']; normal = 'normal' if 'normal' in tc else min(tc, key=lambda c: -tc[c])
    att = {c: n for c, n in tc.items() if c != normal}
    SHARES[ds] = {c: n / sum(att.values()) for c, n in att.items()}
    if a['ttl_cands']:
        V['N_TTL'] = str(len(a['ttl_cands']))
        V['UNSW_ttl_cands'] = ', '.join(f'{t:g}' for t in a['ttl_cands'])
if 'TON_IoT_n_eval' in V and 'UNSW_NB15_n_eval' in V:
    V['N_EVAL'] = V['TON_IoT_n_eval'] if V['TON_IoT_n_eval'] == V['UNSW_NB15_n_eval'] else \
        f"{V['TON_IoT_n_eval']} (TON-IoT) and {V['UNSW_NB15_n_eval']} (UNSW-NB15)"
    V['map_err_max'] = max((V[f'{d}_map_err'] for d in DSS), key=float)

# ── per model: clean performance and every attack (seed means, SD, CI) ─────────
ATT = ([f'bb_volume_b{b}' for b in BETAS] + [f'bb_both_b{b}' for b in BETAS] + [f'wb_action_b{b}' for b in BETAS] +
       ['bb_ttl_only', 'bb_ttl_only_unseen', 'bb_ttl_only_all', 'wbs_action_b0.5', 'wbs_action_b2.0',
        'transfer_action_b0.5', 'transfer_action_b1.0', 'strongest_b0.5', 'strongest_volume_b0.5',
        'bb_dense24_both_b0.5', 'bb_dense24_volume_b0.5', 'bb_grid_plus1000_both_b0.5', 'bb_grid_plus1000_volume_b0.5',
        'ad_none_b0.5', 'ad_cands_b0.5', 'ad_free_b0.5', 'ad_none_b2.0', 'ad_free_b2.0',
        'bb_only_bytes_b0.5', 'bb_only_pkts_b0.5', 'bb_only_delay_b0.5', 'bb_only_bytes_pkts_b0.5'] +
       [f'bb_{w}_b0.5_floor{f}' for w in ('both', 'volume') for f in (0.0, 0.5, 2.0)] +
       [f'bb_unconstrained_e{e}' for e in EPS] + [f'wb_unconstrained_e{e}' for e in EPS] +
       [f'wb_unconstrained_mutable_e{e}' for e in EPS])
CI_KEYS = {'strongest_b0.5': 'ev_strongest_b0.5', 'strongest_volume_b0.5': 'ev_strongest_volume_b0.5',
           'bb_both_b0.5': 'ev_both_b0.5', 'bb_volume_b0.5': 'ev_volume_b0.5'}
VARIANTS = [m + s for m in BASE + DEF for s in ('',)] + [m + '_nottl' for m in ('cnn_none', 'mlp_none', 'rf_none', 'gb_none')] + \
           [m + f'_split{k}' for m in BASE + ('rf_augment', 'gb_augment', 'cnn_at_threat') for k in (1, 2)]
for ds in DSS:
    A[ds] = {}
    for m in VARIANTS:
        R = runs(ds, m)
        if not R:
            continue
        k = f'{ds}_{m}'
        f = [r['macro_f1'] for r in R]
        V[k + '_f1'], V[k + '_sd'] = f3(st.mean(f)), f3(sd(R, 'macro_f1'))
        V[k + '_fpr'] = f3(mean(R, 'benign_fpr'))
        V[k + '_det'] = pct(st.mean(r['n_detected'] / r['n_attacked'] for r in R))
        V[k + '_nseeds'] = str(len(R))
        rare = R[0].get('rare_classes', [])
        if rare:
            V[k + '_rare_f1'] = f3(st.mean(st.mean(r['per_class_f1'][c] for c in rare) for r in R))
        A[ds][m] = {'macro_f1': f, 'det': [r['n_detected'] / r['n_attacked'] for r in R]}
        for att in ATT:
            x = mean(R, att)
            if x is None:
                continue
            V[f'{k}_{att}'] = pct(x)
            V[f'{k}_{att}_sdpp'] = f'{100 * sd(R, att):.1f}'
            A[ds][m][att] = [r[att] for r in R]
        for att in ('strongest_b0.5', 'strongest_volume_b0.5', 'bb_both_b0.5', 'bb_volume_b0.5'):
            if att in A[ds][m]:
                V[f'{k}_{att}_overall'] = pct(st.mean(overall(r, att) for r in R))
                A[ds][m][att + '_overall'] = [overall(r, att) for r in R]
        for att, ev in (CI_KEYS.items() if m in BASE else ()):   # CIs are reported for the undefended models only
            ci = boot_ci(R, ev)
            if ci:
                V[f'{k}_{att}_ci'] = ci_txt(ci)
        if ds in SHARES:
            for att in ('strongest_b0.5', 'strongest_volume_b0.5', 'bb_both_b0.5', 'bb_volume_b0.5'):
                w = [prevalence(r, att + '_per_class', SHARES[ds]) for r in R]
                if all(x is not None for x in w):
                    V[f'{k}_{att}_prev'] = pct(st.mean(w))
        pc = [r.get('strongest_b0.5_per_class') for r in R]
        if all(pc):
            cls = sorted(set.intersection(*(set(p) for p in pc)))
            A[ds][m]['per_class_strongest'] = {c: st.mean(p[c] for p in pc) for c in cls}
        pcv = [r.get('strongest_volume_b0.5_per_class') for r in R]
        if all(pcv):
            cls = sorted(set.intersection(*(set(p) for p in pcv)))
            A[ds][m]['per_class_strongest_volume'] = {c: st.mean(p[c] for p in pcv) for c in cls}
        q = [r.get('queries_first_success_median') for r in R]
        if all(x is not None for x in q):
            V[k + '_queries_median'] = f'{st.median(q) + 1:.0f}'   # zero-based candidate index -> query number
            V[k + '_queries_total'] = str(R[0]['queries_total'])
        if 'ttl_sweep' in R[0]:
            A[ds][m]['ttl_sweep'] = {t: st.mean(r['ttl_sweep'][t] for r in R) for t in R[0]['ttl_sweep']}
        A[ds][m]['eval_s'] = [r.get('eval_s') for r in R]

# ── undefended: clean table and main ASR table (beta 0.5) ──────────────────────
rows = [['Dataset', 'Model', 'Clean macro-F1', 'Rare-class F1', 'Benign FPR', 'Attack flows detected']]
for ds in DSS:
    for m in BASE:
        k = f'{ds}_{m}'
        if k + '_f1' in V:
            rows.append([DSN[ds], NAME[m], f"{V[k + '_f1']} ± {V[k + '_sd']}", V.get(k + '_rare_f1', '–'), V[k + '_fpr'], V[k + '_det']])
T['clean'] = md(rows)


def cell(k, att, sdv=True):
    v = V.get(f'{k}_{att}')
    if v is None:
        return '–'
    return f"{v} ± {V[f'{k}_{att}_sdpp']}" if sdv else v


rows = [['Dataset', 'Model', 'Grid, volume only', 'Grid + adaptive, volume only',
         'Strong white-box PGD', 'Transfer from CNN', 'Strongest, volume only', 'Strongest, all actions [95% CI]']]
for ds in DSS:
    for m in BASE:
        k = f'{ds}_{m}'
        if k + '_f1' not in V:
            continue
        rows.append([DSN[ds], NAME[m], cell(k, 'bb_volume_b0.5'), cell(k, 'ad_none_b0.5'),     # TTL-only and grid-both columns: Table 8 / text (final review round 2)
                     cell(k, 'wbs_action_b0.5') if m in ('cnn_none', 'mlp_none') else 'n/a',
                     cell(k, 'transfer_action_b0.5') if m != 'cnn_none' else 'n/a', cell(k, 'strongest_volume_b0.5'),
                     f"{cell(k, 'strongest_b0.5')} [{V.get(k + '_strongest_b0.5_ci', '–')}]"])
T['asr'] = md(rows)

# end-to-end view: share of all attack flows classified benign, and prevalence-weighted ASR
rows = [['Dataset', 'Model', 'Missed before attack', 'Overall evasion, volume only', 'Overall evasion, all actions',
         'Prevalence-weighted ASR, all actions']]
for ds in DSS:
    for m in BASE:
        k = f'{ds}_{m}'
        if f'{k}_strongest_b0.5_overall' in V:
            rows.append([DSN[ds], NAME[m], pct(1 - st.mean(A[ds][m]['det'])), V[f'{k}_strongest_volume_b0.5_overall'],
                         V[f'{k}_strongest_b0.5_overall'], V.get(f'{k}_strongest_b0.5_prev', '–')])
            V[k + '_miss'] = pct(1 - st.mean(A[ds][m]['det']))
T['overall'] = md(rows)

# search saturation: how much do more queries / adaptive search / white-box add?
rows = [['Dataset', 'Model', 'Grid of 24 points', 'Standard grid', 'Grid + 1,000 points', 'Grid + adaptive (TTL kept)',
         'Grid + adaptive (any TTL)', 'Strongest']]
for ds in DSS:
    for m in BASE:
        k = f'{ds}_{m}'
        if k + '_bb_volume_b0.5' not in V:
            continue
        rows.append([DSN[ds], NAME[m], V.get(k + '_bb_dense24_volume_b0.5', '–'), V[k + '_bb_volume_b0.5'],
                     V.get(k + '_bb_grid_plus1000_volume_b0.5', '–'), V.get(k + '_ad_none_b0.5', '–'),
                     V.get(k + '_ad_free_b0.5', 'n/a') if ds == 'UNSW_NB15' else 'n/a', V.get(k + '_strongest_b0.5', '–')])
T['saturation'] = md(rows)
for ds in DSS:
    gains = []
    for m in BASE:
        a = A[ds].get(m, {})
        if 'bb_grid_plus1000_volume_b0.5' in a:
            gains.append(st.mean(a['bb_grid_plus1000_volume_b0.5']) - st.mean(a['bb_volume_b0.5']))
    if len(gains) == 4:
        V[f'{ds}_dense_gain_max'] = f'{100 * max(gains):.1f} pp'
    g2 = [st.mean(A[ds][m]['ad_none_b0.5']) - st.mean(A[ds][m]['bb_volume_b0.5']) for m in BASE if 'ad_none_b0.5' in A[ds].get(m, {})]
    if len(g2) == 4:
        V[f'{ds}_adaptive_gain_range'] = f'{100 * min(g2):.1f}–{100 * max(g2):.1f} pp'

# per-action ablation and budget floors (beta 0.5, TTL unchanged)
rows = [['Dataset', 'Model', 'Bytes only', 'Packets only', 'Delays only', 'Bytes + packets', 'All (volume only)',
         'Floors × 0', 'Floors × 0.5', 'Floors × 2']]
for ds in DSS:
    for m in BASE:
        k = f'{ds}_{m}'
        if k + '_bb_only_bytes_b0.5' not in V:
            continue
        rows.append([DSN[ds], NAME[m]] + [V.get(f'{k}_bb_only_{a}_b0.5', '–') for a in ('bytes', 'pkts', 'delay', 'bytes_pkts')] +
                    [V[k + '_bb_volume_b0.5']] + [V.get(f'{k}_bb_volume_b0.5_floor{f}', '–') for f in (0.0, 0.5, 2.0)])
T['ablation'] = md(rows)

# cheapest successful action (median over flows, seed 0 = representative; all seeds averaged)
rows = [['Dataset', 'Model', 'Class', 'Seeds', 'Share of budget', 'Bytes', 'Packets', 'Delay (s)']]
for ds in DSS:
    for m in ('cnn_none', 'gb_none'):
        R = runs(ds, m)
        if not R or 'min_action_per_class' not in R[0]:
            continue
        for c in sorted(R[0]['min_action_per_class']):
            vals = [r['min_action_per_class'].get(c, {}) for r in R]

            def g(f):
                v = [x.get(f) for x in vals if x.get(f) is not None]
                return st.mean(v) if v else None
            bf = g('budget_frac')
            if bf is None:
                continue
            ns = sum(x.get('budget_frac') is not None for x in vals)
            rows.append([DSN[ds], NAME[m], c, str(ns), f'{bf:.2f}', f"{g('bytes'):,.0f}", f"{g('pkts'):.1f}", f"{g('delay_s'):.2f}"])
T['minaction'] = md(rows)

# UNSW-NB15: TTL artefact (candidates vs unseen values, and detectors without TTL features)
ds = 'UNSW_NB15'
rows = [['Model', 'TTL only, candidates', 'TTL only, unseen values', 'TTL only, any of 1–255', 'Clean macro-F1 without TTL features',
         'Grid, volume only, with TTL features', 'Grid, volume only, without TTL features',
         'Strongest, volume only, without TTL features']]
for m in ('cnn_none', 'rf_none', 'gb_none', 'mlp_none'):
    k = f'{ds}_{m}'
    if k + '_bb_ttl_only' not in V:
        continue
    kn = f'{ds}_{m}_nottl'
    rows.append([NAME[m], V[k + '_bb_ttl_only'], V.get(k + '_bb_ttl_only_unseen', '–'), V.get(k + '_bb_ttl_only_all', '–'),
                 V.get(kn + '_f1', '–'), V[k + '_bb_volume_b0.5'], V.get(kn + '_bb_volume_b0.5', '–'),
                 V.get(kn + '_strongest_volume_b0.5', '–')])   # same definition as Table 5 (numbers E5)
    if kn + '_f1' in V:
        V[f'{k}_nottl_df1'] = sgn(st.mean(A[ds][m + '_nottl']['macro_f1']) - st.mean(A[ds][m]['macro_f1'][:3]))
T['ttl'] = md(rows)

# networks vs trees: each network against each tree, Welch t-test over seeds, Holm within dataset x attack
NT = []
for ds in DSS:
    for att in ('strongest_b0.5', 'strongest_volume_b0.5', 'bb_volume_b0.5', 'bb_both_b0.5', 'bb_volume_b2.0'):
        fam = []
        for n in ('cnn_none', 'mlp_none'):
            for t in ('rf_none', 'gb_none'):
                if att in A[ds].get(n, {}) and att in A[ds].get(t, {}):
                    x, y = A[ds][n][att], A[ds][t][att]
                    p = stats.ttest_ind(x, y, equal_var=False).pvalue if np.std(x) + np.std(y) > 0 else 1.0
                    fam.append((n, t, st.mean(x) - st.mean(y), p))
        if not fam:
            continue
        adj = holm([f[3] for f in fam])
        for (n, t, d, p), pa in zip(fam, adj):
            V[f'{ds}_{n}_vs_{t}_{att}'] = pp(d)
            V[f'{ds}_{n}_vs_{t}_{att}_p'] = pfmt(pa)
            NT.append((ds, att, n, t, d, pa))
        V[f'{ds}_nets_gt_trees_all_{att}'] = str(all(d > 0 and pa < 0.05 for _, _, d, pa in [(f[0], f[1], f[2], a) for f, a in zip(fam, adj)]))
        V[f'{ds}_nets_lt_trees_all_{att}'] = str(all(d < 0 and pa < 0.05 for _, _, d, pa in [(f[0], f[1], f[2], a) for f, a in zip(fam, adj)]))
rows = [['Dataset', 'Attack', 'CNN − RF', 'CNN − GBDT', 'MLP − RF', 'MLP − GBDT']]
ATTN = {'strongest_b0.5': 'Strongest, all actions', 'strongest_volume_b0.5': 'Strongest, volume only',
        'bb_volume_b0.5': 'Grid, volume only, β = 0.5', 'bb_both_b0.5': 'Grid, both, β = 0.5', 'bb_volume_b2.0': 'Grid, volume only, β = 2'}
rows2 = [rows[0]]
for ds in DSS:
    for att in ATTN:
        if f'{ds}_cnn_none_vs_rf_none_{att}' not in V:
            continue
        if ds == 'TON_IoT' and att in ('strongest_volume_b0.5', 'bb_both_b0.5'):   # identical on TON-IoT (no TTL action)
            continue
        row = [DSN[ds], ATTN[att] if ds == 'UNSW_NB15' else ATTN[att].replace(', all actions', '').replace(', volume only', '')] + \
              [f"{V[f'{ds}_{n}_vs_{t}_{att}']} (p {V[f'{ds}_{n}_vs_{t}_{att}_p']})" for n in ('cnn_none', 'mlp_none') for t in ('rf_none', 'gb_none')]
        (rows if att.startswith('strongest') else rows2).append(row)
T['nettree'] = md(rows)
T['nettree_grid'] = md(rows2)

# white-box vs black-box on the networks (gradient masking check): strong PGD vs the grid of the same budget
for ds in DSS:
    for m in ('cnn_none', 'mlp_none'):
        a = A[ds].get(m, {})
        for b in (0.5, 2.0):
            if f'wbs_action_b{b}' in a and f'bb_both_b{b}' in a:
                V[f'{ds}_{m}_wbs_minus_bb_b{b}'] = pp(st.mean(a[f'wbs_action_b{b}']) - st.mean(a[f'bb_both_b{b}']))
            if f'wbs_action_b{b}' in a and f'wb_action_b{b}' in a:
                V[f'{ds}_{m}_wbs_minus_wb_b{b}'] = pp(st.mean(a[f'wbs_action_b{b}']) - st.mean(a[f'wb_action_b{b}']))
        if 'wb_unconstrained_mutable_e0.05' in a and 'wb_unconstrained_e0.05' in a:
            V[f'{ds}_{m}_mutable_minus_all_e0.05'] = pp(st.mean(a['wb_unconstrained_mutable_e0.05']) - st.mean(a['wb_unconstrained_e0.05']))

# ── defences ─────────────────────────────────────────────────────────────────
DEFTESTS = []   # (ds, m, comparison, att, mean diff, raw p) - secondary family: Holm within dataset over these tests
DUP_TON = ('bb_both_b0.5', 'bb_both_b2.0', 'strongest_volume_b0.5')   # TON-IoT: no TTL action, duplicates of other keys


def paired(ds, m, ref, att, key='asr'):
    """paired difference over the seeds both models have (seed-matched initialisation)"""
    Rm, Rr = runs(ds, m), runs(ds, ref)
    if not Rm or not Rr:
        return None
    sm = {r['cfg']['seed']: r for r in Rm}; sr = {r['cfg']['seed']: r for r in Rr}
    common = sorted(set(sm) & set(sr))
    if key == 'asr':
        if any(att not in sm[s] or att not in sr[s] for s in common):
            return None
        d = [sm[s][att] - sr[s][att] for s in common]
    elif key == 'f1':
        d = [sm[s]['macro_f1'] - sr[s]['macro_f1'] for s in common]
    elif key == 'rare':
        rare = Rm[0].get('rare_classes', [])
        if not rare:
            return None
        d = [st.mean(sm[s]['per_class_f1'][c] for c in rare) - st.mean(sr[s]['per_class_f1'][c] for c in rare) for s in common]
    elif key == 'fpr':
        d = [sm[s]['benign_fpr'] - sr[s]['benign_fpr'] for s in common]
    p = stats.ttest_1samp(d, 0).pvalue if len(d) > 1 and np.std(d) > 0 else 1.0
    return st.mean(d), float(p), len(d)


ROWM = []
rows = [['Dataset', 'Model', 'Seeds', 'Clean macro-F1', 'Δ clean vs undefended', 'Rare-class F1', 'Benign FPR',
         'Strongest ASR, all actions', 'Undefended, same seeds', 'Δ vs undefended', 'Δ vs clean loop', 'Strongest ASR, volume only']]
for ds in DSS:
    for m in BASE + DEF:
        k = f'{ds}_{m}'
        if k + '_f1' not in V:
            continue
        dc = da = dl = '–'
        if m in UNDEF:
            r = paired(ds, m, UNDEF[m], None, 'f1')
            if r:
                dc = V[k + '_dclean'] = sgn(r[0]); DEFTESTS.append((ds, m, 'undef', 'f1', r[0], r[1]))
            for key, nm in (('rare', '_drare'), ('fpr', '_dfpr')):
                r = paired(ds, m, UNDEF[m], None, key)
                if r:
                    V[k + nm] = sgn(r[0])
            for att in ('strongest_b0.5', 'strongest_volume_b0.5', 'bb_both_b0.5', 'bb_volume_b0.5', 'bb_ttl_only',
                        'bb_both_b2.0', 'wbs_action_b0.5', 'ad_none_b0.5', 'ad_free_b0.5', 'bb_unconstrained_e0.05'):
                if ds == 'TON_IoT' and att in DUP_TON:
                    continue
                r = paired(ds, m, UNDEF[m], att)
                if r:
                    V[f'{k}_d_{att}'] = pp(r[0]); V[f'{k}_d_{att}_nseeds'] = str(r[2])
                    if not (m in ('rf_augment', 'gb_augment') and att == 'strongest_b0.5'):   # in the primary family
                        DEFTESTS.append((ds, m, 'undef', att, r[0], r[1]))
                    Rr = {x['cfg']['seed']: x for x in runs(ds, UNDEF[m])}
                    Rm = runs(ds, m)
                    V[f'{k}_{att}_undef_matched'] = pct(st.mean(Rr[x['cfg']['seed']][att] for x in Rm))
            if f'{k}_d_strongest_b0.5' in V:
                da = V[f'{k}_d_strongest_b0.5']
            if m.startswith('cnn_') and m != 'cnn_clean_loop':
                for att in ('strongest_b0.5', 'strongest_volume_b0.5', 'bb_both_b0.5'):
                    if ds == 'TON_IoT' and att in DUP_TON:
                        continue
                    r = paired(ds, m, 'cnn_clean_loop', att)
                    if r:
                        V[f'{k}_dloop_{att}'] = pp(r[0]); DEFTESTS.append((ds, m, 'loop', att, r[0], r[1]))
                r = paired(ds, m, 'cnn_clean_loop', None, 'f1')
                if r:
                    V[f'{k}_dloop_f1'] = sgn(r[0])
                dl = V.get(f'{k}_dloop_strongest_b0.5', '–')
        rows.append([DSN[ds], NAME[m], V[k + '_nseeds'], f"{V[k + '_f1']} ± {V[k + '_sd']}", dc, V.get(k + '_rare_f1', '–'),
                     V[k + '_fpr'], cell(k, 'strongest_b0.5'), V.get(f'{k}_strongest_b0.5_undef_matched', '–'), da, dl,
                     cell(k, 'strongest_volume_b0.5') if ds == 'UNSW_NB15' else '= all actions'])
        ROWM.append(m)
MAINDEF = BASE + ('cnn_clean_loop', 'cnn_at_all', 'cnn_ccat', 'cnn_at_threat', 'cnn_trades', 'mlp_at_threat', 'rf_augment', 'gb_augment')
T['defences'] = md([rows[0]] + [r for r, m in zip(rows[1:], ROWM) if m in MAINDEF])
T['defvar'] = md([rows[0]] + [r for r, m in zip(rows[1:], ROWM) if m not in MAINDEF])

# defence vs defence (paired over common seeds): CNN AT-threat against each other CNN defence; tree variants
D2D = [('cnn_at_threat', o) for o in ('cnn_at_all', 'cnn_ccat', 'cnn_trades', 'cnn_clean_loop')] + \
      [(f'{t}_augment2', f'{t}_augment') for t in ('rf', 'gb')] + [(f'{t}_augment_nodelay', f'{t}_augment') for t in ('rf', 'gb')] + \
      [(f'{t}_augment_nobytes', f'{t}_augment') for t in ('rf', 'gb')] + [('cnn_at_all_bpda', 'cnn_at_all')]
for ds in DSS:
    for m, o in D2D:
        for att in ('strongest_b0.5', 'strongest_volume_b0.5', 'bb_both_b0.5', 'ad_free_b0.5', 'ad_none_b0.5', 'bb_only_delay_b0.5'):
            if ds == 'TON_IoT' and att in DUP_TON:
                continue
            r = paired(ds, m, o, att)
            if r:
                V[f'{ds}_{m}_minus_{o}_{att}'] = pp(r[0]); DEFTESTS.append((ds, m, o, att, r[0], r[1]))
        r = paired(ds, m, o, None, 'f1')
        if r:
            V[f'{ds}_{m}_minus_{o}_f1'] = sgn(r[0])

# Holm within dataset over every defence test
for ds in DSS:
    fam = [t for t in DEFTESTS if t[0] == ds]
    if not fam:
        continue
    adj = holm([t[5] for t in fam])
    V[f'{ds}_n_defence_tests'] = str(len(fam))
    for (d_, m, ref, att, diff, p), pa in zip(fam, adj):
        tag = {'undef': 'd', 'loop': 'dloop'}.get(ref)
        key = f'{ds}_{m}_{tag}_{att if att != "f1" else "clean"}_p' if tag else f'{ds}_{m}_minus_{ref}_{att}_p'
        if tag == 'd' and att == 'f1':
            key = f'{ds}_{m}_dclean_p'
        V[key] = pfmt(pa)
        V[key + 'raw'] = pfmt(p)

# summary statements about the defences
fl = lambda x: float(x.replace('−', '-').split()[0].rstrip('%'))  # noqa: E731
for ds in DSS:
    nd = [m for m in NETDEF if m != 'cnn_clean_loop' and f'{ds}_{m}_d_strongest_b0.5' in V]
    if nd:
        best = min(nd, key=lambda m: fl(V[f'{ds}_{m}_strongest_b0.5']))
        V[f'{ds}_best_netdef'] = NAME[best]
        V[f'{ds}_best_netdef_asr'] = V[f'{ds}_{best}_strongest_b0.5']
    for q in ('drare', 'dfpr'):
        xs = [fl(V[f'{ds}_{m}_{q}']) for m in NETDEF if f'{ds}_{m}_{q}' in V]
        if xs:
            V[f'{ds}_net_{q}_range'] = f'{sgn(min(xs))} to {sgn(max(xs))}'
    td = [m for m in TREEDEF if f'{ds}_{m}_d_strongest_b0.5' in V]
    if td:
        xs = [fl(V[f'{ds}_{m}_d_strongest_b0.5']) for m in td]
        V[f'{ds}_treedef_dstrongest_range'] = f'{sgn(min(xs), 1)} to {sgn(max(xs), 1)} pp'
tm = [abs(fl(V[f'{ds}_{m}_{q}'])) for ds in DSS for m in TREEDEF for q in ('drare', 'dfpr', 'dclean') if f'{ds}_{m}_{q}' in V]
if tm:
    V['tree_aug_max_abs_change'] = f'{max(tm):.3f}'

# per-class strongest ASR (undefended)
rows = None
for ds in DSS:
    have = [m for m in BASE if 'per_class_strongest' in A[ds].get(m, {})]
    if not have:
        continue
    cls = sorted(set.intersection(*(set(A[ds][m]['per_class_strongest']) for m in have)))
    rows = rows or [['Dataset', 'Class', 'Test share of attack flows'] + [NAME[m] for m in BASE]]
    for c in cls:
        rows.append([DSN[ds], c, pct(SHARES.get(ds, {}).get(c))] +
                    [pct(A[ds][m]['per_class_strongest'].get(c)) if m in have else '–' for m in BASE])
        for m in have:
            V[f'{ds}_{m}_pc_{c}'] = pct(A[ds][m]['per_class_strongest'][c])
if rows:
    T['perclass'] = md(rows)

# resplits (seed 0): does the picture hold on other data splits?
rows = [['Dataset', 'Model', 'Split', 'Clean macro-F1', 'Grid, volume only', 'Grid, both', 'Strongest, all actions']]
for ds in DSS:
    for m in BASE + ('cnn_at_threat', 'rf_augment', 'gb_augment'):
        for sfx, nm in (('', 'main'), ('_split1', '1'), ('_split2', '2')):
            R = runs(ds, m + sfx, seeds=(0,))
            if not R:
                continue
            r = R[0]
            rows.append([DSN[ds], NAME[m], nm, f3(r['macro_f1']), pct(r.get('bb_volume_b0.5')), pct(r.get('bb_both_b0.5')),
                         pct(r.get('strongest_b0.5'))])
T['splits'] = md(rows)
# do the network-tree orderings hold on every split (seed 0)?
for ds in DSS:
    ok = []
    for sfx in ('', '_split1', '_split2'):
        R = {m: runs(ds, m + sfx, seeds=(0,)) for m in BASE}
        if not all(R.values()):
            ok = None; break
        s = {m: R[m][0].get('strongest_b0.5') for m in BASE}
        if None in s.values():
            ok = None; break
        ok.append(min(s['cnn_none'], s['mlp_none']) > max(s['rf_none'], s['gb_none']))
    if ok is not None and ok:
        V[f'{ds}_splits_nets_above_trees'] = f'{sum(ok)} of {len(ok)}'

# ── review round 2: further evaluation of the saved models (run_adv_r2.py, results_r2/) ─────────────────────
R2DIR = RES.parent / ('results_r2_test' if SMOKE else 'results_r2')
EPS_R2 = (0.005, 0.01, 0.02, 0.05, 0.1)


def r2seeds(m, ds=None):
    if m == 'cnn_at_all_e0.1' and ds == 'TON_IoT' and all((R2DIR / ds / m / f'seed{s}.json').exists() for s in SEEDS5):
        return SEEDS5   # review round 3: the best seed-0 AT-all configuration retrained on five seeds
    if m in SWEEP:
        return (0,)
    if m.startswith(('rf_', 'gb_')) or '_nottl' in m or m in ('mlp_at_threat', 'cnn_at_all_bpda'):
        return SEEDS3
    return SEEDS5


def r2(ds, m, seeds=None):
    seeds = seeds or r2seeds(m, ds)
    P = [R2DIR / ds / m / f'seed{s}.json' for s in seeds]
    return [json.loads(p.read_text()) for p in P] if all(p.exists() for p in P) else None


def gk(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return None
        d = d[k]
    return d


def mk(R, *ks):
    v = [gk(r, *ks) for r in R]
    return st.mean(v) if R and all(x is not None for x in v) else None


R2M = BASE + NETDEF + TREEDEF + tuple(m + '_nottl' for m in ('cnn_none', 'mlp_none', 'rf_none', 'gb_none')) + SWEEP + ('cnn_at_threat_cs',)
B2 = {}   # (ds, m) -> list of result dicts (round 2)
for ds in DSS:
    for m in R2M:
        R = r2(ds, m)
        if R:
            B2[(ds, m)] = R

# (1) the strongest attack extended with the attacks the v3 union left out (TTL sweep, 24-point grid, per-action grids;
#     on TON-IoT the white-box attack also counts as volume-only): how much does it add?
gains = []
for (ds, m), R in B2.items():
    for a, b in (('strongest_plus_b0.5', 'strongest_b0.5'), ('strongest_plus_volume_b0.5', 'strongest_volume_b0.5')):
        x, y = mk(R, 'union', a), mk(R, 'union', b)
        if x is not None and y is not None:
            gains.append((x - y, ds, m, a))
            V[f'{ds}_{m}_{a}'] = pct(x)
if gains:
    g = max(gains)
    V['union_plus_gain_max'] = f'{100 * g[0]:.1f} pp'
    V['union_plus_gain_max_model'] = f'{NAME.get(g[2], g[2])}, {DSN[g[1]]}'

# (2) operating point: ASR (grid + adaptive for the networks, grid for the trees, beta 0.5, all actions) and end-to-end
#     evasion at the model's own operating point, with the benign threshold matched to the undefended model's validation
#     FPR (same family and seed), and at 1% validation FPR
OPN = {'native': 'own', 'fpr_match': 'matched', 'fpr1': 'fpr1'}
for (ds, m), R in B2.items():
    if 'op' not in R[0]:
        continue
    k = f'{ds}_{m}'
    for nm, lab in OPN.items():
        for f in ('asr', 'asr_volume', 'e2e', 'e2e_volume', 'test_fpr', 'detected'):
            x = mk(R, 'op', nm, f)
            if x is not None:
                V[f'{k}_op_{lab}_{f}'] = f3(x) if f == 'test_fpr' else pct(x)


for (ds, m), R in B2.items():   # review round 3, W1: undefended models under the threshold rule at their own FPR
    if 'opu' in R[0] and all('opu' in r for r in R):
        k = f'{ds}_{m}'
        for f in ('asr', 'asr_volume', 'e2e', 'e2e_volume', 'test_fpr', 'detected'):
            x = mk(R, 'opu', f)
            if x is not None:
                V[f'{k}_op_thr_{f}'] = f3(x) if f == 'test_fpr' else pct(x)


def op_vals(ds, m, nm, f='asr'):
    R = B2.get((ds, m))
    if not R or 'op' not in R[0]:
        return None
    if nm == 'opu':
        if not all('opu' in r for r in R):
            return None
        return {s: gk(r, 'opu', f) for s, r in zip(r2seeds(m, ds), R)}
    return {s: gk(r, 'op', nm, f) for s, r in zip(r2seeds(m, ds), R)}


def op_paired(ds, m, ref, f='asr', nm_m='fpr_match', nm_ref=None):
    """paired difference over common seeds; the undefended reference is taken under the same threshold rule at its
    own validation FPR ('opu', review round 3 W1)"""
    a = op_vals(ds, m, nm_m, f)
    b = op_vals(ds, ref, nm_ref or ('opu' if ref in BASE else 'fpr_match'), f)
    if not a or not b:
        return None
    common = [s for s in a if s in b and a[s] is not None and b[s] is not None]
    d = [a[s] - b[s] for s in common]
    p = stats.ttest_1samp(d, 0).pvalue if len(d) > 1 and np.std(d) > 0 else 1.0
    return st.mean(d), float(p), len(d)


# (3) PRIMARY family (specified in review round 2, M2; endpoint and reference revised in round 3 before the new runs
#     were read): per dataset, Holm over
#     - each network defence vs its undefended model: end-to-end evasion (missed or evaded) at the matched benign FPR,
#       the undefended model under the same threshold rule at its own FPR; UNSW-NB15 volume and timing (6 tests),
#     - CNN AT-threat vs each other CNN defence: the same endpoint (4 tests),
#     - each tree augmentation vs its undefended tree: strongest-attack ASR (2 tests; augmentation leaves the FPR unchanged).
PF = {'TON_IoT': 'e2e', 'UNSW_NB15': 'e2e_volume'}
PRIM_NET = ('cnn_clean_loop', 'cnn_at_all', 'cnn_ccat', 'cnn_at_threat', 'cnn_trades', 'mlp_at_threat')
PRIMARY = {}
for ds in DSS:
    fam = []
    for m in PRIM_NET:
        r = op_paired(ds, m, UNDEF[m], PF[ds])
        if r:
            fam.append((f'{ds}_{m}_prim', r))
    for o in ('cnn_at_all', 'cnn_ccat', 'cnn_trades', 'cnn_clean_loop'):
        r = op_paired(ds, 'cnn_at_threat', o, PF[ds])
        if r:
            fam.append((f'{ds}_cnn_at_threat_prim_vs_{o}', r))
    for m in ('rf_augment', 'gb_augment'):
        r = paired(ds, m, UNDEF[m], 'strongest_b0.5')
        if r:
            fam.append((f'{ds}_{m}_prim', r))
    if not fam:
        continue
    adj = holm([r[1] for _, r in fam])
    V[f'{ds}_n_primary_tests'] = str(len(fam))
    for (key, (d, p, n)), pa in zip(fam, adj):
        V[key] = pp(d); V[key + '_p'] = pfmt(pa); V[key + '_praw'] = pfmt(p); V[key + '_n'] = str(n)
        PRIMARY[key] = (d, p, pa, n)
    V[f'{ds}_n_primary_sig'] = str(sum(pa < 0.05 for pa in adj))

# operating-point table (main text; review round 3: threshold rule for every model, end-to-end endpoint,
# UNSW-NB15 on volume and timing; all-actions version in the supplement)
OPROWS = ('cnn_none', 'cnn_clean_loop', 'cnn_at_all', 'cnn_ccat', 'cnn_at_threat', 'cnn_trades', 'mlp_none', 'mlp_at_threat',
          'rf_none', 'rf_augment', 'gb_none', 'gb_augment')


def op_rows(fa, fe, prim):
    rows = [['Dataset', 'Model', 'Seeds', 'Benign FPR, own', 'ASR, own', 'Benign FPR, matched', 'Missed, matched',
             'Evaded, matched', 'Missed or evaded, matched', 'Δ missed or evaded (p)',
             'Missed or evaded, 1% FPR']]
    for ds in DSS:
        for m in OPROWS:
            R = B2.get((ds, m))
            if not R or 'op' not in R[0]:
                continue
            fa_, fe_ = fa[ds], fe[ds]
            und = m in BASE
            nm = 'opu' if und else 'fpr_match'
            if und and not all('opu' in r for r in R):
                continue
            g = (lambda f: mk(R, 'opu', f)) if und else (lambda f: mk(R, 'op', 'fpr_match', f))
            if und:
                dcell = '–'
            elif prim and m in PRIM_NET:
                dcell = f"{V.get(f'{ds}_{m}_prim', '–')} (Holm {V.get(f'{ds}_{m}_prim_p', '–')})"
            else:
                r = op_paired(ds, m, UNDEF[m], fe_)
                dcell = f'{pp(r[0])} ({pfmt(r[1])}, unadjusted)' if r else '–'      # final review round 2: tree rows are not Holm
            rows.append([DSN[ds], NAME[m], str(len(R)), f3(mk(R, 'op', 'native', 'test_fpr')), pct(mk(R, 'op', 'native', fa_)),
                         f3(g('test_fpr')), pct(1 - g('detected')), pct(g(fe_) - (1 - g('detected'))), pct(g(fe_)), dcell,
                         pct(mk(R, 'op', 'fpr1', fe_))])      # final review round 2 (M2c): missed and evaded shown separately
    return rows


T['op'] = md(op_rows({'TON_IoT': 'asr', 'UNSW_NB15': 'asr_volume'}, PF, True))
T['opall'] = md(op_rows({'TON_IoT': 'asr', 'UNSW_NB15': 'asr'}, {'TON_IoT': 'e2e', 'UNSW_NB15': 'e2e'}, False))
# secondary: the same comparisons on the ASR and, on UNSW-NB15, with all actions (text keys, unadjusted p)
for ds in DSS:
    for m in PRIM_NET + ('rf_augment', 'gb_augment'):
        for f in ('asr', 'asr_volume', 'e2e', 'e2e_volume'):
            r = op_paired(ds, m, UNDEF[m], f)
            if r:
                V[f'{ds}_{m}_opd_{f}'] = pp(r[0]); V[f'{ds}_{m}_opd_{f}_p'] = pfmt(r[1])
        for o in ('cnn_at_all', 'cnn_ccat', 'cnn_trades', 'cnn_clean_loop'):
            if m == 'cnn_at_threat':
                for f in ('asr', 'asr_volume', 'e2e', 'e2e_volume'):
                    r = op_paired(ds, m, o, f)
                    if r:
                        V[f'{ds}_cnn_at_threat_opd_vs_{o}_{f}'] = pp(r[0]); V[f'{ds}_cnn_at_threat_opd_vs_{o}_{f}_p'] = pfmt(r[1])
    # the rule change alone: undefended model, argmax vs threshold at its own FPR
    for m in BASE:
        R = B2.get((ds, m))
        if R and all('opu' in r for r in R):
            for f in ('asr', 'asr_volume', 'e2e', 'e2e_volume'):
                a, b = mk(R, 'opu', f), mk(R, 'op', 'native', f)
                if a is not None and b is not None:
                    V[f'{ds}_{m}_rule_d_{f}'] = pp(a - b)

# BPDA action-space PGD against every network (review round 3, W4)
rows = [['Dataset', 'Model', 'Seeds', 'Strongest attack', 'BPDA action PGD', 'New evasions', 'Strongest with BPDA']]
for ds in DSS:
    for m in ('cnn_none', 'cnn_clean_loop', 'cnn_at_all', 'cnn_ccat', 'cnn_at_threat', 'cnn_trades', 'cnn_at_all_bpda',
              'mlp_none', 'mlp_at_threat'):
        R = B2.get((ds, m))
        if not R or not all('bpa' in r for r in R):
            continue
        k = f'{ds}_{m}'
        st_ = mk(R, 'union', 'strongest_b0.5')
        for f in ('bpda_action_b0.5', 'bpda_action_volume_b0.5', 'bpda_new_vs_strongest_b0.5', 'strongest_with_bpda_b0.5'):
            V[f'{k}_{f}'] = pct(mk(R, 'bpa', f))
        rows.append([DSN[ds], NAME[m], str(len(R)), pct(st_), V[f'{k}_bpda_action_b0.5'], V[f'{k}_bpda_new_vs_strongest_b0.5'],
                     V[f'{k}_strongest_with_bpda_b0.5']])
    xs = [mk(B2[(ds, m)], 'bpa', 'bpda_new_vs_strongest_b0.5') for (d, m) in B2 if d == ds and all('bpa' in r for r in B2[(d, m)])]
    if xs:
        V[f'{ds}_bpda_new_max'] = pct(max(xs))
T['bpda'] = md(rows)
xs = [mk(R, 'bpa', 'bpda_new_vs_strongest_b0.5') for R in B2.values() if all('bpa' in r for r in R)]
if xs:
    V['bpda_new_max'] = pct(max(xs))

# consistency-relaxed realistic attack (review round 3, W3): realistic magnitudes, feature dependencies broken
for (ds, m), R in B2.items():
    if all('relax' in r for r in R):
        k = f'{ds}_{m}'
        for f in ('relax_b0.5', 'relax_union_b0.5', 'realistic_volume_b0.5', 'relax_new_vs_realistic'):
            V[f'{k}_{f}'] = pct(mk(R, 'relax', f))
        V[f'{k}_relax_gain'] = pp(mk(R, 'relax', 'relax_union_b0.5') - mk(R, 'relax', 'realistic_volume_b0.5'))
xs = [(mk(R, 'relax', 'relax_union_b0.5') - mk(R, 'relax', 'realistic_volume_b0.5')) for R in B2.values() if all('relax' in r for r in R)]
if xs:
    V['relax_gain_max'] = f'{100 * max(xs):.1f} pp'

for ds in DSS:
    xs = [mk(B2[(ds, m)], 'op', 'fpr1', PF[ds]) for m in OPROWS if m.startswith(('cnn', 'mlp')) and (ds, m) in B2 and 'op' in B2[(ds, m)][0]]
    if xs:
        V[f'{ds}_net_fpr1_asr_min'] = pct(min(xs))
for ds in DSS:   # tree augmentation against the grid at the matched threshold (secondary)
    for m in ('rf_augment', 'gb_augment', 'rf_augment_nobytes', 'gb_augment_nobytes', 'cnn_at_all_bpda'):
        r = op_paired(ds, m, UNDEF[m])
        if r:
            V[f'{ds}_{m}_op_d'] = pp(r[0]); V[f'{ds}_{m}_op_d_praw'] = pfmt(r[1])

# (4) feature-space reference, strengthened: black-box corners of the full ball, exhaustive {-1,0,+1} patterns on the
#     mutable (volume and timing) features and increase-only patterns, and for the networks BPDA PGD. Crossover = the
#     smallest eps at which the mutable-feature attack evades at least as many detected flows as the realistic
#     volume-and-timing attack (strongest volume-only, beta 0.5), on the same seeds.
def realistic_vol(ds, m, seeds):
    R = runs(ds, m, seeds)
    return st.mean(r['strongest_volume_b0.5'] for r in R) if R else None


def crossover(ds, m, key):
    R = B2.get((ds, m))
    if not R or 'fs' not in R[0]:
        return None, None
    rv = realistic_vol(ds, m, r2seeds(m, ds))
    for e in EPS_R2:
        x = mk(R, 'fs', f'{key}_e{e}')
        if x is not None and x >= rv:
            return e, rv
    return None, rv


def eps_txt(e):
    return '> 0.1' if e is None else ('≤ 0.005' if e == EPS_R2[0] else f'{e:g}')


rows = [['Dataset', 'Model', 'Realistic, volume and timing', 'Realistic range (min–max units)', 'Relaxed realistic',
         'Mutable features, increases only (ε = 0.01 / 0.05)',
         'Mutable features, ± (ε = 0.01 / 0.05)', 'Full ball (ε = 0.01)', 'Crossover ε, ± / increases only']]
for ds in DSS:
    for m in BASE + ('cnn_none_nottl', 'mlp_none_nottl', 'rf_none_nottl', 'gb_none_nottl'):
        R = B2.get((ds, m))
        if not R or 'fs' not in R[0]:
            continue
        k = f'{ds}_{m}'
        e1, rv = crossover(ds, m, 'fs_strongest_mutable')
        e2, _ = crossover(ds, m, 'bb_mut_inc')
        for e in EPS_R2:
            for key in ('fs_strongest_mutable', 'bb_mut_inc', 'fs_strongest', 'bb_corners', 'bb_mut', 'wb_bpda', 'wb_bpda_mutable'):
                x = mk(R, 'fs', f'{key}_e{e}')
                if x is not None:
                    V[f'{k}_{key}_e{e}'] = pct(x)
        V[f'{k}_cross_mut'], V[f'{k}_cross_inc'] = eps_txt(e1), eps_txt(e2)
        V[f'{k}_realistic_vol_r2seeds'] = pct(rv)
        rng_txt = rel_txt = '–'
        if all('relax' in r for r in R):   # review round 3, W3: realistic magnitudes with the feature dependencies broken
            w = max(st.mean(r['relax']['median_width_mm'][j] for r in R) for j in range(len(R[0]['relax']['median_width_mm'])))
            V[f'{k}_relax_width_max'] = rng_txt = f'{w:.0e}'
            rel_txt = V.get(f'{k}_relax_union_b0.5', '–')
        rows.append([DSN[ds], NAME.get(m, NAME.get(m.replace('_nottl', ''), m) + ', no TTL features'), pct(rv), rng_txt, rel_txt,
                     f"{V[f'{k}_bb_mut_inc_e0.01']} / {V[f'{k}_bb_mut_inc_e0.05']}",
                     f"{V[f'{k}_fs_strongest_mutable_e0.01']} / {V[f'{k}_fs_strongest_mutable_e0.05']}",
                     V[f'{k}_fs_strongest_e0.01'], f'{eps_txt(e1)} / {eps_txt(e2)}'])
T['fsref'] = md(rows)
for ds in DSS:
    xs = [V.get(f'{ds}_{m}_cross_mut') for m in BASE]
    if None not in xs:
        V[f'{ds}_cross_mut_list'] = ', '.join(f'{NAME[m]} {x}' for m, x in zip(BASE, xs))
    xs = [V.get(f'{ds}_{m}_cross_inc') for m in BASE]
    if None not in xs:
        V[f'{ds}_cross_inc_list'] = ', '.join(f'{NAME[m]} {x}' for m, x in zip(BASE, xs))
    ws = [V.get(f'{ds}_{m}_cross_mut') for m in BASE]
    if None not in ws:
        num = [float(x.replace('≤ ', '').replace('> ', '')) for x in ws]
        V[f'{ds}_cross_mut_max'] = ws[int(np.argmax(num))]

# defended CNNs under the feature-space attacks: ranking depends on which attack is used
rows = [['Dataset', 'Model', 'White-box ε-PGD (as in training)', 'BPDA PGD, full ball', 'Black-box corners, full ball',
         'Black-box, mutable features ±', 'Black-box, mutable, increases only', 'Realistic strongest']]
for ds in DSS:
    for m in ('cnn_none', 'cnn_clean_loop', 'cnn_at_all', 'cnn_at_all_bpda', 'cnn_ccat', 'cnn_at_threat', 'cnn_trades'):
        R = B2.get((ds, m)); Rv = runs(ds, m, r2seeds(m, ds))
        if not R or 'fs' not in R[0] or not Rv:
            continue
        k = f'{ds}_{m}'
        rows.append([DSN[ds], NAME[m], pct(st.mean(r['wb_unconstrained_e0.05'] for r in Rv)), pct(mk(R, 'fs', 'wb_bpda_e0.05')),
                     pct(mk(R, 'fs', 'bb_corners_e0.05')), pct(mk(R, 'fs', 'bb_mut_e0.05')), pct(mk(R, 'fs', 'bb_mut_inc_e0.05')),
                     pct(st.mean(r['strongest_b0.5'] for r in Rv))])
        V[f'{k}_wb_unconstrained_e0.05_r2seeds'] = pct(st.mean(r['wb_unconstrained_e0.05'] for r in Rv))
        for key in ('wb_bpda', 'bb_corners', 'bb_mut', 'bb_mut_inc'):
            V.setdefault(f'{k}_{key}_e0.05', pct(mk(R, 'fs', f'{key}_e0.05')))
T['fsdef'] = md(rows)
for ds in DSS:
    xs = [mk(B2[(ds, m)], 'fs', 'bb_corners_e0.05') for m in ('cnn_none', 'cnn_clean_loop', 'cnn_at_all', 'cnn_ccat', 'cnn_at_threat', 'cnn_trades')
          if (ds, m) in B2 and 'fs' in B2[(ds, m)][0]]
    if xs:
        V[f'{ds}_cnn_corners_e0.05_min'] = pct(min(xs))
        V[f'{ds}_cnn_corners_e0.05_max'] = pct(max(xs))
    xs = [mk(B2[(ds, m)], 'fs', 'bb_corners_e0.01') for m in BASE if (ds, m) in B2 and 'fs' in B2[(ds, m)][0]]
    if xs:
        V[f'{ds}_base_corners_e0.01_min'] = pct(min(xs))

# (5) physical feasibility: beta 0.5 grid with integer packets and the MTU bound
fl_ = []
for (ds, m), R in B2.items():
    if 'feas' not in R[0]:
        continue
    a, b = mk(R, 'feas', 'grid_both'), mk(R, 'feas', 'feas_both')
    V[f'{ds}_{m}_feas_grid'], V[f'{ds}_{m}_feas_feas'] = pct(a), pct(b)
    fl_.append((a - b, ds, m))
    V[f'{ds}_{m}_feas_lost'] = pct(mk(R, 'feas', 'lost_both'))
if fl_:
    g = max(fl_)
    V['feas_loss_max'] = f'{100 * g[0]:.1f} pp'
    V['feas_loss_max_model'] = f'{NAME.get(g[2], g[2])}, {DSN[g[1]]}'
    V['feas_change_min'] = f'{-100 * max(fl_)[0]:+.1f}'.replace('-', '−')
    V['feas_n_models'] = str(len(fl_))

rows = [['Dataset', 'Model', 'Grid, all actions', 'Grid, integer packets and MTU bound', 'Evasions lost']]
for ds in DSS:
    for m in OPROWS:
        k = f'{ds}_{m}'
        if f'{k}_feas_grid' in V:
            rows.append([DSN[ds], NAME[m], V[f'{k}_feas_grid'], V[f'{k}_feas_feas'], V[f'{k}_feas_lost']])
T['feas'] = md(rows)

# (6) tree saturation on 500 detected flows: regular grids of increasing resolution and a long adaptive search
rows = [['Dataset', 'Model', 'Strongest (v3 union)', 'Regular grids (3 resolutions)', 'Long adaptive search, new flows',
         'Union of all']]
for ds in DSS:
    for m in ('rf_none', 'rf_augment', 'rf_augment_nobytes', 'gb_none', 'gb_augment', 'gb_augment_nobytes'):
        R = B2.get((ds, m))
        if not R or 'sat' not in R[0]:
            continue
        k = f'{ds}_{m}'
        gr = sorted(x for x in R[0]['sat'] if x.startswith('grid_r'))
        gr.sort(key=lambda x: int(x[6:]))
        V[k + '_sat_strongest'] = pct(mk(R, 'sat', 'strongest_sub'))
        V[k + '_sat_union'] = pct(mk(R, 'sat', 'union_sub'))
        V[k + '_sat_gain'] = pp(mk(R, 'sat', 'union_sub') - mk(R, 'sat', 'strongest_sub'))
        V[k + '_sat_adaptive_new'] = pct(mk(R, 'sat', 'long_adaptive_new'))
        res = ', '.join(f"{int(x[6:]) ** (3 if ds == 'TON_IoT' else 4):,}: {pct(mk(R, 'sat', x))}" for x in gr)
        rows.append([DSN[ds], NAME[m], V[k + '_sat_strongest'], res, V[k + '_sat_adaptive_new'], V[k + '_sat_union']])
T['treesat'] = md(rows)
for ds in DSS:
    xs = [mk(B2[(ds, m)], 'sat', 'union_sub') - mk(B2[(ds, m)], 'sat', 'strongest_sub')
          for m in ('rf_none', 'rf_augment', 'gb_none', 'gb_augment') if (ds, m) in B2 and 'sat' in B2[(ds, m)][0]]
    if xs:
        V[f'{ds}_sat_gain_max'] = f'{100 * max(xs):.1f} pp'

# (7) one more configuration per defence (seed 0): AT-all eps and AT-threat beta_train
rows = [['Dataset', 'Model', 'Clean macro-F1', 'Benign FPR', 'Strongest ASR', 'ASR at matched FPR', 'Missed or evaded, matched FPR']]
for ds in DSS:
    for m in ('cnn_at_all_e0.02', 'cnn_at_all', 'cnn_at_all_e0.1', 'cnn_at_threat_b0.25', 'cnn_at_threat', 'cnn_at_threat_b1.0'):
        R = runs(ds, m, (0,)); Q = r2(ds, m, (0,))
        if not R or not Q:
            continue
        r, q = R[0], Q[0]
        nm = NAME[m] if m in SWEEP else NAME[m] + (', ε = 0.05' if 'all' in m else ', β = 0.5')
        k = f'{ds}_{m}_s0'
        V[k + '_strongest'] = pct(r['strongest_b0.5']); V[k + '_op_matched_asr'] = pct(gk(q, 'op', 'fpr_match', 'asr'))
        V[k + '_f1'] = f3(r['macro_f1'])
        rows.append([DSN[ds], nm, f3(r['macro_f1']), f3(r['benign_fpr']), pct(r['strongest_b0.5']),
                     pct(gk(q, 'op', 'fpr_match', 'asr')), pct(gk(q, 'op', 'fpr_match', 'e2e'))])
T['sweep'] = md(rows)
# AT-all eps 0.1 on five TON-IoT seeds (review round 3): same endpoint as the primary family, secondary tests
ds, m = 'TON_IoT', 'cnn_at_all_e0.1'
if r2seeds(m, ds) == SEEDS5:
    Rv = runs(ds, m, SEEDS5)
    V[f'{ds}_{m}_n5'] = '5'
    if Rv:
        V[f'{ds}_{m}_f1_5'] = f3(st.mean(r['macro_f1'] for r in Rv))
        V[f'{ds}_{m}_fpr_5'] = f3(st.mean(r['benign_fpr'] for r in Rv))
        V[f'{ds}_{m}_strongest_5'] = pct(st.mean(r['strongest_b0.5'] for r in Rv))
    for f in ('asr', 'e2e'):
        r = op_paired(ds, m, 'cnn_none', f)
        if r:
            V[f'{ds}_{m}_opd_{f}'] = pp(r[0]); V[f'{ds}_{m}_opd_{f}_p'] = pfmt(r[1])
        r = op_paired(ds, 'cnn_at_threat', m, f)
        if r:
            V[f'{ds}_cnn_at_threat_opd_vs_{m}_{f}'] = pp(r[0]); V[f'{ds}_cnn_at_threat_opd_vs_{m}_{f}_p'] = pfmt(r[1])


# final review round 2 (referee major 1): AT-threat whose inner attack starts at the most benign-scoring corner of the
# budget box (corners + 8 random points, as strong a start as the tree augmentation's candidate grid), five seeds.
# Secondary tests, same endpoint as the primary family; t-based 95% CI over seeds and a Welch test as a check.
def op_diff_ci(ds, m, ref, f):
    a = op_vals(ds, m, 'fpr_match', f); b = op_vals(ds, ref, 'opu' if ref in BASE else 'fpr_match', f)
    if not a or not b:
        return None
    common = [s for s in a if s in b and a[s] is not None and b[s] is not None]
    d = np.array([a[s] - b[s] for s in common])
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    pw = stats.ttest_ind([a[s] for s in common], [b[s] for s in common], equal_var=False).pvalue
    pt = stats.ttest_1samp(d, 0).pvalue
    return d.mean(), d.mean() - h, d.mean() + h, float(pt), float(pw), int((d < 0).sum()), len(d)


def ppci(r):
    return f'{pp(r[0])} [{pp(r[1])}, {pp(r[2])}]'


m = 'cnn_at_threat_cs'
rows = [['Dataset', 'Model', 'Clean macro-F1', 'Benign FPR', 'Strongest ASR (own threshold)', 'Missed or evaded, matched FPR',
         'Δ vs undefended [95% CI] (p; Welch p)', 'Δ vs AT-threat [95% CI] (p; Welch p)']]
for ds in DSS:
    Rv = runs(ds, m, SEEDS5)
    if not Rv or (ds, m) not in B2:
        continue
    V['atcs_done_' + ds] = 'yes'
    k = f'{ds}_{m}'
    V[k + '_f1'] = f3(st.mean(r['macro_f1'] for r in Rv)); V[k + '_fpr'] = f3(st.mean(r['benign_fpr'] for r in Rv))
    V[k + '_strongest'] = pct(st.mean(r['strongest_b0.5'] for r in Rv))
    if ds == 'UNSW_NB15':
        V[k + '_strongest_volume'] = pct(st.mean(r['strongest_volume_b0.5'] for r in Rv))
    for ref in ('cnn_none', 'cnn_at_threat'):
        for f in ('e2e', 'e2e_volume', 'asr'):
            r = op_diff_ci(ds, m, ref, f)
            if r:
                t = f'{k}_d_{ref}_{f}'
                V[t] = pp(r[0]); V[t + '_ci'] = f'[{pp(r[1])}, {pp(r[2])}]'; V[t + '_p'] = pfmt(r[3]); V[t + '_pw'] = pfmt(r[4]); V[t + '_neg'] = str(r[5])
    for ref in ('cnn_at_threat', 'cnn_none'):     # the main AT-threat with the same CI, for the text
        for f in ('e2e', 'e2e_volume'):
            r = op_diff_ci(ds, 'cnn_at_threat', 'cnn_none', f) if ref == 'cnn_none' else None
            if r:
                t = f'{ds}_cnn_at_threat_d_cnn_none_{f}'
                V[t] = pp(r[0]); V[t + '_ci'] = f'[{pp(r[1])}, {pp(r[2])}]'; V[t + '_pw'] = pfmt(r[4])
    f = PF[ds]
    for mm in ('cnn_at_threat', m):
        x = op_vals(ds, mm, 'fpr_match', f)
        V[f'{ds}_{mm}_op_matched_{f}_seeds'] = ', '.join(pct(x[s]) for s in sorted(x))
    e = mk(B2[(ds, m)], 'op', 'fpr_match', f)
    V[k + '_op_matched_pf'] = pct(e)
    r1, r2_ = op_diff_ci(ds, m, 'cnn_none', f), op_diff_ci(ds, m, 'cnn_at_threat', f)
    rows.append([DSN[ds] + ('' if f == 'e2e' else ' (volume and timing)'), NAME[m], V[k + '_f1'], V[k + '_fpr'], V[k + '_strongest'], pct(e),
                 f'{ppci(r1)} ({pfmt(r1[3])}; {pfmt(r1[4])})', f'{ppci(r2_)} ({pfmt(r2_[3])}; {pfmt(r2_[4])})'])
T['atcs'] = md(rows)


# final review round 2 (referee major 3): one destination reply of 40 header bytes per added packet (beta 0.5 grid,
# same candidates with and without replies); TON-IoT dst_pkts/dst_ip_bytes, UNSW-NB15 dbytes/dload; conn_state unchanged
REPLY_M = ('cnn_none', 'mlp_none', 'rf_none', 'gb_none', 'cnn_at_threat', 'cnn_at_threat_cs', 'rf_augment', 'gb_augment')
rows = [['Dataset', 'Model', 'Grid ASR, no reply', 'Grid ASR, with replies', 'Δ (pp)', 'Lost', 'Gained']]
dmax, ndone = [], 0
for ds in DSS:
    for m in REPLY_M:
        R = B2.get((ds, m)) or r2(ds, m)
        if not R or not all('reply' in r for r in R):
            continue
        ndone += 1
        f = 'volume' if ds == 'UNSW_NB15' else 'both'
        a, b = mk(R, 'reply', f'grid_{f}'), mk(R, 'reply', f'reply_{f}')
        lo, ga = mk(R, 'reply', 'lost_volume'), mk(R, 'reply', 'gained_volume')
        k = f'{ds}_{m}_reply'
        V[k + '_grid'], V[k + '_reply'], V[k + '_d'] = pct(a), pct(b), pp(b - a)
        dmax.append((abs(b - a), b - a, ds, m))
        rows.append([DSN[ds] + (' (volume and timing)' if f == 'volume' else ''), NAME[m], pct(a), pct(b), pp(b - a), pct(lo), pct(ga)])
if ndone == 2 * len(REPLY_M):
    V['reply_done'] = 'yes'
    g = max(dmax)
    V['reply_dmax'] = pp(g[1]); V['reply_dmax_model'] = f'{NAME[g[3]]}, {DSN[g[2]]}'
    V['reply_def_max'] = pp(max(x[1] for x in dmax if x[3] not in BASE))
    for ds in DSS:
        xs = [x[1] for x in dmax if x[2] == ds and x[3] in BASE]
        V[f'{ds}_reply_undef_range'] = f'{pp(min(xs))} to {pp(max(xs))}'
        xs = [x[1] for x in dmax if x[2] == ds and x[3] not in BASE]
        V[f'{ds}_reply_def_range'] = f'{pp(min(xs))} to {pp(max(xs))}'
T['reply'] = md(rows)


# final review round 2 (referee major 2a, 2d): attack-aware ROC from run_adv_r2 'roc' (beta 0.5 grid only, all attack-set
# flows; TON-IoT all actions, UNSW-NB15 volume and timing). Missed or evaded against the validation-set benign FPR
# 0.5%-5%, for the benign-probability rule of Table 7 (s1) and for 1 - max attack probability (s2). Partial area =
# mean missed-or-evaded over the FPR range (trapezoid on the FPR axis, normalised by its width).
ROC_M = ('cnn_none', 'cnn_clean_loop', 'cnn_at_all', 'cnn_ccat', 'cnn_at_threat', 'cnn_at_threat_cs', 'cnn_trades',
         'mlp_none', 'mlp_at_threat', 'rf_none', 'rf_augment', 'gb_none', 'gb_augment')
ROC_SHOW = (0.005, 0.01, 0.02, 0.05)


def roc_seeds(ds, m):
    seeds = (0, 1, 2) if m.startswith(('rf_', 'gb_')) or m == 'mlp_at_threat' else SEEDS5
    out = {}
    for sd in seeds:
        f = R2DIR / ds / m / f'seed{sd}.json'
        if f.exists():
            r = json.loads(f.read_text())
            if 'roc' in r:
                out[sd] = r['roc']
    return out if len(out) == len(seeds) else None


def pauc(rows, w):
    x = np.array([r['fpr'] for r in rows]); y = np.array([r[w] for r in rows])
    return float(np.trapz(y, x) / (x[-1] - x[0]))


ROC = {}
rows_t = [['Dataset', 'Model', 'Rule', 'Seeds'] + [f'{100 * f:g}% FPR' for f in ROC_SHOW] + ['Partial area, 0.5–5%', 'Δ partial area vs undefended [95% CI]']]
for ds in DSS:
    w = 'e2e_both' if ds == 'TON_IoT' else 'e2e_vol'
    for m in ROC_M:
        S = roc_seeds(ds, m)
        if S:
            ROC[(ds, m)] = S
            A[ds].setdefault(m, {})['roc'] = {r: [[row[w] for row in S[sd][r]] for sd in S] for r in ('s1', 's2')}
            A[ds][m]['roc_fpr'] = [row['fpr'] for row in next(iter(S.values()))['s1']]
    for rule in ('s1', 's2'):
        for m in ROC_M:
            S = ROC.get((ds, m))
            if not S:
                continue
            pa = {sd: pauc(S[sd][rule], w) for sd in S}
            k = f'{ds}_{m}_roc_{rule}'
            V[k + '_pauc'] = pct(np.mean(list(pa.values())))
            cells = []
            for f in ROC_SHOW:
                v = np.mean([next(r[w] for r in S[sd][rule] if abs(r['fpr'] - f) < 1e-9) for sd in S])
                V[f'{k}_at{100 * f:g}'] = pct(v); cells.append(pct(v))
            dcell = '–'
            u = UNDEF.get(m, m.split('_')[0] + '_none')
            if m not in BASE and ROC.get((ds, u)):
                pu = {sd: pauc(ROC[(ds, u)][sd][rule], w) for sd in ROC[(ds, u)]}
                common = [sd for sd in pa if sd in pu]
                d = np.array([pa[sd] - pu[sd] for sd in common])
                h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                p_ = stats.ttest_1samp(d, 0).pvalue
                V[k + '_dpauc'] = pp(d.mean()); V[k + '_dpauc_ci'] = f'[{pp(d.mean() - h)}, {pp(d.mean() + h)}]'
                V[k + '_dpauc_p'] = pfmt(p_)
                dcell = f"{V[k + '_dpauc']} {V[k + '_dpauc_ci']}"
            rows_t.append([DSN[ds] + ('' if ds == 'TON_IoT' else ' (volume and timing)'), NAME[m],
                           'p(benign)' if rule == 's1' else '1 − max p(attack)', str(len(S))] + cells + [V[k + '_pauc'], dcell])
if len(ROC) == 2 * len(ROC_M):
    V['roc_done'] = 'yes'
T['roc'] = md(rows_t)


# final review round 2, minor 11: tree capacity (tree_capacity.py; GBDT iterations after early stopping, RF leaves per tree)
for ds in DSS:
    f = R2DIR / ds / 'tree_capacity.json'
    if not f.exists():
        continue
    C = json.loads(f.read_text())
    it = [r['n_iter'] for r in C['gb_none']]
    V[f'{ds}_gb_iter_range'] = f'{min(it):,}–{max(it):,}' if min(it) != max(it) else f'{it[0]:,}'
    V[f'{ds}_gb_iter_capped'] = str(sum(x >= 1000 for x in it))
    lv = [r['leaves'] for r in C['rf_none']]
    V[f'{ds}_rf_leaves'] = f'{int(np.mean(lv)):,}'
    V[f'{ds}_rf_leaves_total'] = f'{int(100 * np.mean(lv)):,}'


# final review round 2, minor 4: the primary family as first specified (before the operating-point results):
# ASR at the matched FPR (UNSW-NB15 volume and timing), undefended reference under the class of highest probability
rows = [['Dataset', 'Comparison', 'Δ ASR (pp)', 'Holm p (original family)']]
for ds in DSS:
    f = 'asr' if ds == 'TON_IoT' else 'asr_volume'
    fam = []
    for m in PRIM_NET:
        r = op_paired(ds, m, UNDEF[m], f, nm_ref='native')
        if r:
            fam.append((f'{NAME[m]} − {NAME[UNDEF[m]]}', r))
    for o in ('cnn_at_all', 'cnn_ccat', 'cnn_trades', 'cnn_clean_loop'):
        r = op_paired(ds, 'cnn_at_threat', o, f)
        if r:
            fam.append((f'{NAME["cnn_at_threat"]} − {NAME[o]}', r))
    for m in ('rf_augment', 'gb_augment'):
        r = paired(ds, m, UNDEF[m], 'strongest_b0.5')
        if r:
            fam.append((f'{NAME[m]} − {NAME[UNDEF[m]]} (strongest ASR)', r))
    if not fam:
        continue
    ph = holm([r[1] for _, r in fam])
    for (lab, r), p_ in zip(fam, ph):
        rows.append([DSN[ds], lab, pp(r[0]), pfmt(p_)])
    V[f'{ds}_orig_n_sig'] = str(sum(p_ < 0.05 for p_ in ph))
T['origfam'] = md(rows)

# strongest-attack query budget per flow (worst case, beta 0.5): grid + 1,000-point grid (x TTL choices) + adaptive search
V['TON_IoT_strongest_queries'] = f'{(8 + 120) + 1000 + (1 + 40 * 16):,}'   # dense grid: 1,000 random points (corners dropped in run_adv_v3)
V['UNSW_NB15_strongest_queries'] = f'{(16 + 48) * 5 + 1000 * 5 + 3 * (1 + 40 * 16):,}'


# undefended volume-only ASR on seeds 0-2 (to match the 3-seed detectors without TTL features)
for m in BASE:
    R = runs('UNSW_NB15', m, SEEDS3)
    if R:
        V[f'UNSW_NB15_{m}_strongest_volume_b0.5_s3'] = pct(st.mean(r['strongest_volume_b0.5'] for r in R))

# run times
ev = [x for ds in DSS for m in A[ds] for x in A[ds][m].get('eval_s', []) if x]
if ev:
    V['eval_hours_total'] = f'{sum(ev) / 3600:.0f}'

# derived values quoted in the v3 text
def _pv(k):
    return float(V[k].rstrip('%').replace('−', '-').split()[0])


for ds in DSS:
    if all(f'{ds}_{m}_det' in V for m in BASE):
        V[f'{ds}_min_det'] = min((V[f'{ds}_{m}_det'] for m in BASE), key=lambda s: float(s.rstrip('%')))
    ks = [f'{ds}_{m}_bb_volume_b0.5_floor0.0' for m in BASE]
    if all(k in V for k in ks):
        v = sorted(_pv(k) for k in ks)
        V[f'{ds}_floor0_range'] = f'{v[0]:.1f}–{v[-1]:.1f}%'
    ps = [V.get(f'{ds}_{n}_vs_{t}_strongest_b0.5_p') for n in ('cnn_none', 'mlp_none') for t in ('rf_none', 'gb_none')]
    if None not in ps:
        V[f'{ds}_nettree_strongest_maxp'] = max(ps, key=lambda s: 0.0 if s.startswith('<') else float(s))
for c in ('ransomware', 'scanning'):
    ks = [f'TON_IoT_{m}_pc_{c}' for m in BASE]
    if all(k in V for k in ks):
        V[f'TON_IoT_pc_{c}_min'] = min((V[k] for k in ks), key=lambda s: float(s.rstrip('%')))
ks = [f'{ds}_{m}_wbs_minus_bb_b0.5' for ds in DSS for m in ('cnn_none', 'mlp_none')]
if all(k in V for k in ks):
    v = sorted(_pv(k) for k in ks)
    V['wbs_minus_bb_range'] = f'{-v[-1]:.1f}–{-v[0]:.1f}'
for ds in DSS:
    a, b = f'{ds}_cnn_clean_loop_d_strongest_b0.5', f'{ds}_cnn_at_threat_d_strongest_b0.5'
    if a in V and b in V:
        V[f'{ds}_loop_share'] = f'{100 * _pv(a) / _pv(b):.0f}%'

V['tables'] = T
(HERE / 'values.json').write_text(json.dumps(V, indent=1, ensure_ascii=False), encoding='utf-8')
(HERE / 'asr.json').write_text(json.dumps(A, indent=1), encoding='utf-8')
print(f'{RES}: {len(V) - 1} values, {len(T)} tables; models per dataset:', {ds: len(A[ds]) for ds in DSS})
