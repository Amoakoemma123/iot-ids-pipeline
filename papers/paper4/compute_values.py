"""Collect every number the Paper 4 manuscript quotes into values.json.

Source: Papers pipeline/experiments/paper4_federated/results_v2/<dataset>/<experiment>/seed<r>.json
(written by run_fl_v2.py). Each experiment has 5 repetitions (core) or 3 (extra, dp); a repetition
draws a new client partition, initialisation and batch order. Reported: mean over repetitions with a
t-based 95% CI; paired comparisons use the same repetition (same partition) for both algorithms.
The manuscript only contains {{key}} placeholders that build.py replaces from values.json.
"""
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
PIPE = HERE.parents[1] / 'Papers pipeline'
RES = PIPE / 'experiments' / 'paper4_federated' / 'results_v2'
DS = {'TON_IoT': 'TON', 'UNSW_NB15': 'UNSW'}
PRETTY = {'TON_IoT': 'TON-IoT', 'UNSW_NB15': 'UNSW-NB15'}
V, tables = {}, {}


def f(x, nd=3):
    return 'n/a' if x is None else f'{x:.{nd}f}'


def resdir(ds):
    # P4_NB_OLD=1: use the superseded round-5 N-BaIoT results (only to test this script while N-BaIoT is re-run)
    return RES / '_superseded_r5' / ds if (ds == 'N_BaIoT' and os.environ.get('P4_NB_OLD')) else RES / ds


def runs(ds, exp):
    d = resdir(ds) / exp
    if not d.exists():
        return {}
    return {int(p.stem[4:]): json.loads(p.read_text()) for p in sorted(d.glob('seed*.json'))}


def mci(xs, nd=3):
    """mean ± t-based 95% CI half-width over repetitions."""
    xs = np.asarray([x for x in xs if x is not None], float)
    if len(xs) == 0:
        return None, 'n/a'
    m = xs.mean()
    if len(xs) < 2:
        return m, f(m, nd)
    h = stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / np.sqrt(len(xs))
    return m, f'{m:.{nd}f} ± {h:.{nd}f}'


def dci(xs, nd=3):
    """mean difference over repetitions with its t-based 95% CI in brackets."""
    m, txt = mci(xs, nd)
    if m is None or '±' not in txt:
        return txt
    h = float(txt.split('± ')[1])
    return f'{m:+.{nd}f} [{m - h:+.{nd}f}, {m + h:+.{nd}f}]'


def paired(a, b):
    """mean difference a-b over shared repetitions, t-based CI, paired t-test p."""
    keys = sorted(set(a) & set(b))
    if len(keys) < 2:
        return None
    d = np.array([a[k] - b[k] for k in keys])
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    p = stats.ttest_rel([a[k] for k in keys], [b[k] for k in keys]).pvalue if d.std() > 0 else 1.0
    return dict(mean=float(d.mean()), lo=float(d.mean() - h), hi=float(d.mean() + h), p=float(p), n=len(d))


def pstr(d, nd=3):
    if d is None:
        return 'n/a'
    fp = lambda x: 'p < 0.001' if x < 0.001 else f'p = {x:.3f}'
    hs = f", Holm-adjusted {fp(d['p_holm'])}" if 'p_holm' in d else ''
    return f"{d['mean']:+.{nd}f} [{d['lo']:+.{nd}f}, {d['hi']:+.{nd}f}], {fp(d['p'])}{hs}"


def short(d, nd=3):
    """Primary comparison in the text: mean difference [95% CI], marked † if significant after Holm."""
    if d is None:
        return 'n/a'
    return f"{d['mean']:+.{nd}f} [{d['lo']:+.{nd}f}, {d['hi']:+.{nd}f}]" + ('†' if d.get('p_holm', 1) < 0.05 else '')


def welch(a, b):
    """Difference of means between two run sets that share no partition (Welch t-test, CI from Welch df)."""
    x, y = np.array(list(a.values()), float), np.array(list(b.values()), float)
    if len(x) < 2 or len(y) < 2:
        return None
    t = stats.ttest_ind(x, y, equal_var=False)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    h = stats.t.ppf(0.975, df) * np.sqrt(vx + vy); m = x.mean() - y.mean()
    return dict(mean=float(m), lo=float(m - h), hi=float(m + h), p=float(t.pvalue), n=len(x))


def sel(r):
    return r.get('test_f1_selected', r.get('test_f1_selected_mean'))


def fin(r):
    return r.get('test_f1_final', r.get('test_f1_final_mean'))


LABEL = {
    'central': 'Centralised (same recipe)', 'local_a0.5': 'Local only, α = 0.5',
    'local_a0.1': 'Local only, α = 0.1', 'fedavg_iid': 'FedAvg, IID', 'fedavg_a1': 'FedAvg, α = 1.0',
    'fedavg_a0.5': 'FedAvg, α = 0.5', 'fedavg_a0.1': 'FedAvg, α = 0.1',
    'fedprox_a0.5': 'FedProx, α = 0.5', 'fedprox_a0.1': 'FedProx, α = 0.1',
    'fedadam_a0.5': 'FedAdam, α = 0.5', 'fedadam_a0.1': 'FedAdam, α = 0.1',
    'scaffold_a0.5': 'SCAFFOLD (SGD), α = 0.5', 'scaffold_a0.1': 'SCAFFOLD (SGD), α = 0.1',
    'fedavgsgd_a0.5': 'FedAvg (SGD), α = 0.5', 'fedavgsgd_a0.1': 'FedAvg (SGD), α = 0.1',
    'scale_N2': 'FedAvg, N = 2', 'scale_N8': 'FedAvg, N = 8', 'scale_N16': 'FedAvg, N = 16',
    'fedavg_bn_a0.5': 'FedAvg, BatchNorm, α = 0.5', 'fedavg_bn_a0.1': 'FedAvg, BatchNorm, α = 0.1',
    'central_bn': 'Centralised, BatchNorm',
    'scaffold_a0.1_lr0.1': 'SCAFFOLD (SGD), α = 0.1, lr 0.1',
    'fedlc_a0.5': 'FedLC, α = 0.5', 'fedlc_a0.1': 'FedLC, α = 0.1',
}
CORE = ['central', 'local_a0.5', 'local_a0.1', 'fedavg_iid', 'fedavg_a1', 'fedavg_a0.5', 'fedavg_a0.1',
        'fedprox_a0.5', 'fedprox_a0.1', 'fedadam_a0.5', 'fedadam_a0.1', 'scaffold_a0.5', 'scaffold_a0.1',
        'fedavgsgd_a0.5', 'fedavgsgd_a0.1']
EXTRA = ['scale_N2', 'scale_N8', 'scale_N16', 'fedavg_bn_a0.5', 'fedavg_bn_a0.1', 'central_bn', 'scaffold_a0.1_lr0.1',
         'fedlc_a0.5', 'fedlc_a0.1', 'scaffold_a0.5_lr0.01', 'scaffold_a0.1_lr0.01',
         'local_nat', 'fedavg_nat', 'fedprox_nat', 'fedadam_nat', 'scaffold_nat', 'fedavgsgd_nat', 'fedlc_nat',
         'central_r100', 'fedavg_iid_r100', 'fedavg_a0.5_r100', 'fedavg_a0.1_r100',
         'central_noports', 'fedavg_iid_noports', 'fedavg_a0.5_noports', 'fedavg_a0.1_noports', 'central_bn_recal', 'fedavg_bn_a0.5_recal',
         'local_natr', 'fedavg_natr', 'fedprox_natr', 'fedadam_natr', 'scaffold_natr', 'fedavgsgd_natr', 'fedlc_natr',
         'fedavg_bn_a0.5_fedrecal', 'fedavg_bn_a0.1_fedrecal',
         'fedprox_a0.5_noports', 'fedadam_a0.5_noports', 'fedlc_a0.5_noports', 'scaffold_a0.5_noports', 'fedavgsgd_a0.5_noports']
PRIMARY = [('fedavg_iid', 'central'), ('fedavg_a0.5', 'central'), ('fedavg_a0.1', 'central'),
           ('fedavg_a0.5', 'local_a0.5'), ('fedavg_a0.1', 'local_a0.1'),
           ('fedprox_a0.5', 'fedavg_a0.5'), ('fedprox_a0.1', 'fedavg_a0.1'), ('fedadam_a0.5', 'fedavg_a0.5'), ('fedadam_a0.1', 'fedavg_a0.1'),
           ('fedlc_a0.5', 'fedavg_a0.5'), ('fedlc_a0.1', 'fedavg_a0.1'),
           ('scaffold_a0.5', 'fedavgsgd_a0.5'), ('scaffold_a0.1', 'fedavgsgd_a0.1'),
           ('fedavg_natr', 'central'), ('fedavg_natr', 'local_natr'), ('fedprox_natr', 'fedavg_natr'), ('fedadam_natr', 'fedavg_natr'),
           ('fedlc_natr', 'fedavg_natr'), ('scaffold_natr', 'fedavgsgd_natr')]
EXPLORATORY = [('scaffold_a0.5', 'fedavg_a0.5'), ('scaffold_a0.1', 'fedavg_a0.1'), ('fedavg_a1', 'central'),
               ('fedavg_bn_a0.5', 'fedavg_a0.5'), ('fedavg_bn_a0.1', 'fedavg_a0.1'), ('central_bn', 'central'),
               ('scaffold_a0.1_lr0.1', 'scaffold_a0.1'), ('scaffold_a0.5_lr0.01', 'fedavgsgd_a0.5'), ('scaffold_a0.1_lr0.01', 'fedavgsgd_a0.1'),
               ('fedavg_nat', 'central'), ('fedavg_nat', 'local_nat'), ('scaffold_natr', 'fedavg_natr'),
               ('fedavgsgd_natr', 'fedavg_natr')]
ABLATION_PAIRS = (('fedprox_a0.5_noports', 'fedavg_a0.5_noports'), ('fedadam_a0.5_noports', 'fedavg_a0.5_noports'),
                  ('fedlc_a0.5_noports', 'fedavg_a0.5_noports'), ('scaffold_a0.5_noports', 'fedavgsgd_a0.5_noports'),
                  ('central_noports', 'central'), ('fedavg_a0.5_noports', 'fedavg_a0.5'), ('fedavg_a0.1_noports', 'fedavg_a0.1'),
                  ('fedavg_a0.5_noports', 'central_noports'), ('fedavg_a0.1_noports', 'central_noports'))


def pow10(x):
    """1e-05 -> 10⁻⁵ for powers of ten below 0.01, otherwise the plain value."""
    e = np.log10(x)
    if x < 0.01 and abs(e - round(e)) < 1e-9:
        return '10' + str(int(round(e))).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))
    return f'{x:g}'


def skew(rr):
    top2, ncls = [], []
    for r in rr.values():
        for cc in r.get('client_class_counts') or []:
            cc = np.array(cc); top2.append(np.sort(cc)[::-1][:2].sum() / cc.sum()); ncls.append(int((cc >= 0.01 * cc.sum()).sum()))
    if not top2:
        return None
    return dict(top2=f'{100 * min(top2):.0f}–{100 * max(top2):.0f}%', top2_mean=f'{100 * np.mean(top2):.0f}%',
                cls=f'{min(ncls)}–{max(ncls)}', cls_mean=f'{np.mean(ncls):.1f}')


def cmp_label(e):
    """Readable name of an experiment for the table of paired comparisons."""
    algo = {'fedavg': 'FedAvg', 'fedprox': 'FedProx', 'fedadam': 'FedAdam', 'fedlc': 'FedLC', 'scaffold': 'SCAFFOLD',
            'fedavgsgd': 'FedAvg (SGD)', 'central': 'Centralised', 'local': 'Local only'}
    a, _, rest = e.partition('_')
    part = {'iid': 'IID', 'a1': 'α = 1.0', 'a0.5': 'α = 0.5', 'a0.1': 'α = 0.1', 'natr': 'device partition', 'dev': 'device partition',
            'iid9': 'IID', 'a0.5_N9': 'α = 0.5', 'a0.1_N9': 'α = 0.1'}.get(rest, rest)
    return algo.get(a, a) + (f', {part}' if part else '')


def holm_rows(ds_name, tests, pairs):
    fp = lambda x: '< 0.001' if x < 0.001 else f'{x:.3f}'
    out = []
    for a, b in pairs:
        key = f"_d_{a.replace('.', '')}_vs_{b.replace('.', '')}"
        d = next((v for k, v in tests.items() if k.endswith(key)), None)
        if d:
            lb = cmp_label(b) if a.split('_')[0] == b.split('_')[0] else cmp_label(b).split(',')[0]
            out.append(f"| {ds_name} | {cmp_label(a)} − {lb} |{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] | "
                       f"{fp(d['p'])} | {fp(d['p_holm'])}{' †' if d['p_holm'] < 0.05 else ''} |")
    return out


PAIRED_ROWS = ['| Dataset | Comparison (paired over repetitions) | Mean difference [95% CI] | p | Holm-adjusted p |', '|---|---|---|---|---|']

for ds, tag in DS.items():
    T = json.loads((RES / ds / 'tuned.json').read_text()) if (RES / ds / 'tuned.json').exists() else {}
    for k in ('fedavg', 'fedprox', 'fedadam', 'scaffold', 'fedavgsgd'):
        if k in T:
            V[f'{tag}_tuned_{k}'] = f'{T[k]:g}'
    R = {e: runs(ds, e) for e in CORE + EXTRA}
    S = {e: {k: sel(r) for k, r in rr.items()} for e, rr in R.items()}
    Fn = {e: {k: fin(r) for k, r in rr.items()} for e, rr in R.items()}
    for e, rr in R.items():
        if not rr:
            continue
        k = e.replace('.', '')
        m, s = mci(S[e].values()); V[f'{tag}_{k}_sel'] = s; V[f'{tag}_{k}_sel_mean'] = f(m)
        m, s = mci(Fn[e].values()); V[f'{tag}_{k}_fin'] = s; V[f'{tag}_{k}_fin_mean'] = f(m)
        V[f'{tag}_{k}_n'] = str(len(rr))
        if len(Fn[e]) > 1:
            V[f'{tag}_{k}_fin_sd'] = f(float(np.std(list(Fn[e].values()), ddof=1)))
            V[f'{tag}_{k}_sel_sd'] = f(float(np.std(list(S[e].values()), ddof=1)))
        r0 = next(iter(rr.values()))
        if 'selected_round' in r0:
            V[f'{tag}_{k}_round'] = f"{np.mean([r['selected_round'] for r in rr.values()]):.0f}"
        if 'total_mb' in r0:
            V[f'{tag}_{k}_mb'] = f"{r0['total_mb']:.1f}"
            V[f'{tag}_{k}_mb_round'] = f"{r0['bytes_per_round'] / 1e6:.2f}"
        full = [r.get('test_selected_full') for r in rr.values() if r.get('test_selected_full')]
        if full:
            V[f'{tag}_{k}_fpr'] = mci([x['benign_fpr'] * 100 for x in full], 2)[1]
            V[f'{tag}_{k}_dr'] = mci([x['attack_detection'] * 100 for x in full], 2)[1]
        if e.startswith('local'):
            V[f'{tag}_{k}_min'] = mci([r['test_f1_selected_min'] for r in rr.values()])[1]
            V[f'{tag}_{k}_max'] = mci([r['test_f1_selected_max'] for r in rr.values()])[1]
    tests, expl = {}, {}
    any_r = next((r for e in ('fedavg_a0.5', 'fedavg_iid') for r in R[e].values()), None)
    if any_r:
        V[f'{tag}_model_kb'] = f"{any_r['model_bytes'] / 1e3:.1f}"
        V[f'{tag}_nparams'] = f"{any_r['model_bytes'] // 4:,}"
        V[f'{tag}_raw_mb'] = f"{any_r['raw_train_mb']:.1f}"
    # paired comparisons (same repetition = same partition). PRIMARY comparisons, declared in advance, form the
    # Holm family of each dataset; EXPLORATORY ones are reported with uncorrected p-values and labelled as such.
    for a, b in PRIMARY + EXPLORATORY:
        if not (R.get(a) and R.get(b)):
            continue
        d = paired(S[a], S[b])
        key = f"{tag}_d_{a.replace('.', '')}_vs_{b.replace('.', '')}"
        (tests if (a, b) in PRIMARY else expl)[key] = d
        V[key + '_fin'] = pstr(paired(Fn[a], Fn[b]))
    for a, b in ABLATION_PAIRS + (('fedavg_a0.5_r100', 'fedavg_a0.5'), ('fedavg_a0.1_r100', 'fedavg_a0.1'), ('central_r100', 'central'),
                                  ('fedavg_iid_r100', 'central_r100'), ('fedavg_a0.5_r100', 'central_r100'), ('fedavg_a0.1_r100', 'central_r100')):
        # outside the Holm family (ablations and budget check)
        if R.get(a) and R.get(b):
            d = paired(S[a], S[b]); key = f"{tag}_d_{a.replace('.', '')}_vs_{b.replace('.', '')}"
            V[key] = pstr(d) + (', uncorrected' if d else ''); V[key + '_mean'] = f"{d['mean']:+.3f}" if d else 'n/a'
    # round-to-round swing of validation macro-F1 (rounds 11-50), with and without BN recalibration
    def swing(e, field='val_f1'):
        vals = [np.mean(np.abs(np.diff([h[field] for h in r['history'][10:]]))) for r in R.get(e, {}).values()
                if r['history'] and field in r['history'][0]]
        return f(float(np.mean(vals))) if vals else 'n/a'
    for e, k in (('central', 'central'), ('fedavg_a0.5', 'fedavg_a05'), ('central_bn_recal', 'central_bn'), ('fedavg_bn_a0.5_recal', 'fedavg_bn_a05')):
        V[f'{tag}_{k}_swing'] = swing(e)
        if 'recal' in e:
            V[f'{tag}_{k}_recal_swing'] = swing(e, 'val_f1_recal')
    # Holm correction over the family of paired comparisons of this dataset (validation-selected scores)
    fam = [(k, d) for k, d in tests.items() if d]
    order = sorted(fam, key=lambda kv: kv[1]['p'])
    m_, run = len(order), 0.0
    for i, (k, d) in enumerate(order):
        run = max(run, min(1.0, (m_ - i) * d['p']))
        d['p_holm'] = run
    V[f'{tag}_n_tests'] = str(m_)
    for k, d in tests.items():
        V[k] = short(d)   # primary comparisons: full statistics in the table of paired comparisons
        if d:
            V[k + '_mean'] = f"{d['mean']:+.3f}"
            V[k + '_sig'] = str(d['p'] < 0.05)
            V[k + '_sig_holm'] = str(d['p_holm'] < 0.05)
            V[k + '_pholm'] = f"{d['p_holm']:.3f}"
    for k, d in expl.items():
        V[k] = pstr(d) + (', uncorrected' if d else '')
        if d:
            V[k + '_mean'] = f"{d['mean']:+.3f}"
            V[k + '_sig'] = str(d['p'] < 0.05)
    PAIRED_ROWS.extend(holm_rows(PRETTY[ds], tests, PRIMARY))
    V[f'{tag}_holm_list'] = '; '.join(k.split('_d_')[1].replace('_vs_', ' vs ') for k, d in tests.items() if d)
    # per-client view: does each client gain from joining? (same partition in local-only and FedAvg runs).
    # The repetition is the statistical unit: client differences are averaged within a repetition first.
    def cli_ci(per_rep):
        return mci([np.mean(v) for v in per_rep.values() if v])[1]
    for a in ('0.5', '0.1', 'natr'):
        Lr, Fr = (R.get(f'local_{a}', {}), R.get(f'fedavg_{a}', {})) if a == 'natr' else (R.get(f'local_a{a}', {}), R.get(f'fedavg_a{a}', {}))
        n_cli = n_loc_wins = 0; own_gain, site_gain, common_gain = {}, {}, {}
        n_site_neg = 0
        for rep_, lr in Lr.items():
            fr = Fr.get(rep_)
            if not fr:
                continue
            for k, pc in enumerate(lr['per_client']):
                n_cli += 1
                n_loc_wins += pc['test_f1_selected'] >= fr['test_f1_selected']
                common_gain.setdefault(rep_, []).append(fr['test_f1_selected'] - pc['test_f1_selected'])
                cc = np.array(fr['client_class_counts'][k]); own = [i for i, c in enumerate(cc) if c >= 0.01 * cc.sum()]
                cls = list(fr['test_selected_full']['per_class_f1'])
                fl = np.mean([fr['test_selected_full']['per_class_f1'][cls[i]] for i in own])
                lo_ = np.mean([pc['test_selected_full']['per_class_f1'][cls[i]] for i in own])
                own_gain.setdefault(rep_, []).append(fl - lo_)
                if fr.get('site_selected') and pc.get('site_selected'):
                    g = fr['site_selected'][k]['f1'] - pc['site_selected'][0]['f1']
                    site_gain.setdefault(rep_, []).append(g); n_site_neg += g < 0
        ak = a.replace('.', '')
        if n_cli:
            V[f'{tag}_cli_a{ak}_n'] = str(n_cli)
            V[f'{tag}_cli_a{ak}_localwins'] = str(int(n_loc_wins))
            V[f'{tag}_cli_a{ak}_owngain'] = cli_ci(own_gain)
            V[f'{tag}_cli_a{ak}_ownneg'] = str(int(sum(g < 0 for v in own_gain.values() for g in v)))
            V[f'{tag}_cli_a{ak}_commongain'] = cli_ci(common_gain)
        if site_gain:
            allg = [g for v in site_gain.values() for g in v]
            V[f'{tag}_cli_a{ak}_sitegain'] = cli_ci(site_gain)
            pv = stats.ttest_1samp([np.mean(v) for v in site_gain.values()], 0).pvalue
            V[f'{tag}_cli_a{ak}_sitegain_p'] = 'p < 0.001' if pv < 0.001 else f'p = {pv:.3f}'
            V[f'{tag}_cli_a{ak}_siteneg'] = str(int(n_site_neg))
            V[f'{tag}_cli_a{ak}_siten'] = str(len(allg))
            V[f'{tag}_cli_a{ak}_sitemin'] = f(min(allg)); V[f'{tag}_cli_a{ak}_sitemax'] = f(max(allg))
            # absolute own-traffic scores (mean over sites and repetitions)
            fs = [fr['site_selected'][k]['f1'] for r_, fr in Fr.items() if fr.get('site_selected') for k in range(len(fr['site_selected']))]
            ls = [pc['site_selected'][0]['f1'] for r_, lr in Lr.items() for pc in lr['per_client'] if pc.get('site_selected')]
            V[f'{tag}_cli_a{ak}_site_fed'] = f(float(np.mean(fs))); V[f'{tag}_cli_a{ak}_site_loc'] = f(float(np.mean(ls)))
    # BatchNorm with federated re-estimation of its statistics (deployable; five repetitions). Each run reports
    # (a) the usual score with server-averaged running statistics and (b) the score after federated re-estimation,
    # with the round selected on the re-estimated validation score.
    for a in ('0.5', '0.1'):
        e = f'fedavg_bn_a{a}_fedrecal'; rr = R.get(e, {})
        if not rr:
            continue
        ak = a.replace('.', '')
        fr_sel, fr_fin = {}, {}
        for k_, r in rr.items():
            h = r['history']; i = max(range(len(h)), key=lambda j: (h[j]['val_f1_fedrecal'], -j))
            fr_sel[k_] = h[i]['test_f1_fedrecal']; fr_fin[k_] = h[-1]['test_f1_fedrecal']
        V[f'{tag}_bnfr_a{ak}_run_sel'] = mci(S[e].values())[1]
        V[f'{tag}_bnfr_a{ak}_run_fin'] = mci(Fn[e].values())[1]
        V[f'{tag}_bnfr_a{ak}_run_fin_sd'] = f(float(np.std(list(Fn[e].values()), ddof=1)))
        V[f'{tag}_bnfr_a{ak}_sel'] = mci(fr_sel.values())[1]
        V[f'{tag}_bnfr_a{ak}_fin'] = mci(fr_fin.values())[1]
        V[f'{tag}_bnfr_a{ak}_fin_sd'] = f(float(np.std(list(fr_fin.values()), ddof=1)))
        V[f'{tag}_bnfr_a{ak}_n'] = str(len(rr))
        V[f'{tag}_bnfr_a{ak}_swing'] = swing(e); V[f'{tag}_bnfr_a{ak}_fr_swing'] = swing(e, 'val_f1_fedrecal')
        gn = S.get(f'fedavg_a{a}', {}); gnf = Fn.get(f'fedavg_a{a}', {})
        for lab, x, g in (('run', S[e], gn), ('fr', fr_sel, gn), ('run_fin', Fn[e], gnf), ('fr_fin', fr_fin, gnf)):
            d = paired(x, g); V[f'{tag}_d_bnfr_a{ak}_{lab}_vs_gn'] = pstr(d) + (', uncorrected' if d else '')
            if d:
                V[f'{tag}_d_bnfr_a{ak}_{lab}_vs_gn_mean'] = f"{d['mean']:+.3f}"
        V[f'{tag}_fedavg_a{ak}_swing'] = swing(f'fedavg_a{a}')
    # does the federation gap grow with label skew? linear trend of the per-repetition gap to the centralised
    # model over IID, alpha = 1, 0.5, 0.1 (levels coded 0-3); exploratory
    # model over IID, alpha = 1, 0.5, 0.1 (levels coded 0-3): one least-squares slope per repetition, then a one-sample
    # t-test of the five slopes against zero (the repetition is the unit; exploratory)
    lv = ('fedavg_iid', 'fedavg_a1', 'fedavg_a0.5', 'fedavg_a0.1')
    reps_ = sorted(set.intersection(*[set(S.get(e, {})) for e in lv + ('central',)]))
    if len(reps_) > 2:
        slopes = [np.polyfit(range(4), [S[e][k_] - S['central'][k_] for e in lv], 1)[0] for k_ in reps_]
        pv = stats.ttest_1samp(slopes, 0).pvalue
        V[f'{tag}_trend_slope'] = f'{np.mean(slopes):+.3f}'
        V[f'{tag}_trend_p'] = 'p < 0.001' if pv < 0.001 else f'p = {pv:.3f}'
    for e in ('local_a0.5', 'local_a0.1', 'local_natr'):
        rr = R.get(e, {})
        if rr:
            k = e.replace('.', '')
            # size-weighted mean over clients (the plain mean weights every client equally)
            w = [np.average([pc['test_f1_selected'] for pc in r['per_client']], weights=r['client_sizes']) for r in rr.values()]
            V[f'{tag}_{k}_wmean'] = mci(w)[1]
            V[f'{tag}_{k}_lo'] = f(min(r['test_f1_selected_min'] for r in rr.values()))
            V[f'{tag}_{k}_hi'] = f(max(r['test_f1_selected_max'] for r in rr.values()))
    # final-vs-selected gap (what test-set round picking would have hidden)
    gaps = [S['fedavg_a0.1'][k] - Fn['fedavg_a0.1'][k] for k in S['fedavg_a0.1'] if k in Fn['fedavg_a0.1']]
    if gaps:
        V[f'{tag}_gap_a01'] = mci(gaps)[1]
    # main results table
    rows = ['| Setting | Test macro-F1, validation-selected round | SD | Test macro-F1, final round | Selected round | Benign FPR (%) |',
            '|---|---|---|---|---|---|']
    for e in CORE + [x for x in ('fedlc_a0.5', 'fedlc_a0.1') if R.get(x)]:
        if R.get(e):
            k = e.replace('.', '')
            rows.append(f"| {LABEL.get(e, e)} | {V[f'{tag}_{k}_sel']} | {V.get(f'{tag}_{k}_sel_sd', '–')} | {V[f'{tag}_{k}_fin']} | {V.get(f'{tag}_{k}_round', '–')} | {V.get(f'{tag}_{k}_fpr', '–')} |")
    tables[f'{tag}_main'] = '\n'.join(rows)
    rows = ['| Setting | Test macro-F1, validation-selected round | Test macro-F1, final round | MB per round |', '|---|---|---|---|']
    for e in ('fedavg_a0.5', 'scale_N2', 'scale_N8', 'scale_N16', 'fedavg_bn_a0.5', 'fedavg_bn_a0.1', 'central_bn'):
        if R.get(e):
            k = e.replace('.', '')
            rows.append(f"| {LABEL[e] if e != 'fedavg_a0.5' else 'FedAvg, N = 4, α = 0.5'} | {V[f'{tag}_{k}_sel']} | {V[f'{tag}_{k}_fin']} | {V.get(f'{tag}_{k}_mb_round', '–')} |")
    tables[f'{tag}_extra'] = '\n'.join(rows)
    # partition statistics (repetition 0)
    for e, lab in (('fedavg_iid', 'IID'), ('fedavg_a1', 'α = 1.0'), ('fedavg_a0.5', 'α = 0.5'), ('fedavg_a0.1', 'α = 0.1')):
        r = R[e].get(0)
        if r:
            cc = np.array(r['client_class_counts'])
            V[f"{tag}_part_{e.replace('.', '')}"] = (f"sizes {', '.join(f'{s:,}' for s in r['client_sizes'])}; "
                                                    f"classes per client {', '.join(str(int((c > 0).sum())) for c in cc)}")
    for e in ('fedavg_a1', 'fedavg_a0.5', 'fedavg_a0.1', 'fedavg_natr'):
        sk = skew(R.get(e, {}))
        if sk:
            for kk, vv in sk.items():
                V[f"{tag}_skew_{e.split('_', 1)[1].replace('.', '')}_{kk}"] = vv
    # redrawn device-level partition (TON-IoT): client sizes and label mix over the five repetitions
    NR = R.get('fedavg_natr', {})
    if NR:
        sizes = [x for r in NR.values() for x in r['client_sizes']]
        top2, ncls = [], []
        for r in NR.values():
            for cc in r['client_class_counts']:
                cc = np.array(cc); top2.append(np.sort(cc)[::-1][:2].sum() / cc.sum()); ncls.append(int((cc >= 0.01 * cc.sum()).sum()))
        V[f'{tag}_natr_size_min'], V[f'{tag}_natr_size_max'] = f'{min(sizes):,}', f'{max(sizes):,}'
        V[f'{tag}_natr_top2'] = f'{100 * min(top2):.0f}–{100 * max(top2):.0f}%'
        V[f'{tag}_natr_cls'] = f'{min(ncls)}–{max(ncls)}'
        V[f'{tag}_natr_n'] = str(len(NR))
    rows = ['| Setting | Test macro-F1, validation-selected round | SD | Test macro-F1, final round | Benign FPR (%) | Own-host traffic macro-F1 |', '|---|---|---|---|---|---|']
    for e, lab in (('central', 'Centralised'), ('local_natr', 'Local only (mean over sites)'), ('fedavg_natr', 'FedAvg'), ('fedprox_natr', 'FedProx'),
                   ('fedadam_natr', 'FedAdam'), ('fedlc_natr', 'FedLC'), ('fedavgsgd_natr', 'FedAvg (SGD)'), ('scaffold_natr', 'SCAFFOLD (SGD)')):
        if R.get(e):
            k = e.replace('.', '')
            site = [s_['f1'] for r in R[e].values() for s_ in (r.get('site_selected') or [])] if e != 'local_natr' else                    [pc['site_selected'][0]['f1'] for r in R[e].values() for pc in r['per_client'] if pc.get('site_selected')]
            rows.append(f"| {lab} | {V[f'{tag}_{k}_sel']} | {V.get(f'{tag}_{k}_sel_sd', '–')} | {V[f'{tag}_{k}_fin']} | {V.get(f'{tag}_{k}_fpr', '–')} | {f(float(np.mean(site))) if site else '–'} |")
    if len(rows) > 3:
        tables[f'{tag}_nat'] = '\n'.join(rows)
    for e in ('central_r100', 'fedavg_iid_r100', 'fedavg_a0.5_r100', 'fedavg_a0.1_r100'):
        rr = R.get(e)
        if rr:
            V[f"{tag}_{e.replace('.', '')}_round"] = f"{np.mean([r['selected_round'] for r in rr.values()]):.0f}"
    r01 = R['fedavg_a0.1'].get(0)
    if r01:
        tops, ncls = [], []
        for cc in r01['client_class_counts']:
            cc = np.array(cc); srt = np.sort(cc)[::-1]
            tops.append(f"{100 * srt[:2].sum() / cc.sum():.0f}%"); ncls.append(str(int((cc >= 0.01 * cc.sum()).sum())))
        V[f'{tag}_a01_top2'] = ', '.join(tops[:-1]) + ' and ' + tops[-1]
        V[f'{tag}_a01_cls1'] = ', '.join(ncls[:-1]) + ' and ' + ncls[-1]
    # DP
    dpe = ['dp_noclip_nonoise', 'dp_clip_nonoise', 'dp_sigma0.6', 'dp_sigma1', 'dp_sigma1.5', 'dp_sigma2.5', 'fedavg_short', 'fedavg_gn_short']
    DPR = {e: runs(ds, e) for e in dpe}
    rows = ['| Setting | Noise σ | ε (max over clients, δ = 10⁻⁵) | Test macro-F1, validation-selected | Test macro-F1, final |', '|---|---|---|---|---|']
    lab = {'fedavg_gn_short': 'Non-private FedAvg, same schedule (minibatch Adam, dropout)',
           'dp_noclip_nonoise': 'DP-SGD code, no clipping, no noise', 'dp_clip_nonoise': 'DP-SGD code, clipping C = 1, no noise'}
    for e in ('fedavg_gn_short', 'dp_noclip_nonoise', 'dp_clip_nonoise', 'dp_sigma0.6', 'dp_sigma1', 'dp_sigma1.5', 'dp_sigma2.5'):
        rr = DPR[e]
        if not rr:
            continue
        k = e.replace('.', '')
        V[f'{tag}_{k}_sel'] = mci([sel(r) for r in rr.values()])[1]
        V[f'{tag}_{k}_sel_mean'] = f(mci([sel(r) for r in rr.values()])[0])
        V[f'{tag}_{k}_fin'] = mci([fin(r) for r in rr.values()])[1]
        eps = [r['dp']['epsilon_max'] for r in rr.values() if r.get('dp') and r['dp'].get('epsilon_max')]
        if eps:
            V[f'{tag}_{k}_eps'] = f"{np.mean(eps):.2f} ({min(eps):.2f}–{max(eps):.2f})"
            V[f'{tag}_{k}_eps_mean'] = f"{np.mean(eps):.1f}"
        sig = e.split('sigma')[1] if 'sigma' in e else ('0' if e.startswith('dp') else '–')
        rows.append(f"| {lab.get(e, 'DP-SGD, C = 1')} | {sig} | {V.get(f'{tag}_{k}_eps', '∞' if e.startswith('dp') else '–')} | {V[f'{tag}_{k}_sel']} | {V[f'{tag}_{k}_fin']} |")
    if len(rows) > 2:
        tables[f'{tag}_dp'] = '\n'.join(rows)

# dataset sizes (corrected pipeline) and DP cost in macro-F1 points
import sys, contextlib, io, joblib
sys.path.insert(0, str(PIPE))
from src.data.loader import load_dataset
from experiments.baseline.config import DATASETS as BD
for ds, tag in DS.items():
    with contextlib.redirect_stdout(io.StringIO()):
        raw = load_dataset(*BD[ds], seed=42)
    V[f'{tag}_ntrain'], V[f'{tag}_nval'], V[f'{tag}_ntest'] = (f"{len(raw[k]):,}" for k in ('y_train', 'y_val', 'y_test'))
    cnt = np.bincount(raw['y_test']); pm = cnt.max() / cnt.sum()
    V[f'{tag}_majority_f1'] = f(2 * pm / (1 + pm) / len(cnt))
    V[f'{tag}_random_f1'] = f(1 / len(cnt), 2)
    V[f'{tag}_nfeat'] = str(joblib.load(PIPE / 'experiments' / 'baseline' / 'results' / ds / 'pipeline.pkl').n_features_)
    for sg in ('06', '1', '15', '25'):
        # final round: from the unrounded means (rounded strings gave 4 instead of 5 points for UNSW sigma = 0.6)
        c_ = [sel(r) for r in runs(ds, 'fedavg_gn_short').values()]
        b0 = [sel(r) for r in runs(ds, 'dp_sigma' + {'06': '0.6', '1': '1', '15': '1.5', '25': '2.5'}[sg]).values()]
        if c_ and b0:
            V[f'{tag}_dp_cost_short_{sg}'] = f"{100 * (np.mean(c_) - np.mean(b0)):.0f}"
        a_, b_ = V.get(f'{tag}_dp_noclip_nonoise_sel_mean'), V.get(f'{tag}_dp_sigma{sg}_sel_mean')
        if a_ and b_ and a_ != 'n/a' and b_ != 'n/a':
            V[f'{tag}_dp_cost_{sg}'] = f"{100 * (float(a_) - float(b_)):.0f}"

for ds, tag in DS.items():
    try:
        g = lambda e, c: np.mean([r['test_selected_full']['per_class_f1'][c] for r in runs(ds, e).values()])
        if tag == 'TON':
            V['TON_mitm_nonoise'] = f(g('dp_noclip_nonoise', 'mitm'), 2); V['TON_mitm_s06'] = f(g('dp_sigma0.6', 'mitm'), 2)
        else:
            V['UNSW_shell_nonoise'] = f(g('dp_noclip_nonoise', 'shellcode'), 2)
    except Exception as ex:
        print('per-class DP skipped', ex)

# full-budget DP at epsilon about 3 (TON-IoT; two repetitions)
for e in ('dp_full_sigma1.8', 'dp_full_nonoise'):
    rr = runs('TON_IoT', e)
    if rr:
        k = e.replace('.', '')
        V[f'TON_{k}_sel'] = f(float(np.mean([sel(r) for r in rr.values()])))
        V[f'TON_{k}_n'] = str(len(rr))
        eps = [r['dp']['epsilon_max'] for r in rr.values() if r.get('dp') and r['dp'].get('epsilon_max')]
        if eps:
            V[f'TON_{k}_eps'] = f'{np.mean(eps):.1f}'
if 'TON_dp_full_sigma18_sel' in V and 'TON_dp_full_nonoise_sel' in V:
    V['TON_dp_gap_full'] = f"{100 * (float(V['TON_dp_full_nonoise_sel']) - float(V['TON_dp_full_sigma18_sel'])):.0f}"
# DP tuning (TON-IoT, sigma = 1; two repetitions)
for e in ('dpt_C0.5', 'dpt_C2', 'dpt_C4', 'dpt_lr0.001', 'dpt_lr0.006', 'dpt_r40_sigma1', 'dpt_r40_nonoise'):
    rr = runs('TON_IoT', e)
    if rr:
        k = e.replace('.', '')
        V[f'TON_{k}_sel'] = f(float(np.mean([sel(r) for r in rr.values()])))
        eps = [r['dp']['epsilon_max'] for r in rr.values() if r.get('dp') and r['dp'].get('epsilon_max')]
        if eps:
            V[f'TON_{k}_eps'] = f'{np.mean(eps):.1f}'
if 'TON_dpt_r40_nonoise_sel' in V and 'TON_dpt_r40_sigma1_sel' in V:
    V['TON_dp_gap_r40'] = f"{100 * (float(V['TON_dpt_r40_nonoise_sel']) - float(V['TON_dpt_r40_sigma1_sel'])):.0f}"
for k in ('fedavg', 'fedprox', 'fedadam', 'scaffold', 'fedavgsgd', 'fedlc'):
    for ds, tag in list(DS.items()) + [('N_BaIoT', 'NB')]:
        p3 = resdir(ds) / 'tuned_v3.json'
        if p3.exists():
            T3 = json.loads(p3.read_text())
            if k in T3:
                V[f'{tag}_tuned_{k}'] = pow10(T3[k])
                V[f'{tag}_tuned_{k}_edge'] = str(T3.get(f'{k}_edge'))

V['n_tune_reps'] = str(len(runs('TON_IoT', 'tune3_fedavg_lr0.003')))
V['NB_n_tune_reps'] = str(len(runs('N_BaIoT', 'tune3_fedavg_lr0.003')))
WORD = {2: 'two', 3: 'three', 4: 'four', 5: 'five'}
V['n_tune_reps_word'] = WORD[int(V['n_tune_reps'])]; V['NB_n_tune_reps_word'] = WORD[int(V['NB_n_tune_reps'])]
# check of the centralised reference's learning rate (tuning repetitions, mean validation macro-F1 of the selected epoch)
for ds, tag in DS.items():
    sc = {lr: np.mean([r['val_f1_selected'] for r in runs(ds, f'tunec_central_lr{lr}').values()])
          for lr in ('0.001', '0.003', '0.006', '0.01') if runs(ds, f'tunec_central_lr{lr}')}
    if sc:
        best = max(sc, key=sc.get)
        V[f'{tag}_tunec_best'] = f'{float(best):g}'
        V[f'{tag}_tunec_scores'] = ', '.join(f'{float(k):g}: {v:.3f}' for k, v in sc.items())
        V[f'{tag}_tunec_margin'] = f"{sc[best] - sc.get('0.003', sc[best]):.3f}"
for e in ('dp_full_sigma1.8', 'dp_full_nonoise'):
    V[f"TON_{e.replace('.', '')}_n"] = str(len(runs('TON_IoT', e)))
    V[f"TON_{e.replace('.', '')}_ci"] = mci([sel(r) for r in runs('TON_IoT', e).values()])[1]
V['TON_dpt_n'] = str(len(runs('TON_IoT', 'dpt_C2')))
V['TON_dpt_n_word'] = {2: 'two', 3: 'three'}[int(V['TON_dpt_n'])]
V['TON_dp_full_sigma18_n_word'] = {2: 'two', 3: 'three'}[int(V['TON_dp_full_sigma18_n'])]
# TON-IoT destination hosts (training split), for the device-level partition
with contextlib.redirect_stdout(io.StringIO()):
    rawm = load_dataset(*BD['TON_IoT'], seed=42, return_meta=True)
_, hc = np.unique(rawm['meta_train']['dst_ip'].astype(str).values, return_counts=True); hc = np.sort(hc)[::-1]
V['TON_n_hosts'] = f'{len(hc):,}'; V['TON_host1_share'] = f'{100 * hc[0] / hc.sum():.0f}'
V['TON_host10_share'] = f'{100 * hc[:10].sum() / hc.sum():.0f}'; V['TON_host1_n'] = f'{hc[0]:,}'

# ── N-BaIoT: nine real devices, one client per device ────────────────────────
NBI = json.loads((PIPE / 'dataset4' / 'nbaiot_prepared.json').read_text())
V.update(NB_ntrain=f"{NBI['n_train']:,}", NB_nval=f"{NBI['n_val']:,}", NB_ntest=f"{NBI['n_test']:,}", NB_nfeat=str(NBI['n_features']),
         NB_nclasses=str(len(NBI['classes'])), NB_ndev=str(len(NBI['devices'])), NB_nsampled=f"{NBI['n_sampled']:,}",
         NB_ndup=f"{NBI['n_exact_dup']:,}", NB_nconf=str(NBI['n_conflict']), NB_frac=f"{100 * NBI['sample_frac']:.0f}",
         NB_nrows_full=f"{sum(v['rows'] for v in NBI['files'].values()):,}")
_mir = [i for i, c in enumerate(NBI['classes']) if c.startswith('mirai')]
_nomir = [d for d in NBI['devices'] if not any(NBI['train_counts'][d][i] for i in _mir)]
V['NB_nomirai'] = ' and '.join(d.replace('_', ' ') for d in _nomir); V['NB_n_nomirai'] = str(len(_nomir))
dsz = [sum(v) for v in NBI['train_counts'].values()]
V['NB_dev_min'], V['NB_dev_max'] = f'{min(dsz):,}', f'{max(dsz):,}'
NBE = ['central', 'local_dev', 'fedavg_dev', 'fedprox_dev', 'fedadam_dev', 'fedlc_dev', 'fedavgsgd_dev', 'scaffold_dev',
       'fedavg_iid9', 'fedavg_a0.5_N9', 'fedavg_a0.1_N9']
NR_ = {e: runs('N_BaIoT', e) for e in NBE}
NS = {e: {k: sel(r) for k, r in rr.items()} for e, rr in NR_.items()}
NF = {e: {k: fin(r) for k, r in rr.items()} for e, rr in NR_.items()}
for e, rr in NR_.items():
    if rr:
        k = e.replace('.', '')
        m, s_ = mci(NS[e].values()); V[f'NB_{k}_sel'] = s_; V[f'NB_{k}_sel_mean'] = f(m)
        V[f'NB_{k}_fin'] = mci(NF[e].values())[1]; V[f'NB_{k}_n'] = str(len(rr))
        if len(rr) > 1:
            V[f'NB_{k}_sel_sd'] = f(float(np.std(list(NS[e].values()), ddof=1)))
        full = [r.get('test_selected_full') for r in rr.values() if r.get('test_selected_full')]
        if full:
            V[f'NB_{k}_fpr'] = mci([x['benign_fpr'] * 100 for x in full], 2)[1]
for e in ('fedavg_dev', 'fedavg_a0.5_N9', 'fedavg_a0.1_N9'):
    sk = skew(NR_.get(e, {}))
    if sk:
        for kk, vv in sk.items():
            V[f"NB_skew_{e.split('_', 1)[1].replace('.', '')}_{kk}"] = vv
NB_PRIMARY = [('fedavg_dev', 'central'), ('fedavg_dev', 'local_dev'), ('fedprox_dev', 'fedavg_dev'), ('fedadam_dev', 'fedavg_dev'),
              ('fedlc_dev', 'fedavg_dev'), ('scaffold_dev', 'fedavgsgd_dev'), ('fedavg_iid9', 'central'),
              ('fedavg_a0.5_N9', 'central'), ('fedavg_a0.1_N9', 'central')]
NB_EXPL = [('scaffold_dev', 'fedavg_dev'), ('fedavgsgd_dev', 'fedavg_dev')]
nbt = {}
for a, b in NB_PRIMARY + NB_EXPL:
    if NS.get(a) and NS.get(b):
        d = paired(NS[a], NS[b]); key = f"NB_d_{a.replace('.', '')}_vs_{b.replace('.', '')}"
        if (a, b) in NB_PRIMARY:
            nbt[key] = d
        else:
            V[key] = pstr(d) + ', uncorrected' if d else 'n/a'
            if d:
                V[key + '_mean'] = f"{d['mean']:+.3f}"
fam = sorted([(k, d) for k, d in nbt.items() if d], key=lambda kv: kv[1]['p']); run_ = 0.0
for i, (k, d) in enumerate(fam):
    run_ = max(run_, min(1.0, (len(fam) - i) * d['p'])); d['p_holm'] = run_
V['NB_n_tests'] = str(len(fam))
PAIRED_ROWS.extend(holm_rows('N-BaIoT', nbt, NB_PRIMARY))
tables['paired'] = '\n'.join(PAIRED_ROWS)
for k, d in nbt.items():
    V[k] = short(d)
    if d:
        V[k + '_mean'] = f"{d['mean']:+.3f}"; V[k + '_sig_holm'] = str(d['p_holm'] < 0.05)
# per-device view: each device's own test traffic, FedAvg (device partition) against the device's local model.
# The device partition is the same in every repetition, so the device is the unit (averaged over repetitions).
dev_gain, dev_rows = [], []
for j, dname in enumerate(NBI['devices']):
    ks = [k for k in NR_['fedavg_dev'] if k in NR_['local_dev']]
    gf = [NR_['fedavg_dev'][k]['site_selected'][j]['f1'] for k in ks]
    gl = [NR_['local_dev'][k]['per_client'][j]['site_selected'][0]['f1'] for k in ks]
    gc = [NR_['local_dev'][k]['per_client'][j]['test_f1_selected'] for k in ks]
    if gf:
        dev_gain.append(np.mean(gf) - np.mean(gl))
        dev_rows.append(f"| {dname.replace('_', ' ')} | {sum(NBI['train_counts'][dname]):,} | "
                        f"{sum(1 for c in NBI['train_counts'][dname] if c)} | {np.mean(gl):.3f} | {np.mean(gf):.3f} | "
                        f"{np.mean(gf) - np.mean(gl):+.3f} | {np.mean(gc):.3f} |")
if len(dev_gain) > 1:
    V['NB_dev_gain'] = dci(dev_gain); V['NB_dev_neg'] = str(sum(g < 0 for g in dev_gain)); V['NB_dev_n'] = str(len(dev_gain))
    V['NB_dev_gain_min'], V['NB_dev_gain_max'] = f'{min(dev_gain):+.3f}', f'{max(dev_gain):+.3f}'
    pv = stats.ttest_1samp(dev_gain, 0).pvalue
    V['NB_dev_gain_p'] = 'p < 0.001' if pv < 0.001 else f'p = {pv:.3f}'
    V['NB_dev_reps'] = str(len([k for k in NR_['fedavg_dev'] if k in NR_['local_dev']]))
    tables['NB_dev'] = '\n'.join(['| Device | Training records | Classes | Local model, own traffic | FedAvg, own traffic | Difference | Local model, all test traffic |',
                                  '|---|---|---|---|---|---|---|'] + dev_rows)
rows = ['| Setting | Test macro-F1, validation-selected round | SD | Test macro-F1, final round | Benign FPR (%) |', '|---|---|---|---|---|']
for e, lab in (('central', 'Centralised'), ('local_dev', 'Local only (mean over devices)'), ('fedavg_dev', 'FedAvg, one client per device'),
               ('fedprox_dev', 'FedProx, per device'), ('fedadam_dev', 'FedAdam, per device'), ('fedlc_dev', 'FedLC, per device'),
               ('fedavgsgd_dev', 'FedAvg (SGD), per device'), ('scaffold_dev', 'SCAFFOLD (SGD), per device'),
               ('fedavg_iid9', 'FedAvg, IID, N = 9'), ('fedavg_a0.5_N9', 'FedAvg, α = 0.5, N = 9'), ('fedavg_a0.1_N9', 'FedAvg, α = 0.1, N = 9')):
    if NR_.get(e):
        k = e.replace('.', '')
        rows.append(f"| {lab} | {V[f'NB_{k}_sel']} | {V.get(f'NB_{k}_sel_sd', '–')} | {V[f'NB_{k}_fin']} | {V.get(f'NB_{k}_fpr', '–')} |")
# final round (8-page limit): the main-text table keeps the references and the partitions; all rows go to the supplement
tables['NB_full'] = '\n'.join(rows)
tables['NB_main'] = '\n'.join(r for r in rows if not any(a in r for a in ('FedProx', 'FedAdam', 'FedLC', 'FedAvg (SGD)', 'SCAFFOLD')))
if NR_.get('local_dev'):
    V['NB_local_dev_lo'] = f(min(r['test_f1_selected_min'] for r in NR_['local_dev'].values()))
    V['NB_local_dev_hi'] = f(max(r['test_f1_selected_max'] for r in NR_['local_dev'].values()))


# hyperparameters selected at an edge of their grid
NAMES = {'fedavg': "FedAvg's learning rate", 'fedprox': "FedProx's μ", 'fedadam': "FedAdam's server learning rate",
         'fedlc': "FedLC's τ", 'scaffold': "SCAFFOLD's learning rate", 'fedavgsgd': "the learning rate of FedAvg with SGD"}
edges = []
for ds, tag, pretty in (('TON_IoT', 'TON', 'TON-IoT'), ('UNSW_NB15', 'UNSW', 'UNSW-NB15'), ('N_BaIoT', 'NB', 'N-BaIoT')):
    T3 = json.loads((resdir(ds) / 'tuned_v3.json').read_text())
    for k, nm in NAMES.items():
        if T3.get(f'{k}_edge'):
            sc = T3[f'{k}_scores']; lo = min(sc, key=float)
            edges.append(f"{nm} on {pretty} ({'smallest' if float(lo) == T3[k] else 'largest'} value)")
V['EDGE_SENTENCE'] = ('Only ' + '; '.join(edges) + ' lies at an edge of its grid' + (
    '; because μ → 0 recovers FedAvg, this does not affect the FedProx comparison.' if all('μ' in e for e in edges) else '.')) if edges \
    else 'Every selected value lies inside its grid.'
V['TON_cli_anatr_sitegain_mean'] = V.get('TON_cli_anatr_sitegain', 'n/a').split(' ')[0]
if 'NB_dev_gain' in V:
    V['NB_dev_gain_mean'] = V['NB_dev_gain'].split(' ')[0]
    if not V['NB_dev_gain_mean'].startswith('-'):
        V['NB_dev_gain_mean'] = '+' + V['NB_dev_gain_mean']
if not V['TON_cli_anatr_sitegain_mean'].startswith('-'):
    V['TON_cli_anatr_sitegain_mean'] = '+' + V['TON_cli_anatr_sitegain_mean']


# ══ round 6 (review round 4) ════════════════════════════════════════════════════
import pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    _rt = load_dataset(*BD['TON_IoT'], seed=42)
TON_TRAIN_COUNTS = dict(zip(list(_rt['class_names']), np.bincount(_rt['y_train']).tolist()))
def fp_(x):
    return 'p < 0.001' if x < 0.001 else f'p = {x:.3f}'


def selmap(ds, e):
    return {k: sel(r) for k, r in runs(ds, e).items()}


# partitions compared with each other share no partition: Welch, exploratory
for ds, tag, a, b in (('TON_IoT', 'TON', 'fedavg_natr', 'fedavg_a0.5'), ('TON_IoT', 'TON', 'fedavg_natr', 'fedavg_a0.1'),
                      ('N_BaIoT', 'NB', 'fedavg_dev', 'fedavg_a0.5_N9'), ('N_BaIoT', 'NB', 'fedavg_dev', 'fedavg_a0.1_N9')):
    d = welch(selmap(ds, a), selmap(ds, b))
    if d:
        key = f"{tag}_w_{a.replace('.', '')}_vs_{b.replace('.', '')}"
        V[key] = f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}], {fp_(d['p'])} (Welch, uncorrected)"
        V[key + '_mean'] = f"{d['mean']:+.3f}"; V[key + '_p'] = f"{d['p']:.3f}"


# own traffic: macro-F1, attack detection and benign FPR on each site's own test traffic
def site_list(r, local=False):
    return [pc['site_selected'][0] for pc in r['per_client']] if local else r['site_selected']


def own_traffic(ds, tag, fed, loc, unit):
    F, Lc = runs(ds, fed), runs(ds, loc)
    reps = [k for k in F if k in Lc]
    if not reps:
        return
    for nm, rr, local in (('fed', F, False), ('loc', Lc, True)):
        S_ = [s_ for k in reps for s_ in site_list(rr[k], local)]
        for m, sc in (('f1', 1), ('attack_detection', 100), ('benign_fpr', 100)):
            vals = [s_[m] * sc for s_ in S_ if s_.get(m) is not None]
            V[f'{tag}_own_{nm}_{m}'] = f(float(np.mean(vals)), 1 if sc == 100 else 3)
    S0 = [s_ for k in reps for s_ in site_list(F[k])]
    V[f'{tag}_own_n_min'], V[f'{tag}_own_n_max'] = f"{min(s_['n'] for s_ in S0):,}", f"{max(s_['n'] for s_ in S0):,}"
    V[f'{tag}_own_cls_min'], V[f'{tag}_own_cls_max'] = str(min(len(s_['classes']) for s_ in S0)), str(max(len(s_['classes']) for s_ in S0))
    # repetition as the unit (both datasets), and device as the unit (N-BaIoT, fixed partition)
    g = {k: [a['f1'] - b['f1'] for a, b in zip(site_list(F[k]), site_list(Lc[k], True))] for k in reps}
    allg = [x for v in g.values() for x in v]
    V[f'{tag}_own_rep_gain'] = dci([np.mean(v) for v in g.values()])
    V[f'{tag}_own_rep_p'] = fp_(stats.ttest_1samp([np.mean(v) for v in g.values()], 0).pvalue)
    V[f'{tag}_own_pairs_neg'], V[f'{tag}_own_pairs_n'] = str(sum(x < 0 for x in allg)), str(len(allg))
    return F, Lc, reps


own_traffic('TON_IoT', 'TON', 'fedavg_natr', 'local_natr', 'rep')
own_traffic('N_BaIoT', 'NB', 'fedavg_dev', 'local_dev', 'device')
# Table II with own-traffic columns
rows = ['| Setting | Common test macro-F1 | SD | Own-traffic macro-F1 | Own-traffic attack detection (%) | Own-traffic benign FPR (%) |',
        '|---|---|---|---|---|---|']
for e, lab in (('central', 'Centralised'), ('local_natr', 'Local only (mean over sites)'), ('fedavg_natr', 'FedAvg'), ('fedprox_natr', 'FedProx'),
               ('fedadam_natr', 'FedAdam'), ('fedlc_natr', 'FedLC'), ('fedavgsgd_natr', 'FedAvg (SGD)'), ('scaffold_natr', 'SCAFFOLD (SGD)')):
    rr = runs('TON_IoT', e)
    if not rr:
        continue
    k = e.replace('.', '')
    own = {m: [] for m in ('f1', 'attack_detection', 'benign_fpr')}
    for r in rr.values():
        sl = site_list(r, e.startswith('local')) if (r.get('site_selected') or r.get('per_client')) and e != 'central' else []
        for m in own:
            vals = [s_[m] for s_ in sl if s_.get(m) is not None]
            if vals:
                own[m].append(np.mean(vals))
    cells = [mci(own['f1'])[1] if own['f1'] else '–'] + [mci([x * 100 for x in own[m]], 1)[1] if own[m] else '–' for m in ('attack_detection', 'benign_fpr')]
    rows.append(f"| {lab} | {V[f'TON_{k}_sel']} | {V.get(f'TON_{k}_sel_sd', '–')} | " + ' | '.join(cells) + ' |')
tables['TON_nat'] = '\n'.join(rows)

# own traffic without the port features (TON-IoT host partition)
Fn_, Ln_ = runs('TON_IoT', 'fedavg_natr_noports_ft'), runs('TON_IoT', 'local_natr_noports')
reps = [k for k in Fn_ if k in Ln_]
if reps:
    g = {k: [a['f1'] - b['f1'] for a, b in zip(Fn_[k]['site_selected'], site_list(Ln_[k], True))] for k in reps}
    allg = [x for v in g.values() for x in v]
    V['TON_np_own_gain'] = dci([np.mean(v) for v in g.values()])
    V['TON_np_own_p'] = fp_(stats.ttest_1samp([np.mean(v) for v in g.values()], 0).pvalue)
    V['TON_np_own_neg'], V['TON_np_own_n'] = str(sum(x < 0 for x in allg)), str(len(allg))
    V['TON_np_own_fed'] = f(float(np.mean([s_['f1'] for k in reps for s_ in Fn_[k]['site_selected']])))
    V['TON_np_own_loc'] = f(float(np.mean([s_['f1'] for k in reps for s_ in site_list(Ln_[k], True)])))
    for nm, rr, local in (('fed', Fn_, False), ('loc', Ln_, True)):
        V[f'TON_np_own_{nm}_det'] = f(float(np.mean([s_['attack_detection'] * 100 for k in reps for s_ in site_list(rr[k], local)])), 1)
    d = paired({k: sel(Fn_[k]) for k in reps}, {k: Ln_[k]['test_f1_selected_mean'] for k in reps})
    V['TON_np_common_gain'] = pstr(d) + ', uncorrected'
    V['TON_np_fed_sel'] = mci([sel(Fn_[k]) for k in reps])[1]; V['TON_np_loc_sel'] = mci([Ln_[k]['test_f1_selected_mean'] for k in reps])[1]
    ft1 = {k: [c['1']['site']['f1'] - b['f1'] for c, b in zip(Fn_[k]['finetune'], site_list(Ln_[k], True))] for k in reps}
    V['TON_np_ft1_gain'] = mci([np.mean(v) for v in ft1.values()])[1]
    V['TON_np_ft1_common'] = mci([np.mean([c['1']['test_f1'] for c in Fn_[k]['finetune']]) for k in reps])[1]


# personalisation baselines (same runs supply the federated model): full / low-lr / proximal fine-tuning, and the
# two-model combination (alert if the federated or the adapted model predicts an attack)
def pers_stats(ds, tag, pers, loc, unit):
    P_, Lc = runs(ds, pers), runs(ds, loc)
    reps = [k for k in P_ if k in Lc]
    if not reps:
        return
    nsite = len(P_[reps[0]]['site_selected'])
    loc_site = {k: [b['f1'] for b in site_list(Lc[k], True)] for k in reps}
    fed_site = {k: [a['f1'] for a in P_[k]['site_selected']] for k in reps}
    fed_common = [sel(P_[k]) for k in reps]; loc_common = [Lc[k]['test_f1_selected_mean'] for k in reps]
    V[f'{tag}_pers_fed_common'] = mci(fed_common)[1]; V[f'{tag}_pers_loc_common'] = mci(loc_common)[1]
    agg = (lambda d: [np.mean(v) for v in d.values()]) if unit == 'rep' else \
          (lambda d: [np.mean([d[k][j] for k in reps]) for j in range(nsite)])
    gfed = {k: [a - b for a, b in zip(fed_site[k], loc_site[k])] for k in reps}
    V[f'{tag}_pers_fed_gain'] = mci(agg(gfed))[1]
    for k_, key in (('global_site', 'fed_site'), ('global_test', 'fed_test')):
        for m in ('attack_detection', 'benign_fpr'):
            V[f'{tag}_pers_{key}_{m}'] = f(float(np.mean([P_[k]['finetune'][j][k_][m] * 100 for k in reps for j in range(nsite)
                                                           if P_[k]['finetune'][j][k_][m] is not None])), 1)
    for m in ('attack_detection', 'benign_fpr'):
        V[f'{tag}_pers_loc_site_{m}'] = f(float(np.mean([b[m] * 100 for k in reps for b in site_list(Lc[k], True) if b.get(m) is not None])), 1)
    rows = []
    for mode in ('full', 'low', 'prox'):
        for E in ('1', '2', '5'):
            g = {k: [P_[k]['finetune'][j][mode][E]['site']['f1'] - loc_site[k][j] for j in range(nsite)] for k in reps}
            com = [np.mean([P_[k]['finetune'][j][mode][E]['test_f1'] for j in range(nsite)]) for k in reps]
            give = [(fc - c) / (fc - lc) for fc, c, lc in zip(fed_common, com, loc_common)]
            key = f'{tag}_pers_{mode}{E}'
            V[key + '_gain'] = mci(agg(g))[1]; V[key + '_gain_mean'] = f"{np.mean(agg(g)):+.3f}"
            V[key + '_common'] = mci(com)[1]; V[key + '_giveup'] = f'{100 * np.mean(give):.0f}'
            for where in ('site', 'test'):
                for m in ('attack_detection', 'benign_fpr'):
                    V[f'{key}_either_{where}_{m}'] = f(float(np.mean([P_[k]['finetune'][j][mode][E][f'either_{where}'][m] * 100
                                                                      for k in reps for j in range(nsite)
                                                                      if P_[k]['finetune'][j][mode][E][f'either_{where}'][m] is not None])), 1)
            rows.append((mode, E, V[key + '_gain'], V[key + '_common'], V[key + '_either_site_attack_detection'], V[key + '_either_site_benign_fpr'], V[key + '_either_test_benign_fpr']))
    return rows


LAB = {'full': 'Fine-tuning, same lr', 'low': 'Fine-tuning, lr / 10', 'prox': 'Proximal fine-tuning (μ = 0.1)'}
prow = ['| Dataset | Model | Local epochs | Own-traffic macro-F1 minus local model | Common test macro-F1 | Own-traffic attack detection (%) | Own-traffic benign FPR (%) | Common test benign FPR (%) |',
        '|---|---|---|---|---|---|---|---|']
for ds, tag, pers, loc, unit, nm in (('TON_IoT', 'TON', 'fedavg_natr_pers', 'local_natr', 'rep', 'TON-IoT'),
                                      ('N_BaIoT', 'NB', 'fedavg_dev_pers', 'local_dev', 'device', 'N-BaIoT')):
    rr = pers_stats(ds, tag, pers, loc, unit)
    if rr:
        prow.append(f"| {nm} | Federated model alone | – | {V[f'{tag}_pers_fed_gain']} | {V[f'{tag}_pers_fed_common']} | "
                    f"{V[f'{tag}_pers_fed_site_attack_detection']} | {V[f'{tag}_pers_fed_site_benign_fpr']} | {V[f'{tag}_pers_fed_test_benign_fpr']} |")
        prow.append(f"| {nm} | Local model alone | – | 0 | {V[f'{tag}_pers_loc_common']} | {V[f'{tag}_pers_loc_site_attack_detection']} | {V[f'{tag}_pers_loc_site_benign_fpr']} | – |")
        prow += [f"| {nm} | {LAB[m]} (alert columns: with the federated model) | {E} | {g} | {c} | {d} | {fp} | {ft} |" for m, E, g, c, d, fp, ft in rr]
if len(prow) > 2:
    tables['pers'] = '\n'.join(prow)
diffs = []
for ds, a, b in (('TON_IoT', 'fedavg_natr_pers', 'fedavg_natr'), ('N_BaIoT', 'fedavg_dev_pers', 'fedavg_dev'),
                 ('TON_IoT', 'fedavg_natr_ft', 'fedavg_natr')):
    A, B = runs(ds, a), runs(ds, b)
    diffs += [abs(sel(A[k]) - sel(B[k])) for k in A if k in B]
if diffs:
    V['RERUN_MAXDIFF'] = f(max(diffs)); V['RERUN_N'] = str(len(diffs))

# DP: per-class cost at the full budget, smaller noise levels, one reference for every cost
def pcls(e):
    rr = runs('TON_IoT', e)
    return {c: np.mean([r['test_selected_full']['per_class_f1'][c] for r in rr.values()]) for c in rr[0]['test_selected_full']['per_class_f1']} if rr else {}


NAMES_C = {'mitm': 'MITM', 'xss': 'XSS', 'ddos': 'DDoS', 'dos': 'DoS'}
cap = lambda c: NAMES_C.get(c, c.capitalize())
full0, full18 = pcls('dp_full_nonoise'), pcls('dp_full_sigma1.8')
if full0 and full18:
    att = [c for c in full0 if c != 'normal']
    lost = sorted([c for c in att if full0[c] - full18[c] >= 0.10], key=lambda c: full18[c] - full0[c])
    V['TON_dpfull_nlost'] = str(len(lost)); V['TON_dpfull_natt'] = str(len(att))
    _sp = [cap(c) for c in att + ['normal'] if full0[c] - full18[c] < 0.10]
    V['TON_dpfull_spared'] = (', '.join(_sp[:-1]) + ' and ' + _sp[-1] if len(_sp) > 1 else ''.join(_sp)).replace('Normal', 'benign traffic')
    V['TON_dpfull_worst'] = f"{cap(lost[0])} ({full0[lost[0]]:.2f} to {full18[lost[0]]:.2f})"
s0, ref0 = pcls('dp_noclip_nonoise'), pcls('fedavg_gn_short')
for x, key in (('0.6', '06'), ('2.5', '25')):
    sx = pcls(f'dp_sigma{x}')
    for rname, rd in (('code', s0), ('np', ref0)):
        V[f'TON_dp_nlost_{key}_{rname}'] = str(sum(rd[c] - sx[c] >= 0.10 for c in rd if c != 'normal'))
for sg in ('0.2', '0.3'):
    rr = runs('TON_IoT', f'dp_sigma{sg}')
    if rr:
        k = sg.replace('.', '')
        V[f'TON_dp_sigma{k}_sel'] = mci([sel(r) for r in rr.values()])[1]
        V[f'TON_dp_sigma{k}_eps'] = f"{np.mean([r['dp']['epsilon_max'] for r in rr.values()]):.0f}"
        V[f'TON_dp_cost_short_{k}'] = f"{100 * (np.mean([sel(r) for r in runs('TON_IoT', 'fedavg_gn_short').values()]) - np.mean([sel(r) for r in rr.values()])):.0f}"
V['TON_dp_cost_full_np'] = f"{100 * (float(V['TON_central_sel_mean']) - float(V['TON_dp_full_sigma18_sel'])):.0f}" if 'TON_central_sel_mean' in V else 'n/a'
V['TON_fedavg_a05_sel_mean_dp'] = V.get('TON_fedavg_a05_sel_mean', 'n/a')
V['TON_dp_cost_full_vs_fedavg'] = f"{100 * (float(V['TON_fedavg_a05_sel_mean']) - float(V['TON_dp_full_sigma18_sel'])):.0f}"
for c in ('password', 'backdoor', 'ransomware', 'scanning', 'mitm'):
    V[f'TON_ntrain_{c}'] = f"{TON_TRAIN_COUNTS.get(c, 0):,}"
# DP table on TON-IoT with every configuration in one place (± = 95% CI half-width)
rows = ['| Setting | Rounds × epochs | Noise σ | ε (δ = 10⁻⁵) | Test macro-F1, selected round | Test macro-F1, final round |', '|---|---|---|---|---|---|']
lab = {'fedavg_gn_short': 'Non-private FedAvg', 'dp_noclip_nonoise': 'DP code, no clipping or noise', 'dp_clip_nonoise': 'DP code, clipping only'}
for e in ('fedavg_gn_short', 'dp_noclip_nonoise', 'dp_clip_nonoise', 'dp_sigma0.2', 'dp_sigma0.3', 'dp_sigma0.6', 'dp_sigma1', 'dp_sigma1.5', 'dp_sigma2.5',
          'fedavg_a0.5', 'dp_full_nonoise', 'dp_full_sigma1.8'):
    rr = runs('TON_IoT', e)
    if not rr:
        continue
    full = e.startswith('dp_full') or e == 'fedavg_a0.5'
    sig = r0 = next(iter(rr.values()))
    sg = str(r0['cfg'].get('sigma', 0)) if e.startswith('dp') else '–'
    eps = [r['dp']['epsilon_max'] for r in rr.values() if r.get('dp') and r['dp'].get('epsilon_max')]
    nd_ = 0 if eps and np.mean(eps) > 20 else 2
    eps_s = f"{np.mean(eps):.{nd_}f} ({min(eps):.{nd_}f}–{max(eps):.{nd_}f})" if eps else ('∞' if e.startswith('dp') else '–')
    name = 'Non-private FedAvg' if e == 'fedavg_a0.5' else lab.get(e, 'DP-SGD, C = 1') if e != 'dp_full_nonoise' else 'DP code, no clipping or noise'
    rows.append(f"| {name} | {'50 × 2' if full else '20 × 1'} | {sg if sg != '0' and sg != '0.0' else '0'} | {eps_s} | "
                f"{mci([sel(r) for r in rr.values()])[1]} | {mci([fin(r) for r in rr.values()])[1]} |")
tables['TON_dp'] = '\n'.join(rows)

# minimum detectable paired difference (80% power, two-sided 5%, n = 5): (t_.975 + t_.80) / t_.975 x CI half-width,
# median over the primary algorithm comparisons at alpha = 0.5 and 0.1
for ds, tag in DS.items():
    hs = []
    for alg in ('fedprox', 'fedadam', 'fedlc'):
        for a in ('a0.5', 'a0.1'):
            d = paired(selmap(ds, f'{alg}_{a}'), selmap(ds, f'fedavg_{a}'))
            if d:
                hs.append((d['hi'] - d['lo']) / 2)
    if hs:
        fac = (stats.t.ppf(0.975, 4) + stats.t.ppf(0.80, 4)) / stats.t.ppf(0.975, 4)
        V[f'{tag}_mde'] = f'{fac * np.median(hs):.2f}'
# federation-vs-reference comparisons significant after Holm (all datasets)
fr = [k for k in V if k.endswith('_sig_holm') and ('_vs_central' in k or '_vs_local' in k) and '_d_fedavg' in k]
V['n_fedref_sig'], V['n_fedref'] = str(sum(V[k] == 'True' for k in fr)), str(len(fr))
# feature-vector duplicates between training and test after feature selection (pipeline-level disclosure)
for ds, tag in DS.items():
    pipe = joblib.load(PIPE / 'experiments' / 'baseline' / 'results' / ds / 'pipeline.pkl')
    with contextlib.redirect_stdout(io.StringIO()):
        raw_ = load_dataset(*BD[ds], seed=42)
    Xtr_ = pipe.transform(raw_['X_train'])[0].astype(np.float32); Xte_ = pipe.transform(raw_['X_test'])[0].astype(np.float32)
    h = lambda X: pd.util.hash_pandas_object(pd.DataFrame(X), index=False).values
    tr_lab = {}
    for kk, yy in zip(h(Xtr_), raw_['y_train']):
        tr_lab.setdefault(kk, set()).add(int(yy))
    hte = h(Xte_)
    seen = np.array([kk in tr_lab for kk in hte])
    conf = sum(1 for kk, yy in zip(hte, raw_['y_test']) if kk in tr_lab and int(yy) not in tr_lab[kk])
    V[f'{tag}_test_featdup'] = f'{100 * seen.mean():.0f}'; V[f'{tag}_test_featdup_conf'] = f'{conf:,}'

# N-BaIoT preparation counts (round 6)
V['NB_guard_val'], V['NB_guard_test'] = f"{NBI.get('n_guard_val', 0):,}", f"{NBI.get('n_guard_test', 0):,}"
V['NB_dropped'] = ' and '.join(f"{c.replace('gafgyt_', 'BASHLITE ').upper() if False else c.replace('gafgyt_', 'BASHLITE ')} ({n} distinct records)" for c, n in NBI.get('dropped_classes', {}).items())
# Table S4/S5 (TON_extra, UNSW_extra): add the five-repetition BatchNorm runs with both kinds of statistics
for ds, tag in DS.items():
    if f'{tag}_bnfr_a05_n' not in V:
        continue
    extra = [f"| FedAvg, BatchNorm, α = {a}, server-averaged statistics (five repetitions) | {V[f'{tag}_bnfr_a{ak}_run_sel']} | {V[f'{tag}_bnfr_a{ak}_run_fin']} | – |"
             for a, ak in (('0.5', '05'), ('0.1', '01'))]
    extra += [f"| FedAvg, BatchNorm, α = {a}, federated re-estimation (five repetitions) | {V[f'{tag}_bnfr_a{ak}_sel']} | {V[f'{tag}_bnfr_a{ak}_fin']} | – |"
              for a, ak in (('0.5', '05'), ('0.1', '01'))]
    tables[f'{tag}_extra'] = tables[f'{tag}_extra'].replace('BatchNorm, α = 0.5 |', 'BatchNorm, α = 0.5, server-averaged statistics (three repetitions) |') \
        .replace('BatchNorm, α = 0.1 |', 'BatchNorm, α = 0.1, server-averaged statistics (three repetitions) |') + '\n' + '\n'.join(extra)
# ══ final review round (review_final_*.md, 2026-11-07) ═════════════════════════════
# re-run noise: distribution over all re-run pairs, not only the maximum (referee M6)
if diffs:
    sdiff = []
    for ds, a, b in (('TON_IoT', 'fedavg_natr_pers', 'fedavg_natr'), ('N_BaIoT', 'fedavg_dev_pers', 'fedavg_dev'),
                     ('TON_IoT', 'fedavg_natr_ft', 'fedavg_natr')):
        A, B = runs(ds, a), runs(ds, b)
        sdiff += [sel(A[k]) - sel(B[k]) for k in A if k in B]
    V['RERUN_MEANABS'] = f(float(np.mean(diffs))); V['RERUN_SD'] = f(float(np.std(sdiff, ddof=1)))
# the full-budget DP gap as a paired difference (same seeds, same partitions; referee M3)
d = paired({k: sel(r) for k, r in runs('TON_IoT', 'dp_full_nonoise').items()},
           {k: sel(r) for k, r in runs('TON_IoT', 'dp_full_sigma1.8').items()})
if d:
    V['TON_dp_full_gap_ci'] = f"{d['mean']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"
# equivalence check (TOST at ±0.05, i.e. 90% CI inside ±0.05) for the primary algorithm comparisons (referee M4)
def ci90(a, b):
    k = sorted(set(a) & set(b)); x = np.array([a[i] - b[i] for i in k])
    h = stats.t.ppf(0.95, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h
n_eq = n_alg = 0
for ds, parts in (('TON_IoT', ('a0.5', 'a0.1', 'natr')), ('UNSW_NB15', ('a0.5', 'a0.1')), ('N_BaIoT', ('dev',))):
    for p_ in parts:
        for alg, ref in (('fedprox', 'fedavg'), ('fedadam', 'fedavg'), ('fedlc', 'fedavg'), ('scaffold', 'fedavgsgd')):
            A, B = selmap(ds, f'{alg}_{p_}'), selmap(ds, f'{ref}_{p_}')
            if len(set(A) & set(B)) > 1:
                lo_, hi_ = ci90(A, B); n_alg += 1; n_eq += (lo_ > -0.05 and hi_ < 0.05)
V['n_alg_cmp'], V['n_alg_equiv'] = str(n_alg), str(n_eq)
# per-class details quoted in the DP text
if full0 and full18:
    V['TON_dpfull_ddos_loss'] = f"{100 * (full0['ddos'] - full18['ddos']):.1f}"
for e in ('fedlc_a05', 'fedavg_a05'):
    if f'UNSW_{e}_fpr' in V:
        V[f'UNSW_{e}_fpr_m'] = f"{float(V[f'UNSW_{e}_fpr'].split()[0]):.1f}"
# N-BaIoT local models on their own traffic (Table S3 column, mean over repetitions per device)
_lo = [np.mean([NR_['local_dev'][k]['per_client'][j]['site_selected'][0]['f1'] for k in NR_['local_dev']])
       for j in range(len(NBI['devices']))] if NR_.get('local_dev') else []
if _lo:
    V['NB_loc_own_min'], V['NB_loc_own_max'] = f'{min(_lo):.2f}', f'{max(_lo):.2f}'
for sg, k in (('0.2', '02'), ('0.3', '03'), ('1', '1')):
    pc_ = pcls(f'dp_sigma{sg}')
    if pc_:
        V[f'TON_mitm_s{k}'] = f(pc_['mitm'], 2)

# diagnostic runs of the final round (run_p4final.py; results_p4final). Keys are written only when the runs exist, so
# the text shows the pending placeholders P4F_* until then.
P4F = PIPE / 'experiments' / 'paper4_federated' / 'results_p4final'


def p4runs(ds, e):
    d_ = P4F / ds / e
    return {int(p.stem[4:]): json.loads(p.read_text()) for p in sorted(d_.glob('seed*.json'))} if d_.exists() else {}


def site_pairs(F, Lr, k, field=None):
    """own-traffic macro-F1 differences FedAvg (optionally label-prior-corrected) minus local, per site."""
    out = []
    for a, pc in zip(F[k]['site_selected'], Lr[k]['per_client']):
        b = (pc.get('site_selected') or [{}])[0]
        if a.get('n') and b.get('n'):
            out.append((a[field]['f1'] if field else a['f1']) - b['f1'])
    return out


GAINS = {}
for ds, tag, unit in (('TON_IoT', 'TON', 'hosts'), ('N_BaIoT', 'NB', 'dev')):
    for split, key in (('blocked', 'blk'), ('random', 'rnd')):
        F, Lr = p4runs(ds, f'fedavg_{unit}_{split}'), p4runs(ds, f'local_{unit}_{split}')
        reps = sorted(set(F) & set(Lr))
        if len(reps) < 2:
            continue
        g = {k: site_pairs(F, Lr, k) for k in reps}
        gp = {k: site_pairs(F, Lr, k, 'prior') for k in reps}
        pre = f'P4F_{tag}_{key}'
        V[pre + '_n'] = str(len(reps))
        V[pre + '_own_gain'] = dci([np.mean(v) for v in g.values()])
        GAINS[(tag, key)] = {k: float(np.mean(v)) for k, v in g.items()}
        V[pre + '_prior_gain'] = dci([np.mean(v) for v in gp.values()])
        V[pre + '_pairs_neg'] = str(sum(x < 0 for v in g.values() for x in v)); V[pre + '_pairs_n'] = str(sum(len(v) for v in g.values()))
        V[pre + '_common_gain'] = mci([sel(F[k]) - Lr[k]['test_f1_selected_mean'] for k in reps])[1]
        V[pre + '_fed_f1'] = f(float(np.mean([s_['f1'] for k in reps for s_ in F[k]['site_selected'] if s_.get('n')])))
        V[pre + '_loc_f1'] = f(float(np.mean([pc['site_selected'][0]['f1'] for k in reps for pc in Lr[k]['per_client']
                                               if (pc.get('site_selected') or [{}])[0].get('n')])))
    # label-prior correction on the paper's random split: FedAvg re-runs, local models of results_v2 (same partitions)
    F = p4runs(ds, 'fedavg_natr_prior' if tag == 'TON' else 'fedavg_dev_prior')
    Lr = runs(ds, 'local_natr' if tag == 'TON' else 'local_dev')
    reps = sorted(set(F) & set(Lr))
    if len(reps) >= 2:
        for field, nm in ((None, 'fed'), ('mask', 'mask'), ('prior', 'prior')):
            V[f'P4F_{tag}_orig_{nm}_gain'] = dci([np.mean(site_pairs(F, Lr, k, field)) for k in reps])
        V[f'P4F_{tag}_orig_n'] = str(len(reps))
for tag in ('TON', 'NB'):
    if (tag, 'blk') in GAINS and (tag, 'rnd') in GAINS:
        V[f'P4F_{tag}_blk_vs_rnd'] = pstr(paired(GAINS[(tag, 'blk')], GAINS[(tag, 'rnd')])) + ', uncorrected'
        V[f'P4F_{tag}_blk_shrink'] = f"{100 * (1 - np.mean(list(GAINS[(tag, 'blk')].values())) / np.mean(list(GAINS[(tag, 'rnd')].values()))):.0f}"
# DP optimiser and batch-size check at matched epsilon (TON-IoT, alpha = 0.5)
ref1 = {k: sel(r) for k, r in runs('TON_IoT', 'dp_sigma1').items()}
np_short = float(np.mean([sel(r) for r in runs('TON_IoT', 'fedavg_gn_short').values()]))
sg_runs = {lr: p4runs('TON_IoT', f'dp_sgdm_lr{lr}') for lr in ('0.03', '0.1')}
sg_runs = {lr: rr for lr, rr in sg_runs.items() if len(rr) >= 2}
if sg_runs:
    best = max(sg_runs, key=lambda lr: np.mean([r['val_f1_selected'] for r in sg_runs[lr].values()]))
    for lr, rr in sg_runs.items():
        V[f"P4F_dp_sgdm_lr{lr.replace('.', '')}_sel"] = mci([sel(r) for r in rr.values()])[1]
    rr = sg_runs[best]
    V['P4F_dp_sgdm_lr'] = best; V['P4F_dp_sgdm_sel'] = mci([sel(r) for r in rr.values()])[1]
    V['P4F_dp_sgdm_cost'] = f"{100 * (np_short - np.mean([sel(r) for r in rr.values()])):.0f}"
    V['P4F_d_sgdm_vs_adam'] = pstr(paired({k: sel(r) for k, r in rr.items()}, ref1)) + ', uncorrected'
rr = p4runs('TON_IoT', 'dp_b1024')
if len(rr) >= 2:
    V['P4F_dp_b1024_sel'] = mci([sel(r) for r in rr.values()])[1]
    V['P4F_dp_b1024_sigma'] = f"{np.mean([r['dp']['sigma'] for r in rr.values()]):.2f}"
    V['P4F_dp_b1024_cost'] = f"{100 * (np_short - np.mean([sel(r) for r in rr.values()])):.0f}"
    V['P4F_d_b1024_vs_adam'] = pstr(paired({k: sel(r) for k, r in rr.items()}, ref1)) + ', uncorrected'
    V['P4F_dp_eps'] = f"{np.mean([r['dp']['epsilon_max'] for r in rr.values()]):.1f}"
full_ref = {k: sel(r) for k, r in runs('TON_IoT', 'dp_full_sigma1.8').items()}
for e, k in (('dpfull_sgdm', 'sgdm'), ('dpfull_b1024', 'b1024')):
    rr = p4runs('TON_IoT', e)
    if len(rr) >= 2:
        V[f'P4F_dpfull_{k}_sel'] = mci([sel(r) for r in rr.values()])[1]
        V[f'P4F_dpfull_d_{k}_vs_adam'] = pstr(paired({kk: sel(r) for kk, r in rr.items()}, full_ref)) + ', uncorrected'

V['tables'] = tables
(HERE / 'values.json').write_text(json.dumps(V, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'{len(V) - 1} values, {len(tables)} tables → values.json')
