"""Figures for the Paper 4 v2 manuscript, drawn from results_v2 JSON."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / 'Papers pipeline' / 'experiments' / 'paper4_federated' / 'results_v2'
OUT = HERE / 'figures'; OUT.mkdir(exist_ok=True)
DS = [('TON_IoT', 'TON-IoT'), ('UNSW_NB15', 'UNSW-NB15')]
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False})


def runs(ds, e):
    return [json.loads(p.read_text()) for p in sorted((RES / ds / e).glob('seed*.json'))]


def sel(r):
    return r.get('test_f1_selected', r.get('test_f1_selected_mean'))


def mci(v):
    v = np.asarray(v, float)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
    return v.mean(), h


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches='tight'); plt.close(fig); print('saved', name)


# Fig 1: forest plot of paired differences (same repetition = same partition), 95% t-CI
def pdiff(ds, a, b):
    A = {int(p.stem[4:]): sel(json.loads(p.read_text())) for p in (RES / ds / a).glob('seed*.json')}
    B = {int(p.stem[4:]): sel(json.loads(p.read_text())) for p in (RES / ds / b).glob('seed*.json')}
    k = sorted(set(A) & set(B))
    if len(k) < 2:
        return None
    d = np.array([A[i] - B[i] for i in k]); h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), h


# experiment names per dataset; None = not available on that dataset
NAMES = {
    'TON_IoT': dict(iid='fedavg_iid', a5='fedavg_a0.5', a1='fedavg_a0.1', dev='fedavg_natr', loc5='local_a0.5', loc1='local_a0.1', locd='local_natr'),
    'UNSW_NB15': dict(iid='fedavg_iid', a5='fedavg_a0.5', a1='fedavg_a0.1', dev=None, loc5='local_a0.5', loc1='local_a0.1', locd=None),
    'N_BaIoT': dict(iid='fedavg_iid9', a5='fedavg_a0.5_N9', a1='fedavg_a0.1_N9', dev='fedavg_dev', loc5=None, loc1=None, locd='local_dev'),
}
ALG = {'TON_IoT': dict(a5='a0.5', a1='a0.1', dev='natr'), 'UNSW_NB15': dict(a5='a0.5', a1='a0.1', dev=None), 'N_BaIoT': dict(a5=None, a1=None, dev='dev')}


def row_exps(ds, kind, part):
    n = NAMES[ds]
    if kind == 'central':
        return n[part], 'central'
    if kind == 'local':
        return n[part], n[{'a5': 'loc5', 'a1': 'loc1', 'dev': 'locd'}[part]]
    suf = ALG[ds][part]
    if suf is None:
        return None, None
    if kind == 'scaffold':
        return f'scaffold_{suf}', f'fedavgsgd_{suf}'
    return f'{kind}_{suf}', f'fedavg_{suf}'


PL = {'iid': 'IID', 'a5': 'α = 0.5', 'a1': 'α = 0.1', 'dev': 'device'}
ROWS = [(f'FedAvg − centralised, {PL[p]}', 'central', p) for p in ('iid', 'a5', 'a1', 'dev')] + \
       [(f'FedAvg − local only, {PL[p]}', 'local', p) for p in ('a5', 'a1', 'dev')] + \
       [(f'{lab} − {ref}, {PL[p]}', k, p) for k, lab, ref in (('fedprox', 'FedProx', 'FedAvg'), ('fedadam', 'FedAdam', 'FedAvg'), ('fedlc', 'FedLC', 'FedAvg'),
                                                             ('scaffold', 'SCAFFOLD', 'FedAvg-SGD')) for p in ('a5', 'a1', 'dev')]
DS3 = [('TON_IoT', 'TON-IoT', '#1f5fa8'), ('UNSW_NB15', 'UNSW-NB15', '#d1603d'), ('N_BaIoT', 'N-BaIoT', '#3a8a4f')]
VALS = json.loads((HERE / 'values.json').read_text(encoding='utf-8'))
TAG = {'TON_IoT': 'TON', 'UNSW_NB15': 'UNSW', 'N_BaIoT': 'NB'}
# single-column figures (IEEE column width 3.5 in), 7-8 pt type at print size
fig, ax = plt.subplots(figsize=(3.5, 6.2))
yy = np.arange(len(ROWS))
for off, (ds, lab, col) in zip((-0.24, 0, 0.24), DS3):
    pts, sig = [], []
    for _, kind, part in ROWS:
        a, b = row_exps(ds, kind, part)
        ok_ = a and b and (RES / ds / a).exists() and (RES / ds / b).exists()
        pts.append(pdiff(ds, a, b) if ok_ else None)
        sig.append(ok_ and VALS.get(f"{TAG[ds]}_d_{a.replace('.', '')}_vs_{b.replace('.', '')}_sig_holm") == 'True')
    ok = [i for i, p in enumerate(pts) if p]
    ax.errorbar([pts[i][0] for i in ok], yy[ok] + off, xerr=[pts[i][1] for i in ok], fmt='o', ms=2.6, color=col, capsize=1.5, lw=0.9, label=lab)
    for i in ok:
        if sig[i]:
            ax.text(pts[i][0] + pts[i][1] + 0.008, yy[i] + off, '†', color=col, fontsize=7, va='center')
ax.axvline(0, color='k', lw=0.7)
for y in (3.5, 6.5, 9.5, 12.5, 15.5):
    ax.axhline(y, color='#cccccc', lw=0.5)
ax.set_yticks(yy, [r[0] for r in ROWS], fontsize=7); ax.invert_yaxis()
ax.tick_params(axis='x', labelsize=7)
ax.set_xlabel('Paired difference in test macro-F1\n(mean and 95% CI; † significant after Holm)', fontsize=7.5)
ax.legend(frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(0.35, -0.09), ncol=3, handletextpad=0.2, columnspacing=0.8)
fig.tight_layout(); save(fig, 'fig1_paired_differences.png')

# Fig 2: validation macro-F1 per round, mean over repetitions with min-max band (panels stacked)
fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.4))
CURVES = (('fedavg_a0.5', 'val_f1', '#1f5fa8', '-', 'FedAvg, GroupNorm', 0.15),
          ('fedavg_bn_a0.5_fedrecal', 'val_f1', '#d1603d', '-', 'FedAvg, BatchNorm, server-averaged statistics', 0.15),
          ('fedavg_bn_a0.5_fedrecal', 'val_f1_fedrecal', '#3a8a4f', '--', 'FedAvg, BatchNorm, federated re-estimation', 0.15),
          ('central_bn', 'val_f1', '#777777', ':', 'Centralised, BatchNorm', 0.06))
for ax, (ds, lab) in zip(axes, DS):
    for e, field, col, ls, name, al in CURVES:
        rr = runs(ds, e)
        if not rr:
            continue
        M = np.array([[x[field] for x in r['history']] for r in rr])
        x = np.arange(1, M.shape[1] + 1)
        ax.plot(x, M.mean(0), color=col, ls=ls, lw=1.0, label=f'{name} ({len(rr)} rep.)')
        ax.fill_between(x, M.min(0), M.max(0), color=col, alpha=al, lw=0)
    ax.set_ylabel('Validation macro-F1', fontsize=7.5); ax.set_title(lab, loc='left', fontsize=8); ax.tick_params(labelsize=7)
axes[1].set_xlabel('Round (centralised: 2 epochs)', fontsize=7.5)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=6.5, loc='lower center', ncol=1, bbox_to_anchor=(0.5, -0.17))
fig.tight_layout(); save(fig, 'fig2_normalisation.png')

# Fig 3 (TON-IoT) and Fig. S1 (UNSW-NB15): per-class test F1 under DP
NICE = {'mitm': 'MITM', 'xss': 'XSS', 'ddos': 'DDoS', 'dos': 'DoS', 'normal': 'Benign'}


def dp_heat(ds, exps, name, title):
    P = {e: [r['test_selected_full']['per_class_f1'] for r in runs(ds, e)] for e, _ in exps}
    P = {e: v for e, v in P.items() if v}
    exps = [(e, n) for e, n in exps if e in P]
    classes = [c for c in P[exps[0][0]][0] if c != 'normal'] + ['normal']
    M = np.array([[np.mean([p[c] for p in P[e]]) for c in classes] for e, _ in exps])
    fig, ax = plt.subplots(figsize=(3.5, 0.32 * len(exps) + 1.2))
    ax.imshow(M, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), [NICE.get(c, c.capitalize()) for c in classes], rotation=55, ha='right', fontsize=7)
    ax.set_yticks(range(len(exps)), [n for _, n in exps], fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f'{M[i, j]:.2f}'.lstrip('0') if M[i, j] < 0.995 else '1', ha='center', va='center', fontsize=5.8,
                    color='white' if M[i, j] > 0.6 else 'black')
    ax.set_title(title, loc='left', fontsize=8)
    fig.tight_layout(); save(fig, name)


SHORT = [('dp_noclip_nonoise', '20×1, no clipping, no noise'), ('dp_clip_nonoise', '20×1, clipping only'), ('dp_sigma0.2', '20×1, σ = 0.2'),
         ('dp_sigma0.6', '20×1, σ = 0.6'), ('dp_sigma1', '20×1, σ = 1.0'), ('dp_sigma1.5', '20×1, σ = 1.5'), ('dp_sigma2.5', '20×1, σ = 2.5')]
dp_heat('TON_IoT', SHORT + [('dp_full_nonoise', '50×2, no clipping, no noise'), ('dp_full_sigma1.8', '50×2, σ = 1.8')],
        'fig3_dp_per_class.png', 'TON-IoT: per-class test F1 (rounds × local epochs)')
dp_heat('UNSW_NB15', [x for x in SHORT if x[0] != 'dp_sigma0.2'], 'figS1_dp_per_class_unsw.png', 'UNSW-NB15: per-class test F1')
