"""Paper 1 figures. Fig. 1: test macro-F1 against model size (log scale), one panel per dataset, from points.json
(written by compute_values.py). One hue and marker per model family; int8 networks use open markers of the same hue
as their float32 version. Error bars: SD over five seeds. Dashed line: best mean minus 0.01 (the 'within 0.01' rule
of Section 4.4). The earlier version's forest configuration is ringed.
Palette: validated categorical slots (dataviz validator, light surface, same set as Paper 3); LR is the neutral grey.
Usage: python make_figures.py [points.json]"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
_args = [a for a in sys.argv[1:] if not a.startswith('--')]
PTS = Path(_args[0]) if _args else HERE / 'points.json'
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
DSN = {'TON_IoT': 'TON-IoT', 'UNSW_NB15': 'UNSW-NB15', 'N_BaIoT': 'N-BaIoT'}
FAM = {   # family -> (label, colour, marker)
    'rf': ('Random forest', '#2a78d6', 'o'),
    'dt': ('Decision tree', '#2a78d6', 'v'),
    'gb': ('GBDT (rolled back)', '#eb6834', 's'),
    'gbc': ('GBDT, capped iterations', '#eb6834', 'X'),
    'cnn': ('1D-CNN', '#1baf7a', '^'),
    'mlp': ('MLP', '#6250d6', 'D'),
    'lr': ('Logistic regression', '#5f5e5a', 'P'),
}
INK, MUTED, GRID = '#1f1e1c', '#6b6a65', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': 0.6})


def family(cfg):
    if cfg.startswith('gbc'):
        return 'gbc'
    return 'gb' if cfg.startswith('gb') else cfg.split('_')[0]


P = json.loads(PTS.read_text(encoding='utf-8'))
if not P.get('_final') and '--force' not in sys.argv:
    # the GBDT points would come from the v2 runs (evaluated 20 iterations past their best point): keep the old figure
    sys.exit('points.json is not from the final (rolled-back) runs: figure not regenerated (use --force for a draft)')
dss = [d for d in DSN if P.get(d)]
if not dss:
    sys.exit('no points yet: run compute_values.py after the Paper 1 runs')
fig, axes = plt.subplots(1, len(dss), figsize=(6.9, 2.9), squeeze=False)
for ax, ds in zip(axes[0], dss):
    pts = P[ds]       # final points: every GBDT is rolled back, both learning rates and the capped GBDTs are plotted
    best = max(p['f1'] for p in pts)
    # y-range from the non-LR models; a far-off LR point is drawn on the bottom edge with its value
    lo = min(p['f1'] - p['sd'] for p in pts if p['cfg'] != 'lr')
    ymin, ymax = max(0.0, lo - 0.02), min(1.0, best + 0.02) + 0.005
    ax.axhline(best - 0.01, color=MUTED, lw=0.8, ls='--', zorder=1)
    for p in pts:
        lab, col, mk = FAM[family(p['cfg'])]
        x, y = p['size'] / 1024, p['f1']
        if p['cfg'] == 'lr' and y < ymin:
            ax.plot(x, ymin, marker='v', ms=6, ls='none', color=col, clip_on=False, zorder=3)
            ax.annotate(f'LR {y:.2f}', (x, ymin), xytext=(5, 4), textcoords='offset points', fontsize=7, color=INK)
            continue
        ax.errorbar(x, y, yerr=p['sd'], fmt='none', ecolor=col, elinewidth=0.8, capsize=0, zorder=2)
        ax.plot(x, y, marker=mk, ms=5.5, ls='none', zorder=3, mec=col, mew=1.1, mfc='white' if p['q'] else col)
        if p['cfg'] == 'rf_v1':
            ax.plot(x, y, marker='o', ms=11, mfc='none', mec=INK, mew=0.8, ls='none', zorder=4)
    ax.set_xscale('log')
    ax.set_title(DSN[ds], fontsize=9, color=INK, pad=6)
    ax.set_xlabel('Model size (KB, log scale)')
    ax.set_ylim(ymin, ymax)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
axes[0][0].set_ylabel('Test macro-F1')
h = [Line2D([], [], marker=mk, color=col, mfc=col, ls='none', ms=5.5, label=lab) for lab, col, mk in FAM.values()]
h += [Line2D([], [], marker='^', color=MUTED, mfc='white', ls='none', ms=5.5, label='int8 (open marker)'),
      Line2D([], [], color=MUTED, ls='--', lw=0.8, label='Best − 0.01'),
      Line2D([], [], marker='o', mfc='none', mec=INK, ls='none', ms=10, label='Earlier forest')]
fig.legend(handles=h, loc='lower center', ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=1.2)
out = FIG / ('fig1_accuracy_size.png' if PTS == HERE / 'points.json' else 'fig1_accuracy_size_TEST.png')
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
print('saved', out)

# ── final review round 2: Figures 2-4 from figdata.json (compute_values.py) ──────────────────────────────────────
FD_PATH = HERE / 'figdata.json'
if PTS == HERE / 'points.json' and FD_PATH.exists():
    import numpy as np  # noqa: E402
    FD = json.loads(FD_PATH.read_text(encoding='utf-8'))

    def _style(ax):
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

    # Fig. 2: flows per second on one core (single-record calls) against test macro-F1
    dss2 = [d for d in ('TON_IoT', 'UNSW_NB15') if d in FD['lat']]
    fig, axes = plt.subplots(1, len(dss2), figsize=(6.4, 3.0), squeeze=False)
    for ax, ds in zip(axes[0], dss2):
        for p in P[ds]:
            ms = FD['lat'][ds].get(p['cfg'] + ('_int8' if p['q'] else ''))
            if not ms or family(p['cfg']) not in FAM:
                continue
            lab, col, mk = FAM[family(p['cfg'])]
            ax.scatter(1000.0 / ms, p['f1'], marker=mk, s=22, color=col if not p['q'] else 'white', edgecolors=col, lw=0.8, zorder=3)
        ax.set_xscale('log'); ax.set_title(DSN[ds], fontsize=9, color=INK, pad=6)
        ax.set_xlabel('Flows per second, one core, one record per call'); _style(ax)
        lo = max(min(p['f1'] for p in P[ds]), max(p['f1'] for p in P[ds]) - 0.12)
        ax.set_ylim(lo, max(p['f1'] for p in P[ds]) + 0.005)
    axes[0][0].set_ylabel('Test macro-F1')
    h = [Line2D([], [], marker=mk, color=col, mfc=col, ls='none', ms=5.5, label=lab) for lab, col, mk in FAM.values()]
    h += [Line2D([], [], marker='^', color=MUTED, mfc='white', ls='none', ms=5.5, label='int8 (open marker)')]
    fig.legend(handles=h, loc='lower center', ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=1.2)
    fig.savefig(FIG / 'fig2_throughput.png', dpi=300, bbox_inches='tight', facecolor='white'); plt.close(fig)
    print('saved fig2_throughput.png')

    # Fig. 3: data-split replicates
    RSL = {'best': ('Most accurate tree model', '#eb6834', 's', '-'), 'oversig': ('Smallest tree model significantly above every network', '#eb6834', 'X', ':'),
           'small': ('Best tree model of at most 110 KB', '#2a78d6', 'o', ':'), 'net': ('Most accurate network', '#1baf7a', '^', '-')}
    dss3 = [d for d in ('TON_IoT', 'UNSW_NB15') if d in FD['rs']]
    fig, axes = plt.subplots(1, len(dss3), figsize=(6.4, 3.0), squeeze=False)
    for ax, ds in zip(axes[0], dss3):
        for lab_k, (lab, col, mk, ls) in RSL.items():
            r = FD['rs'][ds].get(lab_k)
            if not r:
                continue
            y = [r['split0']] + r['splits']
            ax.plot(range(6), y, color=col, marker=mk, ls=ls, lw=1.3, ms=5, mfc=col if lab_k != 'small' else 'white', mec=col, label=lab)
        ax.set_xticks(range(6), ['0\n(5 seeds)', '1', '2', '3', '4', '5'])
        ax.set_xlabel('Data split'); ax.set_title(DSN[ds], fontsize=9, color=INK, pad=6); _style(ax)
    axes[0][0].set_ylabel('Test macro-F1')
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.16, 1, 1), w_pad=1.2)
    fig.savefig(FIG / 'fig3_resplits.png', dpi=300, bbox_inches='tight', facecolor='white'); plt.close(fig)
    print('saved fig3_resplits.png')

    # Fig. 4: per-class F1, most accurate tree model against the CNN of width 2
    dss4 = [d for d in ('TON_IoT', 'UNSW_NB15') if d in FD['pc']]
    fig, axes = plt.subplots(1, len(dss4), figsize=(6.4, 3.2), squeeze=False)
    for ax, ds in zip(axes[0], dss4):
        c = FD['pc'][ds]; cls = c['classes']
        t, n = np.array(c['tree_f1']), np.array(c['net_f1'])
        o = np.argsort(t.mean(0) - n.mean(0))
        y = np.arange(len(cls))
        for arr, col, mk, lab in ((n, '#1baf7a', '^', 'CNN of width 2 (float32)'), (t, '#eb6834', 's', 'Most accurate tree model')):
            ax.errorbar(arr.mean(0)[o], y, xerr=[arr.mean(0)[o] - arr.min(0)[o], arr.max(0)[o] - arr.mean(0)[o]], fmt=mk, color=col,
                        ms=4.5, elinewidth=0.8, capsize=0, label=lab, zorder=3)
        ax.hlines(y, np.minimum(t.mean(0), n.mean(0))[o], np.maximum(t.mean(0), n.mean(0))[o], color=GRID, lw=2.5, zorder=1)
        ax.set_yticks(y, [cls[i] for i in o], fontsize=7)
        ax.set_xlabel('Test F1 (mean over seeds; bars: range)'); ax.set_title(DSN[ds], fontsize=9, color=INK, pad=6); _style(ax)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=2.0)
    fig.savefig(FIG / 'fig4_perclass.png', dpi=300, bbox_inches='tight', facecolor='white'); plt.close(fig)
    print('saved fig4_perclass.png')
