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
