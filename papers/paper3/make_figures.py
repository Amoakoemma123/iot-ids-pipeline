"""Paper 3 figures. Fig. 1: what happens to the oldest task (Task 0) as later tasks are learned, at class level
(mean recall of Task 0's attack classes) and at detection level (share of Task 0's attack records flagged as any
attack), next to the benign false-positive rate. Semantic order, mean over five seeds; the band is the seed range.
Palette: validated categorical slots 1, 2, 3, 7 (dataviz validator, light surface: all checks pass; aqua below 3:1
contrast, so every line also has its own marker and the legend sits below the panels); joint retraining is the neutral
reference. LwF is drawn wider underneath fine-tuning, so the two stay visible where they coincide at zero."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1] / 'experiments' / 'paper3_continual'   # repository root (authors' tree: <project>/Papers pipeline)
sys.path.insert(0, str(EXP.parents[1]))
from experiments.paper3_continual.analyze_cl_v2 import metrics  # noqa: E402

RES = EXP / 'results_v3'
RESF = EXP / 'results_p3final'   # corrected A-GEM ('agem2'); the flawed results_v3 'agem' runs are never plotted
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
DSETS = (('TON_IoT', 'TON-IoT'), ('UNSW_NB15', 'UNSW-NB15'), ('N_BaIoT', 'N-BaIoT'))
# strategy, label, colour, marker, linestyle
STR = (('naive', 'Fine-tuning', '#2a78d6', 'o', '-'),
       ('lwf', 'LwF', '#eb6834', 's', '-'),
       ('agem', 'A-GEM', '#1baf7a', '^', '-'),
       ('er2000', 'ER-2000', '#6250d6', 'D', '-'),
       ('joint', 'Joint retraining', '#5f5e5a', 'v', '--'))
INK, MUTED, GRID = '#1f1e1c', '#6b6a65', '#e4e3df'

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6})


AGEM_DONE = all(len(list((RESF / ds / 'main').glob('agem2_o0_s*.json'))) >= 5 for ds, _ in DSETS)
if not AGEM_DONE:   # corrected A-GEM runs pending: leave A-GEM out of the figure (caption marker P3F_fig1_agem)
    STR = tuple(x for x in STR if x[0] != 'agem')


def series(ds, s):
    src, pat = (RESF, 'agem2') if s == 'agem' else (RES, s)
    runs = [metrics(json.loads(p.read_text())) for p in sorted((src / ds / 'main').glob(f'{pat}_o0_s*.json'))]
    assert len(runs) == 5, (ds, s, len(runs))
    cls = np.array([[r['R'][t][0] for t in range(3)] for r in runs])
    det = np.array([[r['Rd'][t][0] for t in range(3)] for r in runs])
    fpr = np.array([r['benign_fpr'] for r in runs])
    return cls, det, fpr


ROWS = ('Task 0, class level\n(mean recall)', 'Task 0, detection level\n(flagged as attack)', 'Benign false-\npositive rate')
fig, axes = plt.subplots(3, 3, figsize=(6.7, 5.6), sharex=True)
data = {(ds, s): series(ds, s) for ds, _ in DSETS for s, *_ in STR}
x = np.arange(3)
FPR_TOP = 1.1 * max(data[k][2].max() for k in data)
DET_LO = np.floor(10 * min(data[k][1].min() for k in data) - 0.5) / 10   # detection row: shared, data-driven lower limit
for c, (ds, dname) in enumerate(DSETS):
    for r in range(3):
        ax = axes[r, c]
        for z, (s, lab, col, mk, ls) in enumerate(STR):
            v = data[(ds, s)][r]
            ax.fill_between(x, v.min(0), v.max(0), color=col, alpha=0.10, lw=0)
            ax.plot(x, v.mean(0), color=col, ls=ls, lw=3.0 if s == 'lwf' else 1.6, marker=mk, ms=4.5, mec='white', mew=0.8, label=lab,
                    zorder=3 + (0 if s == 'lwf' else 1))
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        if r == 0:
            ax.set_ylim(-0.03, 1.03)
        elif r == 1:
            ax.set_ylim(DET_LO, 1.01)   # all values lie well above zero; a shared, truncated axis shows the differences
        else:
            ax.set_ylim(0, FPR_TOP)  # one scale for the whole row, so the near-zero N-BaIoT rates read as such
        if c == 0:
            ax.set_ylabel(ROWS[r], fontsize=8)
        if r == 0:
            ax.set_title(dname, fontsize=9, color=INK, pad=6)
        if r == 2:
            ax.set_xticks(x, ['Task 0', 'Task 1', 'Task 2'])
            ax.set_xlabel('After learning', fontsize=8)
for a in axes.flat:
    a.set_xlim(-0.2, 2.2)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=len(STR), frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.005),
           handlelength=2.2, columnspacing=1.4)
fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=0.8, w_pad=1.0)
out = FIG / 'fig1_task0_retention.png'
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
# caption facts read by build.py (axis start of the detection row, whether the corrected A-GEM is plotted)
(FIG / 'fig1_meta.json').write_text(json.dumps({'fig1_det_lo': f'{DET_LO:.1f}', 'fig1_agem_plotted': AGEM_DONE}))
print('saved', out)
