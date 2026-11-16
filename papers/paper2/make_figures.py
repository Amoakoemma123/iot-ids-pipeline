"""Paper 2 figures, from asr.json (written by compute_values.py).
Fig. 1: black-box ASR against the budget β for the four undefended models (volume and timing actions, TTL unchanged),
        one panel per dataset; on UNSW-NB15 the TTL-only rate (no budget) is a grey dotted reference line.
Fig. 2: CNN under each defence (and the clean-loop control): black-box ASR (volume and TTL, 'both') against β.
Fig. 3: tree defences at β = 0.5: the grid search vs the strongest attack (grid, denser grid, adaptive search,
        any TTL), undefended and each augmentation variant - shows how much of the augmentation gain is in-sample.
Fig. 4 (UNSW-NB15): TTL-only ASR for every initial TTL 1-255, undefended models; candidate TTLs marked.
Mean over seeds; bands = range over seeds. Palette: validated categorical slots (same set as Papers 1 and 3);
the undefended reference is the neutral grey, dashed. Every series also has its own marker."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'asr.json'
TEST = SRC != HERE / 'asr.json'
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
DSN = {'TON_IoT': 'TON-IoT', 'UNSW_NB15': 'UNSW-NB15'}
BETAS = (0.1, 0.25, 0.5, 1.0, 2.0)
INK, MUTED, GRID = '#1f1e1c', '#6b6a65', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': 0.6})
A = json.loads(SRC.read_text())


def panel(ax, ds, series, key):
    for m, (lab, col, mk, ls) in series.items():
        a = A[ds].get(m, {})
        if not all(f'{key}{b}' in a for b in BETAS):
            continue
        v = 100 * np.array([a[f'{key}{b}'] for b in BETAS])        # betas x seeds
        ax.fill_between(BETAS, v.min(1), v.max(1), color=col, alpha=0.12, lw=0)
        ax.plot(BETAS, v.mean(1), color=col, marker=mk, ms=4.5, ls=ls, lw=1.6, mec='white', mew=0.7, label=lab)
    ax.set_xscale('log')
    ax.set_xticks(BETAS, [f'{b:g}' for b in BETAS])
    ax.minorticks_off()
    ax.set_ylim(0, 100)
    ax.set_title(DSN[ds], fontsize=9, color=INK, pad=6)
    ax.set_xlabel('Budget β (relative overhead)')
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def save(fig, name):
    out = FIG / (name.replace('.png', '_TEST.png') if TEST else name)
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print('saved', out)


# Fig. 1: undefended models
S1 = {'cnn_none': ('1D-CNN', '#1baf7a', '^', '-'), 'mlp_none': ('MLP', '#6250d6', 'D', '-'),
      'rf_none': ('Random forest', '#2a78d6', 'o', '-'), 'gb_none': ('Gradient-boosted trees', '#eb6834', 's', '-')}
dss = [d for d in DSN if A.get(d)]
if not dss:
    sys.exit('no results yet: run compute_values.py after the Paper 2 runs')
fig, axes = plt.subplots(1, len(dss), figsize=(6.4, 2.8), squeeze=False, sharey=True)
for ax, ds in zip(axes[0], dss):
    panel(ax, ds, S1, 'bb_volume_b')
    ttl = [np.mean(A[ds][m]['bb_ttl_only']) for m in S1 if 'bb_ttl_only' in A[ds].get(m, {})]
    if ttl:
        ax.axhline(100 * np.mean(ttl), color=MUTED, ls=':', lw=1.0)
        ax.annotate('TTL choice only (mean of models)', (BETAS[-1], 100 * np.mean(ttl)), xytext=(0, -10),
                    textcoords='offset points', fontsize=7, color=MUTED, ha='right')
axes[0][0].set_ylabel('Attack success rate (%)')
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.2)
save(fig, 'fig1_asr_budget.png')

# Fig. 2: CNN defences
S2 = {'cnn_none': ('Undefended', '#5f5e5a', 'v', '--'), 'cnn_clean_loop': ('Clean loop', '#a8a69f', 'x', ':'), 'cnn_at_all': ('AT-all', '#2a78d6', 'o', '-'),
      'cnn_ccat': ('CCAT', '#eb6834', 's', '-'), 'cnn_at_threat': ('AT-threat', '#1baf7a', '^', '-'),
      'cnn_trades': ('TRADES', '#6250d6', 'D', '-')}
fig, axes = plt.subplots(1, len(dss), figsize=(6.4, 2.8), squeeze=False, sharey=True)
for ax, ds in zip(axes[0], dss):
    panel(ax, ds, S2, 'bb_both_b')
axes[0][0].set_ylabel('Attack success rate (%)')
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=6, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.2)
save(fig, 'fig2_defences.png')

# Fig. 3: tree defences, grid (as used for augmentation, fresh candidates) vs strongest attack, beta 0.5
TV = [('none', 'none'), ('augment', 'aug.'), ('augment2', 'aug. ×2'), ('augment_nodelay', 'no delay'), ('augment_nobytes', 'no bytes')]
fig, axes = plt.subplots(1, len(dss), figsize=(6.4, 2.9), squeeze=False, sharey=True)
for ax, ds in zip(axes[0], dss):
    x = 0; ticks, labs = [], []
    for tree, tn in (('rf', 'RF'), ('gb', 'GBDT')):
        for v, vn in TV:
            a = A[ds].get(f'{tree}_{v}', {})
            if 'strongest_b0.5' not in a:
                continue
            G, S_ = 100 * np.array(a['bb_both_b0.5']), 100 * np.array(a['strongest_b0.5'])
            g, s_ = G.mean(), S_.mean()
            ek = dict(ecolor=INK, capsize=1.5, error_kw=dict(lw=0.6))
            ax.bar(x - 0.2, g, 0.38, color='#2a78d6', label='Grid search' if x == 0 else None,
                   yerr=[[g - G.min()], [G.max() - g]], **ek)
            ax.bar(x + 0.2, s_, 0.38, color='#eb6834', label='Strongest attack' if x == 0 else None,
                   yerr=[[s_ - S_.min()], [S_.max() - s_]], **ek)
            ticks.append(x); labs.append(f'{tn}, {vn}'); x += 1
        x += 0.5
    ax.set_xticks(ticks, labs, fontsize=6, rotation=90)
    ax.set_ylim(0, 100); ax.set_title(DSN[ds], fontsize=9, color=INK, pad=6)
    ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
axes[0][0].set_ylabel('Attack success rate, β = 0.5 (%)')
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.2)
save(fig, 'fig3_tree_defences.png')

# Fig. 4: TTL sweep (UNSW-NB15)
if 'UNSW_NB15' in A and any('ttl_sweep' in A['UNSW_NB15'].get(m, {}) for m in S1):
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for m, (lab, col, mk, ls) in S1.items():
        sw = A['UNSW_NB15'].get(m, {}).get('ttl_sweep')
        if not sw:
            continue
        t = sorted(int(k) for k in sw)
        ax.plot(t, [100 * sw[str(k)] for k in t], color=col, lw=1.2, label=lab)
    for c in (1, 31, 62, 254):
        ax.axvline(c, color=MUTED, ls=':', lw=0.8)
    ax.set_xlim(1, 255); ax.set_ylim(0, 100)
    ax.set_xlabel('Initial TTL chosen by the attacker (dotted: TTL candidates)'); ax.set_ylabel('TTL-only ASR (%)')
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    save(fig, 'fig4_ttl_sweep.png')

# Fig. 5 (final review round 2): attack-aware ROC - share of attack flows missed or evaded by the beta 0.5 grid search
# against the validation benign FPR (rule of Table 7). Top: CNN defences; bottom: trees. Mean over seeds, band = range.
S5N = {k: S2[k] for k in ('cnn_none', 'cnn_at_all', 'cnn_at_threat', 'cnn_trades')}
S5N['cnn_at_threat_cs'] = ('AT-threat, corner start', '#1baf7a', '^', ':')
S5T = {'rf_none': ('RF', '#2a78d6', 'o', '--'), 'rf_augment': ('RF, augmented', '#2a78d6', 'o', '-'),
       'gb_none': ('GBDT', '#eb6834', 's', '--'), 'gb_augment': ('GBDT, augmented', '#eb6834', 's', '-')}
if all('roc' in A[ds].get('cnn_none', {}) for ds in dss):
    fig, axes = plt.subplots(2, len(dss), figsize=(7.6, 5.0), squeeze=False, sharex=True)
    for row, series in enumerate((S5N, S5T)):
        for ax, ds in zip(axes[row], dss):
            for m, (lab, col, mk, ls) in series.items():
                a = A[ds].get(m, {})
                if 'roc' not in a:
                    continue
                x = 100 * np.array(a['roc_fpr']); v = 100 * np.array(a['roc']['s1'])     # seeds x fpr
                ax.fill_between(x, v.min(0), v.max(0), color=col, alpha=0.10, lw=0)
                ax.plot(x, v.mean(0), color=col, marker=mk, ms=4, ls=ls, lw=1.5, mec='white', mew=0.6, label=lab)
            ax.set_xscale('log'); ax.set_xticks([0.5, 1, 2, 5], ['0.5', '1', '2', '5']); ax.minorticks_off()
            ax.set_ylim(0, 100); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            if row == 0:
                ax.set_title(DSN[ds] + (' (volume and timing)' if ds == 'UNSW_NB15' else ''), fontsize=9, color=INK, pad=6)
            else:
                ax.set_xlabel('Benign false-positive rate, validation (%)')
        axes[row][0].set_ylabel('Missed or evaded (%)')
        h, l = axes[row][0].get_legend_handles_labels()
        axes[row][-1].legend(h, l, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7)
    fig.tight_layout(h_pad=1.0, w_pad=1.2)
    save(fig, 'fig5_roc.png')
