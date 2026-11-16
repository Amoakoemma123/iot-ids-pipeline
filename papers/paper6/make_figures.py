"""Paper 6 figures (300 dpi PNG in figures/). Panels whose results are missing are drawn empty with a note."""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / 'experiments' / 'paper6_multimodal' / 'results_v4'   # final run (reused v3 runs are copied in, marked reused_from)
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
DSS = ('TON_IoT', 'UNSW_NB15', 'N_BaIoT')
DSN = {'TON_IoT': 'TON-IoT', 'UNSW_NB15': 'UNSW-NB15', 'N_BaIoT': 'N-BaIoT'}
plt.rcParams.update({'font.size': 8.5, 'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})


def load(ds, exp, seed):
    p = RES / ds / exp / f'seed{seed}.json'
    return json.loads(p.read_text()) if p.exists() else None


# ── Figure 1: architectures ──────────────────────────────────────────────────
def box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06', fc=fc, ec='#444', lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=7.5)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='->', lw=0.7, color='#444'))


def fig1():
    fig, axes = plt.subplots(1, 5, figsize=(7.4, 2.9))
    S1, S2, ALL, ENC, HEAD, OUT = '#dbe9f6', '#fde2c8', '#e5e5e5', '#ffffff', '#e8f3e0', '#f2f2f2'
    titles = ['Single stream', 'Early fusion', 'Late fusion', 'Attention fusion', 'Early fusion\nwith head (control)']
    for ax, t in zip(axes, titles):
        ax.set_xlim(0, 3); ax.set_ylim(0, 5); ax.axis('off'); ax.set_title(t, fontsize=8.5)
    # single stream
    ax = axes[0]
    box(ax, 0.75, 4.1, 1.5, 0.55, 'stream k', S1); box(ax, 0.6, 2.9, 1.8, 0.75, 'encoder\n→ 64', ENC)
    box(ax, 0.6, 1.6, 1.8, 0.75, 'dropout\n→ softmax', OUT)
    arrow(ax, 1.5, 4.1, 1.5, 3.65); arrow(ax, 1.5, 2.9, 1.5, 2.35)
    # early fusion, and the early-fusion-with-head control
    for ax, head in ((axes[1], None), (axes[4], 'dense 128')):
        box(ax, 0.2, 4.1, 1.2, 0.55, 'stream 1', S1); box(ax, 1.6, 4.1, 1.2, 0.55, 'stream 2', S2)
        box(ax, 0.35, 2.9, 2.3, 0.75, 'encoder (all\nfeatures) → 64', ENC)
        arrow(ax, 0.8, 4.1, 1.3, 3.65); arrow(ax, 2.2, 4.1, 1.7, 3.65)
        if head:
            box(ax, 0.55, 1.6, 1.9, 0.75, head, HEAD); box(ax, 0.6, 0.35, 1.8, 0.75, 'dropout\n→ softmax', OUT)
            arrow(ax, 1.5, 2.9, 1.5, 2.35); arrow(ax, 1.5, 1.6, 1.5, 1.1)
        else:
            box(ax, 0.6, 1.6, 1.8, 0.75, 'dropout\n→ softmax', OUT); arrow(ax, 1.5, 2.9, 1.5, 2.35)
    # late and attention
    for ax, fuse in ((axes[2], 'concat →\ndense 128'), (axes[3], 'α₁e₁ + α₂e₂\n→ dense 64')):
        box(ax, 0.1, 4.1, 1.3, 0.55, 'stream 1', S1); box(ax, 1.6, 4.1, 1.3, 0.55, 'stream 2', S2)
        box(ax, 0.1, 2.9, 1.3, 0.75, 'encoder 1\n→ e₁ (64)', ENC); box(ax, 1.6, 2.9, 1.3, 0.75, 'encoder 2\n→ e₂ (64)', ENC)
        box(ax, 0.55, 1.6, 1.9, 0.75, fuse, HEAD); box(ax, 0.6, 0.35, 1.8, 0.75, 'dropout\n→ softmax', OUT)
        arrow(ax, 0.75, 4.1, 0.75, 3.65); arrow(ax, 2.25, 4.1, 2.25, 3.65)
        arrow(ax, 0.75, 2.9, 1.2, 2.35); arrow(ax, 2.25, 2.9, 1.8, 2.35); arrow(ax, 1.5, 1.6, 1.5, 1.1)
    fig.tight_layout()
    fig.savefig(FIG / 'fig1_architectures.png', dpi=300)
    plt.close(fig)


# ── Figure 2: semantic vs random splits ──────────────────────────────────────
def fig2():
    archs = [('s1', 'CNN s1'), ('s2', 'CNN s2'), ('late', 'Late'), ('attn', 'Attention'),
             ('rf_s1', 'RF s1'), ('rf_s2', 'RF s2')]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2))
    for ax, ds in zip(axes, DSS):
        ax.set_title(DSN[ds], fontsize=9)
        drawn = False
        for i, (a, lab) in enumerate(archs):
            rnd = [load(ds, f'{a}_rand{r}', r % 5) for r in range(10)]
            sem = [load(ds, a, s) for s in range(5)]
            if all(rnd):
                y = [r['macro_f1'] for r in rnd]
                ax.scatter([i + (j - 4.5) * 0.035 for j in range(10)], y, s=9, color='#888', zorder=2)
                drawn = True
            if all(sem):
                ax.scatter([i], [st.mean(r['macro_f1'] for r in sem)], marker='D', s=30, color='#c0392b', zorder=3)
        ax.set_xticks(range(len(archs)))
        ax.set_xticklabels([l for _, l in archs], fontsize=7, rotation=45, ha='right', rotation_mode='anchor')
        lo, hi = ax.get_ylim()
        if hi - lo < 0.05:     # do not let a saturated panel magnify differences of a few thousandths
            top = min((lo + hi) / 2 + 0.025, 1.002)
            ax.set_ylim(top - 0.05, top)
        ax.set_ylabel('Test macro-F1' if ds == 'TON_IoT' else '')
        ax.grid(axis='y', lw=0.3, alpha=0.6)
        if not drawn:
            ax.text(0.5, 0.5, 'random splits\nnot yet available', transform=ax.transAxes, ha='center', color='#999')
    axes[0].scatter([], [], s=9, color='#888', label='random split')
    axes[0].scatter([], [], marker='D', s=30, color='#c0392b', label='semantic split')
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7, loc='upper center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / 'fig2_random_splits.png', dpi=300)
    plt.close(fig)


# ── Figure 3: attention weights vs stream advantage ─────────────────────────
def _place_labels(fig, ax, pts, bars, fontsize=5.8):
    """Greedy label placement without overlaps (display coordinates), with leader lines for displaced labels.
    pts: (x, y, label); bars: (x, ylo, yhi) error bars to keep clear of."""
    import numpy as np
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    T = ax.transData
    ab = ax.get_window_extent(r)
    obst = []
    for x, y, _ in pts:
        px, py = T.transform((x, y))
        obst.append(Bbox.from_extents(px - 5, py - 5, px + 5, py + 5))
    for x, lo, hi in bars:
        (px, plo), (_, phi) = T.transform((x, lo)), T.transform((x, hi))
        obst.append(Bbox.from_extents(px - 3, plo, px + 3, phi))
    dpi = fig.dpi / 72.0
    placed, leaders = [], []

    def seg_hits_box(p, q, b):
        for t in np.linspace(0, 1, 25):
            if b.contains(p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])):
                return True
        return False

    def segs_cross(a, b, c, d):
        def ccw(p, q, r):
            return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])
        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

    # crowded points first, so that they get the slots next to them
    disp = [T.transform((x, y)) for x, y, _ in pts]
    crowd = [sum(np.hypot(*(np.subtract(d, e))) < 40 * dpi for e in disp) for d in disp]
    order = [pts[i] for i in sorted(range(len(pts)), key=lambda i: -crowd[i])]
    for x, y, lab in order:
        px, py = T.transform((x, y))
        t = ax.text(0, 0, lab, fontsize=fontsize)
        bb = t.get_window_extent(r); w, h = bb.width, bb.height
        t.remove()
        best, best_cost = None, np.inf
        shrunk = [Bbox.from_extents(o.x0 + 1, o.y0 + 1, o.x1 - 1, o.y1 - 1) for o in placed]
        for rad in (6, 10, 15, 21, 28, 36, 45, 56, 70, 90):
            if rad > best_cost:
                break
            for ang in range(0, 360, 15):
                dx, dy = rad * dpi * np.cos(np.radians(ang)), rad * dpi * np.sin(np.radians(ang))
                if abs(dx) < 1e-6:
                    dx = 1e-6
                cx, cy = px + dx, py + dy
                x0 = cx if dx > 0 else cx - w
                y0 = cy - h / 2
                cand = Bbox.from_extents(x0 - 1.5, y0 - 1, x0 + w + 1.5, y0 + h + 1)
                if cand.x0 < ab.x0 + 2 or cand.x1 > ab.x1 - 2 or cand.y0 < ab.y0 + 2 or cand.y1 > ab.y1 + 2:
                    continue
                if any(cand.overlaps(o) for o in obst + placed):
                    continue
                if any(seg_hits_box(a_, b_, cand) for a_, b_ in leaders):
                    continue
                cost = rad
                if rad > 8:
                    anchor = (cx, cy)
                    cost += 40 * sum(segs_cross((px, py), anchor, a_, b_) for a_, b_ in leaders)
                    cost += 40 * sum(seg_hits_box((px, py), anchor, o) for o in shrunk)
                if cost < best_cost:
                    best, best_cost = (cand, dx, dy), cost
        if best is None:      # fall back: right of the point
            best = (Bbox.from_extents(px + 6, py - h / 2, px + 6 + w, py + h / 2), 6 * dpi, 0)
        cand, dx, dy = best
        placed.append(cand)
        far = (dx ** 2 + dy ** 2) ** 0.5 > 8 * dpi
        if far:
            leaders.append(((px, py), (px + dx, py + dy)))
        ax.annotate(lab, (x, y), xytext=(dx / dpi, dy / dpi), textcoords='offset points', fontsize=fontsize, color='#333',
                    ha='left' if dx > 0 else 'right', va='center', annotation_clip=False,
                    arrowprops=dict(arrowstyle='-', lw=0.4, color='#999', shrinkA=0, shrinkB=2) if far else None)


# hand-placed labels for Figure 3 (offset in points from the point; ha), checked visually; others go 6 pt to the right
FIG3_LABELS = {
    'TON_IoT': {'normal': (-6, -5, 'right'), 'ransomware': (9, 20, 'center'), 'xss': (-6, 0, 'right'),
                'mitm': (-6, 0, 'right'), 'password': (-6, -6, 'right'), 'injection': (0, 9, 'center')},
    'UNSW_NB15': {'reconnaissance': (-6, 0, 'right'), 'backdoor': (-4, 12, 'right'), 'exploits': (-6, 0, 'right'),
                  'fuzzers': (-6, -5, 'right'), 'analysis': (18, -17, 'left'), 'shellcode': (0, -44, 'center'),
                  'worms': (6, 3, 'left')},
    'N_BaIoT': {'mirai_syn': (-4, 8, 'right'), 'mirai_scan': (-6, 0, 'right'), 'mirai_udp': (-6, 0, 'right'),
                'mirai_udpplain': (-6, 0, 'right'), 'gafgyt_junk': (0, 9, 'center'), 'gafgyt_scan': (6, -10, 'left'),
                'gafgyt_combo': (6, -8, 'left'), 'normal': (6, 0, 'left'), 'mirai_ack': (6, -2, 'left')},
}
FIG3_XLIM = {'TON_IoT': (-0.24, 0.39), 'UNSW_NB15': (-0.2, 0.73), 'N_BaIoT': (-0.0085, 0.0185)}


def fig3():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.5), gridspec_kw=dict(width_ratios=[1, 1.2, 1]))
    for ax, ds in zip(axes, DSS):
        A = [load(ds, 'attn', s) for s in range(5)]
        S1 = [load(ds, 's1', s) for s in range(5)]
        S2 = [load(ds, 's2', s) for s in range(5)]
        cls = list(A[0]['attn_s1_per_class'])
        pts, bars = [], []
        for c in cls:
            x = st.mean(a['per_class_f1'][c] - b['per_class_f1'][c] for a, b in zip(S1, S2))
            w = [a['attn_s1_per_class'][c] for a in A]
            m = st.mean(w)
            ax.errorbar(x, m, yerr=[[m - min(w)], [max(w) - m]], fmt='o', ms=3.5, color='#2c6fad', elinewidth=0.7, capsize=1.5)
            dx, dy, ha = FIG3_LABELS.get(ds, {}).get(c, (6, 0, 'left'))
            far = (dx ** 2 + dy ** 2) ** 0.5 > 10
            ax.annotate({'reconnaissance': 'recon.'}.get(c, c.replace('_', ' ')), (x, m), xytext=(dx, dy),
                        textcoords='offset points', fontsize=6, color='#333', ha=ha, va='center', annotation_clip=False,
                        arrowprops=dict(arrowstyle='-', lw=0.4, color='#999', shrinkA=1, shrinkB=2.5) if far else None)
        ax.axhline(0.5, lw=0.5, ls=':', color='#777'); ax.axvline(0, lw=0.5, ls=':', color='#777')
        ax.set_ylim(0, 1.08)
        ax.set_xlim(*FIG3_XLIM[ds])
        ax.set_title(DSN[ds], fontsize=9)
        ax.set_xlabel('F1(stream 1) − F1(stream 2)')
        ax.set_ylabel('Attention weight on stream 1' if ds == 'TON_IoT' else '')
    fig.tight_layout()
    fig.savefig(FIG / 'fig3_attention.png', dpi=300)
    plt.close(fig)


# ── Final review round 2: Figure 2 (per-class source of late − early) and Figure 3 (tuned networks against trees) ──
RES_GB = RES.parent / 'results_p6gb'          # the rolled-back GBDT scored in the paper
NET, NET_T, TREE = '#2a78d6', '#1baf7a', '#eb6834'


def load_any(ds, exp, seed):
    if exp.startswith('gb'):
        p = RES_GB / ds / exp / f'seed{seed}.json'
        if p.exists():
            return json.loads(p.read_text())
    return load(ds, exp, seed)


def fig_perclass():
    """per-class contribution to the macro-F1 difference late − early fusion: (F1_late,c − F1_early,c) / K, per seed"""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, ds in zip(axes, ('TON_IoT', 'UNSW_NB15')):
        L = [load(ds, 'late', s) for s in range(5)]; E = [load(ds, 'early', s) for s in range(5)]
        if not all(L) or not all(E):
            ax.text(0.5, 0.5, 'not available', transform=ax.transAxes, ha='center', color='#999'); continue
        cls = list(L[0]['per_class_f1']); K = len(cls)
        D = {c: [(l['per_class_f1'][c] - e['per_class_f1'][c]) / K for l, e in zip(L, E)] for c in cls}
        order = sorted(cls, key=lambda c: st.mean(D[c]))
        y = range(len(order))
        means = [st.mean(D[c]) for c in order]
        ax.barh(list(y), means, height=0.6, color=[NET if m >= 0 else '#a8a69f' for m in means], zorder=2)
        for i, c in enumerate(order):
            ax.scatter(D[c], [i] * 5, s=7, color='#1f1e1c', zorder=3, lw=0)
        ax.axvline(0, color='#6b6a65', lw=0.7)
        ax.set_yticks(list(y)); ax.set_yticklabels(order, fontsize=7)
        tot = st.mean(l['macro_f1'] - e['macro_f1'] for l, e in zip(L, E))
        ax.set_title(f'{DSN[ds]} (macro-F1 difference {tot:+.3f})', fontsize=8.5)
        ax.set_xlabel('Contribution to macro-F1, late − early fusion')
        ax.grid(axis='x', lw=0.3, alpha=0.6); ax.set_axisbelow(True)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG / 'fig2_perclass_late_early.png', dpi=300)
    plt.close(fig)


def fig_nets_trees():
    """test macro-F1 per seed: fusion networks (fixed and tuned configuration) and tree ensembles on all features"""
    models = [('early', 'Early', NET), ('late', 'Late', NET), ('attn', 'Attention', NET),
              ('early_tuned', 'Early,' + chr(10) + 'tuned', NET_T), ('late_tuned', 'Late,' + chr(10) + 'tuned', NET_T),
              ('rf', 'RF', TREE), ('gb', 'GBDT', TREE)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, ds in zip(axes, ('TON_IoT', 'UNSW_NB15')):
        for i, (m, lab, col) in enumerate(models):
            R = [load_any(ds, m, s) for s in range(5)]
            if not all(R):
                continue
            v = [r['macro_f1'] for r in R]
            ax.scatter([i + (j - 2) * 0.07 for j in range(5)], v, s=12, color=col, zorder=3, lw=0)
            ax.hlines(st.mean(v), i - 0.3, i + 0.3, color='#1f1e1c', lw=1.2, zorder=4)
        ax.set_xticks(range(len(models))); ax.set_xticklabels([l for _, l, _ in models], fontsize=6.5)
        ax.set_title(DSN[ds], fontsize=9); ax.set_ylabel('Test macro-F1' if ds == 'TON_IoT' else '')
        ax.grid(axis='y', lw=0.3, alpha=0.6); ax.set_axisbelow(True)
    for col, lab in ((NET, 'network, fixed configuration'), (NET_T, 'network, tuned on validation'), (TREE, 'tree ensemble')):
        axes[0].scatter([], [], s=12, color=col, label=lab)
    axes[0].plot([], [], color='#1f1e1c', lw=1.2, label='mean of five seeds')
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7, loc='upper center', ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.0)
    fig.savefig(FIG / 'fig3_nets_trees.png', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig_perclass(); fig_nets_trees()
    print('figures saved to', FIG)
