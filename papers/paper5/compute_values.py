"""Collect every number the Paper 5 manuscript quotes into values.json (v3 revision).

Sources (all under Papers pipeline/experiments/):
  paper5_xai/results_v3/<ds>/seed<s>/<stage>.json  per-seed v3 stages (seeds 42, 43, 44)
  paper5_xai/results_v3/<ds>/<stage>.json          seedless stages (train_seeds, leakage_decomposition)
  paper5_xai/results_v2/<ds>/{global_shap,ports}.json  global SHAP and port check (corrected data)
  baseline/results (corrected pipeline) and the v1 backup; the raw UNSW CSV for the audit.
Seed 42 gives the reported confidence intervals; seeds 43 and 44 give the 'range over seeds'.
The manuscript only contains {{key}} placeholders that build.py replaces from values.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]
PIPE = HERE.parents[1]   # repository root (authors' tree: <project>/Papers pipeline)
R3 = PIPE / 'experiments' / 'paper5_xai' / 'results_v3'
R2 = PIPE / 'experiments' / 'paper5_xai' / 'results_v2'
V1 = PROJ / '_v1_backup_2026-10-24' / 'experiments' / 'baseline' / 'results'   # earlier-pipeline results, not in this repository
V2 = PIPE / 'experiments' / 'baseline' / 'results'
DS = {'TON_IoT': 'TON', 'UNSW_NB15': 'UNSW'}
SEEDS = (42, 43, 44)
# LIME ranked by its contribution w*(x-mu)/sigma (primary); the *_raw stages rank by the raw weight
LOCAL, DEL, STAB = 'local_c', 'deletion_c', 'stability_c'
MODELS = ('rf', 'cnn')
V, tables = {}, {}
sys.path.insert(0, str(PIPE))


def ci(d, nd=2, pct=False):
    if d is None or d.get('mean') is None:
        return 'n/a'
    s = 100 if pct else 1
    nd = 0 if pct else nd
    return f"{s * d['mean']:.{nd}f} [{s * d['lo']:.{nd}f}, {s * d['hi']:.{nd}f}]"


def f(x, nd=2):
    return f'{x:.{nd}f}'


def pct(x, nd=0):
    return f'{100 * x:.{nd}f}'


def rng_str(vals, nd=2, scale=1):
    """'a–b' over seeds (or a single value if all equal at this precision)."""
    vals = [scale * v for v in vals if v is not None]
    lo, hi = f(min(vals), nd), f(max(vals), nd)
    return lo if lo == hi else f'{lo}–{hi}'


def andjoin(items):
    items = list(items)
    return items[0] if len(items) == 1 else ', '.join(items[:-1]) + ' and ' + items[-1]


def cls_name(c):
    return c.replace('_', ' ')


def load3(ds, stage, seed=None):
    p = R3 / ds / (f'{stage}.json' if seed is None else f'seed{seed}/{stage}.json')
    return json.loads(p.read_text()) if p.exists() else None


def load2(ds, stage):
    p = R2 / ds / f'{stage}.json'
    return json.loads(p.read_text()) if p.exists() else None


def per_seed(ds, stage):
    """{seed: result} for the seeds that have finished this stage."""
    return {s: r for s in SEEDS if (r := load3(ds, stage, s)) is not None}


def get(d, path):
    for k in path.split('.'):
        d = d[k]
    return d


# ── final round: results of run_xai_final.py / twin_mask.py (missing → placeholders stay visible) ──
RF_ = PIPE / 'experiments' / 'paper5_xai' / 'results_final'


def loadF(ds, stage, seed=None):
    p = RF_ / ds / (f'{stage}.json' if seed is None else f'seed{seed}/{stage}.json')
    return json.loads(p.read_text()) if p.exists() else None


TWIN = {ds: set((loadF(ds, 'twin_mask') or {}).get('twin_features_idx', [])) for ds in DS}
TWIN_CLS = {ds: (loadF(ds, 'twin_mask') or {}).get('per_class_twin_features', {}) for ds in DS}


def boot2(groups, n=2000, seed=42):
    """Two-level (seed x record) percentile bootstrap of the mean of per-seed means.
    groups: list of per-seed arrays of per-record values."""
    groups = [np.asarray(g, float) for g in groups if len(g)]
    rng = np.random.default_rng(seed)
    est = float(np.mean([g.mean() for g in groups]))
    bs = []
    for _ in range(n):
        gs = [groups[i] for i in rng.integers(len(groups), size=len(groups))]
        bs.append(np.mean([g[rng.integers(len(g), size=len(g))].mean() for g in gs]))
    return dict(mean=est, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n=int(sum(len(g) for g in groups)))


def binvec(per_class, nkey='n', skey='success', counts=None):
    """Per-record 0/1 outcomes reconstructed from per-class shares and sizes (records within a class
    are exchangeable for a mean)."""
    out = []
    for c, d in per_class.items():
        n = counts[c] if counts is not None else d[nkey]
        s = d if counts is not None else d[skey]
        k = int(round(n * s))
        out += [1] * k + [0] * (n - k)
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def mcnemar_p(a, b):
    """Exact McNemar p-value for paired binary outcomes a, b."""
    from scipy.stats import binomtest
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    return (binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0), n10, n01


def pfmt(p):
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def pfull(p):
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'


# ── dataset audit (UNSW duplicates, file order, sttl) ────────────────────────
u = pd.read_csv(PIPE / 'dataset2' / 'UNSW_NB15.csv', low_memory=False)
u.columns = u.columns.str.strip()
V['unsw_rows_raw'] = f"{len(u):,}"
V['unsw_dupes_with_id'] = f"{int(u.duplicated().sum()):,}"
nid = u.drop(columns=['id'])
V['unsw_dupes'] = f"{int(nid.duplicated().sum()):,}"
V['unsw_dupe_pct'] = f"{100 * nid.duplicated().mean():.1f}"
V['unsw_rows_dedup'] = f"{len(nid.drop_duplicates()):,}"
V['unsw_dupe_group_pct'] = f"{100 * nid.duplicated(keep=False).mean():.1f}"
# records whose features occur with more than one label (after removing exact duplicates)
dd = nid.drop_duplicates()
feat_cols = [c for c in dd.columns if c not in ('label', 'attack_cat')]
V['unsw_conflict_rows'] = f"{int(dd.duplicated(subset=feat_cols, keep=False).sum()):,}"
db = dd.drop(columns=['attack_cat']).drop_duplicates()
V['unsw_conflict_binary'] = f"{int(db.duplicated(subset=feat_cols, keep=False).sum()):,}"
vc_raw, vc_dd = u['attack_cat'].str.strip().value_counts(), dd['attack_cat'].str.strip().value_counts()
V['unsw_generic_raw'] = f"{int(vc_raw['Generic']):,}"
V['unsw_generic_dedup'] = f"{int(vc_dd['Generic']):,}"
cat = u['attack_cat'].astype(str).str.strip()
V['unsw_normal_prefix'] = f"{int((cat != 'Normal').values.argmax()):,}"
gen_pos = np.where(cat.values == 'Generic')[0]
V['unsw_generic_first_pct'] = f"{100 * gen_pos.min() / len(u):.1f}"
st = dd.groupby('sttl')['label'].agg(['size', 'mean']).sort_values('size', ascending=False)
V['unsw_sttl_top3_share'] = f"{100 * st['size'].iloc[:3].sum() / len(dd):.1f}"
V['unsw_sttl_top3_values'] = ', '.join(str(int(x)) for x in st.index[:3])
V['unsw_sttl31_rows'] = f"{int(st.loc[31, 'size']):,}"
V['unsw_sttl31_attack_pct'] = f"{100 * st.loc[31, 'mean']:.1f}"
del u, nid, dd

import contextlib, io
from src.data.loader import load_dataset
from experiments.baseline.config import DATASETS as BASE_DATASETS
with contextlib.redirect_stdout(io.StringIO()):
    _raw = load_dataset(*BASE_DATASETS['UNSW_NB15'], seed=42)
V['unsw_rows_pipeline'] = f"{len(_raw['y_train']) + len(_raw['y_val']) + len(_raw['y_test']):,}"
del _raw
import experiments.paper5_xai.run_xai_v3 as X3
import experiments.paper5_xai.run_xai_v2 as X2
boot_ci_ = X2.boot_ci
for k in ('N_LOCAL', 'N_STAB', 'N_RUNS', 'N_SHAP_RUNS', 'N_CF', 'CF_MAX_K', 'CF_TRIES', 'N_HARD_GEN', 'N_BOOT'):
    V[k] = f"{getattr(X3, k):,}"
V['N_GLOBAL'] = f"{X2.N_GLOBAL:,}"
import experiments.paper5_xai.run_xai_final as XF
V['N_SENS'] = str(XF.N_SENS)
for ds, tag in DS.items():
    V[f'{tag}_attacker_groups'] = andjoin(g.replace('_', ' ') for g in X3.ATTACKER_GROUPS[ds])

# ── Table 1: leakage decomposition (UNSW RF) ─────────────────────────────────
Dc = load3('UNSW_NB15', 'leakage_decomposition')
if Dc:
    A, B, C, D = (Dc[k] for k in ('A_original', 'B_id_removed_duplicates_kept', 'C_id_kept_duplicates_removed', 'D_corrected'))
    V['dec_test_seen_pct'] = pct(A['share_test_seen'], 1)
    for key, r in (('A', A['full_test']), ('Aunseen', A['unseen_test']), ('Aseen', A['seen_test']), ('B', B['full_test']),
                   ('Bunseen', B['unseen_test']), ('C', C['full_test']), ('D', D['full_test'])):
        V[f'dec_{key}_f1'] = f(r['macro_f1'], 4)
        V[f'dec_{key}_acc'] = f(r['accuracy'], 4)
        V[f'dec_{key}_n'] = f"{r['n']:,}"
    V['dec_id_effect'] = f(A['full_test']['macro_f1'] - B['full_test']['macro_f1'], 3)
    V['dec_dup_effect'] = f(abs(B['full_test']['macro_f1'] - D['full_test']['macro_f1']), 3)
    V['dec_A_has_id'] = 'yes' if 'id' in A['features'] else 'no'
    V['dec_C_has_id'] = 'yes' if 'id' in C['features'] else 'no'
    tables['decomposition'] = '\n'.join([
        '| Condition | `id` feature | Duplicates | Test set | Test records | Macro-F1 | Accuracy |',
        '|---|---|---|---|---|---|---|',
        f"| A. Original model | kept | kept | original, all | {V['dec_A_n']} | {V['dec_A_f1']} | {V['dec_A_acc']} |",
        f"| A. Original model | kept | kept | original, no copy in training | {V['dec_Aunseen_n']} | {V['dec_Aunseen_f1']} | {V['dec_Aunseen_acc']} |",
        f"| A. Original model | kept | kept | original, copy in training | {V['dec_Aseen_n']} | {V['dec_Aseen_f1']} | {V['dec_Aseen_acc']} |",
        f"| B | removed | kept | own, all | {V['dec_B_n']} | {V['dec_B_f1']} | {V['dec_B_acc']} |",
        f"| B | removed | kept | own, no copy in training | {V['dec_Bunseen_n']} | {V['dec_Bunseen_f1']} | {V['dec_Bunseen_acc']} |",
        f"| C | kept | removed | own | {V['dec_C_n']} | {V['dec_C_f1']} | {V['dec_C_acc']} |",
        f"| D. Corrected pipeline | removed | removed | own | {V['dec_D_n']} | {V['dec_D_f1']} | {V['dec_D_acc']} |",
    ])

# ── baseline v1 vs corrected ─────────────────────────────────────────────────
import joblib
for ds, tag in DS.items():
    for ver, root in (('v1', V1), ('v2', V2)):
        m = json.loads((root / ds / 'metrics.json').read_text())
        for model in ('rf', 'cnn', 'lr', 'lstm', 'aerf'):
            V[f'{tag}_{ver}_{model}_f1'] = f(m[model]['macro_f1'], 4)
            V[f'{tag}_{ver}_{model}_acc'] = f(m[model]['accuracy'], 4)
    V[f'{tag}_v1_nfeat'] = str(joblib.load(V1 / ds / 'pipeline.pkl').n_features_)
    _p = joblib.load(V2 / ds / 'pipeline.pkl')
    V[f'{tag}_nfeat_mi'] = str(int(np.sum(_p.mi_mask_)))
    V['mi_threshold'] = f'{_p.mi_threshold:g}'
    V['rfecv_cv'] = {3: 'three', 5: 'five'}.get(_p.rfecv_cv, str(_p.rfecv_cv))
    V['rfecv_subsample'] = f'{_p.rfecv_subsample:,}'
    if _p.rfecv_ is not None:
        _nf = list(_p.rfecv_.cv_results_['n_features']); V['rfecv_floor'] = str(int(min(_nf)))
        V[f'{tag}_rfecv_at_floor'] = str(_p.n_features_ == min(_nf))

# ── per-dataset XAI results ──────────────────────────────────────────────────
for ds, tag in DS.items():
    P = per_seed(ds, 'performance')
    if 42 not in P:
        continue
    p42 = P[42]
    V[f'{tag}_ntest'] = f"{p42['n_test']:,}"
    V[f'{tag}_nfeat'] = str(p42['n_features'])
    V[f'{tag}_kw'] = f"{0.75 * np.sqrt(p42['n_features']):.2f}"     # LIME default kernel width
    V[f'{tag}_ncls'] = str(len(p42['classes']))
    V[f'{tag}_features'] = ', '.join(f'`{x}`' for x in p42['features'])
    for m in MODELS:
        V[f'{tag}_{m}_f1'] = f(p42[m]['macro_f1'], 4)
        V[f'{tag}_{m}_f1_seeds'] = rng_str([P[s][m]['macro_f1'] for s in P], 4)
        V[f'{tag}_{m}_fpr'] = pct(p42[m]['benign_fpr'], 1)
        V[f'{tag}_{m}_ms1000'] = f"{p42[m]['ms_per_1000']:.0f}"
    V[f'{tag}_rf_mb'] = f"{p42['rf']['pickle_mb']:.0f}"
    V[f'{tag}_rf_nodes'] = f"{p42['rf']['n_nodes']:,}"
    V[f'{tag}_cnn_params'] = f"{p42['cnn']['n_params']:,}"
    V[f'{tag}_cnn_mb'] = f"{p42['cnn']['file_mb']:.2f}"
    V['nseeds_done_' + tag] = str(len(P))

    # global SHAP (seed 42, corrected data)
    g = load2(ds, 'global_shap')
    if g:
        rows = ['| Rank | RF feature | Mean \\|SHAP\\| | 95% rank interval | CNN feature | Mean \\|SHAP\\| | 95% rank interval |',
                '|---|---|---|---|---|---|---|']
        for r in range(min(10, len(g['rf']['ranking']))):
            a, b = g['rf']['ranking'][r], g['cnn']['ranking'][r]
            rows.append(f"| {r + 1} | `{a['feature']}` | {a['mean_abs_shap']:.4f} | {a['rank_lo']}–{a['rank_hi']} "
                        f"| `{b['feature']}` | {b['mean_abs_shap']:.4f} | {b['rank_lo']}–{b['rank_hi']} |")
        tables[f'{tag}_global'] = '\n'.join(rows)
        for m in MODELS:
            V[f'{tag}_{m}_top1'] = g[m]['ranking'][0]['feature']
            V[f'{tag}_{m}_top3'] = andjoin(f"`{x['feature']}`" for x in g[m]['ranking'][:3])
        V[f'{tag}_xmodel_j5'] = f(g['cross_model']['jaccard@5'])
        V[f'{tag}_xmodel_j10'] = f(g['cross_model']['jaccard@10'])
        if 'rf_interactions' in g:
            V[f'{tag}_inter_top'] = andjoin(f"`{p['a']}`×`{p['b']}`" for p in g['rf_interactions']['top_pairs'][:3])

    # local: agreement + deletion
    L = per_seed(ds, LOCAL)
    if 42 in L:
        L42 = L[42]
        V[f'{tag}_chance_j5'] = f(L42['chance_j5'])
        V[f'{tag}_chance_field_j5'] = f(L42['chance_field_j5'])
        V[f'{tag}_nfields_local'] = str(L42['n_fields'])
        for m in MODELS:
            S = L42[m]['summary']
            for k, d in S.items():
                if isinstance(d, dict) and 'lo' in d:
                    V[f'{tag}_{m}_{k}'] = ci(d)
                    V[f'{tag}_{m}_{k}_mean'] = f(d['mean'])
                    V[f'{tag}_{m}_{k}_seeds'] = rng_str([L[s][m]['summary'][k]['mean'] for s in L])
            V[f'{tag}_{m}_shared_ge3'] = pct(S['share_shared_ge3']['mean'])
            V[f'{tag}_{m}_nlocal'] = str(S['j5']['n'])
            for k, d in L42[m]['paired'].items():
                V[f'{tag}_{m}_p_{k}'] = ci(d)
                V[f'{tag}_{m}_p_{k}_lo'] = f(d['lo'])
            # SHAP above LIME at every k under both replacement strategies, in every seed
            cells = [L[s][m]['paired'][f'{pre}_shap_minus_lime_{k}'] for s in L for pre in ('del', 'dels') for k in (1, 3, 5)]
            V[f'{tag}_{m}_shap_gt_lime_all'] = str(all(d['lo'] > 0 for d in cells))
            V[f'{tag}_{m}_shap_gt_lime_sig'] = str(sum(d['lo'] > 0 for d in cells))
            V[f'{tag}_{m}_shap_gt_lime_pos'] = str(sum(d['mean'] > 0 for d in cells))
            V[f'{tag}_{m}_shap_gt_lime_cells'] = str(len(cells))
            cells = [L[s][m]['paired'][f'{pre}_lime_minus_rand_{k}'] for s in L for pre in ('del', 'dels') for k in (1, 3, 5)]
            V[f'{tag}_{m}_lime_gt_rand_sig'] = str(sum(d['lo'] > 0 for d in cells))
            # per-class agreement from the per-record rows (true class)
            rows = L42[m]['rows']
            classes = p42['classes']
            pc = {}
            for r in rows:
                pc.setdefault(classes[r['true']], []).append(r['j5'])
            pc = {c: float(np.mean(v)) for c, v in pc.items()}
            lo, hi = min(pc, key=pc.get), max(pc, key=pc.get)
            V[f'{tag}_{m}_j5_min'] = f"{pc[lo]:.2f} ({cls_name(lo)})"
            V[f'{tag}_{m}_j5_max'] = f"{pc[hi]:.2f} ({cls_name(hi)})"
            # final round: mean over seeds with seed x record bootstrap; local SHAP vs global top-5
            V[f'{tag}_{m}_j5_3s'] = ci(boot2([[r['j5'] for r in L[s][m]['rows']] for s in L]))
            if g:
                gtop = [p42['features'].index(x['feature']) for x in g[m]['ranking'][:5]]
                jg = [len(set(np.argsort(-np.array(r['shap_abs']))[:5]) & set(gtop)) / len(set(np.argsort(-np.array(r['shap_abs']))[:5]) | set(gtop)) for r in rows]
                V[f'{tag}_{m}_j5_shap_global'] = ci(boot_ci_(jg))
                V[f'{tag}_{m}_j5_shap_global_mean'] = f(np.mean(jg))
            # twins excluded (records identical to a training record in the selected features)
            if TWIN.get(ds):
                keep = [r['j5'] for r in rows if r['test_idx'] not in TWIN[ds]]
                V[f'{tag}_{m}_j5_notwin'] = ci(boot_ci_(keep)); V[f'{tag}_{m}_n_notwin'] = str(len(keep))
                V[f'{tag}_{m}_twin_local_pct'] = pct(1 - len(keep) / len(rows))
                V[f'{tag}_{m}_j5_notwin_seeds'] = rng_str([np.mean([r['j5'] for r in L[s][m]['rows'] if r['test_idx'] not in TWIN[ds]]) for s in L])

    # stability (seed 42 only)
    St = load3(ds, STAB, 42)
    if St:
        for m in MODELS:
            S = St[m]['summary']
            for k in ('stab_1000', 'stab_5000', 'single_lime_vs_shap', 'consensus_lime_vs_shap', 'shap_self'):
                V[f'{tag}_{m}_{k}'] = ci(S[k])
                V[f'{tag}_{m}_{k}_mean'] = f(S[k]['mean'])
            for k, v in St[m]['share_ge_075'].items():
                V[f'{tag}_{m}_{k}_share'] = pct(v)
            V[f'{tag}_{m}_consensus_gain'] = ci(St[m]['paired_consensus_minus_single'], 3)
            _j = S['shap_self']['mean']; V[f'{tag}_{m}_shap_self_shared'] = f(10 * _j / (1 + _j), 1)
            V[f'{tag}_{m}_nstab'] = str(S['stab_1000']['n'])

    # counterfactuals: three search modes
    C = per_seed(ds, 'counterfactual')
    if 42 in C:
        C42 = C[42]
        V[f'{tag}_ngroups'] = str(len(C42['groups']))
        if 'protocol_state' in C42['groups']:
            V[f'{tag}_protocol_state_nfeat'] = str(len(C42['groups']['protocol_state']))
        _sz = sorted((len(v) for v in C42['groups'].values()), reverse=True)
        _lo4 = sum(sorted(_sz)[:4]); _hi4 = sum(_sz[:4])
        V[f'{tag}_max4_nfeat'] = str(_hi4) if _lo4 == _hi4 else f'{_lo4}–{_hi4}'
        V[f'{tag}_support_tau'] = f"{C42['support_tau']:.4f}"
        for m in MODELS:
            for mode in ('random', 'shap', 'attacker'):
                r = C42[m][mode]
                V[f'{tag}_{m}_cf_{mode}_succ'] = pct(r['success'])
                V[f'{tag}_{m}_cf_{mode}_succ_seeds'] = rng_str([C[s][m][mode]['success'] for s in C], 0, 100)
                V[f'{tag}_{m}_cf_{mode}_k'] = ci(r['k'])
                V[f'{tag}_{m}_cf_{mode}_k1'] = pct(r['success_within_1'])
                V[f'{tag}_{m}_cf_{mode}_q'] = f"{r['queries_median']:g}"
                V[f'{tag}_{m}_cf_{mode}_infeas'] = pct(r['infeasible_share'])
                gc_ = sorted(r['group_counts'].items(), key=lambda kv: -kv[1])
                n_succ = max(r['k']['n'], 1)
                V[f'{tag}_{m}_cf_{mode}_groups'] = andjoin(f"{g.replace('_', ' ')} ({100 * c / n_succ:.0f}%)" for g, c in gc_[:3])
                if gc_:
                    V[f'{tag}_{m}_cf_{mode}_top_group'] = gc_[0][0].replace('_', ' ')
                    V[f'{tag}_{m}_cf_{mode}_top_share'] = f"{100 * gc_[0][1] / n_succ:.0f}"
        rows = ['| Attack class | n (RF / CNN) | RF, all groups (%) | RF, attacker-controllable (%) [95% CI] | CNN, all groups (%) | CNN, attacker-controllable (%) [95% CI] |',
                '|---|---|---|---|---|---|']
        for c in C42['rf']['random']['per_class']:
            cells = []
            ns = [C42[m]['random']['per_class'].get(c, {}).get('n', 0) for m in MODELS]
            for m in MODELS:
                for mode in ('random', 'attacker'):
                    pcm = C42[m][mode]['per_class'].get(c)
                    if pcm and mode == 'attacker':
                        lo_, hi_ = wilson(round(pcm['n'] * pcm['success']), pcm['n'])
                        cells.append(f"{pct(pcm['success'])} [{pct(max(lo_, 0))}, {pct(min(hi_, 1))}]")
                    else:
                        cells.append(pct(pcm['success']) if pcm else '–')
            flag = '†' if TWIN_CLS.get(ds, {}).get(c, 0) > 0.5 else ''
            rows.append(f"| {cls_name(c)}{flag} | {ns[0]} / {ns[1]} | " + ' | '.join(cells) + ' |')
        tables[f'{tag}_cf'] = '\n'.join(rows)
        # attacker-controllable success: mean over seeds with a seed x record bootstrap; per seed (RF)
        for m in MODELS:
            g = [binvec(C[s][m]['attacker']['per_class']) for s in C]
            _b = boot2(g)
            V[f'{tag}_{m}_cf_attacker_succ_3s'] = f"{pct(_b['mean'])}% [{pct(_b['lo'])}, {pct(_b['hi'])}]"
            V[f'{tag}_{m}_cf_attacker_succ_3s_mean'] = pct(boot2(g)['mean'])
            V[f'{tag}_{m}_cf_attacker_succ_byseed'] = andjoin(f"{pct(C[s][m]['attacker']['success'])}%" for s in sorted(C))
            V[f'{tag}_{m}_cf_shap_succ'] = pct(C42[m]['shap']['success'])
            V[f'{tag}_{m}_cf_random_succ_exact'] = pct(C42[m]['random']['success'])
        if 'worms' in C42['rf']['random']['per_class']:
            V[f'{tag}_worms_ncf'] = f"{C42['rf']['random']['per_class']['worms']['n']} (RF) and {C42['cnn']['random']['per_class']['worms']['n']} (CNN)"

    # hardening: all groups vs attacker-controllable groups, three seeds
    for variant, stage in (('all', 'hardening'), ('att', 'hardening_attacker')):
        H = per_seed(ds, stage)
        if not H:
            continue
        V[f'{tag}_hard_{variant}_nseeds'] = str(len(H))
        for arm in ('original', 'counterfactual', 'control', 'control_copy'):
            if arm not in H[42]:
                continue
            ev = [H[s][arm]['evasion'] for s in H]
            V[f'{tag}_hard_{variant}_{arm}_ev'] = f"{100 * np.mean(ev):.0f}"
            V[f'{tag}_hard_{variant}_{arm}_ev_seeds'] = rng_str(ev, 0, 100)
            V[f'{tag}_hard_{variant}_{arm}_f1'] = f(np.mean([H[s][arm]['macro_f1'] for s in H]), 4)
            V[f'{tag}_hard_{variant}_{arm}_f1_seeds'] = rng_str([H[s][arm]['macro_f1'] for s in H], 4)
            V[f'{tag}_hard_{variant}_{arm}_fpr'] = f"{100 * np.mean([H[s][arm]['benign_fpr'] for s in H]):.2f}"
        V[f'{tag}_hard_{variant}_naug'] = rng_str([H[s]['n_augmented'] for s in H], 0)
        V[f'{tag}_hard_{variant}_nquery'] = rng_str([H[s]['n_query'] for s in H], 0)
        # counterfactual arm below both original and control in every seed (CIs separate)
        V[f'{tag}_hard_{variant}_cf_beats_both'] = str(all(
            H[s]['counterfactual']['evasion_ci']['hi'] < min(H[s][a]['evasion_ci']['lo'] for a in ('original', 'control', 'control_copy') if a in H[s])
            for s in H))
        V[f'{tag}_hard_{variant}_cf_sep_seeds'] = str(sum(
            H[s]['counterfactual']['evasion_ci']['hi'] < min(H[s][a]['evasion_ci']['lo'] for a in ('original', 'control', 'control_copy') if a in H[s])
            for s in H))
        sep = lambda arms_: str(sum(H[s]['counterfactual']['evasion_ci']['hi'] < min(H[s][a]['evasion_ci']['lo'] for a in arms_ if a in H[s]) for s in H))
        V[f'{tag}_hard_{variant}_sep_ctl'] = sep(('original', 'control'))
        V[f'{tag}_hard_{variant}_sep_copy'] = sep(('control_copy',))
        V[f'{tag}_hard_{variant}_ev42'] = {a: ci(H[42][a]['evasion_ci'], pct=True) for a in H[42] if isinstance(H[42][a], dict) and 'evasion_ci' in H[42][a]}
        for a, v in V[f'{tag}_hard_{variant}_ev42'].items():
            V[f'{tag}_hard_{variant}_{a}_ev42'] = v
        del V[f'{tag}_hard_{variant}_ev42']
        # per-class evasion (mean over seeds) for the attacker variant
        for a in ('original', 'counterfactual', 'control', 'control_copy'):
            if a not in H[42]:
                continue
            g = [binvec({c: v for c, v in H[s][a]['evasion_per_class'].items()}, counts=H[s]['query_per_class']) for s in H]
            V[f'{tag}_hard_{variant}_{a}_ev_3s'] = ci(boot2(g), pct=True)
            for c in H[42][a]['evasion_per_class']:
                vals = [H[s][a]['evasion_per_class'].get(c) for s in H]
                vals = [v for v in vals if v is not None]
                V[f'{tag}_hard_{variant}_{a}_cls_{c}'] = f"{100 * np.mean(vals):.0f}"
    # port check (v2 stage, corrected data)
    Pt = load2(ds, 'ports')
    if Pt:
        if Pt.get('ports'):
            V[f'{tag}_ports'] = andjoin(f'`{x}`' for x in Pt['ports'])
            V[f'{tag}_f1_ports'] = f(Pt['macro_f1_with_ports'], 4)
            V[f'{tag}_f1_noports'] = f(Pt['macro_f1_without_ports'], 4)
        else:
            V[f'{tag}_ports'] = 'none'

    # TTL removed before feature selection
    T = load3(ds, 'ttl_reselect', 42)
    if T and 'rf_macro_f1' in T:
        V[f'{tag}_ttl_removed'] = andjoin(f'`{x}`' for x in T['removed_raw'])
        V[f'{tag}_ttl_rf_f1'] = f(T['rf_macro_f1'], 4)
        V[f'{tag}_ttl_cnn_f1'] = f(T['cnn_macro_f1'], 4)
        V[f'{tag}_ttl_nfeat'] = str(len(T['features_selected']))
        V[f'{tag}_ttl_features'] = ', '.join(f'`{x}`' for x in T['features_selected'])
        r = T['sttl_rule_test']
        V[f'{tag}_sttl_rule_bin'] = pct(r['binary_accuracy'], 1)
        V[f'{tag}_sttl_rule_cls'] = pct(r['class_accuracy'], 1)
        V[f'{tag}_sttl_rule_f1'] = f(r['class_macro_f1'], 4)
        V[f'{tag}_majority_test'] = pct(r['majority_class_accuracy'], 1)
        drops = sorted(((c, p42['rf']['per_class_f1'][c] - T['rf_per_class'][c]) for c in T['rf_per_class']), key=lambda kv: -kv[1])
        V[f'{tag}_ttl_worst'] = andjoin(f"{cls_name(c)} {p42['rf']['per_class_f1'][c]:.2f} → {T['rf_per_class'][c]:.2f}" for c, _ in drops[:2])

    # explanation-quality table (seed 42 CIs, seed range for the headline measures)
    if 42 in L and St:
        S = {m: L[42][m]['summary'] for m in MODELS}
        T2 = {m: St[m]['summary'] for m in MODELS}
        rows = ['| Measure | RF | CNN |', '|---|---|---|']
        add = lambda lab, a, b: rows.append(f'| {lab} | {a} | {b} |')
        add('Top-5 Jaccard, SHAP vs LIME (same record)', ci(S['rf']['j5']), ci(S['cnn']['j5']))
        add('Top-5 Jaccard, range of means over three seeds', V[f'{tag}_rf_j5_seeds'], V[f'{tag}_cnn_j5_seeds'])
        add('Shared top-5 features (of 5)', ci(S['rf']['shared5']), ci(S['cnn']['shared5']))
        add(f"Chance-corrected top-5 Jaccard (chance = {V[f'{tag}_chance_j5']})", ci(S['rf']['j5_chance_corrected']), ci(S['cnn']['j5_chance_corrected']))
        add('Top-5 Jaccard on fields (one-hot columns merged)', ci(S['rf']['fj5']), ci(S['cnn']['fj5']))
        add('Spearman ρ of \\|attribution\\|', ci(S['rf']['rho']), ci(S['cnn']['rho']))
        add('LIME surrogate R²', ci(S['rf']['lime_r2']), ci(S['cnn']['lime_r2']))
        add('LIME self-agreement, 1,000 samples', ci(T2['rf']['stab_1000']), ci(T2['cnn']['stab_1000']))
        add('LIME self-agreement, 5,000 samples', ci(T2['rf']['stab_5000']), ci(T2['cnn']['stab_5000']))
        add('SHAP self-agreement', ci(T2['rf']['shap_self']), ci(T2['cnn']['shap_self']))
        add('Consensus LIME (20 runs) vs SHAP', ci(T2['rf']['consensus_lime_vs_shap']), ci(T2['cnn']['consensus_lime_vs_shap']))
        for pre, plab in (('del', 'training mean'), ('dels', 'random training record')):
            for name, lab in (('shap', 'SHAP'), ('lime', 'LIME'), ('rand', 'random')):
                add(f'Confidence drop, top-3 {lab} features → {plab}', ci(S['rf'][f'{pre}_{name}_3']), ci(S['cnn'][f'{pre}_{name}_3']))
            add(f'Paired difference SHAP − LIME, top-3 → {plab}', ci(L[42]['rf']['paired'][f'{pre}_shap_minus_lime_3']),
                ci(L[42]['cnn']['paired'][f'{pre}_shap_minus_lime_3']))
        tables[f'{tag}_quality'] = '\n'.join(rows)


# ── v3.1: fixed-feature decomposition, deletion, SHAP budget, feature-space copies ──
Df = load3('UNSW_NB15', 'leakage_decomposition_fixed')
if Df:
    for pair, tag_ in (('dups_kept', 'kept'), ('dups_removed', 'removed')):
        R_ = Df[pair]['seeds']
        V[f'decf_{tag_}_with_f1'] = f(R_['42']['with_id']['macro_f1'], 4)
        V[f'decf_{tag_}_without_f1'] = f(R_['42']['without_id']['macro_f1'], 4)
        V[f'decf_{tag_}_with_f1_seeds'] = rng_str([r['with_id']['macro_f1'] for r in R_.values()], 4)
        V[f'decf_{tag_}_without_f1_seeds'] = rng_str([r['without_id']['macro_f1'] for r in R_.values()], 4)
        V[f'decf_{tag_}_effect'] = ci(R_['42']['id_effect'], 3)
        V[f'decf_{tag_}_effect_mean'] = f(R_['42']['id_effect']['mean'], 3)
        V[f'decf_{tag_}_effect_seeds'] = rng_str([r['id_effect']['mean'] for r in R_.values()], 3)
        V[f'decf_{tag_}_sig_seeds'] = str(sum(r['id_effect']['lo'] > 0 for r in R_.values()))
        V[f'decf_{tag_}_n_test'] = f"{Df[pair]['n_test']:,}"
        V[f'decf_{tag_}_with_acc'] = f(R_['42']['with_id']['accuracy'], 4)
        V[f'decf_{tag_}_without_acc'] = f(R_['42']['without_id']['accuracy'], 4)
        _ae = [r['with_id']['accuracy'] - r['without_id']['accuracy'] for r in R_.values()]
        V[f'decf_{tag_}_acc_effect_mean'] = f(float(np.mean(_ae)), 3)
        V[f'decf_{tag_}_acc_effect_seeds'] = rng_str(_ae, 3)
    V['decf_nfeat'] = str(len(Df['features']))
    V['dec_reselect_kept'] = f(Dc['A_original']['full_test']['macro_f1'] - Dc['B_id_removed_duplicates_kept']['full_test']['macro_f1'], 3)
    V['dec_reselect_removed'] = f(Dc['C_id_kept_duplicates_removed']['full_test']['macro_f1'] - Dc['D_corrected']['full_test']['macro_f1'], 3)
    tables['decomposition'] = '\n'.join([
        '| Model and features | `id` | Duplicates | Test set | Test records | Macro-F1 (seed 42) | Macro-F1, range over seeds | Accuracy (seed 42) |',
        '|---|---|---|---|---|---|---|---|',
        f"| Original model (earlier pipeline) | kept | kept | original, all | {V['dec_A_n']} | {V['dec_A_f1']} | – | {V['dec_A_acc']} |",
        f"| Original model (earlier pipeline) | kept | kept | original, no copy in training | {V['dec_Aunseen_n']} | {V['dec_Aunseen_f1']} | – | {V['dec_Aunseen_acc']} |",
        f"| Original model (earlier pipeline) | kept | kept | original, copy in training | {V['dec_Aseen_n']} | {V['dec_Aseen_f1']} | – | {V['dec_Aseen_acc']} |",
        f"| RF, {V['decf_nfeat']} corrected features + `id` | kept | kept | split 1 | {V['decf_kept_n_test']} | {V['decf_kept_with_f1']} | {V['decf_kept_with_f1_seeds']} | {V['decf_kept_with_acc']} |",
        f"| RF, {V['decf_nfeat']} corrected features | removed | kept | split 1 | {V['decf_kept_n_test']} | {V['decf_kept_without_f1']} | {V['decf_kept_without_f1_seeds']} | {V['decf_kept_without_acc']} |",
        f"| RF, {V['decf_nfeat']} corrected features + `id` | kept | removed | split 2 | {V['decf_removed_n_test']} | {V['decf_removed_with_f1']} | {V['decf_removed_with_f1_seeds']} | {V['decf_removed_with_acc']} |",
        f"| RF, {V['decf_nfeat']} corrected features (corrected pipeline) | removed | removed | split 2 | {V['decf_removed_n_test']} | {V['decf_removed_without_f1']} | {V['decf_removed_without_f1_seeds']} | {V['decf_removed_without_acc']} |",
    ])

STRATS = (('mean', 'training mean'), ('record', 'random training record'), ('gauss', "Gaussian (LIME's sampling)"),
          ('condx', 'nearest records, conditional'))
for ds, tag in DS.items():
    Dl = per_seed(ds, DEL)
    if 42 not in Dl:
        continue
    for m in MODELS:
        S = Dl[42][m]['summary']; P = Dl[42][m]['paired']
        for st, _ in STRATS:
            for k in (1, 3, 5):
                for name in ('shap', 'lime', 'rand'):
                    V[f'{tag}_{m}_d_{st}_{name}_{k}'] = ci(S[f'{st}_{name}_{k}'])
                    V[f'{tag}_{m}_d_{st}_{name}_{k}_mean'] = f(S[f'{st}_{name}_{k}']['mean'])
                for a, b in (('shap', 'lime'), ('shap', 'rand'), ('lime', 'rand')):
                    V[f'{tag}_{m}_d_{st}_{a}_{b}_{k}'] = ci(P[f'{st}_{a}_minus_{b}_{k}'])
        cells = lambda a, b, sts: [Dl[s][m]['paired'][f'{st}_{a}_minus_{b}_{k}'] for s in Dl for st in sts for k in (1, 3, 5)]
        for a, b in (('shap', 'lime'), ('shap', 'rand'), ('lime', 'rand')):
            c = cells(a, b, [x for x, _ in STRATS])
            V[f'{tag}_{m}_d_{a}_{b}_sig'] = str(sum(d['lo'] > 0 for d in c))
            V[f'{tag}_{m}_d_{a}_{b}_pos'] = str(sum(d['mean'] > 0 for d in c))
            V[f'{tag}_{m}_d_{a}_{b}_cells'] = str(len(c))
            for st, _ in STRATS:
                c = cells(a, b, [st])
                V[f'{tag}_{m}_d_{st}_{a}_{b}_sig'] = str(sum(d['lo'] > 0 for d in c))
                V[f'{tag}_{m}_d_{st}_{a}_{b}_cells'] = str(len(c))
        V[f'{tag}_{m}_d_cond_ratio'] = f(S['condx_shap_3']['mean'] / S['mean_shap_3']['mean'])
    Lq = load3(ds, LOCAL, 42); Stq = load3(ds, STAB, 42)
    if Lq and Stq:
        Sx = {m: Lq[m]['summary'] for m in MODELS}; T2 = {m: Stq[m]['summary'] for m in MODELS}
        rows = ['| Measure | RF | CNN |', '|---|---|---|']
        add = lambda lab, a, b: rows.append(f'| {lab} | {a} | {b} |')
        add('Top-5 Jaccard, SHAP vs LIME (same record)', ci(Sx['rf']['j5']), ci(Sx['cnn']['j5']))
        add('Top-5 Jaccard, range of means over three seeds', V[f'{tag}_rf_j5_seeds'], V[f'{tag}_cnn_j5_seeds'])
        add('Top-5 Jaccard, mean over three seeds [seed × record bootstrap CI]', V[f'{tag}_rf_j5_3s'], V[f'{tag}_cnn_j5_3s'])
        add('Shared top-5 features (of 5)', ci(Sx['rf']['shared5']), ci(Sx['cnn']['shared5']))
        add(f"Chance-corrected top-5 Jaccard (chance = {V[f'{tag}_chance_j5']})", ci(Sx['rf']['j5_chance_corrected']), ci(Sx['cnn']['j5_chance_corrected']))
        if int(V[f'{tag}_nfields_local']) < int(V[f'{tag}_nfeat']):   # identical to the feature level when no feature is one-hot
            add(f"Top-5 Jaccard on fields (chance = {V[f'{tag}_chance_field_j5']})", ci(Sx['rf']['fj5']), ci(Sx['cnn']['fj5']))
        add('Top-5 Jaccard, local SHAP vs global SHAP top-5 (baseline)', V[f'{tag}_rf_j5_shap_global'], V[f'{tag}_cnn_j5_shap_global'])
        add('Spearman ρ of \\|attribution\\|', ci(Sx['rf']['rho']), ci(Sx['cnn']['rho']))
        add('LIME surrogate R²', ci(Sx['rf']['lime_r2']), ci(Sx['cnn']['lime_r2']))
        add('LIME self-agreement, 1,000 samples (20 runs)', ci(T2['rf']['stab_1000']), ci(T2['cnn']['stab_1000']))
        add('LIME self-agreement, 5,000 samples (5 runs)', ci(T2['rf']['stab_5000']), ci(T2['cnn']['stab_5000']))
        add('SHAP self-agreement (new backgrounds)', ci(T2['rf']['shap_self']), ci(T2['cnn']['shap_self']))
        add('Consensus LIME (20 runs) vs SHAP', ci(T2['rf']['consensus_lime_vs_shap']), ci(T2['cnn']['consensus_lime_vs_shap']))
        for st, lab in STRATS:
            Ssr, Ssc = Dl[42]['rf']['summary'], Dl[42]['cnn']['summary']
            trip = lambda S_: ' / '.join(f(S_[f'{st}_{n}_3']['mean']) for n in ('shap', 'lime', 'rand'))
            add(f'Confidence drop, top-3 → {lab}: SHAP / LIME / random', trip(Ssr), trip(Ssc))
            add('  paired difference SHAP − LIME', V[f'{tag}_rf_d_{st}_shap_lime_3'], V[f'{tag}_cnn_d_{st}_shap_lime_3'])
        tables[f'{tag}_quality'] = '\n'.join(rows)
        # correct-only agreement and deletion (misclassified records excluded)
        for m in MODELS:
            sc = Lq[m]['summary_correct_only']
            V[f'{tag}_{m}_n_correct'] = str(Lq[m]['n_correct'])
            V[f'{tag}_{m}_j5_correct'] = ci(sc['j5'])
            V[f'{tag}_{m}_pct_wrong'] = f"{100 * (1 - Lq[m]['n_correct'] / Lq[m]['summary']['j5']['n']):.0f}"

    # robustness: LIME ranked by raw weight (slope per standard deviation), the earlier analysis
    Lr, Dr, Sr = per_seed(ds, 'local'), per_seed(ds, 'deletion'), load3(ds, 'stability', 42)
    if 42 in Lr and 42 in Dr and Sr:
        for m in MODELS:
            S = Lr[42][m]['summary']
            for k in ('j5', 'shared5', 'rho', 'fj5', 'j5_chance_corrected'):
                V[f'{tag}_{m}_raw_{k}'] = ci(S[k]); V[f'{tag}_{m}_raw_{k}_mean'] = f(S[k]['mean'])
                V[f'{tag}_{m}_raw_{k}_seeds'] = rng_str([Lr[s][m]['summary'][k]['mean'] for s in Lr])
            for k in ('stab_1000', 'stab_5000', 'consensus_lime_vs_shap'):
                V[f'{tag}_{m}_raw_{k}'] = ci(Sr[m]['summary'][k]); V[f'{tag}_{m}_raw_{k}_mean'] = f(Sr[m]['summary'][k]['mean'])
            c = [Dr[s][m]['paired'][f'{st}_shap_minus_lime_{k}'] for s in Dr for st in ('mean', 'record', 'gauss', 'cond') for k in (1, 3, 5)]
            V[f'{tag}_{m}_raw_d_shap_lime_sig'] = str(sum(d['lo'] > 0 for d in c))
            V[f'{tag}_{m}_raw_d_lime_shap_sig'] = str(sum(d['hi'] < 0 for d in c))
            V[f'{tag}_{m}_raw_d_cells'] = str(len(c))
            c = [Dr[s][m]['paired'][f'{st}_lime_minus_rand_{k}'] for s in Dr for st in ('mean', 'record', 'gauss', 'cond') for k in (1, 3, 5)]
            V[f'{tag}_{m}_raw_d_lime_rand_sig'] = str(sum(d['lo'] > 0 for d in c))
            V[f'{tag}_{m}_raw_d_mean_lime_3_mean'] = f(Dr[42][m]['summary']['mean_lime_3']['mean'])
            # excluding exact copies of the record from the conditional donors
            Dc_ = per_seed(ds, DEL)
            V[f'{tag}_{m}_condx_vs_cond_max'] = f(max(abs(Dc_[s][m]['summary'][f'condx_{n}_{k}']['mean'] - Dc_[s][m]['summary'][f'cond_{n}_{k}']['mean'])
                                                   for s in Dc_ for n in ('shap', 'lime', 'rand') for k in (1, 3, 5)), 3)
        # LIME significantly above SHAP in any cell, either ranking
        V[f'{tag}_lime_gt_shap_any'] = str(sum(int(V[f'{tag}_{m}_raw_d_lime_shap_sig']) for m in MODELS) +
                                           sum(sum(d['hi'] < 0 for s in Dl for st, _ in STRATS for k in (1, 3, 5)
                                                   for d in [Dl[s][m]['paired'][f'{st}_shap_minus_lime_{k}']]) for m in MODELS))
    Hr = per_seed(ds, 'hardening_attacker_r3')
    if Hr:
        V[f'{tag}_copy_unsupported'] = rng_str([Hr[s]['n_copy_unsupported'] for s in Hr], 0)
        V[f'{tag}_copy_unsupported_pct'] = rng_str([100 * Hr[s]['n_copy_unsupported'] / Hr[s]['n_augmented'] for s in Hr], 0)

    B_ = load3(ds, 'shap_budget', 42)
    if B_:
        for name, r in B_['cnn'].items():
            V[f'{tag}_cnn_sb_{name}'] = ci(r['self_agreement'])
            V[f'{tag}_cnn_sb_{name}_mean'] = f(r['self_agreement']['mean'])
            _j = r['self_agreement']['mean']; V[f'{tag}_cnn_sb_{name}_shared'] = f(10 * _j / (1 + _j), 1)   # J -> shared of 5
        for name, r in B_['rf'].items():
            V[f'{tag}_rf_sb_{name}'] = ci(r['self_agreement'])

    Cp = load3(ds, 'feature_space_copies')
    if Cp:
        V[f'{tag}_fcopy_pct'] = pct(Cp['share_copy_with_label'], 1)
        V[f'{tag}_fcopy_feat_pct'] = pct(Cp['share_copy_features_only'], 1)
        if 'share_copy_with_label_8dp' in Cp:
            V[f'{tag}_fcopy_8dp_pct'] = pct(Cp['share_copy_with_label_8dp'], 1)
        pcc = sorted(Cp['per_class_copy_with_label'].items(), key=lambda kv: -kv[1])
        V[f'{tag}_fcopy_top'] = andjoin(f"{cls_name(c)} {100 * v:.0f}%" for c, v in pcc[:2])
        for m in MODELS:
            V[f'{tag}_{m}_f1_nocopy'] = f(Cp['seeds']['42'][m]['macro_f1_without_copies'], 4)
            V[f'{tag}_{m}_f1_nocopy_seeds'] = rng_str([r[m]['macro_f1_without_copies'] for r in Cp['seeds'].values()], 4)
            V[f'{tag}_{m}_acc_copies'] = f(Cp['seeds']['42'][m]['accuracy_on_copies'] or 0, 3)
            V[f'{tag}_{m}_acc_nocopies'] = f(Cp['seeds']['42'][m]['accuracy_without_copies'], 3)

_r = [float(V[f'{t}_{m}_d_cond_ratio']) for t in ('TON', 'UNSW') for m in MODELS if f'{t}_{m}_d_cond_ratio' in V]
if _r:
    V['d_cond_ratio_range'] = f'{min(_r):.2f}–{max(_r):.2f}'

# ── final review round: re-analysis of saved outputs ─────────────────────────
for ds, tag in DS.items():
    Dl = per_seed(ds, DEL)
    if 42 not in Dl:
        continue
    for m in MODELS:
        S = Dl[42][m]['summary']
        for nm in ('lime', 'rand'):
            V[f'{tag}_{m}_d_cond_ratio_{nm}'] = f(S[f'condx_{nm}_3']['mean'] / S[f'mean_{nm}_3']['mean'])
        # non-significant SHAP - LIME cells and the strategies they fall in
        ns_ = [(st, k) for s in Dl for st, _ in STRATS for k in (1, 3, 5) if Dl[s][m]['paired'][f'{st}_shap_minus_lime_{k}']['lo'] <= 0]
        V[f'{tag}_{m}_d_shap_lime_nonsig'] = str(len(ns_))
        # twins excluded (UNSW: 13.7% of test records have a feature-space twin in training)
        if TWIN.get(ds):
            for st, _ in STRATS:
                c = []
                for s in Dl:
                    rr = [r for r in Dl[s][m]['rows'] if r['test_idx'] not in TWIN[ds]]
                    for k in (1, 3, 5):
                        c.append(X3.paired_ci([r[f'{st}_shap_{k}'] for r in rr], [r[f'{st}_lime_{k}'] for r in rr]))
                    if s == 42:
                        V[f'{tag}_{m}_d_{st}_shap_lime_3_notwin'] = ci(X3.paired_ci([r[f'{st}_shap_3'] for r in rr], [r[f'{st}_lime_3'] for r in rr]))
                V[f'{tag}_{m}_d_{st}_shap_lime_sig_notwin'] = str(sum(d['lo'] > 0 for d in c))
            allc = [X3.paired_ci([r[f'{st}_shap_{k}'] for r in Dl[s][m]['rows'] if r['test_idx'] not in TWIN[ds]],
                                 [r[f'{st}_lime_{k}'] for r in Dl[s][m]['rows'] if r['test_idx'] not in TWIN[ds]])
                    for s in Dl for st, _ in STRATS for k in (1, 3, 5)]
            V[f'{tag}_{m}_d_shap_lime_sig_notwin'] = str(sum(d['lo'] > 0 for d in allc))
            V[f'{tag}_{m}_d_shap_lime_pos_notwin'] = str(sum(d['mean'] > 0 for d in allc))
    # normalised agreement J / sqrt(J_SHAP,self * J_LIME,self) (means; stability subsample for the ceilings)
    St = load3(ds, STAB, 42); Lq = load3(ds, LOCAL, 42)
    if St and Lq:
        for m in MODELS:
            j = Lq[m]['summary']['j5']['mean']; a_ = St[m]['summary']['shap_self']['mean']; b_ = St[m]['summary']['stab_1000']['mean']
            V[f'{tag}_{m}_j5_norm'] = f(j / np.sqrt(a_ * b_))
            V[f'{tag}_{m}_j5_ceiling'] = f(np.sqrt(a_ * b_))
    P42 = load3(ds, 'performance', 42)
    if P42:
        cc = P42['class_counts_test']; n_ = sum(cc.values()); b = cc['normal'] / n_
        V[f'{tag}_majority_bin'] = pct(max(b, 1 - b), 1)
        if 'worms' in cc:
            V[f'{tag}_worms_ntest'] = str(cc['worms'])
    Ha = per_seed(ds, 'hardening_attacker')
    if Ha and 'worms' in Ha[42]['query_per_class']:
        V[f'{tag}_worms_nhard'] = rng_str([Ha[s]['query_per_class']['worms'] for s in Ha], 0)
if 'TON_cnn_d_shap_lime_pos' in V and 'UNSW_cnn_d_shap_lime_pos' in V:
    _t, _u, _c = V['TON_cnn_d_shap_lime_pos'], V['UNSW_cnn_d_shap_lime_pos'], V['TON_cnn_d_shap_lime_cells']
    V['cnn_pos_phrase'] = f'{_t} of {_c} on each dataset' if _t == _u else f'{_t} on TON-IoT and {_u} on UNSW-NB15'
if 'UNSW_hard_att_control_ev' in V:
    _a1, _a2 = V['UNSW_hard_att_control_ev'], V['UNSW_hard_att_control_copy_ev']
    _r1, _r2 = V['UNSW_hard_att_control_ev_seeds'], V['UNSW_hard_att_control_copy_ev_seeds']
    V['UNSW_hard_ctl_phrase'] = (f'both controls give {_a1}% (range over seeds {_r1}% for each)' if (_a1, _r1) == (_a2, _r2) else
                                 f'the interpolation control gives {_a1}% ({_r1}%) and the benign-donor control {_a2}% ({_r2}%)')
_a = [int(V[f'{t}_{m}_cf_attacker_succ_3s_mean']) for t in ('TON', 'UNSW') for m in MODELS if f'{t}_{m}_cf_attacker_succ_3s_mean' in V]
if _a:
    V['cf_att_min'], V['cf_att_max'] = str(min(_a)), str(max(_a))
for nm in ('lime', 'rand'):
    _r = [float(V[f'{t}_{m}_d_cond_ratio_{nm}']) for t in ('TON', 'UNSW') for m in MODELS if f'{t}_{m}_d_cond_ratio_{nm}' in V]
    if _r:
        V[f'd_cond_ratio_range_{nm}'] = f'{min(_r):.2f}–{max(_r):.2f}'
if Df:
    for pair, tag_ in (('dups_kept', 'kept'), ('dups_removed', 'removed')):
        r42 = Df[pair]['seeds']['42']
        V[f'decf_{tag_}_acc_effect42'] = f(r42['with_id']['accuracy'] - r42['without_id']['accuracy'], 3)

# ── final review round: new runs (run_xai_final.py); absent until the queue has finished ──
STR4 = ('mean', 'record', 'gauss', 'condx')
LS_CFG = (('cont_w0.5', 'w05'), ('cont_w1', 'w1'), ('cont_w2', 'w2'), ('disc', 'disc'))   # (json key, placeholder tag)
_lead, _ncell, _posneg, _pos_any = 0, 0, 0, False
for ds, tag in DS.items():
    Ls = loadF(ds, 'lime_sens', 42)
    if not Ls:
        continue
    V[f'{tag}_ls_kw'] = f"{Ls['default_kernel_width']:.2f}"
    V['ls_nper'] = str(Ls['n_per_class'])
    g = load2(ds, 'global_shap'); feats = load3(ds, 'performance', 42)['features']
    for m in MODELS:
        S, P = Ls[m]['summary'], Ls[m]['paired']
        V[f'{tag}_{m}_ls_n'] = str(Ls[m]['n'])
        for cfg, ct in LS_CFG:
            V[f'{tag}_{m}_ls_{ct}_j5'] = ci(S[f'{cfg}_j5'])
            V[f'{tag}_{m}_ls_{ct}_j5cc'] = ci(S[f'{cfg}_j5_chance_corrected'])
            V[f'{tag}_{m}_ls_{ct}_r2'] = f(S[f'{cfg}_r2']['mean'])
            cells = [P[f'{st}_shap_minus_{cfg}_{k}'] for st in STR4 for k in (1, 3, 5)]
            V[f'{tag}_{m}_ls_{ct}_sig'] = str(sum(d['lo'] > 0 for d in cells))
            V[f'{tag}_{m}_ls_{ct}_neg'] = str(sum(d['hi'] < 0 for d in cells))
            V[f'{tag}_{m}_ls_{ct}_d_mean_3'] = ci(P[f'mean_shap_minus_{cfg}_3'])
            V[f'{tag}_{m}_ls_{ct}_d_condx_3'] = ci(P[f'condx_shap_minus_{cfg}_3'])
            k3 = [P[f'{st}_shap_minus_{cfg}_3']['mean'] > 0 for st in STR4]
            _lead += all(k3); _ncell += 1
        cells = [P[f'{st}_shappos_minus_limepos_{k}'] for st in STR4 for k in (1, 3, 5)]
        V[f'{tag}_{m}_pos_sig'] = str(sum(d['lo'] > 0 for d in cells))
        V[f'{tag}_{m}_pos_neg'] = str(sum(d['hi'] < 0 for d in cells))
        _posneg += sum(d['hi'] < 0 for d in cells); _pos_any = True
        V[f'{tag}_{m}_pos_d_mean_3'] = ci(P['mean_shappos_minus_limepos_3'])
        V[f'{tag}_{m}_pos_d_condx_3'] = ci(P['condx_shappos_minus_limepos_3'])
        V[f'{tag}_{m}_abs_d_mean_3_ls'] = ci(P['mean_shap_minus_cont_w1_3'])
        V[f'{tag}_{m}_shap_npos5'] = f(S['shap_npos5']['mean'], 1)
        V[f'{tag}_{m}_lime_npos5'] = f(S['cont_w1_npos5']['mean'], 1)
        if g:
            gtop = set(feats.index(x['feature']) for x in g[m]['ranking'][:5])
            jl = [len(set(r['cont_w1_top5']) & gtop) / len(set(r['cont_w1_top5']) | gtop) for r in Ls[m]['rows']]
            V[f'{tag}_{m}_j5_lime_global'] = ci(boot_ci_(jl))
if _ncell == 16:      # both datasets finished
    V['ls_lead_cells'] = f'{_lead} of {_ncell}'
    V['ls_lead_all'] = str(_lead == _ncell)
    V['ls_conclusion'] = ("SHAP's lead in the deletion test therefore does not depend on LIME's kernel width or representation."
                          if _lead == _ncell else
                          "SHAP's lead in the deletion test therefore holds for LIME in continuous mode at its default kernel width, "
                          "but not for every LIME configuration, and our faithfulness comparison should be read as one of that configuration.")
    V['pos_conclusion'] = ("Ranking by sign does not reverse the comparison in any combination." if _posneg == 0 else
                           "With signed ranking LIME is significantly more faithful in some combinations, so part of SHAP's lead "
                           "under the absolute ranking comes from the ranking itself.")

for ds, tag in DS.items():
    Hp = {s: r for s in SEEDS if (r := loadF(ds, 'hardening_paired', s)) is not None}
    if len(Hp) < 3:
        continue
    for other, lab in (('control_copy', 'copy'), ('original', 'orig'), ('control', 'ctl')):
        ps, d42 = [], None
        for s in SEEDS:
            R_ = Hp[s]['records']
            p_, n10, n01 = mcnemar_p(R_[f'evaded_{other}'], R_['evaded_counterfactual'])   # n10: evaded other but not cf
            ps.append(p_)
            if s == 42:
                V[f'{tag}_mcn_{lab}_diff42'] = ci(X3.paired_ci(np.array(R_[f'evaded_{other}'], float), np.array(R_['evaded_counterfactual'], float)), pct=True)
                V[f'{tag}_mcn_{lab}_disc42'] = f'{n10} against {n01}'
        V[f'{tag}_mcn_{lab}_p_byseed'] = andjoin(pfull(p) for p in ps)
        V[f'{tag}_mcn_{lab}_nsig'] = str(sum(p < 0.05 for p in ps))
        # pooled over seeds: stratified (Mantel-Haenszel) McNemar = exact test on the summed discordant pairs
        from scipy.stats import binomtest
        b_ = sum(mcnemar_p(Hp[s]['records'][f'evaded_{other}'], Hp[s]['records']['evaded_counterfactual'])[1] for s in SEEDS)
        c_ = sum(mcnemar_p(Hp[s]['records'][f'evaded_{other}'], Hp[s]['records']['evaded_counterfactual'])[2] for s in SEEDS)
        V[f'{tag}_mcn_{lab}_pooled_p'] = pfull(binomtest(b_, b_ + c_, 0.5).pvalue) if b_ + c_ else 'p = 1.000'
        V[f'{tag}_mcn_{lab}_diff_3s'] = ci(boot2([np.array(Hp[s]['records'][f'evaded_{other}'], float) - np.array(Hp[s]['records']['evaded_counterfactual'], float) for s in SEEDS]), pct=True)
    _n = int(V[f'{tag}_mcn_copy_nsig']); _pp = V[f'{tag}_mcn_copy_pooled_p']
    _pooled_sig = _pp.startswith('p <') or float(_pp.split('=')[-1]) < 0.05
    V[f'{tag}_mcn_conclusion'] = (
        "With the pairing used, the counterfactuals beat the benign-donor control in every seed on UNSW-NB15." if _n == 3 else
        f"Pooled over seeds the counterfactuals beat the benign-donor control on UNSW-NB15, but per seed the difference is significant in {_n} of 3 seeds, so it should be read as a consistent tendency rather than a robust per-model effect." if _pooled_sig else
        f"On UNSW-NB15 the advantage of the counterfactuals over benign-donor copies is therefore not robust: it holds in the mean over seeds but is significant in {_n} of 3 seeds and not when pooled.")
    # reproduces Table 9 (hardening_attacker_r3)?
    r3 = per_seed(ds, 'hardening_attacker_r3')
    V[f'{tag}_hp_reproduces'] = str(all(abs(Hp[s][a]['evasion'] - r3[s][a]['evasion']) < 1e-9 for s in SEEDS for a in ('original', 'counterfactual', 'control', 'control_copy')))
    V[f'{tag}_hp_maxdiff'] = 100 * max(abs(Hp[s][a]['evasion'] - r3[s][a]['evasion']) for s in SEEDS for a in ('original', 'counterfactual', 'control', 'control_copy'))
    # local FPR near the added records (mean over seeds) and set sizes
    for arm in ('original', 'counterfactual', 'control', 'control_copy'):
        for st in ('cf_knn5', 'copy_knn5', 'cf_tau', 'copy_tau'):
            v = [Hp[s][arm]['local_fpr'][st]['fpr'] for s in SEEDS if Hp[s][arm]['local_fpr'][st]['fpr'] is not None]
            V[f'{tag}_lfpr_{arm}_{st}'] = f"{100 * np.mean(v):.1f}" if v else 'n/a'
    for st in ('cf_knn5', 'copy_knn5', 'cf_tau', 'copy_tau'):
        V[f'{tag}_lfpr_n_{st}'] = rng_str([Hp[s]['near_sets'][st] for s in SEEDS], 0)
    # non-controllable features already within the support radius of a benign training record
    ev, nev = [], []
    for s in SEEDS:
        R_ = Hp[s]['records']
        for e, w in zip(R_['evaded_original'], R_['noncontrollable_within_tau']):
            (ev if e else nev).append(w)
    V[f'{tag}_prox_evaded'] = pct(np.mean(ev)) if ev else 'n/a'
    V[f'{tag}_prox_notevaded'] = pct(np.mean(nev)) if nev else 'n/a'
    # evasion without feature-space twins
    if TWIN.get(ds):
        for arm in ('original', 'counterfactual', 'control_copy'):
            v = [np.mean([e for t, e in zip(Hp[s]['records']['test_idx'], Hp[s]['records'][f'evaded_{arm}']) if t not in TWIN[ds]]) for s in SEEDS]
            V[f'{tag}_hard_notwin_{arm}_ev'] = f"{100 * np.mean(v):.0f}"
        V[f'{tag}_hard_notwin_nquery'] = rng_str([sum(t not in TWIN[ds] for t in Hp[s]['records']['test_idx']) for s in SEEDS], 0)

_md = [V[k] for k in ('TON_hp_maxdiff', 'UNSW_hp_maxdiff') if k in V]
if len(_md) == 2:
    V['hp_note'] = ('' if max(_md) < 1e-7 else
        f" These tests come from a second run of the hardening experiment with the same seeds, which stores the per-record outcomes; it reproduces the evasion rates above for the original and counterfactual models exactly, and those of the two controls to within {max(_md):.1f} points.")
    for k in ('TON_hp_maxdiff', 'UNSW_hp_maxdiff'):
        V[k] = f"{V[k]:.1f}"

Dci = loadF('UNSW_NB15', 'decomp_ci', 42)
if Dci:
    from sklearn.metrics import f1_score as _f1
    rng_ = np.random.default_rng(42)
    eff, acc = {}, {}
    for pair in ('dups_kept', 'dups_removed'):
        y_ = np.array(Dci[pair]['y']); a_ = np.array(Dci[pair]['pred']['with_id']); b_ = np.array(Dci[pair]['pred']['without_id'])
        acc[pair] = X3.paired_ci((a_ == y_).astype(float), (b_ == y_).astype(float))
        bs = []
        for _ in range(2000):
            i = rng_.integers(len(y_), size=len(y_))
            bs.append(_f1(y_[i], a_[i], average='macro') - _f1(y_[i], b_[i], average='macro'))
        eff[pair] = np.array(bs)
        V[f"decf_{'kept' if pair == 'dups_kept' else 'removed'}_acc_ci"] = f"[{acc[pair]['lo']:.3f}, {acc[pair]['hi']:.3f}]"
    full = {p: _f1(np.array(Dci[p]['y']), np.array(Dci[p]['pred']['with_id']), average='macro') -
               _f1(np.array(Dci[p]['y']), np.array(Dci[p]['pred']['without_id']), average='macro') for p in eff}
    d_ = eff['dups_kept'] - eff['dups_removed']
    V['decf_interaction_ci'] = ci(dict(mean=full['dups_kept'] - full['dups_removed'], lo=np.percentile(d_, 2.5), hi=np.percentile(d_, 97.5)), 3)
    # common test population: records (by raw id) in both test sets
    common = set(Dci['dups_kept']['test_id']) & set(Dci['dups_removed']['test_id'])
    V['decf_common_n'] = f'{len(common):,}'
    for pair, t_ in (('dups_kept', 'kept'), ('dups_removed', 'removed')):
        msk = np.array([t in common for t in Dci[pair]['test_id']])
        y_ = np.array(Dci[pair]['y'])[msk]; a_ = np.array(Dci[pair]['pred']['with_id'])[msk]; b_ = np.array(Dci[pair]['pred']['without_id'])[msk]
        bs = []
        for _ in range(2000):
            i = rng_.integers(len(y_), size=len(y_))
            bs.append(_f1(y_[i], a_[i], average='macro') - _f1(y_[i], b_[i], average='macro'))
        V[f'decf_common_{t_}_effect'] = ci(dict(mean=_f1(y_, a_, average='macro') - _f1(y_, b_, average='macro'),
                                                lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)), 3)
        V[f'decf_common_{t_}_acc'] = ci(X3.paired_ci((a_ == y_).astype(float), (b_ == y_).astype(float)), 3)

Cm = loadF('TON_IoT', 'cf_monotone', 42)
if Cm:
    for m in MODELS:
        V[f'TON_{m}_cfmono_free'] = ci(Cm[m]['summary']['free']['success'], pct=True)
        V[f'TON_{m}_cfmono_mono'] = ci(Cm[m]['summary']['monotone']['success'], pct=True)
        V[f'TON_{m}_cfmono_diff'] = ci(Cm[m]['summary']['paired_monotone_minus_free'], pct=True)

# ── combined tables over datasets ────────────────────────────────────────────
PRETTY = {'TON': 'TON-IoT', 'UNSW': 'UNSW-NB15'}
rows = ['| Dataset | Model | Macro-F1 (seed 42) | Range over seeds | Benign FPR (%) | Size | ms per 1,000 records |',
        '|---|---|---|---|---|---|---|']
for tag in PRETTY:
    if f'{tag}_rf_f1' not in V:
        continue
    rows.append(f"| {PRETTY[tag]} | RF | {V[f'{tag}_rf_f1']} | {V[f'{tag}_rf_f1_seeds']} | {V[f'{tag}_rf_fpr']} | "
                f"{V[f'{tag}_rf_mb']} MB ({V[f'{tag}_rf_nodes']} tree nodes) | {V[f'{tag}_rf_ms1000']} |")
    rows.append(f"| {PRETTY[tag]} | CNN | {V[f'{tag}_cnn_f1']} | {V[f'{tag}_cnn_f1_seeds']} | {V[f'{tag}_cnn_fpr']} | "
                f"{V[f'{tag}_cnn_mb']} MB ({V[f'{tag}_cnn_params']} parameters) | {V[f'{tag}_cnn_ms1000']} |")
tables['perf'] = '\n'.join(rows)
rows = ['| Dataset | Counterfactuals may change | RF trained on | Evasion (%), mean (range over seeds) | Evasion (%), seed 42 [95% CI] | Macro-F1 | Benign FPR (%) |',
        '|---|---|---|---|---|---|---|']
for tag in PRETTY:
    for variant, lab in (('all', 'any field group'), ('att', 'attacker-controllable groups')):
        if f'{tag}_hard_{variant}_nseeds' not in V:
            continue
        for arm, alab in (('original', 'original data'), ('control', '+ interpolated attacks (control)'),
                          ('control_copy', '+ benign-donor copies (control)'), ('counterfactual', '+ counterfactuals')):
            if f'{tag}_hard_{variant}_{arm}_ev' not in V:
                continue
            rows.append(f"| {PRETTY[tag]} | {lab} | {alab} | {V[f'{tag}_hard_{variant}_{arm}_ev']} ({V[f'{tag}_hard_{variant}_{arm}_ev_seeds']}) | "
                        f"{V.get(f'{tag}_hard_{variant}_{arm}_ev42', '–')} | {V[f'{tag}_hard_{variant}_{arm}_f1']} | {V[f'{tag}_hard_{variant}_{arm}_fpr']} |")
tables['hard'] = '\n'.join(rows)

V['tables'] = tables
# phrasing of all-zero counts (final review)
_neg = [V.get(k) for k in ('TON_rf_ls_disc_neg', 'TON_cnn_ls_disc_neg', 'UNSW_rf_ls_disc_neg', 'UNSW_cnn_ls_disc_neg')]
V['ls_disc_neg_phrase'] = ('significantly negative in none of the four for the discretised mode' if all(str(x) == '0' for x in _neg)
                           else 'significantly negative for the discretised mode: ' + ', '.join(map(str, _neg[:3])) + ' and ' + str(_neg[3]))
_neg = [V.get(k) for k in ('TON_rf_pos_neg', 'TON_cnn_pos_neg', 'UNSW_rf_pos_neg', 'UNSW_cnn_pos_neg')]
V['pos_neg_phrase'] = ('and significantly negative in none' if all(str(x) == '0' for x in _neg)
                       else 'and significantly negative in ' + ', '.join(map(str, _neg[:3])) + ' and ' + str(_neg[3]))
(HERE / 'values.json').write_text(json.dumps(V, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'{len(V) - 1} values, {len(tables)} tables → values.json')
