"""Paper 3 v2 analysis: continual-learning metrics from the stored per-stage confusion matrices.

Per stage t (after learning task t) and task j:
  R[t, j]     class-level retention: mean recall of task j's attack classes (argmax over seen classes)
  Rd[t, j]    detection-level retention: share of task j's attack records flagged as any attack
  zero-day    for j > t: share of not-yet-learned task j attack records flagged as an attack
  benign FPR  share of benign test records flagged as an attack
Summary (T = last stage): A = mean_j R[T, j]; plasticity = mean_j R[j, j];
FM = mean_{j<T} (max_{t<T} R[t, j] - R[T, j]); BWT = mean_{j<T} (R[T, j] - R[j, j]); the same for Rd;
final macro-F1 over all classes. Primary family per dataset: each strategy vs naive on final macro-F1,
paired over the five seeds (paired t-test, Holm). Output: results_v3/summary.json
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RES = HERE / 'results_v3'
STRATS = ('naive', 'joint', 'ewc', 'lwf', 'er500', 'er2000', 'agem')


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(run, 1.0)
    return adj


def metrics(r):
    cls = r['classes']; nrm = cls.index('normal')
    tasks = [[cls.index(c) for c in t if c != 'normal'] for t in r['tasks']]
    T = len(tasks)
    R, Rd, Z = np.full((T, T), np.nan), np.full((T, T), np.nan), np.full((T, T), np.nan)
    fpr, f1s = [], []
    for t, st in enumerate(r['stages']):
        cm = np.array(st['cm'], float)
        rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
        det = 1 - cm[:, nrm] / np.maximum(cm.sum(1), 1)
        for j, cj in enumerate(tasks):
            (R if j <= t else Z)[t, j] = np.mean(rec[cj]) if j <= t else np.average(det[cj], weights=cm[cj].sum(1))
            if j <= t:
                Rd[t, j] = np.average(det[cj], weights=cm[cj].sum(1))
        fpr.append(float(det[nrm]))
        f1s.append(st['macro_f1_seen'])
    out = dict(R=R.tolist(), Rd=Rd.tolist(), zero_day=Z.tolist(), benign_fpr=fpr, macro_f1_stage=f1s,
               final_macro_f1=f1s[-1], task0_macro_f1=f1s[0],
               train_s=float(sum(s['train_s'] for s in r['stages'])), buffer=r['stages'][-1]['buffer'])
    for nm, M in (('', R), ('det_', Rd)):
        out[nm + 'A'] = float(np.mean(M[T - 1]))
        out[nm + 'plasticity'] = float(np.mean(np.diag(M)))
        out[nm + 'FM'] = float(np.mean([np.nanmax(M[:T - 1, j]) - M[T - 1, j] for j in range(T - 1)]))
        out[nm + 'BWT'] = float(np.mean([M[T - 1, j] - M[j, j] for j in range(T - 1)]))
    out['zero_day_mean'] = float(np.nanmean(Z))
    return out


def load_group(ds, group):
    d = {}
    for p in sorted((RES / ds / group).glob('*.json')):
        r = json.loads(p.read_text())
        d.setdefault(r['strategy'], {})[(r['order'], r['seed'])] = (r, metrics(r))
    return d


def summarise(ds):
    S = {}
    for group in ('main', 'orders', 'oracle'):
        G = load_group(ds, group)
        if not G:
            continue
        S[group] = {}
        for s, runs in G.items():
            ms = [m for _, m in runs.values()]
            keys = [k for k, v in ms[0].items() if isinstance(v, float)]
            S[group][s] = {k: dict(mean=float(np.mean([m[k] for m in ms])), sd=float(np.std([m[k] for m in ms], ddof=1)) if len(ms) > 1 else 0.0,
                                   values=[m[k] for m in ms]) for k in keys}
            S[group][s]['n'] = len(ms)
            S[group][s]['R_mean'] = np.nanmean([m['R'] for m in ms], 0).tolist()
            S[group][s]['Rd_mean'] = np.nanmean([m['Rd'] for m in ms], 0).tolist()
            # Task-0 identity check: every strategy shares the Task-0 model under a fixed seed
            S[group][s]['task0_f1_by_run'] = {f'o{o}_s{sd}': runs[(o, sd)][1]['task0_macro_f1'] for o, sd in runs}
        if group == 'main' and 'naive' in G:
            fam, rows = [], []
            for s in STRATS:
                if s == 'naive' or s not in G:
                    continue
                keys = sorted(set(G[s]) & set(G['naive']))
                a = np.array([G[s][k][1]['final_macro_f1'] for k in keys]); b = np.array([G['naive'][k][1]['final_macro_f1'] for k in keys])
                p = stats.ttest_rel(a, b).pvalue if len(keys) > 1 else np.nan
                rows.append(dict(strategy=s, diff=float(np.mean(a - b)), p=float(p), n=len(keys)))
                fam.append(p)
            for row, ph in zip(rows, holm(fam)):
                row['p_holm'] = float(ph)
            S['primary_vs_naive'] = rows
            t0 = {s: G[s] for s in G}
            ident = all(len({round(t0[s][k][1]['task0_macro_f1'], 6) for s in t0 if k in t0[s]}) == 1 for k in t0['naive'])
            S['task0_identical_across_strategies'] = bool(ident)
    rf = sorted((RES / ds / 'rf').glob('*.json'))
    if rf:
        from sklearn.metrics import f1_score
        vals = []
        for p in rf:
            cm = np.array(json.loads(p.read_text())['cm'])
            y = np.repeat(np.arange(len(cm)), cm.sum(1)); yp = np.concatenate([np.repeat(np.arange(len(cm)), row) for row in cm])
            vals.append(f1_score(y, yp, average='macro'))
        S['rf_final_macro_f1'] = dict(mean=float(np.mean(vals)), values=vals)
    tn = RES / ds / 'tuned.json'
    if tn.exists():
        S['tuned'] = json.loads(tn.read_text())
    return S


if __name__ == '__main__':
    out = {ds.name: summarise(ds.name) for ds in sorted(RES.iterdir()) if ds.is_dir()}
    (RES / 'summary.json').write_text(json.dumps(out, indent=1))
    for ds, S in out.items():
        print(f'\n## {ds}  (Task-0 identical: {S.get("task0_identical_across_strategies")})')
        for s, m in S.get('main', {}).items():
            print(f"{s:7s} F1 {m['final_macro_f1']['mean']:.3f}±{m['final_macro_f1']['sd']:.3f}  A {m['A']['mean']:.3f}  FM {m['FM']['mean']:.3f}  "
                  f"detA {m['det_A']['mean']:.3f} detFM {m['det_FM']['mean']:.3f}  zero-day {m['zero_day_mean']['mean']:.3f}  t {m['train_s']['mean']:.0f}s")
        for row in S.get('primary_vs_naive', []):
            print('  vs naive', row)
        if 'rf_final_macro_f1' in S:
            print('  RF (retrained on all data):', round(S['rf_final_macro_f1']['mean'], 4))
