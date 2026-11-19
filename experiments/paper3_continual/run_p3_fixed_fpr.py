"""Paper 3: attack recall at a fixed benign false-positive rate (operating-point metric), after every task.

The stored runs (results_v3, results_p3final) keep only per-stage confusion matrices (argmax over seen classes), so
recall at a fixed FPR cannot be derived from them. This script re-runs the primary configuration (semantic order o0,
seeds 0-4, the tuned hyperparameters in results_v3/<ds>/tuned.json, the shared cached Task-0 models) with the
unchanged training code of run_cl_v2.run(); it only intercepts the test-set prediction after every task to also keep
the logits of the test and validation sets. Every re-run is checked against the stored confusion matrices
(bit-identical argmax predictions are expected: deterministic ops, same seeds, same cached Task-0 weights).

Score: attack score = logsumexp(logits of seen attack classes) - logit(benign), i.e. log((1 - p_benign) / p_benign)
under the softmax restricted to the classes seen so far (monotone in the benign probability; no saturation at 0/1).
Random forest: attack score = 1 - P(benign) (predict_proba).

Threshold (no test data): at stage t the threshold is the (1 - alpha) quantile ('higher') of the attack scores of the
benign VALIDATION records of the current task t (the benign share the stage's own early stopping sees; the same
records for every method at a given stage). A record is flagged as an attack if its score > threshold. Sensitivity:
threshold on the benign validation records of all tasks 0..t ('va_upto').
Metrics per stage: realised test benign FPR, recall over test attack records of the classes seen so far ('recall_seen'),
over all test attack records incl. not-yet-learned ones ('recall_all'), plus the argmax operating point and the ROC AUC
(attack vs benign, seen classes) for reference.

Run from `Papers pipeline/` with the project env:
  python -m experiments.paper3_continual.run_p3_fixed_fpr --dataset TON_IoT --phase cnn [--strategies naive,ewc,...]
  python -m experiments.paper3_continual.run_p3_fixed_fpr --dataset TON_IoT --phase rf
  python -m experiments.paper3_continual.run_p3_fixed_fpr --phase analyze
Outputs: results_p3fpr/<ds>/runs/<method>_o0_s<seed>.{json,npz}  (run record + per-record scores)
         results_p3fpr/fixed_fpr.json                              (mean +- sd over seeds per method/dataset/stage)
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.paper3_continual import run_cl_v2 as R  # noqa: E402

OUT = HERE / 'results_p3fpr'
ALPHAS = (0.01, 0.05)
SEEDS = (0, 1, 2, 3, 4)
CNN = ('naive', 'ewc', 'lwf', 'agem2', 'er2000', 'er500', 'joint')
RF = ('rf_er2000', 'rf_joint')
REF = {s: (R.OUT, 'main') for s in CNN}
REF.update(agem2=(R.OUT_FINAL, 'main'), rf_er2000=(R.OUT_FINAL, 'rfmem'), rf_joint=(R.OUT_FINAL, 'rfmem'))


def attack_score(lg, seen, nrm):
    att = [c for c in seen if c != nrm]
    a = lg[:, att]
    m = a.max(1, keepdims=True)
    return (m[:, 0] + np.log(np.exp(a - m).sum(1)) - lg[:, nrm]).astype(np.float32)


def check_and_save(ds, name, seed, r, sc_te, sc_va, D):
    ref_dir, grp = REF[name]
    ref = json.loads((ref_dir / ds / grp / f'{name}_o0_s{seed}.json').read_text())
    same = [np.array_equal(np.array(a['cm']), np.array(b['cm'])) for a, b in zip(r['stages'], ref['stages'])]
    r.update(dataset=ds, order=0, seed=seed, strategy=name, cm_matches_stored=same)
    p = OUT / ds / 'runs' / f'{name}_o0_s{seed}'
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p.with_suffix('.npz'), score_te=np.stack(sc_te), score_va=np.stack(sc_va),
                        yte=D['yte'].astype(np.int16), yva=D['yva'].astype(np.int16))
    p.with_suffix('.json').write_text(json.dumps(r))
    return same


def phase_cnn(ds, strategies, seeds):
    import tensorflow as tf
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    tf.config.experimental.enable_op_determinism()
    D = R.load(ds)
    tasks = R.task_order(D, 0); T = R.build_tasks(D, tasks, 0); S = R.preprocess(D, T, 0)
    tn = R.tuned(ds)
    hp = lambda s: {tn[s]['param']: tn[s]['value']} if s in tn else {}  # noqa: E731
    nrm = D['classes'].index('normal')
    orig = R.predict_logits
    for seed in seeds:
        for s in strategies:
            if (OUT / ds / 'runs' / f'{s}_o0_s{seed}.json').exists():
                continue
            cap = []

            def pl(model, X, bs=8192):
                out = orig(model, X, bs)
                if X is S['te']:   # the post-task test prediction; also keep the validation logits of this model
                    cap.append((out, orig(model, S['va'], bs)))
                return out
            R.predict_logits = pl
            t0 = time.time()
            try:
                r = R.run(D, S, T, s, seed, hp(s), cache_key=f'{ds}_o0_s{seed}_std')
            finally:
                R.predict_logits = orig
            assert len(cap) == len(r['stages'])
            sc_te = [attack_score(lt, st['seen'], nrm) for (lt, _), st in zip(cap, r['stages'])]
            sc_va = [attack_score(lv, st['seen'], nrm) for (_, lv), st in zip(cap, r['stages'])]
            r['wall_s'] = round(time.time() - t0, 1)
            same = check_and_save(ds, s, seed, r, sc_te, sc_va, D)
            print(f'{ds} {s} s{seed}: {r["wall_s"]:.0f} s, cm matches stored {same}', flush=True)


def phase_rf(ds, seeds):
    """rf_stagewise of run_cl_v2 with predict_proba kept (same hyperparameters, same memory generator and update)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    D = R.load(ds)
    T = R.build_tasks(D, R.task_order(D, 0), 0); S = R.preprocess(D, T, 0)
    nrm = D['classes'].index('normal')
    for seed in seeds:
        for mode, name in (('memory', 'rf_er2000'), ('joint', 'rf_joint')):
            if (OUT / ds / 'runs' / f'{name}_o0_s{seed}.json').exists():
                continue
            buf = R.Buffer(2000, np.random.default_rng(seed + 1)) if mode == 'memory' else None
            stages, seen, sc_te, sc_va = [], [], [], []
            t0 = time.time()
            for t, task in enumerate(T):
                seen = sorted(set(seen) | set(task['classes']))
                if mode == 'joint':
                    tr = np.concatenate([T[k]['tr'] for k in range(t + 1)]); Xf, yf = S['tr'][tr], D['ytr'][tr]
                else:
                    Xt, yt = S['tr'][task['tr']], D['ytr'][task['tr']]
                    Xf, yf = (Xt, yt) if buf.y is None else (np.vstack([Xt, buf.X]), np.concatenate([yt, buf.y]))
                rf = RandomForestClassifier(200, max_depth=20, min_samples_leaf=2, n_jobs=-1, random_state=seed)
                rf.fit(Xf, yf)
                pte, pva = rf.predict_proba(S['te']), rf.predict_proba(S['va'])
                pred = rf.classes_[pte.argmax(1)]
                if buf is not None:
                    buf.add(Xt, yt)
                k = list(rf.classes_).index(nrm)
                sc_te.append((1 - pte[:, k]).astype(np.float32)); sc_va.append((1 - pva[:, k]).astype(np.float32))
                stages.append(dict(task=t, seen=[int(c) for c in seen],
                                   cm=confusion_matrix(D['yte'], pred, labels=range(len(D['classes']))).tolist()))
            r = dict(stages=stages, classes=D['classes'], tasks=[[D['classes'][c] for c in t_['classes']] for t_ in T],
                     wall_s=round(time.time() - t0, 1))
            same = check_and_save(ds, name, seed, r, sc_te, sc_va, D)
            print(f'{ds} {name} s{seed}: {r["wall_s"]:.0f} s, cm matches stored {same}', flush=True)


def op_point(sv_ben, ste, yte, nrm, seen, alpha):
    thr = np.quantile(sv_ben, 1 - alpha, method='higher')
    flag = ste > thr
    ben = yte == nrm
    att_seen = np.isin(yte, seen) & ~ben
    return dict(thr=float(thr), val_fpr=float((sv_ben > thr).mean()), test_fpr=float(flag[ben].mean()),
                recall_seen=float(flag[att_seen].mean()), recall_all=float(flag[~ben].mean()))


def phase_analyze():
    from sklearn.metrics import roc_auc_score
    res, per_run = {}, {}
    for ds in R.DATASETS:
        if not (OUT / ds / 'runs').exists():
            continue
        D = R.load(ds)
        T = R.build_tasks(D, R.task_order(D, 0), 0)
        nrm = D['classes'].index('normal')
        yva = D['yva']
        for name in CNN + RF:
            files = sorted((OUT / ds / 'runs').glob(f'{name}_o0_s*.json'))
            if not files:
                continue
            rows = []
            for f in files:
                r = json.loads(f.read_text()); z = np.load(f.with_suffix('.npz'))
                yte = z['yte'].astype(int)
                st_rows = []
                for t, st in enumerate(r['stages']):
                    seen = st['seen']; ste, sva = z['score_te'][t], z['score_va'][t]
                    ben_cur = T[t]['va'][yva[T[t]['va']] == nrm]
                    ben_upto = np.concatenate([T[k]['va'] for k in range(t + 1)]); ben_upto = ben_upto[yva[ben_upto] == nrm]
                    cm = np.array(st['cm'], float)
                    att_seen = [c for c in seen if c != nrm]
                    m_seen = np.isin(yte, seen)
                    row = dict(task=t, n_val_benign_cur=int(len(ben_cur)), n_val_benign_upto=int(len(ben_upto)),
                               argmax_fpr=float(1 - cm[nrm, nrm] / cm[nrm].sum()),
                               argmax_recall_seen=float(1 - cm[att_seen, nrm].sum() / cm[att_seen].sum()),
                               auc_seen=float(roc_auc_score(yte[m_seen] != nrm, ste[m_seen])))
                    for a in ALPHAS:
                        tag = f'{int(round(100 * a))}pct'
                        row[f'cur_{tag}'] = op_point(sva[ben_cur], ste, yte, nrm, seen, a)
                        row[f'upto_{tag}'] = op_point(sva[ben_upto], ste, yte, nrm, seen, a)
                    st_rows.append(row)
                rows.append(dict(seed=r['seed'], cm_matches_stored=r['cm_matches_stored'], stages=st_rows))
            per_run[f'{ds}/{name}'] = rows
            # aggregate: mean and sd (ddof=1) over seeds per stage
            agg = []
            for t in range(len(rows[0]['stages'])):
                A = {}
                def put(key, vals):
                    v = np.asarray(vals, float)
                    A[key] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0)
                for key in ('argmax_fpr', 'argmax_recall_seen', 'auc_seen'):
                    put(key, [x['stages'][t][key] for x in rows])
                for th in ('cur', 'upto'):
                    for a in ALPHAS:
                        tag = f'{th}_{int(round(100 * a))}pct'
                        for m in ('test_fpr', 'recall_seen', 'recall_all', 'val_fpr'):
                            put(f'{tag}_{m}', [x['stages'][t][tag][m] for x in rows])
                A['n_val_benign_cur'] = rows[0]['stages'][t]['n_val_benign_cur']
                A['n_val_benign_upto'] = rows[0]['stages'][t]['n_val_benign_upto']
                agg.append(dict(task=t, **A))
            res.setdefault(ds, {})[name] = dict(n_seeds=len(rows), seeds=[x['seed'] for x in rows],
                                                all_cm_match=all(all(x['cm_matches_stored']) for x in rows), stages=agg)
    meta = dict(source='re-run of the primary configuration (semantic order o0, seeds 0-4) with per-record scores; '
                       'run_p3_fixed_fpr.py', score_cnn='logsumexp(seen attack logits) - benign logit',
                score_rf='1 - P(benign)',
                threshold_primary="'cur': (1-alpha) quantile of attack scores of the current task's benign validation share",
                threshold_sensitivity="'upto': same over the benign validation records of tasks 0..t",
                recall_seen='test attack records of classes seen so far', recall_all='all test attack records',
                sd='sample SD over seeds (ddof=1)')
    OUT.mkdir(exist_ok=True)
    (OUT / 'fixed_fpr.json').write_text(json.dumps(dict(meta=meta, results=res, per_run=per_run), indent=1))
    # console table: final stage, primary threshold
    print(f"{'dataset':10s} {'method':10s} {'n':>2s} {'rec@1%':>15s} {'testFPR@1%':>15s} {'rec@5%':>15s} {'testFPR@5%':>15s} "
          f"{'argmax rec/FPR':>16s} cm_ok")
    for ds, M in res.items():
        for name, v in M.items():
            f = v['stages'][-1]
            c = lambda k: f"{f[k]['mean']:.3f}±{f[k]['sd']:.3f}"  # noqa: E731
            print(f"{ds:10s} {name:10s} {v['n_seeds']:2d} {c('cur_1pct_recall_seen'):>15s} {c('cur_1pct_test_fpr'):>15s} "
                  f"{c('cur_5pct_recall_seen'):>15s} {c('cur_5pct_test_fpr'):>15s} "
                  f"{f['argmax_recall_seen']['mean']:.3f}/{f['argmax_fpr']['mean']:.3f} {v['all_cm_match']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=R.DATASETS)
    ap.add_argument('--phase', choices=('cnn', 'rf', 'analyze'), required=True)
    ap.add_argument('--strategies', default='naive,ewc,lwf,agem2,er2000')
    ap.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(',')]
    if a.phase == 'cnn':
        phase_cnn(a.dataset, a.strategies.split(','), seeds)
    elif a.phase == 'rf':
        phase_rf(a.dataset, seeds)
    else:
        phase_analyze()


if __name__ == '__main__':
    main()
