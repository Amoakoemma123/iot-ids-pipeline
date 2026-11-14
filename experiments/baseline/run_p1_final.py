"""Paper 1 final review round (Revision_v2/Paper1/review_final_*.md, 2026-11-07): tree re-runs and validation scores.

NEW script; run_p1_v2.py (and its make_tree, used by the Paper 2 scripts) is unchanged. Results go to a NEW folder,
results_p1final/<dataset>/<config>/seed<k>.json (+ _pred.npy); results_v2 is never written.

Groups (--group):
  gb_rollback  the 8 GBDT configurations of the v2 grid (7/15/31/63 leaves, lr 0.1 and 0.05), 5 seeds, fitted exactly
               as in run_p1_v2.make_tree and then rolled back to the best internal-validation iteration
               (gbdt_rollback.rollback_best_iter; scikit-learn keeps the 20 non-improving iterations, in which the
               boosting diverged on some seeds: refs/code review C1). Records the pre-rollback ('raw_*') and rolled-back
               test metrics, best_iter, n_iter_raw, number of trees, compact/pickled size, validation-split metrics.
  gb_capped    size-budgeted GBDTs (referee M1): max_iter 50/100/200/400 x 7/15/31 leaves, lr 0.1 (CAP_LR), early
               stopping + rollback, so that GBDTs exist below 100 KB. Config names gbc<iters>_l<leaves>[_lr05].
  small_trees  single decision trees (depth 4/6/8/12) and shallow forests (10 or 25 trees of depth 4/6) (referee M1).
  trees_val    the v2 RF and LR configurations refitted (deterministic: same random_state, same data) to record
               validation-split metrics for model selection on the validation set (referee M3). The test macro-F1
               of the refit is compared with results_v2 ('v2_macro_f1').
  gb_ports     TON-IoT port ablation of gb_l31 / gb05_l31 (src_port, dst_port removed) with rollback -> <cfg>_noports.
  gb_extval    sensitivity (referee M5): the 8 GBDT configurations early-stopped on the provided (non-oversampled)
               validation split instead of an internal 10% of the SMOTE-oversampled training set, with rollback.
               Config names <cfg>_ev. TON-IoT and UNSW-NB15 only.
  nets_val0    validation-split macro-F1 of the saved seed-0 networks of results_v2 (float32 .h5 and int8 .tflite);
               inference only. Written to <cfg>/val_seed0.json.
  nets_val     OPTIONAL full re-training of the 6 networks x 5 seeds (GPU, several hours) recording validation-split
               metrics for float32 and int8; same code as run_p1_v2.run_net otherwise. Not part of the default queue.
Every tree record also stores 'compact_q_bytes' for GBDTs: a quantised encoding (8-bit feature index, 8-bit bin index,
two 16-bit child indices = 6 bytes per internal node, float16 leaf values, plus the float32 bin-edge table of the
features used) and the test macro-F1 with leaf values rounded to float16 ('q16_macro_f1').

Run from `Papers pipeline/`:  python -m experiments.baseline.run_p1_final --dataset D --group G [--smoke]
"""
import argparse, gc, json, os, pickle, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.baseline import run_p1_v2 as R  # noqa: E402
from experiments.baseline.gbdt_rollback import rollback_best_iter  # noqa: E402
from experiments.paper6_multimodal import run_p6_v2 as P6  # noqa: E402

OUT = ROOT / 'experiments' / 'baseline' / 'results_p1final'
V2 = ROOT / 'experiments' / 'baseline' / 'results_v2'
DATASETS = R.DATASETS
SEEDS = R.SEEDS

GB_CFGS = {k: v for k, v in R.TREE_CFGS.items() if v[0] == 'gb'}                     # gb_l*, gb05_l* (8)
RFLR_CFGS = {k: v for k, v in R.TREE_CFGS.items() if v[0] in ('rf', 'lr')}           # 8 RF + LR
CAP_ITERS, CAP_LEAVES, CAP_LR = (50, 100, 200, 400), (7, 15, 31), (0.1,)
CAPPED = {f'gbc{it}_l{l}' + ('' if lr == 0.1 else '_lr05'): ('gb', dict(max_leaf_nodes=l, learning_rate=lr, max_iter=it))
          for lr in CAP_LR for it in CAP_ITERS for l in CAP_LEAVES}
SMALL = {**{f'dt_d{d}': ('dt', dict(max_depth=d)) for d in (4, 6, 8, 12)},
         **{f'rf_t{n}_d{d}': ('rf', dict(n_estimators=n, max_depth=d, min_samples_leaf=1)) for n in (10, 25) for d in (4, 6)}}
EV_DATASETS = ('TON_IoT', 'UNSW_NB15')


# ── models ────────────────────────────────────────────────────────────────────

def make_gb(params, seed, extval=False):
    """identical to run_p1_v2.make_tree('gb', ...) when params has no max_iter and extval is False"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    p = dict(params)
    max_iter = p.pop('max_iter', 1000)
    if not extval and max_iter == 1000:
        return R.make_tree('gb', p, seed)
    kw = dict(validation_fraction=None) if extval else dict(validation_fraction=0.1)
    return HistGradientBoostingClassifier(max_iter=max_iter, early_stopping=True, n_iter_no_change=20,
                                          random_state=seed, **kw, **p)


def make_model(kind, params, seed):
    if kind == 'dt':
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(random_state=seed, **params)
    if kind == 'gb':
        return make_gb(params, seed)
    return R.make_tree(kind, params, seed)


def compact_bytes(t, n_classes, n_features):
    """run_p1_v2.compact_bytes, extended to a single decision tree (same per-node encoding as one RF tree)"""
    from sklearn.tree import DecisionTreeClassifier
    if isinstance(t, DecisionTreeClassifier):
        leaf = t.tree_.children_left == -1
        return 10 * int((~leaf).sum()) + 2 * n_classes * int(leaf.sum())
    return R.compact_bytes(t, n_classes, n_features)


def gb_quantised(t, D, cols):
    """compact_q_bytes and test macro-F1 with float16 leaf values (see module docstring)"""
    from sklearn.metrics import f1_score
    n_int = n_leaf = 0
    used = set()
    for it in t._predictors:
        for pr in it:
            leaf = pr.nodes['is_leaf'].astype(bool)
            n_int += int((~leaf).sum()); n_leaf += int(leaf.sum())
            used.update(int(f) for f in pr.nodes['feature_idx'][~leaf])
    edges = sum(len(t._bin_mapper.bin_thresholds_[f]) for f in used)
    qbytes = 6 * n_int + 2 * n_leaf + 4 * edges
    saved = [[pr.nodes['value'].copy() for pr in it] for it in t._predictors]
    try:
        for it in t._predictors:
            for pr in it:
                pr.nodes['value'] = pr.nodes['value'].astype(np.float16).astype(np.float64)
        pred = t.classes_[t.predict_proba(D['Xte'][:, cols]).argmax(1)]
        f16 = float(f1_score(D['yte'], pred, average='macro'))
    finally:
        for it, vals in zip(t._predictors, saved):
            for pr, v in zip(it, vals):
                pr.nodes['value'] = v
    return dict(compact_q_bytes=int(qbytes), q16_macro_f1=f16, n_internal=n_int, n_leaves=n_leaf, bin_edges=int(edges))


def val_metrics(D, pred, proba=None):
    """validation-split metrics (the split the networks early-stop on; never resampled)"""
    from sklearn.metrics import f1_score
    y = D['yva']
    att = y != D['normal']
    out = dict(val_macro_f1=float(f1_score(y, pred, average='macro')),
               val_benign_fpr=float((pred[~att] != D['normal']).mean()),
               val_attack_detection=float((pred[att] != D['normal']).mean()))
    if proba is not None:
        from sklearn.metrics import log_loss
        out['val_log_loss'] = float(log_loss(y, np.clip(proba, 1e-15, 1), labels=range(len(D['classes']))))
    return out


def predict_full(t, X, n_classes):
    Pc = t.predict_proba(X)
    P = np.zeros((len(Pc), n_classes)); P[:, t.classes_] = Pc
    return P, t.classes_[Pc.argmax(1)]


# ── runs ──────────────────────────────────────────────────────────────────────

def run_tree(D, cfg, kind, params, seed, cols, rollback, extval=False, save_model=False, exp=None):
    nc = len(D['classes'])
    Xf, Xv, Xt = (D['Qfit'], D['Qva'], D['Qte']) if kind == 'lr' else (D['Xfit'], D['Xva'], D['Xte'])
    t = make_gb(params, seed, extval) if kind == 'gb' else make_model(kind, params, seed)
    t0 = time.time()
    if extval:
        t.fit(Xf[:, cols], D['yfit'], X_val=Xv[:, cols], y_val=D['yva'])
    else:
        t.fit(Xf[:, cols], D['yfit'])
    res = dict(train_s=time.time() - t0)
    if kind == 'gb':
        P, pred = predict_full(t, Xt[:, cols], nc)
        raw = P6.metrics(D, pred, P)
        res.update(raw_macro_f1=raw['macro_f1'], raw_benign_fpr=raw['benign_fpr'], raw_clean_macro_f1=raw['clean_macro_f1'],
                   raw_compact_bytes=R.compact_bytes(t, nc, len(cols)))
        if rollback:
            res.update(rollback_best_iter(t))
        res['n_iter'] = int(t.n_iter_)
        res['n_trees'] = int(t.n_iter_ * t.n_trees_per_iteration_)
        res['cap_reached'] = bool(res.get('n_iter_raw', t.n_iter_) >= t.max_iter)
    P, pred = predict_full(t, Xt[:, cols], nc)
    Pv, predv = predict_full(t, Xv[:, cols], nc)
    blob = pickle.dumps(t, protocol=pickle.HIGHEST_PROTOCOL)
    res.update(size_bytes=len(blob), compact_bytes=compact_bytes(t, nc, len(cols)), **P6.metrics(D, pred, P),
               **val_metrics(D, predv, Pv))
    if kind == 'gb':
        res.update(gb_quantised(t, D, cols))
    if kind == 'rf':
        res['nodes'] = int(sum(e.tree_.node_count for e in t.estimators_))
    if kind == 'dt':
        res['nodes'] = int(t.tree_.node_count)
    v2 = V2 / D['ds'] / (exp or cfg) / f'seed{seed}.json'
    if v2.exists():
        res['v2_macro_f1'] = json.loads(v2.read_text())['macro_f1']
    return res, pred, (blob if save_model else None)


def run_net_val(D, cfg, seed, cols, out_dir):
    """run_p1_v2.run_net with validation-split metrics for float32 and int8 (models saved under out_dir, seed 0)"""
    import tensorflow as tf
    kind, w = R.NET_CFGS[cfg]
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    m = R.build_net(kind, w, len(cols), len(D['classes']))
    m.compile(tf.keras.optimizers.Adam(R.LR), 'sparse_categorical_crossentropy', jit_compile=R.JIT)
    bw = P6.BestWeights()
    cb = [tf.keras.callbacks.EarlyStopping(patience=R.PATIENCE, monitor='val_loss'), bw,
          tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5, monitor='val_loss')]
    Xf, Xv, Xt = (D[k][:, cols][..., None] for k in ('Qfit', 'Qva', 'Qte'))
    t0 = time.time()
    h = m.fit(Xf, D['yfit'], validation_data=(Xv, D['yva']), epochs=R.EPOCHS, batch_size=R.BATCH, callbacks=cb, verbose=0)
    res = dict(train_s=time.time() - t0, epochs=len(h.history['loss']), params=int(m.count_params()),
               best_epoch=int(np.argmin(h.history['val_loss'])) + 1, restored_epoch=bw.epoch,
               val_loss_best=float(np.min(h.history['val_loss'])))
    assert bw.epoch == res['best_epoch']
    P = m.predict(Xt, batch_size=4096, verbose=0); pred = P.argmax(1)
    Pv = m.predict(Xv, batch_size=4096, verbose=0)
    res.update(P6.metrics(D, pred, P)); res.update(val_metrics(D, Pv.argmax(1), Pv))
    blob = R.tflite_int8(m, Xf[np.random.default_rng(seed).choice(len(Xf), 500, replace=False)])
    Pq, Pqv = R.tflite_predict(blob, Xt), R.tflite_predict(blob, Xv)
    q, qv = P6.metrics(D, Pq.argmax(1), Pq), val_metrics(D, Pqv.argmax(1), Pqv)
    res['size_bytes'] = 4 * res['params']
    res['int8'] = dict(size_bytes=len(blob), macro_f1=q['macro_f1'], benign_fpr=q['benign_fpr'],
                       attack_detection=q['attack_detection'], macro_pr_auc=q.get('macro_pr_auc'),
                       agreement=float((Pq.argmax(1) == pred).mean()), **qv)
    if seed == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        m.save(out_dir / 'model_seed0.h5')
    return res, pred, blob


def nets_val0(D):
    """validation macro-F1 of the saved seed-0 networks (results_v2), float32 and int8; inference only"""
    import tensorflow as tf
    for cfg in R.NET_CFGS:
        d = V2 / D['ds'] / cfg
        dst = OUT / D['ds'] / cfg / 'val_seed0.json'
        if dst.exists() or not (d / 'model_seed0.h5').exists():
            continue
        m = tf.keras.models.load_model(d / 'model_seed0.h5', compile=False)
        Xv, Xt = D['Qva'][..., None], D['Qte'][..., None]
        Pv = m.predict(Xv, batch_size=4096, verbose=0)
        Pt = m.predict(Xt, batch_size=4096, verbose=0)
        res = dict(val_metrics(D, Pv.argmax(1), Pv), test_macro_f1_check=P6.metrics(D, Pt.argmax(1))['macro_f1'],
                   v2_macro_f1=json.loads((d / 'seed0.json').read_text())['macro_f1'])
        blob = (d / 'model_seed0.tflite').read_bytes()
        Pq = R.tflite_predict(blob, Xv)
        res['int8'] = val_metrics(D, Pq.argmax(1), Pq)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(res, indent=1))
        print(f"  {cfg} seed0: val macro-F1 {res['val_macro_f1']:.4f} int8 {res['int8']['val_macro_f1']:.4f} "
              f"(test check {res['test_macro_f1_check']:.4f} vs v2 {res['v2_macro_f1']:.4f})", flush=True)


def jobs(D, group):
    allc = np.arange(len(D['feat']))
    if group == 'gb_rollback':
        for seed in SEEDS:
            for cfg, (kind, p) in GB_CFGS.items():
                yield cfg, cfg, kind, p, seed, allc, dict(rollback=True, save_model=True)
    elif group == 'gb_capped':
        for seed in SEEDS:
            for cfg, (kind, p) in CAPPED.items():
                yield cfg, cfg, kind, p, seed, allc, dict(rollback=True, save_model=True)
    elif group == 'small_trees':
        for seed in SEEDS:
            for cfg, (kind, p) in SMALL.items():
                yield cfg, cfg, kind, p, seed, allc, dict(rollback=False, save_model=seed == 0)
    elif group == 'trees_val':
        for seed in SEEDS:
            for cfg, (kind, p) in RFLR_CFGS.items():
                yield cfg, cfg, kind, p, seed, allc, dict(rollback=False, save_model=False)
    elif group == 'gb_ports':     # TON-IoT port ablation of the two GBDTs of run_p1_v2 'ports', with rollback
        if D['ds'] == 'TON_IoT':
            keep = np.array([i for i, n in enumerate(D['feat']) if n not in R.PORTS])
            assert len(keep) == len(D['feat']) - 2
            for seed in SEEDS:
                for cfg in ('gb_l31', 'gb05_l31'):
                    yield cfg, f'{cfg}_noports', 'gb', GB_CFGS[cfg][1], seed, keep, dict(rollback=True, save_model=False)
    elif group == 'gb_extval':
        if D['ds'] in EV_DATASETS:
            for seed in SEEDS:
                for cfg, (kind, p) in GB_CFGS.items():
                    yield cfg, f'{cfg}_ev', kind, p, seed, allc, dict(rollback=True, save_model=False, extval=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='all', choices=('all',) + DATASETS)
    ap.add_argument('--group', required=True,
                    choices=('gb_rollback', 'gb_capped', 'small_trees', 'trees_val', 'gb_ports', 'gb_extval', 'nets_val0', 'nets_val'))
    ap.add_argument('--smoke', action='store_true', help='seed 0, two configurations, results_p1final_smoke/')
    a = ap.parse_args()
    global OUT, SEEDS, GB_CFGS, CAPPED, SMALL, RFLR_CFGS
    if a.smoke:
        OUT, SEEDS = OUT.parent / 'results_p1final_smoke', (0,)
        GB_CFGS = {k: GB_CFGS[k] for k in ('gb_l7', 'gb05_l7')}
        CAPPED = {k: CAPPED[k] for k in ('gbc50_l7',)}
        SMALL = {k: SMALL[k] for k in ('dt_d4', 'rf_t10_d4')}
        RFLR_CFGS = {k: RFLR_CFGS[k] for k in ('rf_t25_d8', 'lr')}
    if a.group.startswith('nets'):
        import tensorflow as tf
        for g in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(g, True)
    for ds in DATASETS if a.dataset == 'all' else (a.dataset,):
        if (a.group == 'gb_extval' and ds not in EV_DATASETS) or (a.group == 'gb_ports' and ds != 'TON_IoT'):
            continue
        D = P6.load(ds)
        print(f'== {ds} [{a.group}]: {len(D["feat"])} features, fit {len(D["yfit"]):,}, val {len(D["yva"]):,}, '
              f'test {len(D["yte"]):,}', flush=True)
        (OUT / ds).mkdir(parents=True, exist_ok=True)
        if a.group == 'nets_val0':
            nets_val0(D); del D; gc.collect(); continue
        if a.group == 'nets_val':
            allc = np.arange(len(D['feat']))
            for seed in SEEDS:
                for cfg in R.NET_CFGS:
                    path = OUT / ds / cfg / f'seed{seed}.json'
                    if path.exists():
                        continue
                    t0 = time.time()
                    res, pred, blob = run_net_val(D, cfg, seed, allc, OUT / ds / cfg)
                    res['cfg'] = dict(config=cfg, exp=cfg, seed=seed, kind=R.NET_CFGS[cfg][0], width=R.NET_CFGS[cfg][1],
                                      epochs_budget=R.EPOCHS, jit_compile=R.JIT, group='nets_val')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path.with_name(f'seed{seed}_pred.npy'), pred.astype(np.int8))
                    if seed == 0:
                        path.with_name('model_seed0.tflite').write_bytes(blob)
                    path.write_text(json.dumps(res, indent=1))
                    print(f"  {cfg} seed{seed}: macro-F1 {res['macro_f1']:.4f} val {res['val_macro_f1']:.4f} "
                          f"int8 {res['int8']['macro_f1']:.4f} ({time.time() - t0:.0f}s)", flush=True)
                    gc.collect()
            del D; gc.collect(); continue
        for cfg, exp, kind, p, seed, cols, opt in jobs(D, a.group):
            path = OUT / ds / exp / f'seed{seed}.json'
            if path.exists():
                continue
            t0 = time.time()
            res, pred, blob = run_tree(D, cfg, kind, p, seed, cols, exp=exp, **opt)
            res['cfg'] = dict(config=cfg, exp=exp, seed=seed, kind=kind, group=a.group, **p,
                              early_stopping_on=('validation split' if opt.get('extval') else
                                                 ('internal 10% of the oversampled training set' if kind == 'gb' else None)),
                              rollback=opt['rollback'])
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path.with_name(f'seed{seed}_pred.npy'), pred.astype(np.int8))
            if blob is not None:
                path.with_name(f'model_seed{seed}.pkl').write_bytes(blob)
            path.write_text(json.dumps(res, indent=1))
            extra = (f" (raw {res['raw_macro_f1']:.4f}, iter {res.get('best_iter', res['n_iter'])}/{res.get('n_iter_raw', res['n_iter'])})"
                     if kind == 'gb' else '')
            chk = f" v2 {res['v2_macro_f1']:.4f}" if 'v2_macro_f1' in res and kind != 'gb' else ''
            print(f"  {exp} seed{seed}: macro-F1 {res['macro_f1']:.4f}{extra}{chk} val {res['val_macro_f1']:.4f} "
                  f"size {res['compact_bytes'] / 1024:.0f} KB ({time.time() - t0:.0f}s)", flush=True)
            gc.collect()
        del D; gc.collect()


if __name__ == '__main__':
    main()
