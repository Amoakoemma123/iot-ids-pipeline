"""Paper 1 v2: detection accuracy against model footprint for lightweight IoT intrusion detection.

Replaces run_experiment.py (v1) for the paper. v1 problems fixed here:
  * CNN inputs were min–max scaled, heavy-tailed flow counts with batch normalisation (the silent failure
    found in Papers 3 and 6) -> networks get the quantile-transformed features (same as Paper 6);
  * Keras 2.10 restores best weights only when early stopping fires -> BestWeights always restores;
  * the LSTM slid a window over randomly shuffled records (no temporal meaning) -> dropped;
  * single seed, untuned and unbounded random forest (95 MB) compared with a 24k-parameter CNN ->
    five seeds, size-budgeted random forests and gradient-boosted trees, networks of three widths,
    post-training int8 quantisation of the networks;
  * port-number shortcut not tested -> 'ports' group drops src_port/dst_port on TON-IoT.
Data: exactly the Paper 4-6 data (experiments.paper6_multimodal.run_p6_v2.load, split 0): corrected
pipeline for TON-IoT/UNSW-NB15 (SMOTE on the training split), N-BaIoT as prepared for Paper 4.
Results: results_v2/<dataset>/<config>/seed<k>.json (+ _pred.npy); seed-0 models saved for the
latency benchmark (bench_p1_latency.py).

Run from `Papers pipeline/`:  python -m experiments.baseline.run_p1_v2 --dataset D --group trees|nets|ports
"""
import argparse, gc, json, os, pickle, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.paper6_multimodal import run_p6_v2 as P6  # noqa: E402

OUT = ROOT / 'experiments' / 'baseline' / 'results_v2'
DATASETS = ('TON_IoT', 'UNSW_NB15', 'N_BaIoT')
SEEDS = (0, 1, 2, 3, 4)
PORTS = ('src_port', 'dst_port')
EPOCHS, PATIENCE, BATCH, LR = 100, 15, 256, 1e-3     # v1 budget for the CNN (epochs 100, patience 15, batch 256)
JIT = True          # XLA-compile the training step (about 5x faster for these small nets on the GTX 1650)

# size budgets: every configuration is a point on the accuracy-footprint plane
TREE_CFGS = {
    **{f'rf_t{n}_d{d}': ('rf', dict(n_estimators=n, max_depth=d, min_samples_leaf=1))
       for n in (25, 100) for d in (8, 12, 16)},
    'rf_t200_dNone': ('rf', dict(n_estimators=200, max_depth=None, min_samples_leaf=1)),
    'rf_v1': ('rf', dict(n_estimators=200, max_depth=20, min_samples_leaf=2)),       # v1 configuration
    **{f'gb_l{l}': ('gb', dict(max_leaf_nodes=l, learning_rate=0.1)) for l in (7, 15, 31, 63)},
    **{f'gb05_l{l}': ('gb', dict(max_leaf_nodes=l, learning_rate=0.05)) for l in (7, 15, 31, 63)},   # lr 0.1 collapses on some seeds
    'lr': ('lr', dict(C=1.0)),
}
NET_CFGS = {f'{k}_w{w}': (k, w) for k in ('cnn', 'mlp') for w in (0.5, 1, 2)}


# ── models ────────────────────────────────────────────────────────────────────

def make_tree(kind, params, seed):
    if kind == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(max_features='sqrt', n_jobs=-1, random_state=seed, **params)
    if kind == 'gb':
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=1000, early_stopping=True, validation_fraction=0.1,
                                              n_iter_no_change=20, random_state=seed, **params)
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000, random_state=seed, **params)


def build_net(kind, w, n_features, n_classes):
    """cnn = the v1 lightweight 1D-CNN (Conv32-BN-Pool, Conv64-BN-Pool, GAP, Dense128-Drop, Dense64-Drop),
    widths scaled by w; mlp = Dense(128w)-BN-Dense(64w)-BN-Drop with a 64w head."""
    import tensorflow as tf
    from tensorflow.keras import layers as L
    r = lambda n: max(4, round(n * w))
    inp = L.Input((n_features, 1))
    if kind == 'cnn':
        x = L.Conv1D(r(32), 3, activation='relu', padding='same')(inp)
        x = L.BatchNormalization()(x)
        x = L.MaxPooling1D(2, padding='same')(x)
        x = L.Conv1D(r(64), 3, activation='relu', padding='same')(x)
        x = L.BatchNormalization()(x)
        x = L.MaxPooling1D(2, padding='same')(x)
        x = L.GlobalAveragePooling1D()(x)
        x = L.Dropout(0.3)(L.Dense(r(128), activation='relu')(x))
        x = L.Dropout(0.3)(L.Dense(r(64), activation='relu')(x))
    else:
        x = L.Flatten()(inp)
        x = L.BatchNormalization()(L.Dense(r(128), activation='relu')(x))
        x = L.BatchNormalization()(L.Dense(r(64), activation='relu')(x))
        x = L.Dropout(0.3)(L.Dense(r(64), activation='relu')(x))
    return tf.keras.Model(inp, L.Dense(n_classes, activation='softmax')(x))


def tflite_int8(m, Xcal):
    """post-training full-integer quantisation (weights and activations int8, float input/output)"""
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_keras_model(m)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([Xcal[i:i + 1]] for i in range(len(Xcal)))
    return conv.convert()


def tflite_predict(blob, X):
    import tensorflow as tf
    it = tf.lite.Interpreter(model_content=blob, num_threads=1)
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    it.resize_tensor_input(inp['index'], [len(X), *X.shape[1:]]); it.allocate_tensors()
    it.set_tensor(inp['index'], X.astype(np.float32)); it.invoke()
    return it.get_tensor(out['index'])


# ── runs ──────────────────────────────────────────────────────────────────────

def save(ds, cfg, seed, res, pred, model_blob=None, suffix='.pkl'):
    d = OUT / ds / cfg
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f'seed{seed}_pred.npy', pred.astype(np.int8))
    if model_blob is not None and seed == 0:
        (d / f'model_seed0{suffix}').write_bytes(model_blob)
    (d / f'seed{seed}.json').write_text(json.dumps(res, indent=1))


def compact_bytes(t, n_classes, n_features):
    """size of a plain deployable encoding (documented in the paper):
    internal node = int16 feature + float32 threshold + 2 x uint16 child = 10 bytes;
    RF leaf = class probabilities as float16 (2 bytes x classes); GBDT leaf = float32 value (4 bytes);
    LR = float32 weights and intercepts."""
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    if isinstance(t, RandomForestClassifier):
        tot = 0
        for e in t.estimators_:
            leaf = e.tree_.children_left == -1
            tot += 10 * int((~leaf).sum()) + 2 * n_classes * int(leaf.sum())
        return tot
    if isinstance(t, HistGradientBoostingClassifier):
        tot = 0
        for it in t._predictors:
            for pr in it:
                leaf = pr.nodes['is_leaf'].astype(bool)
                tot += 10 * int((~leaf).sum()) + 4 * int(leaf.sum())
        return tot
    return 4 * (t.coef_.size + t.intercept_.size)


def run_tree(D, cfg, seed, cols):
    kind, params = TREE_CFGS[cfg]
    Xf, Xt = (D['Qfit'], D['Qte']) if kind == 'lr' else (D['Xfit'], D['Xte'])   # LR on the quantile features
    t = make_tree(kind, params, seed)
    t0 = time.time(); t.fit(Xf[:, cols], D['yfit']); fit_s = time.time() - t0
    Pc = t.predict_proba(Xt[:, cols])
    P = np.zeros((len(Pc), len(D['classes']))); P[:, t.classes_] = Pc
    pred = t.classes_[Pc.argmax(1)]
    blob = pickle.dumps(t, protocol=pickle.HIGHEST_PROTOCOL)
    res = dict(train_s=fit_s, size_bytes=len(blob), compact_bytes=compact_bytes(t, len(D['classes']), len(cols)),
               **P6.metrics(D, pred, P))
    if kind == 'rf':
        res['nodes'] = int(sum(e.tree_.node_count for e in t.estimators_))
    if kind == 'gb':
        res['n_iter'] = int(t.n_iter_)
    return res, pred, blob


def run_net(D, cfg, seed, cols, exp=None):
    import tensorflow as tf
    kind, w = NET_CFGS[cfg]
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    m = build_net(kind, w, len(cols), len(D['classes']))
    m.compile(tf.keras.optimizers.Adam(LR), 'sparse_categorical_crossentropy', jit_compile=JIT)
    bw = P6.BestWeights()
    cb = [tf.keras.callbacks.EarlyStopping(patience=PATIENCE, monitor='val_loss'), bw,
          tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5, monitor='val_loss')]
    keys = ('Xfit', 'Xva', 'Xte') if (exp or '').endswith('_minmax') else ('Qfit', 'Qva', 'Qte')   # _minmax: the v1 input scaling
    Xf, Xv, Xt = (D[k][:, cols][..., None] for k in keys)
    t0 = time.time()
    h = m.fit(Xf, D['yfit'], validation_data=(Xv, D['yva']), epochs=EPOCHS, batch_size=BATCH, callbacks=cb, verbose=0)
    res = dict(train_s=time.time() - t0, epochs=len(h.history['loss']), params=int(m.count_params()),
               best_epoch=int(np.argmin(h.history['val_loss'])) + 1, restored_epoch=bw.epoch)
    assert bw.epoch == res['best_epoch']
    P = m.predict(Xt, batch_size=4096, verbose=0)
    pred = P.argmax(1)
    res.update(P6.metrics(D, pred, P))
    # float32 model size (weights only) and int8 post-training quantisation
    blob = tflite_int8(m, Xf[np.random.default_rng(seed).choice(len(Xf), 500, replace=False)])
    Pq = tflite_predict(blob, Xt)
    q = P6.metrics(D, Pq.argmax(1), Pq)
    res['size_bytes'] = 4 * res['params']
    res['int8'] = dict(size_bytes=len(blob), macro_f1=q['macro_f1'], benign_fpr=q['benign_fpr'],
                       attack_detection=q['attack_detection'], macro_pr_auc=q.get('macro_pr_auc'),
                       agreement=float((Pq.argmax(1) == pred).mean()))
    if seed == 0:
        d = OUT / D['ds'] / (exp or cfg); d.mkdir(parents=True, exist_ok=True)   # exp: e.g. cnn_w1_noports must not overwrite cnn_w1
        m.save(d / 'model_seed0.h5')
    return res, pred, blob


def jobs(D, group):
    allc = np.arange(len(D['feat']))
    if group == 'trees':
        for seed in SEEDS:
            for cfg in TREE_CFGS:
                yield cfg, cfg, seed, allc
    elif group == 'nets':
        for seed in SEEDS:
            for cfg in NET_CFGS:
                yield cfg, cfg, seed, allc
    elif group == 'ports':        # TON-IoT only: the same models without the two port numbers
        keep = np.array([i for i, n in enumerate(D['feat']) if n not in PORTS])
        assert len(keep) == len(D['feat']) - 2
        for seed in SEEDS:
            for cfg in ('rf_t100_d16', 'rf_v1', 'gb_l31', 'gb05_l31', 'cnn_w1', 'mlp_w1'):   # gb05: lr 0.1 collapses on some seeds
                yield cfg, f'{cfg}_noports', seed, keep
    elif group == 'minmax':       # TON-IoT and UNSW-NB15: CNN width 1 on min-max inputs, as in the earlier version
        for seed in SEEDS:
            yield 'cnn_w1', 'cnn_w1_minmax', seed, allc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='all', choices=('all',) + DATASETS)
    ap.add_argument('--group', default='trees', choices=('trees', 'nets', 'ports', 'minmax'))
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    global OUT, SEEDS, EPOCHS, TREE_CFGS, NET_CFGS
    if a.smoke:
        OUT, SEEDS, EPOCHS = OUT.parent / 'results_v2_smoke', (0,), 2
        keep_t = ('rf_t100_d16', 'rf_v1', 'gb_l31') if a.group == 'ports' else ('rf_t25_d8', 'gb_l7', 'lr')
        keep_n = ('cnn_w1', 'mlp_w1') if a.group in ('ports', 'minmax') else ('cnn_w0.5', 'mlp_w0.5')
        TREE_CFGS = {k: TREE_CFGS[k] for k in keep_t}
        NET_CFGS = {k: NET_CFGS[k] for k in keep_n}
    if a.group != 'trees':
        import tensorflow as tf
        for g in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(g, True)
    for ds in DATASETS if a.dataset == 'all' else (a.dataset,):
        if (a.group == 'ports' and ds != 'TON_IoT') or (a.group == 'minmax' and ds == 'N_BaIoT'):
            continue
        D = P6.load(ds)
        print(f'== {ds}: {len(D["feat"])} features, fit {len(D["yfit"]):,}, test {len(D["yte"]):,}', flush=True)
        (OUT / ds).mkdir(parents=True, exist_ok=True)
        (OUT / ds / 'features.json').write_text(json.dumps(D['feat']))
        np.save(OUT / ds / 'clean_split0.npy', D['clean']); np.save(OUT / ds / 'yte.npy', D['yte'].astype(np.int8))
        for cfg, exp, seed, cols in jobs(D, a.group):
            if (OUT / ds / exp / f'seed{seed}.json').exists():
                continue
            t0 = time.time()
            is_net = cfg in NET_CFGS
            res, pred, blob = run_net(D, cfg, seed, cols, exp) if is_net else run_tree(D, cfg, seed, cols)
            res['cfg'] = dict(config=cfg, exp=exp, seed=seed, features=[D['feat'][i] for i in cols],
                              **({'kind': NET_CFGS[cfg][0], 'width': NET_CFGS[cfg][1], 'epochs_budget': EPOCHS, 'jit_compile': JIT} if is_net
                                 else {'kind': TREE_CFGS[cfg][0], **TREE_CFGS[cfg][1]}))
            save(ds, exp, seed, res, pred, blob, '.tflite' if is_net else '.pkl')
            extra = f" int8 {res['int8']['macro_f1']:.4f}" if is_net else ''
            print(f'  {exp} seed{seed}: macro-F1 {res["macro_f1"]:.4f}{extra} size {res["size_bytes"] / 1024:.0f} KB '
                  f'({time.time() - t0:.0f}s)', flush=True)
            gc.collect()
        del D; gc.collect()


if __name__ == '__main__':
    main()
