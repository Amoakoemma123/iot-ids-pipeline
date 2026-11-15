"""Paper 6: two-stream (feature-group) fusion for IoT intrusion detection on the corrected data.

Round 2 (results_v4) after the 2026-11-01 review. Changes against results_v3 (commit f5b08e9):
  * BUG FIX: Keras 2.10 EarlyStopping(restore_best_weights=True) restores the best weights only when
    early stopping fires; runs that reached the epoch cap were evaluated with last-epoch weights.
    BestWeights (below) always restores the weights of the epoch with the lowest validation loss.
  * Epoch budget 120 (patience 10) for every network. Runs that stopped early under the old 60-epoch
    budget would stop at the same epoch under 120, so only capped v3 runs are re-run where v3 files
    are reused (see reuse_v3); the main group is re-run in full so that predictions are saved.
  * Test predictions saved (seed<k>_pred.npy) for every new run; per split a 'clean' test mask marks
    test records whose selected-feature vector does not occur in the training split (the loader's
    cross-split guard runs before feature selection).
  * Tree references tuned on the validation split (group 'tune'): RF and histogram gradient boosting.
  * New groups: capacity (early_wide, early_head, late_all), resplit (5 further stratified splits of
    TON-IoT/UNSW-NB15 with the same selected features, scaling/SMOTE/quantile refitted per split),
    mlp (MLP encoders: column order irrelevant, semantic + random splits), colorder (early fusion on
    5 random column orders), attention ablation per class inside the attention model.
Streams: a fixed semantic split of each dataset's selected features (GROUPS) and random splits of the
same sizes (split r trained with seed r mod 5).
CNN/MLP inputs: quantile transform fitted on the training (SMOTE-augmented) rows; trees use min–max values.
Results: results_v4/<dataset>/<experiment>/seed<k>.json (resumable).

Run from `Papers pipeline/`:  python -m experiments.paper6_multimodal.run_p6_v2 --dataset D --group G
"""
import argparse, contextlib, gc, io, json, shutil, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'experiments' / 'paper6_multimodal' / 'results_v4'
V3 = ROOT / 'experiments' / 'paper6_multimodal' / 'results_v3'
BASE_RES = ROOT / 'experiments' / 'baseline' / 'results'
DATASETS = ('TON_IoT', 'UNSW_NB15', 'N_BaIoT')

# stream 1 = volume and timing of the flow (host-level statistics on N-BaIoT);
# stream 2 = protocol, service, state and TTL fields (channel/socket-level statistics on N-BaIoT)
GROUPS = {
    'TON_IoT': ['duration', 'src_bytes', 'dst_bytes', 'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes'],
    'UNSW_NB15': ['sbytes', 'dbytes', 'sload', 'dload', 'smean', 'ackdat'],
    'N_BaIoT': 'MI_dir_|H_L',          # prefix rule: MI_dir_* and H_L* are host-level
}
SEEDS = (0, 1, 2, 3, 4)
N_RANDOM = 10
RESPLITS = (1, 2, 3, 4, 5)
ARCHS = ('s1', 's2', 'early', 'late', 'attn')
TREES = ('rf', 'rf_s1', 'rf_s2', 'gb', 'gb_s1', 'gb_s2')
EPOCHS, PATIENCE, BATCH, LR = 120, 10, 512, 1e-3
OLD_CAP = 60               # v3 budget; v3 runs with epochs == OLD_CAP are re-run, others reused
DROPOUT, WMULT = 0.3, 1.0  # final review round 2 (M3): network grid knobs (run_p6_nettune.py); defaults = the paper's networks
WIDE = 1.6                 # early_wide: width factor matching late fusion's parameter count
RF_GRID = [dict(max_depth=d, min_samples_leaf=l) for d in (20, None) for l in (1, 2)]
GB_GRID = [dict(learning_rate=lr, max_leaf_nodes=n) for lr in (0.05, 0.1) for n in (31, 127)]


# ── data ──────────────────────────────────────────────────────────────────────

def _finish(D, ds, split):
    D['ds'], D['split'] = ds, split
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(n_quantiles=1000, subsample=200000, random_state=0).fit(D['Xfit'])
    for k in ('Xfit', 'Xva', 'Xte'):
        D['Q' + k[1:]] = qt.transform(D[k]).astype(np.float32)
    D['qt'] = qt
    D['normal'] = [c.lower() for c in D['classes']].index('normal')
    for k in ('ytr', 'yva', 'yte', 'yfit'):
        D[k] = np.asarray(D[k]).astype(int)
    # test records whose selected-feature vector also occurs in the (pre-SMOTE) training split
    tr = {r.tobytes() for r in np.ascontiguousarray(D['Xtr'])}
    D['clean'] = np.array([r.tobytes() not in tr for r in np.ascontiguousarray(D['Xte'])])
    return D


def load(ds, split=0, keep_raw=False):
    """split 0 = the corrected pipeline's split (seed 42); split k >= 1 = a new stratified 70/15/15 split
    (loader seed 42 + k) with the same selected features, min–max scaling and SMOTE refitted on it."""
    if ds == 'N_BaIoT':
        assert split == 0, 'no resplits for N-BaIoT'
        z = np.load(ROOT / 'dataset4' / 'nbaiot_prepared.npz')
        info = json.loads((ROOT / 'dataset4' / 'nbaiot_prepared.json').read_text())
        D = dict(feat=info['features'], classes=info['classes'], Xtr=z['Xtr'], ytr=z['ytr'],
                 Xva=z['Xva'], yva=z['yva'], Xte=z['Xte'], yte=z['yte'])
        D['Xfit'], D['yfit'] = D['Xtr'], D['ytr']          # near-balanced: no SMOTE
        return _finish(D, ds, split)
    import joblib
    from experiments.baseline.config import DATASETS as BASE
    from src.data.loader import load_dataset
    pipe = joblib.load(BASE_RES / ds / 'pipeline.pkl')
    csv, label = BASE[ds]
    with contextlib.redirect_stdout(io.StringIO()):
        raw = load_dataset(csv, label_col=label, seed=42 + split)
        if split == 0:
            f = lambda X: pipe.transform(X)[0].astype(np.float32)
            Xsm, ysm = pipe.transform(raw['X_train'], apply_smote=True, y=raw['y_train'])
        else:
            from sklearn.preprocessing import MinMaxScaler
            names = list(raw['feature_names'])
            miss = set(pipe.selected_features_) - set(names)
            assert not miss, f'selected features missing in split {split}: {miss}'
            cols = [names.index(n) for n in pipe.selected_features_]
            sc = MinMaxScaler().fit(raw['X_train'][:, cols])
            f = lambda X: sc.transform(X[:, cols]).astype(np.float32)
            p = pipe.__class__(seed=split, smote_k=pipe.smote_k)
            Xsm, ysm = p._apply_smote(f(raw['X_train']), np.asarray(raw['y_train']))
    if keep_raw:      # raw (unscaled, all columns) split arrays, for problem-space attacks (Paper 2)
        D_raw = dict(names=list(raw['feature_names']), Rtr=raw['X_train'], Rva=raw['X_val'], Rte=raw['X_test'],
                     scaler=pipe.scaler_ if split == 0 else sc)
    D = dict(feat=list(pipe.selected_features_), classes=list(raw['class_names']),
             Xtr=f(raw['X_train']), ytr=raw['y_train'], Xva=f(raw['X_val']), yva=raw['y_val'],
             Xte=f(raw['X_test']), yte=raw['y_test'], Xfit=np.asarray(Xsm, np.float32), yfit=np.asarray(ysm, int))
    if keep_raw:
        D['raw'] = D_raw
    return _finish(D, ds, split)


def semantic_split(D):
    g = GROUPS[D['ds']]
    if isinstance(g, str):
        import re
        s1 = [i for i, n in enumerate(D['feat']) if re.match(g, n)]
    else:
        missing = set(g) - set(D['feat'])
        assert not missing, f'stream-1 features not selected: {missing}'
        s1 = [D['feat'].index(n) for n in g]
    s2 = [i for i in range(len(D['feat'])) if i not in s1]
    return np.array(s1), np.array(s2)


def random_split(D, r):
    s1, s2 = semantic_split(D)
    perm = np.random.default_rng(1000 + r).permutation(len(D['feat']))
    return np.sort(perm[:len(s1)]), np.sort(perm[len(s1):])


def column_order(D, r):
    """random order of all features for early fusion (the 1D-CNN depends on column order)"""
    return np.random.default_rng(2000 + r).permutation(len(D['feat']))


# ── models ────────────────────────────────────────────────────────────────────

def _encoder(inp, p, w=1.0, kind='cnn'):
    from tensorflow.keras import layers as L
    if kind == 'mlp':
        x = L.Flatten(name=f'{p}_flat')(inp)
        x = L.Dense(round(128 * w), activation='relu', name=f'{p}_d1')(x)
        x = L.BatchNormalization(name=f'{p}_bn1')(x)
        x = L.Dense(round(128 * w), activation='relu', name=f'{p}_d2')(x)
        x = L.BatchNormalization(name=f'{p}_bn2')(x)
        return L.Dense(round(64 * w), activation='relu', name=f'{p}_emb')(x)
    x = L.Conv1D(round(64 * w), 3, activation='relu', padding='same', name=f'{p}_c1')(inp)
    x = L.BatchNormalization(name=f'{p}_bn1')(x)
    if inp.shape[1] >= 2:
        x = L.MaxPooling1D(2, padding='same', name=f'{p}_pool')(x)
    x = L.Conv1D(round(128 * w), 3, activation='relu', padding='same', name=f'{p}_c2')(x)
    x = L.BatchNormalization(name=f'{p}_bn2')(x)
    x = L.GlobalAveragePooling1D(name=f'{p}_gap')(x)
    return L.Dense(round(64 * w), activation='relu', name=f'{p}_emb')(x)


def base_arch(arch):
    """'m_late' -> ('late', 'mlp');  'early_perm3' -> ('early', 'cnn')"""
    kind = 'mlp' if arch.startswith('m_') else 'cnn'
    a = arch[2:] if kind == 'mlp' else arch
    return a.split('_perm')[0], kind


def build(arch, n1, n2, n_classes):
    from tensorflow.keras import layers as L, Model
    a, kind = base_arch(arch)
    if a in ('s1', 's2', 'early', 'early_wide', 'early_head'):
        n = {'s1': n1, 's2': n2}.get(a, n1 + n2)
        inp = L.Input((n, 1))
        e = _encoder(inp, a, (WIDE if a == 'early_wide' else 1.0) * WMULT, kind)
        if a == 'early_head':      # same 128-unit head as late fusion
            e = L.Dense(128, activation='relu')(e)
        return Model(inp, L.Dense(n_classes, activation='softmax')(L.Dropout(DROPOUT)(e)))
    i1, i2 = L.Input((n1, 1)), L.Input((n2, 1))
    e1, e2 = _encoder(i1, 'e1', WMULT, kind=kind), _encoder(i2, 'e2', WMULT, kind=kind)
    if a in ('late', 'late_all'):
        x = L.Dropout(DROPOUT)(L.Dense(128, activation='relu')(L.Concatenate()([e1, e2])))
        return Model([i1, i2], L.Dense(n_classes, activation='softmax')(x))
    att = L.Dense(2, activation='softmax', name='modality_attention')(L.Concatenate()([e1, e2]))
    fused = L.Add()([L.Multiply()([e1, L.Lambda(lambda t: t[:, :1])(att)]),
                     L.Multiply()([e2, L.Lambda(lambda t: t[:, 1:])(att)])])
    x = L.Dropout(DROPOUT)(L.Dense(64, activation='relu', name='head_d')(fused))
    return Model([i1, i2], L.Dense(n_classes, activation='softmax', name='head_out')(x))


def inputs(arch, X, s1, s2, D=None):
    X = X[..., None]
    a, _ = base_arch(arch)
    if a in ('s1', 's2'):
        return X[:, s1] if a == 's1' else X[:, s2]
    if a in ('early', 'early_wide', 'early_head'):
        return X[:, s1] if '_perm' in arch else X[:, np.r_[s1, s2]]   # perm: s1 holds the full order
    if a == 'late_all':
        return [X, X]
    return [X[:, s1], X[:, s2]]


def metrics(D, pred, proba=None):
    from sklearn.metrics import f1_score, accuracy_score, average_precision_score
    y = D['yte']
    att = y != D['normal']
    out = dict(macro_f1=float(f1_score(y, pred, average='macro')), accuracy=float(accuracy_score(y, pred)),
               per_class_f1=dict(zip(D['classes'], map(float, f1_score(y, pred, average=None, labels=range(len(D['classes'])))))),
               attack_detection=float((pred[att] != D['normal']).mean()),
               benign_fpr=float((pred[~att] != D['normal']).mean()))
    c = D['clean']
    out['clean_macro_f1'] = float(f1_score(y[c], pred[c], average='macro', labels=range(len(D['classes']))))
    out['clean_n'] = int(c.sum())
    if proba is not None:     # threshold-free: macro average precision (PR-AUC), one-vs-rest
        K = len(D['classes'])
        Y = np.eye(K)[y]
        ap = [average_precision_score(Y[:, k], proba[:, k]) for k in range(K) if Y[:, k].any()]
        out['macro_pr_auc'] = float(np.mean(ap))
    return out


class BestWeights:
    """Always restore the weights of the epoch with the lowest validation loss (Keras 2.10's
    EarlyStopping(restore_best_weights=True) does so only when early stopping fires)."""

    def __new__(cls):
        import tensorflow as tf

        class _CB(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.best, self.w, self.epoch = np.inf, None, 0

            def on_epoch_end(self, epoch, logs=None):
                v = logs.get('val_loss')
                if v is not None and v < self.best:
                    self.best, self.w, self.epoch = v, self.model.get_weights(), epoch + 1

            def on_train_end(self, logs=None):
                if self.w is not None:
                    self.model.set_weights(self.w)
        return _CB()


def attention_analysis(m, D, Xin, out):
    """per-class stream weights, norm-weighted contributions and in-model ablation of each stream"""
    import tensorflow as tf
    from sklearn.metrics import f1_score
    sub = tf.keras.Model(m.inputs, [m.get_layer('e1_emb').output, m.get_layer('e2_emb').output,
                                    m.get_layer('modality_attention').output])
    E1, E2, A = sub.predict(Xin, batch_size=4096, verbose=0)
    head = lambda F: m.get_layer('head_out')(m.get_layer('head_d')(F)).numpy().argmax(1)
    y, cls = D['yte'], D['classes']
    a = A[:, 0]
    n1 = np.linalg.norm(A[:, :1] * E1, axis=1); n2 = np.linalg.norm(A[:, 1:] * E2, axis=1)
    share = n1 / (n1 + n2 + 1e-12)
    out['attn_s1_mean'] = float(a.mean())
    out['attn_s1_per_class'] = {c: float(a[y == i].mean()) for i, c in enumerate(cls) if (y == i).any()}
    out['attn_s1_sd_within'] = float(np.mean([a[y == i].std() for i in range(len(cls)) if (y == i).any()]))
    out['contrib_s1_per_class'] = {c: float(share[y == i].mean()) for i, c in enumerate(cls) if (y == i).any()}
    base = head(A[:, :1] * E1 + A[:, 1:] * E2)
    # ablation: replace one stream's embedding by its mean over the test set (attention weights kept)
    abl = {}
    for k, (Ea, Eb, ia, ib) in {'s1': (E1, E2, 0, 1), 's2': (E2, E1, 1, 0)}.items():
        Em = np.broadcast_to(Ea.mean(0), Ea.shape)
        F = A[:, ia:ia + 1] * Em + A[:, ib:ib + 1] * Eb
        p = head(F)
        abl[k] = dict(macro_f1=float(f1_score(y, p, average='macro')),
                      per_class_f1=dict(zip(cls, map(float, f1_score(y, p, average=None, labels=range(len(cls)))))))
    out['attn_ablation'] = abl
    out['attn_base_check'] = float((base == out.pop('_pred')).mean())


def run_cnn(D, arch, s1, s2, seed):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    nc = len(D['classes'])
    a, _ = base_arch(arch)
    n1, n2 = (len(s1), len(s2)) if a != 'late_all' else (len(s1) + len(s2),) * 2
    if '_perm' in arch:
        n1 = len(s1)
    m = build(arch, n1, n2, nc)
    m.compile(tf.keras.optimizers.Adam(LR), 'sparse_categorical_crossentropy')
    bw = BestWeights()
    cb = [tf.keras.callbacks.EarlyStopping(patience=PATIENCE, monitor='val_loss'), bw,
          tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-5, monitor='val_loss')]
    t0 = time.time()
    h = m.fit(inputs(arch, D['Qfit'], s1, s2), D['yfit'], validation_data=(inputs(arch, D['Qva'], s1, s2), D['yva']),
              epochs=EPOCHS, batch_size=BATCH, callbacks=cb, verbose=0)
    out = dict(train_s=time.time() - t0, epochs=len(h.history['loss']), params=int(m.count_params()),
               best_epoch=int(np.argmin(h.history['val_loss'])) + 1, restored_epoch=bw.epoch,
               val_loss_best=float(np.min(h.history['val_loss'])))
    assert bw.epoch == out['best_epoch'], 'restored epoch differs from best validation epoch'
    from sklearn.metrics import f1_score as _f1     # validation macro-F1 of the restored model (network grid, M3)
    out['val_macro_f1'] = float(_f1(D['yva'], m.predict(inputs(arch, D['Qva'], s1, s2), batch_size=4096, verbose=0).argmax(1), average='macro'))
    Xin = inputs(arch, D['Qte'], s1, s2)
    P = m.predict(Xin, batch_size=4096, verbose=0)
    pred = P.argmax(1)
    out.update(metrics(D, pred, P))
    if a == 'attn':
        out['_pred'] = pred
        attention_analysis(m, D, Xin, out)
    return out, pred


def make_tree(kind, params, seed):
    if kind == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed, **params)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=1000, early_stopping=True, validation_fraction=0.1,
                                          n_iter_no_change=20, random_state=seed, **params)


def tuned(ds):
    p = OUT / ds / 'tune_trees.json'
    T = json.loads(p.read_text()) if p.exists() else None
    assert T, f'run --group tune for {ds} first'
    return T


def run_tree(D, arch, s1, s2, seed):
    kind = arch.split('_')[0]
    cols = {'': np.r_[s1, s2], 's1': s1, 's2': s2}[arch[len(kind) + 1:] if '_' in arch else '']
    t = make_tree(kind, tuned(D['ds'])[kind]['best'], seed)
    t0 = time.time(); t.fit(D['Xfit'][:, cols], D['yfit'])
    out = dict(train_s=time.time() - t0)
    Pc = t.predict_proba(D['Xte'][:, cols])
    P = np.zeros((len(Pc), len(D['classes'])))
    P[:, t.classes_] = Pc
    pred = t.classes_[Pc.argmax(1)]
    out.update(metrics(D, pred, P))
    if kind == 'gb':
        out['n_iter'] = int(t.n_iter_)
    return out, pred


def tune_trees(D):
    """choose RF and GBDT hyperparameters on validation macro-F1 (all features, semantic split, seed 0)"""
    from sklearn.metrics import f1_score
    res = {}
    for kind, grid in (('rf', RF_GRID), ('gb', GB_GRID)):
        scores = []
        for g in grid:
            t = make_tree(kind, g, 0).fit(D['Xfit'], D['yfit'])
            scores.append(float(f1_score(D['yva'], t.predict(D['Xva']), average='macro')))
            print(f'    tune {kind} {g}: val macro-F1 {scores[-1]:.4f}', flush=True)
        res[kind] = dict(grid=grid, val_macro_f1=scores, best=grid[int(np.argmax(scores))])
    return res


# ── driver ────────────────────────────────────────────────────────────────────

def jobs(D, group):
    s1, s2 = semantic_split(D)
    if group == 'main':
        for seed in SEEDS:
            for arch in TREES + ARCHS:
                yield arch, arch, s1, s2, seed
    elif group == 'capacity':
        for seed in SEEDS:
            for arch in ('early_wide', 'early_head', 'late_all'):
                yield arch, arch, s1, s2, seed
    elif group == 'random':
        for r in range(N_RANDOM):
            r1, r2 = random_split(D, r)
            for arch in ('s1', 's2', 'late', 'attn', 'rf_s1', 'rf_s2', 'gb_s1', 'gb_s2'):
                yield arch, f'{arch}_rand{r}', r1, r2, r % 5
    elif group == 'mlp':
        for seed in SEEDS:
            for arch in ARCHS:
                yield f'm_{arch}', f'm_{arch}', s1, s2, seed
        for r in range(N_RANDOM):
            r1, r2 = random_split(D, r)
            for arch in ('s1', 's2', 'late', 'attn'):
                yield f'm_{arch}', f'm_{arch}_rand{r}', r1, r2, r % 5
    elif group == 'colorder':
        for r in range(5):
            yield f'early_perm{r}', f'early_perm{r}', column_order(D, r), np.array([], int), r
    elif group == 'resplit':      # D is already the resplit; one seed per split (= split index)
        k = D['split']
        for arch in ('rf', 'gb', 'rf_s1', 'rf_s2', 's1', 's2', 'early', 'late', 'attn'):
            yield arch, f'{arch}_split{k}', s1, s2, k


def reuse_v3(ds, exp, seed):
    """copy a v3 result when it is valid under the fixed protocol: a CNN run that stopped early before the
    old 60-epoch cap (identical stopping under a 120 budget; best weights were restored). Trees are never
    reused (now tuned). Returns True if copied."""
    random_cnn = '_rand' in exp and exp.split('_rand')[0] in ('s1', 's2', 'late', 'attn')
    nbaiot_main = ds == 'N_BaIoT' and exp in ('s1', 's2', 'early', 'late')   # saturated; attn re-run for the ablation
    if not (random_cnn or nbaiot_main):
        return False
    src = V3 / ds / exp / f'seed{seed}.json'
    if not src.exists():
        return False
    r = json.loads(src.read_text())
    if r['epochs'] >= OLD_CAP:
        return False
    r['reused_from'] = 'results_v3 (stopped early at epoch %d < 60; no saved predictions)' % r['epochs']
    dst = OUT / ds / exp / f'seed{seed}.json'
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(r, indent=1))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='all', choices=('all',) + DATASETS)
    ap.add_argument('--group', default='main', choices=('tune', 'main', 'capacity', 'random', 'mlp', 'colorder', 'resplit'))
    ap.add_argument('--no-reuse', action='store_true', help='final review round 2: never copy results_v3 runs; '
                    're-train them (the reused files are moved to results_v4/_superseded_reused first)')
    ap.add_argument('--smoke', action='store_true', help='2 epochs, seed 0 / split 0 only, results_v4_smoke/')
    a = ap.parse_args()
    global OUT, EPOCHS, SEEDS, N_RANDOM, RESPLITS
    if a.smoke:
        sm = OUT.parent / 'results_v4_smoke'
        if (OUT / a.dataset / 'tune_trees.json').exists():
            (sm / a.dataset).mkdir(parents=True, exist_ok=True)
            shutil.copy(OUT / a.dataset / 'tune_trees.json', sm / a.dataset / 'tune_trees.json')
        OUT, EPOCHS, SEEDS, N_RANDOM, RESPLITS = sm, 2, (0,), 1, (1,)
    import tensorflow as tf
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    for ds in DATASETS if a.dataset == 'all' else (a.dataset,):
        splits = RESPLITS if a.group == 'resplit' else (0,)
        if a.group == 'resplit' and ds == 'N_BaIoT':
            continue
        for split in splits:
            D = load(ds, split)
            s1, s2 = semantic_split(D)
            print(f'== {ds} split {split}: {len(D["feat"])} features, stream 1 {len(s1)}, stream 2 {len(s2)}, '
                  f'fit {len(D["yfit"]):,}, test {len(D["yte"]):,} (clean {int(D["clean"].sum()):,})', flush=True)
            (OUT / ds).mkdir(parents=True, exist_ok=True)
            np.save(OUT / ds / f'clean_split{split}.npy', D['clean'])
            np.save(OUT / ds / f'yte_split{split}.npy', D['yte'].astype(np.int8))
            if split == 0:
                (OUT / ds / 'streams.json').write_text(json.dumps(dict(
                    s1=[D['feat'][i] for i in s1], s2=[D['feat'][i] for i in s2],
                    random={r: [D['feat'][i] for i in random_split(D, r)[0]] for r in range(N_RANDOM)},
                    colorder={r: [D['feat'][i] for i in column_order(D, r)] for r in range(5)}), indent=1))
            if a.group == 'tune':
                if not (OUT / ds / 'tune_trees.json').exists():
                    (OUT / ds / 'tune_trees.json').write_text(json.dumps(tune_trees(D), indent=1))
                continue
            for arch, exp, i1, i2, seed in jobs(D, a.group):
                path = OUT / ds / exp / f'seed{seed}.json'
                if path.exists() or (not a.smoke and not a.no_reuse and reuse_v3(ds, exp, seed)):
                    continue
                t0 = time.time()
                res, pred = (run_tree if arch.split('_')[0] in ('rf', 'gb') else run_cnn)(D, arch, i1, i2, seed)
                res.update(cfg=dict(arch=arch, exp=exp, seed=seed, split=split, epochs_budget=EPOCHS,
                                    s1=[D['feat'][i] for i in i1]))
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path.with_name(f'seed{seed}_pred.npy'), pred.astype(np.int8))
                path.write_text(json.dumps(res, indent=1))
                print(f'  {exp} seed{seed}: macro-F1 {res["macro_f1"]:.4f} clean {res["clean_macro_f1"]:.4f} '
                      f'({time.time() - t0:.0f}s, epochs {res.get("epochs", "-")})', flush=True)
                gc.collect()
            del D; gc.collect()


if __name__ == '__main__':
    main()
