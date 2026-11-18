"""Cheap evaluation checks for machine-learning intrusion detection (Paper 7, Section 6).

Pipeline-independent: each function takes plain arrays, lists or dicts. Requires numpy and scikit-learn
(TensorFlow only for RestoreBest). Run `python ids_checks.py` for a self-test on synthetic data.

    identifier_columns(columns)                       -> names that look like identifiers / testbed fields   (F1, F3)
    exact_copies(X_train, X_test, y_train, y_test)    -> share of test rows seen in training (exact match)  (F2)
    tree_reference(X_train, y_train, X_test, y_test)  -> macro-F1 of a random forest on the same features    (F4, F16)
    majority_f1(y_train, y_test)                      -> macro-F1 of always predicting the majority class    (F4)
    flag_network(net_f1, tree_f1, majority, best_epoch, ...) -> list of warnings for a network result      (F4)
    cap_share(epochs_trained, cap)                    -> share of runs that reached the epoch cap            (F4)
    grid_edge(grid, best)                             -> True if the selected value is at an edge of its grid (F9)
    rerun_spread(scores)                              -> max - min over repeated identical runs              (F8)
    false_alarms_per_day(fpr, benign_per_day)         -> expected false alerts at a stated benign volume     (F15)
    RestoreBest                                       -> Keras callback: always restores best weights and records the epoch (F5)
"""
import re

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

ID_PATTERNS = (r'^id$', r'^index$', r'^unnamed', r'(^|_)(src|dst|source|dest)_?ip', r'^srcip$', r'^dstip$',
               r'(^|_)ts$', r'time', r'^stime$', r'^ltime$', r'uid', r'flow_?id')
TESTBED_PATTERNS = (r'ttl', r'(^|_)(src|dst|s|d)_?port', r'^sport$', r'^dsport$')


def identifier_columns(columns):
    """Columns whose names suggest identifiers (drop them) or testbed-set fields (ablate them).
    A name heuristic: patterns such as 'time' over-match (e.g. duration-like fields), so treat the output as a list
    to review, not as a verdict."""
    cols = [str(c) for c in columns]
    ident = [c for c in cols if any(re.search(p, c.lower()) for p in ID_PATTERNS)]
    testbed = [c for c in cols if c not in ident and any(re.search(p, c.lower()) for p in TESTBED_PATTERNS)]
    return {'identifiers': ident, 'testbed': testbed}


def exact_copies(X_train, X_test, y_train=None, y_test=None):
    """Share of test rows whose feature vector (and label, if given) occurs exactly in training.
    Uses the bytes of float32 rows: no rounding (rounding inflated a copy rate from 0.1% to 17% in our series)."""
    Xtr = np.ascontiguousarray(np.asarray(X_train, dtype=np.float32))
    Xte = np.ascontiguousarray(np.asarray(X_test, dtype=np.float32))
    seen = {r.tobytes() for r in Xtr}
    feat = np.array([r.tobytes() in seen for r in Xte])
    out = {'features_only': float(feat.mean())}
    if y_train is not None and y_test is not None:
        seen_l = {r.tobytes() + str(l).encode() for r, l in zip(Xtr, y_train)}
        out['with_label'] = float(np.mean([r.tobytes() + str(l).encode() in seen_l for r, l in zip(Xte, y_test)]))
    out['clean_mask'] = ~feat
    return out


def tree_reference(X_train, y_train, X_test, y_test, n_estimators=200, seed=0, n_jobs=-1):
    """Macro-F1 of an UNTUNED random forest trained on exactly the features a network sees.
    An untuned forest is enough to expose a failed network (Paper 7, F4); tune it (and consider boosted trees)
    before reporting it as a comparator (F16)."""
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs)
    rf.fit(X_train, y_train)
    return float(f1_score(y_test, rf.predict(X_test), average='macro'))


def majority_f1(y_train, y_test):
    vals, counts = np.unique(y_train, return_counts=True)
    pred = np.full(len(y_test), vals[np.argmax(counts)])
    return float(f1_score(y_test, pred, average='macro'))


def flag_network(net_f1, tree_f1, majority=None, best_epoch=None, max_epochs=None, gap=0.05, early=4):
    """Warnings for a network result. A large gap to the tree reference was the only rule that caught all
    three silent/graded failures in our series; the early-peak rule caught only the most severe one.
    gap=0.05: the smallest failure in our series left a gap of 0.076 (min-max CNN vs forest), while correctly
    trained networks were within about 0.01-0.04 of the forest; adjust to the dataset's seed-to-seed spread."""
    w = []
    if tree_f1 - net_f1 > gap:
        w.append(f'network {net_f1:.3f} is {tree_f1 - net_f1:.3f} below a tree on the same features ({tree_f1:.3f})')
    if majority is not None and net_f1 <= majority + 0.02:
        w.append(f'network {net_f1:.3f} is at or below the majority-class predictor ({majority:.3f})')
    if best_epoch is not None and best_epoch <= early:
        w.append(f'best epoch {best_epoch} is within the first {early} epochs')
    if best_epoch is not None and max_epochs is not None and best_epoch >= max_epochs:
        w.append(f'best epoch {best_epoch} is the last allowed epoch ({max_epochs}): budget may be too small')
    return w


def cap_share(epochs_trained, cap):
    e = np.asarray(epochs_trained)
    return float(np.mean(e >= cap))


def grid_edge(grid, best):
    g = sorted(grid)
    return best in (g[0], g[-1])


def rerun_spread(scores):
    s = np.asarray(scores, dtype=float)
    return float(s.max() - s.min())


def false_alarms_per_day(fpr, benign_per_day=1_000_000):
    return fpr * benign_per_day


try:  # optional Keras part
    import tensorflow as tf

    class RestoreBest(tf.keras.callbacks.Callback):
        """Keep the weights of the best epoch and ALWAYS restore them at the end of training.
        (tf.keras 2.10-2.15 EarlyStopping(restore_best_weights=True) restores only when early stopping fires.)
        After fit(), `restored_epoch` holds the 1-based epoch whose weights the model has; store it with the result."""

        def __init__(self, monitor='val_loss', mode='min'):
            super().__init__()
            self.monitor, self.sign = monitor, (1 if mode == 'min' else -1)
            self.best, self.best_w, self.restored_epoch = np.inf, None, None

        def on_epoch_end(self, epoch, logs=None):
            v = (logs or {}).get(self.monitor)
            if v is not None and self.sign * v < self.best:
                self.best, self.best_w, self.restored_epoch = self.sign * v, self.model.get_weights(), epoch + 1

        def on_train_end(self, logs=None):
            if self.best_w is not None:
                self.model.set_weights(self.best_w)
except ImportError:  # pragma: no cover
    RestoreBest = None


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 6)); y = (X[:, 0] + X[:, 1] > 0).astype(int)
    Xtr, Xte, ytr, yte = X[:1500], np.vstack([X[1500:], X[:50]]), y[:1500], np.concatenate([y[1500:], y[:50]])
    print(identifier_columns(['id', 'dur', 'sbytes', 'sttl', 'srcip', 'dsport', 'ts']))
    c = exact_copies(Xtr, Xte, ytr, yte); print('copies', round(c['features_only'], 3), round(c['with_label'], 3))
    t = tree_reference(Xtr, ytr, Xte, yte, n_estimators=50, n_jobs=1); m = majority_f1(ytr, yte)
    print('tree', round(t, 3), 'majority', round(m, 3))
    print(flag_network(0.30, t, m, best_epoch=2, max_epochs=60))
    print('cap share', cap_share([60, 41, 60, 12], 60), 'edge', grid_edge([1, 10, 100], 1),
          'spread', rerun_spread([0.72, 0.61, 0.70]), 'FP/day', false_alarms_per_day(0.015))
