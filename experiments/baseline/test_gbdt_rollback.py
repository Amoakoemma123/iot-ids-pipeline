"""Synthetic check of gbdt_rollback.rollback_best_iter (no IDS data; a few seconds on one core).

Run from `Papers pipeline/`:  python -m experiments.baseline.test_gbdt_rollback
Checks, for a multi-class and a binary problem and several seeds:
  1. the rolled-back model keeps best_iter = argmax(validation_score_) iterations (n_iter_ == best_iter < n_iter_raw);
  2. its predict_proba equals staged_predict_proba of the unrolled model at that iteration;
  3. its predict_proba equals that of a model refitted with max_iter=best_iter on the same data and random_state
     (same internal 10% validation split), and the refitted model's validation scores equal the truncated ones;
  4. a pickle round trip preserves the predictions.
"""
import copy, pickle, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.baseline.gbdt_rollback import rollback_best_iter  # noqa: E402


def check(n_classes, seed):
    from sklearn.datasets import make_classification
    from sklearn.ensemble import HistGradientBoostingClassifier
    X, y = make_classification(n_samples=3000, n_features=12, n_informative=6, n_classes=n_classes,
                               n_clusters_per_class=1, flip_y=0.15, random_state=seed)
    Xte = np.random.default_rng(seed).normal(size=(500, 12))
    kw = dict(early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, learning_rate=0.3,
              max_leaf_nodes=31, random_state=seed)
    m = HistGradientBoostingClassifier(max_iter=500, **kw).fit(X, y)
    raw = copy.deepcopy(m)
    info = rollback_best_iter(m)
    k = info['best_iter']
    assert info['rolled_back'] and m.n_iter_ == k < raw.n_iter_, info
    assert k == int(np.argmax(raw.validation_score_)), (k, np.argmax(raw.validation_score_))
    assert raw.n_iter_ - k == 20, (raw.n_iter_, k)             # early stopping fired n_iter_no_change after the best
    P = m.predict_proba(Xte)
    staged = list(raw.staged_predict_proba(Xte))
    assert np.allclose(P, staged[k - 1], atol=1e-12), 'differs from staged_predict_proba at best_iter'
    ref = HistGradientBoostingClassifier(max_iter=k, **kw).fit(X, y)
    assert ref.n_iter_ == k
    assert np.allclose(ref.validation_score_, m.validation_score_, rtol=0, atol=1e-12), 'internal split differs'
    assert np.allclose(P, ref.predict_proba(Xte), atol=1e-12), 'differs from a refit with max_iter=best_iter'
    assert (m.predict(Xte) == ref.predict(Xte)).all()
    m2 = pickle.loads(pickle.dumps(m))
    assert np.array_equal(m2.predict_proba(Xte), P) and m2.n_iter_ == k
    return k, raw.n_iter_


def main():
    for n_classes in (4, 2):
        for seed in (0, 1, 2):
            k, n = check(n_classes, seed)
            print(f'classes {n_classes} seed {seed}: fitted {n} iterations, kept {k} -> OK')
    # no early stopping: unchanged
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=500, random_state=0)
    m = HistGradientBoostingClassifier(max_iter=30, early_stopping=False).fit(X, y)
    info = rollback_best_iter(m)
    assert not info['rolled_back'] and m.n_iter_ == 30
    print('early_stopping=False: unchanged -> OK')
    print('ALL ROLLBACK CHECKS PASSED')


if __name__ == '__main__':
    main()
