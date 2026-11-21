"""A small SMOTE (Synthetic Minority Over-sampling Technique, Chawla et al. 2002) that works inside a
scikit-learn Pipeline, so oversampling is only applied to the training folds, never to test data."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neighbors import NearestNeighbors


def smote(X, y, k: int = 5, ratio: float = 1.0, random_state: int = 0):
    """Oversample every minority class up to `ratio` × the size of the majority class."""
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    target = int(counts.max() * ratio)
    Xs, ys = [X], [y]
    for c, n in zip(classes, counts):
        if n >= target or n < 2:
            continue
        Xc = X[y == c]
        nn = NearestNeighbors(n_neighbors=min(k, n - 1) + 1).fit(Xc)
        neigh = nn.kneighbors(Xc, return_distance=False)[:, 1:]
        m = target - n
        base = rng.integers(0, n, m)
        pick = neigh[base, rng.integers(0, neigh.shape[1], m)]
        gap = rng.random((m, 1))
        Xs.append(Xc[base] + gap * (Xc[pick] - Xc[base]))
        ys.append(np.full(m, c))
    return np.vstack(Xs), np.concatenate(ys)


class SMOTEClassifier(BaseEstimator, ClassifierMixin):
    """Wraps a classifier: fit() oversamples with SMOTE first, predict() is unchanged."""

    def __init__(self, estimator=None, k=5, ratio=0.5, random_state=0):
        self.estimator = estimator
        self.k = k
        self.ratio = ratio
        self.random_state = random_state

    def fit(self, X, y):
        Xr, yr = smote(X, y, self.k, self.ratio, self.random_state)
        self.estimator_ = clone(self.estimator).fit(Xr, yr)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)
