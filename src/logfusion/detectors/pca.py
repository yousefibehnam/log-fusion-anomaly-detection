"""PCA subspace anomaly detection with the Q-statistic threshold.

Event-count vectors are TF-IDF weighted and centred, then split into a "normal"
subspace spanned by the leading principal components (those explaining ``variance``
of the total variance) and a residual subspace. A sample's squared prediction error
(SPE), its squared distance from the normal subspace, is compared with the
Jackson-Mudholkar (1979) approximation of the SPE's upper ``1 - alpha`` quantile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np


@dataclass
class PCADetector:
    variance: float = 0.95
    alpha: float = 0.001
    tfidf: bool = True

    mean_: np.ndarray = field(init=False, repr=False)
    idf_: np.ndarray = field(init=False, repr=False)
    components_: np.ndarray = field(init=False, repr=False)
    threshold_: float = field(init=False)
    n_components_: int = field(init=False)

    def fit(self, x: np.ndarray) -> PCADetector:
        x = np.asarray(x, dtype=float)
        n = x.shape[0]
        if n < 2:
            raise ValueError("need at least two samples to fit")
        if self.tfidf:
            df = (x > 0).sum(axis=0)
            self.idf_ = np.where(df > 0, np.log(n / np.maximum(df, 1)), 0.0)
        else:
            self.idf_ = np.ones(x.shape[1])
        weighted = x * self.idf_
        self.mean_ = weighted.mean(axis=0)
        centred = weighted - self.mean_

        _, s, vt = np.linalg.svd(centred, full_matrices=False)
        eigenvalues = s**2 / (n - 1)
        total = eigenvalues.sum()
        if total == 0:
            self.n_components_ = 0
            self.components_ = np.zeros((0, x.shape[1]))
            self.threshold_ = 0.0
            return self
        explained = np.cumsum(eigenvalues) / total
        k = int(np.searchsorted(explained, self.variance) + 1)
        k = min(k, len(eigenvalues))
        self.n_components_ = k
        self.components_ = vt[:k]
        self.threshold_ = q_statistic(eigenvalues[k:], self.alpha)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        """Squared prediction error of each row."""
        centred = np.asarray(x, dtype=float) * self.idf_ - self.mean_
        projected = centred @ self.components_.T @ self.components_
        residual = centred - projected
        return np.einsum("ij,ij->i", residual, residual)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.score(x) > self.threshold_).astype(int)


def q_statistic(residual_eigenvalues: np.ndarray, alpha: float) -> float:
    """Jackson-Mudholkar upper control limit for the squared prediction error."""
    lam = np.asarray(residual_eigenvalues, dtype=float)
    theta1, theta2, theta3 = lam.sum(), (lam**2).sum(), (lam**3).sum()
    if theta1 <= 0 or theta2 <= 0:
        return 0.0
    h0 = 1.0 - 2.0 * theta1 * theta3 / (3.0 * theta2**2)
    c_alpha = NormalDist().inv_cdf(1.0 - alpha)
    term = (
        c_alpha * np.sqrt(2.0 * theta2 * h0**2) / theta1
        + 1.0
        + theta2 * h0 * (h0 - 1.0) / theta1**2
    )
    return float(theta1 * term ** (1.0 / h0))
