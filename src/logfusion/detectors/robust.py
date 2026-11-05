"""Robust (median/MAD) z-scores per feature.

Each feature is scaled by its median and median absolute deviation on training data.
This is the modified z-score of Iglewicz and Hoaglin (1993), for which 3.5 is the
customary outlier cut-off. A feature whose training MAD is zero would make every change
infinitely surprising, so the scale has a floor, ``min_scale``, in feature units.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAD_TO_SIGMA = 1.4826  # 1 / Phi^-1(0.75): makes MAD consistent with sigma for normal data


@dataclass
class RobustZ:
    threshold: float = 3.5
    min_scale: float = 1.0

    median_: np.ndarray = field(init=False, repr=False)
    scale_: np.ndarray = field(init=False, repr=False)

    def fit(self, x: np.ndarray) -> RobustZ:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("expected a non-empty 2-D array")
        self.median_ = np.median(x, axis=0)
        mad = np.median(np.abs(x - self.median_), axis=0)
        self.scale_ = np.maximum(MAD_TO_SIGMA * mad, self.min_scale)
        return self

    def z(self, x: np.ndarray) -> np.ndarray:
        """Absolute robust z-score of every value."""
        return np.abs(np.asarray(x, dtype=float) - self.median_) / self.scale_

    def score(self, x: np.ndarray) -> np.ndarray:
        """The largest z-score in each row."""
        return self.z(x).max(axis=1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.score(x) > self.threshold).astype(int)
