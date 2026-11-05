"""Window-level anomaly detection from one or several sources.

For every source and every service, a robust z-score model is fitted on fault-free
training windows; a service's raw score in a window is its largest feature z-score. Each
source is then calibrated on separate fault-free windows: its threshold is the highest
window score seen there, so on the calibration windows no source raises an alarm.
Dividing by that threshold puts all sources on one scale (1.0 = alarm level), and the
fused score of a service is the largest normalised score over the selected sources.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from logfusion.detectors.robust import RobustZ
from logfusion.fusion.sources import SourceFeatures


@dataclass
class FusionDetector:
    min_scale: float = 1.0
    raw_: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    thresholds_: dict[str, float] = field(default_factory=dict, init=False)

    def fit(
        self, sources: Sequence[SourceFeatures], train: slice, calibrate: slice
    ) -> FusionDetector:
        for source in sources:
            values = source.values
            raw = np.zeros(values.shape[:2])
            for s in range(values.shape[1]):
                model = RobustZ(min_scale=self.min_scale).fit(values[train, s, :])
                raw[:, s] = model.z(values[:, s, :]).max(axis=1) if values.shape[2] else 0.0
            self.raw_[source.name] = raw
            calib_max = float(raw[calibrate].max()) if raw[calibrate].size else 0.0
            self.thresholds_[source.name] = calib_max if calib_max > 0 else 1.0
        return self

    @property
    def sources(self) -> list[str]:
        return list(self.raw_)

    def service_scores(self, sources: Sequence[str] | None = None) -> np.ndarray:
        """Normalised scores of shape (windows, services); values above 1 are alarms."""
        names = list(sources) if sources is not None else self.sources
        if not names:
            raise ValueError("select at least one source")
        return np.max([self.raw_[n] / self.thresholds_[n] for n in names], axis=0)

    def window_scores(self, sources: Sequence[str] | None = None) -> np.ndarray:
        return self.service_scores(sources).max(axis=1)

    def predict(self, sources: Sequence[str] | None = None) -> np.ndarray:
        return (self.window_scores(sources) > 1.0).astype(int)
