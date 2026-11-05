"""Evaluation measures: detection quality, ranking quality and parsing accuracy."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_dict(self) -> dict[str, float]:
        return {**asdict(self), "precision": self.precision, "recall": self.recall, "f1": self.f1}


def detection(y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray) -> Detection:
    """Confusion counts for binary labels (1 = anomaly)."""
    t = np.asarray(y_true).astype(bool)
    p = np.asarray(y_pred).astype(bool)
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: {t.shape} vs {p.shape}")
    return Detection(
        tp=int(np.sum(t & p)),
        fp=int(np.sum(~t & p)),
        fn=int(np.sum(t & ~p)),
        tn=int(np.sum(~t & ~p)),
    )


def rank_of(ranking: Sequence[Hashable], target: Hashable) -> int:
    """1-based position of ``target`` in ``ranking``; ``len(ranking) + 1`` if absent."""
    try:
        return list(ranking).index(target) + 1
    except ValueError:
        return len(ranking) + 1


def top_k_accuracy(ranks: Sequence[int], k: int) -> float:
    """Fraction of cases where the true item is ranked within the first ``k``."""
    return float(np.mean([r <= k for r in ranks])) if ranks else 0.0


def mean_reciprocal_rank(ranks: Sequence[int]) -> float:
    return float(np.mean([1.0 / r for r in ranks])) if ranks else 0.0


def grouping_accuracy(predicted: Sequence[Hashable], truth: Sequence[Hashable]) -> float:
    """Share of messages whose predicted group contains exactly the messages of their true group."""
    if len(predicted) != len(truth):
        raise ValueError("predicted and truth differ in length")
    if not truth:
        return 0.0
    pred_groups: dict[Hashable, set[int]] = defaultdict(set)
    true_groups: dict[Hashable, set[int]] = defaultdict(set)
    for i, (p, t) in enumerate(zip(predicted, truth, strict=True)):
        pred_groups[p].add(i)
        true_groups[t].add(i)
    correct = sum(
        1 for p, t in zip(predicted, truth, strict=True) if pred_groups[p] == true_groups[t]
    )
    return correct / len(truth)
