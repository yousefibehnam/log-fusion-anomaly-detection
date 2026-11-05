"""Root-cause ranking on the service call graph.

Faults propagate upstream: when a dependency fails or slows down, its callers show the
symptoms too, and a caller can even look *more* anomalous than the faulty service. Ranking
services by their own anomaly score therefore often blames a caller. The graph-based
ranking runs a personalised random walk that follows the anomaly downstream:

* from service ``u`` the walk moves to a callee ``v`` with weight ``score(v)``,
  so mass flows towards anomalous dependencies;
* it moves back to a caller ``c`` with the smaller weight ``backward * score(c)``,
  so it can recover from a wrong turn;
* it stays at ``u`` with weight ``max(0, score(u) - max score of u's callees)``: a service
  that is more anomalous than everything it depends on is a plausible origin;
* with probability ``1 - damping`` it restarts at a service chosen in proportion to its score.

Services are ranked by their stationary probability.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np


def rank_by_score(scores: Mapping[str, float]) -> list[str]:
    """Services from most to least anomalous (ties broken by name)."""
    return sorted(scores, key=lambda s: (-scores[s], s))


def anomaly_walk(
    edges: Iterable[tuple[str, str]],
    scores: Mapping[str, float],
    damping: float = 0.85,
    backward: float = 0.2,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> dict[str, float]:
    """Stationary distribution of the anomaly-guided random walk described above."""
    if not 0.0 < damping < 1.0:
        raise ValueError("damping must be in (0, 1)")
    services = sorted(scores)
    index = {s: i for i, s in enumerate(services)}
    n = len(services)
    if n == 0:
        return {}
    s = np.array([max(float(scores[x]), 0.0) for x in services])

    callees: list[set[int]] = [set() for _ in range(n)]
    callers: list[set[int]] = [set() for _ in range(n)]
    for caller, callee in edges:
        if caller in index and callee in index and caller != callee:
            callees[index[caller]].add(index[callee])
            callers[index[callee]].add(index[caller])

    transition = np.zeros((n, n))
    for u in range(n):
        for v in callees[u]:
            transition[u, v] += s[v]
        for c in callers[u]:
            transition[u, c] += backward * s[c]
        deepest = max((s[v] for v in callees[u]), default=0.0)
        transition[u, u] += max(0.0, s[u] - deepest)
        row = transition[u].sum()
        if row > 0:
            transition[u] /= row
        else:
            transition[u, u] = 1.0  # nothing anomalous around: stay

    restart = s / s.sum() if s.sum() > 0 else np.full(n, 1.0 / n)
    p = restart.copy()
    for _ in range(max_iter):
        nxt = damping * p @ transition + (1.0 - damping) * restart
        if np.abs(nxt - p).sum() < tol:
            p = nxt
            break
        p = nxt
    return {x: float(p[index[x]]) for x in services}


def rank_by_walk(
    edges: Iterable[tuple[str, str]], scores: Mapping[str, float], **kwargs
) -> list[str]:
    """Services ordered by their stationary probability in :func:`anomaly_walk`."""
    probability = anomaly_walk(edges, scores, **kwargs)
    return sorted(probability, key=lambda x: (-probability[x], x))
