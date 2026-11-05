"""Read the three raw sources and align them on (time window, service).

* Traces (JSON lines, one span per line) give the call volume, the number of failed calls
  and the latency of each service, and, through parent/child links, the call graph.
* Business logs (free text) are parsed into event templates with the same miner used for
  HDFS; each template's count per window becomes a feature.
* Metrics (long CSV) give one value per service, metric and window.

All three are turned into arrays of shape (windows, services, features).
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from logfusion.parsing.miner import TemplateMiner

LOG_LINE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}:\d{2}:\d{2}\.\d{3}) (?P<level>[A-Z]+) "
    r"\[(?P<service>[^\]]+)\] (?P<message>.*)$"
)


def parse_time(text: str) -> float:
    """Seconds since the epoch for ``2018-01-01T00:00:01.469Z`` or ``2018-01-01 00:00:01.469``."""
    text = text.rstrip("Z").replace(" ", "T")
    return datetime.fromisoformat(text).replace(tzinfo=UTC).timestamp()


@dataclass(frozen=True)
class Clock:
    """Maps timestamps to window indices."""

    start: float
    window_seconds: int
    windows: int

    def window(self, timestamp: float) -> int | None:
        w = int((timestamp - self.start) // self.window_seconds)
        return w if 0 <= w < self.windows else None


@dataclass
class SourceFeatures:
    """One source as an array of shape (windows, services, features)."""

    name: str
    features: list[str]
    values: np.ndarray


@dataclass
class TraceData:
    features: SourceFeatures
    edges: Counter[tuple[str, str]] = field(default_factory=Counter)


def load_traces(path: Path, services: list[str], clock: Clock) -> TraceData:
    index = {s: i for i, s in enumerate(services)}
    durations: dict[tuple[int, int], list[float]] = defaultdict(list)
    errors = np.zeros((clock.windows, len(services)))
    span_service: dict[str, str] = {}
    parents: list[tuple[str, str]] = []
    with path.open() as f:
        for line in f:
            span = json.loads(line)
            service = span["service"]
            span_service[span["span_id"]] = service
            if span["parent_id"] is not None:
                parents.append((span["parent_id"], service))
            w = clock.window(parse_time(span["start"]))
            if w is None or service not in index:
                continue
            durations[(w, index[service])].append(float(span["duration_ms"]))
            if span["status"] != "OK":
                errors[w, index[service]] += 1

    values = np.zeros((clock.windows, len(services), 4))
    for (w, s), d in durations.items():
        arr = np.asarray(d)
        values[w, s] = (len(arr), errors[w, s], arr.mean(), np.percentile(arr, 95))
    edges: Counter[tuple[str, str]] = Counter()
    for parent_id, child in parents:
        caller = span_service.get(parent_id)
        if caller is not None:
            edges[(caller, child)] += 1
    names = ["calls", "errors", "latency_mean_ms", "latency_p95_ms"]
    return TraceData(SourceFeatures("traces", names, values), edges)


def load_business_logs(
    path: Path, services: list[str], clock: Clock, miner: TemplateMiner | None = None
) -> SourceFeatures:
    """Count each event template per window and service."""
    miner = miner or TemplateMiner()
    index = {s: i for i, s in enumerate(services)}
    counts: Counter[tuple[int, int, int]] = Counter()
    with path.open() as f:
        for line in f:
            m = LOG_LINE.match(line.rstrip("\n"))
            if m is None or m["service"] not in index:
                continue
            w = clock.window(parse_time(f"{m['date']} {m['time']}"))
            template = miner.add(f"{m['level']} {m['message']}")
            if w is not None:
                counts[(w, index[m["service"]], template.id)] += 1
    values = np.zeros((clock.windows, len(services), len(miner.templates)))
    for (w, s, t), c in counts.items():
        values[w, s, t] = c
    return SourceFeatures("logs", [t.text for t in miner.templates], values)


def load_metrics(path: Path, services: list[str], clock: Clock) -> SourceFeatures:
    """Average each metric per window and service."""
    index = {s: i for i, s in enumerate(services)}
    sums: dict[tuple[int, int, str], float] = defaultdict(float)
    counts: Counter[tuple[int, int, str]] = Counter()
    names: list[str] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["service"] not in index:
                continue
            w = clock.window(parse_time(row["timestamp"]))
            if w is None:
                continue
            if row["metric"] not in names:
                names.append(row["metric"])
            key = (w, index[row["service"]], row["metric"])
            sums[key] += float(row["value"])
            counts[key] += 1
    values = np.zeros((clock.windows, len(services), len(names)))
    for (w, s, metric), total in sums.items():
        values[w, s, names.index(metric)] = total / counts[(w, s, metric)]
    return SourceFeatures("metrics", names, values)
