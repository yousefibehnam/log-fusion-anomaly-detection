"""Generate synthetic traces, business logs and metrics with injected, labelled faults.

The simulation runs in fixed windows (60 s by default). In each window a Poisson number
of user requests enters at the entry service and fans out along the call graph. Every
service call becomes a trace span and produces business-log lines; every service reports
CPU and memory once per window. Faults are injected at one root service for a few
windows. The four fault types show up in different sources:

=============  ===========================  ==========================  ==================
fault          traces                       business logs               metrics
=============  ===========================  ==========================  ==================
``cpu``        root is 2.5x slower          (nothing new)               CPU +35-45 points
``memory``     root up to 30% slower        GC warnings at the root     memory ramps up
``latency``    +150-250 ms at the root      callers log slow responses  (nothing new)
``error``      root fails 35% of calls,     ERROR lines at the root     (nothing new)
               callers fail in turn         and at failing callers
=============  ===========================  ==========================  ==================

Slowness and errors propagate to callers through the call structure itself, as they would
in a real system, so symptoms appear upstream of the root cause.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from logfusion.synthetic.topology import Topology, default_topology

FAULT_TYPES = ("cpu", "memory", "latency", "error")
NOMINAL_START = datetime(2018, 1, 1, tzinfo=UTC)  # a label for the simulated clock, nothing more
SLOW_CALL_MS = 120.0

# Business-log message patterns per service; {n} is replaced by a random number.
INFO_MESSAGES: dict[str, tuple[str, ...]] = {
    "frontend": (
        "GET /product/{n} served",
        "GET /home served for session {n}",
        "POST /cart/{n} accepted",
    ),
    "auth": ("token issued for user {n}", "token validated for user {n}"),
    "catalog": ("listed {n} products in category books", "product {n} details loaded"),
    "cart": ("cart {n} updated with {n} items", "cart {n} loaded"),
    "checkout": ("order {n} created with {n} items", "order {n} confirmed"),
    "payment": ("payment authorised for order {n} amount {n}", "refund issued for order {n}"),
    "inventory": ("stock reserved for sku {n} qty {n}", "stock level for sku {n} is {n}"),
    "shipping": ("shipment quote computed for order {n}", "label created for order {n}"),
    "cache": ("cache hit for key {n}", "cache miss for key {n}"),
    "db": ("query executed in {n} ms rows {n}", "transaction {n} committed"),
}
ERROR_MESSAGES: dict[str, str] = {
    "frontend": "request {n} failed with status 500",
    "auth": "token service unavailable for user {n}",
    "catalog": "failed to load product {n}",
    "cart": "cart {n} could not be saved",
    "checkout": "order {n} rejected: downstream failure",
    "payment": "payment gateway error for order {n}: timeout",
    "inventory": "stock reservation failed for sku {n}",
    "shipping": "carrier api returned error for order {n}",
    "cache": "cache node unreachable for key {n}",
    "db": "connection pool exhausted after {n} ms",
}
NOISE_WARNINGS = (
    "retrying request {n} after transient error",
    "slow consumer detected on queue {n}",
)


@dataclass(frozen=True)
class Fault:
    kind: str
    root: str
    start: int  # first window (inclusive)
    end: int  # last window (exclusive)

    def active(self, window: int) -> bool:
        return self.start <= window < self.end


@dataclass(frozen=True)
class GeneratorConfig:
    windows: int = 720
    window_seconds: int = 60
    requests_per_window: float = 30.0
    normal_fraction: float = 0.3
    faults: int = 10
    min_duration: int = 5
    max_duration: int = 15
    min_gap: int = 10
    noise_warning_rate: float = 0.02
    seed: int = 0


@dataclass
class SyntheticDataset:
    config: GeneratorConfig
    faults: list[Fault]
    spans: list[dict] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    metrics: list[tuple[str, str, str, float]] = field(default_factory=list)

    def write(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "traces.jsonl").open("w") as f:
            for span in sorted(self.spans, key=lambda sp: sp["start"]):
                f.write(json.dumps(span, separators=(",", ":")) + "\n")
        (out_dir / "business.log").write_text("\n".join(sorted(self.logs)) + "\n")
        with (out_dir / "metrics.csv").open("w") as f:
            f.write("timestamp,service,metric,value\n")
            for ts, service, metric, value in self.metrics:
                f.write(f"{ts},{service},{metric},{value:.3f}\n")
        truth = {
            "synthetic": True,
            "config": asdict(self.config),
            "faults": [asdict(fault) for fault in self.faults],
        }
        (out_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2) + "\n")


class Generator:
    def __init__(self, config: GeneratorConfig | None = None, topology: Topology | None = None):
        self.config = config or GeneratorConfig()
        self.topology = topology or default_topology()
        self.rng = np.random.default_rng(self.config.seed)
        self._ids = 0
        self._leak_mb = {name: 0.0 for name in self.topology.names}

    # ---------------------------------------------------------------- schedule
    def schedule_faults(self) -> list[Fault]:
        """Non-overlapping faults after the fault-free prefix, at random non-entry services."""
        c = self.config
        candidates = [n for n in self.topology.names if n != self.topology.entry]
        first = int(c.windows * c.normal_fraction)
        faults: list[Fault] = []
        cursor = first + c.min_gap
        for i in range(c.faults):
            remaining = c.faults - i
            room = c.windows - cursor - remaining * (c.max_duration + c.min_gap)
            start = cursor + int(self.rng.integers(0, max(room // remaining, 1)))
            duration = int(self.rng.integers(c.min_duration, c.max_duration + 1))
            if start + duration > c.windows:
                break
            kind = FAULT_TYPES[i % len(FAULT_TYPES)]
            root = candidates[int(self.rng.integers(len(candidates)))]
            faults.append(Fault(kind, root, start, start + duration))
            cursor = start + duration + c.min_gap
        order = self.rng.permutation(len(faults))  # shuffle kinds over time, keep positions
        kinds = [faults[i].kind for i in order]
        return [Fault(k, f.root, f.start, f.end) for k, f in zip(kinds, faults, strict=True)]

    # ---------------------------------------------------------------- simulation
    def generate(self) -> SyntheticDataset:
        c = self.config
        faults = self.schedule_faults()
        data = SyntheticDataset(c, faults)
        leak_mb = self._leak_mb

        for w in range(c.windows):
            active = [f for f in faults if f.active(w)]
            window_start = NOMINAL_START + timedelta(seconds=w * c.window_seconds)
            load = c.requests_per_window * (1.0 + 0.3 * np.sin(2 * np.pi * w / c.windows))
            calls = {name: 0 for name in self.topology.names}

            for _ in range(int(self.rng.poisson(load))):
                offset = float(self.rng.uniform(0, c.window_seconds * 1000.0))
                trace_id = self._next_id()
                self._span(
                    self.topology.entry, None, trace_id, window_start, offset, active, data, calls
                )

            for f in active:
                if f.kind == "memory":
                    leak_mb[f.root] += float(self.rng.uniform(25.0, 40.0))
            for name in leak_mb:
                if not any(f.kind == "memory" and f.root == name for f in active):
                    leak_mb[name] = 0.0  # the leaking process was restarted
            self._metrics(w, window_start, active, calls, leak_mb, data)
            self._noise(window_start, data)
        return data

    def _span(self, service, parent, trace_id, window_start, offset_ms, active, data, calls):
        """Simulate one call to ``service``; returns (duration_ms, failed)."""
        spec = self.topology.service(service)
        calls[service] += 1
        span_id = self._next_id()
        own = float(self.rng.lognormal(np.log(spec.latency_ms), 0.25))
        failed = False
        for f in active:
            if f.root != service:
                continue
            if f.kind == "cpu":
                own *= 2.5
            elif f.kind == "latency":
                own += float(self.rng.uniform(150.0, 250.0))
            elif f.kind == "memory":
                own *= 1.0 + min(
                    0.3, self._leak_mb[service] / 1000.0
                )  # GC pressure grows with the leak
            elif f.kind == "error" and self.rng.random() < 0.35:
                failed = True

        children_ms = 0.0
        for callee, p in self.topology.calls.get(service, ()):
            if self.rng.random() >= p:
                continue
            child_ms, child_failed = self._span(
                callee,
                span_id,
                trace_id,
                window_start,
                offset_ms + own + children_ms,
                active,
                data,
                calls,
            )
            children_ms += child_ms
            if child_ms > SLOW_CALL_MS:
                self._log(
                    data,
                    window_start,
                    offset_ms,
                    "WARN",
                    service,
                    f"slow response from {callee} took {int(child_ms)} ms",
                )
            if child_failed and self.rng.random() < 0.8:
                failed = True

        duration = own + children_ms
        data.spans.append(
            {
                "trace_id": f"{trace_id:x}",
                "span_id": f"{span_id:x}",
                "parent_id": None if parent is None else f"{parent:x}",
                "service": service,
                "start": _iso(window_start + timedelta(milliseconds=offset_ms)),
                "duration_ms": round(duration, 3),
                "status": "ERROR" if failed else "OK",
            }
        )
        if failed:
            self._log(
                data, window_start, offset_ms, "ERROR", service, self._fill(ERROR_MESSAGES[service])
            )
        else:
            pattern = INFO_MESSAGES[service][int(self.rng.integers(len(INFO_MESSAGES[service])))]
            self._log(data, window_start, offset_ms, "INFO", service, self._fill(pattern))
        return duration, failed

    def _metrics(self, w, window_start, active, calls, leak_mb, data):
        ts = _iso(window_start + timedelta(seconds=self.config.window_seconds - 1))
        for spec in self.topology.services:
            cpu = (
                spec.cpu_base
                + spec.cpu_per_call * calls[spec.name]
                + float(self.rng.normal(0, 1.5))
            )
            if any(f.kind == "cpu" and f.root == spec.name for f in active):
                cpu += float(self.rng.uniform(35.0, 45.0))
            memory = spec.memory_mb + leak_mb[spec.name] + float(self.rng.normal(0, 8.0))
            if leak_mb[spec.name] > 100 and self.rng.random() < 0.5:
                self._log(
                    data,
                    window_start,
                    30_000.0,
                    "WARN",
                    spec.name,
                    f"GC overhead high: pause {int(self.rng.integers(200, 900))} ms",
                )
            data.metrics.append((ts, spec.name, "cpu_pct", min(cpu, 100.0)))
            data.metrics.append((ts, spec.name, "memory_mb", memory))

    def _noise(self, window_start, data):
        """Rare, harmless warnings so that not every warning signals a fault."""
        for name in self.topology.names:
            if self.rng.random() < self.config.noise_warning_rate:
                pattern = NOISE_WARNINGS[int(self.rng.integers(len(NOISE_WARNINGS)))]
                offset = float(self.rng.uniform(0, self.config.window_seconds * 1000.0))
                self._log(data, window_start, offset, "WARN", name, self._fill(pattern))

    def _log(self, data, window_start, offset_ms, level, service, message):
        ts = (window_start + timedelta(milliseconds=offset_ms)).strftime("%Y-%m-%d %H:%M:%S.%f")[
            :-3
        ]
        data.logs.append(f"{ts} {level} [{service}] {message}")

    def _fill(self, pattern: str) -> str:
        while "{n}" in pattern:
            pattern = pattern.replace("{n}", str(int(self.rng.integers(1, 100_000))), 1)
        return pattern

    def _next_id(self) -> int:
        self._ids += 1
        return self._ids


def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def generate(out_dir: Path, config: GeneratorConfig | None = None) -> SyntheticDataset:
    """Generate a dataset and write it to ``out_dir``."""
    data = Generator(config).generate()
    data.write(out_dir)
    return data
