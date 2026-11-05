import json

import pytest

from logfusion.synthetic.generator import FAULT_TYPES, Generator, GeneratorConfig, generate
from logfusion.synthetic.topology import Service, Topology, default_topology

SMALL = GeneratorConfig(windows=200, faults=4, requests_per_window=8.0, seed=3)


def test_default_topology_is_consistent():
    topology = default_topology()
    assert len(topology.names) == 10
    assert topology.callers("db") == ["inventory", "payment", "auth"]
    assert set(topology.callees("checkout")) == {"cart", "payment", "inventory", "shipping"}


def test_topology_rejects_cycles_and_unknown_services():
    a, b = Service("a", 1, 1, 0, 1), Service("b", 1, 1, 0, 1)
    with pytest.raises(ValueError, match="acyclic"):
        Topology((a, b), {"a": (("b", 1.0),), "b": (("a", 1.0),)}, entry="a")
    with pytest.raises(ValueError, match="undefined"):
        Topology((a,), {"a": (("zzz", 1.0),)}, entry="a")


def test_fault_schedule_is_valid():
    faults = Generator(SMALL).schedule_faults()
    assert len(faults) == 4
    first_allowed = int(SMALL.windows * SMALL.normal_fraction)
    for f in faults:
        assert f.kind in FAULT_TYPES and f.root != "frontend"
        assert SMALL.min_duration <= f.end - f.start <= SMALL.max_duration
        assert f.start >= first_allowed and f.end <= SMALL.windows
    for earlier, later in zip(faults, faults[1:], strict=False):
        assert later.start >= earlier.end + SMALL.min_gap


def test_generation_is_deterministic_per_seed():
    a, b = Generator(SMALL).generate(), Generator(SMALL).generate()
    assert a.faults == b.faults and a.logs == b.logs and a.spans == b.spans


def test_written_files(tmp_path):
    data = generate(tmp_path, SMALL)
    truth = json.loads((tmp_path / "ground_truth.json").read_text())
    assert truth["synthetic"] is True and len(truth["faults"]) == len(data.faults)
    assert sum(1 for _ in (tmp_path / "traces.jsonl").open()) == len(data.spans)
    assert (tmp_path / "metrics.csv").read_text().startswith("timestamp,service,metric,value\n")
    assert len(data.metrics) == SMALL.windows * 10 * 2


def test_error_faults_produce_error_spans_and_logs():
    data = Generator(GeneratorConfig(windows=200, faults=4, seed=5)).generate()
    error_faults = [f for f in data.faults if f.kind == "error"]
    assert error_faults, "seed 5 schedules at least one error fault"
    root = error_faults[0].root
    assert any(s["service"] == root and s["status"] == "ERROR" for s in data.spans)
    assert any(f" ERROR [{root}] " in line for line in data.logs)
