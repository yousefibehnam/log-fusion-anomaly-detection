import numpy as np
import pytest

from logfusion.fusion.detector import FusionDetector
from logfusion.fusion.sources import (
    Clock,
    SourceFeatures,
    load_business_logs,
    load_metrics,
    load_traces,
    parse_time,
)

SERVICES = ["api", "db"]
START = parse_time("2018-01-01T00:00:00Z")
CLOCK = Clock(START, 60, 3)


def test_clock_windows():
    assert CLOCK.window(START) == 0
    assert CLOCK.window(START + 125) == 2
    assert CLOCK.window(START + 180) is None and CLOCK.window(START - 1) is None


def test_load_traces_features_and_graph(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text(
        '{"trace_id":"1","span_id":"a","parent_id":null,"service":"api","start":"2018-01-01T00:00:01.000Z","duration_ms":30.0,"status":"OK"}\n'
        '{"trace_id":"1","span_id":"b","parent_id":"a","service":"db","start":"2018-01-01T00:00:01.010Z","duration_ms":10.0,"status":"ERROR"}\n'
        '{"trace_id":"2","span_id":"c","parent_id":null,"service":"api","start":"2018-01-01T00:01:05.000Z","duration_ms":50.0,"status":"OK"}\n'
    )
    data = load_traces(path, SERVICES, CLOCK)
    v = data.features.values
    assert data.features.features == ["calls", "errors", "latency_mean_ms", "latency_p95_ms"]
    np.testing.assert_allclose(v[0, 0], [1, 0, 30, 30])
    np.testing.assert_allclose(v[0, 1], [1, 1, 10, 10])
    np.testing.assert_allclose(v[1, 0], [1, 0, 50, 50])
    assert dict(data.edges) == {("api", "db"): 1}


def test_load_business_logs_counts_templates(tmp_path):
    path = tmp_path / "business.log"
    path.write_text(
        "2018-01-01 00:00:01.000 INFO [api] order 12 created\n"
        "2018-01-01 00:00:02.000 INFO [api] order 99 created\n"
        "2018-01-01 00:01:03.000 ERROR [db] connection pool exhausted after 300 ms\n"
        "not a log line\n"
    )
    logs = load_business_logs(path, SERVICES, CLOCK)
    assert logs.features == [
        "INFO order <NUM> created",
        "ERROR connection pool exhausted after <NUM> ms",
    ]
    assert logs.values[0, 0, 0] == 2 and logs.values[1, 1, 1] == 1 and logs.values.sum() == 3


def test_load_metrics_averages(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text(
        "timestamp,service,metric,value\n"
        "2018-01-01T00:00:10.000Z,api,cpu_pct,10\n"
        "2018-01-01T00:00:50.000Z,api,cpu_pct,20\n"
        "2018-01-01T00:02:00.000Z,db,memory_mb,512\n"
    )
    metrics = load_metrics(path, SERVICES, CLOCK)
    assert metrics.features == ["cpu_pct", "memory_mb"]
    assert metrics.values[0, 0, 0] == 15 and metrics.values[2, 1, 1] == 512


def test_fusion_detector_calibrates_and_fuses():
    # A periodic signal (period 10) so that the calibration windows cover its full range.
    wave = np.sin(2 * np.pi * np.arange(40) / 10)[:, None, None]
    a = 10 + 2 * wave * np.ones((40, 2, 1))
    b = 5 + 2 * wave * np.ones((40, 2, 1))
    a[35, 0, 0] += 50  # only source "a" sees window 35
    b[37, 1, 0] += 50  # only source "b" sees window 37
    sources = [SourceFeatures("a", ["x"], a), SourceFeatures("b", ["y"], b)]
    det = FusionDetector().fit(sources, slice(0, 20), slice(20, 30))

    calibration = det.predict()[20:30]
    assert calibration.sum() == 0  # by construction: no alarm on calibration windows
    assert det.predict(["a"])[35] == 1 and det.predict(["a"])[37] == 0
    assert det.predict(["b"])[37] == 1 and det.predict(["b"])[35] == 0
    assert det.predict(["a", "b"])[[35, 37]].tolist() == [1, 1]
    assert det.service_scores()[35].argmax() == 0 and det.service_scores()[37].argmax() == 1


def test_fusion_detector_needs_a_source():
    det = FusionDetector().fit(
        [SourceFeatures("a", ["x"], np.ones((5, 1, 1)))], slice(0, 3), slice(3, 5)
    )
    with pytest.raises(ValueError):
        det.service_scores([])
