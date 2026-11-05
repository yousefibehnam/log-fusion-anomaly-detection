import json

from logfusion.cli import main
from logfusion.experiments.synthetic import parse_seeds, run_seed, summarise
from logfusion.synthetic.generator import GeneratorConfig


def test_generate_command(tmp_path, capsys):
    assert main(["generate", "--seed", "2", "--windows", "150", "--out", str(tmp_path)]) == 0
    assert "faults" in capsys.readouterr().out
    assert json.loads((tmp_path / "ground_truth.json").read_text())["config"]["windows"] == 150


def test_hdfs_command_without_data(tmp_path, capsys):
    assert main(["hdfs", "--data", str(tmp_path), "--out", str(tmp_path / "out")]) == 2
    assert "download_data.sh" in capsys.readouterr().err


def test_parse_seeds():
    assert parse_seeds("1-3") == [1, 2, 3]
    assert parse_seeds("4,7") == [4, 7]


def test_synthetic_experiment_end_to_end(tmp_path):
    config = GeneratorConfig(windows=300, faults=4, requests_per_window=10.0)
    run = run_seed(11, tmp_path, config)
    assert set(run["detection"]) == {
        "traces",
        "logs",
        "metrics",
        "traces+logs",
        "traces+metrics",
        "logs+metrics",
        "traces+logs+metrics",
    }
    assert len(run["rootcause"]) == 4
    assert "frontend->auth" in run["learned_edges"]
    summary = summarise([run])
    assert summary["rootcause"]["faults"] == 4
    fused = summary["detection"]["traces+logs+metrics"]
    assert 0.0 <= fused["f1"]["mean"] <= 1.0
