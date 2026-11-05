"""Three-source fusion and root-cause experiment on SYNTHETIC data.

For each seed: generate a dataset, load its traces, business logs and metrics, fit the
detectors on the first 20% of windows, calibrate thresholds on the next 10% (both
fault-free), and evaluate on the remaining 70%.

* Detection is scored per window (anomalous = some fault is active) for every non-empty
  combination of sources.
* For every fault that the fused detector catches in at least one window, services are
  ranked by their mean fused score over the detected windows ("score") and by the
  anomaly-guided random walk over the call graph learned from the traces ("walk").
"""

from __future__ import annotations

import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from logfusion.detectors.rootcause import rank_by_score, rank_by_walk
from logfusion.evaluation import detection, mean_reciprocal_rank, rank_of, top_k_accuracy
from logfusion.fusion.detector import FusionDetector
from logfusion.fusion.sources import Clock, load_business_logs, load_metrics, load_traces
from logfusion.synthetic.generator import FAULT_TYPES, NOMINAL_START, GeneratorConfig, generate
from logfusion.synthetic.topology import default_topology

SOURCES = ("traces", "logs", "metrics")
TRAIN_FRACTION = 0.2
CALIBRATION_FRACTION = 0.1


def source_combinations() -> list[tuple[str, ...]]:
    return [c for k in range(1, len(SOURCES) + 1) for c in itertools.combinations(SOURCES, k)]


def run_seed(seed: int, work_dir: Path, config: GeneratorConfig | None = None) -> dict:
    config = replace(config or GeneratorConfig(), seed=seed)
    data_dir = work_dir / f"seed{seed}"
    if not (data_dir / "ground_truth.json").exists():
        generate(data_dir, config)
    truth = json.loads((data_dir / "ground_truth.json").read_text())
    faults = truth["faults"]
    windows = truth["config"]["windows"]

    services = default_topology().names
    clock = Clock(NOMINAL_START.timestamp(), truth["config"]["window_seconds"], windows)
    traces = load_traces(data_dir / "traces.jsonl", services, clock)
    logs = load_business_logs(data_dir / "business.log", services, clock)
    metrics = load_metrics(data_dir / "metrics.csv", services, clock)

    train_end = int(windows * TRAIN_FRACTION)
    calib_end = int(windows * (TRAIN_FRACTION + CALIBRATION_FRACTION))
    if any(f["start"] < calib_end for f in faults):
        raise ValueError("faults overlap the fault-free training/calibration windows")
    detector = FusionDetector().fit(
        [traces.features, logs, metrics], slice(0, train_end), slice(train_end, calib_end)
    )

    labels = np.zeros(windows, dtype=int)
    kind_of_window = np.full(windows, "", dtype=object)
    for f in faults:
        labels[f["start"] : f["end"]] = 1
        kind_of_window[f["start"] : f["end"]] = f["kind"]
    test = slice(calib_end, windows)

    result: dict = {"seed": seed, "detection": {}, "recall_by_fault": {}, "rootcause": []}
    for combo in source_combinations():
        name = "+".join(combo)
        predicted = detector.predict(combo)
        result["detection"][name] = detection(labels[test], predicted[test]).as_dict()
        result["recall_by_fault"][name] = {
            kind: float(predicted[kind_of_window == kind].mean())
            for kind in FAULT_TYPES
            if (kind_of_window == kind).any()
        }

    fused_scores = detector.service_scores(SOURCES)
    fused_alarm = detector.predict(SOURCES)
    for f in faults:
        detected = [w for w in range(f["start"], f["end"]) if fused_alarm[w]]
        entry = {
            "kind": f["kind"],
            "root": f["root"],
            "windows": f["end"] - f["start"],
            "detected_windows": len(detected),
        }
        if detected:
            scores = {s: float(fused_scores[detected, i].mean()) for i, s in enumerate(services)}
            entry["rank_score"] = rank_of(rank_by_score(scores), f["root"])
            entry["rank_walk"] = rank_of(rank_by_walk(traces.edges, scores), f["root"])
        result["rootcause"].append(entry)

    result["learned_edges"] = sorted(f"{a}->{b}" for a, b in traces.edges)
    result["log_templates"] = len(logs.features)
    return result


def summarise(runs: list[dict]) -> dict:
    """Mean and standard deviation over seeds, plus pooled root-cause statistics."""
    summary: dict = {"seeds": [r["seed"] for r in runs], "detection": {}, "recall_by_fault": {}}
    for combo in runs[0]["detection"]:
        per_metric = {}
        for metric in ("precision", "recall", "f1"):
            values = [r["detection"][combo][metric] for r in runs]
            per_metric[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        summary["detection"][combo] = per_metric
        summary["recall_by_fault"][combo] = {
            kind: float(
                np.mean(
                    [
                        r["recall_by_fault"][combo][kind]
                        for r in runs
                        if kind in r["recall_by_fault"][combo]
                    ]
                )
            )
            for kind in FAULT_TYPES
        }

    cases = [c for r in runs for c in r["rootcause"]]
    found = [c for c in cases if c["detected_windows"] > 0]
    rca: dict = {"faults": len(cases), "detected": len(found), "methods": {}, "by_fault": {}}
    for method in ("score", "walk"):
        ranks = [c[f"rank_{method}"] for c in found]
        rca["methods"][method] = {
            "top1": top_k_accuracy(ranks, 1),
            "top3": top_k_accuracy(ranks, 3),
            "mrr": mean_reciprocal_rank(ranks),
        }
    for kind in FAULT_TYPES:
        subset = [c for c in found if c["kind"] == kind]
        rca["by_fault"][kind] = {
            "faults": sum(1 for c in cases if c["kind"] == kind),
            "detected": len(subset),
            **{
                f"{m}_top1": top_k_accuracy([c[f"rank_{m}"] for c in subset], 1)
                for m in ("score", "walk")
            },
        }
    summary["rootcause"] = rca
    return summary


def run(seeds: list[int], work_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [run_seed(seed, work_dir) for seed in seeds]
    summary = summarise(runs)
    (out_dir / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "summary.md").write_text(render_markdown(summary))
    return summary


def render_markdown(summary: dict) -> str:
    seeds = summary["seeds"]
    lines = [
        "# Synthetic multi-service experiment",
        "",
        "**All data in this experiment are synthetic** (generated by `logfusion.synthetic`). "
        f"Seeds: {seeds[0]}–{seeds[-1]} ({len(seeds)} runs); seed 0 was used during development "
        "and is not included. Values are mean ± standard deviation over seeds.",
        "",
        "## Window-level detection (test windows)",
        "",
        "| Sources | Precision | Recall | F1 |",
        "|---|---:|---:|---:|",
    ]
    for combo, m in summary["detection"].items():
        cells = " | ".join(
            f"{m[k]['mean']:.3f} ± {m[k]['std']:.3f}" for k in ("precision", "recall", "f1")
        )
        lines.append(f"| {combo} | {cells} |")
    lines += [
        "",
        "## Recall by fault type (share of fault windows detected)",
        "",
        "| Sources | " + " | ".join(FAULT_TYPES) + " |",
        "|---|" + "---:|" * len(FAULT_TYPES),
    ]
    for combo, by_kind in summary["recall_by_fault"].items():
        lines.append(f"| {combo} | " + " | ".join(f"{by_kind[k]:.3f}" for k in FAULT_TYPES) + " |")
    rca = summary["rootcause"]
    lines += [
        "",
        "## Root-cause ranking (fused scores)",
        "",
        f"{rca['detected']} of {rca['faults']} injected faults were detected in at least one "
        "window; rankings are computed for those. 10 services are ranked (random top-1 = 0.10).",
        "",
        "| Method | Top-1 | Top-3 | MRR |",
        "|---|---:|---:|---:|",
    ]
    labels = {"score": "Own anomaly score", "walk": "Anomaly-guided walk on call graph"}
    for method, m in rca["methods"].items():
        lines.append(f"| {labels[method]} | {m['top1']:.3f} | {m['top3']:.3f} | {m['mrr']:.3f} |")
    lines += [
        "",
        "| Fault type | Faults | Detected | Top-1 (score) | Top-1 (walk) |",
        "|---|---:|---:|---:|---:|",
    ]
    for kind, m in rca["by_fault"].items():
        lines.append(
            f"| {kind} | {m['faults']} | {m['detected']} | {m['score_top1']:.3f} | "
            f"{m['walk_top1']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def parse_seeds(text: str) -> list[int]:
    """``"1-10"`` or ``"1,2,5"``."""
    if "-" in text:
        lo, hi = (int(x) for x in text.split("-", 1))
        return list(range(lo, hi + 1))
    return [int(x) for x in text.split(",") if x]
