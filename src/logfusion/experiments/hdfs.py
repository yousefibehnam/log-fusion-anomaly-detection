"""Log parsing and statistical anomaly detection on the real HDFS_v1 data set.

* Parsing: the template miner is run over all 11,175,629 lines; its grouping accuracy is
  measured on the 2,000 lines that Loghub labelled by hand (``HDFS_2k``).
* Detection: blocks are ordered by their first log line and split in half. Detectors
  are fitted on the first half **without labels** (unsupervised, as in Xu et al. 2009)
  and evaluated against the real block labels on the second half. This is the main,
  pre-declared protocol (PCA keeps components explaining 95% of the variance).
* Protocol study: PCA is also run (a) with 90% and 99% variance, (b) fitted and evaluated
  on all blocks, the protocol of earlier HDFS studies, and (c) semi-supervised, fitted only
  on training blocks labelled normal. Each is run on this project's templates and on the
  29 reference events shipped with HDFS_v1 (``Event_occurrence_matrix.csv``), so the
  effect of the parser can be separated from the effect of the protocol.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from logfusion.detectors.pca import PCADetector
from logfusion.detectors.robust import RobustZ
from logfusion.evaluation import detection, grouping_accuracy
from logfusion.hdfs import BlockEvents, build_block_events, load_labels
from logfusion.parsing.miner import TemplateMiner

TRAIN_FRACTION = 0.5


def parse_or_load(data_dir: Path, cache: Path, log=print) -> tuple[BlockEvents, float]:
    """Build the block x template matrix from ``HDFS.log`` (or load it from ``cache``)."""
    if cache.exists():
        return BlockEvents.load(cache), 0.0
    started = time.perf_counter()
    with (data_dir / "HDFS.log").open(errors="replace") as f:
        events = build_block_events(f, progress=lambda n: log(f"  parsed {n:,} lines"))
    seconds = time.perf_counter() - started
    cache.parent.mkdir(parents=True, exist_ok=True)
    events.save(cache)
    return events, seconds


def sample_grouping_accuracy(structured_csv: Path) -> dict:
    """Grouping accuracy of a fresh miner on Loghub's hand-labelled 2k sample."""
    with structured_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    miner = TemplateMiner()
    predicted = [miner.add(r["Content"]).id for r in rows]
    truth = [r["EventId"] for r in rows]
    return {
        "lines": len(rows),
        "true_events": len(set(truth)),
        "mined_templates": len(miner.templates),
        "grouping_accuracy": grouping_accuracy(predicted, truth),
    }


def run(data_dir: Path, out_dir: Path, sample_csv: Path | None = None, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    events, parse_seconds = parse_or_load(data_dir, data_dir / "block_events.npz", log)
    labels_by_block = load_labels(data_dir / "preprocessed" / "anomaly_label.csv")
    missing = [b for b in events.blocks if b not in labels_by_block]
    if missing:
        raise ValueError(f"{len(missing)} blocks have no label, e.g. {missing[:3]}")
    y = np.array([labels_by_block[b] for b in events.blocks])
    x = events.counts.astype(float)

    split = int(len(y) * TRAIN_FRACTION)
    train, test = slice(0, split), slice(split, None)
    result: dict = {
        "parsing": {
            "lines": events.lines,
            "malformed_lines": events.malformed,
            "templates": len(events.templates),
            "seconds": round(parse_seconds, 1) if parse_seconds else None,
        },
        "blocks": {
            "total": len(y),
            "anomalous": int(y.sum()),
            "train": split,
            "train_anomalous": int(y[train].sum()),
            "test": len(y) - split,
            "test_anomalous": int(y[test].sum()),
        },
        "detectors": {},
    }
    if sample_csv is not None and sample_csv.exists():
        result["parsing"]["hdfs_2k_sample"] = sample_grouping_accuracy(sample_csv)

    variants = {
        "PCA (TF-IDF, 95% variance, alpha=0.001)": PCADetector(),
        "PCA (raw counts, 95% variance, alpha=0.001)": PCADetector(tfidf=False),
        "Robust z-score (max |z| > 3.5)": RobustZ(),
    }
    for name, detector in variants.items():
        detector.fit(x[train])
        scores = detector.score(x[test])
        predicted = detector.predict(x[test])
        entry = detection(y[test], predicted).as_dict()
        if isinstance(detector, PCADetector):
            entry["components"] = detector.n_components_
            entry["threshold"] = detector.threshold_
        entry["auc"] = roc_auc(y[test], scores)
        result["detectors"][name] = entry
        log(
            f"  {name}: precision={entry['precision']:.3f} recall={entry['recall']:.3f} "
            f"f1={entry['f1']:.3f}"
        )

    result["protocols"] = protocol_study(x, y, split, reference_matrix(data_dir, events.blocks))
    result["templates"] = events.templates
    (out_dir / "hdfs_results.json").write_text(json.dumps(result, indent=2) + "\n")
    (out_dir / "hdfs_results.md").write_text(render_markdown(result))
    return result


def reference_matrix(data_dir: Path, blocks: list[str]) -> np.ndarray | None:
    """Loghub's per-block counts of the 29 reference events, in ``blocks`` order."""
    path = data_dir / "preprocessed" / "Event_occurrence_matrix.csv"
    if not path.exists():
        return None
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        first_event = header.index("E1")
        rows = {r[0]: [float(v) for v in r[first_event:]] for r in reader}
    return np.array([rows[b] for b in blocks])


def protocol_study(x: np.ndarray, y: np.ndarray, split: int, reference: np.ndarray | None) -> list:
    feature_sets = {"this parser": x}
    if reference is not None:
        feature_sets["Loghub reference events"] = reference
    train, test = slice(0, split), slice(split, None)
    normal_train = np.flatnonzero(y[train] == 0)
    rows = []
    for variance in (0.90, 0.95, 0.99):
        rows.append((f"Unsupervised, temporal split, {variance:.0%} variance", variance, "split"))
    rows.append(("Unsupervised, fit and evaluate on all blocks, 95% variance", 0.95, "all"))
    rows.append(("Semi-supervised (normal training blocks only), 95% variance", 0.95, "normal"))
    study = []
    for label, variance, protocol in rows:
        entry: dict = {"protocol": label}
        for name, features in feature_sets.items():
            detector = PCADetector(variance=variance)
            if protocol == "all":
                detector.fit(features)
                d = detection(y, detector.predict(features))
            else:
                fit_rows = (
                    features[train][normal_train] if protocol == "normal" else features[train]
                )
                detector.fit(fit_rows)
                d = detection(y[test], detector.predict(features[test]))
            entry[name] = {**d.as_dict(), "components": detector.n_components_}
        study.append(entry)
    return study


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve (probability a random anomaly outscores a random normal block)."""
    y_true = np.asarray(y_true).astype(bool)
    positives, negatives = int(y_true.sum()), int((~y_true).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = np.asarray(scores)[order]
    i = 0
    while i < len(scores):  # average ranks over ties
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y_true].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def render_markdown(result: dict) -> str:
    p, b = result["parsing"], result["blocks"]
    lines = [
        "# HDFS_v1 results (real data)",
        "",
        "Data: Loghub HDFS_v1 (Xu et al., SOSP 2009). Blocks are ordered by their first log line; "
        "detectors are fitted on the first half without labels and evaluated on the second half "
        "against the real block labels.",
        "",
        "## Parsing",
        "",
        f"- Log lines: {p['lines']:,} ({p['malformed_lines']} malformed)",
        f"- Templates mined: {p['templates']}",
    ]
    if p.get("seconds"):
        lines.append(f"- Parsing time: {p['seconds']:.0f} s (single process)")
    sample = p.get("hdfs_2k_sample")
    if sample:
        lines.append(
            f"- Grouping accuracy on Loghub's hand-labelled HDFS_2k sample: "
            f"{sample['grouping_accuracy']:.4f} ({sample['mined_templates']} templates mined, "
            f"{sample['true_events']} true events)"
        )
    lines += [
        "",
        "## Blocks",
        "",
        f"- Total: {b['total']:,} ({b['anomalous']:,} anomalous, "
        f"{b['anomalous'] / b['total']:.2%})",
        f"- Training half: {b['train']:,} ({b['train_anomalous']:,} anomalous; labels not used)",
        f"- Test half: {b['test']:,} ({b['test_anomalous']:,} anomalous)",
        "",
        "## Detection on the test half",
        "",
        "| Detector | Precision | Recall | F1 | ROC AUC | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, d in result["detectors"].items():
        lines.append(
            f"| {name} | {d['precision']:.3f} | {d['recall']:.3f} | {d['f1']:.3f} | "
            f"{d['auc']:.3f} | {d['tp']:,} | {d['fp']:,} | {d['fn']:,} |"
        )
    study = result.get("protocols", [])
    if study:
        names = [k for k in study[0] if k != "protocol"]
        lines += [
            "",
            "## PCA protocol study",
            "",
            "Precision / recall / F1 (k = number of principal components kept). The first group "
            'of rows uses the same split as above; "all blocks" fits and evaluates on every '
            "block, as earlier HDFS studies did; the semi-supervised row uses the training labels "
            "to drop anomalous blocks before fitting.",
            "",
            "| Protocol | " + " | ".join(names) + " |",
            "|---|" + "---:|" * len(names),
        ]
        for row in study:
            cells = [
                f"{row[n]['precision']:.3f} / {row[n]['recall']:.3f} / **{row[n]['f1']:.3f}** "
                f"(k={row[n]['components']})"
                for n in names
            ]
            lines.append(f"| {row['protocol']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
