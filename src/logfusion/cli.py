"""Command-line entry point: ``logfusion {generate,synthetic,hdfs}``."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from logfusion.experiments import hdfs as hdfs_experiment
from logfusion.experiments import synthetic as synthetic_experiment
from logfusion.synthetic.generator import GeneratorConfig, generate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="logfusion", description="Log fusion, anomaly detection and root-cause ranking."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write one SYNTHETIC multi-service dataset")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--windows", type=int, default=GeneratorConfig.windows)
    gen.add_argument("--out", type=Path, required=True)

    syn = sub.add_parser("synthetic", help="fusion + root-cause experiment on synthetic data")
    syn.add_argument("--seeds", default="1-10", help='e.g. "1-10" or "1,2,3" (default: 1-10)')
    syn.add_argument("--work", type=Path, default=Path("data/synthetic"))
    syn.add_argument("--out", type=Path, default=Path("results/synthetic"))

    hd = sub.add_parser("hdfs", help="parsing + detection experiment on Loghub HDFS_v1")
    hd.add_argument("--data", type=Path, default=Path("data/HDFS_v1"))
    hd.add_argument("--out", type=Path, default=Path("results/hdfs"))
    hd.add_argument(
        "--sample",
        type=Path,
        default=Path("tests/data/loghub/HDFS_2k.log_structured.csv"),
        help="hand-labelled sample for parsing accuracy",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        config = replace(GeneratorConfig(), seed=args.seed, windows=args.windows)
        data = generate(args.out, config)
        print(
            f"wrote {len(data.spans):,} spans, {len(data.logs):,} log lines, "
            f"{len(data.metrics):,} metric values and {len(data.faults)} faults to {args.out}"
        )
    elif args.command == "synthetic":
        seeds = synthetic_experiment.parse_seeds(args.seeds)
        summary = synthetic_experiment.run(seeds, args.work, args.out)
        print(synthetic_experiment.render_markdown(summary))
    elif args.command == "hdfs":
        if not (args.data / "HDFS.log").exists():
            print(
                f"error: {args.data / 'HDFS.log'} not found; run scripts/download_data.sh first",
                file=sys.stderr,
            )
            return 2
        result = hdfs_experiment.run(args.data, args.out, args.sample)
        print(json.dumps(result["detectors"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
