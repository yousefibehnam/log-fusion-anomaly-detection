"""Loading the HDFS_v1 console log (Xu et al., 2009) as per-block event-count vectors.

Each HDFS log line names one or more data blocks. Following the original study, all
lines that mention a block form that block's session, and a session is represented by
how often each event template occurs in it. The ground-truth labels are per block.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from logfusion.parsing.miner import TemplateMiner

LINE = re.compile(
    r"^(?P<date>\d{6}) (?P<time>\d{6}) (?P<pid>\d+) (?P<level>\w+) "
    r"(?P<component>[^:]+): (?P<content>.*)$"
)
BLOCK = re.compile(r"blk_-?\d+")


@dataclass(frozen=True)
class LogRecord:
    date: str
    time: str
    pid: int
    level: str
    component: str
    content: str


def parse_line(line: str) -> LogRecord | None:
    """Split an HDFS log line into its header fields and message; ``None`` if malformed."""
    m = LINE.match(line.rstrip("\n"))
    if m is None:
        return None
    return LogRecord(m["date"], m["time"], int(m["pid"]), m["level"], m["component"], m["content"])


def block_ids(content: str) -> list[str]:
    """Distinct block ids mentioned in a message, in order of appearance."""
    return list(dict.fromkeys(BLOCK.findall(content)))


@dataclass
class BlockEvents:
    """Event counts: one row per block (in order of first appearance), one column per template."""

    blocks: list[str]
    counts: np.ndarray
    templates: list[str]
    lines: int
    malformed: int

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            blocks=np.array(self.blocks),
            counts=self.counts,
            templates=np.array(self.templates),
            stats=np.array([self.lines, self.malformed]),
        )

    @classmethod
    def load(cls, path: Path) -> BlockEvents:
        with np.load(path) as data:
            lines, malformed = (int(x) for x in data["stats"])
            return cls(
                list(data["blocks"]), data["counts"], list(data["templates"]), lines, malformed
            )


def build_block_events(
    lines: Iterable[str],
    miner: TemplateMiner | None = None,
    progress: Callable[[int], None] | None = None,
    progress_every: int = 1_000_000,
) -> BlockEvents:
    """Parse log lines, mine templates and count template occurrences per block."""
    miner = miner or TemplateMiner()
    per_block: dict[str, Counter[int]] = {}
    n = malformed = 0
    for line in lines:
        n += 1
        record = parse_line(line)
        if record is None:
            malformed += 1
            continue
        template = miner.add(record.content)
        for block in block_ids(record.content):
            per_block.setdefault(block, Counter())[template.id] += 1
        if progress is not None and n % progress_every == 0:
            progress(n)

    blocks = list(per_block)
    counts = np.zeros((len(blocks), len(miner.templates)), dtype=np.int32)
    for row, block in enumerate(blocks):
        for template_id, c in per_block[block].items():
            counts[row, template_id] = c
    return BlockEvents(blocks, counts, [t.text for t in miner.templates], n, malformed)


def load_labels(path: Path) -> dict[str, int]:
    """Read ``anomaly_label.csv``: 1 for ``Anomaly``, 0 for ``Normal``."""
    with path.open(newline="") as f:
        return {row["BlockId"]: int(row["Label"] == "Anomaly") for row in csv.DictReader(f)}
