"""Replace obviously variable tokens (ids, addresses, numbers) with placeholders.

Masking before template mining keeps the miner from splitting one event type into
many templates just because a block id or an IP address differs. Masks are applied
in order, so more specific patterns (block ids, IP:port) run before generic numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Mask:
    """A named regular expression and the placeholder that replaces its matches."""

    name: str
    pattern: re.Pattern[str]
    placeholder: str


def _mask(name: str, regex: str) -> Mask:
    return Mask(name, re.compile(regex), f"<{name.upper()}>")


DEFAULT_MASKS: tuple[Mask, ...] = (
    _mask("blk", r"blk_-?\d+"),
    _mask("ip", r"/?\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"),
    _mask("hex", r"\b0x[0-9a-fA-F]+\b"),
    _mask("num", r"(?<![\w.<>])-?\d+(?:\.\d+)?(?![\w.<>])"),
)


def mask(message: str, masks: tuple[Mask, ...] = DEFAULT_MASKS) -> str:
    """Return ``message`` with every match of every mask replaced by its placeholder."""
    for m in masks:
        message = m.pattern.sub(m.placeholder, message)
    return message


def tokenize(message: str) -> list[str]:
    """Split a (masked) message into whitespace-separated tokens."""
    return message.split()
