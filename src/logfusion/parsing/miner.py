"""Online event-template mining by similarity clustering.

Messages are first bucketed by token count and by their first token (when that token
is a constant word), then each message is compared with the templates in its bucket.
Similarity is the fraction of positions where the template has the same token or a
wildcard. A message joins the most similar template if that fraction reaches
``threshold``; positions that disagree become wildcards. Otherwise it starts a new
template. Template ids never change, even when a template gains wildcards.

Messages that are identical after masking are cached and always go to the template
the first copy went to. Templates only ever gain wildcards, so that template still
matches them; the cache makes long logs (millions of lines) fast to process.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from logfusion.parsing.masking import DEFAULT_MASKS, Mask, mask, tokenize

WILDCARD = "<*>"


@dataclass
class Template:
    """One event type: its id, current token pattern and number of matched messages."""

    id: int
    tokens: list[str]
    count: int = 0

    @property
    def text(self) -> str:
        return " ".join(self.tokens)


@dataclass
class TemplateMiner:
    """Learns templates incrementally from a stream of messages."""

    threshold: float = 0.7
    masks: tuple[Mask, ...] = DEFAULT_MASKS
    templates: list[Template] = field(default_factory=list)
    _buckets: dict[tuple[int, str], list[Template]] = field(default_factory=dict, repr=False)
    _cache: dict[str, Template] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold}")

    def add(self, message: str) -> Template:
        """Assign ``message`` to a template, creating or generalising one if needed."""
        masked = mask(message, self.masks)
        cached = self._cache.get(masked)
        if cached is not None:
            cached.count += 1
            return cached
        tokens = tokenize(masked)
        bucket = self._buckets.setdefault(self._key(tokens), [])
        best, best_sim = self._best_match(bucket, tokens)
        if best is not None and best_sim >= self.threshold:
            best.tokens = [
                t if t == m else WILDCARD for t, m in zip(best.tokens, tokens, strict=True)
            ]
        else:
            best = Template(id=len(self.templates), tokens=list(tokens))
            self.templates.append(best)
            bucket.append(best)
        best.count += 1
        self._cache[masked] = best
        return best

    def match(self, message: str) -> Template | None:
        """Return the template that ``message`` would join, without learning from it."""
        tokens = tokenize(mask(message, self.masks))
        best, best_sim = self._best_match(self._buckets.get(self._key(tokens), []), tokens)
        return best if best is not None and best_sim >= self.threshold else None

    @staticmethod
    def _key(tokens: list[str]) -> tuple[int, str]:
        first = tokens[0] if tokens else ""
        constant = first and not first.startswith("<") and not any(c.isdigit() for c in first)
        return len(tokens), first if constant else WILDCARD

    @staticmethod
    def _best_match(bucket: list[Template], tokens: list[str]) -> tuple[Template | None, float]:
        if not tokens:
            return (bucket[0], 1.0) if bucket else (None, 0.0)
        best, best_sim = None, -1.0
        for template in bucket:
            same = sum(
                1 for t, m in zip(template.tokens, tokens, strict=True) if t in (m, WILDCARD)
            )
            sim = same / len(tokens)
            if sim > best_sim:
                best, best_sim = template, sim
        return best, best_sim
