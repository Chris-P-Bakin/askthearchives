"""Section-aware chunking of a report's stitched page text.

Pure functions, no I/O. Pages are stitched with "\\n" before splitting so that
paragraphs and sections that cross a page break stay whole. Chunk offsets index
into that stitched text: ``"\\n".join(text.rstrip("\\n") for _, text in pages)``.
"""
import re
from dataclasses import dataclass
from typing import Callable

HEADING_MIN_CHARS = 3
HEADING_MAX_CHARS = 80
HEADING_MIN_UPPER_RATIO = 0.95


@dataclass
class Chunk:
    pages: list[int]
    section: str
    text: str
    char_start: int
    char_end: int


def is_heading(line: str) -> bool:
    """ALL-CAPS section-heading line: short, mostly letters, (nearly) no lowercase."""
    s = line.strip()
    if not HEADING_MIN_CHARS <= len(s) <= HEADING_MAX_CHARS:
        return False
    alpha = [ch for ch in s if ch.isalpha()]
    if len(alpha) < HEADING_MIN_CHARS:
        return False
    return sum(ch.isupper() for ch in alpha) / len(alpha) >= HEADING_MIN_UPPER_RATIO


def _paragraphs(doc: str) -> list[tuple[int, int, bool]]:
    """(start, end, starts_with_heading) spans. Blank lines and heading lines end a paragraph."""
    paras: list[tuple[int, int, bool]] = []
    start = end = None
    heading = False
    pos = 0
    for line in doc.split("\n"):
        line_start, pos = pos, pos + len(line) + 1
        stripped = line.strip()
        if not stripped:
            if start is not None:
                paras.append((start, end, heading))
                start = None
            continue
        line_end = line_start + len(line.rstrip())
        if start is not None and is_heading(line):
            paras.append((start, end, heading))
            start = None
        if start is None:
            start = line_start + (len(line) - len(line.lstrip()))
            heading = is_heading(line)
        end = line_end
    if start is not None:
        paras.append((start, end, heading))
    return paras


def _hard_split(doc: str, start: int, end: int, max_tokens: int,
                count_tokens: Callable[[str], int]) -> list[tuple[int, int]]:
    """Split [start, end) into spans of <= max_tokens, preferring sentence then whitespace
    boundaries near the middle. A run with no whitespace is returned whole."""
    if count_tokens(doc[start:end]) <= max_tokens:
        return [(start, end)]
    mid = (start + end) // 2
    lo, hi = start + (end - start) // 4, end - (end - start) // 4
    window = doc[lo:hi]
    cut = None
    best = None
    for m in re.finditer(r"[.!?]\s+", window):
        if best is None or abs(lo + m.end() - mid) < abs(best - mid):
            best = lo + m.end()
    if best is not None:
        cut = best
    else:
        for m in re.finditer(r"\s+", window):
            if cut is None or abs(lo + m.start() - mid) < abs(cut - mid):
                cut = lo + m.start()
    if cut is None:
        return [(start, end)]
    left_end = len(doc[start:cut].rstrip()) + start
    right_start = cut + (len(doc[cut:end]) - len(doc[cut:end].lstrip()))
    if left_end <= start or right_start >= end:
        return [(start, end)]
    return (_hard_split(doc, start, left_end, max_tokens, count_tokens)
            + _hard_split(doc, right_start, end, max_tokens, count_tokens))


def chunk_report(pages: list[tuple[int, str]], title: str, target_words: int = 200,
                 max_tokens: int = 2000,
                 count_tokens: Callable[[str], int] | None = None) -> list[Chunk]:
    count = count_tokens or (lambda t: len(t.split()))
    # Trailing newlines are stripped so a page ending in "\n" doesn't fake a paragraph break.
    pages = [(n, text.rstrip("\n")) for n, text in pages]
    doc = "\n".join(text for _, text in pages)

    spans, pos = [], 0
    for number, text in pages:
        if text:
            spans.append((number, pos, pos + len(text)))
        pos += len(text) + 1

    def pages_for(s: int, e: int) -> list[int]:
        return sorted({n for n, ps, pe in spans if ps < e and pe > s})

    chunks: list[Chunk] = []

    def emit(s: int, e: int, section: str) -> None:
        chunks.append(Chunk(pages_for(s, e), section, doc[s:e], s, e))

    section = title
    group: list[tuple[int, int]] = []   # paragraph spans being packed
    group_words = group_tokens = 0

    def flush() -> None:
        nonlocal group, group_words, group_tokens
        if group:
            emit(group[0][0], group[-1][1], group_section)
        group, group_words, group_tokens = [], 0, 0

    group_section = section
    for p_start, p_end, is_head in _paragraphs(doc):
        if is_head:
            flush()
            section = doc[p_start:p_end].split("\n", 1)[0].strip()
        text = doc[p_start:p_end]
        words, tokens = len(text.split()), count(text)
        if tokens > max_tokens:
            flush()
            for s, e in _hard_split(doc, p_start, p_end, max_tokens, count):
                emit(s, e, section)
            continue
        if group and (group_words + words > target_words or group_tokens + tokens > max_tokens):
            flush()
        if not group:
            group_section = section
        group.append((p_start, p_end))
        group_words += words
        group_tokens += tokens
    flush()
    return chunks
