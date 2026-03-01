"""Deterministic, word-boundary-aware text chunker with character overlap."""

from __future__ import annotations

import re

_WHITESPACE = re.compile(r"\s+")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Split ``text`` into overlapping chunks of at most ``chunk_size`` characters.

    - Whitespace is normalized to single spaces first.
    - Chunk boundaries snap back to the last space so words aren't split at the end,
      and the overlap start snaps forward to a space so chunks begin on a whole word.
    - Adjacent chunks share up to ``overlap`` characters to preserve context across cuts.
    - Deterministic: identical input always yields identical output.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    text = _WHITESPACE.sub(" ", text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        # Back up by ``overlap``, then snap forward to the next space so the
        # next chunk begins on a whole word (never a fragment). Progress is
        # guaranteed because ``end`` is strictly greater than ``start``.
        nxt = max(end - overlap, start + 1)
        space = text.find(" ", nxt, end)
        if space != -1:
            nxt = space + 1
        start = nxt
    return chunks
