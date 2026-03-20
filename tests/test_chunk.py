from app.ingest.chunk import chunk_text


def test_empty_and_whitespace_yield_no_chunks():
    assert chunk_text("", 800, 120) == []
    assert chunk_text("    \n\t  ", 800, 120) == []


def test_short_text_is_single_chunk():
    assert chunk_text("hello world", 800, 120) == ["hello world"]


def test_whitespace_is_normalized():
    assert chunk_text("hello   \n  world", 800, 120) == ["hello world"]


def test_long_text_splits_into_multiple_chunks():
    text = " ".join(f"token{i}" for i in range(1000))
    chunks = chunk_text(text, 200, 40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_adjacent_chunks_overlap():
    text = " ".join(f"token{i}" for i in range(500))
    chunks = chunk_text(text, 120, 40)
    assert len(chunks) > 1
    for a, b in zip(chunks, chunks[1:], strict=False):
        # With a 40-char overlap, consecutive chunks must share at least one token.
        assert set(a.split()) & set(b.split()), "expected overlapping tokens between chunks"


def test_chunking_is_deterministic():
    text = " ".join(f"token{i}" for i in range(500))
    assert chunk_text(text, 200, 40) == chunk_text(text, 200, 40)


def test_all_tokens_are_covered():
    text = " ".join(f"token{i}" for i in range(300))
    chunks = chunk_text(text, 150, 30)
    covered = set()
    for c in chunks:
        covered |= set(c.split())
    assert covered == set(text.split())
