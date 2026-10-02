from docuparser.ingestion.chunking import split_text


def test_short_text_single_chunk():
    assert split_text("hello world", 100, 10) == ["hello world"]


def test_long_text_respects_size_and_overlaps():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_text(text, 200, 40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    assert "word0" in chunks[0] and "word499" in chunks[-1]


def test_empty():
    assert split_text("   ") == []
