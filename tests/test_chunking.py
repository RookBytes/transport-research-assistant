from transport_rag.ingestion.chunking import chunk_text


def test_chunk_text_preserves_line_ranges():
    text = "\n".join(f"line {i}" for i in range(1, 31))
    chunks = chunk_text(text, "sample.txt", target_chars=60, overlap_lines=2)

    assert len(chunks) > 1
    assert chunks[0].start_line == 1
    assert chunks[0].end_line >= chunks[0].start_line
    assert all(c.source_path == "sample.txt" for c in chunks)


def test_empty_text_has_no_chunks():
    assert chunk_text("", "sample.txt") == []
