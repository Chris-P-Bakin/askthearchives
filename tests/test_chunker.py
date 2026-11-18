from chunker import Chunk, chunk_report, is_heading


def _stitch(pages):
    return "\n".join(t.rstrip("\n") for _, t in pages)


def _words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_paragraph_crossing_page_break_lands_in_one_chunk():
    pages = [(1, "Intro text.\n\nThe flow separated near the"), (2, "trailing edge of the wing.\n\nNext.")]
    chunks = chunk_report(pages, "T")
    crossing = [c for c in chunks if "near the\ntrailing edge" in c.text]
    assert len(crossing) == 1
    assert crossing[0].pages == [1, 2]


def test_trailing_newlines_do_not_split_cross_page_paragraph():
    pages = [(1, "The wing stall was continued on\n"), (2, "the next page.\n")]
    chunks = chunk_report(pages, "T")
    assert len(chunks) == 1
    assert chunks[0].pages == [1, 2]
    assert chunks[0].text == "The wing stall was continued on\nthe next page."


def test_heading_starts_new_section():
    pages = [(1, "Preamble words here.\n\nINTRODUCTION\n\nBody one.\n\nBody two.")]
    chunks = chunk_report(pages, "T")
    intro = [c for c in chunks if "Body one." in c.text]
    assert intro and intro[0].section == "INTRODUCTION"
    assert chunks[0].section != "INTRODUCTION"


def test_packs_to_target_words():
    paras = "\n\n".join(_words(40, f"p{i}x") for i in range(50))
    chunks = chunk_report([(1, paras)], "T", target_words=200)
    counts = [len(c.text.split()) for c in chunks]
    assert len(chunks) < 50
    assert all(c <= 200 for c in counts)
    assert max(counts) >= 160


def test_giant_paragraph_hard_split_by_token_budget():
    count = lambda t: len(t.split())
    chunks = chunk_report([(1, _words(12000))], "T", max_tokens=300, count_tokens=count)
    assert len(chunks) > 1
    assert all(count(c.text) <= 300 for c in chunks)


def test_offsets_roundtrip():
    pages = [(1, "A para.\n\nINTRO\n\n" + _words(500)), (3, ""), (4, _words(60) + "\n\nLast one.")]
    stitched = _stitch(pages)
    chunks = chunk_report(pages, "T", max_tokens=100)
    assert chunks
    for c in chunks:
        assert stitched[c.char_start:c.char_end] == c.text


def test_empty_inputs_do_not_crash():
    assert chunk_report([], "T") == []
    assert chunk_report([(1, ""), (2, "  \n ")], "T") == []


def test_pages_sorted_and_skip_empty_pages():
    pages = [(5, "alpha beta"), (6, ""), (7, "gamma delta")]
    chunks = chunk_report(pages, "T")
    assert chunks[0].pages == [5, 7]


def test_is_heading():
    assert is_heading("APPARATUS AND METHODS")
    assert not is_heading("The flow was turbulent.")
    assert not is_heading("A")
    assert not is_heading("TABLE II" * 20)
    assert not is_heading("0.123 0.456 0.789")
