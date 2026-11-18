import json

from agreement import page_agreement, collect_pages, bucket


def test_page_agreement_identical_is_zero():
    r = page_agreement("The quick brown fox.", "The  quick brown fox.")
    assert r["cer"] == 0.0


def test_page_agreement_known_edit():
    # normalized texts differ by exactly 1 char over max length 4
    r = page_agreement("abcd", "abed")
    assert r["cer"] == 0.25
    assert r["a_chars"] == 4 and r["b_chars"] == 4


def test_page_agreement_both_empty():
    r = page_agreement("", "")
    assert r["cer"] == 0.0


def test_bucket_thresholds():
    assert bucket(0.01) == "agree"
    assert bucket(0.05) == "minor"
    assert bucket(0.20) == "conflict"


def _mk_page(rdir, vdir, n, text, failed=False):
    d = rdir / vdir
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{n}.txt").write_text(text)
    (d / f"{n}.meta.json").write_text(json.dumps(
        {"config_hash": "x", "loop_guard": {"vision_failed": failed}}))


def test_collect_pages_skips_missing_and_failed(tmp_path):
    rdir = tmp_path / "99999"
    _mk_page(rdir, "vision_27bq4", 1, "page one a")
    _mk_page(rdir, "vision_haiku2", 1, "page one b")
    _mk_page(rdir, "vision_27bq4", 2, "only in a")
    _mk_page(rdir, "vision_27bq4", 3, "page three a")
    _mk_page(rdir, "vision_haiku2", 3, "garbage loop", failed=True)
    pages = list(collect_pages(tmp_path))
    assert [(p[0], p[1]) for p in pages] == [("99999", 1)]
    assert pages[0][2] == "page one a" and pages[0][3] == "page one b"


def test_run_custom_dirs(tmp_path):
    from agreement import run
    rdir = tmp_path / "doc1"
    _mk_page(rdir, "vision_8blora", 1, "same text here")
    _mk_page(rdir, "vision_27bq4", 1, "same text here")
    out = tmp_path / "agr.jsonl"
    s = run(tmp_path, out, 0.02, 0.10, dir_a="vision_8blora", dir_b="vision_27bq4")
    assert s["pages"] == 1
    header = json.loads(out.read_text().splitlines()[0])
    assert header["dir_a"] == "vision_8blora"
    assert header["dir_b"] == "vision_27bq4"


def test_resolve_out_guards_canonical_file(tmp_path):
    from pathlib import Path
    from agreement import resolve_out, DIR_A, DIR_B, DATA
    # explicit --out always wins
    assert resolve_out("x/y.jsonl", "a", "b") == Path("x/y.jsonl")
    # default dirs -> canonical file
    assert resolve_out(None, DIR_A, DIR_B) == DATA / "agreement.jsonl"
    # custom dirs with no --out must NOT clobber the canonical file
    p = resolve_out(None, "vision_8blora", "vision_27bq4")
    assert p == DATA / "agreement_vision_8blora__vision_27bq4.jsonl"
