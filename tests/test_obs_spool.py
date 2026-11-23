import json

from obs_spool import read_spool


def _line(obj) -> bytes:
    return json.dumps(obj).encode() + b"\n"


def test_reads_new_lines_and_persists_offset(tmp_path):
    spool = tmp_path / "spool.ndjson"
    off = tmp_path / "offset"
    spool.write_bytes(_line({"n": 1}) + _line({"n": 2}))

    assert read_spool(spool, off) == [{"n": 1}, {"n": 2}]
    assert int(off.read_text()) == spool.stat().st_size

    with spool.open("ab") as f:
        f.write(_line({"n": 3}))
    assert read_spool(spool, off) == [{"n": 3}]
    assert read_spool(spool, off) == []


def test_partial_line_skipped_offset_advances(tmp_path):
    spool = tmp_path / "spool.ndjson"
    off = tmp_path / "offset"
    complete = _line({"a": 1})
    partial = b'{"b":'
    spool.write_bytes(complete + partial)

    assert read_spool(spool, off) == [{"a": 1}]
    # Offset stops at the end of the last complete line; the partial is kept.
    assert int(off.read_text()) == len(complete)

    with spool.open("ab") as f:
        f.write(b"2}\n")
    assert read_spool(spool, off) == [{"b": 2}]
    assert read_spool(spool, off) == []


def test_complete_garbage_line_skipped(tmp_path):
    spool = tmp_path / "spool.ndjson"
    off = tmp_path / "offset"
    garbage = b"this is not json\n"
    spool.write_bytes(_line({"a": 1}) + garbage + _line({"b": 2}))

    assert read_spool(spool, off) == [{"a": 1}, {"b": 2}]
    # The complete bad line is consumed, so the offset reaches end of file.
    assert int(off.read_text()) == spool.stat().st_size
    assert read_spool(spool, off) == []


def test_stale_offset_resets(tmp_path):
    spool = tmp_path / "spool.ndjson"
    off = tmp_path / "offset"
    spool.write_bytes(b'{"ab":12}\n')  # exactly 10 bytes
    assert spool.stat().st_size == 10
    off.write_text("9999")

    assert read_spool(spool, off) == [{"ab": 12}]


def test_rotation_truncates_at_cap(tmp_path):
    spool = tmp_path / "spool.ndjson"
    off = tmp_path / "offset"
    spool.write_bytes(_line({"n": 1}) + _line({"n": 2}))

    assert read_spool(spool, off, rotate_bytes=1) == [{"n": 1}, {"n": 2}]
    assert spool.stat().st_size == 0
    assert int(off.read_text()) == 0

    with spool.open("ab") as f:
        f.write(_line({"n": 3}))
    assert read_spool(spool, off, rotate_bytes=1) == [{"n": 3}]


def test_missing_spool_empty(tmp_path):
    assert read_spool(tmp_path / "absent.ndjson", tmp_path / "offset") == []


def test_corrupt_offset_file_treated_as_zero(tmp_path):
    spool = tmp_path / "spool.ndjson"
    off = tmp_path / "offset"
    spool.write_bytes(_line({"a": 1}))
    off.write_text("not-a-number")

    assert read_spool(spool, off) == [{"a": 1}]
