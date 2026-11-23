import json

import batch_validate as bv

GOOD_TEXT = "A page of ordinary prose about boundary layers. " * 12 + "".join(
    f" word{i}x{i * 7 % 13}" for i in range(80))
HASH = "6ecf7d0dbeb2"


def manifest(n=3):
    return [{"unit_id": f"19930001-p{i}", "ntrs_id": "19930001", "page": i,
             "config_hash": HASH} for i in range(n)]


def ok_line(uid, h=HASH):
    return json.dumps({"unit_id": uid, "status": "ok", "text": GOOD_TEXT,
                       "meta": {"config_hash": h, "loop_guard": {"vision_failed": False},
                                "finish_reason": "stop"}})


def err_line(uid):
    return json.dumps({"unit_id": uid, "status": "error", "error": "server_http_404"})


def test_valid_batch_passes():
    m = manifest()
    ok, reason = bv.validate_batch(m, [ok_line(u["unit_id"]) for u in m])
    assert ok, reason


def test_unknown_hash_anywhere_invalidates_whole():
    m = manifest()
    lines = [ok_line(m[0]["unit_id"]), ok_line(m[1]["unit_id"], h="deadbeefdead"),
             ok_line(m[2]["unit_id"])]
    ok, reason = bv.validate_batch(m, lines)
    assert not ok and "config_hash" in reason


def test_malformed_json_line_invalidates_whole():
    m = manifest()
    lines = [ok_line(m[0]["unit_id"]), "{not json", ok_line(m[2]["unit_id"])]
    ok, reason = bv.validate_batch(m, lines)
    assert not ok and "malformed" in reason


def test_missing_page_without_error_line_invalidates():
    m = manifest()
    ok, reason = bv.validate_batch(m, [ok_line(m[0]["unit_id"]), ok_line(m[1]["unit_id"])])
    assert not ok and "missing" in reason


def test_honest_error_line_keeps_batch_valid():
    m = manifest()
    lines = [ok_line(m[0]["unit_id"]), err_line(m[1]["unit_id"]), ok_line(m[2]["unit_id"])]
    ok, reason = bv.validate_batch(m, lines)
    assert ok, reason


def test_duplicate_unit_id_invalidates():
    m = manifest(2)
    ok, reason = bv.validate_batch(m, [ok_line(m[0]["unit_id"])] * 2)
    assert not ok and "malformed" in reason


def test_non_dict_json_line_invalidates_not_crashes():
    m = manifest(2)
    ok, reason = bv.validate_batch(m, [ok_line(m[0]["unit_id"]), "5"])
    assert not ok and "malformed" in reason
