import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from unit_validate import validate_result
import units

FIX = Path(__file__).resolve().parent / "fixtures"
JUNK_V21 = (FIX / "junk_v21_p80.txt").read_text()
GOOD = "A page of ordinary prose about boundary layers. " * 3

UNIT = {"unit_id": "19930001-p3", "ntrs_id": "19930001", "page": 3,
        "image_key": "k", "image_sha256": "x", "model_id": units.MODEL_ID,
        "prompt_sha": units.PROMPT_SHA, "config_hash": units.CONFIG_HASH}
META = {"config_hash": units.CONFIG_HASH, "config": {}, "server": "s",
        "image_px": [10, 10], "seconds": 1.0, "usage": {}, "finish_reason": "stop",
        "output_chars": len(GOOD),
        "loop_guard": {"ratio": None, "retried": False, "vision_failed": False,
                       "temperature_used": 0.0}}


def meta(**kw):
    m = copy.deepcopy(META)
    m.update(kw)
    return m


def test_accepts_good_result():
    assert validate_result(UNIT, GOOD, meta()) == (True, "")


def test_rejects_hash_not_accepted():
    assert validate_result(UNIT, GOOD, meta(config_hash="deadbeef0000")) == (False, "config_hash_mismatch")


def test_rejects_hash_differing_from_unit():
    u = dict(UNIT, config_hash="other")
    assert validate_result(u, GOOD, meta()) == (False, "config_hash_mismatch")


def test_rejects_empty_output():
    assert validate_result(UNIT, "  \n", meta()) == (False, "empty_output")


def test_rejects_degenerate_output():
    ok, reason = validate_result(UNIT, JUNK_V21, meta(finish_reason="length"))
    assert (ok, reason) == (False, "degenerate_output")


@pytest.mark.parametrize("bad", [None, [], "x", {}, {"config_hash": units.CONFIG_HASH},
                                 {"config_hash": units.CONFIG_HASH, "loop_guard": "no"}])
def test_rejects_malformed_meta(bad):
    assert validate_result(UNIT, GOOD, bad) == (False, "meta_malformed")


def test_vision_failed_result_is_valid():
    m = meta()
    m["loop_guard"]["vision_failed"] = True
    assert validate_result(UNIT, "", m) == (True, "")
    assert validate_result(UNIT, JUNK_V21, dict(m, finish_reason="length")) == (True, "")


def test_cli_exit_codes(tmp_path):
    (tmp_path / "u.json").write_text(json.dumps(UNIT))
    (tmp_path / "r.txt").write_text(GOOD)
    (tmp_path / "r.meta.json").write_text(json.dumps(META))
    script = Path(__file__).resolve().parent.parent / "pipeline" / "unit_validate.py"
    args = [sys.executable, str(script), str(tmp_path / "u.json"),
            str(tmp_path / "r.txt"), str(tmp_path / "r.meta.json")]
    assert subprocess.run(args).returncode == 0
    (tmp_path / "r.txt").write_text("")
    p = subprocess.run(args, capture_output=True, text=True)
    assert p.returncode == 1 and p.stdout.strip() == "empty_output"


def test_forged_stop_finish_reason_bypasses_junk_check():
    # DOCUMENTED TRUST BOUNDARY: pipe_table/runaway detection is length-gated, so a
    # worker forging finish_reason="stop" passes. Closed by CER quorum for untrusted
    # workers; changing this behavior must be a conscious decision.
    v1 = (FIX / "junk_v1_p80.txt").read_text()
    from guard import degenerate
    assert degenerate(v1, "stop")[0] is False  # precondition: zlib alone doesn't catch it
    assert validate_result(UNIT, v1, meta(finish_reason="stop")) == (True, "")
