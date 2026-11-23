import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import pass4a
from pass4a import pseudo_table_junk

FIX = Path(__file__).resolve().parent / "fixtures"
JUNK_V1 = (FIX / "junk_v1_p80.txt").read_text()
JUNK_V21 = (FIX / "junk_v21_p80.txt").read_text()
GOOD_1135 = (FIX / "good_table_1135_p37.txt").read_text()


def test_junk_fixtures_detected():
    assert pseudo_table_junk(JUNK_V1) is True
    assert pseudo_table_junk(JUNK_V21) is True


def test_genuine_dense_table_passes():
    assert pseudo_table_junk(GOOD_1135) is False


def test_prose_passes():
    assert pseudo_table_junk("A page of ordinary prose.\nNo pipes at all.") is False


def test_guard_config_bumped():
    assert pass4a.LOOP_GUARD["version"] == "v2"
    assert pass4a.LOOP_GUARD["pipe_line_floor"] == 0.4
    assert pass4a.LOOP_GUARD["unique_floor"] == 0.6


def test_degenerate_gates_pseudo_table_on_length():
    assert pass4a.degenerate(JUNK_V21, "length")[0] is True
    assert pass4a.degenerate(GOOD_1135, "length")[0] is False
    assert pass4a.degenerate(GOOD_1135, "stop")[0] is False


def test_degenerate_judges_pseudo_table_only_when_truncated():
    # junk-shaped but zlib-clean text: flagged at finish_reason == "length" only
    bad, _, _ = pass4a.degenerate(JUNK_V1, "length")
    assert bad is True


def test_junk_v1_not_degenerate_when_not_truncated():
    # zlib-clean (ratio ~0.36), so only the length-gated pipe rule can catch it
    assert pass4a.degenerate(JUNK_V1, "stop")[0] is False


def test_junk_v21_is_zlib_caught_and_pipe_rule_exercised_directly(monkeypatch):
    assert pass4a.zratio(JUNK_V21) < 0.05            # ~0.025: zlib alone catches it
    assert pass4a.degenerate(JUNK_V21, "stop")[0] is True
    # disable the zlib floors so only the length-gated pipe rule can fire
    monkeypatch.setitem(pass4a.LOOP_GUARD, "ratio_floor", 0.0)
    monkeypatch.setitem(pass4a.LOOP_GUARD, "length_ratio_floor", 0.0)
    assert pass4a.degenerate(JUNK_V21, "length")[0] is True
    assert pass4a.degenerate(JUNK_V21, "stop")[0] is False


def test_runaway_numeric_share_is_a_guard_param(monkeypatch):
    line = " ".join(["1.5"] * 800)                    # >2000 chars, all numeric
    text = "Intro words here.\n" + line
    assert pass4a.pseudo_table_junk(text) is True
    monkeypatch.setitem(pass4a.LOOP_GUARD, "runaway_numeric_share", 1.01)
    assert pass4a.pseudo_table_junk(text) is False


import subprocess, os
import guard as guard_mod


def test_degenerate_returns_zlib_reason():
    bad, ratio, reason = guard_mod.degenerate(JUNK_V21, "stop")
    assert bad is True and reason == "zlib" and ratio < 0.15


def test_degenerate_returns_runaway_reason():
    g = {**guard_mod.LOOP_GUARD, "ratio_floor": 0.0, "length_ratio_floor": 0.0}
    bad, ratio, reason = guard_mod.degenerate(JUNK_V1, "length", g)
    assert bad is True and reason == "runaway"
    assert guard_mod.junk_reason(JUNK_V1, g) == "runaway"


def test_junk_reason_pipe_table():
    g = {**guard_mod.LOOP_GUARD, "ratio_floor": 0.0, "length_ratio_floor": 0.0}
    assert guard_mod.junk_reason(JUNK_V21, g) == "pipe_table"
    assert guard_mod.junk_reason("plain prose\nonly") is None


def test_clean_text_has_no_reason():
    bad, ratio, reason = guard_mod.degenerate(GOOD_1135, "length")
    assert (bad, reason) == (False, None) and ratio is not None


def test_pass4a_reexports_guard_objects():
    assert pass4a.LOOP_GUARD is guard_mod.LOOP_GUARD
    assert pass4a.degenerate is guard_mod.degenerate


def test_guard_imports_standalone_without_env():
    root = Path(__file__).resolve().parent.parent / "pipeline"
    out = subprocess.run([sys.executable, "-c", "import guard, sys; print('common' in sys.modules)"],
                         cwd=root, env={"PATH": os.environ.get("PATH", "")},
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_config_hash_unchanged_after_extraction():
    root = Path(__file__).resolve().parent.parent
    env = {**os.environ, "NACA_VLM_MODEL": "qwen3-vl-8b-lora-v2_1",
           "NACA_VLM_PROMPT_FILE": "pipeline/prompts/transcribe_v2.txt"}
    out = subprocess.run([sys.executable, "-c",
                          "import sys; sys.path.insert(0,'pipeline'); import pass4a; print(pass4a.CONFIG_HASH)"],
                         cwd=root, env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "6ecf7d0dbeb2"


def test_min_chars_override_via_g_is_honored():
    short = "abc" * 100                         # 300 chars: below default min_chars (400)
    assert guard_mod.degenerate(short, "stop") == (False, None, None)
    g = {**guard_mod.LOOP_GUARD, "min_chars": 50}
    bad, ratio, reason = guard_mod.degenerate(short, "stop", g)
    assert ratio is not None and bad is True and reason == "zlib"
    assert guard_mod.zratio(short, g) is not None and guard_mod.zratio(short) is None
