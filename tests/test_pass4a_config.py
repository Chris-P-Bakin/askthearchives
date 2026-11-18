import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PROBE = (
    "import json, pass4a; "
    "print(json.dumps({'hash': pass4a.CONFIG_HASH, "
    "'think_key': pass4a.CONFIG.get('enable_thinking', 'absent')}))"
)


def _probe(**env_extra):
    import os
    env = {**os.environ,
           "NACA_VLM_MODEL": "Qwen3.8-27B-UD-Q4_K_M",
           "NACA_VLM_PROMPT_FILE": "prompts/transcribe_v2.txt",
           "NACA_VLM_THINK": "1",
           **env_extra}
    out = subprocess.run(
        [sys.executable, "-c", PROBE], cwd=ROOT / "pipeline",
        env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_every_loop_guard_value_pinned():
    sys.path.insert(0, str(ROOT / "pipeline"))
    # all detector thresholds live in LOOP_GUARD and so in CONFIG_HASH
    import pass4a
    assert pass4a.LOOP_GUARD == {
        "version": "v2", "min_chars": 400, "ratio_floor": 0.15,
        "length_ratio_floor": 0.25, "retry_temperature": 0.4,
        "pipe_line_floor": 0.4, "unique_floor": 0.6,
        "runaway_line_chars": 2000, "runaway_line_share": 0.5,
        "runaway_numeric_share": 0.95,
    }


def test_think_default_keeps_v2_guard_hash():
    # default (thinking on) must not add the key, so hashes minted under guard v2 stand (v2 re-pinned for runaway_numeric_share; earlier: v1 f86c00c0ef5d, v2-initial b29ebd12ea68)
    r = _probe()
    assert r["think_key"] == "absent"
    assert r["hash"] == "6a169a212aac"


def test_no_think_changes_hash():
    r = _probe(NACA_VLM_THINK="0")
    assert r["think_key"] is False
    assert r["hash"] != "6a169a212aac"
