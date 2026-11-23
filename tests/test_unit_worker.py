import io, json, struct, sys, zlib
from pathlib import Path
import urllib.error

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "worker"))
import unit_worker as uw
import units
from unit_validate import validate_result

GOOD = "A page of ordinary prose about boundary layers. " * 12 + "".join(
    f" word{i}x{i*7%13}" for i in range(80))
JUNK = "x " * 600


def png(w=8, h=6):
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d))
    raw = b"".join(b"\x00" + b"\x00" * w for _ in range(h))
    return (uw.PNG_MAGIC + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


UNIT = {"unit_id": "1-p1", "ntrs_id": "1", "page": 1, "image_key": "k", "image_sha256": "x",
        "model_id": units.MODEL_ID, "prompt_sha": units.PROMPT_SHA, "config_hash": units.CONFIG_HASH}


class FakeResp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


def fake_server(monkeypatch, replies):
    """replies: list of (text, finish) or Exception; records request bodies."""
    seen = []
    def urlopen(req, timeout=None):
        seen.append(json.loads(req.data))
        r = replies[min(len(seen) - 1, len(replies) - 1)]
        if isinstance(r, Exception):
            raise r
        text, fin = r
        return FakeResp(json.dumps({"choices": [{"message": {"content": text}, "finish_reason": fin}],
                                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode())
    monkeypatch.setattr(uw.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(uw.time, "sleep", lambda s: None)
    return seen


def test_config_hash_matches_campaign():
    assert uw.CONFIG_HASH == units.CONFIG_HASH


def test_happy_path_validates(monkeypatch):
    seen = fake_server(monkeypatch, [(GOOD, "stop")])
    text, meta = uw.process_unit(UNIT, png(), "http://s", uw.PROMPT)
    assert text == GOOD.strip()
    assert validate_result(UNIT, text, meta) == (True, "")
    assert meta["loop_guard"]["retried"] is False and meta["worker"]
    b = seen[0]
    assert b["temperature"] == 0.0 and b["max_tokens"] == 12288 and b["dry_multiplier"] == 0.8
    assert b["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert meta["image_px"] == [8, 6]


def test_loop_guard_retry(monkeypatch):
    seen = fake_server(monkeypatch, [(JUNK, "stop"), (GOOD, "stop")])
    text, meta = uw.process_unit(UNIT, png(), "http://s", uw.PROMPT)
    assert meta["loop_guard"]["retried"] is True and meta["loop_guard"]["vision_failed"] is False
    assert meta["loop_guard"]["temperature_used"] == 0.4 and seen[1]["temperature"] == 0.4
    assert text == GOOD.strip()
    assert validate_result(UNIT, text, meta)[0]


def test_guard_failed_still_valid(monkeypatch):
    fake_server(monkeypatch, [(JUNK, "stop"), (JUNK, "stop")])
    text, meta = uw.process_unit(UNIT, png(), "http://s", uw.PROMPT)
    assert meta["loop_guard"]["vision_failed"] is True
    assert validate_result(UNIT, text, meta) == (True, "")


def test_server_error_is_structured_failure(monkeypatch, tmp_path):
    fake_server(monkeypatch, [urllib.error.URLError("refused")])
    with pytest.raises(uw.UnitError):
        uw.process_unit(UNIT, png(), "http://s", uw.PROMPT)
    fake_server(monkeypatch, [urllib.error.HTTPError("u", 400, "bad", {}, None)])
    with pytest.raises(uw.UnitError) as e:
        uw.process_unit(UNIT, png(), "http://s", uw.PROMPT)
    assert "400" in e.value.reason
    # CLI: error result, exit 2, no result.txt
    (tmp_path / "u.json").write_text(json.dumps(UNIT))
    (tmp_path / "i.png").write_bytes(png())
    fake_server(monkeypatch, [urllib.error.URLError("refused")])
    rc = uw.run_file_mode(tmp_path / "u.json", tmp_path / "i.png", tmp_path / "out", "http://s")
    assert rc == 2
    assert "error" in json.loads((tmp_path / "out/result.meta.json").read_text())
    assert not (tmp_path / "out/result.txt").exists()


def test_truncated_image_is_structured_failure(monkeypatch):
    seen = fake_server(monkeypatch, [(GOOD, "stop")])
    for bad in (png()[:-20], b"notapng" * 10, png()[:20]):
        with pytest.raises(uw.UnitError):
            uw.process_unit(UNIT, bad, "http://s", uw.PROMPT)
    assert seen == []   # never sent


def test_file_mode_success(monkeypatch, tmp_path):
    fake_server(monkeypatch, [(GOOD, "stop")])
    import hashlib
    unit = dict(UNIT, image_sha256=hashlib.sha256(png()).hexdigest())
    (tmp_path / "u.json").write_text(json.dumps(unit))
    (tmp_path / "i.png").write_bytes(png())
    assert uw.run_file_mode(tmp_path / "u.json", tmp_path / "i.png", tmp_path / "out", "http://s") == 0
    meta = json.loads((tmp_path / "out/result.meta.json").read_text())
    assert validate_result(UNIT, (tmp_path / "out/result.txt").read_text(), meta)[0]


def test_empty_output_is_structured_failure(monkeypatch, tmp_path):
    import hashlib
    fake_server(monkeypatch, [("", "stop")])
    with pytest.raises(uw.UnitError) as e:
        uw.process_unit(UNIT, png(), "http://s", uw.PROMPT)
    assert e.value.reason == "empty_output"
    unit = dict(UNIT, image_sha256=hashlib.sha256(png()).hexdigest())
    (tmp_path / "u.json").write_text(json.dumps(unit))
    (tmp_path / "i.png").write_bytes(png())
    assert uw.run_file_mode(tmp_path / "u.json", tmp_path / "i.png", tmp_path / "out", "http://s") == 2
    assert not (tmp_path / "out/result.txt").exists()


def test_worker_config_equals_pass4a_config():
    import os, subprocess
    env = {**os.environ, "NACA_VLM_MODEL": "qwen3-vl-8b-lora-v2_1",
           "NACA_VLM_PROMPT_FILE": "pipeline/prompts/transcribe_v2.txt"}
    for k in ("NACA_VLM_THINK", "NACA_VLM_MAX_PX", "NACA_VLM_DRY"):
        env.pop(k, None)
    out = subprocess.run([sys.executable, "-c",
                          "import sys, json; sys.path.insert(0,'pipeline'); import pass4a; "
                          "print(json.dumps([pass4a.CONFIG, pass4a.CONFIG_HASH]))"],
                         cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    cfg, h = json.loads(out.stdout)
    assert uw.CONFIG == cfg
    assert uw.CONFIG_HASH == h


def test_missing_prompt_file_is_fatal(tmp_path):
    with pytest.raises(SystemExit):
        uw.load_prompt(str(tmp_path / "nope.txt"))
    empty = tmp_path / "e.txt"
    empty.write_text("  \n")
    with pytest.raises(SystemExit):
        uw.load_prompt(str(empty))
    assert uw.load_prompt(uw.PROMPT_FILE) == uw.PROMPT
