import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
pytest.importorskip("anthropic")
import pass4b

FIX = Path(__file__).resolve().parent / "fixtures"
JUNK_V21 = (FIX / "junk_v21_p80.txt").read_text()
GOOD = (FIX / "good_table_1135_p37.txt").read_text()


def _run(tmp_path, monkeypatch, outputs):
    from PIL import Image
    rd = tmp_path / "r"
    (rd / "pages").mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(rd / "pages" / "1.png")
    monkeypatch.setattr(pass4b, "report_dir", lambda _id: rd)
    it = iter(outputs)
    monkeypatch.setattr(pass4b, "request", lambda b64: next(it))
    status = pass4b.transcribe("x", 1, force=True)
    meta = json.loads((rd / pass4b.OUT_DIR / "1.meta.json").read_text())
    return status, meta["loop_guard"]


def test_pass4b_records_reasons_on_retry_success(tmp_path, monkeypatch):
    u = {"prompt_tokens": 1, "completion_tokens": 1}
    status, g = _run(tmp_path, monkeypatch,
                     [(JUNK_V21, "stop", dict(u)), (GOOD, "stop", dict(u))])
    assert status == "ok"
    assert g["reason"] == "zlib" and g["retried"] is True and g["retry_reason"] is None


def test_pass4b_clean_first_attempt(tmp_path, monkeypatch):
    u = {"prompt_tokens": 1, "completion_tokens": 1}
    status, g = _run(tmp_path, monkeypatch, [(GOOD, "stop", u)])
    assert status == "ok" and g["reason"] is None and "retry_reason" not in g
