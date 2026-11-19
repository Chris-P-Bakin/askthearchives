import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import escalate  # noqa: E402


def _row(nid, page):
    return {"unit_id": f"{nid}:{page}", "ntrs_id": nid, "page": page, "kind": "vision_failed"}


def _write_meta(data, row, hash=escalate.ESC_27B_HASH, failed=False):
    d = data / row["ntrs_id"] / escalate.LAYER_27B
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{row['page']}.meta.json").write_text(json.dumps(
        {"config_hash": hash, "loop_guard": {"vision_failed": failed}}))


def _fake_run(data, rows, calls, **meta_kw):
    def run(cmd, **kw):
        calls.append((cmd, kw))
        for r in rows:
            _write_meta(data, r, **meta_kw)
        return subprocess.CompletedProcess(cmd, 0)
    return run


def test_run_27b_writes_pages_file_and_invokes_pass4a(tmp_path, monkeypatch):
    rows = [_row("A", 1), _row("B", 2)]
    calls = []
    monkeypatch.setattr(escalate.subprocess, "run", _fake_run(tmp_path, rows, calls))
    monkeypatch.setenv("NACA_VLM_MODEL", "campaign-model")
    res = escalate.run_27b(rows, data=tmp_path, review_path=tmp_path / "rq.jsonl")
    assert res["done"] == 2 and res["vision_failed"] == 0 and res["skipped"] == 0
    cmd, kw = calls[0]
    pf = Path(cmd[cmd.index("--pages-file") + 1])
    assert pf.parent == tmp_path / "escalation_runs" and pf.name.endswith("_27b.pages")
    got = [json.loads(l) for l in pf.read_text().splitlines()]
    assert got == [{"doc": "A", "page": 1}, {"doc": "B", "page": 2}]
    env = kw["env"]
    for k, v in escalate.ENV_27B.items():
        assert env[k] == v
    assert env["NACA_VLM_MODEL"] == "Qwen3.8-27B-UD-Q4_K_M"
    assert env["NACA_VLM_DIR"] == "vision_27b_v2" and env["NACA_REPORTS_CSV"] == str(escalate.common.ROOT / "data" / "roster.csv")
    assert escalate.is_done(rows[0], data=tmp_path)


def test_27b_vision_failed_goes_to_review_queue(tmp_path, monkeypatch):
    rows = [_row("A", 1)]
    monkeypatch.setattr(escalate.subprocess, "run",
                        _fake_run(tmp_path, rows, [], failed=True))
    rq = tmp_path / "rq.jsonl"
    res = escalate.run_27b(rows, data=tmp_path, review_path=rq)
    assert res["vision_failed"] == 1 and res["done"] == 0
    rev = [json.loads(l) for l in rq.read_text().splitlines()]
    assert rev[0]["reason"] == "vision_failed_27b" and rev[0]["unit_id"] == "A:1"
    assert escalate.is_done(rows[0], data=tmp_path)


def test_27b_skips_pages_already_done(tmp_path, monkeypatch):
    rows = [_row("A", 1), _row("B", 2)]
    _write_meta(tmp_path, rows[0])
    calls = []
    monkeypatch.setattr(escalate.subprocess, "run", _fake_run(tmp_path, [rows[1]], calls))
    res = escalate.run_27b(rows, data=tmp_path, review_path=tmp_path / "rq.jsonl")
    assert res["skipped"] == 1 and res["done"] == 1
    pf = Path(calls[0][0][calls[0][0].index("--pages-file") + 1])
    assert [json.loads(l)["doc"] for l in pf.read_text().splitlines()] == ["B"]


def test_27b_hash_mismatch_aborts(tmp_path, monkeypatch):
    rows = [_row("A", 1)]
    monkeypatch.setattr(escalate.subprocess, "run",
                        _fake_run(tmp_path, rows, [], hash="deadbeef0000"))
    with pytest.raises(RuntimeError, match="drift"):
        escalate.run_27b(rows, data=tmp_path, review_path=tmp_path / "rq.jsonl")
    assert not escalate.is_done(rows[0], data=tmp_path)


def test_27b_dry_run_and_limit(tmp_path, monkeypatch):
    rows = [_row("A", 1), _row("B", 2)]
    calls = []
    monkeypatch.setattr(escalate.subprocess, "run", _fake_run(tmp_path, rows, calls))
    res = escalate.run_27b(rows, limit=1, dry_run=True, data=tmp_path)
    assert calls == [] and res["done"] == 0 and res["planned"] == 1
    assert not (tmp_path / "escalation_runs").exists()


def test_env_27b_paths_exist_and_invocation_pinned(tmp_path, monkeypatch):
    for k in ("NACA_VLM_PROMPT_FILE", "NACA_REPORTS_CSV"):
        assert Path(escalate.ENV_27B[k]).is_absolute() and Path(escalate.ENV_27B[k]).exists()
    rows = [_row("A", 1)]
    calls = []
    monkeypatch.setattr(escalate.subprocess, "run", _fake_run(tmp_path, rows, calls))
    escalate.run_27b(rows, data=tmp_path, review_path=tmp_path / "rq.jsonl")
    cmd, kw = calls[0]
    assert cmd[1] == str(escalate.common.ROOT / "pipeline" / "pass4a.py")
    assert cmd[cmd.index("--workers") + 1] == "1"
    assert kw["cwd"] == str(escalate.common.ROOT)


def test_27b_nonzero_returncode_recorded_and_stale_meta_not_drift(tmp_path, monkeypatch):
    rows = [_row("A", 1)]
    _write_meta(tmp_path, rows[0], hash="stale0000000")   # stale leftover, different hash
    monkeypatch.setattr(escalate.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 3))
    res = escalate.run_27b(rows, data=tmp_path, review_path=tmp_path / "rq.jsonl")
    assert res["pass4a_error"] == 3 and res["failed"] == 1 and res["done"] == 0
    assert not escalate.is_done(rows[0], data=tmp_path)


def test_27b_empty_output_goes_to_review_not_done(tmp_path, monkeypatch):
    rows = [_row("A", 1)]

    def run(cmd, **kw):
        _write_meta(tmp_path, rows[0])
        (tmp_path / "A" / escalate.LAYER_27B / "1.txt").write_text("")
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(escalate.subprocess, "run", run)
    rq = tmp_path / "rq.jsonl"
    res = escalate.run_27b(rows, data=tmp_path, review_path=rq)
    assert res["done"] == 0 and res["empty"] == 1
    rev = [json.loads(l) for l in rq.read_text().splitlines()]
    assert rev[0]["reason"] == "empty_output_27b" and rev[0]["unit_id"] == "A:1"
    assert not escalate.is_done(rows[0], data=tmp_path)


def test_review_queue_is_lane_owned_file():
    assert escalate.REVIEW_QUEUE.name == "escalation_review.jsonl"
