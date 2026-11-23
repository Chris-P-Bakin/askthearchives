import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import escalate  # noqa: E402


def _row(nid, page):
    return {"unit_id": f"{nid}:{page}", "ntrs_id": nid, "page": page, "kind": "table_flags"}


def _cfg(v2=True):
    return {"model": escalate.ENV_235B["NACA_VLM_MODEL"],
            "prompt_version": "file:abc123" if v2 else "v1",
            "loop_guard": {"version": "v2" if v2 else "v1"}}


def _write(data, row, cost=0.01, v2=True, text="plain text", failed=False):
    d = data / row["ntrs_id"] / escalate.LAYER_235B
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{row['page']}.txt").write_text(text)
    (d / f"{row['page']}.meta.json").write_text(json.dumps({
        "config_hash": "fresh1234567" if v2 else "3669475d3a4b", "config": _cfg(v2),
        "usage": {"cost": cost}, "loop_guard": {"vision_failed": failed}}))


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("NACA_VLM_KEY", "test-not-a-real-key")


def _fake_dispatch(data, rows, **kw):
    calls = []

    def d(chunk, data_, tag):
        calls.append([r["unit_id"] for r in chunk])
        for r in chunk:
            _write(data, r, **kw)
        return 0
    return d, calls


def test_env_paths_absolute_and_exist():
    for k in ("NACA_VLM_PROMPT_FILE", "NACA_REPORTS_CSV"):
        p = Path(escalate.ENV_235B[k])
        assert p.is_absolute() and p.exists(), k
    assert "NACA_VLM_KEY" not in escalate.ENV_235B


def test_spend_sums_meta_costs(tmp_path):
    _write(tmp_path, _row("A", 1), cost=0.25)
    _write(tmp_path, _row("B", 2), cost=0.5, v2=False)
    escalate.mark_done(_row("C", 3), "done", data=tmp_path, layer="vision_235b", cost=1.0)
    escalate.mark_done(_row("D", 4), "done", data=tmp_path, layer="vision_27b_v2", cost=9.0)
    assert escalate.spend(data=tmp_path) == pytest.approx(1.75)


def test_trial_meta_not_done_v2_is():
    assert not escalate.is_v2_235b_meta({"config": _cfg(False)})
    assert escalate.is_v2_235b_meta({"config": _cfg(True)})
    assert not escalate.is_v2_235b_meta(None)


def test_chunk_refused_when_projected_over_budget(tmp_path, monkeypatch):
    _write(tmp_path, _row("X", 1), cost=119.5)
    rows = [_row("A", i) for i in range(25)]
    d, calls = _fake_dispatch(tmp_path, rows)
    monkeypatch.setattr(escalate, "_dispatch_235b", d)
    res = escalate.run_235b(rows, data=tmp_path, review_path=tmp_path / "rq.jsonl")
    assert res["budget_stop"] is True and calls == [] and res["done"] == 0


def test_spend_recomputed_between_chunks(tmp_path, monkeypatch):
    rows = [_row("A", i) for i in range(4)]
    d, calls = _fake_dispatch(tmp_path, rows, cost=5.0)
    monkeypatch.setattr(escalate, "_dispatch_235b", d)
    res = escalate.run_235b(rows, chunk=2, budget=11.0, data=tmp_path,
                            review_path=tmp_path / "rq.jsonl")
    assert len(calls) == 1 and res["budget_stop"] and res["done"] == 2
    assert res["spent"] == pytest.approx(10.0)


def test_cleared_table_flag_marks_done_still_flagged_goes_review(tmp_path, monkeypatch):
    rows = [_row("A", 1), _row("B", 2)]
    flagged = {"B"}

    def vt(text):
        return {"tables": 1, "flags": [{"check": "monotonic"}] if text == "BAD" else []}
    monkeypatch.setattr(escalate.table_verify, "verify_text", vt)

    def d2(chunk, data_, tag):
        for r in chunk:
            _write(tmp_path, r, text="BAD" if r["ntrs_id"] in flagged else "ok")
        return 0
    monkeypatch.setattr(escalate, "_dispatch_235b", d2)
    rq = tmp_path / "rq.jsonl"
    res = escalate.run_235b(rows, data=tmp_path, review_path=rq)
    assert res["cleared"] == 1 and res["still_flagged"] == 1
    rev = [json.loads(l) for l in rq.read_text().splitlines()]
    assert len(rev) == 1 and rev[0]["unit_id"] == "B:2"
    assert rev[0]["reason"] == "table_flags_after_235b"
    assert all(escalate.is_done(r, data=tmp_path) for r in rows)


def test_done_235b_page_never_redispatched(tmp_path, monkeypatch):
    rows = [_row("A", 1), _row("B", 2), _row("C", 3)]
    _write(tmp_path, rows[0])                      # v2 meta -> done
    _write(tmp_path, rows[1], v2=False)            # trial meta -> NOT done
    escalate.mark_done(rows[2], "done", data=tmp_path, layer="vision_235b")
    d, calls = _fake_dispatch(tmp_path, rows)
    monkeypatch.setattr(escalate, "_dispatch_235b", d)
    res = escalate.run_235b(rows, data=tmp_path, review_path=tmp_path / "rq.jsonl")
    assert calls == [["B:2"]] and res["skipped"] == 1
    assert res["recovered"] == 1 and res["done"] == 2


def test_stale_meta_not_counted_done(tmp_path, monkeypatch):
    rows = [_row("A", 1)]
    _write(tmp_path, rows[0], v2=False)
    monkeypatch.setattr(escalate, "_dispatch_235b", lambda c, d, t: 3)
    res = escalate.run_235b(rows, data=tmp_path, review_path=tmp_path / "rq.jsonl")
    assert res["failed"] == 1 and res["done"] == 0 and res["pass4a_error"] == 3


def test_missing_key_fatal_before_dispatch(tmp_path, monkeypatch):
    monkeypatch.delenv("NACA_VLM_KEY")
    called = []
    monkeypatch.setattr(escalate, "_dispatch_235b", lambda *a: called.append(1) or 0)
    with pytest.raises(RuntimeError, match="NACA_VLM_KEY"):
        escalate.run_235b([_row("A", 1)], data=tmp_path)
    assert not called


def test_dispatch_env_and_cwd(tmp_path, monkeypatch):
    seen = {}

    def run(cmd, **kw):
        seen.update(cmd=cmd, **kw)
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(escalate.subprocess, "run", run)
    monkeypatch.setenv("NACA_VLM_MODEL", "campaign-model")
    escalate._dispatch_235b([_row("A", 1)], tmp_path, 0)
    assert seen["env"]["NACA_VLM_MODEL"] == escalate.ENV_235B["NACA_VLM_MODEL"]
    assert seen["env"]["NACA_VLM_KEY"] == "test-not-a-real-key"
    assert seen["cwd"] == str(escalate.common.ROOT)


def test_interrupted_v2_meta_without_ledger_is_recovered_not_redispatched(tmp_path, monkeypatch):
    rows = [_row("A", 1), _row("B", 2)]
    _write(tmp_path, rows[0], text="BAD", cost=0.07)   # meta landed, run died before mark
    monkeypatch.setattr(escalate.table_verify, "verify_text",
                        lambda t: {"tables": 1, "flags": [{"check": "x"}] if t == "BAD" else []})
    d, calls = _fake_dispatch(tmp_path, rows)
    monkeypatch.setattr(escalate, "_dispatch_235b", d)
    rq = tmp_path / "rq.jsonl"
    res = escalate.run_235b(rows, data=tmp_path, review_path=rq)
    assert calls == [["B:2"]]                      # A never sent to the API again
    assert res["recovered"] == 1 and res["still_flagged"] == 1
    assert json.loads(rq.read_text().splitlines()[0])["unit_id"] == "A:1"
    assert escalate.is_done(rows[0], data=tmp_path)
    assert escalate.spend(data=tmp_path) == pytest.approx(0.08)
    # no key needed when only recovery work remains
    monkeypatch.delenv("NACA_VLM_KEY")
    _write(tmp_path, _row("C", 3))
    assert escalate.run_235b([_row("C", 3)], data=tmp_path, review_path=rq)["recovered"] == 1


def test_empty_235b_output_goes_to_review_not_cleared(tmp_path, monkeypatch):
    rows = [_row("A", 1)]
    d, _ = _fake_dispatch(tmp_path, rows, text="  \n")
    monkeypatch.setattr(escalate, "_dispatch_235b", d)
    rq = tmp_path / "rq.jsonl"
    res = escalate.run_235b(rows, data=tmp_path, review_path=rq)
    assert res["cleared"] == 0 and res["done"] == 0 and res["failed"] == 1
    rev = [json.loads(l) for l in rq.read_text().splitlines()]
    assert rev[0]["reason"] == "empty_output_235b"
    assert not escalate.is_done(rows[0], data=tmp_path)


def test_redispatch_preserves_trial_meta_cost(tmp_path, monkeypatch):
    rows = [_row("A", 1)]
    _write(tmp_path, rows[0], cost=0.40, v2=False)      # trial meta: $0.40 already spent
    d, _ = _fake_dispatch(tmp_path, rows, cost=0.05)    # re-dispatch overwrites it
    monkeypatch.setattr(escalate, "_dispatch_235b", d)
    res = escalate.run_235b(rows, data=tmp_path)
    assert res["spent"] == pytest.approx(0.45)          # 0.40 banked + 0.05 fresh


def test_trial_cost_banked_once_across_reruns(tmp_path, monkeypatch):
    rows = [_row("A", 1)]
    _write(tmp_path, rows[0], cost=0.40, v2=False)
    monkeypatch.setattr(escalate, "_dispatch_235b", lambda c, d, t: 0)  # dispatch lands nothing
    escalate.run_235b(rows, data=tmp_path)
    d, _ = _fake_dispatch(tmp_path, rows, cost=0.05)
    monkeypatch.setattr(escalate, "_dispatch_235b", d)
    res = escalate.run_235b(rows, data=tmp_path)
    assert res["spent"] == pytest.approx(0.45)          # banked exactly once
