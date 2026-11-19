import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import escalate  # noqa: E402
import units  # noqa: E402

PNG = b"fakepng"


def _setup(tmp_path, page=3):
    data = tmp_path / "data"
    (data / "N1" / "pages").mkdir(parents=True)
    (data / "N1" / "pages" / f"{page}.png").write_bytes(PNG)
    u = {"unit_id": f"N1-p{page}", "ntrs_id": "N1", "page": page,
         "image_sha256": hashlib.sha256(PNG).hexdigest()}
    uf = tmp_path / "units.jsonl"
    uf.write_text(json.dumps(u) + "\n")
    row = {"unit_id": u["unit_id"], "ntrs_id": "N1", "page": page, "kind": "wu_error",
           "error_code": 1, "wu_name": f"naca_{u['unit_id']}"}
    return data, uf, row


class Runner:
    def __init__(self, fail_names=()):
        self.calls, self.fail_names = [], set(fail_names)

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        if "bin/create_work" in argv:
            name = argv[argv.index("--wu_name") + 1]
            if name in self.fail_names:
                return subprocess.CompletedProcess(argv, 1, "", f"duplicate {name}")
        return subprocess.CompletedProcess(argv, 0, "", "")

    def cw(self):
        return [c for c in self.calls if "bin/create_work" in c]


def _names(r):
    return [c[c.index("--wu_name") + 1] for c in r.cw()]


def _done(data):
    return [json.loads(l) for l in (data / "escalation_done.jsonl").read_text().splitlines()]


def test_already_recovered_marks_done_without_resubmit(tmp_path):
    data, uf, row = _setup(tmp_path)
    d = data / "N1" / units.CAMPAIGN_LAYER
    d.mkdir()
    (d / "3.meta.json").write_text(json.dumps({"config_hash": next(iter(units.ACCEPTED_HASHES))}))
    r = Runner()
    res = escalate.run_wu_error([row], uf, data=data, run=r, stage_dir=tmp_path / "st")
    assert res["already_recovered"] == 1 and res["resubmitted"] == 0
    assert r.calls == []
    assert _done(data)[0]["disposition"] == "already_recovered"


def test_resubmit_marks_done_with_wu_name(tmp_path):
    data, uf, row = _setup(tmp_path)
    r = Runner()
    res = escalate.run_wu_error([row], uf, data=data, run=r, stage_dir=tmp_path / "st")
    assert res["resubmitted"] == 1 and _names(r) == ["naca_N1-p3"]
    e = _done(data)[0]
    assert e["disposition"] == "resubmitted" and e["wu_name"] == "naca_N1-p3"
    assert not list((tmp_path / "st").iterdir())  # staging cleaned


def test_resubmit_uses_retry_suffix_on_name_collision(tmp_path):
    data, uf, row = _setup(tmp_path)
    r = Runner(fail_names={"naca_N1-p3"})
    res = escalate.run_wu_error([row], uf, data=data, run=r, stage_dir=tmp_path / "st")
    assert _names(r) == ["naca_N1-p3", "naca_N1-p3__r2"]
    assert res["resubmitted"] == 1 and res["failed"] == 0
    e = _done(data)[0]
    assert e["wu_name"] == "naca_N1-p3__r2" and "duplicate" in e["first_error"]


def test_both_attempts_fail_surfaces_both_errors(tmp_path):
    data, uf, row = _setup(tmp_path)
    r = Runner(fail_names={"naca_N1-p3", "naca_N1-p3__r2"})
    res = escalate.run_wu_error([row], uf, data=data, run=r, stage_dir=tmp_path / "st")
    assert res["failed"] == 1 and res["resubmitted"] == 0
    assert len(res["errors"][0]["errors"]) == 2
    assert not (data / "escalation_done.jsonl").exists()


def test_sha_mismatch_fails_without_create_work(tmp_path):
    data, uf, row = _setup(tmp_path)
    (data / "N1" / "pages" / "3.png").write_bytes(b"other")
    r = Runner()
    res = escalate.run_wu_error([row], uf, data=data, run=r, stage_dir=tmp_path / "st")
    assert res["failed"] == 1 and r.cw() == []


def test_dry_run_and_ledgered_rows_do_nothing(tmp_path):
    data, uf, row = _setup(tmp_path)
    r = Runner()
    assert escalate.run_wu_error([row], uf, dry_run=True, data=data, run=r)["planned"] == 1
    assert r.calls == [] and not (data / "escalation_done.jsonl").exists()
    escalate.mark_done(row, "resubmitted", data=data)
    res = escalate.run_wu_error([row], uf, data=data, run=r, stage_dir=tmp_path / "st")
    assert res["skipped"] == 1 and r.calls == []


def _ledger(data, row, ts):
    with open(data / "escalation_done.jsonl", "a") as f:
        f.write(json.dumps({"unit_id": row["unit_id"], "kind": "wu_error",
                            "disposition": "resubmitted", "ts": ts}) + "\n")


def test_newer_row_reopens_resubmitted_unit_next_name_r3(tmp_path):
    data, uf, row = _setup(tmp_path)
    _ledger(data, row, "2026-01-01T00:00:00Z")
    row = dict(row, wu_name="naca_N1-p3__r2", ts="2026-01-02T00:00:00Z")
    r = Runner()
    res = escalate.run_wu_error([row], uf, data=data, run=r, stage_dir=tmp_path / "st")
    assert _names(r) == ["naca_N1-p3__r3"] and res["resubmitted"] == 1
    assert _done(data)[-1]["wu_name"] == "naca_N1-p3__r3"


def test_cap_reached_marks_exhausted_and_enqueues_review(tmp_path):
    data, uf, row = _setup(tmp_path)
    _ledger(data, row, "2026-01-01T00:00:00Z")
    row = dict(row, wu_name="naca_N1-p3__r4", ts="2026-01-02T00:00:00Z")
    r = Runner()
    rp = tmp_path / "review.jsonl"
    res = escalate.run_wu_error([row], uf, data=data, run=r, stage_dir=tmp_path / "st",
                                review_path=rp)
    assert r.calls == [] and res["retry_exhausted"] == 1
    assert _done(data)[-1]["disposition"] == "retry_exhausted"
    rv = json.loads(rp.read_text())
    assert rv["reason"] == "wu_error_exhausted" and rv["unit_id"] == row["unit_id"]


def test_older_or_equal_ts_row_stays_skipped(tmp_path):
    data, uf, row = _setup(tmp_path)
    _ledger(data, row, "2026-01-02T00:00:00Z")
    r = Runner()
    for ts in ("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"):
        res = escalate.run_wu_error([dict(row, ts=ts)], uf, data=data, run=r,
                                    stage_dir=tmp_path / "st")
        assert res["skipped"] == 1
    assert r.calls == []


def _hooklib():
    import importlib.machinery, importlib.util
    root = Path(__file__).resolve().parent.parent
    p = str(root / "boinc/glue/bin/naca_hooklib.py")
    spec = importlib.util.spec_from_file_location("naca_hooklib_t", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, root


def test_producer_to_lane_errored_resubmission_reopens_to_r3(tmp_path, monkeypatch):
    """C1: record_wu_error on naca_<uid>__r2 must queue unit <uid>, append past the existing
    (unit_id, wu_error) row, route to the wu_error lane and reopen -> __r3."""
    h, root = _hooklib()
    data = tmp_path / "data"
    (data / "19930091026" / "pages").mkdir(parents=True)
    (data / "19930091026" / "pages" / "9.png").write_bytes(PNG)
    uid = "19930091026-p9"
    uf = tmp_path / "units.jsonl"
    uf.write_text(json.dumps({"unit_id": uid, "ntrs_id": "19930091026", "page": 9,
                              "image_sha256": hashlib.sha256(PNG).hexdigest()}) + "\n")
    monkeypatch.setattr(h, "PIPELINE", str(root / "pipeline"))
    monkeypatch.setattr(h, "DATA_ROOT", str(data))
    assert h.record_wu_error(["--error", "-1", f"naca_{uid}", "1", "1"])
    first = json.loads((data / "escalation_queue.jsonl").read_text().splitlines()[0])
    escalate.mark_done(first, "resubmitted", data=data)
    import time
    time.sleep(1.1)   # row ts has 1s resolution; the errored resubmission must be strictly newer
    assert h.record_wu_error(["--error", "-1", f"naca_{uid}__r2", "2", "1"])
    lines = (data / "escalation_queue.jsonl").read_text().splitlines()
    assert len(lines) == 2
    routed = escalate.route(escalate.read_queue(data / "escalation_queue.jsonl"), data)
    assert [r["unit_id"] for r in routed["wu_error"]] == [uid] and not routed["skipped_missing"]
    r = Runner()
    res = escalate.run_wu_error(routed["wu_error"], uf, data=data, run=r, stage_dir=tmp_path / "st")
    assert res["resubmitted"] == 1 and _names(r) == ["naca_19930091026-p9__r3"]


def test_status_counts_reopened_wu_error_pending(tmp_path, monkeypatch, capsys):
    """I1: status must use the same latest-ts law as run_wu_error."""
    data, uf, row = _setup(tmp_path)
    q = data / "escalation_queue.jsonl"
    old = dict(row, ts="2026-01-01T00:00:00Z")
    q.write_text(json.dumps(old) + "\n")
    _ledger(data, old, "2026-01-02T00:00:00Z")
    monkeypatch.setattr(escalate.common, "DATA", data)
    escalate.main(["status"])
    assert "wu_error: 0 pending" in capsys.readouterr().out
    q.write_text(json.dumps(old) + "\n" + json.dumps(dict(row, wu_name="naca_N1-p3__r2",
                                                          ts="2026-01-03T00:00:00Z")) + "\n")
    escalate.main(["status"])
    assert "wu_error: 1 pending" in capsys.readouterr().out
