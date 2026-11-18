import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import units
from unit_assimilate import assimilate

UNIT = {"unit_id": "19930001-p3", "ntrs_id": "19930001", "page": 3,
        "image_key": "k", "image_sha256": "x", "model_id": units.MODEL_ID,
        "prompt_sha": units.PROMPT_SHA, "config_hash": units.CONFIG_HASH}
META = {"config_hash": units.CONFIG_HASH, "config": {}, "server": "s",
        "image_px": [10, 10], "seconds": 1.0, "usage": {}, "finish_reason": "stop",
        "output_chars": 10,
        "loop_guard": {"ratio": None, "retried": False, "vision_failed": False,
                       "temperature_used": 0.0}}
PROSE = "A page of ordinary prose about boundary layers. " * 3


def _table(bad=False):
    rows = ["| x | y |", "|---|---|"]
    for i in range(12):
        v = 4.8e-4 - i * 1e-5
        if bad and i == 5:
            v *= 100
        rows.append(f"| {1000 + i} | {v:.6f} |")
    return "\n".join(rows)


def queue(root):
    p = root / "escalation_queue.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def paths(root):
    d = root / "19930001" / "vision_8blora_v2_1"
    return d / "3.txt", d / "3.meta.json"


def test_atomic_write_and_extended_meta(tmp_path):
    r = assimilate(UNIT, PROSE, dict(META, worker="w7"), data_root=tmp_path)
    assert r["status"] == "assimilated"
    t, m = paths(tmp_path)
    assert t.read_text() == PROSE
    meta = json.loads(m.read_text())
    assert meta["unit_id"] == "19930001-p3" and meta["worker"] == "w7"
    assert meta["assimilated_ts"] and meta["config_hash"] == units.CONFIG_HASH
    assert not list(t.parent.glob("*.tmp"))


def test_duplicate_delivery_noop(tmp_path):
    bad = _table(bad=True)
    assert assimilate(UNIT, bad, dict(META), data_root=tmp_path)["status"] == "assimilated"
    t, m = paths(tmp_path)
    before = (t.stat().st_mtime_ns, m.stat().st_mtime_ns)
    n = len(queue(tmp_path))
    assert n == 1
    assert assimilate(UNIT, "different", dict(META), data_root=tmp_path) == {"status": "duplicate"}
    assert (t.stat().st_mtime_ns, m.stat().st_mtime_ns) == before
    assert t.read_text() == bad
    assert len(queue(tmp_path)) == n


def test_table_flags_enqueued(tmp_path):
    text = _table(bad=True)
    assert sum(c.isdigit() for c in text) / len(text) > 0.15
    assimilate(UNIT, text, dict(META), data_root=tmp_path)
    q = queue(tmp_path)
    assert len(q) == 1
    row = q[0]
    assert row["kind"] == "table_flags" and row["unit_id"] == "19930001-p3"
    assert row["ntrs_id"] == "19930001" and row["page"] == 3
    assert row["flags"] and row["config_hash"] == units.CONFIG_HASH and row["ts"]


def test_clean_table_not_enqueued(tmp_path):
    assimilate(UNIT, _table(), dict(META), data_root=tmp_path)
    assert queue(tmp_path) == []


def test_vision_failed_enqueued(tmp_path):
    m = copy.deepcopy(META)
    m["loop_guard"]["vision_failed"] = True
    r = assimilate(UNIT, "", m, data_root=tmp_path)
    assert r["status"] == "assimilated"
    q = queue(tmp_path)
    assert [x["kind"] for x in q] == ["vision_failed"]
    assert paths(tmp_path)[0].exists()


def test_clean_prose_not_enqueued(tmp_path):
    assimilate(UNIT, PROSE, dict(META), data_root=tmp_path)
    assert queue(tmp_path) == []


def test_queue_dedup_when_page_rewritten(tmp_path):
    # stale-hash result on disk is overwritten, but the queue row is not doubled
    assimilate(UNIT, _table(bad=True), dict(META), data_root=tmp_path)
    t, m = paths(tmp_path)
    old = json.loads(m.read_text())
    old["config_hash"] = "000000000000"
    m.write_text(json.dumps(old))
    assert assimilate(UNIT, _table(bad=True), dict(META), data_root=tmp_path)["status"] == "assimilated"
    assert len(queue(tmp_path)) == 1


def test_default_root_is_common_data(tmp_path, monkeypatch):
    import common
    monkeypatch.setattr(common, "DATA", tmp_path)
    assimilate(UNIT, PROSE, dict(META))
    assert paths(tmp_path)[0].exists()


def test_cli(tmp_path):
    (tmp_path / "u.json").write_text(json.dumps(UNIT))
    (tmp_path / "r.txt").write_text(PROSE)
    (tmp_path / "r.meta.json").write_text(json.dumps(META))
    script = Path(__file__).resolve().parent.parent / "pipeline" / "unit_assimilate.py"
    root = tmp_path / "data"
    p = subprocess.run([sys.executable, str(script), str(tmp_path / "u.json"),
                        str(tmp_path / "r.txt"), str(tmp_path / "r.meta.json"),
                        "--data-root", str(root)], capture_output=True, text=True)
    assert p.returncode == 0 and json.loads(p.stdout)["status"] == "assimilated"
    assert paths(root)[0].exists()


def test_bad_hash_rejected_writes_nothing(tmp_path):
    r = assimilate(UNIT, _table(bad=True), dict(META, config_hash="deadbeef0000"), data_root=tmp_path)
    assert r == {"status": "rejected", "reason": "config_hash_mismatch"}
    assert list(tmp_path.iterdir()) == []


def test_empty_text_not_vision_failed_rejected(tmp_path):
    r = assimilate(UNIT, "", dict(META), data_root=tmp_path)
    assert r == {"status": "rejected", "reason": "empty_output"}
    assert list(tmp_path.iterdir()) == []


def test_cli_rejected_exit_nonzero(tmp_path):
    (tmp_path / "u.json").write_text(json.dumps(UNIT))
    (tmp_path / "r.txt").write_text(PROSE)
    (tmp_path / "r.meta.json").write_text(json.dumps(dict(META, config_hash="deadbeef0000")))
    script = Path(__file__).resolve().parent.parent / "pipeline" / "unit_assimilate.py"
    root = tmp_path / "data"
    p = subprocess.run([sys.executable, str(script), str(tmp_path / "u.json"),
                        str(tmp_path / "r.txt"), str(tmp_path / "r.meta.json"),
                        "--data-root", str(root)], capture_output=True, text=True)
    assert p.returncode == 1 and json.loads(p.stdout)["reason"] == "config_hash_mismatch"
    assert not root.exists()


def test_duplicate_path_reenqueues_lost_flag_row(tmp_path):
    """Crash window: page landed but the escalation row was lost; redelivery self-heals."""
    meta = dict(META)
    assert assimilate(UNIT, _table(bad=True), meta, data_root=tmp_path)["status"] == "assimilated"
    (tmp_path / "escalation_queue.jsonl").unlink()           # simulate the lost row
    res = assimilate(UNIT, _table(bad=True), dict(META), data_root=tmp_path)
    assert res["status"] == "duplicate"
    assert [r["kind"] for r in queue(tmp_path)] == ["table_flags"]
    assimilate(UNIT, _table(bad=True), dict(META), data_root=tmp_path)   # idempotent
    assert len(queue(tmp_path)) == 1


def test_duplicate_path_reenqueues_vision_failed(tmp_path):
    meta = dict(META, loop_guard=dict(META["loop_guard"], vision_failed=True))
    assimilate(UNIT, PROSE, meta, data_root=tmp_path)
    (tmp_path / "escalation_queue.jsonl").unlink()
    assert assimilate(UNIT, PROSE, dict(meta), data_root=tmp_path)["status"] == "duplicate"
    assert [r["kind"] for r in queue(tmp_path)] == ["vision_failed"]


def test_unrolled_low_digit_page_enqueued(tmp_path):
    # Cell-per-line unroll of non-numeric labels: digit fraction under the
    # gate, but structurally broken output must still reach the table lane.
    import random
    rng = random.Random(7)
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    text = "\n".join("".join(rng.choice(letters) for _ in range(rng.randint(2, 7)))
                     for _ in range(300))
    assert sum(c.isdigit() for c in text) / len(text) <= 0.15
    assimilate(UNIT, text, dict(META), data_root=tmp_path)
    q = queue(tmp_path)
    assert len(q) == 1
    assert q[0]["kind"] == "table_flags"
    assert {"check": "unrolled"} in q[0]["flags"]


def test_assimilate_nara_collection_lands_under_collection_root(tmp_path, monkeypatch):
    monkeypatch.setattr(units.registry, "ROOT", tmp_path)
    monkeypatch.setattr(units.registry, "_cache", {
        "naca": {"id": "naca", "root": "data", "doc_prefix": ""},
        "nara-rg255": {"id": "nara-rg255", "root": "data/collections/nara-rg255",
                       "doc_prefix": "nara255-"}})
    u = dict(UNIT, ntrs_id="nara255-1", unit_id="nara255-1-p3", collection="nara-rg255")
    assert assimilate(u, PROSE, dict(META), data_root=tmp_path / "data")["status"] == "assimilated"
    d = tmp_path / "data" / "collections" / "nara-rg255" / "nara255-1" / units.CAMPAIGN_LAYER
    assert (d / f"{UNIT['page']}.txt").exists()
    assert not (tmp_path / "data" / "nara255-1").exists()


def test_queue_row_carries_collection_only_for_non_naca(tmp_path, monkeypatch):
    monkeypatch.setattr(units.registry, "ROOT", tmp_path)
    monkeypatch.setattr(units.registry, "_cache", {
        "naca": {"id": "naca", "root": "data", "doc_prefix": ""},
        "nara-rg255": {"id": "nara-rg255", "root": "data/collections/nara-rg255",
                       "doc_prefix": "nara255-"}})
    root = tmp_path / "data"
    u = dict(UNIT, ntrs_id="nara255-1", unit_id="nara255-1-p3", collection="nara-rg255")
    assimilate(u, _table(bad=True), dict(META), data_root=root)
    assimilate(UNIT, _table(bad=True), dict(META), data_root=root)
    rows = {r["ntrs_id"]: r for r in queue(root)}
    assert rows["nara255-1"]["collection"] == "nara-rg255"
    assert "collection" not in rows["19930001"]
