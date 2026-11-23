import hashlib
import json
import re
from pathlib import Path

import pytest

import boinc_submit as bs

PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 40


def make_unit(tmp_path, nid="19930001", page=3, png=PNG, sha=None):
    pdir = tmp_path / "data" / nid / "pages"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / f"{page}.png").write_bytes(png)
    return {"unit_id": f"{nid}-p{page}", "ntrs_id": nid, "page": page,
            "image_key": f"naca/pages/{nid}/p-{page}.png",
            "image_sha256": sha or hashlib.sha256(png).hexdigest(),
            "model_id": "m", "prompt_sha": "p", "config_hash": "6ecf7d0dbeb2"}


def test_names():
    assert bs.wu_name("19930001-p3") == "naca_19930001-p3"
    assert bs.phys_names("19930001-p3") == ("19930001-p3.unit.json", "19930001-p3.page.png")


def test_create_work_args_exact():
    a = bs.create_work_args({"unit_id": "19930001-p3"})
    assert a == ["bin/create_work", "--appname", "naca_vision",
                 "--wu_name", "naca_19930001-p3",
                 "--wu_template", "templates/naca_vision_in",
                 "--result_template", "templates/naca_vision_out",
                 "--min_quorum", "1", "--target_nresults", "1",
                 "19930001-p3.unit.json", "19930001-p3.page.png"]


def test_stage_unit_writes_files(tmp_path):
    u = make_unit(tmp_path)
    stage = tmp_path / "stage"
    j, p = bs.stage_unit(u, tmp_path / "data", stage)
    assert json.loads(j.read_text()) == u
    assert p.read_bytes() == PNG
    assert j.parent == p.parent == stage


def test_stage_unit_rejects_sha_mismatch(tmp_path):
    u = make_unit(tmp_path, sha="0" * 64)
    with pytest.raises(bs.StageError, match="sha256"):
        bs.stage_unit(u, tmp_path / "data", tmp_path / "stage")


def test_stage_unit_missing_image(tmp_path):
    u = make_unit(tmp_path)
    (tmp_path / "data/19930001/pages/3.png").unlink()
    with pytest.raises(bs.StageError, match="missing"):
        bs.stage_unit(u, tmp_path / "data", tmp_path / "stage")


class FakeRun:
    def __init__(self, existing=()):
        self.calls = []
        self.existing = existing

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        r = R()
        if "mysql" in " ".join(cmd):
            r.stdout = "".join(n + "\n" for n in self.existing)
        return r


def test_submit_stages_then_creates(tmp_path):
    units = [make_unit(tmp_path, page=1), make_unit(tmp_path, page=2)]
    run = FakeRun()
    res = bs.submit(units, tmp_path / "data", tmp_path / "stage", run=run, container="c1")
    assert res == {"submitted": 2, "skipped_existing": 0, "failed": 0}
    flat = [" ".join(c) for c in run.calls]
    stage_calls = [c for c in flat if "bin/stage_file" in c]
    create_calls = [c for c in flat if "bin/create_work" in c]
    assert len(stage_calls) == 4 and len(create_calls) == 2
    assert all(c.startswith("docker exec -u boincadm -w /home/boincadm/project c1 ")
               for c in stage_calls + create_calls)
    assert "--copy /naca/stage/19930001-p1.unit.json" in stage_calls[0]
    # every stage precedes its create_work
    assert flat.index(create_calls[0]) > flat.index(stage_calls[1])
    # staging dir is cleaned after staging
    assert not list((tmp_path / "stage").iterdir())


def test_submit_skips_existing(tmp_path):
    units = [make_unit(tmp_path, page=1), make_unit(tmp_path, page=2)]
    run = FakeRun(existing=["naca_19930001-p1"])
    res = bs.submit(units, tmp_path / "data", tmp_path / "stage", run=run, container="c1")
    assert res == {"submitted": 1, "skipped_existing": 1, "failed": 0}
    assert not any("19930001-p1" in " ".join(c) and "create_work" in " ".join(c) for c in run.calls)


def test_submit_counts_stage_error_and_continues(tmp_path):
    bad = make_unit(tmp_path, page=1, sha="0" * 64)
    good = make_unit(tmp_path, page=2)
    run = FakeRun()
    res = bs.submit([bad, good], tmp_path / "data", tmp_path / "stage", run=run, container="c1")
    assert res == {"submitted": 1, "skipped_existing": 0, "failed": 1}


def test_submit_limit(tmp_path):
    units = [make_unit(tmp_path, page=i) for i in range(1, 5)]
    res = bs.submit(units, tmp_path / "data", tmp_path / "stage", run=FakeRun(),
                    container="c1", limit=2)
    assert res["submitted"] == 2


def test_dry_run_runs_nothing(tmp_path):
    units = [make_unit(tmp_path, page=1)]
    run = FakeRun()
    res = bs.submit(units, tmp_path / "data", tmp_path / "stage", run=run, container="c1", dry_run=True)
    assert run.calls == [] and res["submitted"] == 0


def test_hooklib_unit_id_from_output_path():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "boinc/glue/bin"))
    import naca_hooklib as h
    p = "/home/boincadm/project/upload/35f/naca_19930091026-p4_1_r1207525140_0"
    assert h.unit_id_from_output_path(p) == "19930091026-p4"
    assert h.unit_id_from_output_path("naca_19930001-p12_0_r5_1") == "19930001-p12"
    with pytest.raises(ValueError):
        h.unit_id_from_output_path("other_0_r5_1")


def _hl():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "boinc/glue/bin"))
    import naca_hooklib as h
    return h


def test_classify_rc_mapping():
    h = _hl()
    assert h.classify_rc(0, "") == 0
    assert h.classify_rc(1, "") == 1                       # real INVALID verdict
    assert h.classify_rc(1, "Traceback (most recent call last):\n...") == h.CRASH_RC
    assert h.classify_rc(2, "") == h.CRASH_RC
    assert h.classify_rc(0, "Traceback x") == h.CRASH_RC


def test_run_cli_missing_unit_is_crash_not_invalid(tmp_path, monkeypatch):
    h = _hl()
    monkeypatch.setattr(h, "unit_json_path", lambda uid: str(tmp_path / "nope.json"))
    t, m = tmp_path / "r.txt", tmp_path / "m.json"
    t.write_text("x"); m.write_text("{}")
    assert h.run_cli("unit_validate.py", "u", str(t), str(m)) == h.CRASH_RC


def test_run_cli_real_verdicts(tmp_path, monkeypatch):
    h = _hl()
    root = Path(__file__).resolve().parent.parent
    monkeypatch.setattr(h, "PIPELINE", str(root / "pipeline"))
    u = tmp_path / "u.json"
    u.write_text(json.dumps({"unit_id": "1-p1", "config_hash": "6ecf7d0dbeb2"}))
    monkeypatch.setattr(h, "unit_json_path", lambda uid: str(u))
    t, m = tmp_path / "r.txt", tmp_path / "m.json"
    t.write_text("x")
    m.write_text("{}")                                      # malformed meta -> INVALID verdict
    assert h.run_cli("unit_validate.py", "1-p1", str(t), str(m)) == 1
    m.write_text("not json")                                # unit_validate handles -> INVALID
    assert h.run_cli("unit_validate.py", "1-p1", str(t), str(m)) == 1
    u.write_text("not json")                                # crash inside CLI -> defer
    assert h.run_cli("unit_validate.py", "1-p1", str(t), str(m)) == h.CRASH_RC


def test_existing_wu_names_excludes_errored(tmp_path):
    seen = []
    def run(cmd, **kw):
        seen.append(" ".join(cmd))
        class R:
            returncode = 0; stdout = ""; stderr = ""
        return R()
    bs.existing_wu_names(run, "c1")
    assert "error_mask=0" in seen[0].replace(" ", "")


def test_record_wu_error_enqueues_once(tmp_path, monkeypatch):
    h = _hl()
    root = Path(__file__).resolve().parent.parent
    monkeypatch.setattr(h, "PIPELINE", str(root / "pipeline"))
    monkeypatch.setattr(h, "DATA_ROOT", str(tmp_path))
    argv = ["--error", "-1", "naca_19930001-p3", "77", "1"]
    h.record_wu_error(argv)
    h.record_wu_error(argv)
    rows = [json.loads(l) for l in (tmp_path / "escalation_queue.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    r = rows[0]
    assert (r["unit_id"], r["ntrs_id"], r["page"], r["kind"]) == ("19930001-p3", "19930001", 3, "wu_error")
    assert r["error_code"] == "-1"


def test_assimilate_error_branch_records_and_exits_0(tmp_path):
    import os, subprocess, sys
    root = Path(__file__).resolve().parent.parent
    env = dict(os.environ, NACA_PIPELINE=str(root / "pipeline"), NACA_DATA_ROOT=str(tmp_path))
    r = subprocess.run([sys.executable, str(root / "boinc/glue/bin/naca_assimilate"),
                        "--error", "-1", "naca_19930001-p3", "77", "1"], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert (tmp_path / "escalation_queue.jsonl").exists()


def test_undecodable_result_is_invalid_not_crash(tmp_path, monkeypatch):
    h = _hl()
    root = Path(__file__).resolve().parent.parent
    monkeypatch.setattr(h, "PIPELINE", str(root / "pipeline"))
    u = tmp_path / "u.json"
    u.write_text(json.dumps({"unit_id": "1-p1", "config_hash": "6ecf7d0dbeb2"}))
    monkeypatch.setattr(h, "unit_json_path", lambda uid: str(u))
    t, m = tmp_path / "r.txt", tmp_path / "m.json"
    t.write_bytes(b"\xff\xfe\x80 not utf8")
    m.write_text("{}")
    assert h.run_cli("unit_validate.py", "1-p1", str(t), str(m)) == 1
    # a missing unit.json stays transient
    monkeypatch.setattr(h, "unit_json_path", lambda uid: str(tmp_path / "nope.json"))
    assert h.run_cli("unit_validate.py", "1-p1", str(t), str(m)) == h.CRASH_RC


# --- batched app (naca_vision_b) ---

def test_batch_wu_name_deterministic_and_gt_prefix():
    us = [{"unit_id": "19930080813-p10"}, {"unit_id": "19930080813-p11"}]
    n = bs.batch_wu_name(us)
    assert n == bs.batch_wu_name(us) and n.startswith("naca_b_")
    assert not n.startswith("naca_b_gt_")
    g = bs.batch_wu_name(us, gt=True)
    assert g.startswith("naca_b_gt_") and g != n


def test_batch_template_lists_manifest_sticky_models_and_n_pages():
    xml = bs.build_batch_template(3)
    assert xml.count("<file_ref>") == 6  # manifest + 2 sticky + 3 pages
    assert xml.count("<sticky/>") == 2 and xml.count("<no_delete/>") == 2
    # sticky/no_delete belong in <file_info>, where the client honors them,
    # never in <file_ref> (C1: file_deleter deleted the staged models)
    for blk in re.findall(r"<file_ref>.*?</file_ref>", xml, re.S):
        assert "<sticky/>" not in blk and "<no_delete/>" not in blk
    for blk in re.findall(r"<file_info>.*?</file_info>", xml, re.S)[1:3]:
        assert "<sticky/>" in blk and "<no_delete/>" in blk
    assert "model.gguf" in xml and "mmproj.gguf" in xml
    assert "manifest.json" in xml and "page_000.png" in xml and "page_002.png" in xml
    # sticky model refs are served from the download hierarchy, never copied
    assert xml.count("<copy_file/>") == 4  # manifest + 3 pages only


def test_batch_create_work_args_use_batched_app_and_template():
    us = [{"unit_id": "19930001-p3"}, {"unit_id": "19930001-p4"}]
    a = bs.batch_create_work_args(us)
    name = bs.batch_wu_name(us)
    assert a[:5] == ["bin/create_work", "--appname", "naca_vision_b", "--wu_name", name]
    assert "templates/naca_vision_b_in_2" in a and "templates/naca_vision_b_out" in a
    # input files in template order: manifest, sticky models, pages
    assert a[-5:] == [f"{name}.manifest.json", bs.MODEL_PHYS, bs.MMPROJ_PHYS,
                      "19930001-p3.page.png", "19930001-p4.page.png"]


def test_submit_resolves_collection_page_path(tmp_path, monkeypatch):
    monkeypatch.setattr(bs.units_mod.registry, "ROOT", tmp_path)
    monkeypatch.setattr(bs.units_mod.registry, "_cache", {
        "naca": {"id": "naca", "root": "data", "doc_prefix": ""},
        "nara-rg255": {"id": "nara-rg255", "root": "data/collections/nara-rg255",
                       "doc_prefix": "nara255-"}})
    d = tmp_path / "data" / "collections" / "nara-rg255" / "nara255-1" / "pages"
    d.mkdir(parents=True)
    (d / "3.png").write_bytes(PNG)
    u = {"unit_id": "nara255-1-p3", "ntrs_id": "nara255-1", "page": 3,
         "collection": "nara-rg255", "image_sha256": hashlib.sha256(PNG).hexdigest()}
    j, p = bs.stage_unit(u, tmp_path / "data", tmp_path / "stage")
    assert p.read_bytes() == PNG
