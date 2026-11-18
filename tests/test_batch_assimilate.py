import json

import batch_assimilate as ba
from test_batch_validate import HASH, GOOD_TEXT, manifest, ok_line, err_line


def rows(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_pages_assimilate_per_page(tmp_path):
    m = manifest(2)
    lines = [ok_line(u["unit_id"]) for u in m]
    summary = ba.assimilate_batch(m, lines, data_root=tmp_path, wu_name="naca_b_abc")
    assert summary["assimilated"] == 2 and summary["escalated"] == 0
    for u in m:
        assert (tmp_path / u["ntrs_id"] / "vision_8blora_v2_1" / f"{u['page']}.txt").exists()


def test_error_pages_enqueue_wu_error(tmp_path):
    m = manifest(2)
    lines = [ok_line(m[0]["unit_id"]), err_line(m[1]["unit_id"])]
    summary = ba.assimilate_batch(m, lines, data_root=tmp_path, wu_name="naca_b_abc")
    assert summary["assimilated"] == 1 and summary["escalated"] == 1
    q = rows(tmp_path / "escalation_queue.jsonl")
    assert len(q) == 1 and q[0]["kind"] == "wu_error"
    assert q[0]["unit_id"] == m[1]["unit_id"] and q[0]["wu_name"] == "naca_b_abc"


def test_rejected_page_escalates_not_fatal(tmp_path):
    # content-check failure (empty text claimed as success) -> escalate that page only
    m = manifest(2)
    bad = json.dumps({"unit_id": m[1]["unit_id"], "status": "ok", "text": "",
                      "meta": {"config_hash": HASH,
                               "loop_guard": {"vision_failed": False},
                               "finish_reason": "stop"}})
    summary = ba.assimilate_batch(m, [ok_line(m[0]["unit_id"]), bad],
                                  data_root=tmp_path, wu_name="naca_b_abc")
    assert summary["assimilated"] == 1 and summary["escalated"] == 1
    assert rows(tmp_path / "escalation_queue.jsonl")[0]["unit_id"] == m[1]["unit_id"]


def test_gt_layer_override_leaves_campaign_untouched(tmp_path):
    m = manifest(1)
    u = m[0]
    camp = tmp_path / u["ntrs_id"] / "vision_8blora_v2_1"
    camp.mkdir(parents=True)
    (camp / f"{u['page']}.txt").write_text("existing campaign text")
    (camp / f"{u['page']}.meta.json").write_text(json.dumps({"config_hash": HASH}))
    before = (camp / f"{u['page']}.txt").read_bytes()
    summary = ba.assimilate_batch(m, [ok_line(u["unit_id"])], data_root=tmp_path,
                                  wu_name="naca_b_gt_abc", layer="vision_8bwin")
    assert summary["assimilated"] == 1
    gt = tmp_path / u["ntrs_id"] / "vision_8bwin" / f"{u['page']}.txt"
    assert gt.read_text() == GOOD_TEXT
    assert (camp / f"{u['page']}.txt").read_bytes() == before
