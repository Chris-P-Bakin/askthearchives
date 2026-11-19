import json

from pass5 import decide_pair, run_pass5


def test_decide_pair_buckets():
    assert decide_pair({"cer": 0.01, "a_chars": 500, "b_chars": 500, "bucket": "agree"}) \
        == ("vision_27bq4", "agreed")
    assert decide_pair({"cer": 0.05, "a_chars": 500, "b_chars": 500, "bucket": "minor"}) \
        == ("vision_27bq4", "minor")
    assert decide_pair({"cer": 0.30, "a_chars": 500, "b_chars": 600, "bucket": "conflict"}) \
        == ("vision_27bq4", "conflict")


def test_decide_pair_empty_sides():
    # both engines say blank page
    assert decide_pair({"cer": 0.0, "a_chars": 0, "b_chars": 0, "bucket": "agree"}) \
        == ("vision_27bq4", "blank")
    # 27B produced nothing but haiku has text -> haiku, single-engine confidence
    assert decide_pair({"cer": 1.0, "a_chars": 0, "b_chars": 47, "bucket": "conflict"}) \
        == ("vision_haiku2", "single")


def _page(rdir, vdir, n, text, failed=False):
    d = rdir / vdir
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{n}.txt").write_text(text)
    (d / f"{n}.meta.json").write_text(json.dumps(
        {"config_hash": "x", "loop_guard": {"vision_failed": failed}}))


def test_run_pass5_writes_records_and_queue(tmp_path):
    rdir = tmp_path / "77777"
    _page(rdir, "vision_27bq4", 1, "same text")
    _page(rdir, "vision_haiku2", 1, "same text")
    _page(rdir, "vision_27bq4", 2, "x" * 600)
    _page(rdir, "vision_haiku2", 2, "y" * 600)
    # 27B loop-guard failure with healthy haiku -> single/haiku
    _page(rdir, "vision_27bq4", 3, "garbage", failed=True)
    _page(rdir, "vision_haiku2", 3, "haiku rescue text")

    agreement = tmp_path / "agreement.jsonl"
    with agreement.open("w") as f:
        f.write(json.dumps({"norm_hash": "n", "t_agree": 0.02, "t_conflict": 0.1}) + "\n")
        f.write(json.dumps({"doc": "77777", "page": 1, "cer": 0.0,
                            "a_chars": 9, "b_chars": 9, "bucket": "agree"}) + "\n")
        f.write(json.dumps({"doc": "77777", "page": 2, "cer": 1.0,
                            "a_chars": 600, "b_chars": 600, "bucket": "conflict"}) + "\n")

    queue = tmp_path / "review_queue.jsonl"
    counts = run_pass5(tmp_path, agreement, queue)
    assert counts == {"agreed": 1, "conflict": 1, "single": 1}

    r1 = json.loads((rdir / "reconcile" / "1.json").read_text())
    assert r1 == {"page_source": "vision_27bq4", "confidence": "agreed",
                  "inter_cer": 0.0, "decided_by": "agreement_v1"}
    r3 = json.loads((rdir / "reconcile" / "3.json").read_text())
    assert r3["page_source"] == "vision_haiku2" and r3["confidence"] == "single"

    q = [json.loads(l) for l in queue.read_text().splitlines()]
    # vision_failed page outranks the conflict page
    assert [(r["doc"], r["page"], r["reason"]) for r in q] == \
        [("77777", 3, "vision_failed"), ("77777", 2, "conflict")]
