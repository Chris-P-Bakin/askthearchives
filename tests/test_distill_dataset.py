import json

from PIL import Image

from distill_dataset import build_manifest


def _page(root, doc, n, bucket, label="some transcribed text", px=(280, 280),
          gt_done=False):
    rdir = root / doc
    (rdir / "reconcile").mkdir(parents=True, exist_ok=True)
    (rdir / "reconcile" / f"{n}.json").write_text(json.dumps(
        {"page_source": "vision_27bq4", "confidence": bucket,
         "inter_cer": 0.01, "decided_by": "agreement_v1"}))
    (rdir / "vision_27bq4").mkdir(exist_ok=True)
    (rdir / "vision_27bq4" / f"{n}.txt").write_text(label)
    (rdir / "pages").mkdir(exist_ok=True)
    Image.new("L", px, 255).save(rdir / "pages" / f"{n}.png")
    if gt_done:
        (rdir / "ground_truth").mkdir(exist_ok=True)
        (rdir / "ground_truth" / f"{n}.txt").write_text("gt")
        (rdir / "ground_truth" / f"{n}.json").write_text(json.dumps({"status": "done"}))


def _prompt(root):
    p = root / "prompts"
    p.mkdir()
    (p / "transcribe_v2.txt").write_text("Transcribe this page.")
    return p / "transcribe_v2.txt"


def test_build_manifest_selects_buckets_and_excludes_gt(tmp_path):
    _page(tmp_path, "11111", 1, "agreed")
    _page(tmp_path, "11111", 2, "minor")
    _page(tmp_path, "11111", 3, "conflict")
    _page(tmp_path, "11111", 4, "agreed", gt_done=True)
    m = build_manifest(tmp_path, prompt_path=_prompt(tmp_path))
    got = {(e["ntrs_id"], e["page"]) for e in m["examples"]}
    assert got == {("11111", 1), ("11111", 2)}
    assert m["teacher_config_hash"] == "be1cd2673d3b"
    assert len(m["prompt_sha256"]) == 64


def test_build_manifest_pairs_image_and_label_same_page(tmp_path):
    import hashlib
    _page(tmp_path, "11111", 1, "agreed", label="page one text")
    _page(tmp_path, "11111", 2, "agreed", label="page two text")
    m = build_manifest(tmp_path, prompt_path=_prompt(tmp_path))
    for e in m["examples"]:
        assert e["image"].endswith(f"{e['ntrs_id']}/pages/{e['page']}.png")
        want = hashlib.sha256(
            (tmp_path / e["ntrs_id"] / "vision_27bq4" / f"{e['page']}.txt")
            .read_bytes()).hexdigest()
        assert e["label_sha256"] == want


def test_build_manifest_split_deterministic_and_disjoint(tmp_path):
    for n in range(1, 41):
        _page(tmp_path, "11111", n, "agreed")
    prompt = _prompt(tmp_path)
    m1 = build_manifest(tmp_path, prompt_path=prompt, seed=13)
    m2 = build_manifest(tmp_path, prompt_path=prompt, seed=13)
    assert m1["examples"] == m2["examples"]
    val = [e for e in m1["examples"] if e["split"] == "val"]
    train = [e for e in m1["examples"] if e["split"] == "train"]
    assert len(val) + len(train) == 40
    assert 1 <= len(val) <= 4  # ~5% of 40


def test_build_manifest_drops_over_budget(tmp_path):
    _page(tmp_path, "11111", 1, "agreed")
    _page(tmp_path, "11111", 2, "agreed", label="x" * 400000)
    m = build_manifest(tmp_path, prompt_path=_prompt(tmp_path))
    assert [(e["ntrs_id"], e["page"]) for e in m["examples"]] == [("11111", 1)]
    assert m["dropped_over_budget"] == [["11111", 2]]


def test_build_manifest_samples_blanks(tmp_path):
    for n in range(1, 31):
        _page(tmp_path, "11111", n, "blank", label="")
    _page(tmp_path, "11111", 99, "agreed")
    prompt = _prompt(tmp_path)
    m1 = build_manifest(tmp_path, prompt_path=prompt, blank_sample=20, seed=13)
    blanks1 = [e["page"] for e in m1["examples"] if e["bucket"] == "blank"]
    assert len(blanks1) == 20
    m2 = build_manifest(tmp_path, prompt_path=prompt, blank_sample=20, seed=13)
    assert blanks1 == [e["page"] for e in m2["examples"] if e["bucket"] == "blank"]


def test_est_tokens_vision_at_32px_per_merged_token(tmp_path):
    # Qwen3-VL: 16px patches, 2x2 merge -> one token per 32x32 px block.
    from distill_dataset import _est_tokens
    img = tmp_path / "p.png"
    Image.new("L", (320, 320), 255).save(img)
    assert _est_tokens(img, "", "") == 100 + 64  # ceil(320/32)^2 + fixed overhead


def _label_235b(root, doc, n, text):
    d = root / doc / "vision_235b"
    d.mkdir(exist_ok=True)
    (d / f"{n}.txt").write_text(text)


def test_build_manifest_admits_extra_pages_from_relabel_dir(tmp_path):
    import hashlib
    _page(tmp_path, "11111", 1, "agreed")
    _page(tmp_path, "11111", 3, "conflict")
    _label_235b(tmp_path, "11111", 3, "tiebreak text")
    m = build_manifest(tmp_path, prompt_path=_prompt(tmp_path),
                       extra_pages={("11111", 3)})
    by = {(e["ntrs_id"], e["page"]): e for e in m["examples"]}
    assert set(by) == {("11111", 1), ("11111", 3)}
    e = by[("11111", 3)]
    assert e["bucket"] == "tiebreak"
    assert e["label_dir"] == "vision_235b"
    assert e["label_sha256"] == hashlib.sha256(b"tiebreak text").hexdigest()
    assert by[("11111", 1)]["label_dir"] == "vision_27bq4"


def test_build_manifest_relabels_legacy_pages(tmp_path):
    import hashlib
    _page(tmp_path, "11111", 1, "agreed", label="27b table text")
    _label_235b(tmp_path, "11111", 1, "235b table text")
    m = build_manifest(tmp_path, prompt_path=_prompt(tmp_path),
                       relabel_pages={("11111", 1)})
    (e,) = m["examples"]
    assert e["bucket"] == "agreed"
    assert e["label_dir"] == "vision_235b"
    assert e["label_sha256"] == hashlib.sha256(b"235b table text").hexdigest()


def test_build_manifest_drops_relabel_page_missing_235b(tmp_path):
    _page(tmp_path, "11111", 1, "agreed")
    _page(tmp_path, "11111", 2, "agreed")
    m = build_manifest(tmp_path, prompt_path=_prompt(tmp_path),
                       relabel_pages={("11111", 2)})
    assert [(e["ntrs_id"], e["page"]) for e in m["examples"]] == [("11111", 1)]
    assert m["dropped_missing_relabel"] == [["11111", 2]]


def test_tiebreak_pages_union_of_agreement_files(tmp_path):
    from distill_dataset import tiebreak_pages
    rows_t = [{"norm_hash": "x"},
              {"doc": "111", "page": 1, "bucket": "agree"},
              {"doc": "111", "page": 2, "bucket": "conflict"}]
    rows_h = [{"doc": "111", "page": 1, "bucket": "minor"},
              {"doc": "222", "page": 5, "bucket": "minor"}]
    (tmp_path / "agreement_vision_235b__vision_27bq4.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows_t) + "\n")
    (tmp_path / "agreement_vision_235b__vision_haiku2.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows_h) + "\n")
    assert tiebreak_pages(tmp_path) == {("111", 1), ("222", 5)}


def test_table_heavy_pages_by_digit_fraction(tmp_path):
    from distill_dataset import table_heavy_pages
    _page(tmp_path, "11111", 1, "agreed", label="1234567890" * 10)   # all digits
    _page(tmp_path, "11111", 2, "minor", label="prose " * 50)        # no digits
    _page(tmp_path, "11111", 3, "conflict", label="1234567890" * 10) # wrong bucket
    assert table_heavy_pages(tmp_path) == {("11111", 1)}


def test_tiebreak_pages_bucket_filter(tmp_path):
    from distill_dataset import tiebreak_pages
    rows_t = [{"doc": "111", "page": 1, "bucket": "agree"},
              {"doc": "111", "page": 2, "bucket": "minor"}]
    rows_h = [{"doc": "222", "page": 5, "bucket": "minor"},
              {"doc": "222", "page": 6, "bucket": "agree"}]
    (tmp_path / "agreement_vision_235b__vision_27bq4.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows_t) + "\n")
    (tmp_path / "agreement_vision_235b__vision_haiku2.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows_h) + "\n")
    assert tiebreak_pages(tmp_path, buckets=("agree",)) == {("111", 1), ("222", 6)}
