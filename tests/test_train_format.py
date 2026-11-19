import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from train_8b_lora import format_example


def _mk(tmp_path, doc, n, label):
    rdir = tmp_path / doc
    (rdir / "vision_27bq4").mkdir(parents=True)
    (rdir / "vision_27bq4" / f"{n}.txt").write_text(label)
    (rdir / "pages").mkdir()
    (rdir / "pages" / f"{n}.png").write_bytes(b"png")
    import hashlib
    return {"ntrs_id": doc, "page": n, "bucket": "agreed",
            "image": str(rdir / "pages" / f"{n}.png"),
            "label_sha256": hashlib.sha256(label.encode()).hexdigest(),
            "split": "train", "est_tokens": 100}


def test_format_example_reads_label_and_prompt(tmp_path):
    ex = _mk(tmp_path, "11111", 5, "the page text")
    msgs = format_example(ex, "Transcribe this page.", tmp_path)["messages"]
    user, assistant = msgs
    assert user["role"] == "user"
    img, txt = user["content"]
    assert img["type"] == "image" and Path(img["image"]).is_absolute()
    assert img["image"].endswith("11111/pages/5.png")
    assert txt == {"type": "text", "text": "Transcribe this page."}
    assert assistant == {"role": "assistant",
                         "content": [{"type": "text", "text": "the page text"}]}


def test_format_example_blank_label_is_empty_string(tmp_path):
    ex = _mk(tmp_path, "11111", 7, "")
    msgs = format_example(ex, "Transcribe this page.", tmp_path)["messages"]
    assert msgs[1]["content"][0]["text"] == ""


def test_format_example_rejects_tampered_label(tmp_path):
    ex = _mk(tmp_path, "11111", 9, "original label text")
    (tmp_path / "11111" / "vision_27bq4" / "9.txt").write_text("tampered")
    try:
        format_example(ex, "Transcribe this page.", tmp_path)
    except ValueError as e:
        assert "label_sha256" in str(e)
    else:
        raise AssertionError("tampered label did not raise")


def test_mark_epoch_save_sets_checkpoint_flag():
    from types import SimpleNamespace
    from train_8b_lora import mark_epoch_save
    control = SimpleNamespace(should_save=False)
    assert mark_epoch_save(control) is control
    assert control.should_save is True


def _manifest(tmp_path, n_train):
    exs = []
    for i in range(n_train):
        ex = _mk(tmp_path, f"d{i:05d}", i, f"label text {i}")
        exs.append(ex)
    return {"examples": exs}


def test_load_split_frac_subsamples_deterministically(tmp_path):
    from train_8b_lora import load_split
    m = _manifest(tmp_path, 8)
    full = load_split(m, "P.", "train", data_root=tmp_path)
    half_a = load_split(m, "P.", "train", frac=0.5, data_root=tmp_path)
    half_b = load_split(m, "P.", "train", frac=0.5, data_root=tmp_path)
    assert len(full) == 8 and len(half_a) == 4
    assert half_a == half_b  # seeded: same subset every call
    full_labels = {m["messages"][1]["content"][0]["text"] for m in full}
    assert all(x["messages"][1]["content"][0]["text"] in full_labels for x in half_a)


def test_format_example_honors_label_dir(tmp_path):
    import hashlib
    ex = _mk(tmp_path, "11111", 5, "wrong 27b text")
    d = tmp_path / "11111" / "vision_235b"
    d.mkdir()
    (d / "5.txt").write_text("235b text")
    ex["label_dir"] = "vision_235b"
    ex["label_sha256"] = hashlib.sha256(b"235b text").hexdigest()
    msgs = format_example(ex, "Transcribe this page.", tmp_path)["messages"]
    assert msgs[1]["content"][0]["text"] == "235b text"
