import hashlib
import json
from pathlib import Path

import pytest

import units
import common


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "DATA", tmp_path)
    (tmp_path / "A1" / "pages").mkdir(parents=True)
    (tmp_path / "A1" / "pages" / "3.png").write_bytes(b"pngbytes")
    return tmp_path


def test_unit_id_and_make_unit_shape(tree):
    assert units.unit_id("A1", 3) == "A1-p3"
    u = units.make_unit("A1", 3)
    assert u == {
        "unit_id": "A1-p3", "ntrs_id": "A1", "page": 3,
        "image_key": "naca/pages/A1/p-3.png",
        "image_sha256": hashlib.sha256(b"pngbytes").hexdigest(),
        "model_id": "qwen3-vl-8b-lora-v2_1",
        "prompt_sha": units.PROMPT_SHA,
        "config_hash": "6ecf7d0dbeb2",
    }


def test_prompt_sha_pinned():
    p = Path(__file__).resolve().parent.parent / "pipeline" / "prompts" / "transcribe_v2.txt"
    assert units.PROMPT_SHA == hashlib.sha256(p.read_bytes()).hexdigest()
    assert units.ACCEPTED_HASHES == {"6ecf7d0dbeb2"}
    assert units.CAMPAIGN_LAYER == "vision_8blora_v2_1"


def test_read_write_units_roundtrip(tmp_path):
    us = [{"unit_id": "a-p1"}, {"unit_id": "b-p2"}]
    out = tmp_path / "sub" / "u.jsonl"
    units.write_units(us, out)
    assert units.read_units(out) == us
    assert not list(out.parent.glob("*.tmp"))


def test_make_unit_default_naca_backcompat(tree):
    u = units.make_unit("A1", 3)
    assert "collection" not in u
    assert units.unit_collection(u) == "naca"
    assert units.unit_doc_dir(u) == tree / "A1"


def test_make_unit_nara_collection_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(units.registry, "ROOT", tmp_path)
    monkeypatch.setattr(units.registry, "_cache", {
        "naca": {"id": "naca", "root": "data", "doc_prefix": ""},
        "nara-rg255": {"id": "nara-rg255", "root": "data/collections/nara-rg255",
                       "doc_prefix": "nara255-"}})
    croot = tmp_path / "data" / "collections" / "nara-rg255"
    (croot / "nara255-1" / "pages").mkdir(parents=True)
    (croot / "nara255-1" / "pages" / "3.png").write_bytes(b"x")
    u = units.make_unit("nara255-1", 3, collection="nara-rg255")
    assert u["ntrs_id"] == "nara255-1" and u["collection"] == "nara-rg255"
    assert u["image_key"] == "nara-rg255/pages/1/p-3.png"
    assert units.unit_doc_dir(u) == croot / "nara255-1"
    assert units.unit_doc_dir(u, tmp_path / "other") == \
        tmp_path / "other" / "collections" / "nara-rg255" / "nara255-1"
