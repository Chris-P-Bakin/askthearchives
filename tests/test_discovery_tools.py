import importlib.util
import json
from pathlib import Path

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from mcp_server import obs, server

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "catalog_build_mod2", ROOT / "pipeline" / "catalog_build.py")
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)


def _anc(*pairs):
    return [{"naId": n, "levelOfDescription": lv, "title": f"t{n}", "distance": d}
            for n, lv, d in pairs]


def _item(n, parent, digitized):
    r = {"naId": n, "title": f"Item {n}", "levelOfDescription": "item",
         "ancestors": _anc((parent, "series", 1), (582, "recordGroup", 2))}
    if digitized:
        r["digitalObjects"] = [{"objectType": "PDF", "objectUrl": "http://x/a.pdf"}]
    return r


@pytest.fixture
def cat(tmp_path, monkeypatch):
    d = tmp_path / "m"
    d.mkdir()
    recs = [
        {"naId": 582, "title": "Records of NASA", "levelOfDescription": "recordGroup",
         "inclusiveStartDate": {"year": 1903}, "inclusiveEndDate": {"year": 2006}},
        {"naId": 10, "title": "Speech Files", "levelOfDescription": "series",
         "scopeAndContentNote": "Correspondence about wind tunnels.",
         "ancestors": _anc((582, "recordGroup", 1)),
         "inclusiveStartDate": {"year": 1968}, "inclusiveEndDate": {"year": 1969}},
        {"naId": 20, "title": "Budget Files", "levelOfDescription": "series",
         "scopeAndContentNote": "Tunnels budget.",
         "ancestors": _anc((582, "recordGroup", 1))},
        _item(101, 10, True), _item(102, 10, False), _item(103, 20, False)]
    (d / "a.jsonl").write_text("\n".join(json.dumps({"record": r}) for r in recs) + "\n")
    db = tmp_path / "catalog.db"
    cb.build(db, {"nara-rg255": d}, {"nara-rg255"})
    monkeypatch.setattr(server, "CATALOG_DB", db)
    monkeypatch.setattr(server, "MIDDLEWARE", [])
    monkeypatch.setattr(obs, "SPOOL", str(tmp_path / "spool.ndjson"))
    monkeypatch.setattr(server, "stats_cached", lambda: {"collections": [
        {"id": "naca", "reports": 3, "pages": 30}]})
    return db


def test_list_collections_merges_registry_and_catalog(cat):
    rows = {r["id"]: r for r in server.list_collections()}
    assert rows["naca"]["reports"] == 3 and rows["naca"]["status"] == "transcribed"
    assert rows["nara-rg255"]["catalog"] == {
        "n_record_groups": 1, "n_series": 2, "n_items": 3, "n_digitized": 1}
    assert "reports" not in rows["nara-rg255"]
    assert rows["nara-rg255"]["title"]


def test_describe_archive_children_and_url(cat):
    d = server.describe_archive(582)
    assert d["record_url"] == "https://catalog.archives.gov/id/582"
    assert d["years"] == "1903-2006" and d["n_items"] == 3
    assert d["children"]["total"] == 2
    assert {c["naId"] for c in d["children"]["items"]} == {10, 20}
    with pytest.raises(ToolError):
        server.describe_archive(999999)
    with pytest.raises(ToolError):
        server.describe_archive(-1)  # synthetic collection ids are not addressable


def test_search_catalog_filters(cat):
    hits = server.search_catalog("tunnels")
    assert {h["naId"] for h in hits} == {10, 20}
    assert [h["naId"] for h in server.search_catalog("tunnels", digitized_only=True)] == [10]
    assert server.search_catalog("tunnels", level="recordGroup") == []
    assert {h["level"] for h in server.search_catalog("nasa", level="recordGroup")} == {"recordGroup"}
    assert len(server.search_catalog("tunnels", limit=1)) == 1
    assert server.search_catalog("NOT (a OR") == []  # no FTS syntax errors
    with pytest.raises(ToolError):
        server.search_catalog("x", level="bogus")


def test_tools_error_when_catalog_missing(cat, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "CATALOG_DB", tmp_path / "absent.db")
    with pytest.raises(ToolError, match="catalog not built"):
        server.search_catalog("x")
    with pytest.raises(ToolError, match="catalog not built"):
        server.describe_archive(582)
    rows = {r["id"]: r for r in server.list_collections()}  # degrades to registry
    assert "catalog" not in rows["naca"] and rows["naca"]["reports"] == 3


def test_discovery_calls_recorded_as_events(cat, tmp_path):
    server.list_collections()
    server.describe_archive(582)
    server.search_catalog("tunnels")
    evs = [json.loads(x) for x in (tmp_path / "spool.ndjson").read_text().splitlines()]
    assert [e["tool"] for e in evs] == ["list_collections", "describe_archive", "search_catalog"]
    assert all(e["status"] == "ok" for e in evs)
