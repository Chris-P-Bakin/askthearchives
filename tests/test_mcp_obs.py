import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from mcp_server import obs, server
from mcp_server.db import NotFoundError


@pytest.fixture
def spool(tmp_path, monkeypatch):
    p = tmp_path / "spool.ndjson"
    monkeypatch.setattr(obs, "SPOOL", str(p))
    monkeypatch.setattr(server, "MIDDLEWARE", [])
    return p


def _lines(p):
    return [json.loads(x) for x in p.read_text().splitlines()]


def test_ok_event_fields(spool):
    out = server._run("search", lambda: {"results": [1, 2], "degraded": False})
    assert out == {"results": [1, 2], "degraded": False}
    (ev,) = _lines(spool)
    assert ev["tool"] == "search" and ev["status"] == "ok"
    assert ev["results"] == 2 and ev["bytes"] > 0 and ev["degraded"] == 0
    assert ev["ts"] > 0 and ev["duration_ms"] >= 0


def test_degraded_and_list_results(spool):
    server._run("search", lambda: {"results": [1], "degraded": True})
    server._run("search", lambda: [1, 2, 3])
    a, b = _lines(spool)
    assert a["degraded"] == 1 and b["results"] == 3 and b["degraded"] == 0


def test_rejected_event_on_middleware_raise(spool):
    def boom(tool, kwargs):
        raise ToolError("rate limited")
    server.MIDDLEWARE.append(boom)
    with pytest.raises(ToolError):
        server._run("search", lambda: [1])
    (ev,) = _lines(spool)
    assert ev["status"] == "rejected" and ev["bytes"] == 0 and ev["results"] == 0


def test_error_event_on_tool_raise(spool):
    def fn():
        raise NotFoundError("nope")
    with pytest.raises(ToolError):
        server._run("get_page", fn)
    (ev,) = _lines(spool)
    assert ev["status"] == "error" and ev["tool"] == "get_page"
    assert ev["bytes"] == 0 and ev["results"] == 0


def test_spool_write_failure_never_breaks_call(tmp_path, monkeypatch):
    monkeypatch.setattr(obs, "SPOOL", str(tmp_path / "no" / "such" / "spool"))
    monkeypatch.setattr(server, "MIDDLEWARE", [])
    assert server._run("search", lambda: [1]) == [1]


def test_unserializable_result_still_recorded(spool):
    o = object()
    assert server._run("search", lambda: o) is o
    (ev,) = _lines(spool)
    assert ev["status"] == "ok" and ev["bytes"] == 0 and ev["results"] == 0


def test_disabled_when_unset(tmp_path, monkeypatch):
    monkeypatch.setattr(obs, "SPOOL", None)
    monkeypatch.setattr(server, "MIDDLEWARE", [])
    monkeypatch.chdir(tmp_path)
    assert server._run("search", lambda: [1]) == [1]
    assert list(tmp_path.iterdir()) == []


def test_bytes_result_size_recorded(spool):
    blob = b"\x89PNG" + b"\x00" * 100
    assert server._run("get_page_image", lambda: blob) == blob
    (ev,) = _lines(spool)
    assert ev["bytes"] == len(blob)
