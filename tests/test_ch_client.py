import io
import json
import urllib.error
import urllib.request

import pytest

from ch_client import CHClient, CHUnavailable


class _FakeResp:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_urlopen(monkeypatch):
    calls = []
    state = {"body": b"", "error": None}

    def _open(req, timeout=None):
        calls.append({
            "url": req.full_url,
            "method": req.get_method(),
            "headers": {k.lower(): v for k, v in req.header_items()},
            "body": req.data,
            "timeout": timeout,
        })
        if state["error"] is not None:
            raise state["error"]
        return _FakeResp(state["body"])

    monkeypatch.setattr(urllib.request, "urlopen", _open)
    return calls, state


def test_insert_posts_jsoneachrow(fake_urlopen):
    calls, _ = fake_urlopen
    client = CHClient("http://ch:8123", "writer", "s3cret")
    client.insert("archives.samples", [{"a": 1}, {"a": 2, "b": "x"}])

    assert len(calls) == 1
    call = calls[0]
    assert call["method"] == "POST"
    assert call["url"].startswith("http://ch:8123/?query=")
    assert "INSERT%20INTO%20archives.samples%20FORMAT%20JSONEachRow" in call["url"]
    assert call["body"].decode("utf-8") == '{"a": 1}\n{"a": 2, "b": "x"}'
    assert call["headers"]["x-clickhouse-user"] == "writer"
    assert call["headers"]["x-clickhouse-key"] == "s3cret"


def test_insert_empty_rows_no_request(fake_urlopen):
    calls, _ = fake_urlopen
    CHClient("http://ch:8123", "u", "p").insert("archives.samples", [])
    assert calls == []


def test_http_error_raises_with_body_snippet(fake_urlopen, monkeypatch):
    long_body = b"Code: 60. DB::Exception: Table archives.samples does not exist. " + b"x" * 500
    err = urllib.error.HTTPError(
        "http://ch:8123/", 500, "Internal Server Error", {}, io.BytesIO(long_body))
    _, state = fake_urlopen
    state["error"] = err

    client = CHClient("http://ch:8123", "u", "p")
    with pytest.raises(CHUnavailable) as exc:
        client.insert("archives.samples", [{"a": 1}])

    msg = str(exc.value)
    text = long_body.decode()
    assert "500" in msg
    assert text[:300] in msg
    assert text[:301] not in msg


def test_query_returns_text(fake_urlopen):
    calls, state = fake_urlopen
    state["body"] = b"42\n"
    result = CHClient("http://ch:8123", "u", "p").query("SELECT count() FROM archives.samples")

    assert result == "42\n"
    assert calls[0]["method"] == "POST"
    assert calls[0]["body"] == b"SELECT count() FROM archives.samples"


def test_transport_failure_raises_chunavailable(fake_urlopen):
    _, state = fake_urlopen
    state["error"] = urllib.error.URLError("connection refused")
    with pytest.raises(CHUnavailable):
        CHClient("http://ch:8123", "u", "p").query("SELECT 1")


def test_incomplete_read_raises_chunavailable(fake_urlopen):
    import http.client
    _, state = fake_urlopen
    state["error"] = http.client.IncompleteRead(b"par")
    with pytest.raises(CHUnavailable):
        CHClient("http://ch:8123", "u", "p").query("SELECT 1")
