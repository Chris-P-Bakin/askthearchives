import json
import socket
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from qdrant_client import Qdrant, QdrantUnavailable


class _Stub:
    def __init__(self, exists=False, status=200, search_result=None):
        self.requests = []  # (method, path, body)
        self.exists = exists
        self.status = status
        self.search_result = search_result or []
        stub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _reply(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _handle(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n)) if n else None
                stub.requests.append((self.command, self.path, body))
                if stub.status != 200:
                    return self._reply(stub.status, {"status": "error"})
                p = self.path.split("?")[0]
                if self.command == "GET":
                    if stub.exists:
                        return self._reply(200, {"result": {"status": "green"}})
                    return self._reply(404, {"status": {"error": "not found"}})
                if p.endswith("/points/search"):
                    return self._reply(200, {"result": stub.search_result})
                if p.endswith("/points/count"):
                    return self._reply(200, {"result": {"count": 7}})
                return self._reply(200, {"result": True})

            do_GET = do_PUT = do_POST = _handle

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def stub():
    s = _Stub()
    yield s
    s.close()


def test_ensure_collection_idempotent():
    s = _Stub(exists=False)
    try:
        q = Qdrant(s.url)
        q.ensure_collection()
        puts = [r for r in s.requests if r[0] == "PUT"]
        assert len(puts) == 1
        assert puts[0][1] == "/collections/chunks"
        assert puts[0][2] == {
            "vectors": {"size": 1024, "distance": "Cosine"},
            "quantization_config": {"scalar": {"type": "int8"}},
        }
        s.exists = True
        s.requests.clear()
        q.ensure_collection()
        assert [r[0] for r in s.requests] == ["GET"]
    finally:
        s.close()


def test_upsert_payload_shape(stub):
    q = Qdrant(stub.url)
    payload = {"chunk_id": "d1:0001", "doc_id": "d1", "pages": [1, 2],
               "section": "Intro", "collection": "ntrs"}
    q.upsert([("d1:0001", [0.1, 0.2], payload)])
    method, path, body = stub.requests[0]
    assert method == "PUT"
    assert path == "/collections/chunks/points?wait=true"
    pt = body["points"][0]
    assert pt["id"] == str(uuid.uuid5(uuid.NAMESPACE_URL, "d1:0001"))
    assert pt["id"] == q.point_id("d1:0001")
    assert pt["vector"] == [0.1, 0.2]
    assert set(pt["payload"]) == {"chunk_id", "doc_id", "pages", "section", "collection"}
    assert "text" not in pt["payload"]


def test_upsert_rejects_extra_payload_keys(stub):
    q = Qdrant(stub.url)
    bad = {"chunk_id": "d1:0001", "doc_id": "d1", "pages": [1],
           "section": "", "collection": "c", "text": "leak"}
    with pytest.raises(ValueError):
        q.upsert([("d1:0001", [0.1], bad)])
    assert stub.requests == []


def test_delete_by_doc_sends_filter(stub):
    Qdrant(stub.url).delete_by_doc("d1")
    method, path, body = stub.requests[0]
    assert method == "POST"
    assert path == "/collections/chunks/points/delete?wait=true"
    assert body == {"filter": {"must": [{"key": "doc_id", "match": {"value": "d1"}}]}}


def test_search_returns_chunk_ids_ranked():
    s = _Stub(search_result=[
        {"id": "x", "score": 0.9, "payload": {"chunk_id": "a:0001"}},
        {"id": "y", "score": 0.5, "payload": {"chunk_id": "b:0002"}},
    ])
    try:
        out = Qdrant(s.url).search([0.1, 0.2], 2)
        assert out == [("a:0001", 0.9), ("b:0002", 0.5)]
        _, path, body = s.requests[0]
        assert path == "/collections/chunks/points/search"
        assert body == {"vector": [0.1, 0.2], "limit": 2, "with_payload": True}
    finally:
        s.close()


def test_count(stub):
    assert Qdrant(stub.url).count() == 7
    assert stub.requests[0][1] == "/collections/chunks/points/count"
    assert stub.requests[0][2] == {"exact": True}


def test_non_2xx_raises():
    s = _Stub(status=500)
    try:
        with pytest.raises(QdrantUnavailable):
            Qdrant(s.url).search([0.1], 1)
        with pytest.raises(QdrantUnavailable):
            Qdrant(s.url).ensure_collection()
    finally:
        s.close()


def test_unavailable_raises():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with pytest.raises(QdrantUnavailable):
        Qdrant(f"http://127.0.0.1:{port}", timeout=1).search([0.1], 1)


def test_upsert_batches_large_point_sets(stub):
    q = Qdrant(stub.url)
    payload = {"chunk_id": "", "doc_id": "d1", "pages": [1],
               "section": "", "collection": "test"}
    items = [(f"d1:{i:04d}", [0.1, 0.2], {**payload, "chunk_id": f"d1:{i:04d}"})
             for i in range(600)]
    q.upsert(items)
    puts = [r for r in stub.requests if r[0] == "PUT"]
    assert len(puts) == 3  # 600 points at UPSERT_BATCH=256 -> 256+256+88
    sent = [p["payload"]["chunk_id"] for _, _, body in puts for p in body["points"]]
    assert sent == [f"d1:{i:04d}" for i in range(600)]
