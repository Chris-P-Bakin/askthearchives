import json
import math
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from embed_client import QUERY_INSTRUCT, EmbedClient, EmbedUnavailable


class _Stub:
    def __init__(self, status=200):
        self.requests = []  # (path, body)
        self.status = status
        self.shuffle = False  # return data reversed with explicit "index" fields
        stub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _empty(self, code):
                self.send_response(code)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n))
                stub.requests.append((self.path, body))
                if stub.status != 200:
                    return self._empty(stub.status)
                if self.path == "/v1/embeddings":
                    out = {"data": [{"embedding": [3.0, 4.0]} for _ in body["input"]]}
                    if stub.shuffle:
                        # vector i is [1, i] so order is checkable post-normalise
                        out = {"data": [{"index": i, "embedding": [1.0, float(i)]}
                                        for i in range(len(body["input"]))][::-1]}
                elif self.path == "/tokenize":
                    out = {"tokens": list(range(len(body["content"].split())))}
                elif self.path == "/detokenize":
                    out = {"content": " ".join(f"w{t}" for t in body["tokens"])}
                else:
                    return self._empty(404)
                data = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


@pytest.fixture
def stub():
    s = _Stub()
    yield s
    s.close()


def test_embed_normalizes(stub):
    vecs = EmbedClient(stub.url).embed(["a b", "c"])
    assert len(vecs) == 2
    assert vecs[0] == pytest.approx([0.6, 0.8])
    assert math.isclose(sum(x * x for x in vecs[1]), 1.0)


def test_query_prefix_applied(stub):
    EmbedClient(stub.url).embed_query("lunar regolith")
    sent = [b for p, b in stub.requests if p == "/v1/embeddings"][0]["input"]
    assert sent == [QUERY_INSTRUCT + "lunar regolith"]
    assert sent[0].startswith("Instruct: Given a web search query")


def test_overlong_text_truncated_via_tokenize(stub):
    text = " ".join(f"x{i}" for i in range(3000))
    vecs = EmbedClient(stub.url, max_tokens=100).embed([text])
    assert len(vecs) == 1
    detok = [b for p, b in stub.requests if p == "/detokenize"]
    assert len(detok) == 1
    assert len(detok[0]["tokens"]) == 100
    sent = [b for p, b in stub.requests if p == "/v1/embeddings"][0]["input"][0]
    assert len(sent.split()) == 100


def test_short_text_not_detokenized(stub):
    c = EmbedClient(stub.url, max_tokens=100)
    assert c.truncate("a b c") == "a b c"
    assert c.count_tokens("a b c") == 3
    assert "/detokenize" not in [p for p, _ in stub.requests]


def test_connection_refused_raises_embed_unavailable():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    with pytest.raises(EmbedUnavailable):
        EmbedClient(f"http://127.0.0.1:{port}", timeout=2).embed(["x"])


def test_non_200_raises_embed_unavailable():
    s = _Stub(status=400)
    try:
        with pytest.raises(EmbedUnavailable):
            EmbedClient(s.url).embed(["x"])
    finally:
        s.close()


def test_embed_reorders_by_response_index(stub):
    stub.shuffle = True
    vecs = EmbedClient(stub.url).embed(["a", "b", "c"])
    for i, v in enumerate(vecs):
        n = math.sqrt(1 + i * i)
        assert v == pytest.approx([1 / n, i / n])
