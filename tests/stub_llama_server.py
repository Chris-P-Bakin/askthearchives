"""Stand-in llama-server binary for launcher tests.

Invoked by launcher.py exactly like the real server (flags ignored except
--port). Serves /health and /v1/chat/completions with loop-guard-passing text.
Env: STUB_LOG — file to append one line per completion request (sha12 of the
image payload); STUB_FAIL_NTH_UNIQUE — 404 every request for the Nth (1-based)
unique image seen, persistently.
"""
import hashlib
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

GOOD = "A page of ordinary prose about boundary layers. " * 12 + "".join(
    f" word{i}x{i * 7 % 13}" for i in range(80))
if os.environ.get("STUB_NONASCII"):
    # NACA pages are full of these; they are outside cp1252/ASCII
    GOOD += " where α ≈ 0.002, β = 2°, Σc_f λ θ ± x²"

SEEN: list[str] = []   # unique image hashes in arrival order


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = b'{"status":"ok"}'
        self.send_response(200 if self.path == "/health" else 404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        img = req["messages"][0]["content"][0]["image_url"]["url"]
        h = hashlib.sha256(img.encode()).hexdigest()[:12]
        if h not in SEEN:
            SEEN.append(h)
        if log := os.environ.get("STUB_LOG"):
            with open(log, "a") as f:
                f.write(h + "\n")
        nth = int(os.environ.get("STUB_FAIL_NTH_UNIQUE", "0"))
        if nth and SEEN.index(h) + 1 == nth:
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps({"choices": [{"message": {"content": GOOD},
                                        "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 1, "completion_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        die = int(os.environ.get("STUB_DIE_AFTER", "0"))
        if die and len(SEEN) >= die:
            os._exit(1)   # simulate the server crashing mid-batch


def main():
    port = int(sys.argv[sys.argv.index("--port") + 1])
    HTTPServer(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":
    main()
