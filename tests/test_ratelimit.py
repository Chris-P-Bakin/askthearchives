import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp_server import ratelimit, server  # noqa: E402
from mcp_server.server import ToolError  # noqa: E402


def test_bucket_allows_burst_then_rejects():
    t = [0.0]
    b = ratelimit.TokenBucket(rate=1, burst=3, clock=lambda: t[0])
    for _ in range(3):
        b.middleware("search", {})
    with pytest.raises(ToolError, match="rate limit: retry in"):
        b.middleware("search", {})


def test_bucket_refills_with_time():
    t = [0.0]
    b = ratelimit.TokenBucket(rate=1, burst=2, clock=lambda: t[0])
    b.middleware("search", {})
    b.middleware("search", {})
    with pytest.raises(ToolError):
        b.middleware("search", {})
    t[0] += 1.0
    b.middleware("search", {})  # exactly one more
    with pytest.raises(ToolError):
        b.middleware("search", {})


def test_refill_capped_at_burst():
    t = [0.0]
    b = ratelimit.TokenBucket(rate=1, burst=2, clock=lambda: t[0])
    t[0] += 1000.0
    b.middleware("search", {})
    b.middleware("search", {})
    with pytest.raises(ToolError):
        b.middleware("search", {})


def test_concurrent_burst_all_get_toolerror_not_crash():
    b = ratelimit.TokenBucket(rate=1, burst=1)
    start = time.monotonic()
    results = []
    lock = threading.Lock()

    def hit():
        try:
            b.middleware("search", {})
            out = "ok"
        except ToolError:
            out = "limited"
        except Exception as e:  # noqa: BLE001
            out = e
        with lock:
            results.append(out)

    threads = [threading.Thread(target=hit) for _ in range(20)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    elapsed = time.monotonic() - start
    assert len(results) == 20
    assert all(r in ("ok", "limited") for r in results), results
    assert results.count("ok") <= 1 + elapsed * 1 + 1e-9


def test_install_limiter_appends_middleware_and_guard_rejects(monkeypatch):
    monkeypatch.setattr(server, "MIDDLEWARE", [])
    b = ratelimit.install_limiter(1, 1)
    assert server.MIDDLEWARE == [b.middleware]
    server._guard("search", {})
    with pytest.raises(ToolError):
        server._guard("search", {})
