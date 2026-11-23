"""Global token-bucket rate limiter for the public MCP endpoint."""
import math
import threading
import time

from mcp.server.mcpserver.exceptions import ToolError


class TokenBucket:
    def __init__(self, rate: float, burst: int, clock=time.monotonic):
        self.rate = float(rate)
        self.burst = int(burst)
        self._clock = clock
        self._tokens = float(burst)
        self._last = clock()
        self._lock = threading.Lock()

    def middleware(self, tool: str, kwargs: dict) -> None:
        with self._lock:
            now = self._clock()
            elapsed = max(0.0, now - self._last)
            self._last = now
            self._tokens = min(float(self.burst),
                               self._tokens + elapsed * self.rate)
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            wait = (1.0 - self._tokens) / self.rate if self.rate > 0 else 1.0
        raise ToolError(f"rate limit: retry in ~{max(1, math.ceil(wait))}s")


def install_limiter(rate: float, burst: int) -> TokenBucket:
    from mcp_server import server  # late import: avoids a cycle, honors patched MIDDLEWARE
    bucket = TokenBucket(rate, burst)
    server.MIDDLEWARE.append(bucket.middleware)
    return bucket
