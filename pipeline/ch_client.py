"""Minimal ClickHouse HTTP client (stdlib only) for the observability tables.

Writes use JSONEachRow over the HTTP interface; auth is via the
X-ClickHouse-User / X-ClickHouse-Key headers so the password never lands in a URL.
"""
import http.client
import json
import urllib.error
import urllib.parse
import urllib.request


class CHUnavailable(Exception):
    """ClickHouse unreachable, timed out, or returned a non-2xx status."""


class CHClient:
    def __init__(self, url: str, user: str, password: str, timeout: float = 10.0):
        self.url = url.rstrip("/")
        self.user = user
        self.password = password
        self.timeout = timeout

    def _post(self, query: str | None, body: bytes) -> str:
        target = self.url + "/"
        if query is not None:
            target += "?" + urllib.parse.urlencode({"query": query}, quote_via=urllib.parse.quote)
        req = urllib.request.Request(
            target, data=body, method="POST",
            headers={"X-ClickHouse-User": self.user, "X-ClickHouse-Key": self.password})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            try:
                detail = e.read()[:300].decode("utf-8", "replace")
            except OSError:
                detail = ""
            finally:
                e.close()
            raise CHUnavailable(f"HTTP {e.code} {detail}".rstrip()) from e
        except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as e:
            raise CHUnavailable(str(e)) from e

    def insert(self, table: str, rows: list[dict]) -> None:
        if not rows:
            return
        body = "\n".join(json.dumps(r) for r in rows).encode("utf-8")
        self._post(f"INSERT INTO {table} FORMAT JSONEachRow", body)

    def query(self, sql: str) -> str:
        return self._post(None, sql.encode("utf-8"))
