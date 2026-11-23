"""Client for a llama.cpp embedding server (--embedding --pooling last).

Texts are truncated by token count (never characters): digit-dense table chunks
tokenize far beyond char-count estimates and overflow the server context (HTTP 400).
"""
import json
import math
import urllib.error
import urllib.request

QUERY_INSTRUCT = ("Instruct: Given a web search query, retrieve relevant "
                  "passages that answer the query\nQuery: ")


class EmbedUnavailable(Exception):
    """Embed server unreachable, timed out, or returned non-200."""


class EmbedClient:
    def __init__(self, base_url: str, timeout: float = 60.0, max_tokens: int = 2000):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status != 200:
                    raise EmbedUnavailable(f"{path}: HTTP {resp.status}")
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            e.close()
            raise EmbedUnavailable(f"{path}: HTTP {e.code}") from e
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise EmbedUnavailable(f"{path}: {e}") from e

    def _tokens(self, text: str) -> list:
        return self._post("/tokenize", {"content": text})["tokens"]

    def count_tokens(self, text: str) -> int:
        return len(self._tokens(text))

    def truncate(self, text: str) -> str:
        tokens = self._tokens(text)
        if len(tokens) <= self.max_tokens:
            return text
        return self._post("/detokenize",
                          {"tokens": tokens[:self.max_tokens]})["content"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        body = self._post("/v1/embeddings", {"input": [self.truncate(t) for t in texts]})
        out = []
        items = sorted(enumerate(body["data"]),
                       key=lambda p: p[1].get("index", p[0]))
        for _, item in items:
            v = item["embedding"]
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out

    def embed_query(self, q: str) -> list[float]:
        return self.embed([QUERY_INSTRUCT + q])[0]
