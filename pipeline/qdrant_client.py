"""Minimal Qdrant REST client (stdlib only) for the chunk vector store.

Point ids are uuid5 of the chunk_id (Qdrant accepts only uint64/UUID ids). The
payload carries identifiers only; chunk text lives in chunks.db.
"""
import json
import urllib.error
import urllib.request
import uuid

UPSERT_BATCH = 256
PAYLOAD_KEYS = {"chunk_id", "doc_id", "pages", "section", "collection"}


class QdrantUnavailable(Exception):
    """Qdrant unreachable, timed out, or returned a non-2xx status."""


class Qdrant:
    def __init__(self, base_url: str, collection: str = "chunks", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.collection = collection
        self.timeout = timeout

    @classmethod
    def for_collection(cls, base_url: str, cid: str, timeout: float = 30.0):
        """Client for archive cid: naca keeps the historical `chunks` name."""
        name = "chunks" if cid == "naca" else f"chunks_{cid}"
        return cls(base_url, name, timeout=timeout)

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path, data=data, method=method,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            try:
                detail = e.read()[:300].decode("utf-8", "replace")
            except OSError:
                detail = ""
            finally:
                e.close()
            raise QdrantUnavailable(
                f"{method} {path}: HTTP {e.code} {detail}".rstrip()) from e
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise QdrantUnavailable(f"{method} {path}: {e}") from e

    def _coll(self, suffix: str = "") -> str:
        return f"/collections/{self.collection}{suffix}"

    @staticmethod
    def point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def ensure_collection(self, dim: int = 1024) -> None:
        try:
            self._call("GET", self._coll())
            return
        except QdrantUnavailable as e:
            if "HTTP 404" not in str(e):
                raise
        self._call("PUT", self._coll(), {
            "vectors": {"size": dim, "distance": "Cosine"},
            "quantization_config": {"scalar": {"type": "int8"}},
        })

    def upsert(self, items: list[tuple[str, list[float], dict]]) -> None:
        points = []
        for chunk_id, vector, payload in items:
            if set(payload) != PAYLOAD_KEYS:
                raise ValueError(
                    f"payload keys must be exactly {sorted(PAYLOAD_KEYS)}, "
                    f"got {sorted(payload)}")
            points.append({"id": self.point_id(chunk_id), "vector": vector,
                           "payload": payload})
        # Batched: one request per UPSERT_BATCH points. A large doc's chunks in
        # a single request can exceed Qdrant's JSON payload limit (32MB).
        for i in range(0, len(points), UPSERT_BATCH):
            self._call("PUT", self._coll("/points?wait=true"),
                       {"points": points[i:i + UPSERT_BATCH]})

    def delete_by_doc(self, doc_id: str) -> None:
        self._call("POST", self._coll("/points/delete?wait=true"), {
            "filter": {"must": [{"key": "doc_id", "match": {"value": doc_id}}]}})

    def search(self, vector: list[float], limit: int) -> list[tuple[str, float]]:
        body = self._call("POST", self._coll("/points/search"),
                          {"vector": vector, "limit": limit, "with_payload": True})
        return [(r["payload"]["chunk_id"], r["score"]) for r in body["result"]]

    def count(self) -> int:
        return self._call("POST", self._coll("/points/count"),
                          {"exact": True})["result"]["count"]
