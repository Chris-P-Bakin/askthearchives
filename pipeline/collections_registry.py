"""Collection registry: which archive a document belongs to, and where its data lives.

Entries come from collections.json at the repo root. Doc ids are routed by
longest matching doc_prefix; the empty prefix (naca) is the fallback.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Containers mount pipeline/ without the repo root; pipeline/collections.json
# is a committed symlink to ../collections.json so both layouts resolve.
_CANDIDATES = (ROOT / "collections.json",
               Path(__file__).resolve().parent / "collections.json")
REGISTRY_PATH = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])
_SAFE = re.compile(r"^[A-Za-z0-9_-]*$")

_cache: dict[str, dict] | None = None


def load() -> dict[str, dict]:
    """Return id -> entry for every collection in collections.json (cached)."""
    global _cache
    if _cache is None:
        entries = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        reg = {}
        for e in entries:
            if not _SAFE.match(e["doc_prefix"]):
                raise ValueError(f"unsafe doc_prefix for {e['id']}: {e['doc_prefix']!r}")
            reg[e["id"]] = e
        _cache = reg
    return _cache


def for_doc(doc_id: str) -> dict:
    """Return the entry whose doc_prefix is the longest prefix of doc_id."""
    best = None
    for entry in load().values():
        prefix = entry["doc_prefix"]
        if doc_id.startswith(prefix) and (best is None or len(prefix) > len(best["doc_prefix"])):
            best = entry
    if best is None:
        raise KeyError(f"no collection matches doc id {doc_id!r}")
    return best


def root(cid: str) -> Path:
    """Absolute data root for collection cid. KeyError if cid is unknown."""
    return ROOT / load()[cid]["root"]


def record_url(entry: dict, doc_id: str) -> str:
    """Public record URL for doc_id, with the entry's doc_prefix stripped first."""
    prefix = entry["doc_prefix"]
    bare = doc_id[len(prefix):] if prefix and doc_id.startswith(prefix) else doc_id
    return entry["record_url"].replace("{id}", bare)
