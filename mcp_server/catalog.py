"""Read-only discovery over catalog.db (built by pipeline/catalog_build.py).

Collection-level rows carry synthetic negative naIds that can renumber across
rebuilds, so they are only ever addressed through the `collection` column.
"""
import sqlite3
from pathlib import Path

from mcp_server.db import NotFoundError, _fts_query

LEVELS = ("recordGroup", "series", "collection")
RECORD_URL = "https://catalog.archives.gov/id/{}"
CHILD_LIMIT = 50


class CatalogMissing(NotFoundError):
    def __init__(self):
        super().__init__("catalog not built: the archive catalog index is "
                         "unavailable on this server")


def _conn(path: Path) -> sqlite3.Connection:
    if not Path(path).exists():
        raise CatalogMissing()
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _years(r) -> str | None:
    s, e = r["start_year"], r["end_year"]
    if s is None and e is None:
        return None
    if s is None:
        return str(e)
    return str(s) if e is None or s == e else f"{s}-{e}"


def aggregates(path: Path) -> dict[str, dict]:
    """Per-collection catalog aggregates; {} when the catalog is absent."""
    try:
        conn = _conn(path)
    except CatalogMissing:
        return {}
    try:
        out: dict[str, dict] = {}
        for r in conn.execute(
                "SELECT collection, level, COUNT(*) n, SUM(n_items) i, SUM(n_digitized) d "
                "FROM archives WHERE level IN ('recordGroup','series') "
                "GROUP BY collection, level"):
            a = out.setdefault(r["collection"], {"n_record_groups": 0, "n_series": 0,
                                                  "n_items": 0, "n_digitized": 0})
            if r["level"] == "recordGroup":  # RG totals already contain their series
                a["n_record_groups"] = r["n"]
                a["n_items"] = r["i"] or 0
                a["n_digitized"] = r["d"] or 0
            else:
                a["n_series"] = r["n"]
        return out
    finally:
        conn.close()


def describe(path: Path, na_id: int, collection_status: dict[str, str]) -> dict:
    conn = _conn(path)
    try:
        if na_id <= 0:
            raise NotFoundError(f"naId {na_id} is not a valid archive id")
        r = conn.execute("SELECT * FROM archives WHERE naId=?", (na_id,)).fetchone()
        if r is None:
            raise NotFoundError(f"naId {na_id} not found in the catalog")
        kids = conn.execute(
            "SELECT naId, level, title, start_year, end_year, n_items, n_digitized "
            "FROM archives WHERE parent_naId=? ORDER BY n_digitized DESC, naId LIMIT ?",
            (na_id, CHILD_LIMIT)).fetchall()
        n_kids = conn.execute("SELECT COUNT(*) FROM archives WHERE parent_naId=?",
                              (na_id,)).fetchone()[0]
        return {
            "naId": r["naId"], "level": r["level"], "title": r["title"],
            "scope_note": r["scope_note"], "collection": r["collection"],
            "parent_naId": r["parent_naId"], "years": _years(r),
            "n_items": r["n_items"], "n_digitized": r["n_digitized"],
            "n_objects": r["n_objects"],
            "object_types": r["object_types"].split(",") if r["object_types"] else [],
            "our_status": collection_status.get(r["collection"]),
            "record_url": RECORD_URL.format(r["naId"]),
            "children": {"total": n_kids, "shown": len(kids), "items": [
                {"naId": k["naId"], "level": k["level"], "title": k["title"],
                 "years": _years(k), "n_items": k["n_items"],
                 "n_digitized": k["n_digitized"]} for k in kids]},
        }
    finally:
        conn.close()


def search(path: Path, query: str, level: str | None, digitized_only: bool,
           limit: int, collection_status: dict[str, str]) -> list[dict]:
    if level is not None and level not in LEVELS:
        raise NotFoundError(f"unknown level {level!r}; valid levels: {list(LEVELS)}")
    conn = _conn(path)
    try:
        sql = ("SELECT a.* FROM archives_fts f JOIN archives a ON a.naId = f.rowid "
               "WHERE archives_fts MATCH ?")
        args: list = [_fts_query(query)]
        if level:
            sql += " AND a.level = ?"
            args.append(level)
        if digitized_only:
            sql += " AND a.n_digitized > 0"
        sql += " ORDER BY f.rank LIMIT ?"
        args.append(limit)
        return [{
            "naId": r["naId"], "level": r["level"], "title": r["title"],
            "collection": r["collection"], "years": _years(r),
            "n_items": r["n_items"], "n_digitized": r["n_digitized"],
            "our_status": collection_status.get(r["collection"]),
            "record_url": RECORD_URL.format(r["naId"]) if r["naId"] > 0 else None,
        } for r in conn.execute(sql, args)]
    finally:
        conn.close()
