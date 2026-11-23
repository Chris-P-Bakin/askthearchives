"""Build catalog.db: discovery metadata over NARA description mirrors.

Discovery is metadata search, not passage retrieval, so this DB is separate
from chunks.db. Tables:
  archives(naId PK, level, title, parent_naId, start_year, end_year, n_items,
           n_digitized, n_objects, object_types, collection, scope_note)
    + archives_fts (external content over title, scope_note)
  items(doc_id PK, naId, title, digitized, rights, our_status)   -- ingested only
    + items_fts (external content over title)

Every registry collection gets a level='collection' row (negative synthetic
naId) so list_collections/search_catalog see naca too. RG/series rows are
built for every mirror dir; item rows only for collections in `ingested`.
Records at levels other than recordGroup/series (item, fileUnit, ...) count
as items. Rebuild writes a temp DB and atomically replaces the target.
"""
import argparse
import importlib.util
import json
import os
import sqlite3
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_AGG = ("recordGroup", "series")


def _registry():
    # pipeline/registry module (renamed from collections.py, which shadowed the stdlib name); load by path.
    mod = sys.modules.get("collection_registry")
    if mod is None:
        spec = importlib.util.spec_from_file_location("collection_registry", _HERE / "collections_registry.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["collection_registry"] = mod
        spec.loader.exec_module(mod)
    return mod


SCHEMA = """
CREATE TABLE archives(naId INTEGER PRIMARY KEY, level TEXT, title TEXT, parent_naId INTEGER,
  start_year INT, end_year INT, n_items INT, n_digitized INT, n_objects INT,
  object_types TEXT, collection TEXT, scope_note TEXT);
CREATE TABLE items(doc_id TEXT PRIMARY KEY, naId INT, title TEXT, digitized INT,
  rights TEXT, our_status TEXT);
CREATE VIRTUAL TABLE archives_fts USING fts5(title, scope_note, content='archives', content_rowid='naId');
CREATE VIRTUAL TABLE items_fts USING fts5(title, content='items');
CREATE INDEX archives_parent ON archives(parent_naId);
CREATE INDEX archives_collection ON archives(collection);
CREATE INDEX items_naid ON items(naId);
"""


def _year(rec, key):
    d = rec.get(key)
    y = d.get("year") if isinstance(d, dict) else None
    return y if isinstance(y, int) else None


def _parent(rec):
    best = None
    for a in rec.get("ancestors") or []:
        if isinstance(a, dict) and isinstance(a.get("naId"), int):
            if best is None or a.get("distance", 99) < best.get("distance", 99):
                best = a
    return best["naId"] if best else None


def _digital(rec):
    d = rec.get("digitalObjects")
    return [o for o in d if isinstance(o, dict)] if isinstance(d, list) else []


def _row(na_id, level, title, parent, start, end, coll, scope=None):
    return dict(naId=na_id, level=level, title=title, parent=parent, start=start, end=end,
                n_items=0, n_digitized=0, n_objects=0, types=set(), collection=coll, scope=scope)


def build(db_path, meta_dirs: dict, ingested: set, done: dict | None = None) -> dict:
    """Rebuild db_path from meta_dirs {collection_id: dir of *.jsonl}.

    done: optional {collection_id: set(doc_id)} of transcribed docs -> items.our_status.
    """
    reg = _registry()
    done = done or {}
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_name(db_path.name + ".tmp")
    for p in (tmp, Path(str(tmp) + "-journal")):
        p.unlink(missing_ok=True)
    counts = dict(scanned=0, corrupt=0, archives=0, items=0, collections=0)
    con = sqlite3.connect(tmp)
    try:
        con.executescript(SCHEMA)
        arch: dict[int, dict] = {}
        for cid, mdir in meta_dirs.items():
            prefix = reg.load()[cid]["doc_prefix"]
            keep_items = cid in ingested
            done_ids = done.get(cid, set())
            for f in sorted(Path(mdir).glob("*.jsonl")):
                with open(f, encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        counts["scanned"] += 1
                        try:
                            rec = json.loads(line)["record"]
                            na_id = rec["naId"]
                            level = rec["levelOfDescription"]
                            if not isinstance(na_id, int) or not isinstance(level, str):
                                raise ValueError("shape")
                        except (ValueError, KeyError, TypeError):
                            counts["corrupt"] += 1
                            continue
                        ancs = [a for a in (rec.get("ancestors") or [])
                                if isinstance(a, dict) and isinstance(a.get("naId"), int)]
                        for a in ancs:  # ancestors seed rows the mirror may lack
                            if a.get("levelOfDescription") in _AGG and a["naId"] not in arch:
                                arch[a["naId"]] = _row(
                                    a["naId"], a["levelOfDescription"], a.get("title"), None,
                                    _year(a, "inclusiveStartDate"), _year(a, "inclusiveEndDate"), cid)
                        if level in _AGG:
                            r = arch.setdefault(na_id, _row(na_id, level, None, None, None, None, cid))
                            r.update(level=level, title=rec.get("title"), parent=_parent(rec),
                                     start=_year(rec, "inclusiveStartDate"),
                                     end=_year(rec, "inclusiveEndDate"), collection=cid,
                                     scope=rec.get("scopeAndContentNote"))
                            continue
                        objs = _digital(rec)
                        for a in ancs:
                            if a.get("levelOfDescription") in _AGG:
                                r = arch[a["naId"]]
                                r["n_items"] += 1
                                r["n_digitized"] += 1 if objs else 0
                                r["n_objects"] += len(objs)
                                r["types"].update(o["objectType"] for o in objs
                                                  if isinstance(o.get("objectType"), str))
                        if keep_items:
                            doc_id = f"{prefix}{na_id}"
                            con.execute(
                                "INSERT OR REPLACE INTO items VALUES (?,?,?,?,?,?)",
                                (doc_id, na_id, rec.get("title"), 1 if objs else 0,
                                 (rec.get("useRestriction") or {}).get("status"),
                                 "transcribed" if doc_id in done_ids else None))
                            counts["items"] += 1
        for r in arch.values():
            con.execute("INSERT INTO archives VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (r["naId"], r["level"], r["title"], r["parent"], r["start"], r["end"],
                         r["n_items"], r["n_digitized"], r["n_objects"],
                         ",".join(sorted(r["types"])) or None, r["collection"], r["scope"]))
        counts["archives"] = len(arch)
        for i, (cid, e) in enumerate(sorted(reg.load().items()), 1):
            con.execute("INSERT INTO archives VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (-i, "collection", e["title"], None, None, None, None, None, None,
                         None, cid, f"{e['title']} ({e['source']})"))
            counts["collections"] += 1
        con.execute("INSERT INTO archives_fts(archives_fts) VALUES('rebuild')")
        con.execute("INSERT INTO items_fts(items_fts) VALUES('rebuild')")
        con.commit()
    except BaseException:
        con.close()
        tmp.unlink(missing_ok=True)
        raise
    con.close()
    os.replace(tmp, db_path)
    return counts


def _read_done(cid) -> set:
    p = _registry().root(cid) / "ingest_done.jsonl"
    if not p.exists():
        return set()
    return {json.loads(l)["doc_id"] for l in p.read_text().splitlines() if l.strip()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=str(_registry().ROOT / "data" / "catalog.db"))
    ap.add_argument("--meta", action="append", default=[], metavar="CID=DIR",
                    help="collection id and its JSONL mirror dir (repeatable)")
    ap.add_argument("--ingested", action="append", default=[], metavar="CID",
                    help="collection whose item rows are kept (repeatable)")
    a = ap.parse_args(argv)
    meta = {}
    for spec in a.meta:
        cid, _, d = spec.partition("=")
        meta[cid] = Path(d)
    done = {cid: _read_done(cid) for cid in a.ingested}
    print(json.dumps(build(a.db, meta, set(a.ingested), done), sort_keys=True))


if __name__ == "__main__":
    main()
