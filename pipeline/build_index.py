"""Build corpus.db: the MCP server's index over the pipeline's data layer.

One row per page with the best-available text (ground truth > vision_27bq4 > vision2 > vision_haiku2 >
tesseract, recording source/quality/config_hash), FTS5 keyword index, and
(unless --no-embed) sqlite-vec embeddings via fastembed. Incremental:
a page re-indexes only when its chosen source file's mtime or config hash
changed. Safe to re-run while the pipeline writes new pages.
"""
import argparse, json, sqlite3, struct, sys, time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, load_reports, report_dir, log

SCHEMA_VERSION = "1"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384


@dataclass
class PageText:
    text: str
    source: str
    quality: str
    config_hash: str
    src_path: Path
    src_mtime: float
    flagged: bool = False


_REVIEW_CACHE: dict = {}


def _review_pages(path) -> set:
    """(ntrs_id, page) pairs in the escalation review file; cached by mtime."""
    path = Path(path)
    try:
        key = (str(path), path.stat().st_mtime)
    except OSError:
        return set()
    hit = _REVIEW_CACHE.get(str(path))
    if hit and hit[0] == key:
        return hit[1]
    out = set()
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
            out.add((str(r["ntrs_id"]), int(r["page"])))
        except (ValueError, KeyError, TypeError):
            continue
    _REVIEW_CACHE[str(path)] = (key, out)
    return out


_RESOLVES = {"table_flags": ("vision_235b",),
             "vision_failed": ("vision_235b", "vision_27b_v2")}


def _queue_rows(path) -> dict:
    """(ntrs_id, page) -> set of pending kinds in escalation_queue.jsonl."""
    path = Path(path)
    try:
        key = (str(path), path.stat().st_mtime)
    except OSError:
        return {}
    hit = _REVIEW_CACHE.get("q:" + str(path))
    if hit and hit[0] == key:
        return hit[1]
    out: dict = {}
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
            if r.get("kind") in _RESOLVES:
                out.setdefault((str(r["ntrs_id"]), int(r["page"])), set()).add(r["kind"])
        except (ValueError, KeyError, TypeError):
            continue
    _REVIEW_CACHE["q:" + str(path)] = (key, out)
    return out


def _meta_flags(m: dict) -> bool:
    tv = m.get("table_verify")
    if isinstance(tv, dict) and tv.get("flags"):
        return True
    return bool(m.get("table_flags"))


def _campaign_text(rdir: Path, n: int):
    """Campaign layers in priority order; a layer counts only if its meta passes
    its predicate and the loop guard did not mark the page failed."""
    import escalate, units
    gates = (
        ("vision_235b", escalate.is_v2_235b_meta),
        (escalate.LAYER_27B, lambda m: m.get("config_hash") == escalate.ESC_27B_HASH),
        ("vision_8blora_v2_1", lambda m: m.get("config_hash") in units.ACCEPTED_HASHES),
    )
    for layer, ok in gates:
        v, mp = rdir / layer / f"{n}.txt", rdir / layer / f"{n}.meta.json"
        if not (v.exists() and mp.exists()):
            continue
        try:
            m = json.loads(mp.read_text())
        except ValueError:
            continue
        if not isinstance(m, dict) or not ok(m):
            continue
        if (m.get("loop_guard") or {}).get("vision_failed") is True:
            continue
        return PageText(v.read_text(), layer, "vision", m.get("config_hash", ""),
                        v, v.stat().st_mtime, flagged=_meta_flags(m))
    return None


def best_text(rdir: Path, n: int, campaign_only: bool = False,
              review_path=None, queue_path=None) -> PageText | None:
    pt = _best_text(rdir, n, campaign_only)
    if pt is not None:
        if review_path is None:
            import escalate
            review_path = escalate.REVIEW_QUEUE
        if (Path(rdir).name, n) in _review_pages(review_path):
            pt.flagged = True
        if queue_path is None:
            import escalate
            queue_path = escalate.QUEUE
        for kind in _queue_rows(queue_path).get((Path(rdir).name, n), ()):
            if pt.source not in _RESOLVES[kind]:
                pt.flagged = True
    return pt


def _best_text(rdir: Path, n: int, campaign_only: bool) -> PageText | None:
    gt = rdir / "ground_truth" / f"{n}.txt"
    gt_meta = rdir / "ground_truth" / f"{n}.json"
    if gt.exists() and gt_meta.exists():
        if json.loads(gt_meta.read_text()).get("status") == "done":
            return PageText(gt.read_text(), "ground_truth", "verified",
                            "human", gt, gt.stat().st_mtime)
    pt = _campaign_text(rdir, n)
    if pt is not None:
        return pt
    if not campaign_only:
        rec_p = rdir / "reconcile" / f"{n}.json"
        if rec_p.exists():
            rec = json.loads(rec_p.read_text())
            v = rdir / rec["page_source"] / f"{n}.txt"
            if v.exists():
                return PageText(v.read_text(), rec["page_source"],
                                "vision_" + rec["confidence"],
                                rec.get("decided_by", ""), v, v.stat().st_mtime)
        for vdir in ("vision_27bq4", "vision2", "vision_haiku2"):
            v = rdir / vdir / f"{n}.txt"
            v_meta = rdir / vdir / f"{n}.meta.json"
            if v.exists() and v_meta.exists():
                m = json.loads(v_meta.read_text())
                if not (m.get("loop_guard") or {}).get("vision_failed"):
                    return PageText(v.read_text(), vdir, "vision",
                                    m.get("config_hash", ""), v, v.stat().st_mtime)
    oj = rdir / "ocr" / f"{n}.json"
    if oj.exists():
        o = json.loads(oj.read_text())
        return PageText(o.get("text", ""), "tesseract", "ocr_draft",
                        o.get("config_hash", ""), oj, oj.stat().st_mtime)
    return None


DDL = """
CREATE TABLE IF NOT EXISTS reports(
  doc_id TEXT PRIMARY KEY, ntrs_id TEXT, report_number TEXT, title TEXT,
  series TEXT, decade TEXT, year TEXT, pages INTEGER, source_url TEXT);
CREATE TABLE IF NOT EXISTS pages(
  doc_id TEXT, page INTEGER, text TEXT, source TEXT, quality TEXT,
  config_hash TEXT, chars INTEGER, src_mtime REAL,
  PRIMARY KEY(doc_id, page));
CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
  text, content='pages', content_rowid='rowid');
CREATE TABLE IF NOT EXISTS docs(
  ntrs_id TEXT PRIMARY KEY, disposition TEXT, pages INTEGER, title TEXT, year TEXT);
CREATE TABLE IF NOT EXISTS index_meta(key TEXT PRIMARY KEY, value TEXT);
"""


def _embedder():
    from fastembed import TextEmbedding
    return TextEmbedding(EMBED_MODEL)


def embed_texts(texts: list[str], model=None) -> list[list[float]]:
    model = model or _embedder()
    return [e.tolist() for e in model.embed(texts, batch_size=32)]


def _vec(v: list[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


def _vec_delete(conn, rowid: int) -> None:
    """Remove a page's vector regardless of this run's embed setting, so
    mixed embed/--no-embed runs never leave stale or orphaned vectors."""
    try:
        conn.execute("DELETE FROM pages_vec WHERE rowid=?", (rowid,))
    except sqlite3.OperationalError:
        pass  # pages_vec doesn't exist yet


def _write_doc_row(conn, nid, rep, rdir, pdir) -> None:
    """docs row for every roster id: pages-bearing, no_pdf, or still pending."""
    meta = {}
    mp = rdir / "meta.json"
    if mp.exists():
        try:
            meta = json.loads(mp.read_text())
        except ValueError:
            meta = {}
    csv_row = meta.get("csv_row") or {}
    npages = (len(list(pdir.glob("[0-9]*.png"))) - len(list(pdir.glob("*.raw.png")))
              if pdir.is_dir() else 0)
    if meta.get("pdf_status") == "no_pdf_in_record" or meta.get("disposition") == "no_pdf":
        disp = "no_pdf"
    elif npages:
        disp = "ok"
    else:
        disp = meta.get("disposition") or "pending"
    title = rep.get("title") or csv_row.get("title", "")
    year = (rep.get("year") or (rep.get("pub_date") or "")[:4]
            or rep.get("published_year") or csv_row.get("published_year")
            or csv_row.get("year") or None)
    conn.execute("INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?)",
                 (nid, disp, npages, title, year))


def ingested_ids(data_dir: Path) -> list[str]:
    """Sorted ids of data/<id>/ dirs that have a pages/ subdirectory."""
    return sorted(d.name for d in Path(data_dir).iterdir()
                  if d.is_dir() and (d / "pages").is_dir())


def roster_meta(csv_path: Path, ids: list[str]) -> list[dict]:
    """Roster rows for ids; an id absent from the roster is fatal."""
    by_id = {r["ntrs_id"]: r for r in load_reports(str(csv_path))}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"ingested ids missing from roster {csv_path}: "
                         + ", ".join(missing))
    return [by_id[i] for i in ids]


def build(db_path: Path, data_dir: Path, ids: list[str] | None = None,
          rebuild: bool = False, embed: bool = True,
          campaign_only: bool = False, shard_meta: dict | None = None,
          review_path=None, queue_path=None,
          reports: list[dict] | None = None) -> dict:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    # Always load the vec extension: drops/deletes on an existing pages_vec
    # need the vec0 module even in --no-embed runs.
    import sqlite_vec
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    if rebuild:
        for t in ("pages_vec", "pages_fts", "pages", "reports", "docs", "index_meta"):
            conn.execute(f"DROP TABLE IF EXISTS {t}")
    conn.executescript(DDL)
    if embed:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS pages_vec "
                     f"USING vec0(embedding float[{EMBED_DIM}])")

    if reports is None:
        reports = load_reports()
    if ids:
        want = set(ids)
        reports = [r for r in reports if r["ntrs_id"] in want]
    counts = {"indexed": 0, "skipped": 0, "pages": 0}
    seen_ids = {r["ntrs_id"] for r in reports}
    for extra in (ids or []):  # ids passed but absent from the roster CSV
        if extra not in seen_ids:
            reports.append({"ntrs_id": extra})
    model = None
    for rep in reports:
        nid = rep["ntrs_id"]
        rdir = Path(data_dir) / nid
        pdir = rdir / "pages"
        _write_doc_row(conn, nid, rep, rdir, pdir)
        if not pdir.is_dir():
            continue
        conn.execute(
            "INSERT OR REPLACE INTO reports VALUES (?,?,?,?,?,?,?,?,?)",
            (nid, nid, rep.get("report_number", ""), rep.get("title", ""),
             rep.get("series", ""), rep.get("decade", ""),
             rep.get("year") or rep.get("pub_date", "")[:4] or None,
             len(list(pdir.glob("[0-9]*.png"))) -
             len(list(pdir.glob("*.raw.png"))),
             rep.get("pdf_url") or rep.get("source_url", "")))
        page_nums = sorted(int(p.name.split(".")[0]) for p in pdir.glob("*.png")
                           if ".raw" not in p.name)
        todo = []  # (rowid, page, text) needing embedding
        for n in page_nums:
            pt = best_text(rdir, n, campaign_only=campaign_only,
                           review_path=review_path, queue_path=queue_path)
            if pt is None:
                continue
            counts["pages"] += 1
            row = conn.execute(
                "SELECT rowid, src_mtime, config_hash, source FROM pages "
                "WHERE doc_id=? AND page=?", (nid, n)).fetchone()
            if row and row[1] == pt.src_mtime and row[2] == pt.config_hash \
                    and row[3] == pt.source:
                counts["skipped"] += 1
                if embed and not conn.execute(
                        "SELECT 1 FROM pages_vec WHERE rowid=?",
                        (row[0],)).fetchone():
                    # page was indexed by a --no-embed run: backfill its vector
                    todo.append((row[0],
                                 f"{rep.get('report_number','')} "
                                 f"{rep.get('title','')}\n{pt.text[:1500]}"))
                continue
            if row:
                conn.execute("INSERT INTO pages_fts(pages_fts, rowid, text) "
                             "VALUES('delete', ?, ?)",
                             (row[0], conn.execute(
                                 "SELECT text FROM pages WHERE rowid=?",
                                 (row[0],)).fetchone()[0]))
                conn.execute("DELETE FROM pages WHERE rowid=?", (row[0],))
                _vec_delete(conn, row[0])
            cur = conn.execute(
                "INSERT INTO pages VALUES (?,?,?,?,?,?,?,?)",
                (nid, n, pt.text, pt.source, pt.quality, pt.config_hash,
                 len(pt.text), pt.src_mtime))
            rowid = cur.lastrowid
            conn.execute("INSERT INTO pages_fts(rowid, text) VALUES (?,?)",
                         (rowid, pt.text))
            counts["indexed"] += 1
            if embed:
                todo.append((rowid, f"{rep.get('report_number','')} "
                                    f"{rep.get('title','')}\n{pt.text[:1500]}"))
        if embed and todo:
            model = model or _embedder()
            vecs = embed_texts([t for _, t in todo], model)
            conn.executemany(
                "INSERT INTO pages_vec(rowid, embedding) VALUES (?,?)",
                [(rid, _vec(v)) for (rid, _), v in zip(todo, vecs)])
        conn.commit()
    for k, v in (("schema_version", SCHEMA_VERSION), ("embed_model", EMBED_MODEL),
                 ("embed_dim", str(EMBED_DIM)),
                 ("campaign_only", str(bool(campaign_only))),
                 *((("shard_meta", json.dumps(shard_meta)),) if shard_meta else ()),
                 ("built_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))):
        conn.execute("INSERT OR REPLACE INTO index_meta VALUES (?,?)", (k, v))
    # sweep any orphans left by historical mixed-mode runs
    try:
        conn.execute("DELETE FROM pages_vec WHERE rowid NOT IN "
                     "(SELECT rowid FROM pages)")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    # Leave a single self-contained file: read-only consumers (the MCP
    # container mounts the db ro) cannot open a WAL database.
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.close()
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--ids", help="comma-separated ntrs ids")
    ap.add_argument("--all-ingested", action="store_true",
                    help="index every data/<id>/ with pages/, metadata from --roster")
    ap.add_argument("--roster", default=str(DATA / "roster.csv"))
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--campaign-only", action="store_true")
    ap.add_argument("--db", default=str(DATA / "corpus.db"))
    args = ap.parse_args()
    if args.all_ingested and args.ids:
        ap.error("--all-ingested and --ids are mutually exclusive")
    ids = args.ids.split(",") if args.ids else None
    reports = None
    if args.all_ingested:
        ids = ingested_ids(DATA)
        reports = roster_meta(Path(args.roster), ids)
    counts = build(Path(args.db), DATA, ids=ids, reports=reports,
                   rebuild=args.rebuild, embed=not args.no_embed,
                   campaign_only=args.campaign_only)
    log("index_built", **counts, db=args.db)


if __name__ == "__main__":
    main()
