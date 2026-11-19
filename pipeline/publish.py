"""Batch exporter: assemble an immutable, campaign-only publication tree.

data/publish/batch_NNN/
  {ntrs_id}/source.pdf        hardlink of the NTRS PDF
  {ntrs_id}/pages/{n}.png     hardlinks of the deskewed page images
  {ntrs_id}/ocr/{n}.json      hardlinks of the tesseract word-box layer
  {ntrs_id}/text/{n}.txt      campaign best_text (campaign layers > tesseract)
  {ntrs_id}/text/{n}.meta.json provenance: source layer, config_hash, quality, flagged
  {ntrs_id}/flags.jsonl       that doc's escalation review + pending queue rows
  index_shard.db              Task-1 campaign-only index over the batch (no embeddings)
  BATCH_README.md             provenance statement
  manifest.jsonl              path/sha256/bytes for every other file; written LAST
                              (its presence is the commit marker; reruns skip
                              files already present and byte-identical)
Legacy text layers (vision_27bq4/vision2/vision_haiku2/reconcile) never enter the tree.
"""
import argparse, hashlib, json, os, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import escalate
from build_index import _RESOLVES, best_text, build

README = """# NACA reports, batch {batch:03d}

{docs} documents ({ok} with pages, {no_pdf} no_pdf{pending_txt}); {pages} pages with text. Public domain: released under CC0 1.0.

## Layout

Per report `{{ntrs_id}}/`: `source.pdf` (NTRS original), `pages/{{n}}.png` (page images),
`ocr/{{n}}.json` (tesseract word boxes), `text/{{n}}.txt` plus `text/{{n}}.meta.json`
(best text and its provenance), and `flags.jsonl` when any page is flagged.
`docs.jsonl` has one row per document in the batch (disposition, pages exported, title, year),
including those without pages. `index_shard.db` is a SQLite FTS5 index of the text; `manifest.jsonl` lists every file
with its sha256 and byte size, paths relative to this directory.

## Provenance

Text is machine transcription, not hand-corrected, drawn only from the campaign engines:
the 8B-LoRA campaign engine (config hash `6ecf7d0dbeb2`, ground-truth gate passed at
median CER 0.0940), the 27B escalation engine (config hash `6a169a212aac`, gate passed at
median CER 0.0975), and the 235B escalation engine under its v2 predicate (prompt v2,
loop guard v2). Gate threshold was median CER <= 0.1032 on 30 hand-verified pages. Pages
no campaign engine produced fall back to the raw tesseract OCR layer
(`source` in each `text/{{n}}.meta.json` says which). Earlier trial engines are excluded.

## Flags

A page with `"flagged": true` is published deliberately, not withheld: its text may contain
errors, most often wrong digits in tables. `{{ntrs_id}}/flags.jsonl` carries the rows behind
each flag, verbatim, with `status` "review" (escalation exhausted, needs a human) or
"pending" (queued for further escalation when exported). Treat flagged numbers as unverified.

## Citing

Cite the NASA NTRS id of the report (the directory name), for example
`NTRS 19930091026`, and the original NACA report number. The page images here are the
citable originals; the text is a derived reading aid.
"""


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _link(src: Path, dst: Path) -> bool:
    """Hardlink src to dst; True if it created/replaced the link."""
    if dst.exists():
        if os.path.samefile(src, dst):
            return False
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, dst)
    return True


def _write(dst: Path, data: str | bytes) -> bool:
    """Write fresh; True if the file is new or its content changed."""
    b = data.encode() if isinstance(data, str) else data
    if b"/home/" in b:
        raise ValueError(f"absolute local path would be exported in {dst.name}")
    if dst.exists() and dst.read_bytes() == b:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_bytes(b)
    os.replace(tmp, dst)
    return True


def _rows(path) -> dict:
    out: dict = {}
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            r = json.loads(line)
            out.setdefault(str(r["ntrs_id"]), []).append(r)
        except (ValueError, KeyError, TypeError):
            continue
    return out


def _verify(root: Path) -> list:
    """Paths whose manifest sha256/size no longer matches (empty = intact)."""
    bad = []
    for line in (root / "manifest.jsonl").read_text().splitlines():
        r = json.loads(line)
        p = root / r["path"]
        if not p.is_file() or p.stat().st_size != r["bytes"] or _sha(p) != r["sha256"]:
            bad.append(r["path"])
    return bad


def _queue_status(row: dict, psrc: dict) -> str:
    """Same supersession rule as build_index.best_text: a queue row is resolved when the
    page's chosen source is one of the layers that resolves its kind."""
    try:
        src = psrc.get(int(row.get("page")))
    except (TypeError, ValueError):
        src = None
    return "resolved" if src in _RESOLVES.get(row.get("kind"), ()) else "pending"


def export_batch(batch: int, out_root=None, data_dir=None, review_path=None,
                 queue_path=None, batches_dir=None, force: bool = False) -> dict:
    """A batch whose manifest.jsonl exists is COMMITTED: verified, never rewritten
    (corrections are errata's job). force=True is only for a deliberate re-export
    and still requires the operator to have deleted manifest.jsonl first."""
    data_dir = Path(data_dir or common.DATA)
    out_root = Path(out_root or data_dir / "publish")
    batches_dir = Path(batches_dir or data_dir / "batches")
    review_path = Path(review_path or escalate.REVIEW_QUEUE)
    queue_path = Path(queue_path or escalate.QUEUE)
    ids = json.loads((batches_dir / f"batch_{batch:03d}.json").read_text())["ids"]
    root = out_root / f"batch_{batch:03d}"
    if (root / "manifest.jsonl").exists():
        if force:
            raise SystemExit("--force requires manifest.jsonl to be absent: delete the "
                             "batch tree deliberately before re-exporting")
        bad = _verify(root)
        return {"batch": batch, "committed": True, "new_files": 0,
                "verified": not bad, "mismatches": bad, "root": str(root)}
    root.mkdir(parents=True, exist_ok=True)
    review, queue = _rows(review_path), _rows(queue_path)
    new = 0
    pages = flagged = no_pages = 0
    sources: Counter = Counter()
    for nid in ids:
        psrc: dict = {}
        rdir, dest = data_dir / nid, root / nid
        pdir = rdir / "pages"
        if not pdir.is_dir():
            no_pages += 1
            continue
        if (rdir / "source.pdf").exists():
            new += _link(rdir / "source.pdf", dest / "source.pdf")
        nums = sorted(int(p.name.split(".")[0]) for p in pdir.glob("[0-9]*.png")
                      if ".raw" not in p.name)
        for n in nums:
            new += _link(pdir / f"{n}.png", dest / "pages" / f"{n}.png")
            if (rdir / "ocr" / f"{n}.json").exists():
                new += _link(rdir / "ocr" / f"{n}.json", dest / "ocr" / f"{n}.json")
            pt = best_text(rdir, n, campaign_only=True,
                           review_path=review_path, queue_path=queue_path)
            if pt is None:
                continue
            pages += 1
            flagged += bool(pt.flagged)
            sources[pt.source] += 1
            psrc[n] = pt.source
            new += _write(dest / "text" / f"{n}.txt", pt.text)
            new += _write(dest / "text" / f"{n}.meta.json", json.dumps(
                {"page": n, "source": pt.source, "config_hash": pt.config_hash,
                 "quality": pt.quality, "flagged": bool(pt.flagged)}, indent=1) + "\n")
        rows = ([dict(r, status="review") for r in review.get(nid, [])]
                + [dict(r, status=_queue_status(r, psrc)) for r in queue.get(nid, [])
                   if r.get("kind") != "wu_error"])  # wu_error = ops artifact, not a flag
        if rows:
            new += _write(dest / "flags.jsonl",
                          "".join(json.dumps(r) + "\n" for r in rows))
    db = root / "index_shard.db"
    # uncommitted here (manifest absent): always rebuild, never trust a crash-partial db
    for suf in ("", "-wal", "-shm"):
        Path(str(db) + suf).unlink(missing_ok=True)
    build(db, data_dir, ids=ids, embed=False, campaign_only=True,
          shard_meta={"batch": batch}, review_path=review_path,
          queue_path=queue_path)
    new += 1
    import sqlite3
    c = sqlite3.connect(db)
    drows = {r[0]: r for r in c.execute("SELECT ntrs_id, disposition, title, year FROM docs")}
    c.close()
    docs = []
    for nid in ids:
        r = drows.get(nid, (nid, "pending", "", None))
        n = len(list((root / nid / "pages").glob("*.png"))) if (root / nid / "pages").is_dir() else 0
        docs.append({"ntrs_id": nid, "disposition": r[1], "pages_exported": n,
                     "title": r[2], "year": r[3]})
    new += _write(root / "docs.jsonl", "".join(json.dumps(d) + "\n" for d in docs))
    cnt = Counter(d["disposition"] for d in docs)
    other = len(docs) - cnt["ok"] - cnt["no_pdf"]
    new += _write(root / "BATCH_README.md", README.format(
        batch=batch, docs=len(docs), ok=cnt["ok"], no_pdf=cnt["no_pdf"],
        pending_txt=f", {other} pending" if other else "", pages=pages))
    man = "".join(
        json.dumps({"path": p.relative_to(root).as_posix(), "sha256": _sha(p),
                    "bytes": p.stat().st_size}) + "\n"
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name not in ("manifest.jsonl",) and not p.name.endswith(".tmp")
        and not p.name.startswith("index_shard.db-"))
    _write(root / "manifest.jsonl", man)  # last: commit marker
    return {"batch": batch, "docs": len(ids), "no_pages_docs": no_pages, "pages": pages,
            "flagged_pages": flagged, "sources": dict(sources), "new_files": new,
            "files": man.count("\n"), "committed": False, "root": str(root),
            "consulted": [review_path.name, queue_path.name]}


def _committed(out_root: Path) -> list:
    return sorted(d for d in out_root.glob("batch_[0-9][0-9][0-9]*")
                  if (d / "manifest.jsonl").is_file())


def _trackers(out_root: Path) -> list:
    f = out_root / "trackers.txt"
    if not f.is_file():
        return []
    return [u.strip() for u in f.read_text().splitlines()
            if u.strip() and not u.strip().startswith("#")]


def _torrent(path: Path, dest: Path, out_root: Path) -> Path:
    import torf
    t = torf.Torrent(path=path, trackers=_trackers(out_root) or None,
                     private=False, created_by="naca-publish", creation_date=None)
    t.generate()
    dest.parent.mkdir(parents=True, exist_ok=True)
    t.write(dest, overwrite=True)
    return dest


def _assert_intact(root: Path):
    """Refuse to torrent a batch tree that drifted from its manifest (changed bytes,
    missing files, or on-disk files the manifest does not list)."""
    bad = _verify(root)
    listed = {json.loads(l)["path"] for l in (root / "manifest.jsonl").read_text().splitlines()}
    extra = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                   if p.is_file() and p.name != "manifest.jsonl"
                   and p.relative_to(root).as_posix() not in listed)
    if bad or extra:
        raise SystemExit(f"{root.name} drifted from manifest.jsonl; refusing to torrent. "
                         f"mismatched/missing: {bad[:20]}; not in manifest: {extra[:20]}")


def make_torrent(batch: int, out_root=None, data_dir=None, **_) -> Path:
    """Torrent over one COMMITTED, verified batch tree (trackerless unless trackers.txt exists)."""
    out_root = Path(out_root or Path(data_dir or common.DATA) / "publish")
    root = out_root / f"batch_{batch:03d}"
    if not (root / "manifest.jsonl").is_file():
        raise SystemExit(f"batch {batch:03d} is not committed (no manifest.jsonl)")
    _assert_intact(root)
    return _torrent(root, out_root / "torrents" / f"batch_{batch:03d}.torrent", out_root)


def _ts(mtime: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


RANK = {"ground_truth": 4, "vision_235b": 3, "vision_27b_v2": 2,
        "vision_8blora_v2_1": 1, "tesseract": 0}


def _jl_rows(path: Path) -> list:
    try:
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    except OSError:
        return []


def _jl_write(path: Path, rows: list):
    rows = sorted({json.dumps(r, sort_keys=True) for r in rows})
    if rows:
        _write(path, "".join(r + "\n" for r in rows))
    else:
        path.unlink(missing_ok=True)


def _flag_reasons(nid: str, n: int, review: dict, queue: dict, pt) -> tuple:
    reasons, tss = [], []
    for r in review.get(nid, []):
        if int(r.get("page", -1)) == n:
            reasons.append(f"review:{r.get('reason', '')}")
            tss.append(str(r.get("ts", "")))
    for r in queue.get(nid, []):
        if int(r.get("page", -1)) == n:
            reasons.append(f"queue:{r.get('kind', '')}")
            tss.append(str(r.get("ts", "")))
    ts = max(tss) if tss else _ts(pt.src_mtime)
    return sorted(set(reasons)), ts


def gen_errata(batch: int, out_root=None, data_dir=None, review_path=None,
               queue_path=None, **_) -> int:
    """Overlay corrections for a committed batch; never touches the batch tree.
    - text differs and current layer ranks >= shipped: {n}.txt + {n}.meta.json (superseded)
    - text differs and current layer ranks LOWER: regressions.jsonl row, no overlay
    - flagged state differs from shipped meta: {n}.flags.json (no .txt if text unchanged)
    - overlay files from an earlier run that no longer apply: removed, tombstoned in
      retractions.jsonl. Deterministic: ts comes from file mtime / source rows, never now()."""
    data_dir = Path(data_dir or common.DATA)
    out_root = Path(out_root or data_dir / "publish")
    review_path = Path(review_path or escalate.REVIEW_QUEUE)
    queue_path = Path(queue_path or escalate.QUEUE)
    root = out_root / f"batch_{batch:03d}"
    if not (root / "manifest.jsonl").is_file():
        raise SystemExit(f"batch {batch:03d} is not committed (no manifest.jsonl)")
    review, queue = _rows(review_path), _rows(queue_path)
    ebase = out_root / "errata"
    edir = ebase / f"batch_{batch:03d}"
    keep, regress, count = set(), [], 0
    for line in (root / "manifest.jsonl").read_text().splitlines():
        r = json.loads(line)
        parts = r["path"].split("/")
        if len(parts) != 3 or parts[1] != "text" or not parts[2].endswith(".txt") \
                or not parts[2][:-4].isdigit():
            continue
        nid, n = parts[0], int(parts[2][:-4])
        pt = best_text(data_dir / nid, n, campaign_only=True,
                       review_path=review_path, queue_path=queue_path)
        if pt is None:
            continue
        try:
            sm = json.loads((root / nid / "text" / f"{n}.meta.json").read_text())
        except (OSError, ValueError):
            sm = {}
        ship_src = sm.get("source", "")
        if hashlib.sha256(pt.text.encode()).hexdigest() != r["sha256"]:
            if RANK.get(pt.source, -1) < RANK.get(ship_src, -1):
                regress.append({"batch": batch, "ntrs_id": nid, "page": n,
                                "shipped_source": ship_src, "current_source": pt.source,
                                "ts": _ts(pt.src_mtime)})
            else:
                _write(edir / nid / f"{n}.txt", pt.text)
                _write(edir / nid / f"{n}.meta.json", json.dumps(
                    {"page": n, "source": pt.source, "config_hash": pt.config_hash,
                     "quality": pt.quality, "flagged": bool(pt.flagged),
                     "reason": "superseded", "supersedes_sha256": r["sha256"],
                     "ts": _ts(pt.src_mtime)}, indent=1) + "\n")
                keep |= {edir / nid / f"{n}.txt", edir / nid / f"{n}.meta.json"}
                count += 1
        if bool(pt.flagged) != bool(sm.get("flagged", False)):
            reasons, ts = _flag_reasons(nid, n, review, queue, pt)
            _write(edir / nid / f"{n}.flags.json", json.dumps(
                {"page": n, "flagged": bool(pt.flagged), "was": bool(sm.get("flagged", False)),
                 "reasons": reasons, "ts": ts}, indent=1) + "\n")
            keep.add(edir / nid / f"{n}.flags.json")
            count += 1
    tomb = []
    if edir.is_dir():
        for p in sorted(edir.rglob("*")):
            if not p.is_file() or p in keep or p.name.endswith(".meta.json"):
                continue
            nid, n = p.parent.name, p.name.split(".")[0]
            kind = "flags" if p.name.endswith(".flags.json") else "text"
            ts = ""
            try:
                ts = json.loads((p if kind == "flags" else
                                 p.with_name(f"{n}.meta.json")).read_text()).get("ts", "")
            except (OSError, ValueError):
                pass
            tomb.append({"batch": batch, "ntrs_id": nid, "page": int(n), "kind": kind,
                         "retracted_sha256": _sha(p), "ts": ts})
        for p in sorted(edir.rglob("*"), reverse=True):
            if p.is_file() and p not in keep:
                p.unlink()
            elif p.is_dir() and not any(p.iterdir()):
                p.rmdir()
    if tomb:
        _jl_write(ebase / "retractions.jsonl", _jl_rows(ebase / "retractions.jsonl") + tomb)
    old = [r for r in _jl_rows(ebase / "regressions.jsonl") if r.get("batch") != batch]
    if old or regress:
        _jl_write(ebase / "regressions.jsonl", old + regress)
    return count


def make_cumulative(date: str | None = None, out_root=None, data_dir=None,
                    review_path=None, queue_path=None, batches_dir=None,
                    no_embed: bool = False, **_) -> Path:
    """Torrent over every committed batch + errata/ + corpus.db (merged campaign-only
    hybrid index WITH embeddings, as archives-mcp needs; no_embed=True is a test escape
    hatch). Errata are regenerated for every committed batch first (deterministic) so
    the shipped corpus.db never disagrees with the batch trees without an erratum.
    Staged as hardlinks in cumulative_YYYYMMDD/."""
    import shutil
    from datetime import date as _d
    data_dir = Path(data_dir or common.DATA)
    out_root = Path(out_root or data_dir / "publish")
    batches_dir = Path(batches_dir or data_dir / "batches")
    date = date or _d.today().strftime("%Y%m%d")
    name, k = f"cumulative_{date}", 1
    while (out_root / "torrents" / f"{name}.torrent").exists():
        k += 1
        name = f"cumulative_{date}-{k}"
    committed = _committed(out_root)
    if not committed:
        raise SystemExit("no committed batches")
    for b in committed:
        _assert_intact(b)
    for b in committed:
        gen_errata(int(b.name[6:9]), out_root=out_root, data_dir=data_dir,
                   review_path=review_path, queue_path=queue_path)
    ids: list = []
    for b in committed:
        for nid in json.loads((batches_dir / f"{b.name[:9]}.json").read_text())["ids"]:
            if nid not in ids:
                ids.append(nid)
    db = out_root / "corpus.db"
    for suf in ("", "-wal", "-shm"):
        Path(str(db) + suf).unlink(missing_ok=True)
    build(db, data_dir, ids=ids, embed=not no_embed, campaign_only=True,
          review_path=review_path or escalate.REVIEW_QUEUE,
          queue_path=queue_path or escalate.QUEUE)
    stage = out_root / name
    if stage.exists():
        shutil.rmtree(stage)

    def _stage(src: Path, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    srcs = [p for b in committed for p in b.rglob("*") if p.is_file()]
    if (out_root / "errata").is_dir():
        srcs += [p for p in (out_root / "errata").rglob("*") if p.is_file()]
    srcs.append(db)
    for p in sorted(srcs):
        _stage(p, stage / p.relative_to(out_root))
    return _torrent(stage, out_root / "torrents" / f"{name}.torrent", out_root)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--batch", type=int, required=True)
    e.add_argument("--out-root")
    e.add_argument("--force", action="store_true")
    t = sub.add_parser("torrent")
    t.add_argument("--batch", type=int)
    t.add_argument("--out-root")
    c = sub.add_parser("cumulative")
    c.add_argument("--out-root")
    g = sub.add_parser("errata")
    g.add_argument("--batch", type=int, required=True)
    g.add_argument("--out-root")
    args = ap.parse_args()
    if args.cmd == "torrent":
        if args.batch is None:
            ap.error("torrent requires --batch N (use 'cumulative' for the cumulative torrent)")
        print(make_torrent(args.batch, out_root=args.out_root)); return
    if args.cmd == "cumulative":
        print(make_cumulative(out_root=args.out_root)); return
    if args.cmd == "errata":
        print(json.dumps({"batch": args.batch,
                          "errata": gen_errata(args.batch, out_root=args.out_root)})); return
    print(json.dumps(export_batch(args.batch, out_root=args.out_root, force=args.force), indent=1))


if __name__ == "__main__":
    main()
