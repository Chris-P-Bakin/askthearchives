"""Staged index refresh: rebuild a copy of the live db, swap it in atomically.

The live db may be open in the MCP server at any moment, so it is never
written in place: build on `<db>.staging`, then os.replace (atomic, same dir).
"""
import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _signature(db: Path):
    """(pages rowcount, max page src_mtime) -- the content-change fingerprint."""
    if not Path(db).exists():
        return None
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return conn.execute("SELECT count(*), max(src_mtime) FROM pages").fetchone()
    except sqlite3.DatabaseError:
        return None
    finally:
        conn.close()


def _default_build(data_dir: Path, embed: bool = True):
    import build_index

    def run(staging: Path):
        ids = build_index.ingested_ids(data_dir)
        reports = build_index.roster_meta(Path(data_dir) / "roster.csv", ids)
        build_index.build(staging, data_dir, ids=ids, reports=reports, embed=embed)
    return run


def _rm(p: Path):
    try:
        p.unlink()
    except FileNotFoundError:
        pass


def refresh(db: Path, data_dir: Path, build_fn=None,
            restart_cmd: list[str] | None = None, embed: bool = True) -> bool:
    db, data_dir = Path(db), Path(data_dir)
    staging = db.with_suffix(".staging")
    for p in (staging, *(db.parent / (staging.name + s) for s in ("-wal", "-shm"))):
        _rm(p)
    if db.exists():
        shutil.copy2(db, staging)
    build_fn = build_fn or _default_build(data_dir, embed)
    try:
        build_fn(staging)
    except BaseException:
        for s in ("-wal", "-shm"):
            _rm(db.parent / (staging.name + s))
        raise
    before, after = _signature(db), _signature(staging)
    for s in ("-wal", "-shm"):
        _rm(db.parent / (staging.name + s))
    if after is None or not after[0]:
        _rm(staging)
        raise RuntimeError(f"refusing swap: staging index {staging} is "
                           "unreadable or has no pages; live db untouched")
    os.replace(staging, db)
    changed = before != after
    if changed and restart_cmd:
        subprocess.run(restart_cmd, check=True)
    return changed


def main():
    from common import DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA / "corpus.db"))
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--chunks-db", default=str(DATA / "chunks.db"))
    ap.add_argument("--no-chunks", action="store_true",
                    help="skip the incremental chunk refresh")
    ap.add_argument("--no-embed", action="store_true",
                    help="skip page-vector embedding in the corpus build")
    ap.add_argument("--restart-unit", help="systemd user unit to restart on change")
    a = ap.parse_args()
    cmd = ["systemctl", "--user", "restart", a.restart_unit] if a.restart_unit else None
    changed = refresh(Path(a.db), Path(a.data), restart_cmd=cmd,
                      embed=not a.no_embed)
    print("changed" if changed else "unchanged")
    if a.no_chunks:
        return
    # The corpus swap above is final; a chunk failure only fails the unit.
    from chunk_refresh import refresh_chunks
    from embed_client import EmbedClient
    from qdrant_client import Qdrant
    try:
        counts = refresh_chunks(
            Path(a.db), Path(a.chunks_db), Path(a.data),
            EmbedClient(os.environ.get("ARCHIVES_EMBED_URL", "http://127.0.0.1:8201")),
            Qdrant(os.environ.get("ARCHIVES_QDRANT_URL", "http://127.0.0.1:6333")),
            Path(a.data) / "roster.csv")
    except Exception as e:
        print(f"chunk refresh failed: {e!r}", file=sys.stderr)
        sys.exit(1)
    print("chunks", counts)


if __name__ == "__main__":
    main()
