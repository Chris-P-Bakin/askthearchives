"""Batch work generator: gated pages of a batch manifest -> page-granular units."""
import argparse, json, sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from common import report_dir
from units import ACCEPTED_HASHES, CAMPAIGN_LAYER, doc_dir, make_unit, registry, write_units


def _cid(nid: str) -> str:
    return registry.for_doc(nid)["id"]


def gated_pages(ntrs_ids: Iterable[str]) -> list[tuple[str, int]]:
    """Gate v1 (data/gate_v1.json, hash 1e931d767294):
    escalate if tess_mean_conf < 70 or figure_area_frac > 0.2."""
    pages = []
    for nid in ntrs_ids:
        rdir = doc_dir(nid, _cid(nid))
        for oj in sorted((rdir / "ocr").glob("[0-9]*.json"), key=lambda p: int(p.stem)):
            n = int(oj.stem)
            o = json.loads(oj.read_text())
            confs = [wd["conf"] for wd in o["words"]]
            conf = sum(confs) / len(confs) if confs else 0.0
            frac = 0.0
            lay = rdir / "layout" / f"{n}.json"
            if lay.exists():
                frac = sum(f["area_frac"] for f in json.loads(lay.read_text())["figures"])
            if conf < 70 or frac > 0.2:
                pages.append((nid, n))
    return pages


def _done(nid: str, n: int) -> bool:
    meta = doc_dir(nid, _cid(nid)) / CAMPAIGN_LAYER / f"{n}.meta.json"
    if not meta.exists():
        return False
    try:
        return json.loads(meta.read_text()).get("config_hash") in ACCEPTED_HASHES
    except (ValueError, OSError):
        return False


def pending_units(manifest: dict) -> list[dict]:
    return [make_unit(nid, n, _cid(nid)) for nid, n in gated_pages(manifest["ids"])
            if not _done(nid, n)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    data = common.DATA
    manifest = json.loads((data / "batches" / f"batch_{args.batch:03d}.json").read_text())
    out = Path(args.out) if args.out else data / "units" / f"batch_{args.batch:03d}_units.jsonl"
    if args.out and not out.is_absolute():
        out = common.ROOT / out
    n_pages = sum(len(list((doc_dir(i, _cid(i)) / "ocr").glob("[0-9]*.json"))) for i in manifest["ids"])
    gated = gated_pages(manifest["ids"])
    us = pending_units(manifest)
    write_units(us, out)
    rate = len(gated) / n_pages if n_pages else 0.0
    print(f"batch {args.batch}: docs={len(manifest['ids'])} ocr_pages={n_pages} "
          f"gated={len(gated)} ({rate:.1%}) units={len(us)} -> {out}")


if __name__ == "__main__":
    main()
