"""Build the distillation training manifest (spec: 2026-11-10-8b-distillation-design.md §2).

Selects agreed + minor reconcile pages (plus a seeded sample of blanks), excludes the
GT eval pages and over-budget examples, splits ~5% validation, and records provenance
(prompt hash, teacher config hash, label hashes) in data/distill_v1_manifest.json.
"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from PIL import Image

from common import DATA

TEACHER_HASH = "be1cd2673d3b"
TIEBREAKER_HASH = "3669475d3a4b"
PROMPT_PATH = Path(__file__).parent / "prompts" / "transcribe_v2.txt"
TRAIN_BUCKETS = {"agreed", "minor"}
RELABEL_DIR = "vision_235b"


def tiebreak_pages(data_root: Path, buckets: tuple = ("agree", "minor")) -> set:
    """(doc, page) conflict pages the 235B tie-breaker attested (named buckets
    against either the teacher or Haiku)."""
    pages = set()
    for other in ("vision_27bq4", "vision_haiku2"):
        p = data_root / f"agreement_vision_235b__{other}.jsonl"
        for line in open(p):
            r = json.loads(line)
            if r.get("bucket") in buckets:
                pages.add((r["doc"], r["page"]))
    return pages


def table_heavy_pages(data_root: Path, threshold: float = 0.15) -> set:
    """(doc, page) agreed/minor pages whose 27B label text is digit-dense —
    table pages whose 27B digits the attestation found untrustworthy."""
    pages = set()
    for rdir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        rec_dir = rdir / "reconcile"
        if not rec_dir.is_dir():
            continue
        for rp in rec_dir.glob("*.json"):
            if json.loads(rp.read_text())["confidence"] not in TRAIN_BUCKETS:
                continue
            n = int(rp.stem)
            lp = rdir / "vision_27bq4" / f"{n}.txt"
            if not lp.exists():
                continue
            t = lp.read_text()
            if t and sum(c.isdigit() for c in t) / len(t) > threshold:
                pages.add((rdir.name, n))
    return pages


def _est_tokens(img_path: Path, prompt: str, label: str) -> int:
    with Image.open(img_path) as im:
        w, h = im.size
    # Qwen3-VL: 16px patches with 2x2 merge -> one token per 32x32 px block.
    vision = math.ceil(w / 32) * math.ceil(h / 32)
    return vision + len(prompt) // 4 + len(label) // 4 + 64


def _gt_done(rdir: Path, n: int) -> bool:
    mp = rdir / "ground_truth" / f"{n}.json"
    if not mp.exists():
        return False
    try:
        return json.loads(mp.read_text()).get("status") == "done"
    except json.JSONDecodeError:
        return False


def build_manifest(data_root: Path, prompt_path: Path = PROMPT_PATH, seed: int = 13,
                   blank_sample: int = 20, max_tokens: int = 16384 - 512,
                   extra_pages: set = frozenset(),
                   relabel_pages: set = frozenset()) -> dict:
    prompt = prompt_path.read_text().strip()
    rng = random.Random(seed)
    candidates, blanks, dropped, missing_relabel = [], [], [], []

    for rdir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        rec_dir = rdir / "reconcile"
        if not rec_dir.is_dir():
            continue
        for rp in sorted(rec_dir.glob("*.json"), key=lambda p: int(p.stem)):
            n = int(rp.stem)
            rec = json.loads(rp.read_text())
            bucket = rec["confidence"]
            key = (rdir.name, n)
            is_extra = key in extra_pages
            if bucket not in TRAIN_BUCKETS and bucket != "blank" and not is_extra:
                continue
            if _gt_done(rdir, n):
                continue
            label_dir = RELABEL_DIR if is_extra or key in relabel_pages \
                else "vision_27bq4"
            label_p = rdir / label_dir / f"{n}.txt"
            img_p = rdir / "pages" / f"{n}.png"
            if not label_p.exists():
                if key in relabel_pages:
                    missing_relabel.append([rdir.name, n])
                continue
            if not img_p.exists():
                continue
            label = label_p.read_text()
            if _est_tokens(img_p, prompt, label) > max_tokens:
                dropped.append([rdir.name, n])
                continue
            ex = {
                "ntrs_id": rdir.name, "page": n,
                "bucket": "tiebreak" if is_extra else bucket,
                "label_dir": label_dir,
                "image": str(img_p),
                "label_sha256": hashlib.sha256(label.encode()).hexdigest(),
                "est_tokens": _est_tokens(img_p, prompt, label),
            }
            (blanks if bucket == "blank" else candidates).append(ex)

    if len(blanks) > blank_sample:
        blanks = rng.sample(blanks, blank_sample)
        blanks.sort(key=lambda e: (e["ntrs_id"], e["page"]))
    examples = candidates + blanks

    n_val = max(1, round(len(examples) * 0.05)) if examples else 0
    val_idx = set(rng.sample(range(len(examples)), n_val)) if examples else set()
    for i, ex in enumerate(examples):
        ex["split"] = "val" if i in val_idx else "train"

    return {
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "teacher_config_hash": TEACHER_HASH,
        "seed": seed,
        "dropped_over_budget": dropped,
        "dropped_missing_relabel": missing_relabel,
        "examples": examples,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--v2", action="store_true",
                    help="grown pool: admit 235B-attested conflict pages and "
                         "relabel digit-dense legacy pages from vision_235b")
    ap.add_argument("--tiebreak-buckets", default="agree,minor",
                    help="agreement buckets admitting tiebreak pages (--v2 only)")
    args = ap.parse_args()
    extra = tiebreak_pages(DATA, tuple(args.tiebreak_buckets.split(","))) \
        if args.v2 else frozenset()
    relabel = table_heavy_pages(DATA) if args.v2 else frozenset()
    out = args.out or str(DATA / ("distill_v2_manifest.json" if args.v2
                                  else "distill_v1_manifest.json"))
    m = build_manifest(DATA, seed=args.seed, extra_pages=extra,
                       relabel_pages=relabel)
    if args.v2:
        m["tiebreaker_config_hash"] = TIEBREAKER_HASH
    args.out = out
    Path(args.out).write_text(json.dumps(m, indent=1))
    counts: dict[str, int] = {}
    for e in m["examples"]:
        counts[e["bucket"]] = counts.get(e["bucket"], 0) + 1
        counts[e["split"]] = counts.get(e["split"], 0) + 1
    print(json.dumps({"examples": len(m["examples"]), "dropped": len(m["dropped_over_budget"]),
                      **counts, "out": args.out}, indent=1))


if __name__ == "__main__":
    main()
