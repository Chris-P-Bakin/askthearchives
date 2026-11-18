"""Stage a Library of Congress "By the People" sample as pipeline pages.

By the People (crowd.loc.gov) transcriptions are written by one volunteer and
accepted by another (status "completed"), and released as public-domain CSV
datasets on loc.gov. That makes them an outside ground truth: material the
pipeline was never tuned on, checked by people who are not us.

Default dataset: "Historical Legal Reports from the Law Library of Congress"
(LCCN 2023446314, 9,807 images). Typed mid-century government reports chosen
by LoC because OCR did badly on them -- the closest BTP match to NACA scans.

For each sampled page this writes, under data/btp-<asset>/:
  pages/1.png              page image (what pass4a reads)
  ground_truth/1.txt       the reviewed volunteer transcription
  ground_truth/1.json      {"status": "btp_reviewed", source metadata}
and data/btp/pages.jsonl ({doc, page} rows for pass4a --pages-file).

The GT status is deliberately not "done": score.py and bench_report.py only
read "done" pages, so BTP pages never leak into the NACA benchmark table.
Score them with bench/btp/score_btp.py instead.

Usage:
  python bench/btp/prep.py --n 30                  # download zip, sample, fetch images
  python bench/btp/prep.py --csv path/to.csv --n 30
"""
import argparse, csv, io, json, pathlib, random, re, sys, time, urllib.request, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = DATA / "btp"

DATASETS = {
    # lccn: (zip url, short name)
    "2023446314": ("https://tile.loc.gov/storage-services/master/gdc/gdcdatasets/2023446314/2023446314.zip",
                   "legal-reports"),
}
UA = "askthearchives-bench/1.0 (OCR accuracy research)"
MIN_CHARS = 400        # skip near-empty pages (covers, blanks): CER is noisy on them
MAX_CHARS = 6000       # skip pages too long for one transcript call
PAUSE = 2.0            # seconds between loc.gov requests (be polite)

csv.field_size_limit(sys.maxsize)


def fetch(url: str, tries: int = 4) -> bytes:
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 -- retried, then re-raised
            last = e
            time.sleep(5 * (i + 1))
    raise last


def resolve_zip(lccn: str) -> str:
    """Zip URL from the item's loc.gov JSON; falls back to the known storage path."""
    try:
        item = json.loads(fetch(f"https://www.loc.gov/item/{lccn}/?fo=json"))
        for res in item.get("resources", []):
            for f in [res] + [x for grp in res.get("files", []) for x in grp]:
                u = f.get("url") or f.get("download") or ""
                if u.lower().endswith(".zip"):
                    return u
    except Exception as e:  # noqa: BLE001
        print(f"item JSON lookup failed ({e!r}); using known path")
    return DATASETS[lccn][0]


def col(row: dict, *names):
    """First present column, case/space-insensitive (BTP exports vary a little)."""
    norm = {re.sub(r"[\s_]", "", k).lower(): v for k, v in row.items() if k}
    for n in names:
        v = norm.get(re.sub(r"[\s_]", "", n).lower())
        if v is not None:
            return v
    return None


def load_rows(csv_bytes: bytes) -> list:
    text = csv_bytes.decode("utf-8-sig", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


def eligible(row: dict) -> bool:
    status = (col(row, "AssetStatus", "Status") or "").strip().lower()
    text = col(row, "Transcription") or ""
    url = col(row, "DownloadUrl", "Download URL", "ImageUrl", "Image URL") or ""
    if status not in ("completed", "complete"):
        return False
    if not url.startswith("http"):
        return False
    if "nothing to transcribe" in text.lower()[:80]:
        return False
    return MIN_CHARS <= len(text.strip()) <= MAX_CHARS


def sample(rows: list, n: int, seed: int, per_item: int = 1) -> list:
    """Seeded sample, at most per_item pages from any one item (spread across reports)."""
    rng = random.Random(seed)
    pool = [r for r in rows if eligible(r)]
    rng.shuffle(pool)
    picked, per = [], {}
    for r in pool:
        item = col(r, "ItemId", "Item") or ""
        if per.get(item, 0) >= per_item:
            continue
        per[item] = per.get(item, 0) + 1
        picked.append(r)
        if len(picked) == n:
            break
    return picked


def doc_id(row: dict) -> str:
    asset = col(row, "Asset") or col(row, "DownloadUrl") or ""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", asset.strip().rstrip("/").split("/")[-1]).strip("-")
    return "btp-" + (slug or "asset")[-60:]


def stage(row: dict, lccn: str, fetch_fn=fetch) -> dict:
    from PIL import Image
    did = doc_id(row)
    d = DATA / did
    (d / "pages").mkdir(parents=True, exist_ok=True)
    (d / "ground_truth").mkdir(parents=True, exist_ok=True)
    png = d / "pages" / "1.png"
    url = col(row, "DownloadUrl", "Download URL", "ImageUrl", "Image URL")
    if not png.exists():
        img = Image.open(io.BytesIO(fetch_fn(url)))
        img.convert("L" if img.mode in ("1", "L", "LA") else "RGB").save(png)
        time.sleep(PAUSE)
    (d / "ground_truth" / "1.txt").write_text(col(row, "Transcription").strip() + "\n")
    meta = {"status": "btp_reviewed", "source": "loc.gov By the People", "dataset_lccn": lccn,
            "campaign": col(row, "Campaign"), "project": col(row, "Project"),
            "item": col(row, "Item"), "item_id": col(row, "ItemId"),
            "asset": col(row, "Asset"), "image_url": url}
    (d / "ground_truth" / "1.json").write_text(json.dumps(meta, indent=1))
    return {"doc": did, "page": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lccn", default="2023446314", choices=sorted(DATASETS))
    ap.add_argument("--csv", default="", help="local CSV instead of downloading the zip")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.csv:
        rows = load_rows(pathlib.Path(args.csv).read_bytes())
    else:
        zpath = OUT / f"{args.lccn}.zip"
        if not zpath.exists():
            zurl = resolve_zip(args.lccn)
            print(f"downloading {zurl}")
            zpath.write_bytes(fetch(zurl))
        with zipfile.ZipFile(zpath) as z:
            member = next(m for m in z.namelist() if m.lower().endswith(".csv"))
            rows = load_rows(z.read(member))
    picked = sample(rows, args.n, args.seed)
    print(f"{len(rows)} rows, {sum(map(eligible, rows))} eligible, sampled {len(picked)}")
    out_rows = []
    for i, r in enumerate(picked, 1):
        out_rows.append(stage(r, args.lccn))
        print(f"  [{i}/{len(picked)}] {out_rows[-1]['doc']}")
    (OUT / "pages.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out_rows))
    print(f"wrote {OUT / 'pages.jsonl'}")


if __name__ == "__main__":
    main()
