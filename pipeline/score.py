"""Score engine output against ground truth: CER, WER, term recall.

Engines: baseline (1990s embedded text layer), tesseract (pass 3a), and any
future engine that leaves a text file at a known path. Normalization is frozen
by NORM + its hash; change NORM only deliberately (it invalidates comparisons).
"""
import argparse, hashlib, json, re, sys, time, unicodedata

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import DATA, report_dir, known_words  # noqa: F401  (known_words unused, kept for parity)

NORM = {
    "ligatures": True,        # fi/fl/ff/ffi/ffl -> letters
    "quotes": "straight",     # curly -> straight
    "dashes": "hyphen",       # en/em dash -> -
    "rejoin_hyphens": True,   # word-\nword -> wordword (engine side; GT already rejoined)
    "drop_table_pipes": True, # GT " | " cell separators -> space
    "drop_latex_delims": True,# $ and $$ removed, \ kept
    "drop_illegible": True,   # GT <?> markers removed (hyp chars there count as insertions)
    "strip_markup": True,     # YAML front matter, HTML/XML tags, entities removed (engine-format neutral)
    "collapse_ws": True,      # all whitespace runs -> single space
    "case": "preserve",
}
NORM_HASH = hashlib.sha256(json.dumps(NORM, sort_keys=True).encode()).hexdigest()[:12]

_LIG = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
_QUO = {"‘": "'", "’": "'", "“": '"', "”": '"'}
_DSH = {"–": "-", "—": "-", "−": "-"}


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFC", text)
    for table in (_LIG, _QUO, _DSH):
        for k, v in table.items():
            t = t.replace(k, v)
    t = t.replace("­", "")              # soft hyphen
    t = re.sub(r"(\w)-[ \t]*\n[ \t]*(?=\w)", r"\1", t)  # rejoin line-break hyphens
    t = re.sub(r"\A---\n.*?\n---\n", " ", t, flags=re.S)
    t = t.replace("<?>", "\x00Q\x00")
    t = re.sub(r"</?[A-Za-z][^>]{0,600}>", " ", t)
    t = t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&nbsp;", " ")
    t = t.replace("\x00Q\x00", " ").replace("⟨?⟩", " ")
    t = t.replace("$$", " ").replace("$", " ")
    t = t.replace("|", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    ai = np.frombuffer(a.encode("utf-32-le"), dtype=np.uint32)
    bi = np.frombuffer(b.encode("utf-32-le"), dtype=np.uint32)
    idx = np.arange(len(bi) + 1)
    prev = idx.copy()
    for i, ca in enumerate(ai, 1):
        cur = np.empty_like(prev)
        cur[0] = i
        np.minimum(prev[:-1] + (bi != ca), prev[1:] + 1, out=cur[1:])
        # deletions: cur[j] = min over k<=j of cur[k] + (j - k), via prefix min
        np.minimum(cur, np.minimum.accumulate(cur - idx) + idx, out=cur)
        prev = cur
    return int(prev[-1])


_tok = re.compile(r"\S+")


def wer(gt: str, hyp: str) -> float:
    g, h = _tok.findall(gt), _tok.findall(hyp)
    if not g:
        return 0.0 if not h else 1.0
    vocab = {w: chr(i + 1) for i, w in enumerate(dict.fromkeys(g + h))}
    return levenshtein("".join(vocab[w] for w in g), "".join(vocab[w] for w in h)) / len(g)


def load_terms():
    terms = set()
    for line in open(DATA / "term_list_v1.txt"):
        w = line.strip().lower()
        if len(w) >= 3:
            terms.add(w)
    return terms


def term_recall(gt: str, hyp: str, terms) -> tuple:
    gt_toks = {w.lower().strip(".,;:()") for w in gt.split()}
    hyp_toks = {w.lower().strip(".,;:()") for w in hyp.split()}
    present = gt_toks & terms
    if not present:
        return None, 0
    return len(present & hyp_toks) / len(present), len(present)


def engine_texts(ntrs_id: str, n: int) -> dict:
    rdir = report_dir(ntrs_id)
    out = {}
    bl = rdir / "baseline" / f"{n}.txt"
    if bl.exists():
        out["baseline"] = bl.read_text(errors="replace")
    oc = rdir / "ocr" / f"{n}.json"
    if oc.exists():
        out["tesseract"] = json.loads(oc.read_text())["text"]
    for vdir in sorted(rdir.glob("vision*")):
        vt = vdir / f"{n}.txt"
        if vt.exists():
            out[vdir.name] = vt.read_text(errors="replace")
    return out


def score_page(ntrs_id: str, n: int, terms) -> dict:
    gt_path = report_dir(ntrs_id) / "ground_truth" / f"{n}.txt"
    meta = json.loads((report_dir(ntrs_id) / "ground_truth" / f"{n}.json").read_text())
    gt = normalize(gt_path.read_text())
    res = {"ntrs_id": ntrs_id, "page": n, "gt_status": meta["status"],
           "gt_chars": len(gt), "norm_hash": NORM_HASH, "engines": {}}
    for eng, raw in engine_texts(ntrs_id, n).items():
        hyp = normalize(raw)
        dist = levenshtein(gt, hyp)
        cer = dist / max(1, len(gt))
        tr, nterms = term_recall(gt, hyp, terms)
        res["engines"][eng] = {
            "cer": round(cer, 4), "wer": round(wer(gt, hyp), 4),
            "term_recall": None if tr is None else round(tr, 4),
            "n_terms": nterms, "hyp_chars": len(hyp),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", action="store_true", help="include draft ground truth")
    ap.add_argument("--page", nargs=2, metavar=("NTRS_ID", "N"), help="single page")
    args = ap.parse_args()
    terms = load_terms()
    pages = []
    if args.page:
        pages = [(args.page[0], int(args.page[1]))]
    else:
        for gj in sorted(DATA.glob("*/ground_truth/*.json")):
            meta = json.loads(gj.read_text())
            if meta["status"] == "done" or args.drafts:
                pages.append((gj.parent.parent.name, int(gj.stem)))
    results = []
    hdr = f'{"report page":<22} {"gt":<6} {"engine":<10} {"CER":>7} {"WER":>7} {"term_rec":>8}'
    print(hdr + "\n" + "-" * len(hdr))
    for nid, n in pages:
        r = score_page(nid, n, terms)
        results.append(r)
        for eng, m in r["engines"].items():
            tr = "-" if m["term_recall"] is None else f'{m["term_recall"]:.3f}'
            print(f'{nid + " p" + str(n):<22} {r["gt_status"]:<6} {eng:<10}'
                  f' {m["cer"]:>7.3f} {m["wer"]:>7.3f} {tr:>8}')
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(DATA / "scores.jsonl", "a") as f:
        for r in results:
            f.write(json.dumps({"ts": ts, **r}) + "\n")
    (DATA / "scores_latest.json").write_text(json.dumps(
        {"ts": ts, "norm_hash": NORM_HASH, "pages": results}, indent=1))


if __name__ == "__main__":
    main()
