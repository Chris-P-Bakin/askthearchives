"""Shared plumbing for the NACA OCR pilot pipeline."""
import csv, fcntl, json, os, re, time, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CSV = ROOT / "naca_pilot_sample.csv"
LOG = DATA / "pass1.log.jsonl"
TERMS = DATA / "term_list_v1.txt"
ENGLISH_WORDS = pathlib.Path("/usr/share/dict/words")

def load_reports(csv_path: str | None = None):
    """Explicit arg wins, else env NACA_REPORTS_CSV, else the pilot CSV."""
    path = csv_path or os.environ.get("NACA_REPORTS_CSV") or CSV
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def report_dir(ntrs_id: str) -> pathlib.Path:
    return DATA / ntrs_id

def log(event: str, **kw):
    DATA.mkdir(exist_ok=True)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, **kw}
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[{rec['ts']}] {event} " + " ".join(f"{k}={v}" for k, v in kw.items()))

def ntrs_wait(min_interval: float = 4.0, lock_path=None) -> None:
    """Block until >= min_interval s since the last NTRS request by ANY process.

    State file holds the last-request epoch; flock serializes processes, so
    concurrent callers queue and each stamps its own slot. Default lives beside LOG.
    """
    path = pathlib.Path(lock_path) if lock_path else LOG.parent / ".ntrs.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            try:
                last = float(f.read().strip() or 0)
            except ValueError:
                last = 0.0
            wait = min_interval - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            f.seek(0)
            f.truncate()
            f.write(repr(time.time()))
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

_token_re = re.compile(r"[A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff][A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff0-9'-]*")
_latex_re = re.compile(r"\\[A-Za-z]+|\$")

def seed_term_list():
    """Term list v1, seeded from the pilot CSV per the spec: titles, report numbers, series."""
    if TERMS.exists():
        return
    terms = set()
    for r in load_reports():
        for field in ("title", "report_number", "other_report_numbers", "series_name"):
            for tok in _token_re.findall(r.get(field, "")):
                terms.add(tok.lower())
        # report-number fragments: NACA-TR-96 -> naca, tr, tr-96, naca-tr-96
        for num in (r["report_number"], r.get("other_report_numbers", "")):
            if num:
                terms.add(num.lower())
                parts = num.lower().split("-")
                for i in range(len(parts)):
                    terms.add("-".join(parts[i:]))
    DATA.mkdir(exist_ok=True)
    TERMS.write_text("\n".join(sorted(terms)) + "\n")
    log("term_list_seeded", version=1, terms=len(terms))

def known_words():
    """English dictionary + the seeded aeronautics/CSV term list, lowercased."""
    seed_term_list()
    words = set(w.strip().lower() for w in ENGLISH_WORDS.read_text().splitlines())
    words |= set(w.strip() for w in TERMS.read_text().splitlines())
    return words

def dict_share(text: str, words: set) -> tuple[int, int, float]:
    toks = _token_re.findall(_latex_re.sub(" ", text))
    if not toks:
        return 0, 0, 0.0
    known = sum(1 for t in toks if t.lower() in words or t.lower().rstrip("'s") in words)
    return len(toks), known, round(known / len(toks), 4)
