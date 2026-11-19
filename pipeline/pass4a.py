"""Pass 4a: local VLM transcription via an OpenAI-compatible chat server.

Writes data/{id}/{OUT_DIR}/{n}.txt (transcript) and {n}.meta.json (tokens,
timing, image size, loop-guard verdict). score.py picks vision*/{n}.txt up
automatically, labeling the engine by directory name.

Loop guard: greedy decoding can fall into a repetition attractor on sparse
figure pages (observed on Qwen3-VL-8B, Qwen3.8-27B and Chandra). Looping
text compresses far better than real transcription (measured: loops < 0.17
zlib ratio, worst legitimate page 0.193, median page 0.43), so output with
ratio < 0.15 -- or < 0.25 when generation hit the token limit -- is retried
once at temperature 0.4 (any perturbation breaks the attractor). If the
retry is degenerate too, the less-degenerate attempt is kept and the page
is flagged vision_failed for human review.
"""
import argparse, base64, hashlib, io, json, os, sys, time, urllib.error, urllib.request, zlib

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import DATA, report_dir, load_reports, log
from guard import LOOP_GUARD, zratio, junk_reason, pseudo_table_junk, degenerate  # noqa: F401 (re-exported)

SERVER = os.environ.get("NACA_VLM_URL", "http://127.0.0.1:8780")
OUT_DIR = os.environ.get("NACA_VLM_DIR", "vision")   # engine output dir under data/{id}/
MODEL = os.environ.get("NACA_VLM_MODEL", "")          # sent in body when set (vLLM needs it)
DRY = os.environ.get("NACA_VLM_DRY", "1") == "1"      # llama.cpp-only sampler params
PROMPT_FILE = os.environ.get("NACA_VLM_PROMPT_FILE", "")  # per-engine canonical prompt
API_KEY = os.environ.get("NACA_VLM_KEY", "")   # Authorization: Bearer (hosted endpoints)
THINK = os.environ.get("NACA_VLM_THINK", "1") == "1"  # Qwen3 thinking mode (0 = chat_template_kwargs enable_thinking:false)
MAX_PX = int(os.environ.get("NACA_VLM_MAX_PX", "0"))  # cap image long side before send; 0 = native

CONFIG = {
    "model": MODEL or "Qwen3-VL-8B-Instruct-Q8_0",
    "temperature": 0.0,
    "max_tokens": 12288,
    "dry_multiplier": 0.8,
    "dry_allowed_length": 4,
    "prompt_version": "v1",
    "loop_guard": LOOP_GUARD,
}
if not THINK:
    CONFIG["enable_thinking"] = False  # key absent when thinking on: legacy hashes stand
if MAX_PX:
    CONFIG["max_px"] = MAX_PX          # key absent at native res: legacy hashes stand
CONFIG_HASH = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:12]


def cap_image(png: bytes, max_px: int) -> bytes:
    """Downscale so the long side is at most max_px (aspect kept); 0 or within-cap = unchanged."""
    if not max_px:
        return png
    from PIL import Image
    im = Image.open(io.BytesIO(png))
    w, h = im.size
    long_side = max(w, h)
    if long_side <= max_px:
        return png
    scale = max_px / long_side
    im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

PROMPT = """Transcribe ALL printed text on this scanned NACA report page, exactly as printed.

Rules:
- Reading order: left column then right, top to bottom.
- Include running headers and printed page numbers on their own lines.
- Rejoin words hyphenated at line breaks (write "moment", not "mo-ment").
- Tables: one printed row per line, cells separated by " | ", including header rows.
- Equations: LaTeX; display equations as $$...$$ on their own line, inline math as $...$.
- Figure pages: transcribe titles, captions, legend text, axis labels and tick numerals. Do NOT describe the imagery.
- Struck-through classification stamps: transcribe the stamped word if legible under the strike.
- If a span is genuinely unreadable, write <?> in its place. NEVER invent plausible text you cannot actually read. If the page is blank, output nothing.

Output the transcription only - no commentary, no markdown fences."""

if PROMPT_FILE:
    PROMPT = open(PROMPT_FILE).read().strip()
    CONFIG["prompt_version"] = "file:" + hashlib.sha256(PROMPT.encode()).hexdigest()[:10]
    CONFIG_HASH = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:12]


def request(img_b64: str, temperature: float) -> dict:
    """One chat completion; retries transport/5xx errors (server hiccups)."""
    body = {
        "temperature": temperature,
        "max_tokens": CONFIG["max_tokens"],
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_b64}},
            {"type": "text", "text": PROMPT},
        ]}],
    }
    if MODEL:
        body["model"] = MODEL
    if not THINK:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if DRY:
        body["dry_multiplier"] = CONFIG["dry_multiplier"]
        body["dry_allowed_length"] = CONFIG["dry_allowed_length"]
    data = json.dumps(body).encode()
    last = None
    for attempt in range(3):
        try:
            hdrs = {"Content-Type": "application/json"}
            if API_KEY:
                hdrs["Authorization"] = "Bearer " + API_KEY
            req = urllib.request.Request(SERVER + "/v1/chat/completions", data, hdrs)
            with urllib.request.urlopen(req, timeout=1800) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code < 500:
                raise
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = e
        time.sleep(15 * (attempt + 1))
    raise last


def transcribe(ntrs_id: str, n: int, force=False) -> str:
    vdir = report_dir(ntrs_id) / OUT_DIR
    out, meta_p = vdir / f"{n}.txt", vdir / f"{n}.meta.json"
    if out.exists() and meta_p.exists() and not force:
        if json.loads(meta_p.read_text()).get("config_hash") == CONFIG_HASH:
            return "skipped"
    img = cap_image((report_dir(ntrs_id) / "pages" / f"{n}.png").read_bytes(), MAX_PX)
    from PIL import Image
    w, h = Image.open(io.BytesIO(img)).size  # size as sent (post-cap)
    img_b64 = base64.b64encode(img).decode()

    t0 = time.time()
    resp = request(img_b64, CONFIG["temperature"])
    fin = resp["choices"][0].get("finish_reason")
    text = (resp["choices"][0]["message"].get("content") or "").strip()
    bad, ratio, reason = degenerate(text, fin)
    guard = {"ratio": ratio, "reason": reason, "retried": False, "vision_failed": False,
             "temperature_used": CONFIG["temperature"]}
    usage = resp.get("usage", {})

    if bad:
        log("loop_guard_retry", report=ntrs_id, page=n, ratio=ratio, fin=fin)
        resp2 = request(img_b64, LOOP_GUARD["retry_temperature"])
        fin2 = resp2["choices"][0].get("finish_reason")
        text2 = (resp2["choices"][0]["message"].get("content") or "").strip()
        bad2, ratio2, reason2 = degenerate(text2, fin2)
        guard.update(retried=True, retry_ratio=ratio2, retry_reason=reason2)
        u2 = resp2.get("usage", {})
        usage = {k: usage.get(k, 0) + u2.get(k, 0)
                 for k in set(usage) | set(u2)
                 if isinstance(usage.get(k, 0), (int, float)) and isinstance(u2.get(k, 0), (int, float))}
        if not bad2:
            text, fin = text2, fin2
            guard["temperature_used"] = LOOP_GUARD["retry_temperature"]
        else:
            guard["vision_failed"] = True
            if (ratio2 or 0) > (ratio or 0):   # keep the less-degenerate attempt
                text, fin = text2, fin2
                guard["temperature_used"] = LOOP_GUARD["retry_temperature"]
            log("loop_guard_failed", report=ntrs_id, page=n,
                ratio=ratio, retry_ratio=ratio2)
    dt = time.time() - t0

    vdir.mkdir(exist_ok=True)
    out.write_text(text + ("\n" if text else ""))
    meta_p.write_text(json.dumps({
        "config_hash": CONFIG_HASH, "config": CONFIG, "server": SERVER,
        "image_px": [w, h], "seconds": round(dt, 1),
        "usage": usage,
        "finish_reason": fin,
        "output_chars": len(text),
        "loop_guard": guard,
    }, indent=1))
    return "vision_failed" if guard["vision_failed"] else "ok"


def bucket_pages(agreement_path, bucket: str) -> list:
    """(doc, page) rows of one agreement bucket, sorted; header rows have no bucket."""
    pages = []
    with open(agreement_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("bucket") == bucket:
                pages.append((r["doc"], r["page"]))
    return sorted(pages)


def pages_file_pages(path) -> list:
    """(doc, page) rows of a JSONL file with doc/page fields, sorted."""
    pages = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pages.append((r["doc"], int(r["page"])))
    return sorted(pages)


def escalated_pages():
    """Gate v1 over the loaded reports; logic lives in workgen.gated_pages
    (data/gate_v1.json, hash 1e931d767294)."""
    from workgen import gated_pages
    return gated_pages([rep["ntrs_id"] for rep in load_reports()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuning", action="store_true")
    ap.add_argument("--escalated", action="store_true",
                    help="all corpus pages selected by gate v1")
    ap.add_argument("--page", nargs=2, metavar=("NTRS_ID", "N"))
    ap.add_argument("--sample", type=int, default=0,
                    help="random N of the escalated set (seed 0; for speed tests)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N pages (smoke test)")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent requests (vLLM batches; keep 1 for llama.cpp)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--bucket", default="",
                    help="pages of this bucket in data/agreement.jsonl (e.g. conflict)")
    ap.add_argument("--pages-file", default="",
                    help="JSONL file of {doc, page} rows to transcribe")
    args = ap.parse_args()
    if args.page:
        pages = [(args.page[0], int(args.page[1]))]
    elif args.tuning:
        import csv
        pages = [(r["ntrs_id"], int(r["pdf_page"]))
                 for r in csv.DictReader(open(DATA / "tuning_pages.csv"))]
    elif args.bucket:
        pages = bucket_pages(DATA / "agreement.jsonl", args.bucket)
    elif args.pages_file:
        pages = pages_file_pages(args.pages_file)
    elif args.escalated:
        pages = escalated_pages()
    elif args.sample:
        import random
        pages = escalated_pages()
        random.seed(0)
        pages = sorted(random.sample(pages, args.sample))
    else:
        ap.error("--tuning, --escalated, --bucket, --pages-file or --page required")
    if args.limit:
        pages = pages[:args.limit]
    log("pass4a_start", pages=len(pages), out_dir=OUT_DIR, workers=args.workers,
        config_hash=CONFIG_HASH)

    counts = {"ok": 0, "skipped": 0, "failed": 0, "vision_failed": 0}
    tok_in = tok_out = 0
    t_start = time.time()
    done = 0

    def run_one(job):
        nid, n = job
        try:
            return nid, n, transcribe(nid, n, args.force), None
        except Exception as e:
            return nid, n, "failed", repr(e)[:300]

    def report(nid, n, st, err):
        nonlocal tok_in, tok_out, done
        done += 1
        counts[st] += 1
        if err:
            log("pass4a_page_error", report=nid, page=n, error=err)
            return
        if st == "skipped":
            return
        m = json.loads((report_dir(nid) / OUT_DIR / f"{n}.meta.json").read_text())
        u = m.get("usage", {})
        tok_in += u.get("prompt_tokens", 0)
        tok_out += u.get("completion_tokens", 0)
        rate = done / max(1e-9, time.time() - t_start)
        eta_h = (len(pages) - done) / max(1e-9, rate) / 3600
        flag = " LOOP-RETRY" if m["loop_guard"]["retried"] else ""
        flag += " VISION_FAILED" if m["loop_guard"]["vision_failed"] else ""
        print(f"  [{done}/{len(pages)} eta {eta_h:.1f}h] {nid} p{n}: {st} "
              f"{m['seconds']}s out={m['output_chars']}ch fin={m.get('finish_reason')}{flag}")

    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_one, j) for j in pages]
            for f in as_completed(futs):
                report(*f.result())
    else:
        for j in pages:
            report(*run_one(j))

    log("pass4a_done", **counts, tokens_in=tok_in, tokens_out=tok_out,
        hours=round((time.time() - t_start) / 3600, 2), config_hash=CONFIG_HASH)


if __name__ == "__main__":
    main()
