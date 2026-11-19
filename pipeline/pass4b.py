"""Pass 4b: frontier-API transcription trial (Claude Haiku 5.5 by default).

Same job and prompt as pass 4a, but through the Anthropic SDK. Writes
data/{id}/{OUT_DIR}/{n}.txt + {n}.meta.json in the same shape pass4a uses, so
score.py and timing.py pick the engine up automatically (engine = dir name).

Loop guard: same zlib detector as pass4a. Haiku 5.5 rejects non-default
temperature, so the retry is a plain resend (API sampling is not
deterministic, which is usually enough to break a repetition attractor).

Caveat for scoring: 28/30 tuning ground truths are Claude-drafted and
human-accepted, so Claude-family engines have a home-field advantage on CER.
"""
import argparse, base64, hashlib, json, os, sys, time, zlib

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import DATA, report_dir, log
from pass4a import PROMPT, LOOP_GUARD, degenerate, escalated_pages

import anthropic

MODEL = os.environ.get("NACA_4B_MODEL", "claude-haiku-5-5")
OUT_DIR = os.environ.get("NACA_4B_DIR", "vision_haiku")
CONFIG = {
    "model": MODEL,
    "max_tokens": 12288,
    "effort": "low",
    "prompt_version": __import__("pass4a").CONFIG["prompt_version"],
    "loop_guard": {**LOOP_GUARD, "retry_temperature": None},  # plain resend retry
}
CONFIG_HASH = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:12]

client = anthropic.Anthropic()  # ANTHROPIC_API_KEY from env


def request(img_b64: str):
    resp = client.messages.create(
        model=MODEL,
        max_tokens=CONFIG["max_tokens"],
        output_config={"effort": CONFIG["effort"]},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                "media_type": "image/png", "data": img_b64}},
            {"type": "text", "text": PROMPT},
        ]}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    fin = "length" if resp.stop_reason == "max_tokens" else resp.stop_reason
    usage = {"prompt_tokens": resp.usage.input_tokens,
             "completion_tokens": resp.usage.output_tokens}
    return text, fin, usage


def transcribe(ntrs_id: str, n: int, force=False) -> str:
    vdir = report_dir(ntrs_id) / OUT_DIR
    out, meta_p = vdir / f"{n}.txt", vdir / f"{n}.meta.json"
    if out.exists() and meta_p.exists() and not force:
        if json.loads(meta_p.read_text()).get("config_hash") == CONFIG_HASH:
            return "skipped"
    img_b64 = base64.standard_b64encode(
        (report_dir(ntrs_id) / "pages" / f"{n}.png").read_bytes()).decode()
    from PIL import Image
    import io
    w, h = Image.open(io.BytesIO(base64.b64decode(img_b64))).size

    t0 = time.time()
    text, fin, usage = request(img_b64)
    bad, ratio, reason = degenerate(text, fin)
    guard = {"ratio": ratio, "reason": reason, "retried": False, "vision_failed": False}
    if bad:
        log("loop_guard_retry", report=ntrs_id, page=n, ratio=ratio, fin=fin, engine=OUT_DIR)
        text2, fin2, u2 = request(img_b64)
        bad2, ratio2, reason2 = degenerate(text2, fin2)
        guard.update(retried=True, retry_ratio=ratio2, retry_reason=reason2)
        usage = {k: usage[k] + u2[k] for k in usage}
        if not bad2:
            text, fin = text2, fin2
        else:
            guard["vision_failed"] = True
            if (ratio2 or 0) > (ratio or 0):
                text, fin = text2, fin2
    dt = time.time() - t0

    vdir.mkdir(exist_ok=True)
    out.write_text(text + ("\n" if text else ""))
    meta_p.write_text(json.dumps({
        "config_hash": CONFIG_HASH, "config": CONFIG, "server": "anthropic-api",
        "image_px": [w, h], "seconds": round(dt, 1),
        "usage": usage, "finish_reason": fin,
        "output_chars": len(text), "loop_guard": guard,
    }, indent=1))
    return "vision_failed" if guard["vision_failed"] else "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuning", action="store_true")
    ap.add_argument("--page", nargs=2, metavar=("NTRS_ID", "N"))
    ap.add_argument("--escalated", action="store_true",
                    help="all corpus pages selected by gate v1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.page:
        pages = [(args.page[0], int(args.page[1]))]
    elif args.tuning:
        import csv
        pages = [(r["ntrs_id"], int(r["pdf_page"]))
                 for r in csv.DictReader(open(DATA / "tuning_pages.csv"))]
    elif args.escalated:
        pages = escalated_pages()
    else:
        ap.error("--tuning, --escalated or --page required")
    if args.limit:
        pages = pages[:args.limit]
    log("pass4b_start", pages=len(pages), model=MODEL, out_dir=OUT_DIR,
        config_hash=CONFIG_HASH)
    counts = {"ok": 0, "skipped": 0, "failed": 0, "vision_failed": 0}
    tok_in = tok_out = 0

    def run_one(job):
        nid, n = job
        try:
            return nid, n, transcribe(nid, n, args.force), None
        except anthropic.AuthenticationError as e:
            raise SystemExit(f"API key rejected ({e.status_code}) - aborting run: "
                             f"{str(e)[:160]}")
        except anthropic.APIStatusError as e:
            return nid, n, "failed", f"HTTP {e.status_code}: {str(e)[:200]}"
        except anthropic.APIConnectionError as e:
            return nid, n, "failed", f"connection: {str(e)[:200]}"
        except Exception as e:
            return nid, n, "failed", repr(e)[:300]

    done = 0
    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, j) for j in pages]
        try:
            fut_iter = as_completed(futs)
        except BaseException:
            raise
        for f in fut_iter:
            try:
                nid, n, st, err = f.result()
            except SystemExit:
                ex.shutdown(cancel_futures=True)
                raise
            counts[st] += 1
            done += 1
            if done % 50 == 0:
                rate = done / (time.time() - t0)
                print(f"  [{done}/{len(pages)} eta {(len(pages)-done)/rate/3600:.1f}h]",
                      flush=True)
            if err:
                log("pass4b_page_error", report=nid, page=n, error=err)
                continue
            if st == "skipped":
                continue
            m = json.loads((report_dir(nid) / OUT_DIR / f"{n}.meta.json").read_text())
            u = m["usage"]
            tok_in += u["prompt_tokens"]; tok_out += u["completion_tokens"]
            flag = " LOOP-RETRY" if m["loop_guard"]["retried"] else ""
            flag += " VISION_FAILED" if m["loop_guard"]["vision_failed"] else ""
            print(f"  {nid} p{n}: {st} {m['seconds']}s out={m['output_chars']}ch "
                  f"fin={m['finish_reason']}{flag}")
    cost = tok_in * 0.10 / 1e6 + tok_out * 0.50 / 1e6
    log("pass4b_done", **counts, tokens_in=tok_in, tokens_out=tok_out,
        est_cost_usd=round(cost, 4), config_hash=CONFIG_HASH)


if __name__ == "__main__":
    main()
