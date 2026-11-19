"""Unit worker: transcribe one NACA page with the campaign VLM engine.

Stdlib only (plus guard.py, which sits beside this file in the container or
in ../pipeline in the repo). The request shape and CONFIG are byte-equivalent
to pipeline/pass4a.py so the config hash is 6ecf7d0dbeb2.

    python worker/unit_worker.py <unit.json> <image.png> <out_dir>      # BOINC file mode
    python worker/unit_worker.py --standalone <units.jsonl> --data-root <path>

File mode writes <out_dir>/result.txt + result.meta.json, or on failure
<out_dir>/result.meta.json = {"error": reason} and exits 2. A failure is never
an empty-text success.
"""
import argparse, base64, hashlib, json, os, socket, struct, sys, time
import urllib.error, urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "pipeline"):
    sys.path.insert(0, str(_p))
from guard import LOOP_GUARD, degenerate  # noqa: E402

def _pick_server():
    # Multi-GPU boxes run one llama-server per card; NACA_VLM_URLS lists them.
    # Each BOINC slot is its own process, so pid-mod spreads slots across cards.
    urls = [u for u in os.environ.get("NACA_VLM_URLS", "").split(",") if u]
    if not urls:
        return os.environ.get("NACA_VLM_URL", "http://127.0.0.1:8782")
    return urls[os.getpid() % len(urls)]


SERVER = _pick_server()
MODEL = "qwen3-vl-8b-lora-v2_1"
CAMPAIGN_LAYER = "vision_8blora_v2_1"
ACCEPTED_HASHES = {"6ecf7d0dbeb2"}
PROMPT_FILE = os.environ.get(
    "NACA_PROMPT_FILE",
    str(next((p for p in (_HERE / "transcribe_v2.txt",
                          _HERE.parent / "pipeline/prompts/transcribe_v2.txt") if p.exists()),
             _HERE / "transcribe_v2.txt")))
MAX_SLOTS = 2   # serve_vlm_8blora.sh runs -np 2: never more threads than slots
RETRY_SLEEP = 15
HTTP_ATTEMPTS = 3


def load_prompt(path: str) -> str:
    """A missing/empty prompt would silently change prompt_version -> CONFIG_HASH, making
    every result invalid after burning GPU time: fail fast at startup instead."""
    try:
        text = open(path, encoding="utf-8").read().strip()
    except OSError as e:
        raise SystemExit(f"fatal: prompt file unreadable: {path}: {e}")
    if not text:
        raise SystemExit(f"fatal: prompt file empty: {path}")
    return text


PROMPT = load_prompt(PROMPT_FILE)

CONFIG = {
    "model": MODEL,
    "temperature": 0.0,
    "max_tokens": 12288,
    "dry_multiplier": 0.8,
    "dry_allowed_length": 4,
    "prompt_version": "file:" + hashlib.sha256(PROMPT.encode()).hexdigest()[:10],
    "loop_guard": LOOP_GUARD,
}
CONFIG_HASH = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:12]

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PNG_IEND = b"\x00\x00\x00\x00IEND\xaeB`\x82"


class UnitError(Exception):
    """Structured failure: carries a short machine-readable reason."""
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def png_size(png: bytes) -> tuple[int, int]:
    """Validate magic, IHDR and the IEND trailer (truncation); return (w, h)."""
    if len(png) < 33 or png[:8] != PNG_MAGIC or png[12:16] != b"IHDR":
        raise UnitError("bad_image: not a PNG")
    w, h = struct.unpack(">II", png[16:24])
    if not (w and h):
        raise UnitError("bad_image: zero dimension")
    if not png.endswith(PNG_IEND):
        raise UnitError("bad_image: truncated PNG")
    return w, h


def request(server_url: str, prompt: str, img_b64: str, temperature: float) -> dict:
    body = {
        "temperature": temperature,
        "max_tokens": CONFIG["max_tokens"],
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_b64}},
            {"type": "text", "text": prompt},
        ]}],
        "model": MODEL,
        "dry_multiplier": CONFIG["dry_multiplier"],
        "dry_allowed_length": CONFIG["dry_allowed_length"],
    }
    data = json.dumps(body).encode()
    last = None
    for attempt in range(HTTP_ATTEMPTS):
        try:
            req = urllib.request.Request(server_url + "/v1/chat/completions", data,
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1800) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code < 500:
                raise UnitError(f"server_http_{e.code}")
            last = f"server_http_{e.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = f"server_unreachable: {e!r}"[:200]
        except ValueError as e:
            last = f"server_bad_json: {e!r}"[:200]
        if attempt < HTTP_ATTEMPTS - 1:
            time.sleep(RETRY_SLEEP * (attempt + 1))
    raise UnitError(last or "server_error")


def _choice(resp: dict):
    try:
        c = resp["choices"][0]
        return (c.get("message", {}).get("content") or "").strip(), c.get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError):
        raise UnitError("server_malformed_response")


def process_unit(unit: dict, image_bytes: bytes, server_url: str, prompt: str) -> tuple[str, dict]:
    w, h = png_size(image_bytes)
    img_b64 = base64.b64encode(image_bytes).decode()
    t0 = time.time()
    resp = request(server_url, prompt, img_b64, CONFIG["temperature"])
    text, fin = _choice(resp)
    bad, ratio, reason = degenerate(text, fin)
    guard = {"ratio": ratio, "reason": reason, "retried": False, "vision_failed": False,
             "temperature_used": CONFIG["temperature"]}
    usage = resp.get("usage", {})

    if bad:
        resp2 = request(server_url, prompt, img_b64, LOOP_GUARD["retry_temperature"])
        text2, fin2 = _choice(resp2)
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
    if not text.strip() and not guard["vision_failed"]:
        # validator rejects empty_output; never persist it as a success
        raise UnitError("empty_output")
    meta = {
        "config_hash": CONFIG_HASH, "config": CONFIG, "server": server_url,
        "image_px": [w, h], "seconds": round(time.time() - t0, 1),
        "usage": usage, "finish_reason": fin, "output_chars": len(text),
        "loop_guard": guard, "worker": socket.gethostname(),
    }
    return text, meta


def check_image(unit: dict, image_bytes: bytes) -> None:
    sha = unit.get("image_sha256")
    if sha and hashlib.sha256(image_bytes).hexdigest() != sha:
        raise UnitError("bad_image: sha256 mismatch")


def _atomic(path: Path, data: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


def write_result(out_dir: Path, text: str, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic(out_dir / "result.txt", text + ("\n" if text else ""))
    _atomic(out_dir / "result.meta.json", json.dumps(meta, indent=1))


def run_file_mode(unit_path, image_path, out_dir, server=SERVER) -> int:
    out_dir = Path(out_dir)
    try:
        unit = json.loads(Path(unit_path).read_text(encoding="utf-8"))
        try:
            img = Path(image_path).read_bytes()
        except OSError as e:
            raise UnitError(f"bad_image: {e.__class__.__name__}")
        check_image(unit, img)
        text, meta = process_unit(unit, img, server, PROMPT)
    except UnitError as e:
        out_dir.mkdir(parents=True, exist_ok=True)
        _atomic(out_dir / "result.meta.json", json.dumps({"error": e.reason}))
        print(f"error: {e.reason}", file=sys.stderr)
        return 2
    write_result(out_dir, text, meta)
    return 0


def _done(data_root: Path, unit: dict) -> bool:
    p = data_root / unit["ntrs_id"] / CAMPAIGN_LAYER / f"{unit['page']}.meta.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("config_hash") in ACCEPTED_HASHES
    except (OSError, ValueError):
        return False


def run_standalone(units_path, data_root, results_dir=None, server=SERVER, workers=MAX_SLOTS,
                   limit=0, assimilate=False) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    data_root = Path(data_root)
    results_dir = Path(results_dir) if results_dir else data_root / "worker_results"
    units = [json.loads(l) for l in open(units_path, encoding="utf-8") if l.strip()]
    pending = [u for u in units if not _done(data_root, u)]
    skipped = len(units) - len(pending)
    if limit:
        pending = pending[:limit]
    workers = max(1, min(workers, MAX_SLOTS))
    counts = {"done": 0, "failed": 0, "skipped": skipped}

    def one(u):
        rd = results_dir / u["unit_id"]
        try:
            img = (data_root / u["ntrs_id"] / "pages" / f"{u['page']}.png").read_bytes()
            check_image(u, img)
            text, meta = process_unit(u, img, server, PROMPT)
        except OSError:
            return u, None, None, "bad_image: missing"
        except UnitError as e:
            return u, None, None, e.reason
        write_result(rd, text, meta)
        return u, text, meta, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for u, text, meta, err in ex.map(one, pending):
            if err:
                counts["failed"] += 1
                rd = results_dir / u["unit_id"]
                rd.mkdir(parents=True, exist_ok=True)
                _atomic(rd / "result.meta.json", json.dumps({"error": err}))
                print(f"{u['unit_id']}: FAILED {err}", flush=True)
                continue
            counts["done"] += 1
            st = ""
            if assimilate:
                import unit_assimilate
                st = " " + unit_assimilate.assimilate(u, text, meta, data_root)["status"]
            print(f"{u['unit_id']}: {meta['seconds']}s chars={meta['output_chars']} "
                  f"retried={meta['loop_guard']['retried']} "
                  f"vision_failed={meta['loop_guard']['vision_failed']}{st}", flush=True)
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("positional", nargs="*", help="<unit.json> <image.png> <out_dir>")
    ap.add_argument("--standalone", metavar="UNITS_JSONL")
    ap.add_argument("--data-root")
    ap.add_argument("--results-dir")
    ap.add_argument("--server", default=SERVER)
    ap.add_argument("--workers", type=int, default=MAX_SLOTS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--assimilate", action="store_true",
                    help="standalone: validate+assimilate into the data tree (needs pipeline/)")
    a = ap.parse_args(argv)
    if a.standalone:
        if not a.data_root:
            ap.error("--standalone needs --data-root")
        c = run_standalone(a.standalone, a.data_root, a.results_dir, a.server, a.workers,
                           a.limit, a.assimilate)
        print(json.dumps(c))
        return 2 if c["failed"] else 0
    if len(a.positional) != 3:
        ap.error("need <unit.json> <image.png> <out_dir> or --standalone")
    return run_file_mode(*a.positional, server=a.server)


if __name__ == "__main__":
    sys.exit(main())
