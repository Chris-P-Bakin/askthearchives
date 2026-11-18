"""Pipeline dashboard + transcription app, one process.

GET  /                 status dashboard (read-only)
GET  /transcribe       split-view transcription editor for the tuning set
GET  /api/status       JSON scan of data/
GET  /api/tuning       tuning queue with per-page transcription status
GET  /api/page?...     baseline seed + existing transcript for one page
GET  /img/{id}/{n}[.raw].png   page render
POST /api/transcript   save transcript -> data/{id}/ground_truth/{n}.txt (+ .json)

ntrs_id is validated against the pilot CSV; page must be an integer.
"""
import csv, json, math, re, shutil, sqlite3, subprocess, time, threading, urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import common

PORT = 8777
VALID_IDS = {r["ntrs_id"] for r in common.load_reports()}
_cache = {"t": 0.0, "data": None}
_lock = threading.Lock()

ROOT = common.DATA.parent
TRAIN_LOG = ROOT / "training" / "logs" / "v1_loss.jsonl"
TRAIN_RUNS = ROOT / "training" / "runs" / "v1"
MANIFEST = common.DATA / "distill_v1_manifest.json"
CORPUS_DB = common.DATA / "corpus.db"
CURVE_POINTS = 160

# quality-tier display order, best first (index tier ladder)
TIER_ORDER = ["verified", "vision_agreed", "vision_minor", "vision_single",
              "vision", "vision_conflict", "vision_blank", "ocr_draft"]


def training_state(log_path, alive: bool, total_steps: int) -> dict:
    rows = []
    try:
        for ln in log_path.read_text().splitlines():
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "step" in r:
                rows.append(r)
    except FileNotFoundError:
        pass
    loss = eval_loss = epoch = None
    for r in rows:
        loss = r.get("loss", loss)
        eval_loss = r.get("eval_loss", eval_loss)
        epoch = r.get("epoch", epoch)
    step = rows[-1]["step"] if rows else 0
    curve = [(r["step"], r["loss"]) for r in rows if "loss" in r]
    if len(curve) > CURVE_POINTS:
        idx = [round(i * (len(curve) - 1) / (CURVE_POINTS - 1)) for i in range(CURVE_POINTS)]
        curve = [curve[i] for i in dict.fromkeys(idx)]
    return {"alive": alive, "step": step, "total_steps": total_steps,
            "pct": round(100 * step / total_steps) if total_steps else 0,
            "loss": loss, "eval_loss": eval_loss, "epoch": epoch, "curve": curve}


COLLECTION_FILE = common.DATA / "collection.json"
COLLECTION_DEFAULTS = {"name": "Document Pipeline", "subtitle": "", "short": "Pipeline"}


def collection_meta(path=COLLECTION_FILE) -> dict:
    meta = dict(COLLECTION_DEFAULTS)
    try:
        meta.update(json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError):
        pass
    return meta


def tier_counts(db_path) -> dict:
    if not db_path.exists():
        return {}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = conn.execute("SELECT quality, count(*) FROM pages GROUP BY quality").fetchall()
        conn.close()
    except sqlite3.Error:
        return {}
    return dict(rows)


def _pgrep(pattern: str) -> bool:
    out = subprocess.run(["pgrep", "-fc", pattern], capture_output=True, text=True)
    return out.stdout.strip() not in ("", "0")


def _train_total_steps() -> int:
    try:
        m = json.loads(MANIFEST.read_text())
        n_train = sum(1 for e in m["examples"] if e["split"] == "train")
        return math.ceil(n_train / 8) * 2
    except Exception:
        return 0


_cpu_prev = {}

def _cpu_pct() -> float | None:
    try:
        parts = open("/proc/stat").readline().split()[1:]
        vals = [int(x) for x in parts]
    except (OSError, ValueError):
        return None
    total, idle = sum(vals), vals[3] + (vals[4] if len(vals) > 4 else 0)
    p = _cpu_prev
    pct = None
    if p and total > p["total"]:
        pct = round(100 * (1 - (idle - p["idle"]) / (total - p["total"])), 1)
    _cpu_prev.update(total=total, idle=idle)
    return pct


def system_scan() -> dict:
    gpu = None
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
         "--format=csv,noheader,nounits"], capture_output=True, text=True)
    if out.returncode == 0 and out.stdout.strip():
        u, mu, mt, temp, pw = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
        gpu = {"util": int(u), "mem_used": int(mu), "mem_total": int(mt),
               "temp": int(temp), "power": float(pw)}
    mem = None
    try:
        mi = dict(ln.split(":")[:2] for ln in open("/proc/meminfo").read().splitlines() if ":" in ln)
        tot = int(mi["MemTotal"].split()[0]); avail = int(mi["MemAvailable"].split()[0])
        mem = {"used_gb": round((tot - avail) / 1e6, 1), "total_gb": round(tot / 1e6, 1)}
    except (OSError, KeyError, ValueError):
        pass
    try:
        du = shutil.disk_usage(common.DATA)
        disk = {"used_gb": round((du.total - du.free) / 1e9), "total_gb": round(du.total / 1e9)}
    except OSError:
        disk = None
    return {"gpu": gpu, "cpu_pct": _cpu_pct(), "mem": mem, "disk": disk,
            "generated": time.strftime("%H:%M:%S")}


def _engine_active(vdir_name: str, window: float = 180) -> bool:
    """Any page written to this engine dir recently? Dir mtime updates on file create."""
    now = time.time()
    for nid in VALID_IDS:
        d = common.report_dir(nid) / vdir_name
        try:
            if now - d.stat().st_mtime < window:
                return True
        except OSError:
            continue
    return False


def pipeline_state() -> dict:
    """Which schematic stations are active right now (cheap: pgrep + dir mtimes)."""
    return {
        "p1": _pgrep("pass1.py"), "p2": _pgrep("pass2.py"), "p3": _pgrep("pass3.py"),
        "v27": _pgrep("pass4a.py") or _engine_active("vision_27bq4"),
        "haiku": _pgrep("pass4b.py") or _engine_active("vision_haiku2"),
        "agree": _pgrep("agreement.py"), "rec": _pgrep("pass5.py"),
        "index": _pgrep("build_index.py"), "train": _pgrep("train_8b_lora.py"),
    }

GUIDE_DEFAULT = """# Transcription guide (v1)

- Transcribe what is printed, in reading order: left column then right, top to bottom.
- Include running headers and printed page numbers, each on its own line, where they appear.
- Rejoin words hyphenated at line breaks (write "moment", not "mo-ment").
- Expand ligatures (fi, fl) and write the long s as a normal s. Keep original spelling otherwise.
- Tables: one printed row per line, cells separated by " | ". Transcribe the header rows the same way.
- Equations: LaTeX, display equations as $$...$$ on their own line, inline math as $...$.
- Illegible span: write <?> in its place. It is excluded from scoring - never guess.
- Figure pages: transcribe the caption and any axis labels you can read; nothing else.
- Blank page: leave the editor empty and mark done.
"""

def _log_state():
    latest, tail = {}, []
    try:
        lines = common.LOG.read_text().splitlines()
    except FileNotFoundError:
        return latest, tail
    for ln in lines:
        try:
            e = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if "ntrs_id" in e:
            latest[e["ntrs_id"]] = e
        tail.append(e)
    return latest, tail[-12:]

def scan():
    rows = common.load_reports()
    log_latest, log_tail = _log_state()
    reports, totals = [], {"done": 0, "processing": 0, "queued": 0, "failed": 0,
                           "missing": 0, "pages_rendered": 0, "baseline_pages": 0,
                           "flagged_pages": 0, "baseline_absent": 0}
    for r in rows:
        d = common.report_dir(r["ntrs_id"])
        rep = {"ntrs_id": r["ntrs_id"], "report": r["report_number"],
               "series": r["series"], "decade": r["decade"],
               "pages": None, "raw": 0, "deskewed": 0, "baseline": 0,
               "dict_share": None, "flags": 0, "status": "queued"}
        ev = log_latest.get(r["ntrs_id"], {})
        meta_p, stats_p = d / "meta.json", d / "baseline" / "stats.json"
        if (d / "pages").is_dir():
            pngs = list((d / "pages").glob("*.png"))
            rep["raw"] = sum(1 for p in pngs if p.name.endswith(".raw.png"))
            rep["deskewed"] = len(pngs) - rep["raw"]
        if (d / "baseline").is_dir():
            rep["baseline"] = len(list((d / "baseline").glob("*.txt")))
        if meta_p.exists():
            meta = json.loads(meta_p.read_text())
            rep["status"] = "missing" if meta.get("sha256") is None else "processing"
        if stats_p.exists():
            s = json.loads(stats_p.read_text())
            rep["pages"] = s["pdf_pages"]
            shares = [v["dict_share"] for v in s["pages"].values() if v["words"] > 0]
            rep["dict_share"] = round(sum(shares) / len(shares), 3) if shares else 0.0
            rep["flags"] = sum(1 for v in s["pages"].values() if "flag" in v)
            totals["flagged_pages"] += rep["flags"]
            if s["baseline_absent"]:
                totals["baseline_absent"] += 1
            if rep["raw"] == rep["deskewed"] == rep["baseline"] == s["pdf_pages"]:
                rep["status"] = "done"
        if ev.get("event") == "report_failed":
            rep["status"] = "failed"; rep["error"] = ev.get("error", "")[:120]
        totals[rep["status"]] = totals.get(rep["status"], 0) + 1
        totals["pages_rendered"] += rep["raw"]
        totals["baseline_pages"] += rep["baseline"]
        reports.append(rep)
    running = subprocess.run(["pgrep", "-fc", "pass[1-9].py"], capture_output=True,
                             text=True).stdout.strip()
    du = subprocess.run(["du", "-sh", str(common.DATA)], capture_output=True, text=True)
    return {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pass1_running": running not in ("", "0"),
            "disk": du.stdout.split()[0] if du.stdout else "?",
            "n_reports": len(rows), "totals": totals,
            "reports": reports, "log_tail": log_tail}

def status_cached():
    with _lock:
        if time.time() - _cache["t"] > 5:
            _cache["data"] = scan()
            _cache["t"] = time.time()
        return _cache["data"]

# ---------------------------------------------------------------- performance
_perf_cache = {"t": 0.0, "data": None}
_steps_cache = {"t": 0.0, "data": None}

def _tstats(xs):
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    return {"n": n, "median": round(xs[n // 2], 2), "mean": round(sum(xs) / n, 2),
            "p90": round(xs[min(n - 1, int(n * 0.9))], 2),
            "total_h": round(sum(xs) / 3600, 2)}

def _cpu_steps():
    """Per-page seconds for passes 1-3 (heavy file walk; cached 15 min)."""
    if time.time() - _steps_cache["t"] < 900 and _steps_cache["data"]:
        return _steps_cache["data"]
    lay, ocr, ras, base = [], [], [], []
    npages = 0
    for nid in VALID_IDS:
        d = common.report_dir(nid)
        if (d / "layout").is_dir():
            for lj in (d / "layout").glob("[0-9]*.json"):
                sec = json.loads(lj.read_text()).get("seconds")
                if sec is not None:
                    lay.append(sec)
        if (d / "ocr").is_dir():
            for oj in (d / "ocr").glob("[0-9]*.json"):
                npages += 1
                sec = json.loads(oj.read_text()).get("seconds")
                if sec is not None:
                    ocr.append(sec)
        mp = d / "meta.json"
        if mp.exists():
            t = json.loads(mp.read_text()).get("timings")
            if t and t.get("pages"):
                ras.append(t["rasterize_s"] / t["pages"])
                base.append(t["baseline_s"] / t["pages"])
    steps = []
    for name, xs in (("1 rasterize", ras), ("1 baseline", base),
                     ("2 layout", lay), ("3a tesseract", ocr)):
        st = _tstats(xs)
        if st:
            st.update(step=name, coverage=round(st["n"] / max(1, npages), 3))
            steps.append(st)
    _steps_cache.update(t=time.time(), data=(steps, npages))
    return steps, npages

def perf_scan():
    steps, npages = _cpu_steps()
    now = time.time()
    engines = {}
    for nid in VALID_IDS:
        for vd in common.report_dir(nid).glob("vision*"):
            if not vd.is_dir():
                continue
            e = engines.setdefault(vd.name, {
                "pages": 0, "retried": 0, "vision_failed": 0, "tok_in": 0,
                "tok_out": 0, "secs": [], "recent": 0, "last_ts": 0.0})
            for mj in vd.glob("*.meta.json"):
                try:
                    m = json.loads(mj.read_text())
                except Exception:
                    continue
                e["pages"] += 1
                if "seconds" in m:
                    e["secs"].append(m["seconds"])
                g = m.get("loop_guard") or {}
                e["retried"] += bool(g.get("retried"))
                e["vision_failed"] += bool(g.get("vision_failed"))
                u = m.get("usage") or {}
                e["tok_in"] += u.get("prompt_tokens", 0)
                e["tok_out"] += u.get("completion_tokens", 0)
                mt = mj.stat().st_mtime
                e["last_ts"] = max(e["last_ts"], mt)
                if now - mt < 900:
                    e["recent"] += 1
    vrows = []
    for name, e in sorted(engines.items()):
        st = _tstats(e["secs"]) or {"n": e["pages"]}
        st.update(step="4a " + name, coverage=None,
                  retried=e["retried"], vision_failed=e["vision_failed"],
                  tok_in=e["tok_in"], tok_out=e["tok_out"],
                  rate_ph=e["recent"] * 4, active=now - e["last_ts"] < 180)
        vrows.append(st)
    run = None
    try:
        tgt = int(json.loads((common.DATA / "gate_v1.json").read_text())
                  ["calibration"]["corpus_escalated"].split("/")[0])
    except Exception:
        tgt = None
    v2 = engines.get("vision2")
    if tgt and v2:
        rate = v2["recent"] * 4
        done = min(v2["pages"], tgt)
        run = {"target": tgt, "done": done, "rate_ph": rate,
               "eta_h": round((tgt - done) / rate, 1) if rate and done < tgt else None,
               "retried": v2["retried"], "vision_failed": v2["vision_failed"],
               "tok_in": v2["tok_in"], "tok_out": v2["tok_out"],
               "active": now - v2["last_ts"] < 180}
    return {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "steps": steps + vrows, "npages": npages, "run": run}

def perf_cached():
    with _lock:
        if time.time() - _perf_cache["t"] > 20:
            _perf_cache["data"] = perf_scan()
            _perf_cache["t"] = time.time()
        return _perf_cache["data"]

_live_cache = {"t": 0.0, "data": None}

def live_scan():
    stations = pipeline_state()
    train = training_state(TRAIN_LOG, alive=_pgrep("train_8b_lora.py"),
                           total_steps=_train_total_steps())
    ckpts = sorted(TRAIN_RUNS.glob("checkpoint-*")) if TRAIN_RUNS.is_dir() else []
    train["checkpoints"] = len(ckpts)
    train["adapter_saved"] = (TRAIN_RUNS / "adapter_final").is_dir()
    tiers = tier_counts(CORPUS_DB)
    docs = 0
    if CORPUS_DB.exists():
        try:
            conn = sqlite3.connect(f"file:{CORPUS_DB}?mode=ro", uri=True)
            docs = conn.execute("SELECT count(DISTINCT doc_id) FROM pages").fetchone()[0]
            conn.close()
        except sqlite3.Error:
            pass
    return {"stations": stations, "training": train, "system": system_scan(),
            "collection": collection_meta(),
            "tiers": {k: tiers[k] for k in TIER_ORDER if k in tiers},
            "docs": docs, "pages": sum(tiers.values())}

def live_cached():
    with _lock:
        if time.time() - _live_cache["t"] > 5:
            _live_cache["data"] = live_scan()
            _live_cache["t"] = time.time()
        return _live_cache["data"]

# ---------------------------------------------------------------- transcription
def gt_paths(nid, page):
    d = common.report_dir(nid) / "ground_truth"
    return d / f"{page}.txt", d / f"{page}.json"

def tuning_queue():
    out = []
    with open(common.DATA / "tuning_pages.csv", newline="") as f:
        for row in csv.DictReader(f):
            txt_p, meta_p = gt_paths(row["ntrs_id"], int(row["pdf_page"]))
            st = "todo"
            if meta_p.exists():
                st = json.loads(meta_p.read_text()).get("status", "draft")
            elif txt_p.exists():
                st = "draft"
            out.append({"ntrs_id": row["ntrs_id"], "page": int(row["pdf_page"]),
                        "report": row["report_number"], "series": row["series"],
                        "category": row["category"], "focus": row["focus"], "status": st})
    guide_p = common.DATA / "transcription_guide.md"
    if not guide_p.exists():
        guide_p.write_text(GUIDE_DEFAULT)
    return {"queue": out, "guide": guide_p.read_text()}

def page_payload(nid, page):
    base = common.report_dir(nid) / "baseline" / f"{page}.txt"
    ocr_p = common.report_dir(nid) / "ocr" / f"{page}.json"
    ocr = json.loads(ocr_p.read_text()) if ocr_p.exists() else None
    txt_p, meta_p = gt_paths(nid, page)
    return {"baseline": base.read_text(errors="replace") if base.exists() else "",
            "ocr": ocr["text"] if ocr else None,
            "ocr_engine": f"{ocr['engine']} {ocr['version']}" if ocr else None,
            "transcript": txt_p.read_text() if txt_p.exists() else None,
            "meta": json.loads(meta_p.read_text()) if meta_p.exists() else None}

def save_transcript(body):
    nid, page = body["ntrs_id"], int(body["page"])
    if nid not in VALID_IDS:
        raise ValueError("unknown ntrs_id")
    status = body.get("status", "draft")
    if status not in ("draft", "done"):
        raise ValueError("bad status")
    txt_p, meta_p = gt_paths(nid, page)
    txt_p.parent.mkdir(exist_ok=True)
    txt_p.write_text(body.get("text", ""))
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seeded_from": body.get("seeded_from", "baseline")}
    meta.update(status=status, updated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                chars=len(body.get("text", "")))
    meta_p.write_text(json.dumps(meta, indent=1))
    return {"ok": True, "status": status, "chars": meta["chars"]}

COMMON_CSS = r"""
:root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
 --grid:#e1e0d9;--border:rgba(11,11,11,.10);--accent:#2a78d6;
 --good:#0ca30c;--serious:#ec835a;--critical:#d03b3b;--good-text:#006300}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;
 --ink2:#c3c2b7;--grid:#2c2c2a;--border:rgba(255,255,255,.10);--accent:#3987e5;--good-text:#0ca30c}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);
 font:15px/1.45 system-ui,sans-serif}
button,select{font:13px system-ui;background:var(--surface);color:var(--ink);
 border:1px solid var(--border);border-radius:6px;padding:5px 10px;cursor:pointer}
button:hover{border-color:var(--accent)}
a{color:var(--accent);text-decoration:none}
"""

DASH_HTML = (common.DATA.parent / "pipeline" / "dashboard.html").read_text()

TRANS_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Transcribe</title><style>__CSS__
body{display:flex;flex-direction:column;height:100vh;overflow:hidden}
header{display:flex;gap:8px;align-items:center;padding:8px 12px;flex-wrap:wrap;
 border-bottom:1px solid var(--border);background:var(--surface)}
header .meta{font-size:12px;color:var(--ink2);margin-right:auto}
#prog{font-size:12px;color:var(--ink2)}
#state{font-size:12px;font-weight:600}
.st-done{color:var(--good-text)}.st-draft{color:var(--accent)}.st-todo{color:var(--muted)}
main{display:flex;flex:1;min-height:0}
#viewer{flex:1;overflow:hidden;position:relative;background:var(--page);cursor:grab;
 border-right:1px solid var(--border)}
#viewer img{position:absolute;transform-origin:0 0;user-select:none;-webkit-user-drag:none}
#right{flex:1;display:flex;flex-direction:column;min-width:0}
#guide{font-size:12.5px;color:var(--ink2);border-bottom:1px solid var(--border);
 padding:4px 12px;max-height:30vh;overflow-y:auto;background:var(--surface)}
#guide pre{white-space:pre-wrap;margin:6px 0;font:12px/1.5 system-ui}
textarea{flex:1;resize:none;border:0;outline:0;padding:12px 14px;background:var(--page);
 color:var(--ink);font:14px/1.55 ui-monospace,Menlo,Consolas,monospace;white-space:pre-wrap}
footer{display:flex;gap:10px;padding:6px 12px;font-size:12px;color:var(--ink2);
 border-top:1px solid var(--border);background:var(--surface)}
@media (max-width:900px){main{flex-direction:column}#viewer{border-right:0;
 border-bottom:1px solid var(--border)}}
</style></head><body>
<header>
 <a href="/">← dashboard</a>
 <select id="queue"></select>
 <button id="prev">‹ prev</button><button id="next">next ›</button>
 <span class="meta" id="meta"></span>
 <span id="prog"></span><span id="state"></span>
 <button id="unc" title="Ctrl+.">insert ⟨?⟩</button>
 <button id="reseed" title="replace editor with the 1990s baseline text">reseed</button>
 <button id="rawtog">raw scan</button>
 <button id="save" title="Ctrl+S">save draft</button>
 <button id="donebtn">mark done ✓</button>
</header>
<main>
 <div id="viewer"><img id="pg" draggable="false"></div>
 <div id="right">
  <details id="gd"><summary style="padding:4px 12px;font-size:12px;cursor:pointer">transcription guide</summary>
   <div id="guide"><pre id="guidetext"></pre></div></details>
  <textarea id="ed" spellcheck="false" placeholder="loading…"></textarea>
  <details id="bd"><summary style="padding:4px 12px;font-size:12px;cursor:pointer;border-top:1px solid var(--border)">reference: 1990s baseline layer (what NTRS search sees today)</summary>
   <pre id="basetext" style="max-height:22vh;overflow:auto;margin:0;padding:8px 12px;font:12px/1.5 ui-monospace,monospace;color:var(--ink2);background:var(--surface);white-space:pre-wrap"></pre></details>
  <footer><span id="chars"></span><span id="savemsg"></span>
   <span style="margin-left:auto">zoom: wheel · pan: drag · reset: double-click</span></footer>
 </div>
</main>
<script>
let Q=[], idx=0, dirty=false, raw=false, scale=0.28, tx=0, ty=0, seededFrom="baseline";
const $=id=>document.getElementById(id);
const cur=()=>Q[idx];

function applyT(){ $("pg").style.transform=`translate(${tx}px,${ty}px) scale(${scale})`; }
function resetView(){ const v=$("viewer"), im=$("pg");
 if(!im.naturalWidth) return;
 scale=Math.min(v.clientWidth/im.naturalWidth, v.clientHeight/im.naturalHeight);
 tx=(v.clientWidth-im.naturalWidth*scale)/2; ty=0; applyT(); }
$("pg").onload=resetView;
$("viewer").addEventListener("wheel",e=>{e.preventDefault();
 const f=e.deltaY<0?1.15:1/1.15, r=$("viewer").getBoundingClientRect();
 const mx=e.clientX-r.left, my=e.clientY-r.top;
 tx=mx-(mx-tx)*f; ty=my-(my-ty)*f; scale*=f; applyT();},{passive:false});
let drag=null;
$("viewer").addEventListener("pointerdown",e=>{drag={x:e.clientX-tx,y:e.clientY-ty};
 $("viewer").setPointerCapture(e.pointerId);});
$("viewer").addEventListener("pointermove",e=>{if(drag){tx=e.clientX-drag.x;ty=e.clientY-drag.y;applyT();}});
$("viewer").addEventListener("pointerup",()=>drag=null);
$("viewer").addEventListener("dblclick",resetView);

function imgUrl(){ return `/img/${cur().ntrs_id}/${cur().page}${raw?".raw":""}.png`; }
function paintState(st){ cur().status=st;
 $("state").textContent={todo:"○ not started",draft:"● draft",done:"✓ done"}[st];
 $("state").className="st-"+st;
 const done=Q.filter(q=>q.status==="done").length;
 $("prog").textContent=`${done}/${Q.length} done`;
 $("queue").options[idx].text=optLabel(Q[idx]); }
function optLabel(q){ return `${q.status==="done"?"✓ ":q.status==="draft"?"● ":"○ "}${q.report} p${q.page} — ${q.category}`; }

async function load(i){
 if(dirty) await save("draft",true);
 idx=i; const q=cur(); $("queue").selectedIndex=i;
 $("meta").textContent=`${q.report} page ${q.page} · ${q.series} · ${q.focus}`;
 raw=false; $("rawtog").textContent="raw scan";
 $("pg").src=imgUrl();
 const r=await fetch(`/api/page?ntrs_id=${q.ntrs_id}&page=${q.page}`);
 const d=await r.json();
 const seed = d.ocr!==null ? d.ocr : d.baseline;
 seededFrom = d.ocr!==null ? (d.ocr_engine||"tesseract") : "baseline";
 $("ed").value = d.transcript!==null ? d.transcript : seed;
 $("basetext").textContent = d.baseline.trim() || "(baseline layer empty for this page)";
 dirty=false; $("savemsg").textContent = d.transcript!==null?"loaded saved transcript":"seeded from "+seededFrom;
 paintState(q.status); count(); $("ed").focus();
}
function count(){ $("chars").textContent=$("ed").value.length+" chars"; }
$("ed").addEventListener("input",()=>{dirty=true;count();$("savemsg").textContent="unsaved changes";});

async function save(status,quiet){
 const q=cur();
 const r=await fetch("/api/transcript",{method:"POST",headers:{"Content-Type":"application/json"},
  body:JSON.stringify({ntrs_id:q.ntrs_id,page:q.page,text:$("ed").value,status:status,seeded_from:seededFrom})});
 if(r.ok){ dirty=false; if(!quiet)$("savemsg").textContent="saved "+new Date().toLocaleTimeString();
  paintState(status);}
 else $("savemsg").textContent="SAVE FAILED "+r.status;
}
$("save").onclick=()=>save(cur().status==="done"?"done":"draft");
$("donebtn").onclick=()=>save("done");
$("unc").onclick=()=>{const e=$("ed"),s=e.selectionStart;
 e.setRangeText("⟨?⟩",s,e.selectionEnd,"end");dirty=true;count();e.focus();};
$("reseed").onclick=async()=>{ if(!confirm("Replace editor contents with the OCR seed?"))return;
 const q=cur(); const d=await (await fetch(`/api/page?ntrs_id=${q.ntrs_id}&page=${q.page}`)).json();
 $("ed").value=d.ocr!==null?d.ocr:d.baseline; dirty=true; count();};
$("rawtog").onclick=()=>{raw=!raw;$("rawtog").textContent=raw?"deskewed":"raw scan";$("pg").src=imgUrl();};
$("prev").onclick=()=>idx>0&&load(idx-1);
$("next").onclick=()=>idx<Q.length-1&&load(idx+1);
$("queue").onchange=e=>load(e.target.selectedIndex);
document.addEventListener("keydown",e=>{
 if((e.ctrlKey||e.metaKey)&&e.key==="s"){e.preventDefault();$("save").click();}
 if((e.ctrlKey||e.metaKey)&&e.key==="."){e.preventDefault();$("unc").click();}});
setInterval(()=>{if(dirty)save(cur().status==="done"?"done":"draft",true);},30000);
window.addEventListener("beforeunload",e=>{if(dirty)e.preventDefault();});

(async()=>{
 const d=await (await fetch("/api/tuning")).json();
 Q=d.queue; $("guidetext").textContent=d.guide;
 $("queue").innerHTML=Q.map(q=>`<option>${optLabel(q)}</option>`).join("");
 const first=Q.findIndex(q=>q.status!=="done");
 load(first<0?0:first);
})();
</script></body></html>"""

class H(BaseHTTPRequestHandler):
    def _send(self, body, ct, code=200, cache=False):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "max-age=86400")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(url.query)
        try:
            if url.path == "/api/status":
                self._send(json.dumps(status_cached()).encode(), "application/json")
            elif url.path == "/api/perf":
                self._send(json.dumps(perf_cached()).encode(), "application/json")
            elif url.path == "/api/live":
                self._send(json.dumps(live_cached()).encode(), "application/json")
            elif url.path == "/api/tuning":
                self._send(json.dumps(tuning_queue()).encode(), "application/json")
            elif url.path == "/api/page":
                nid, page = qs["ntrs_id"][0], int(qs["page"][0])
                if nid not in VALID_IDS:
                    self.send_error(400); return
                self._send(json.dumps(page_payload(nid, page)).encode(), "application/json")
            elif url.path == "/":
                self._send(DASH_HTML.encode(), "text/html; charset=utf-8")
            elif url.path == "/transcribe":
                self._send(TRANS_HTML.replace("__CSS__", COMMON_CSS).encode(), "text/html; charset=utf-8")
            else:
                m = re.fullmatch(r"/img/(\d+)/(\d+)(\.raw)?\.png", url.path)
                if m and m.group(1) in VALID_IDS:
                    p = common.report_dir(m.group(1)) / "pages" / f"{m.group(2)}{m.group(3) or ''}.png"
                    if p.exists():
                        self._send(p.read_bytes(), "image/png", cache=True); return
                self.send_error(404)
        except Exception as e:
            self.send_error(500, str(e)[:120])

    def do_POST(self):
        if self.path != "/api/transcript":
            self.send_error(404); return
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self._send(json.dumps(save_transcript(body)).encode(), "application/json")
        except Exception as e:
            self.send_error(400, str(e)[:120])

    def log_message(self, *a):
        pass

if __name__ == "__main__":
    print(f"dashboard http://0.0.0.0:{PORT}/  ·  transcription http://0.0.0.0:{PORT}/transcribe", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
