#!/usr/bin/env python3
"""Escalation-queue consumer core: read/dedupe the queue, route pages to lanes, skip pages
a lane already finished, and keep append-only ledgers for non-filesystem outcomes."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import boinc_submit
import common
import table_verify
import units as units_mod

ESC_27B_HASH = "6a169a212aac"
LAYER_27B = "vision_27b_v2"
LAYER_235B = "vision_235b"
REVIEW_QUEUE = common.DATA / "escalation_review.jsonl"   # lane-owned; pass5 owns review_queue.jsonl
QUEUE = common.DATA / "escalation_queue.jsonl"
DONE = common.DATA / "escalation_done.jsonl"


def _data(data):
    return Path(data) if data is not None else common.DATA


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_jsonl(path):
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if isinstance(e, dict):
            out.append(e)
    return out


def _append(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


KINDS = ("table_flags", "vision_failed", "wu_error")


def read_queue_ex(path=None):
    """(rows, stats). Nothing vanishes: every non-blank line is a kept row, a dedupe drop,
    or malformed (raw line retained). stats = {lines, dedupe_dropped, malformed: [raw]}."""
    best, lines, malformed, dropped = {}, 0, [], 0
    p = Path(path if path is not None else QUEUE)
    for line in (p.read_text().splitlines() if p.exists() else []):
        if not line.strip():
            continue
        lines += 1
        try:
            r = json.loads(line)
        except ValueError:
            malformed.append(line)
            continue
        if not (isinstance(r, dict) and isinstance(r.get("unit_id"), str)
                and isinstance(r.get("kind"), str)):
            malformed.append(line)
            continue
        k = (r["unit_id"], r["kind"])
        if k in best:
            dropped += 1
            if str(r.get("ts", "")) < str(best[k].get("ts", "")):
                continue
        best[k] = r
    return list(best.values()), {"lines": lines, "dedupe_dropped": dropped,
                                 "malformed": malformed}


def read_queue(path=None) -> list:
    """Parsed rows deduped on (unit_id, kind), keeping the latest ts. Malformed lines are
    excluded here; use read_queue_ex for the accounting."""
    return read_queue_ex(path)[0]


def _page_png(row, data):
    if row.get("page") is None or not row.get("ntrs_id"):
        return None
    return data / str(row["ntrs_id"]) / "pages" / f"{row['page']}.png"


def route(rows, data=None) -> dict:
    data = _data(data)
    out = {"235b": [], "27b": [], "wu_error": [], "skipped_missing": [],
           "unknown_kind": [], "suppressed": []}
    table_pages = {(r.get("ntrs_id"), r.get("page")) for r in rows if r["kind"] == "table_flags"}
    for r in rows:
        png = _page_png(r, data)
        if r["kind"] not in KINDS:
            out["unknown_kind"].append(r)
        elif png is None or not png.exists():
            out["skipped_missing"].append(r)
        elif r["kind"] == "table_flags":
            out["235b"].append(r)
        elif r["kind"] == "vision_failed":
            if (r.get("ntrs_id"), r.get("page")) not in table_pages:
                out["27b"].append(r)
            else:
                out["suppressed"].append(r)
        elif r["kind"] == "wu_error":
            out["wu_error"].append(r)
    return out


def pending(rows, layer, hash, data=None) -> list:
    data = _data(data)
    keep = []
    for r in rows:
        m = data / str(r.get("ntrs_id")) / layer / f"{r.get('page')}.meta.json"
        try:
            if json.loads(m.read_text()).get("config_hash") == hash:
                continue
        except (OSError, ValueError):
            pass
        keep.append(r)
    return keep


def _done_path(data):
    return _data(data) / "escalation_done.jsonl"


def is_done(row, data=None) -> bool:
    k = (row["unit_id"], row["kind"])
    return any((e.get("unit_id"), e.get("kind")) == k for e in _read_jsonl(_done_path(data)))


def mark_done(row, disposition, data=None, **extra):
    _append(_done_path(data), dict(extra, unit_id=row["unit_id"], kind=row["kind"],
                                   disposition=disposition, ts=_now()))


def enqueue_review(row, reason, source_layer, config_hash, path=None) -> bool:
    path = Path(path) if path is not None else REVIEW_QUEUE
    for e in _read_jsonl(path):
        if e.get("unit_id") == row["unit_id"] and e.get("reason") == reason:
            return False
    _append(path, {"unit_id": row["unit_id"], "ntrs_id": row.get("ntrs_id"),
                   "page": row.get("page"), "reason": reason, "source_layer": source_layer,
                   "config_hash": config_hash, "ts": _now()})
    return True


ENV_27B = {
    "NACA_VLM_URL": "http://127.0.0.1:8781",
    "NACA_VLM_MODEL": "Qwen3.8-27B-UD-Q4_K_M",
    "NACA_VLM_PROMPT_FILE": str(common.ROOT / "pipeline" / "prompts" / "transcribe_v2.txt"),
    "NACA_VLM_THINK": "1",
    "NACA_VLM_DIR": "vision_27b_v2",
    "NACA_REPORTS_CSV": str(common.ROOT / "data" / "roster.csv"),
}


def run_27b(rows, limit=0, dry_run=False, data=None, review_path=None) -> dict:
    """Route vision_failed rows through the 27B teacher via pass4a. Counts:
    done / vision_failed / skipped (already done) / failed (no meta written) / planned."""
    data = _data(data)
    todo = pending(rows, LAYER_27B, ESC_27B_HASH, data=data)
    todo = [r for r in todo if not is_done(r, data)]
    res = {"done": 0, "vision_failed": 0, "skipped": len(rows) - len(todo),
           "failed": 0, "planned": 0, "empty": 0}
    if limit:
        todo = todo[:limit]
    if dry_run or not todo:
        res["planned"] = len(todo)
        return res
    runs = data / "escalation_runs"
    runs.mkdir(parents=True, exist_ok=True)
    pf = runs / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_27b.pages"
    pf.write_text("".join(json.dumps({"doc": r["ntrs_id"], "page": int(r["page"])}) + "\n"
                          for r in todo))
    env = {k: v for k, v in os.environ.items() if not k.startswith("NACA_VLM_")}
    env.update(ENV_27B)
    def _sig(r):
        mp = data / str(r["ntrs_id"]) / LAYER_27B / f"{r['page']}.meta.json"
        try:
            st = mp.stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None
    before = {r["unit_id"]: _sig(r) for r in todo}
    proc = subprocess.run([sys.executable, str(common.ROOT / "pipeline" / "pass4a.py"),
                    "--pages-file", str(pf), "--workers", "1"],
                   env=env, cwd=str(common.ROOT), check=False)
    rc = getattr(proc, "returncode", 0)
    if rc:
        res["pass4a_error"] = rc
    for r in todo:
        mp = data / str(r["ntrs_id"]) / LAYER_27B / f"{r['page']}.meta.json"
        if _sig(r) is None or _sig(r) == before[r["unit_id"]]:
            res["failed"] += 1   # no fresh meta this run (crash, or stale leftover)
            continue
        try:
            m = json.loads(mp.read_text())
        except (OSError, ValueError):
            res["failed"] += 1
            continue
        if m.get("config_hash") != ESC_27B_HASH:
            raise RuntimeError(f"27B config drift: {mp} has config_hash "
                               f"{m.get('config_hash')!r}, expected {ESC_27B_HASH!r}; "
                               "check ENV_27B vs server/prompt")
        try:
            txt = mp.with_suffix("").with_suffix(".txt").read_text()
        except OSError:
            txt = None
        if m.get("output_chars") == 0 or (txt is not None and not txt.strip()):
            # the loop guard skips text under min_chars, so empty output slips through
            enqueue_review(r, "empty_output_27b", LAYER_27B, ESC_27B_HASH, path=review_path)
            res["empty"] += 1
            continue
        if (m.get("loop_guard") or {}).get("vision_failed"):
            enqueue_review(r, "vision_failed_27b", LAYER_27B, ESC_27B_HASH, path=review_path)
            mark_done(r, "review", data=data, layer=LAYER_27B)
            res["vision_failed"] += 1
        else:
            mark_done(r, "done", data=data, layer=LAYER_27B)
            res["done"] += 1
    return res


ENV_235B = {
    "NACA_VLM_URL": "https://openrouter.ai/api",
    "NACA_VLM_MODEL": "qwen/qwen3-vl-235b-a22b-instruct",
    "NACA_VLM_PROMPT_FILE": str(common.ROOT / "pipeline" / "prompts" / "transcribe_v2.txt"),
    "NACA_VLM_DIR": "vision_235b",
    "NACA_REPORTS_CSV": str(common.ROOT / "data" / "roster.csv"),
}
COST_FLOOR_235B = 0.02   # per-page projection floor ($)


def is_v2_235b_meta(meta) -> bool:
    """True iff a vision_235b meta was written under the v2-prompt config (trial-era
    metas: v1 prompt/guard, hash 3669475d3a4b, do NOT qualify)."""
    if not isinstance(meta, dict):
        return False
    cfg = meta.get("config") or {}
    return (str(cfg.get("prompt_version", "")).startswith("file:")
            and cfg.get("model") == ENV_235B["NACA_VLM_MODEL"]
            and (cfg.get("loop_guard") or {}).get("version") == "v2")


def spend(data=None) -> float:
    """Total $ spent on the 235B lane: usage.cost over every vision_235b meta plus
    "cost" extras on done-ledger rows for that layer (pages with no meta cost)."""
    data = _data(data)
    total = 0.0
    for mp in data.glob(f"*/{LAYER_235B}/*.meta.json"):
        try:
            c = (json.loads(mp.read_text()).get("usage") or {}).get("cost")
        except (OSError, ValueError):
            continue
        if isinstance(c, (int, float)) and not isinstance(c, bool):
            total += c
    for e in _read_jsonl(_done_path(data)):
        c = e.get("cost")
        if e.get("layer") == LAYER_235B and isinstance(c, (int, float)) \
                and not isinstance(c, bool):
            total += c
    return total


def _meta_235b(r, data):
    return data / str(r["ntrs_id"]) / LAYER_235B / f"{r['page']}.meta.json"


def _sig_235b(r, data):
    try:
        st = _meta_235b(r, data).stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _already_235b(r, data) -> bool:
    try:
        return is_v2_235b_meta(json.loads(_meta_235b(r, data).read_text()))
    except (OSError, ValueError):
        return False


def _dispatch_235b(chunk, data, tag) -> int:
    runs = data / "escalation_runs"
    runs.mkdir(parents=True, exist_ok=True)
    pf = runs / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_235b_{tag}.pages"
    pf.write_text("".join(json.dumps({"doc": r["ntrs_id"], "page": int(r["page"])}) + "\n"
                          for r in chunk))
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("NACA_VLM_") or k == "NACA_VLM_KEY"}
    env.update(ENV_235B)
    proc = subprocess.run([sys.executable, str(common.ROOT / "pipeline" / "pass4a.py"),
                           "--pages-file", str(pf), "--workers", "1"],
                          env=env, cwd=str(common.ROOT), check=False)
    return getattr(proc, "returncode", 0)


def _finish_235b(r, data, res, review_path) -> bool:
    """Verify one page's fresh v2 meta/text and mark it (done or review). False if the
    meta/text is unreadable (counted failed by the caller)."""
    mp = _meta_235b(r, data)
    try:
        m = json.loads(mp.read_text())
    except (OSError, ValueError):
        return False
    if not is_v2_235b_meta(m):
        raise RuntimeError(f"235B config drift: {mp} not written under the v2 "
                           "prompt/guard config; check ENV_235B")
    h = m.get("config_hash")
    if (m.get("loop_guard") or {}).get("vision_failed"):
        enqueue_review(r, "vision_failed_235b", LAYER_235B, h, path=review_path)
        mark_done(r, "review", data=data, layer=LAYER_235B)
        res["vision_failed"] += 1
        return True
    try:
        text = mp.with_suffix("").with_suffix(".txt").read_text()
    except OSError:
        return False
    if not text.strip():
        enqueue_review(r, "empty_output_235b", LAYER_235B, h, path=review_path)
        res["empty"] = res.get("empty", 0) + 1
        return False
    if table_verify.verify_text(text)["flags"]:
        enqueue_review(r, "table_flags_after_235b", LAYER_235B, h, path=review_path)
        mark_done(r, "review", data=data, layer=LAYER_235B)
        res["still_flagged"] += 1
    else:
        mark_done(r, "done", data=data, layer=LAYER_235B)
        res["cleared"] += 1
    res["done"] += 1
    return True


def _bank_trial_cost(r, data):
    """A re-dispatch overwrites the page's existing (trial) meta; bank its usage.cost in
    the done ledger first so spend() keeps counting it. Idempotent per unit."""
    try:
        c = (json.loads(_meta_235b(r, data).read_text()).get("usage") or {}).get("cost")
    except (OSError, ValueError):
        return
    if isinstance(c, (int, float)) and not isinstance(c, bool) \
            and not is_done({"unit_id": r["unit_id"], "kind": "trial_cost"}, data):
        _append(_done_path(data), {"unit_id": r["unit_id"], "kind": "trial_cost",
                                   "layer": LAYER_235B, "cost": c,
                                   "disposition": "superseded_trial_meta", "ts": _now()})


def run_235b(rows, limit=0, chunk=25, budget=120.0, dry_run=False, data=None,
             review_path=None) -> dict:
    """Route table_flags rows through the hosted 235B via pass4a in budget-gated chunks.
    Done = ledger entry. A page with a v2 meta but no ledger entry (interrupted run) is
    recovered: verified and marked without re-dispatch. Counts: done (incl. cleared +
    still_flagged), cleared, still_flagged, vision_failed, recovered, skipped (ledgered),
    failed, spent; budget_stop=True if a chunk was refused."""
    data = _data(data)
    todo = [r for r in rows if not is_done(r, data)]
    recover = [r for r in todo if _already_235b(r, data)]
    fresh = [r for r in todo if r not in recover]
    res = {"done": 0, "cleared": 0, "still_flagged": 0, "vision_failed": 0,
           "recovered": 0, "skipped": len(rows) - len(todo), "failed": 0,
           "spent": spend(data)}
    if limit:
        fresh = fresh[:limit]
    if dry_run:
        res["planned"] = len(fresh)
        res["recoverable"] = len(recover)
        return res
    if fresh and not os.environ.get("NACA_VLM_KEY"):
        raise RuntimeError("NACA_VLM_KEY is not set; refusing to dispatch the 235B lane")
    for r in recover:
        if _finish_235b(r, data, res, review_path):
            res["recovered"] += 1
        else:
            res["failed"] += 1
    run_cost, run_pages = 0.0, 0
    for i in range(0, len(fresh), chunk):
        part = fresh[i:i + chunk]
        spent = spend(data)
        per_page = max(run_cost / run_pages if run_pages else 0.0, COST_FLOOR_235B)
        if spent + per_page * len(part) >= budget:
            res["budget_stop"] = True
            res["spent"] = spent
            return res
        before = {r["unit_id"]: _sig_235b(r, data) for r in part}
        for r in part:
            _bank_trial_cost(r, data)
        rc = _dispatch_235b(part, data, i // chunk)
        if rc:
            res["pass4a_error"] = rc
        run_pages += len(part)
        for r in part:
            sig = _sig_235b(r, data)
            if sig is None or sig == before[r["unit_id"]]:
                res["failed"] += 1
                continue
            try:
                c = (json.loads(_meta_235b(r, data).read_text()).get("usage") or {}).get("cost")
            except (OSError, ValueError):
                c = None
            if isinstance(c, (int, float)) and not isinstance(c, bool):
                run_cost += c
            if not _finish_235b(r, data, res, review_path):
                res["failed"] += 1
    res["spent"] = spend(data)
    return res

def _already_recovered(r, data) -> bool:
    m = data / str(r.get("ntrs_id")) / units_mod.CAMPAIGN_LAYER / f"{r.get('page')}.meta.json"
    try:
        return json.loads(m.read_text()).get("config_hash") in units_mod.ACCEPTED_HASHES
    except (OSError, ValueError, AttributeError):
        return False


MAX_WU_RETRY = 4   # highest __rN suffix tried before the unit is declared exhausted


def _latest_done_ts(row, data):
    k = (row["unit_id"], row["kind"])
    ts = [str(e.get("ts", "")) for e in _read_jsonl(_done_path(data))
          if (e.get("unit_id"), e.get("kind")) == k]
    return max(ts) if ts else None


def _wu_attempt(row, uid):
    """(base_name, attempt_no) from the failed row's wu_name: naca_<uid> -> 1, __rN -> N."""
    base = boinc_submit.wu_name(uid)
    n = str(row.get("wu_name") or base)
    if n.startswith(base + "__r") and n[len(base) + 3:].isdigit():
        return base, int(n[len(base) + 3:])
    return base, 1


def wu_error_open(row, data) -> bool:
    """One law for the wu_error ledger: open unless a done mark is at least as new as the row."""
    done_ts = _latest_done_ts(row, data)
    return done_ts is None or str(row.get("ts", "")) > done_ts


def run_wu_error(rows, units_file, dry_run=False, data=None, run=subprocess.run,
                 stage_dir=None, review_path=None, container=boinc_submit.CONTAINER) -> dict:
    """Resubmit BOINC workunits that errored server-side. Pages already recovered by another
    path (accepted-hash v8blora meta) are marked done without resubmit. create_work refuses a
    name that already exists (even errored), so a failure on the original name is retried once
    as naca_<unit_id>__r2 (stage_file registrations are name-independent, so no restage).
    Counts: resubmitted, already_recovered, failed (+ skipped = ledgered, planned on dry-run)."""
    data = _data(data)
    stage_dir = stage_dir if stage_dir is not None else boinc_submit.DEFAULT_STAGE
    by_id = {}
    for line in Path(units_file).read_text().splitlines():
        if line.strip():
            u = json.loads(line)
            by_id[u["unit_id"]] = u
    res = {"resubmitted": 0, "already_recovered": 0, "failed": 0, "skipped": 0}
    for r in rows:
        # A queue row newer than the unit's last ledger entry is a fresh wu_error (e.g. the
        # resubmitted WU itself errored): it reopens the unit. Older/equal stays skipped.
        if not wu_error_open(r, data):
            res["skipped"] += 1
            continue
        if _already_recovered(r, data):
            if not dry_run:
                mark_done(r, "already_recovered", data=data)
            res["already_recovered"] += 1
            continue
        if dry_run:
            res["planned"] = res.get("planned", 0) + 1
            continue
        base, k = _wu_attempt(r, r["unit_id"])
        if k >= MAX_WU_RETRY:
            enqueue_review(r, "wu_error_exhausted", None, None, path=review_path)
            mark_done(r, "retry_exhausted", data=data, wu_name=str(r.get("wu_name")))
            res["retry_exhausted"] = res.get("retry_exhausted", 0) + 1
            continue
        u = by_id.get(r["unit_id"])
        if u is None:
            print(f"{r['unit_id']}: FAILED unit not in {units_file}", file=sys.stderr)
            res["failed"] += 1
            continue
        try:
            boinc_submit.stage_into_project(u, data, stage_dir, run, container)
        except boinc_submit.StageError as e:
            print(f"{r['unit_id']}: FAILED {e}", file=sys.stderr)
            res["failed"] += 1
            continue
        # first-level errors also try the original name (create_work may accept it if the
        # errored WU row is gone); later levels go straight to the next suffix.
        names = [base, base + "__r2"] if k == 1 else [f"{base}__r{k + 1}"]
        errs = []
        used = None
        for name in names:
            p = run(boinc_submit._exec(container, boinc_submit.create_work_args(u, name)),
                    capture_output=True, text=True)
            if p.returncode == 0:
                used = name
                break
            errs.append(f"{name}: {(p.stderr or '').strip()}")
        if used is None:
            print(f"{r['unit_id']}: FAILED create_work: " + " | ".join(errs), file=sys.stderr)
            res["failed"] += 1
            res.setdefault("errors", []).append({"unit_id": r["unit_id"], "errors": errs})
            continue
        extra = {"wu_name": used}
        if errs:
            extra["first_error"] = errs[0]
        mark_done(r, "resubmitted", data=data, **extra)
        res["resubmitted"] += 1
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p27 = sub.add_parser("run-27b")
    p27.add_argument("--limit", type=int, default=0)
    p27.add_argument("--dry-run", action="store_true")
    p235 = sub.add_parser("run-235b")
    p235.add_argument("--limit", type=int, default=0)
    p235.add_argument("--budget", type=float, default=120.0)
    p235.add_argument("--dry-run", action="store_true")
    pw = sub.add_parser("run-wu-error")
    pw.add_argument("--units", required=True)
    pw.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        data = common.DATA
        rows, st = read_queue_ex(data / "escalation_queue.jsonl")
        routed = route(rows, data)
        lane_layer = {"27b": (LAYER_27B, ESC_27B_HASH)}
        print(f"lines: {st['lines']}  dedupe_dropped: {st['dedupe_dropped']}  "
              f"malformed: {len(st['malformed'])}")
        for lane, lrows in routed.items():
            total = len(lrows)
            if lane in lane_layer:
                lrows = pending(lrows, *lane_layer[lane], data=data)
            if lane == "wu_error":
                lrows = [r for r in lrows if wu_error_open(r, data)]
            else:
                lrows = [r for r in lrows if not is_done(r, data)]
            print(f"{lane}: {len(lrows)} pending ({total} routed)")
        acct = (st["dedupe_dropped"] + len(st["malformed"])
                + sum(len(v) for v in routed.values()))
        print(f"reconciliation: {acct} accounted / {st['lines']} lines "
              f"{'OK' if acct == st['lines'] else 'MISMATCH'}")
    elif a.cmd == "run-27b":
        rows = route(read_queue(QUEUE), common.DATA)["27b"]
        print(json.dumps(run_27b(rows, limit=a.limit, dry_run=a.dry_run)))
    elif a.cmd == "run-235b":
        rows = route(read_queue(QUEUE), common.DATA)["235b"]
        print(json.dumps(run_235b(rows, limit=a.limit, budget=a.budget, dry_run=a.dry_run)))
    elif a.cmd == "run-wu-error":
        rows = route(read_queue(QUEUE), common.DATA)["wu_error"]
        c = run_wu_error(rows, a.units, dry_run=a.dry_run)
        print(json.dumps(c))
        return 1 if c["failed"] else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
