"""Submit page-image transcription jobs to the Claude Message Batches API.

Used for distillation labels (hard table pages) and GT drafting. Half-price
batch processing; results within 24h, usually minutes to hours.

  submit:  python pipeline/claude_batch.py submit JOBS.jsonl --prompt FILE \
               [--model claude-sonnet-4-5] [--max-tokens 8000] [--tag NAME]
           JOBS.jsonl lines: {"custom_id": "...", "png": "path/to/page.png"}
  poll:    python pipeline/claude_batch.py poll BATCH_ID
  fetch:   python pipeline/claude_batch.py fetch BATCH_ID --out DIR
           writes <custom_id>.txt per succeeded request + a summary json

Key from ANTHROPIC_API_KEY (repo .env). Stdlib only.
"""
import argparse
import base64
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

API = "https://api.anthropic.com/v1"
ROOT = Path(__file__).resolve().parent.parent


def _key():
    k = os.environ.get("ANTHROPIC_API_KEY")
    if not k:
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith(("ANTHROPIC_API_KEY=", "ANTHROPIC_KEY=")):
                k = line.split("=", 1)[1].strip()
    if not k:
        raise SystemExit("ANTHROPIC_API_KEY not set (env or .env)")
    return k


def _call(path, payload=None, method=None):
    req = urllib.request.Request(
        f"{API}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        method=method or ("POST" if payload is not None else "GET"),
        headers={"x-api-key": _key(), "anthropic-version": "2023-06-01",
                 "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def submit(jobs_path, prompt_path, model, max_tokens, tag, chunk=40):
    prompt = Path(prompt_path).read_text()
    jobs = [json.loads(l) for l in open(jobs_path) if l.strip()]
    for ci in range(0, len(jobs), chunk):
        _submit_chunk(jobs[ci:ci + chunk], prompt, model, max_tokens,
                      f"{tag}-{ci // chunk}" if len(jobs) > chunk else tag)


def _submit_chunk(jobs, prompt, model, max_tokens, tag):
    requests = []
    for j in jobs:
        png = Path(j["png"])
        media = "image/jpeg" if png.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        b64 = base64.b64encode(png.read_bytes()).decode()
        requests.append({
            "custom_id": j["custom_id"],
            "params": {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": 0,
                "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64",
                     "media_type": media, "data": b64}},
                    {"type": "text", "text": prompt}]}]}})
    batch = _call("/messages/batches", {"requests": requests})
    rec = {"batch_id": batch["id"], "tag": tag, "n": len(requests),
           "model": model, "submitted": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    ledger = ROOT / "data" / "claude_batches.jsonl"
    with open(ledger, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


def poll(batch_id):
    b = _call(f"/messages/batches/{batch_id}")
    print(json.dumps({"id": b["id"], "status": b["processing_status"],
                      "counts": b.get("request_counts")}))
    return b


def fetch(batch_id, out_dir):
    b = poll(batch_id)
    if b["processing_status"] != "ended":
        raise SystemExit("batch not ended yet")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    url = b["results_url"]
    req = urllib.request.Request(url, headers={
        "x-api-key": _key(), "anthropic-version": "2023-06-01"})
    ok = err = 0
    usage_in = usage_out = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            res = json.loads(line)
            cid = res["custom_id"]
            if res["result"]["type"] == "succeeded":
                msg = res["result"]["message"]
                text = "".join(c["text"] for c in msg["content"]
                               if c["type"] == "text")
                (out / f"{cid}.txt").write_text(text)
                usage_in += msg["usage"]["input_tokens"]
                usage_out += msg["usage"]["output_tokens"]
                ok += 1
            else:
                (out / f"{cid}.error.json").write_text(json.dumps(res["result"]))
                err += 1
    summary = {"batch_id": batch_id, "succeeded": ok, "errored": err,
               "input_tokens": usage_in, "output_tokens": usage_out,
               "batch_cost_usd": round(usage_in * 1.5e-6 + usage_out * 7.5e-6, 2)}
    (out / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("jobs")
    s.add_argument("--prompt", required=True)
    s.add_argument("--model", default="claude-sonnet-4-5")
    s.add_argument("--max-tokens", type=int, default=8000)
    s.add_argument("--tag", default="")
    s.add_argument("--chunk", type=int, default=40)
    p = sub.add_parser("poll")
    p.add_argument("batch_id")
    f = sub.add_parser("fetch")
    f.add_argument("batch_id")
    f.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "submit":
        submit(a.jobs, a.prompt, a.model, a.max_tokens, a.tag, a.chunk)
    elif a.cmd == "poll":
        poll(a.batch_id)
    else:
        fetch(a.batch_id, a.out)


if __name__ == "__main__":
    main()
