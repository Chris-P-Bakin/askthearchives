"""SPIKE (throwaway): build a retrieval bench of question -> (doc, page) pairs.

Samples GT pages + random campaign-layer pages, asks the 235B (OpenRouter,
text-only) for one researcher-style question per page, answerable from that
page alone, without naming the report. Writes queries.jsonl.
"""
import json, os, random, sys, time, urllib.request
from pathlib import Path

ROOT = Path("~/data_normalizing")
OUT = Path(__file__).parent / "queries.jsonl"
MODEL = "qwen/qwen3-vl-235b-a22b-instruct"
N_RANDOM = 70

def key():
    for line in open(ROOT / ".env"):
        if line.startswith("NACA_VLM_KEY="):
            return line.strip().split("=", 1)[1]
    raise SystemExit("no key")

PROMPT = """Below is one page from a NACA technical report (1920-1958 aeronautics).
Write ONE question a modern aerospace researcher might type into a search
engine, answerable from THIS page's content alone. Requirements:
- Do not mention the report number, title, authors, or the word "page".
- Ask about the technical substance (phenomena, methods, measured values,
  conclusions), the way someone who has never seen this report would ask.
- One sentence, no preamble, output the question only.

PAGE:
{page}"""

def ask(text, k):
    body = json.dumps({"model": MODEL, "max_tokens": 120, "temperature": 0.7,
                       "messages": [{"role": "user",
                                     "content": PROMPT.format(page=text[:6000])}]}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", body,
                                 {"Content-Type": "application/json",
                                  "Authorization": f"Bearer {k}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()

def main():
    random.seed(1958)
    k = key()
    pool = []
    for row in map(json.loads, open(ROOT / "data/gt_pages.jsonl")):
        p = ROOT / "data" / row["doc"] / "ground_truth" / f"{row['page']}.txt"
        if not p.exists():
            p = ROOT / "data" / row["doc"] / "vision_8bwin" / f"{row['page']}.txt"
        if p.exists():
            pool.append((row["doc"], row["page"], p))
    camp = list(ROOT.glob("data/*/vision_8blora_v2_1/*.txt"))
    random.shuffle(camp)
    for p in camp:
        if len(pool) >= 30 + N_RANDOM:
            break
        if len(p.read_text().split()) > 80:   # skip near-empty pages
            pool.append((p.parts[-3], int(p.stem), p))
    print(len(pool), "pages sampled")
    out = []
    for i, (doc, page, path) in enumerate(pool):
        try:
            q = ask(path.read_text(), k)
            out.append({"query": q, "doc": doc, "page": page})
            if i % 10 == 0:
                print(i, q[:90], flush=True)
        except Exception as e:
            print("skip", doc, page, repr(e)[:80], flush=True)
        time.sleep(0.3)
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print("wrote", len(out), "queries")

if __name__ == "__main__":
    main()
