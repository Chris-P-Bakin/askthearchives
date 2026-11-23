"""SPIKE (throwaway): section-aware paragraph chunking + Qwen3-Embedding-0.6B trial.

Chunks every ingested doc's best-text pages (stitched across page breaks),
embeds via the local llama.cpp server, writes chunks.jsonl + embs.npy.
"""
import json, re, sys, time, urllib.request
from pathlib import Path

ROOT = Path("~/data_normalizing")
sys.path.insert(0, str(ROOT / "pipeline"))
import build_index as bi  # noqa: E402

OUT = Path(__file__).parent
EMBED_URL = "http://127.0.0.1:8201/v1/embeddings"
HEADING = re.compile(r"^[A-Z][A-Z0-9 ,.\-()']{3,60}$")
TARGET_WORDS, MAX_WORDS = 200, 280

def doc_meta():
    import csv
    return {r["ntrs_id"]: r for r in csv.DictReader(open(ROOT / "data/roster.csv"))}

def pages_of(doc: Path):
    nums = sorted(int(p.stem) for p in (doc / "pages").glob("*.png")
                  if not p.name.endswith(".raw.png"))
    for n in nums:
        pt = bi.best_text(doc, n)
        if pt and pt.text.strip():
            yield n, pt.text

def chunk_doc(doc: Path, title: str):
    """Stitch pages, split into heading-aware paragraph packs."""
    section = ""
    buf, buf_pages, out = [], set(), []

    def flush():
        nonlocal buf, buf_pages
        words = " ".join(buf).split()
        if len(words) >= 15:
            out.append({"text": " ".join(words), "section": section,
                        "pages": sorted(buf_pages)})
        buf, buf_pages = [], set()

    for n, text in pages_of(doc):
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if not para:
                continue
            first = para.splitlines()[0].strip()
            if HEADING.match(first) and len(para.split()) < 40:
                flush()
                section = first.title()
                rest = "\n".join(para.splitlines()[1:]).strip()
                if not rest:
                    continue
                para = rest
            if sum(len(b.split()) for b in buf) + len(para.split()) > MAX_WORDS:
                flush()
            buf.append(para)
            buf_pages.add(n)
            if sum(len(b.split()) for b in buf) >= TARGET_WORDS:
                flush()
    flush()
    for c in out:
        prefix = title[:120]
        if c["section"]:
            prefix += " - " + c["section"]
        c["embed_text"] = f"{prefix}: {c['text']}"
    return out

def embed_batch(texts):
    body = json.dumps({"input": texts}).encode()
    req = urllib.request.Request(EMBED_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return [row["embedding"] for row in sorted(d["data"], key=lambda x: x["index"])]

def main():
    import numpy as np
    meta = doc_meta()
    docs = sorted(p for p in (ROOT / "data").iterdir()
                  if p.is_dir() and p.name[0].isdigit() and (p / "pages").exists())
    chunks, t0 = [], time.time()
    for i, doc in enumerate(docs):
        title = meta.get(doc.name, {}).get("title", "")
        for c in chunk_doc(doc, title):
            c["doc"] = doc.name
            chunks.append(c)
        if i % 100 == 0:
            print(f"chunked {i}/{len(docs)} docs, {len(chunks)} chunks", flush=True)
    print(f"TOTAL {len(chunks)} chunks from {len(docs)} docs in {time.time()-t0:.0f}s", flush=True)

    with open(OUT / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps({k: c[k] for k in ("doc", "pages", "section", "text")}) + "\n")

    vecs, B = [], 64
    t0 = time.time()
    for i in range(0, len(chunks), B):
        batch = [c["embed_text"][:6000] for c in chunks[i:i + B]]
        vecs.extend(embed_batch(batch))
        if (i // B) % 50 == 0:
            el = time.time() - t0
            rate = (i + B) / el if el else 0
            print(f"embedded {i+len(batch)}/{len(chunks)} ({rate:.0f} chunks/s)", flush=True)
    a = np.asarray(vecs, dtype=np.float16)
    np.save(OUT / "embs.npy", a)
    print(f"DONE embeddings {a.shape} in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
