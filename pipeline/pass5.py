"""Pass 5 — reconcile: assign each escalated page a text source + confidence.

v1 makes no new text. It reads the cross-engine agreement map (agreement.py)
and writes one decision record per page to data/{id}/reconcile/{n}.json:

  {"page_source": <engine dir>, "confidence": agreed|minor|conflict|single|blank,
   "inter_cer": float|null, "decided_by": "agreement_v1"}

Rules: text is the 27B's (measured best engine) except where it produced
nothing or its loop guard fired, in which case Haiku serves with confidence
"single". A later, smarter reconcile (arbiter engine, human review) replaces
records under a new decided_by value.

Also writes data/review_queue.jsonl ranked by expected payoff of human review:
vision_failed pages first, then conflicts by inter_cer * page size.
"""

import argparse
import json
from pathlib import Path

from common import DATA

DIR_A = "vision_27bq4"
DIR_B = "vision_haiku2"
DECIDER = "agreement_v1"


def decide_pair(row: dict) -> tuple[str, str]:
    """(page_source, confidence) for a page scored by both engines."""
    if row["a_chars"] == 0 and row["b_chars"] == 0:
        return DIR_A, "blank"
    if row["a_chars"] == 0:
        return DIR_B, "single"
    bucket = row["bucket"]
    if bucket == "agree":
        return DIR_A, "agreed"
    if bucket == "minor":
        return DIR_A, "minor"
    return DIR_A, "conflict"


def _meta_failed(rdir: Path, vdir: str, n: int) -> bool | None:
    """True if vision_failed, False if healthy, None if page absent."""
    mp = rdir / vdir / f"{n}.meta.json"
    if not mp.exists():
        return None
    try:
        m = json.loads(mp.read_text())
    except json.JSONDecodeError:
        return None
    return bool((m.get("loop_guard") or {}).get("vision_failed"))


def _write_record(rdir: Path, n: int, source: str, confidence: str,
                  inter_cer: float | None) -> None:
    out = rdir / "reconcile"
    out.mkdir(exist_ok=True)
    (out / f"{n}.json").write_text(json.dumps(
        {"page_source": source, "confidence": confidence,
         "inter_cer": inter_cer, "decided_by": DECIDER}))


def run_pass5(data_root: Path, agreement_path: Path, queue_path: Path) -> dict:
    counts: dict[str, int] = {}
    queue: list[dict] = []
    seen: set[tuple[str, int]] = set()

    lines = agreement_path.read_text().splitlines()
    for line in lines[1:]:  # line 0 is the header record
        row = json.loads(line)
        source, confidence = decide_pair(row)
        rdir = data_root / row["doc"]
        _write_record(rdir, row["page"], source, confidence, row["cer"])
        counts[confidence] = counts.get(confidence, 0) + 1
        seen.add((row["doc"], row["page"]))
        if confidence == "conflict":
            queue.append({"doc": row["doc"], "page": row["page"], "reason": "conflict",
                          "score": row["cer"] * max(row["a_chars"], row["b_chars"])})

    # pages absent from the agreement map: 27B failed or missing, haiku healthy
    for rdir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        bdir = rdir / DIR_B
        if not bdir.is_dir():
            continue
        for txt in bdir.glob("*.txt"):
            n = int(txt.stem)
            if (rdir.name, n) in seen:
                continue
            if _meta_failed(rdir, DIR_B, n) is not False:
                continue  # haiku itself failed or has no meta
            a_failed = _meta_failed(rdir, DIR_A, n)
            if a_failed is False:
                continue  # both healthy -> agreement map covers it (or should)
            _write_record(rdir, n, DIR_B, "single", None)
            counts["single"] = counts.get("single", 0) + 1
            if a_failed is True:
                queue.append({"doc": rdir.name, "page": n,
                              "reason": "vision_failed", "score": float("inf")})

    queue.sort(key=lambda r: -r["score"])
    with queue_path.open("w") as f:
        for r in queue:
            score = None if r["score"] == float("inf") else round(r["score"], 1)
            f.write(json.dumps({"doc": r["doc"], "page": r["page"],
                                "reason": r["reason"], "score": score}) + "\n")
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agreement", default=str(DATA / "agreement.jsonl"))
    ap.add_argument("--queue", default=str(DATA / "review_queue.jsonl"))
    args = ap.parse_args()
    counts = run_pass5(DATA, Path(args.agreement), Path(args.queue))
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
