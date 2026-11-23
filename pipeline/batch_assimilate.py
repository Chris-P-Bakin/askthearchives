"""Assimilate a validated batched (naca_vision_b) result page by page.

    python pipeline/batch_assimilate.py <manifest.json> <results.jsonl> \
        <batch.meta.json> --data-root R [--layer vision_8bwin] [--wu-name N]

Ok pages go through unit_assimilate.assimilate (content checks included —
a page rejected there escalates, it never fails the batch). Error pages and
rejected pages enqueue a wu_error row keyed (unit_id, kind, wu_name) so a
resubmission that fails again reopens the unit. --layer routes GT acceptance
batches (wu names naca_b_gt_*) to vision_8bwin without touching the campaign
layer.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unit_assimilate  # noqa: E402


def _escalate(root: Path, unit: dict, reason: str, wu_name: str, ts: str) -> None:
    row = {"unit_id": unit["unit_id"], "ntrs_id": unit.get("ntrs_id"),
           "page": unit.get("page"), "kind": "wu_error", "error_code": reason,
           "wu_name": wu_name, "ts": ts}
    unit_assimilate._enqueue(root, row, key=("unit_id", "kind", "wu_name"))


def assimilate_batch(manifest: list[dict], lines: list[str], data_root,
                     wu_name: str = "", layer: str | None = None) -> dict:
    root = Path(data_root)
    units = {u["unit_id"]: u for u in manifest}
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary = {"assimilated": 0, "duplicate": 0, "escalated": 0}
    for line in lines:
        row = json.loads(line)
        unit = units[row["unit_id"]]
        if row["status"] == "ok":
            res = unit_assimilate.assimilate(unit, row["text"], row["meta"],
                                             data_root=root, layer=layer)
            if res["status"] == "rejected":
                _escalate(root, unit, f"rejected:{res['reason']}", wu_name, ts)
                summary["escalated"] += 1
            else:
                summary[res["status"] if res["status"] in summary else "assimilated"] += 1
        else:
            _escalate(root, unit, row.get("error", "error"), wu_name, ts)
            summary["escalated"] += 1
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("results")
    ap.add_argument("batch_meta")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--layer")
    ap.add_argument("--wu-name", default="")
    a = ap.parse_args(argv)
    manifest = json.loads(Path(a.manifest).read_text())
    lines = [l for l in Path(a.results).read_text().splitlines() if l.strip()]
    s = assimilate_batch(manifest, lines, a.data_root, wu_name=a.wu_name, layer=a.layer)
    print(json.dumps(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
