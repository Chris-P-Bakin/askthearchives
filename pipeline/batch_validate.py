"""Validity gate for a batched (naca_vision_b) result — whole-result verdict.

    python pipeline/batch_validate.py <manifest.json> <results.jsonl> <batch.meta.json>

Exit 0 = valid, 1 = invalid with the reason on stdout. Spec
(2026-11-15-windows-native-worker-design): the WHOLE result is invalid on a
wrong/unknown config hash anywhere, a malformed results.jsonl line, or a
manifest page missing with no error line (truncated or hostile client).
Honest per-page error lines — and pages that fail CONTENT checks — keep the
result valid; batch_assimilate escalates those pages individually.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from units import ACCEPTED_HASHES  # noqa: E402


def validate_batch(manifest: list[dict], lines: list[str]) -> tuple[bool, str]:
    units = {u["unit_id"]: u for u in manifest}
    seen = set()
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except ValueError:
            return False, f"malformed_line_{i}"
        if not isinstance(row, dict):
            return False, f"malformed_line_{i}"
        uid = row.get("unit_id")
        if (not isinstance(row, dict) or uid not in units or uid in seen
                or row.get("status") not in ("ok", "error")):
            return False, f"malformed_line_{i}"
        seen.add(uid)
        if row["status"] == "ok":
            meta = row.get("meta")
            if not isinstance(row.get("text"), str) or not isinstance(meta, dict):
                return False, f"malformed_line_{i}"
            h = meta.get("config_hash")
            if h not in ACCEPTED_HASHES or h != units[uid].get("config_hash"):
                return False, f"config_hash_mismatch:{uid}"
        elif not isinstance(row.get("error"), str):
            return False, f"malformed_line_{i}"
    missing = set(units) - seen
    if missing:
        return False, f"missing_pages:{len(missing)}"
    return True, "ok"


def main(argv=None) -> int:
    manifest_p, results_p, _meta_p = (argv or sys.argv[1:])[:3]
    manifest = json.loads(Path(manifest_p).read_text())
    lines = [l for l in Path(results_p).read_text().splitlines() if l.strip()]
    ok, reason = validate_batch(manifest, lines)
    print(reason)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
