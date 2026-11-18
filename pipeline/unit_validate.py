"""Server-side quality gate for worker results (BOINC script_validator entry).

    python pipeline/unit_validate.py <unit.json> <result.txt> <result.meta.json>

Exit 0 = valid, 1 = invalid with the reject reason on stdout.

TRUST MODEL: only the zlib-ratio check is unconditional. The non-zlib junk
checks (pipe_table, runaway) are gated on finish_reason == "length" (running
them on stop-finished pages would false-reject legitimate dense tables), so
they rely on the worker reporting an honest finish_reason. That is the
accepted trust model for the trusted-worker phase (min_quorum=1); semantic
CER quorum closes it when untrusted workers join (spec: Approaches/validator).
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard
from units import ACCEPTED_HASHES


def validate_result(unit: dict, text: str, meta: dict) -> tuple[bool, str]:
    if (not isinstance(meta, dict) or "config_hash" not in meta
            or not isinstance(meta.get("loop_guard"), dict)):
        return False, "meta_malformed"
    h = meta["config_hash"]
    if h not in ACCEPTED_HASHES or h != unit.get("config_hash"):
        return False, "config_hash_mismatch"
    # a guard-flagged page is a valid result: it assimilates as flagged
    if meta["loop_guard"].get("vision_failed") is True:
        return True, ""
    text = text or ""
    if not text.strip():
        return False, "empty_output"
    bad, _, _ = guard.degenerate(text, meta.get("finish_reason"))
    if bad:
        return False, "degenerate_output"
    return True, ""


def main(argv=None) -> int:
    a = (argv if argv is not None else sys.argv)[1:]
    if len(a) != 3:
        print("usage: unit_validate.py <unit.json> <result.txt> <result.meta.json>")
        return 2
    unit = json.loads(Path(a[0]).read_text())
    text = Path(a[1]).read_text()
    try:
        meta = json.loads(Path(a[2]).read_text())
    except (ValueError, OSError):
        meta = None
    ok, reason = validate_result(unit, text, meta)
    if not ok:
        print(reason)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
