"""Per-request serving events, appended as NDJSON to a spool file.

Best-effort: unset ARCHIVES_SPOOL disables recording, and no failure here
may ever reach the caller. No client identification is recorded.
"""
import json
import os
import time

SPOOL = os.environ.get("ARCHIVES_SPOOL")


def _size(result) -> int:
    if result is None:
        return 0
    if isinstance(result, (bytes, bytearray)):
        return len(result)
    try:
        return len(json.dumps(result))
    except Exception:
        return 0


def _count(result) -> int:
    try:
        if isinstance(result, list):
            return len(result)
        if isinstance(result, dict) and "results" in result:
            return len(result["results"])
    except Exception:
        pass
    return 0


def record(tool: str, t0: float, status: str, result) -> None:
    if not SPOOL:
        return
    try:
        ev = {"ts": time.time(), "tool": tool,
              "duration_ms": (time.monotonic() - t0) * 1000.0,
              "bytes": _size(result), "results": _count(result),
              "status": status,
              "degraded": int(isinstance(result, dict)
                              and result.get("degraded") is True)}
        with open(SPOOL, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev) + "\n")
    except Exception:
        pass
