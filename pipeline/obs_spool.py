"""Consumer side of the observability NDJSON spool.

The MCP server appends one JSON object per line to a spool file. ``read_spool``
returns the complete, parseable events appended since the last call, using a
persisted byte offset. It never raises on malformed data: bad lines are skipped,
and a partial trailing line is left for the next call.
"""

import json
import os
from pathlib import Path

DEFAULT_ROTATE_BYTES = 64 * 2**20


def _load_offset(offset_file: Path) -> int:
    try:
        return int(offset_file.read_text().strip())
    except (OSError, ValueError):
        return 0


def _store_offset(offset_file: Path, offset: int) -> None:
    tmp = offset_file.with_name(offset_file.name + ".tmp")
    tmp.write_text(str(offset))
    os.replace(tmp, offset_file)


def read_spool(
    spool: Path,
    offset_file: Path,
    rotate_bytes: int = DEFAULT_ROTATE_BYTES,
) -> list[dict]:
    """Return events appended to ``spool`` since the offset in ``offset_file``.

    Complete lines are parsed as JSON; unparsable lines and non-object values
    are skipped but still consumed. A trailing line without ``\\n`` is not
    consumed. When the spool is fully drained and at least ``rotate_bytes``
    long, it is truncated and the offset is reset to 0.
    """
    try:
        size = spool.stat().st_size
        with spool.open("rb") as f:
            offset = _load_offset(offset_file)
            if offset < 0 or offset > size:
                offset = 0
            f.seek(offset)
            data = f.read()
    except OSError:
        return []

    end = data.rfind(b"\n") + 1  # bytes through the last complete line
    events: list[dict] = []
    for line in data[:end].split(b"\n"):
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            events.append(obj)

    new_offset = offset + end
    try:
        _store_offset(offset_file, new_offset)
        if new_offset == size and size >= rotate_bytes:
            os.truncate(spool, 0)
            _store_offset(offset_file, 0)
    except OSError:
        pass
    return events
