import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"

# mcp_server.server reads this at import time, so it must go before collection:
# an inherited value would flip legacy direct-call tests onto the chunk path.
os.environ.pop("ARCHIVES_CHUNKS_DB", None)


import pytest


@pytest.fixture(autouse=True)
def _isolate_log(tmp_path_factory, monkeypatch):
    """Keep test events out of the production pass1 log (an ingest may be live)."""
    import common
    monkeypatch.setattr(common, "LOG", tmp_path_factory.mktemp("log") / "pass1.log.jsonl")

