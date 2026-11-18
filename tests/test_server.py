import asyncio
import pytest
from conftest import DATA

from build_index import build
from mcp_server import server as srv
from mcp_server.db import Corpus, NotFoundError


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    db = tmp_path_factory.mktemp("idx") / "corpus.db"
    build(db, DATA, ids=["19930092929"])
    return Corpus(db, DATA)


def test_tools_registered():
    names = set(asyncio.run(srv.app.list_tools().__anext__()) if False else
                [t.name for t in asyncio.run(srv.app.list_tools())])
    assert {"search", "get_page", "get_report", "get_figures",
            "get_page_image", "cite"} <= names


def test_startup_rejects_bad_index(tmp_path):
    with pytest.raises(SystemExit):
        srv.validate_index(tmp_path / "missing.db")


def test_startup_rejects_wrong_schema(tmp_path):
    import sqlite3
    db = tmp_path / "bad.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE index_meta(key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO index_meta VALUES ('schema_version','999'),"
                 "('embed_model','BAAI/bge-small-en-v1.5')")
    conn.commit(); conn.close()
    with pytest.raises(SystemExit):
        srv.validate_index(db)


def test_page_image_bbox_clamped(corpus):
    full = corpus.page_image("19930092929", 43, bbox=[-100, -100, 999999, 999999])
    assert full[:8] == b"\x89PNG\r\n\x1a\n"


def test_page_image_zero_area_bbox_errors(corpus):
    with pytest.raises(NotFoundError):
        corpus.page_image("19930092929", 43, bbox=[10, 10, 10, 10])


def test_page_image_max_px_clamp(corpus):
    from PIL import Image
    import io
    big = corpus.page_image("19930092929", 43, max_px=99999)
    assert max(Image.open(io.BytesIO(big)).size) <= 2400
    dflt = corpus.page_image("19930092929", 43)
    assert max(Image.open(io.BytesIO(dflt)).size) <= 1200


def test_instructions_mention_citation_and_illegible():
    assert "<?>" in srv.INSTRUCTIONS
    assert "cite" in srv.INSTRUCTIONS.lower()


def test_startup_rejects_no_embed_index(tmp_path):
    from build_index import build
    db = tmp_path / "corpus.db"
    build(db, DATA, ids=["19930093282"], embed=False)
    with pytest.raises(SystemExit):
        srv.validate_index(db)


def test_run_translates_operational_error_to_tool_error():
    import sqlite3
    from mcp.server.mcpserver.exceptions import ToolError
    def boom():
        raise sqlite3.OperationalError("unable to open database file")
    with pytest.raises(ToolError, match="rebuild"):
        srv._run("search", boom)
