import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# pipeline/collections.py shadows the stdlib `collections` name once pipeline/
# is on sys.path, so load it by file path under a private module name.
_spec = importlib.util.spec_from_file_location(
    "collection_registry", ROOT / "pipeline" / "collections_registry.py")
registry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(registry)


def test_load_has_naca_and_nara():
    reg = registry.load()
    assert set(reg) == {"naca", "nara-rg255"}
    naca = reg["naca"]
    assert naca["citation"] == "hosted-original"
    assert naca["doc_prefix"] == ""
    assert naca["root"] == "data"
    assert naca["label"] == "NTRS"
    assert naca["record_url"] == "https://ntrs.nasa.gov/citations/{id}"
    nara = reg["nara-rg255"]
    assert nara["citation"] == "link-only"
    assert nara["doc_prefix"] == "nara255-"
    assert nara["root"] == "data/collections/nara-rg255"
    assert nara["label"] == "NARA"
    assert nara["record_url"] == "https://catalog.archives.gov/id/{id}"
    for entry in reg.values():
        assert {"id", "title", "source", "record_url", "citation",
                "doc_prefix", "root", "status", "label"} <= set(entry)
        assert set(entry) - {"doc_label_prefix"} == {
            "id", "title", "source", "record_url", "citation",
            "doc_prefix", "root", "status", "label"}
        assert entry["id"] in reg


def test_doc_label_prefix_optional_on_naca_only():
    reg = registry.load()
    assert reg["naca"]["doc_label_prefix"] == "NACA"
    assert "doc_label_prefix" not in reg["nara-rg255"]


def test_for_doc_prefix_routing():
    assert registry.for_doc("19930091026")["id"] == "naca"
    assert registry.for_doc("nara255-486937833")["id"] == "nara-rg255"


def test_for_doc_unknown_prefix_is_naca_only_if_bare():
    # The empty prefix is the fallback: an id matching no other prefix routes to naca.
    assert registry.for_doc("zz9-1")["id"] == "naca"


def test_record_url_strips_prefix():
    nara = registry.load()["nara-rg255"]
    assert registry.record_url(nara, "nara255-486937833") == \
        "https://catalog.archives.gov/id/486937833"
    naca = registry.load()["naca"]
    assert registry.record_url(naca, "19930091026") == \
        "https://ntrs.nasa.gov/citations/19930091026"


def test_roots_absolute():
    naca_root = registry.root("naca")
    nara_root = registry.root("nara-rg255")
    assert naca_root.is_absolute() and nara_root.is_absolute()
    assert naca_root == ROOT / "data"
    assert nara_root == ROOT / "data" / "collections" / "nara-rg255"
    try:
        registry.root("no-such-collection")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown collection id must raise KeyError")
