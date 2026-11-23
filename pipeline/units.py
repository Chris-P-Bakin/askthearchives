"""Page-granular work-unit format for the compute plane.

A unit is one page to transcribe. Validator, assimilator, stager, worker and
the BOINC submitter all consume this dict; keep the keys stable.
"""
import hashlib, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

CAMPAIGN_LAYER = "vision_8blora_v2_1"
# GT-passed only, per spec engine-variants gate; Task 1 is the authority.
# 117f54c0ab18 = same engine/prompt with max_tokens 6144 (2026-11-23 cap);
# 6ecf7d0dbeb2 stays accepted for all previously-produced results.
ACCEPTED_HASHES = {"6ecf7d0dbeb2", "117f54c0ab18"}
MODEL_ID = "qwen3-vl-8b-lora-v2_1"
CONFIG_HASH = "6ecf7d0dbeb2"

PROMPT_SHA = "450d70bc3632026d6612410183562309960babd4fd607a2fc748b7d9b43b2ebe"


def _load_registry():
    # pipeline/registry module (renamed from collections.py, which shadowed the stdlib name) on sys.path; load by path.
    import importlib.util
    mod = sys.modules.get("collection_registry")
    if mod is None:
        spec = importlib.util.spec_from_file_location(
            "collection_registry", Path(__file__).resolve().parent / "collections_registry.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["collection_registry"] = mod
        spec.loader.exec_module(mod)
    return mod


registry = _load_registry()


def unit_collection(unit: dict) -> str:
    """Collection id of a unit; units predating the field are naca."""
    return unit.get("collection", "naca")


def doc_dir(doc_id: str, collection: str = "naca", data_root=None) -> Path:
    """Directory holding a doc's files. The dir name is the full doc_id.

    naca: <data_root or common.DATA>/<doc_id> (exactly as before).
    others: the registry root; an explicit data_root (the naca data root) is
    re-based so the registry's data-relative path is kept under it.
    """
    if collection == "naca":
        base = Path(data_root) if data_root is not None else Path(common.DATA)
        return base / doc_id
    croot = registry.root(collection)
    if data_root is not None:
        croot = Path(data_root) / croot.relative_to(registry.ROOT / "data")
    return croot / doc_id


def unit_doc_dir(unit: dict, data_root=None) -> Path:
    return doc_dir(unit["ntrs_id"], unit_collection(unit), data_root)


def unit_id(ntrs_id: str, n: int) -> str:
    return f"{ntrs_id}-p{n}"


def make_unit(ntrs_id: str, n: int, collection: str = "naca") -> dict:
    """The wire key "ntrs_id" carries the full doc_id for every collection."""
    png = doc_dir(ntrs_id, collection) / "pages" / f"{n}.png"
    if collection == "naca":
        image_key = f"naca/pages/{ntrs_id}/p-{n}.png"
    else:
        prefix = registry.load()[collection]["doc_prefix"]
        bare = ntrs_id[len(prefix):] if prefix and ntrs_id.startswith(prefix) else ntrs_id
        image_key = f"{collection}/pages/{bare}/p-{n}.png"
    unit = {
        "unit_id": unit_id(ntrs_id, n),
        "ntrs_id": ntrs_id,
        "page": n,
        "image_key": image_key,
        "image_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
        "model_id": MODEL_ID,
        "prompt_sha": PROMPT_SHA,
        "config_hash": CONFIG_HASH,
    }
    if collection != "naca":  # naca units stay byte-identical; absent key means naca
        unit["collection"] = collection
    return unit


def read_units(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_units(units, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        for u in units:
            f.write(json.dumps(u, sort_keys=True) + "\n")
    os.replace(tmp, path)
