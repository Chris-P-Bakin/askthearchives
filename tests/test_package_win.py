import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "boinc" / "win"))
import package  # noqa: E402


def sha(b):
    return hashlib.sha256(b).hexdigest()


def make_vendor(tmp_path):
    """Tiny stand-ins for the four vendor zips + pins file with real hashes."""
    vendor = tmp_path / "vendor"
    vendor.mkdir()

    def zp(name, members):
        p = vendor / name
        with zipfile.ZipFile(p, "w") as z:
            for m, data in members.items():
                z.writestr(m, data)
        return p

    llama = zp("llama.zip", {"llama-server.exe": "exe", "ggml-cuda.dll": "dll"})
    cudart = zp("cudart.zip", {"cudart64_12.dll": "dll"})
    wrapper = zp("wrapper.zip", {"wrapper_26204_windows_x86_64.exe": "exe"})
    python = zp("python.zip", {"python.exe": "exe", "python312.zip": "stdlib",
                               "python312.dll": "dll"})
    pins = {"artifacts": {
        "llama_win_cuda": {"file": "llama.zip", "sha256": sha(llama.read_bytes()),
                           "strip_prefix": ""},
        "cudart_win": {"file": "cudart.zip", "sha256": sha(cudart.read_bytes()),
                       "strip_prefix": ""},
        "wrapper_win": {"file": "wrapper.zip", "sha256": sha(wrapper.read_bytes()),
                        "strip_prefix": ""},
        "python_embed": {"file": "python.zip", "sha256": sha(python.read_bytes()),
                         "strip_prefix": "", "subdir": "python"},
    }, "linux_engine": {"build": "test"}}
    pins_p = tmp_path / "pins.json"
    pins_p.write_text(json.dumps(pins))
    return vendor, pins_p


def test_verify_rejects_hash_mismatch(tmp_path):
    vendor, pins_p = make_vendor(tmp_path)
    pins = package.load_pins(pins_p)
    pins["artifacts"]["llama_win_cuda"]["sha256"] = "0" * 64
    out = tmp_path / "out"
    with pytest.raises(package.PinMismatch):
        package.assemble(pins, vendor, out)
    assert not out.exists() or not any(out.iterdir())


def test_assemble_builds_version_xml(tmp_path):
    vendor, pins_p = make_vendor(tmp_path)
    out = tmp_path / "out"
    vdir = package.assemble(package.load_pins(pins_p), vendor, out)
    assert vdir.name == f"naca_vision_b_{package.VERSION}_windows_x86_64__cuda_10g"
    names = {p.relative_to(vdir).as_posix() for p in vdir.rglob("*") if p.is_file()}
    for want in ("llama-server.exe", "ggml-cuda.dll", "cudart64_12.dll",
                 "wrapper_26204_windows_x86_64.exe", "python/python.exe",
                 "launcher.py", "unit_worker.py", "guard.py", "transcribe_v2.txt",
                 "job.xml", "version.xml"):
        assert want in names, f"missing {want}"
    xml = (vdir / "version.xml").read_text()
    assert "<main_program/>" in xml
    # C2: every non-main file must be copied into the slot under its logical
    # name, or python/llama-server find XML link files instead of their DLLs
    assert xml.count("<copy_file/>") == xml.count("<file>") - 1
    assert package.VERSION in package.PHYS_PREFIX and package.VERSION in package.VERSION_DIR
    assert xml.count("<file>") == len(names) - 1  # every file listed except version.xml
    assert "wrapper_26204_windows_x86_64.exe" in xml.split("<main_program/>")[0].rsplit(
        "<physical_name>", 1)[-1]
    # job.xml's logical application path maps to the bundled python
    assert "<logical_name>python/python.exe</logical_name>" in xml


def test_assemble_is_deterministic(tmp_path):
    vendor, pins_p = make_vendor(tmp_path)
    a = package.assemble(package.load_pins(pins_p), vendor, tmp_path / "a")
    b = package.assemble(package.load_pins(pins_p), vendor, tmp_path / "b")
    assert (a / "version.xml").read_text() == (b / "version.xml").read_text()
