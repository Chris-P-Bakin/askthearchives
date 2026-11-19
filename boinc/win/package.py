"""Assemble and sign the naca_vision_b windows_x86_64 app version from pinned
official binaries — no compiler, every artifact sha256-pinned in pins.json.

    .venv/bin/python boinc/win/package.py fetch      # download into boinc/vendor/, verify
    .venv/bin/python boinc/win/package.py assemble   # build boinc/win/build/<version dir>
    .venv/bin/python boinc/win/package.py sign       # sign_executable inside the apache container
    .venv/bin/python boinc/win/package.py stage      # copy into project apps/ + update_versions

The bundle: BOINC wrapper.exe (main program) + job.xml + embeddable CPython +
llama.cpp win-cuda server + our launcher/unit_worker/guard/prompt. Physical
names are prefixed for download-space uniqueness; logical names keep the
bundle-relative paths job.xml expects.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
VERSION = "2.02"   # single source: dir name, physical-name prefix, apps path
VERSION_DIR = f"naca_vision_b_{VERSION}_windows_x86_64__cuda_10g"
PHYS_PREFIX = f"naca_vision_b_{VERSION}_"
CONTAINER = "naca-boinc-apache"
PROJECT_DIR = "/home/boincadm/project"
REPO_FILES = {   # bundle path -> repo source
    "launcher.py": "worker/win/launcher.py",
    "job.xml": "worker/win/job.xml",
    "unit_worker.py": "worker/unit_worker.py",
    "guard.py": "pipeline/guard.py",
    "transcribe_v2.txt": "pipeline/prompts/transcribe_v2.txt",
}


class PinMismatch(Exception):
    pass


def load_pins(path) -> dict:
    return json.loads(Path(path).read_text())


def verify(path: Path, sha256: str) -> None:
    got = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if got != sha256:
        raise PinMismatch(f"{path}: sha256 {got} != pinned {sha256}")


def _extract(zip_path: Path, dest: Path, strip_prefix: str) -> None:
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            if m.is_dir():
                continue
            rel = m.filename
            if strip_prefix and rel.startswith(strip_prefix):
                rel = rel[len(strip_prefix):]
            target = dest / rel
            if not target.resolve().is_relative_to(dest.resolve()):
                raise PinMismatch(f"unsafe zip member {m.filename} in {zip_path.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(m))


def assemble(pins: dict, vendor_dir, out_dir) -> Path:
    vendor_dir, out_dir = Path(vendor_dir), Path(out_dir)
    arts = pins["artifacts"]
    for a in arts.values():            # verify everything before extracting anything
        verify(vendor_dir / a["file"], a["sha256"])
    vdir = out_dir / VERSION_DIR
    if vdir.exists():
        shutil.rmtree(vdir)
    vdir.mkdir(parents=True)
    for a in arts.values():
        dest = vdir / a["subdir"] if a.get("subdir") else vdir
        _extract(vendor_dir / a["file"], dest, a.get("strip_prefix", ""))
    for bundle_name, src in REPO_FILES.items():
        shutil.copyfile(ROOT / src, vdir / bundle_name)
    files = sorted(p.relative_to(vdir).as_posix()
                   for p in vdir.rglob("*") if p.is_file())
    main = next(f for f in files if f.startswith("wrapper_") and f.endswith(".exe"))
    entries = []
    for f in files:
        phys = PHYS_PREFIX + f.replace("/", "_")
        e = [f"        <physical_name>{phys}</physical_name>"]
        if f == main:
            e.append("        <main_program/>")
        else:
            # copy_file: a real file lands in the slot at the logical name —
            # without it python/llama-server would open BOINC XML link files
            e.append(f"        <logical_name>{f}</logical_name>")
            e.append("        <copy_file/>")
        entries.append("    <file>\n" + "\n".join(e) + "\n    </file>")
    (vdir / "version.xml").write_text(
        "<version>\n" + "\n".join(entries) + "\n</version>\n")
    return vdir


def fetch(pins: dict, vendor_dir: Path) -> None:
    vendor_dir.mkdir(parents=True, exist_ok=True)
    for name, a in pins["artifacts"].items():
        dest = vendor_dir / a["file"]
        if not dest.exists():
            print(f"fetching {name}: {a['url']}")
            urllib.request.urlretrieve(a["url"], dest)
        verify(dest, a["sha256"])
        print(f"{name}: ok")


def _rename_physical(vdir: Path) -> Path:
    """update_versions expects files at their physical names: emit a staging copy."""
    staged = vdir.parent / (vdir.name + ".staged")
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir()
    shutil.copyfile(vdir / "version.xml", staged / "version.xml")
    for p in vdir.rglob("*"):
        if p.is_file() and p.name != "version.xml":
            rel = p.relative_to(vdir).as_posix()
            shutil.copyfile(p, staged / (PHYS_PREFIX + rel.replace("/", "_")))
    return staged


def sign_in_project(dest: str) -> None:
    """Sign every staged file in place (writes <name>.sig).

    The container's keys/code_sign_private is /dev/null by upstream design; the
    real key lives in boinc/server/secrets-local/ (gitignored). It is copied in
    only for the duration of the signing run."""
    key = ROOT / "boinc/server/secrets-local/code_sign_private"
    if not key.exists():
        raise RuntimeError(f"missing {key}: generate with bin/crypt_prog -genkey 1024")
    subprocess.run(["docker", "cp", str(key), f"{CONTAINER}:/tmp/cs_priv"], check=True)
    try:
        script = (f'cd {dest} && for f in *; do '
                  'case "$f" in version.xml|*.sig) continue;; esac; '
                  f'{PROJECT_DIR}/bin/sign_executable "$f" /tmp/cs_priv '
                  '> "$f.sig" || exit 1; done && ls *.sig | wc -l')
        r = subprocess.run(["docker", "exec", CONTAINER, "bash", "-c", script],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"sign_executable failed: {r.stderr.strip()}")
    finally:
        subprocess.run(["docker", "exec", CONTAINER, "rm", "-f", "/tmp/cs_priv"], check=False)
    print(f"signed {r.stdout.strip()} files")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "assemble", "sign", "stage"])
    ap.add_argument("--pins", default=str(ROOT / "boinc/win/pins.json"))
    ap.add_argument("--vendor", default=str(ROOT / "boinc/vendor"))
    ap.add_argument("--build", default=str(ROOT / "boinc/win/build"))
    a = ap.parse_args(argv)
    pins = load_pins(a.pins)
    build = Path(a.build)
    if a.cmd == "fetch":
        fetch(pins, Path(a.vendor))
    elif a.cmd == "assemble":
        vdir = assemble(pins, a.vendor, build)
        staged = _rename_physical(vdir)
        print(f"assembled {vdir}\nstaged (physical names) {staged}")
    elif a.cmd in ("sign", "stage"):
        staged = build / (VERSION_DIR + ".staged")
        dest = f"{PROJECT_DIR}/apps/naca_vision_b/{VERSION}/windows_x86_64__cuda_10g"
        subprocess.run(["docker", "exec", "-u", "boincadm", CONTAINER,
                        "mkdir", "-p", dest], check=True)
        subprocess.run(["docker", "cp", f"{staged}/.", f"{CONTAINER}:{dest}/"], check=True)
        subprocess.run(["docker", "exec", CONTAINER, "chown", "-R",
                        "boincadm:boincadm", f"{PROJECT_DIR}/apps/naca_vision_b"],
                       check=True)
        sign_in_project(dest)
        if a.cmd == "stage":
            r = subprocess.run(["docker", "exec", "-i", "-u", "boincadm", "-w", PROJECT_DIR,
                                CONTAINER, "bash", "-c", "yes | bin/update_versions"],
                               capture_output=True, text=True)
            print(r.stdout[-2000:], r.stderr[-1000:], sep="\n")
            return r.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
