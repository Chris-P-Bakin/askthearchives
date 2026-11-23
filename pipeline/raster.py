"""PDF -> deskewed 300 dpi page PNGs for collections whose docs are not NTRS reports.

pass1.rasterize is welded to report_dir()/source.pdf, so this takes explicit paths.
The deskew/normalize and page-count logic are pass1's, imported not copied.
"""
import os, shutil, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pass1


def raster_pdf(pdf: Path, out_dir: Path) -> int:
    """Write out_dir/{n}.png (deskewed, 1-based) for every page; return page count."""
    pdf, out_dir = Path(pdf), Path(out_dir)
    n = pass1.pdf_pages(pdf)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir.parent / "_raster"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        timeout = int(os.environ.get("NACA_RASTER_TIMEOUT", "7200"))
        subprocess.run(["pdftoppm", "-r", "300", "-png", str(pdf), str(tmp / "p")],
                       check=True, timeout=timeout)
        for f in sorted(tmp.glob("p-*.png")):
            page = int(f.stem.split("-")[-1])
            pass1.deskew_and_normalize(f, out_dir / f"{page}.png")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return n


def raster_pages(pdf: Path, out_dir: Path, pages) -> int:
    """Raster only the given 1-based pages; returns how many were written."""
    pdf, out_dir = Path(pdf), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir.parent / "_raster"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    timeout = int(os.environ.get("NACA_RASTER_TIMEOUT", "7200"))
    done = 0
    try:
        for n in sorted(set(int(p) for p in pages)):
            subprocess.run(["pdftoppm", "-r", "300", "-png",
                            "-f", str(n), "-l", str(n), str(pdf), str(tmp / "p")],
                           check=True, timeout=timeout)
            outs = sorted(tmp.glob("p-*.png"))
            if not outs:
                continue
            pass1.deskew_and_normalize(outs[-1], out_dir / f"{n}.png")
            for f in outs:
                f.unlink(missing_ok=True)
            done += 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return done
