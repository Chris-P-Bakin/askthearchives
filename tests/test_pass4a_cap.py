import io
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from pass4a import cap_image


def _png(w, h):
    buf = io.BytesIO()
    Image.new("L", (w, h), 200).save(buf, format="PNG")
    return buf.getvalue()


def _size(png):
    return Image.open(io.BytesIO(png)).size


def test_cap_zero_is_identity():
    png = _png(2500, 3200)
    assert cap_image(png, 0) is png


def test_within_cap_unchanged():
    png = _png(800, 1000)
    assert cap_image(png, 1024) is png


def test_landscape_downscaled_to_cap():
    out = cap_image(_png(3000, 2000), 1500)
    assert _size(out) == (1500, 1000)


def test_portrait_downscaled_preserving_aspect():
    out = cap_image(_png(2500, 3200), 1024)
    w, h = _size(out)
    assert h == 1024
    assert abs(w - round(2500 * 1024 / 3200)) <= 1
