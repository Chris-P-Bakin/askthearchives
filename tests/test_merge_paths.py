import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from merge_lora import ROOT, paths_for


def test_paths_for_v1_matches_original_layout():
    adapter, out = paths_for("v1")
    assert adapter == ROOT / "training" / "runs" / "v1" / "adapter_final"
    assert out == ROOT / "models" / "qwen3-vl-8b-lora-v1" / "merged-bf16"


def test_paths_for_learning_curve_run():
    adapter, out = paths_for("lc25")
    assert adapter == ROOT / "training" / "runs" / "lc25" / "adapter_final"
    assert out == ROOT / "models" / "qwen3-vl-8b-lora-lc25" / "merged-bf16"
