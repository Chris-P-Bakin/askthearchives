"""Merge a LoRA adapter onto the bf16 base and save merged weights for GGUF conversion.

Usage: merge_lora.py [RUN_NAME]  (default v1; reads training/runs/RUN_NAME/adapter_final)

Loading the adapter dir with load_in_4bit=False makes Unsloth resolve the
bnb-4bit base name to its full-precision twin; the assertion below refuses
to continue if anything quantized slipped in (spec: merge onto bf16 ONLY).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def paths_for(run_name: str) -> tuple[Path, Path]:
    adapter = ROOT / "training" / "runs" / run_name / "adapter_final"
    out = ROOT / "models" / f"qwen3-vl-8b-lora-{run_name}" / "merged-bf16"
    return adapter, out


def main() -> None:
    run_name = sys.argv[1] if len(sys.argv) > 1 else "v1"
    adapter, out = paths_for(run_name)
    if not adapter.is_dir():
        sys.exit(f"adapter not found: {adapter}")

    import torch
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(str(adapter), load_in_4bit=False)
    dtypes = {p.dtype for p in model.parameters()}
    print("base resolved:", model.config._name_or_path)
    print("param dtypes:", dtypes)
    bad = dtypes - {torch.bfloat16, torch.float16, torch.float32}
    if bad:
        sys.exit(f"REFUSING MERGE: quantized param dtypes present: {bad}")

    model.save_pretrained_merged(str(out), processor)
    print("merged saved:", out)


if __name__ == "__main__":
    main()
