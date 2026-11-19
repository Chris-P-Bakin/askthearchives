"""QLoRA distillation of Qwen3-VL-8B on the 27B pseudo-label manifest.

Spec: docs/superpowers/specs/2026-11-10-8b-distillation-design.md (section 5).
Run via training/.venv (see setup.sh). `--smoke N` trains N steps on the N longest
train examples to validate memory headroom before the real run.
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
MANIFEST = DATA_ROOT / "distill_v1_manifest.json"
PROMPT_PATH = ROOT / "pipeline" / "prompts" / "transcribe_v2.txt"
RUN_DIR = ROOT / "training" / "runs" / "v1"
LOSS_LOG = ROOT / "training" / "logs" / "v1_loss.jsonl"

BASE_MODEL = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
MAX_SEQ = 16384
SEED = 13


def mark_epoch_save(control):
    """Epoch-end checkpoint: spec's overfit guard needs the best-val epoch on disk."""
    control.should_save = True
    return control


def format_example(ex: dict, prompt: str, data_root: Path) -> dict:
    label = (Path(data_root) / ex["ntrs_id"] / ex.get("label_dir", "vision_27bq4")
             / f"{ex['page']}.txt").read_text()
    want = ex.get("label_sha256")
    if want:
        got = hashlib.sha256(label.encode()).hexdigest()
        if got != want:
            raise ValueError(
                f"label_sha256 mismatch for {ex['ntrs_id']} p{ex['page']}: "
                f"manifest {want[:12]} vs on-disk {got[:12]} — "
                "the teacher layer changed since the manifest was built")
    image = str(Path(ex["image"]).resolve())
    return {"messages": [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": label}]},
    ]}


def load_split(manifest: dict, prompt: str, split: str, smoke_n: int = 0,
               frac: float = 1.0, data_root: Path | None = None) -> list[dict]:
    exs = [e for e in manifest["examples"] if e["split"] == split]
    if frac < 1.0:
        # Seeded subsample for the learning-curve experiment (spec section 6).
        rnd = random.Random(SEED)
        exs = sorted(rnd.sample(exs, round(len(exs) * frac)),
                     key=lambda e: (e["ntrs_id"], e["page"]))
    if smoke_n:
        exs = sorted(exs, key=lambda e: -e["est_tokens"])[:smoke_n]
    return [format_example(e, prompt, data_root or DATA_ROOT) for e in exs]


class LossLogger:
    """Appends {'step', 'loss' | 'eval_loss'} lines; file is the dashboard's view."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("a")

    def on_log(self, args, state, control, logs=None, **kw):
        if logs:
            row = {"step": state.global_step}
            for k in ("loss", "eval_loss", "learning_rate", "epoch"):
                if k in logs:
                    row[k] = logs[k]
            self.f.write(json.dumps(row) + "\n")
            self.f.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", type=int, default=0,
                    help="train N steps on the N longest examples, then exit")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--frac", type=float, default=1.0,
                    help="seeded fraction of the train split (learning curve)")
    ap.add_argument("--manifest", default=str(MANIFEST),
                    help="training manifest JSON (default: v1 manifest)")
    ap.add_argument("--run-name", default="v1",
                    help="subdir under training/runs and loss-log prefix")
    args = ap.parse_args()

    global RUN_DIR, LOSS_LOG
    RUN_DIR = ROOT / "training" / "runs" / args.run_name
    LOSS_LOG = ROOT / "training" / "logs" / f"{args.run_name}_loss.jsonl"

    manifest = json.loads(Path(args.manifest).read_text())
    prompt = PROMPT_PATH.read_text().strip()
    import hashlib
    if hashlib.sha256(prompt.encode()).hexdigest() != manifest["prompt_sha256"]:
        sys.exit("prompt file no longer matches manifest prompt_sha256 — rebuild manifest")

    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    model, processor = FastVisionModel.from_pretrained(
        BASE_MODEL, load_in_4bit=True, max_seq_length=MAX_SEQ,
        use_gradient_checkpointing="unsloth")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16, lora_alpha=32, lora_dropout=0, bias="none", random_state=SEED)

    train_ds = load_split(manifest, prompt, "train", smoke_n=args.smoke,
                          frac=args.frac)
    val_ds = load_split(manifest, prompt, "val")
    print(f"train={len(train_ds)} val={len(val_ds)} smoke={args.smoke}")

    class CB(TrainerCallback):
        logger = LossLogger(LOSS_LOG)

        def on_log(self, *a, **kw):
            self.logger.on_log(*a, **kw)

    class EpochEndSave(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kw):
            return mark_epoch_save(control)

    FastVisionModel.for_training(model)
    cfg = SFTConfig(
        output_dir=str(RUN_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        max_steps=args.smoke if args.smoke else -1,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=1,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="no" if args.smoke else "epoch",
        optim="adamw_8bit",
        bf16=True,
        seed=SEED,
        report_to="none",
        max_length=MAX_SEQ,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataloader_num_workers=2,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=processor.tokenizer,
        data_collator=UnslothVisionDataCollator(model, processor, resize="max"),
        train_dataset=train_ds,
        eval_dataset=None if args.smoke else val_ds,
        args=cfg,
        callbacks=[CB(), EpochEndSave()],
    )
    trainer.train(resume_from_checkpoint=True if args.resume else None)

    if not args.smoke:
        out = RUN_DIR / "adapter_final"
        model.save_pretrained(str(out))
        processor.save_pretrained(str(out))
        print("adapter saved:", out)


if __name__ == "__main__":
    main()
