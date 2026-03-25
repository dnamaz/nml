#!/usr/bin/env python3
"""
NML Stage-1 SFT Training Script
Fine-tunes Qwen2.5-Coder on the NML corpus using QLoRA.

Supports:
  - Intel Arc GPU via IPEX-LLM (XPU backend) — primary target
  - NVIDIA GPU via standard bitsandbytes CUDA
  - CPU fallback (slow, for testing only)

Usage:
    # Intel Arc (primary)
    python train_nml.py

    # Smaller model for quick test
    python train_nml.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --steps 2000

    # NVIDIA GPU
    python train_nml.py --device cuda

    # Custom paths
    python train_nml.py --train domain/output/training/train.jsonl \
                        --valid domain/output/training/valid.jsonl \
                        --output domain/output/model/nml-stage1-adapters

Requirements (Intel Arc):
    pip install ipex-llm[xpu] bitsandbytes-intel transformers peft trl datasets accelerate
    (from https://pytorch-extension.intel.com/release-whl/stable/xpu/us/)

Requirements (NVIDIA):
    pip install bitsandbytes transformers peft trl datasets accelerate
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "transpilers"))

# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_TRAIN   = ROOT / "domain" / "output" / "training" / "train.jsonl"
DEFAULT_VALID   = ROOT / "domain" / "output" / "training" / "valid.jsonl"
DEFAULT_OUTPUT  = ROOT / "domain" / "output" / "model" / "nml-stage1-adapters"
DEFAULT_STEPS   = 30_000
DEFAULT_LR      = 2e-4
DEFAULT_BATCH   = 4
DEFAULT_GRAD_ACC = 4       # effective batch = 4 * 4 = 16
DEFAULT_MAX_SEQ = 1024
DEFAULT_SAVE_STEPS = 2000
DEFAULT_LORA_R  = 64
DEFAULT_LORA_ALPHA = 128
DEFAULT_LORA_DROPOUT = 0.05


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="NML SFT training — Stage 1")
    p.add_argument("--model",    default=DEFAULT_MODEL,   help="HuggingFace model ID")
    p.add_argument("--train",    default=str(DEFAULT_TRAIN),  help="Training JSONL")
    p.add_argument("--valid",    default=str(DEFAULT_VALID),  help="Validation JSONL")
    p.add_argument("--output",   default=str(DEFAULT_OUTPUT), help="Adapter output directory")
    p.add_argument("--steps",    type=int, default=DEFAULT_STEPS)
    p.add_argument("--lr",       type=float, default=DEFAULT_LR)
    p.add_argument("--batch",    type=int, default=DEFAULT_BATCH)
    p.add_argument("--grad-acc", type=int, default=DEFAULT_GRAD_ACC)
    p.add_argument("--max-seq",  type=int, default=DEFAULT_MAX_SEQ)
    p.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS)
    p.add_argument("--lora-r",   type=int, default=DEFAULT_LORA_R)
    p.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    p.add_argument("--device",   default="auto",
                   help="auto | xpu | cuda | cpu")
    p.add_argument("--no-quantize", action="store_true",
                   help="Disable 4-bit quantization (use BF16 full LoRA — requires more VRAM)")
    p.add_argument("--no-token-expansion", action="store_true",
                   help="Skip adding NML tokens to the tokenizer vocab (saves ~2GB VRAM on 7B)")
    p.add_argument("--eval-size", type=int, default=500,
                   help="Max validation records to use for eval (default 500; use 0 for full set)")
    p.add_argument("--no-eval", action="store_true",
                   help="Disable evaluation entirely (fastest; avoids XPU eval slowdown)")
    p.add_argument("--offline", action="store_true",
                   help="Load model from local cache only — no HF Hub network check "
                        "(equivalent to HF_HUB_OFFLINE=1)")
    p.add_argument("--val-grammar", action="store_true",
                   help="Run grammar validity check on validation samples every save-steps")
    p.add_argument("--resume", metavar="CHECKPOINT",
                   help="Resume from a checkpoint directory, e.g. "
                        "domain/output/model/test-run/checkpoint-100  "
                        "(use 'last' to auto-detect the latest checkpoint in --output)")
    return p.parse_args()


# ─── Device detection ─────────────────────────────────────────────────────────

def detect_device(preference: str) -> str:
    if preference != "auto":
        return preference
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def device_info(device: str):
    if device == "xpu":
        try:
            name = torch.xpu.get_device_name(0)
            vram = torch.xpu.get_device_properties(0).total_memory / 1e9
            print(f"  Intel XPU: {name}  ({vram:.1f} GB)")
        except Exception:
            print("  Intel XPU: detected (properties unavailable)")
    elif device == "cuda":
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  NVIDIA CUDA: {name}  ({vram:.1f} GB)")
    else:
        print("  CPU (training will be very slow)")


# ─── NML token registration ───────────────────────────────────────────────────

def add_nml_tokens(tokenizer) -> int:
    """Add NML-specific tokens to Qwen's existing tokenizer vocabulary."""
    try:
        from nml_grammar import (
            _CLASSIC_OPCODES, _SYMBOLIC_TO_CANONICAL, _VERBOSE_TO_CANONICAL,
            _NUMBERED_LIST, _GREEK_LIST, _VERBOSE_REG_LIST,
        )
    except ImportError:
        print("  WARNING: nml_grammar not found — skipping NML token additions")
        return 0

    new_tokens = set()
    new_tokens.update(_CLASSIC_OPCODES)
    new_tokens.update(_SYMBOLIC_TO_CANONICAL.keys())
    new_tokens.update(_VERBOSE_TO_CANONICAL.keys())
    new_tokens.update(_NUMBERED_LIST)
    new_tokens.update(_GREEK_LIST)
    new_tokens.update(_VERBOSE_REG_LIST)
    # Structural tokens
    new_tokens.update(["@", "#", "shape=", "dtype=", "f32", "f64", "i32",
                        "dest=", "left=", "right=", "src="])

    # Only add tokens not already in the vocabulary
    existing = set(tokenizer.get_vocab().keys())
    to_add = sorted(new_tokens - existing)

    added = tokenizer.add_tokens(to_add)
    return added


# ─── Dataset loading ──────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_record(record: dict, tokenizer) -> str:
    """Apply the model's chat template to a messages record."""
    return tokenizer.apply_chat_template(
        record["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


# ─── Grammar validation callback ──────────────────────────────────────────────

class GrammarValidationCallback(TrainerCallback):
    """Periodically sample from validation set and report grammar pass rate."""

    def __init__(self, val_records: list[dict], sample_size: int = 200, save_steps: int = 2000):
        self.val_records = val_records
        self.sample_size = sample_size
        self.save_steps  = save_steps
        try:
            from nml_grammar import validate_grammar
            self.validate = validate_grammar
        except ImportError:
            self.validate = None

    def on_save(self, args, state, control, **kwargs):
        if self.validate is None:
            return
        import random, re
        sample = random.sample(self.val_records, min(self.sample_size, len(self.val_records)))
        passed = 0
        for rec in sample:
            assistant_text = next(
                (m["content"] for m in rec["messages"] if m["role"] == "assistant"), ""
            )
            # Only check records that look like NML code
            if re.search(r"\b(LD|ST|MMUL|RELU|HALT|TNET|LOOP)\b", assistant_text):
                if self.validate(assistant_text).valid:
                    passed += 1
        total = len(sample)
        pct = 100 * passed / total if total else 0
        print(f"\n[Grammar check @ step {state.global_step}] "
              f"{passed}/{total} valid ({pct:.1f}%)\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("\nNML Stage-1 SFT Training")
    print("=" * 60)

    # ── Device setup ────────────────────────────────────────────
    device = detect_device(args.device)
    print(f"Device: {device}")
    device_info(device)

    # IPEX-LLM: patch for Intel Arc
    if device == "xpu":
        try:
            import intel_extension_for_pytorch as ipex  # noqa: F401
            print("  IPEX loaded")
        except ImportError:
            print("  WARNING: intel_extension_for_pytorch not found.")
            print("  Install: pip install ipex-llm[xpu] --extra-index-url "
                  "https://pytorch-extension.intel.com/release-whl/stable/xpu/us/")

    # ── Output directory ─────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # ── Tokenizer ────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    added = 0 if args.no_token_expansion else add_nml_tokens(tokenizer)
    print(f"  Added {added} NML-specific tokens -> vocab size: {len(tokenizer)}")

    # ── Quantization config ──────────────────────────────────────
    # XPU note: bitsandbytes NF4 on XPU requires IPEX-LLM's own quantization
    # path.  Standard bitsandbytes is CUDA-only and causes CONVERSION errors in
    # transformers 5.x when passed to a plain from_pretrained on XPU.
    # We therefore defer bnb_config creation for XPU until after we know
    # whether IPEX-LLM is available.
    _nf4_cfg = lambda: BitsAndBytesConfig(  # noqa: E731
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    bnb_config = None
    if not args.no_quantize:
        if device == "cuda":
            bnb_config = _nf4_cfg()
            print("  Quantization: NF4 QLoRA (bitsandbytes CUDA)")
        elif device == "xpu":
            pass  # decided below after IPEX-LLM probe
        else:
            print("  Quantization: disabled (CPU)")
    else:
        print("  Quantization: disabled (--no-quantize)")

    # ── Base load kwargs (no quantization_config yet) ─────────────
    print(f"\nLoading model: {args.model}")
    offline = args.offline or bool(os.environ.get("HF_HUB_OFFLINE", ""))
    if offline:
        print("  Offline mode — using local cache only (no HF Hub check)")
    base_kwargs = dict(
        trust_remote_code=True,
        dtype=torch.bfloat16,         # transformers 5.x: torch_dtype was renamed back to dtype
        attn_implementation="eager",  # sdpa/flash_attn not stable on XPU fallback
        local_files_only=offline,
    )

    # ── Model loading ─────────────────────────────────────────────
    if device == "xpu":
        # Prefer IPEX-LLM for optimised XPU kernels + its own NF4 path.
        try:
            from ipex_llm.transformers import AutoModelForCausalLM as IpexAutoModel
            ipex_kwargs = dict(base_kwargs, optimize_model=False)
            if not args.no_quantize:
                ipex_kwargs["load_in_low_bit"] = "nf4"
                print("  Quantization: NF4 via IPEX-LLM")
            else:
                print("  Quantization: disabled (--no-quantize)")
            model = IpexAutoModel.from_pretrained(args.model, **ipex_kwargs)
            print("  Loaded via IPEX-LLM (XPU-optimized)")
        except ImportError:
            # Standard transformers fallback — plain BF16, no bitsandbytes.
            # bitsandbytes NF4 is CUDA-only; passing its config here produces
            # CONVERSION entries that transformers 5.x raises as RuntimeError.
            print("  IPEX-LLM not available — loading BF16 without quantization "
                  "(install ipex-llm[xpu] for NF4 and better XPU performance)")
            model = AutoModelForCausalLM.from_pretrained(args.model, **base_kwargs)
    else:
        load_kwargs = dict(base_kwargs)
        if bnb_config:
            load_kwargs["quantization_config"] = bnb_config
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

    # Resize embeddings if NML tokens were added
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))
        print(f"  Resized embedding table: +{added} rows (randomly initialized)")

    # ── Prepare for k-bit training + LoRA ────────────────────────
    if bnb_config:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
    else:
        # BF16 path: enable gradient checkpointing to reduce activation memory
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Qwen2.5 target modules (attention + MLP projections)
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj",        # MLP
    ]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=DEFAULT_LORA_DROPOUT,
        target_modules=target_modules,
        bias="none",
        # Also fine-tune the new NML embedding rows
        modules_to_save=["embed_tokens", "lm_head"] if added > 0 else [],
        # Required when model has tied embeddings (embed_tokens shares weights with lm_head)
        ensure_weight_tying=added > 0,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # HF Trainer moves the model to the detected device automatically.
    # For non-XPU/non-CUDA paths, place explicitly.
    if device == "cpu":
        model = model.to("cpu")

    # ── Dataset ───────────────────────────────────────────────────
    print(f"\nLoading dataset:")
    print(f"  Train: {args.train}")
    train_records = load_jsonl(args.train)
    valid_records = load_jsonl(args.valid)
    if args.eval_size and len(valid_records) > args.eval_size:
        import random
        valid_records = random.sample(valid_records, args.eval_size)
    print(f"  {len(train_records):,} train  |  {len(valid_records):,} valid")

    # Format to text using the model's chat template
    print("  Applying chat template...", end=" ", flush=True)
    train_texts = [format_record(r, tokenizer) for r in train_records]
    valid_texts = [format_record(r, tokenizer) for r in valid_records]
    print("done")

    train_dataset = Dataset.from_dict({"text": train_texts})
    valid_dataset = Dataset.from_dict({"text": valid_texts})

    # ── Training arguments ────────────────────────────────────────
    # use_cpu=True forces CPU; False lets HF auto-detect XPU/CUDA
    use_cpu = (device == "cpu")

    # --no-eval: skip evaluation entirely (avoids the XPU eval slowdown that
    # previously caused the 78-minute eval and system crash).
    # When eval is disabled, load_best_model_at_end must also be False.
    do_eval = not args.no_eval
    eval_strategy  = "steps" if do_eval else "no"
    eval_steps_val = args.save_steps if do_eval else None

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        max_steps=args.steps,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        fp16=False,
        logging_steps=50,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps_val,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=do_eval,
        metric_for_best_model="eval_loss" if do_eval else None,
        greater_is_better=False if do_eval else None,
        report_to="none",          # disable wandb/tensorboard by default
        dataloader_pin_memory=device == "cuda",
        use_cpu=use_cpu,
        warmup_steps=100,          # warmup_ratio deprecated in transformers 5.x
        max_length=args.max_seq,   # trl 0.29+: max_length (was max_seq_length)
        dataset_text_field="text",
        packing=True,              # pack short examples together for efficiency
    )

    # ── Callbacks ─────────────────────────────────────────────────
    callbacks = []
    if args.val_grammar:
        callbacks.append(
            GrammarValidationCallback(valid_records, save_steps=args.save_steps)
        )

    # ── Trainer ───────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset if do_eval else None,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # ── Resume checkpoint resolution ──────────────────────────────
    resume_ckpt = None
    if args.resume:
        if args.resume.lower() == "last":
            # Auto-detect the latest checkpoint in the output directory
            from transformers.trainer_utils import get_last_checkpoint
            last = get_last_checkpoint(str(output_dir))
            if last:
                resume_ckpt = last
                print(f"  Resuming from: {last}")
            else:
                print("  WARNING: --resume last specified but no checkpoint found "
                      f"in {output_dir}. Starting from scratch.")
        else:
            resume_ckpt = args.resume
            print(f"  Resuming from: {resume_ckpt}")

    # ── Train ─────────────────────────────────────────────────────
    print(f"\nTraining config:")
    print(f"  Steps:          {args.steps:,}")
    print(f"  Batch size:     {args.batch} × {args.grad_acc} grad acc = {args.batch * args.grad_acc} effective")
    print(f"  Learning rate:  {args.lr}")
    print(f"  Max seq length: {args.max_seq}")
    print(f"  LoRA rank:      {args.lora_r}  alpha: {args.lora_alpha}")
    print(f"  Save every:     {args.save_steps} steps")
    print(f"  Eval:           {'disabled (--no-eval)' if not do_eval else f'every {args.save_steps} steps  eval-size={args.eval_size}'}")
    print()

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume_ckpt)
    elapsed = time.time() - t0

    # ── Save ──────────────────────────────────────────────────────
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    hours = elapsed / 3600
    print(f"\nTraining complete in {hours:.1f}h")
    print(f"Adapter saved to: {final_dir}")
    print()
    print("Next steps:")
    print("  1. Evaluate:  python evaluate_nml.py --adapter", final_dir)
    print("  2. Merge:     python merge_adapter.py --base", args.model, "--adapter", final_dir)
    print("  3. Run stage 2 fix passes with --steps 2000 --lr 5e-5")


if __name__ == "__main__":
    main()
