#!/usr/bin/env python3
"""
NML Adapter Merge Script

Merges a LoRA adapter into the base model weights, producing a single
standalone model that can be used for inference without PEFT loaded.

The merged model:
  - Has no adapter overhead at inference time
  - Can be loaded with plain AutoModelForCausalLM (no peft import needed)
  - Is required for stage-2 fine-tuning (use merged as the new base)
  - Includes the resized tokenizer (NML tokens preserved)

Usage:
    # Auto-detect base model from adapter_config.json
    python merge_adapter.py --adapter domain/output/model/test-run/final

    # Explicit base model + output path
    python merge_adapter.py \
        --base Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --adapter domain/output/model/test-run/final \
        --output domain/output/model/test-run/merged

    # Offline (model already cached)
    python merge_adapter.py --adapter domain/output/model/test-run/final --offline

    # Save in float16 to halve disk size (slight precision loss vs bfloat16)
    python merge_adapter.py --adapter domain/output/model/test-run/final --dtype float16
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "transpilers"))


# ─── Helpers ──────────────────────────────────────────────────────────────────

def detect_device() -> str:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def dtype_from_str(s: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[s]


def read_base_model(adapter_path: Path) -> str:
    cfg = adapter_path / "adapter_config.json"
    if not cfg.exists():
        return None
    with open(cfg) as f:
        return json.load(f).get("base_model_name_or_path")


def model_size_gb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / 1e9


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Merge NML LoRA adapter into base model")
    p.add_argument("--adapter", required=True,
                   help="Path to adapter directory (e.g. domain/output/model/test-run/final)")
    p.add_argument("--base-model", default=None,
                   help="Base model ID or path (auto-read from adapter_config.json if omitted)")
    p.add_argument("--output", default=None,
                   help="Output directory for merged model "
                        "(default: <adapter-dir>/../merged)")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="Save dtype (default: bfloat16)")
    p.add_argument("--offline", action="store_true",
                   help="Use local model cache only (no HF Hub check)")
    p.add_argument("--safe-serialization", action="store_true", default=True,
                   help="Save as .safetensors (default: True)")
    p.add_argument("--no-safe-serialization", dest="safe_serialization",
                   action="store_false",
                   help="Save as .bin instead of .safetensors")
    p.add_argument("--skip-verify", action="store_true",
                   help="Skip the post-merge forward-pass sanity check")
    args = p.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        print(f"ERROR: adapter not found: {adapter_path}")
        sys.exit(1)

    # ── Resolve base model ─────────────────────────────────────────
    base_model = args.base_model or read_base_model(adapter_path)
    if not base_model:
        print("ERROR: --base-model required (or adapter_config.json must contain "
              "base_model_name_or_path)")
        sys.exit(1)

    # ── Resolve output path ────────────────────────────────────────
    output_path = Path(args.output) if args.output else adapter_path.parent / "merged"

    offline = args.offline or bool(os.environ.get("HF_HUB_OFFLINE", ""))
    save_dtype = dtype_from_str(args.dtype)
    device = detect_device()

    print("\nNML Adapter Merge")
    print("=" * 55)
    print(f"  Base model  : {base_model}")
    print(f"  Adapter     : {adapter_path}")
    print(f"  Output      : {output_path}")
    print(f"  Save dtype  : {args.dtype}")
    print(f"  Device      : {device}")
    print("=" * 55)

    t0 = time.time()

    # ── Load tokenizer ────────────────────────────────────────────
    print("\n[1/5] Loading tokenizer from adapter...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path),
        trust_remote_code=True,
        local_files_only=offline,
    )
    print(f"  Vocab size: {len(tokenizer):,} tokens")

    # ── Load base model in BF16 on CPU ────────────────────────────
    # Always merge on CPU to avoid VRAM OOM — merging is memory-intensive
    # because both the base weights and delta weights must be held at once.
    print(f"\n[2/5] Loading base model: {base_model}  (CPU, BF16)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=offline,
        device_map="cpu",           # keep on CPU throughout merge
    )

    # Resize embeddings to match adapter's tokenizer (NML tokens)
    base_vocab = model.config.vocab_size
    if len(tokenizer) != base_vocab:
        model.resize_token_embeddings(len(tokenizer))
        print(f"  Resized embeddings: {base_vocab:,} → {len(tokenizer):,} tokens")

    # ── Load LoRA adapter ─────────────────────────────────────────
    print(f"\n[3/5] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        model,
        str(adapter_path),
        device_map="cpu",
    )

    # ── Merge and unload ──────────────────────────────────────────
    print("\n[4/5] Merging adapter weights into base model...")
    model = model.merge_and_unload()

    # Cast to requested save dtype
    if save_dtype != torch.bfloat16:
        print(f"  Casting to {args.dtype}...")
        model = model.to(save_dtype)

    print(f"  Merge complete  ({time.time() - t0:.1f}s so far)")

    # ── Optional sanity check ─────────────────────────────────────
    if not args.skip_verify:
        print("\n  Sanity check: running a forward pass on dummy input...")
        try:
            dummy = tokenizer("MMUL R0 R1 R2\nHALT", return_tensors="pt")
            with torch.no_grad():
                out = model(**dummy)
            logits_shape = tuple(out.logits.shape)
            print(f"  Forward pass OK  — logits shape: {logits_shape}")
        except Exception as e:
            print(f"  WARNING: forward pass failed ({e}) — merge may be incomplete")

    # ── Save ─────────────────────────────────────────────────────
    print(f"\n[5/5] Saving merged model to: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    # embed_tokens and lm_head are now separate (modules_to_save untied them)
    model.config.tie_word_embeddings = False

    model.save_pretrained(
        str(output_path),
        safe_serialization=args.safe_serialization,
    )
    tokenizer.save_pretrained(str(output_path))

    elapsed = time.time() - t0
    size_gb = model_size_gb(output_path)

    print(f"\n{'='*55}")
    print(f"  Merge complete in {elapsed:.1f}s")
    print(f"  Saved to  : {output_path}")
    print(f"  Disk size : {size_gb:.2f} GB")
    print(f"{'='*55}")
    print(f"\nNext steps:")
    print(f"  Evaluate  : python evaluate_nml.py --adapter {output_path} --grammar-only")
    print(f"  Inference : python -c \"")
    print(f"    from transformers import pipeline")
    print(f"    pipe = pipeline('text-generation', model='{output_path}', device_map='auto')")
    print(f"    print(pipe([{{\\\"role\\\": \\\"user\\\", \\\"content\\\": \\\"Write NML to add two vectors.\\\"}}], max_new_tokens=256)[0])\"")
    print(f"  Stage 2   : python train_nml.py --model {output_path} --steps 2000 --lr 5e-5")
    print()


if __name__ == "__main__":
    main()
