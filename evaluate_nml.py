#!/usr/bin/env python3
"""
NML Adapter Evaluation Script

Two evaluation modes:

  1. Grammar-only (fast, no GPU generation needed):
     Checks grammar validity of assistant responses already in the validation
     set.  Useful for confirming the training data is clean.

  2. Generation (full eval, requires GPU):
     Loads the fine-tuned adapter, generates NML programs from held-out
     prompts, then checks grammar validity and execution correctness via
     the C runtime.

Usage:
    # Grammar check on validation set (no model load)
    python evaluate_nml.py --adapter domain/output/model/test-run/final --grammar-only

    # Full generation eval (default, 100 prompts)
    python evaluate_nml.py --adapter domain/output/model/test-run/final

    # Offline + custom count
    python evaluate_nml.py --adapter domain/output/model/test-run/final --count 200 --offline

    # NVIDIA GPU
    python evaluate_nml.py --adapter domain/output/model/test-run/final --device cuda
"""

import argparse
import json
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "transpilers"))

# ─── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_VALID   = ROOT / "domain" / "output" / "training" / "valid.jsonl"
_NML_EXE        = "nml.exe" if platform.system() == "Windows" else "nml"
DEFAULT_RUNTIME = Path(os.environ.get("NML_RUNTIME", str(ROOT / _NML_EXE)))

GENERATION_PROMPTS = [
    # ── Core arithmetic ───────────────────────────────────────────
    "Write NML to add two vectors element-wise.",
    "Write NML to subtract one tensor from another using MSUB.",
    "Write NML to multiply two matrices.",
    "Write NML to element-wise multiply two tensors using EMUL.",
    "Write NML to element-wise divide two tensors using EDIV.",
    "Write NML to scale a tensor by 3.5.",
    "Write NML to add a scalar 2.0 to every element using SADD.",
    "Write NML to multiply a tensor by scalar 0.5 using SCLR.",
    "Write NML to subtract scalar 1.0 from a tensor using SSUB.",
    "Write NML to compute the dot product of two vectors.",
    "Write NML to transpose a matrix.",
    "Write NML to compute modulo of two values.",
    # ── Activations ───────────────────────────────────────────────
    "Write NML to apply ReLU activation.",
    "Write NML to apply sigmoid activation.",
    "Write NML to apply tanh activation.",
    "Write NML to compute softmax of a vector using SOFT.",
    "Write NML for layer normalization followed by GELU.",
    # ── Control flow ──────────────────────────────────────────────
    "Write NML to sum integers 1 to 10 using a LOOP.",
    "Write NML to compute 5 factorial using a loop.",
    "Write NML that uses a subroutine to double a value.",
    "Write NML that jumps to a label using JUMP.",
    "Write NML that conditionally skips a block when flag is false using JMPF.",
    "Write NML that jumps when a register is non-zero using JMPT.",
    # ── Dense layers ──────────────────────────────────────────────
    "Write NML for a single dense layer with ReLU.",
    "Write NML for a 2-layer MLP with sigmoid output.",
    "Write NML for a feedforward layer with GELU activation.",
    # ── Vision ────────────────────────────────────────────────────
    "Write NML for conv followed by max pooling.",
    "Write NML to upsample a feature map using UPSC.",
    "Write NML to zero-pad a tensor using PADZ.",
    # ── Transformer ───────────────────────────────────────────────
    "Write NML for self-attention on a sequence.",
    "Write NML to embed token IDs using EMBD.",
    # ── Reduction ─────────────────────────────────────────────────
    "Write NML to find the maximum using RDUC.",
    "Write NML to compute the mean of a tensor.",
    "Write NML to clamp values between 0 and 1.",
    "Write NML to select values where a condition holds using WHER.",
    "Write NML to compare and replace values using CMPR.",
    # ── Signal ────────────────────────────────────────────────────
    "Write NML to apply a FIR filter using FILT.",
    "Write NML to compute the FFT of a signal.",
    # ── Training ──────────────────────────────────────────────────
    "Write NML to train a 2-layer network using TNET.",
    "Write NML to train a 2-layer MLP using TNDEEP with 1000 epochs.",
    "Write NML to compute MSE loss using LOSS.",
    "Write NML using BKWD for a backprop pass.",
    "Write NML to compute the backward pass for MMUL using MMUL_BK.",
    "Write NML to compute the backward pass for RELU using RELU_BK.",
    "Write NML to compute the backward pass for SIGM using SIGM_BK.",
    "Write NML to propagate gradients through softmax using SOFTBK.",
    "Write NML to use WUPD to update weights.",
    "Write NML to apply dropout during training using DROP.",
    "Write NML to apply L2 weight decay using WDECAY.",
    "Write NML to apply batch normalization using BN.",
    # ── General purpose ───────────────────────────────────────────
    "Write NML to convert float to int using FTOI.",
    "Write NML to convert int to float using ITOF.",
    "Write NML to compute cosine distance between two embeddings.",
    # ── M2M ───────────────────────────────────────────────────────
    "Write NML to verify a program signature using VRFY.",
    "Write NML to cast a consensus vote using VOTE.",
    # ── Symbolic syntax ───────────────────────────────────────────
    "Write NML in symbolic syntax to multiply two matrices using ×.",
    "Write NML in symbolic syntax to add two tensors using ⊕.",
    "Write NML in symbolic syntax to apply ReLU using ⌐.",
    "Write NML in symbolic syntax to load a tensor with ↓ and store with ↑.",
    "Write NML in symbolic syntax to apply softmax using Σ.",
    # ── Verbose syntax ────────────────────────────────────────────
    "Write NML in verbose syntax to multiply two matrices using MATRIX_MULTIPLY.",
    "Write NML in verbose syntax to load a tensor using LOAD and store using STORE.",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def detect_device(preference: str) -> str:
    if preference != "auto":
        return preference
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def check_grammar(text: str) -> bool:
    try:
        from nml_grammar import validate_grammar
        return validate_grammar(text).valid
    except ImportError:
        # Fall back to a simple HALT presence check if grammar module missing
        return "HALT" in text


FIXTURES = Path(__file__).parent / "tests" / "eval_fixtures.nml.data"


def execute_nml(text: str, runtime: Path) -> bool:
    if not runtime.exists():
        return False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".nml", delete=False, encoding="utf-8"
        ) as f:
            f.write(text)
            tmp = f.name
        cmd = [str(runtime), tmp]
        if FIXTURES.exists():
            cmd.append(str(FIXTURES))
        cmd += ["--max-cycles", "100000"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def _looks_like_nml(text: str) -> bool:
    # Classic opcodes
    if re.search(
        r"^\s*(?:LD|ST|MMUL|MADD|RELU|SIGM|TANH|SOFT|GELU|HALT|TNET|LOOP|LEAF|ALLC)\b",
        text, re.MULTILINE,
    ):
        return True
    # Symbolic opcodes (×, ⊕, ⊖, ⊗, ⌐, σ, Σ, ↓, ↑, ◼)
    if re.search(r"[×⊕⊖⊗⊘⊙⊛⌐στΣ↓↑◼∗·÷∔∸]", text):
        return True
    # Verbose opcodes
    if re.search(
        r"^\s*(?:MATRIX_MULTIPLY|LOAD|STORE|SIGMOID|SOFTMAX|SCALE|SCALAR_ADD|SCALAR_SUB)\b",
        text, re.MULTILINE,
    ):
        return True
    return False


# ─── Mode 1: grammar check on validation set (no generation) ──────────────────

def eval_grammar_only(valid_path: Path, count: int) -> None:
    print(f"\nLoading validation data: {valid_path}")
    records = load_jsonl(valid_path)
    if count and len(records) > count:
        records = random.sample(records, count)
    print(f"  Checking {len(records):,} records...")

    nml_total = nml_pass = 0
    non_nml = 0

    for rec in records:
        asst = next(
            (m["content"] for m in rec.get("messages", []) if m["role"] == "assistant"),
            "",
        )
        if not _looks_like_nml(asst):
            non_nml += 1
            continue
        nml_total += 1
        if check_grammar(asst):
            nml_pass += 1

    pct = 100 * nml_pass / nml_total if nml_total else 0
    print(f"\n{'='*50}")
    print(f"  Grammar-only evaluation")
    print(f"{'='*50}")
    print(f"  NML records checked : {nml_total:,}")
    print(f"  Grammar pass        : {nml_pass:,}  ({pct:.1f}%)")
    print(f"  Non-NML records     : {non_nml:,}  (skipped)")
    print(f"{'='*50}")


# ─── Mode 2: generation + grammar + execution eval ────────────────────────────

def eval_generation(
    adapter_path: Path,
    base_model: str,
    device: str,
    count: int,
    max_tokens: int,
    runtime: Path,
    offline: bool,
) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    # Detect merged model: no adapter_config.json means weights are already baked in
    is_merged = not (adapter_path / "adapter_config.json").exists()

    model_path = str(adapter_path) if is_merged else base_model
    print(f"\nLoading {'merged model' if is_merged else 'tokenizer from adapter'}: {adapter_path}")
    tok_kwargs = dict(trust_remote_code=True, local_files_only=offline)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-pad for generation

    print(f"Loading model: {model_path}")
    model_kwargs = dict(
        trust_remote_code=True,
        dtype=torch.bfloat16,
        local_files_only=offline,
    )
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    if not is_merged:
        # Resize to match the saved adapter's vocab (NML tokens were added)
        if len(tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
            print(f"  Resized embeddings to {len(tokenizer):,} tokens")
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
    else:
        print(f"  Merged model — no adapter to load")

    model.eval()

    # Move to device
    if device == "xpu":
        model = model.to("xpu")
    elif device == "cuda":
        model = model.to("cuda")

    print(f"  Device: {device}")

    # Pick prompts
    prompts = list(GENERATION_PROMPTS)
    random.shuffle(prompts)
    prompts = prompts[:count]

    grammar_pass = exec_pass = errors = 0
    results = []

    print(f"\nGenerating {len(prompts)} programs...\n")
    t0 = time.time()

    for i, prompt in enumerate(prompts, 1):
        chat = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt")
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                gen_cfg = GenerationConfig(
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                out = model.generate(**inputs, generation_config=gen_cfg)
            generated = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
        except Exception as e:
            errors += 1
            results.append({"prompt": prompt, "generated": "", "grammar": False,
                             "exec": False, "error": str(e)})
            continue

        g_ok = check_grammar(generated) if _looks_like_nml(generated) else False
        e_ok = execute_nml(generated, runtime) if g_ok else False

        if g_ok:
            grammar_pass += 1
        if e_ok:
            exec_pass += 1

        status = "EXEC_OK" if e_ok else ("GRAMMAR_OK" if g_ok else "FAIL")
        if i % 10 == 0 or i == len(prompts):
            elapsed = time.time() - t0
            print(f"  [{i:3d}/{len(prompts)}]  grammar={grammar_pass}  "
                  f"exec={exec_pass}  elapsed={elapsed:.0f}s")

        results.append({
            "prompt": prompt,
            "generated": generated,
            "grammar": g_ok,
            "exec": e_ok,
        })

    elapsed = time.time() - t0
    n = len(prompts)
    g_pct = 100 * grammar_pass / n if n else 0
    e_pct = 100 * exec_pass / n if n else 0

    print(f"\n{'='*50}")
    print(f"  Generation evaluation  ({device})")
    print(f"{'='*50}")
    print(f"  Prompts evaluated : {n}")
    print(f"  Grammar pass      : {grammar_pass:3d} / {n}  ({g_pct:.1f}%)")
    print(f"  Execution pass    : {exec_pass:3d} / {n}  ({e_pct:.1f}%)")
    if errors:
        print(f"  Generation errors : {errors}")
    print(f"  Elapsed           : {elapsed:.1f}s")
    print(f"{'='*50}")

    # Save detailed results
    out_path = adapter_path / "eval_results.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nDetailed results saved to: {out_path}")

    # Print a few failures for inspection (grammar failures first, then exec-only)
    gram_fail = [r for r in results if not r.get("grammar") and r.get("generated")]
    exec_fail = [r for r in results if r.get("grammar") and not r.get("exec") and r.get("generated")]
    failures  = gram_fail + exec_fail
    if failures:
        print(f"\nSample failures (first 5):")
        for r in failures[:5]:
            kind = "grammar+exec" if not r.get("grammar") else "exec only"
            print(f"  [{kind}] Prompt : {r['prompt']}")
            snippet = r["generated"][:120].replace("\n", " ")
            print(f"  Output : {snippet}...")
            print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Evaluate NML fine-tuned adapter")
    p.add_argument("--adapter", required=True,
                   help="Path to saved adapter directory (e.g. domain/output/model/test-run/final)")
    p.add_argument("--base-model", default=None,
                   help="Base model ID (auto-read from adapter_config.json if omitted)")
    p.add_argument("--valid", default=str(DEFAULT_VALID),
                   help="Validation JSONL for grammar-only mode")
    p.add_argument("--runtime", default=str(DEFAULT_RUNTIME),
                   help="Path to NML runtime binary")
    p.add_argument("--count", type=int, default=100,
                   help="Number of prompts to evaluate in generation mode, "
                        "or max validation records in grammar-only mode (default 100)")
    p.add_argument("--max-tokens", type=int, default=1024,
                   help="Max new tokens per generation (default 512)")
    p.add_argument("--device", default="auto",
                   help="auto | xpu | cuda | cpu")
    p.add_argument("--grammar-only", action="store_true",
                   help="Skip generation — only grammar-check the validation set")
    p.add_argument("--offline", action="store_true",
                   help="Use local model cache only (no HF Hub check)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        print(f"ERROR: adapter directory not found: {adapter_path}")
        sys.exit(1)

    offline = args.offline or bool(os.environ.get("HF_HUB_OFFLINE", ""))

    print("\nNML Adapter Evaluation")
    print("=" * 50)
    print(f"  Adapter : {adapter_path}")

    if args.grammar_only:
        eval_grammar_only(Path(args.valid), args.count)
        return

    # Read base model from adapter_config.json if not specified
    base_model = args.base_model
    is_merged = not (adapter_path / "adapter_config.json").exists()
    if not base_model:
        cfg_path = adapter_path / "adapter_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                base_model = json.load(f).get("base_model_name_or_path")
        if not base_model:
            if is_merged:
                base_model = str(adapter_path)  # merged model is its own base
            else:
                print("ERROR: --base-model required (or provide adapter_config.json)")
                sys.exit(1)
    print(f"  Base    : {base_model}")

    device = detect_device(args.device)
    print(f"  Device  : {device}")
    print(f"  Runtime : {args.runtime}")

    eval_generation(
        adapter_path=adapter_path,
        base_model=base_model,
        device=device,
        count=args.count,
        max_tokens=args.max_tokens,
        runtime=Path(args.runtime),
        offline=offline,
    )


if __name__ == "__main__":
    main()
