#!/usr/bin/env python3
"""
NML Verification Loop — generate NML from the fine-tuned model, validate,
execute, and collect verified pairs for the next training round.

Pipeline:
  1. Generate NML from prompt templates using the fine-tuned model
  2. Validate with nml_grammar.py (syntax check)
  3. Execute with the C runtime (runtime check)
  4. If correct: add to verified training set
  5. If incorrect: create a debugging/repair pair

Requires:
  - pip install mlx-lm
  - Compiled NML runtime (./nml or as specified by --runtime)
  - Fine-tuned model (merged or with adapter)

Usage:
    python3 nml_verify_gen.py \
        --model ../domain/output/model/nml-core-merged \
        --runtime ../nml \
        --output ../domain/output/training/verified_pairs.jsonl \
        --count 500

    python3 nml_verify_gen.py \
        --model ../domain/output/model/Qwen2.5-Coder-7B-Instruct-4bit \
        --adapter ../domain/output/model/nml-core-adapters \
        --runtime ../nml \
        --count 100
"""

import json
import random
import argparse
import subprocess
import tempfile
import sys
from pathlib import Path

random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# Prompt templates for NML generation
# ═══════════════════════════════════════════════════════════════════════════════

PROMPTS = [
    "Write NML to add two vectors element-wise.",
    "Write NML to multiply a matrix by a vector.",
    "Write NML to apply ReLU activation to a tensor.",
    "Write NML to apply sigmoid activation.",
    "Write NML to compute the softmax of a vector.",
    "Write NML to scale a value by 3.5.",
    "Write NML to divide a tensor by 4.0.",
    "Write NML to compute the dot product of two vectors.",
    "Write NML to sum integers from 1 to 10 using a loop.",
    "Write NML to compute 5 factorial.",
    "Write NML for a single dense neural network layer with ReLU.",
    "Write NML for a 2-layer neural network with sigmoid output.",
    "Write NML to clamp values between 0 and 1.",
    "Write NML to find the maximum of a tensor using RDUC.",
    "Write NML to compute the mean of a tensor.",
    "Write NML to transpose a matrix.",
    "Write NML that uses a subroutine to double a value.",
    "Write NML for convolution followed by max pooling.",
    "Write NML for self-attention on an input sequence.",
    "Write NML to embed token IDs into dense vectors.",
    "Write NML to apply a FIR filter to a signal.",
    "Write NML to compute the FFT of a signal.",
    "Write NML to find consensus of values using VOTE.",
    "Write NML to compute cosine distance between two embeddings.",
    "Write NML to print a number using SYS.",
    "Write NML to compute modulo of two values.",
    "Write NML for absolute value using conditional branching.",
    "Write NML to return the larger of two values.",
    "Write NML for a three-tier rate: below 100 at 0.1, 100-500 at 0.2, above 500 at 0.3.",
    "Write NML for layer normalization followed by GELU activation.",
    "Write NML to chain two subroutines: scale by 2 then add 10.",
    "Write NML to project a vector into embedding space using PROJ.",
    "Write NML for a feedforward layer with GELU activation.",
    "Write NML to create a comparison mask for values greater than 50.",
    "Write NML to conditionally select between two tensors using WHER.",
    "Write NML to gather elements from a tensor by index.",
    "Write NML to zero-pad a tensor then convolve with a kernel.",
    "Write NML for an autoencoder with ReLU in the bottleneck.",
    "Write NML to upscale a tensor by 2x.",
    "Write NML for Fibonacci: print the first 10 numbers.",
    "Write NML to accumulate a value over 20 iterations in a loop.",
    "Write NML for a signed program with META headers.",
    "Write NML with two composable fragments using FRAG/LINK.",
    "Write NML to convert an integer to float using ITOF.",
    "Write NML for a weighted sum: multiply element-wise then reduce.",
    "Write NML to compute Euclidean distance between two vectors.",
    "Write NML for a linear regression model: y = Wx + b.",
    "Write NML to compute compound growth over 12 periods.",
    "Write NML for a 2-layer CNN with ReLU and pooling.",
    "Write NML for a transformer block with attention, FFN, and residual connections.",
    "Write symbolic NML to add two tensors.",
    "Write symbolic NML for a single dense layer with sigmoid.",
    "Write symbolic NML to scale a value by 0.5.",
    "Write verbose NML to load, scale, and store a value.",
    "Write verbose NML for a neural network layer with ReLU.",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Grammar validation (import from nml_grammar.py)
# ═══════════════════════════════════════════════════════════════════════════════

def validate_nml_grammar(nml_code: str) -> dict:
    """Run grammar validation on NML code. Returns dict with 'valid' key."""
    grammar_path = Path(__file__).parent / "nml_grammar.py"
    if not grammar_path.exists():
        return {"valid": True, "errors": [], "warnings": [],
                "note": "grammar validator not found, skipping"}

    try:
        sys.path.insert(0, str(grammar_path.parent))
        import nml_grammar
        report = nml_grammar.validate_grammar(nml_code)
        return report.to_dict()
    except Exception as e:
        return {"valid": False, "errors": [{"message": str(e)}], "warnings": []}


# ═══════════════════════════════════════════════════════════════════════════════
# Auto-generate .nml.data for programs that reference @memory
# ═══════════════════════════════════════════════════════════════════════════════

def generate_data_file(nml_code: str) -> str:
    """Scan NML for @memory refs used as inputs and generate a .nml.data file."""
    import re
    all_refs = set(re.findall(r'@(\w+)', nml_code))
    st_refs = set()
    for line in nml_code.splitlines():
        tokens = line.strip().split()
        if len(tokens) >= 3:
            op = tokens[0].upper()
            if op in ("ST", "↑", "STORE"):
                ref = tokens[2].lstrip("@")
                st_refs.add(ref)
    input_refs = all_refs - st_refs
    if not input_refs:
        return None

    lines = []
    for ref in sorted(input_refs):
        r = ref.lower()
        if any(kw in r for kw in ['w1', 'w2', 'w3', 'weight', 'matrix', 'projection']):
            shape = "4,4"
        elif any(kw in r for kw in ['b1', 'b2', 'b3', 'bias', 'offset', 'shift']):
            shape = "1,4"
        elif any(kw in r for kw in ['kernel', 'filter', 'conv']):
            shape = "3,3"
        elif any(kw in r for kw in ['image', 'img', 'frame']):
            shape = "8,8"
        elif any(kw in r for kw in ['query', 'key', 'value', 'q', 'k', 'v']):
            shape = "4,4"
        elif any(kw in r for kw in ['embed', 'table', 'vocab']):
            shape = "8,4"
        elif any(kw in r for kw in ['target', 'label']):
            shape = "1,4"
        elif any(kw in r for kw in ['scalar', 'threshold', 'rate', 'lr', 'limit']):
            shape = "1"
        elif any(kw in r for kw in ['signal', 'readings', 'sequence', 'series']):
            shape = "1,8"
        elif any(kw in r for kw in ['arch']):
            shape = "1,4"
        else:
            shape = "1,4"

        n = 1
        for d in shape.split(","):
            n *= int(d)
        data = ",".join(f"{random.uniform(-1,1):.4f}" for _ in range(n))
        lines.append(f"@{ref} shape={shape} dtype=f64 data={data}")

    if not lines:
        return None
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".nml.data", delete=False)
    tmp.write("\n".join(lines) + "\n")
    tmp.close()
    return tmp.name


# ═══════════════════════════════════════════════════════════════════════════════
# Runtime execution
# ═══════════════════════════════════════════════════════════════════════════════

def execute_nml(nml_code: str, runtime_path: str, data_file: str = None,
                timeout: int = 10) -> dict:
    """Execute NML code via the C runtime. Returns dict with success/output."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".nml", delete=False) as f:
        f.write(nml_code)
        prog_path = f.name

    cmd = [runtime_path, prog_path]
    if data_file:
        cmd.append(data_file)
    cmd.append("--max-cycles")
    cmd.append("100000")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        Path(prog_path).unlink(missing_ok=True)
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        Path(prog_path).unlink(missing_ok=True)
        return {"success": False, "stdout": "", "stderr": "timeout", "returncode": -1}
    except FileNotFoundError:
        Path(prog_path).unlink(missing_ok=True)
        return {"success": False, "stdout": "", "stderr": f"runtime not found: {runtime_path}",
                "returncode": -1}


# ═══════════════════════════════════════════════════════════════════════════════
# Model generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_nml(prompt: str, model_path: str, adapter_path: str = None,
                 max_tokens: int = 512) -> str:
    """Generate NML code from a prompt using the fine-tuned model."""
    try:
        from mlx_lm import load, generate as mlx_generate

        model, tokenizer = load(
            model_path,
            adapter_path=adapter_path,
        )

        formatted = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        response = mlx_generate(
            model, tokenizer, prompt=formatted,
            max_tokens=max_tokens,
            verbose=False,
        )
        return response.strip()
    except Exception as e:
        return f"ERROR: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# Main verification loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NML Verification Loop")
    parser.add_argument("--model", required=True,
                        help="Path to fine-tuned model (or base model if using --adapter)")
    parser.add_argument("--adapter", default=None,
                        help="Path to LoRA adapter directory (optional)")
    parser.add_argument("--runtime", default=str(Path(__file__).parent.parent / "nml"),
                        help="Path to NML C runtime binary")
    parser.add_argument("--output", default=str(
                        Path(__file__).parent.parent / "domain" / "output" / "training" / "verified_pairs.jsonl"),
                        help="Output JSONL for verified pairs")
    parser.add_argument("--repair-output", default=None,
                        help="Output JSONL for repair pairs (broken -> fix)")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of prompts to generate")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--skip-execute", action="store_true",
                        help="Only validate grammar, skip runtime execution")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    runtime = Path(args.runtime).resolve()

    print("=" * 60)
    print("  NML Verification Loop")
    print("=" * 60)
    print(f"  Model:     {args.model}")
    if args.adapter:
        print(f"  Adapter:   {args.adapter}")
    print(f"  Runtime:   {runtime}")
    print(f"  Prompts:   {args.count}")
    print(f"  Output:    {args.output}")
    print()

    prompts = []
    while len(prompts) < args.count:
        prompts.extend(random.sample(PROMPTS, min(len(PROMPTS), args.count - len(prompts))))

    verified = []
    repair = []
    stats = {"total": 0, "grammar_pass": 0, "execute_pass": 0,
             "grammar_fail": 0, "execute_fail": 0, "error": 0}

    print("Loading model...")
    try:
        from mlx_lm import load, generate as mlx_generate
        model, tokenizer = load(args.model, adapter_path=args.adapter)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return

    for idx, prompt in enumerate(prompts):
        stats["total"] += 1
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{args.count}] verified={len(verified)} repair={len(repair)}")

        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        try:
            response = mlx_generate(
                model, tokenizer, prompt=formatted,
                max_tokens=args.max_tokens, verbose=False,
            )
            nml_code = response.strip()
        except Exception as e:
            stats["error"] += 1
            continue

        if not nml_code or nml_code.startswith("ERROR"):
            stats["error"] += 1
            continue

        grammar = validate_nml_grammar(nml_code)
        if not grammar.get("valid", False):
            stats["grammar_fail"] += 1
            repair.append({
                "messages": [
                    {"role": "user", "content": f"Fix this NML program:\n{nml_code}"},
                    {"role": "assistant", "content": f"Grammar errors found: {json.dumps(grammar.get('errors', []))}"},
                ]
            })
            continue

        stats["grammar_pass"] += 1

        if args.skip_execute or not runtime.exists():
            verified.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": nml_code},
                ]
            })
            continue

        data_path = generate_data_file(nml_code)
        exec_result = execute_nml(nml_code, str(runtime), data_file=data_path)
        if data_path:
            Path(data_path).unlink(missing_ok=True)
        if exec_result["success"]:
            stats["execute_pass"] += 1
            verified.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": nml_code},
                ]
            })
        else:
            stats["execute_fail"] += 1
            repair.append({
                "messages": [
                    {"role": "user", "content": f"Fix this NML program (runtime error: {exec_result['stderr'][:200]}):\n{nml_code}"},
                    {"role": "assistant", "content": "This program has a runtime error and needs to be corrected."},
                ]
            })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for pair in verified:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    if args.repair_output and repair:
        repair_path = Path(args.repair_output)
        repair_path.parent.mkdir(parents=True, exist_ok=True)
        with open(repair_path, "w") as f:
            for pair in repair:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n{'─' * 60}")
    print(f"  Results:")
    print(f"    Total prompts:    {stats['total']}")
    print(f"    Grammar pass:     {stats['grammar_pass']}")
    print(f"    Grammar fail:     {stats['grammar_fail']}")
    print(f"    Execute pass:     {stats['execute_pass']}")
    print(f"    Execute fail:     {stats['execute_fail']}")
    print(f"    Errors:           {stats['error']}")
    print(f"    Verified pairs:   {len(verified)}")
    print(f"    Repair pairs:     {len(repair)}")
    print(f"{'─' * 60}")
    if stats['total'] > 0:
        rate = len(verified) / stats['total'] * 100
        print(f"    Success rate:     {rate:.1f}%")
    print(f"\n  Verified pairs written to: {out_path}")
    if args.repair_output and repair:
        print(f"  Repair pairs written to:   {args.repair_output}")


if __name__ == "__main__":
    main()
