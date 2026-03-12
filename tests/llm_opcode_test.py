#!/usr/bin/env python3
"""
LLM Opcode Coverage Test — generate NML from the trained model for each
opcode category, validate grammar, execute on the C runtime, and report
which opcodes the model can produce correctly.

Usage:
    python3 tests/llm_opcode_test.py \
        --model domain/output/model/nml-equalized-merged \
        --runtime ./nml
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TRANSPILER_DIR = Path(__file__).parent.parent / "transpilers"
sys.path.insert(0, str(TRANSPILER_DIR))

OPCODE_PROMPTS = [
    {
        "id": "arithmetic",
        "opcodes": ["MMUL", "MADD", "MSUB", "EMUL", "EDIV", "SDOT", "SCLR", "SDIV"],
        "prompts": [
            "Write NML to multiply a 2x3 matrix by a 3x1 vector using MMUL, add a bias with MADD, then store the result. Include the .nml.data file with sample data.",
            "Write NML to subtract two tensors with MSUB, multiply element-wise with EMUL, and divide element-wise with EDIV. Include the .nml.data file.",
            "Write NML to compute the dot product of two vectors using SDOT, scale by 2.0 with SCLR, then divide by 4.0 with SDIV. Include the .nml.data file.",
        ],
    },
    {
        "id": "activation",
        "opcodes": ["RELU", "SIGM", "TANH", "SOFT", "GELU"],
        "prompts": [
            "Write NML to load a value, apply ReLU activation with RELU, then store the result. Include the .nml.data file.",
            "Write NML to load a tensor and apply sigmoid activation with SIGM, then store. Include the .nml.data file.",
            "Write NML to apply TANH activation to a value, then apply SOFT (softmax) to a vector. Include the .nml.data file.",
            "Write NML for a feedforward layer: MMUL, MADD, then GELU activation. Include the .nml.data file with sample weights and input.",
        ],
    },
    {
        "id": "memory",
        "opcodes": ["LD", "ST", "MOV", "ALLC", "LEAF"],
        "prompts": [
            "Write NML that uses LEAF to set R0 to 42.0, MOV to copy it to R1, ALLC to allocate a zero tensor in R2, then ST to store all three.",
            "Write NML to LD a tensor from memory, modify it with SCLR, and ST the result. Include the .nml.data file.",
        ],
    },
    {
        "id": "dataflow",
        "opcodes": ["RSHP", "TRNS", "SPLT", "MERG"],
        "prompts": [
            "Write NML to load a 2x3 matrix, transpose it with TRNS, then reshape with RSHP. Include the .nml.data file.",
            "Write NML to split a tensor with SPLT and merge two tensors with MERG. Include the .nml.data file.",
        ],
    },
    {
        "id": "compare",
        "opcodes": ["CMP", "CMPI", "CMPF"],
        "prompts": [
            "Write NML to compare a register to an immediate value with CMPI, branch with JMPT. Include the .nml.data file.",
            "Write NML to compare two registers with CMP, then conditionally store different values. Include the .nml.data file.",
            "Write NML to use CMPF for a tree feature comparison on a tensor. Include the .nml.data file.",
        ],
    },
    {
        "id": "control",
        "opcodes": ["JMPT", "JMPF", "JUMP", "LOOP", "ENDP", "CALL", "RET"],
        "prompts": [
            "Write NML to sum integers from 1 to 10 using LOOP and ENDP.",
            "Write NML with a subroutine using CALL and RET that doubles a value.",
            "Write NML that uses JMPT, JMPF, and JUMP for if/else branching.",
        ],
    },
    {
        "id": "tree",
        "opcodes": ["LEAF", "TACC"],
        "prompts": [
            "Write NML to use LEAF to load constants and TACC to accumulate a sum of 5 values.",
            "Write NML for a decision tree that uses LEAF, TACC, and CMPF. Include the .nml.data file with sample feature data.",
        ],
    },
    {
        "id": "system",
        "opcodes": ["HALT", "SYNC"],
        "prompts": [
            "Write NML that loads a value, uses SYNC, stores the result, and ends with HALT.",
        ],
    },
    {
        "id": "general",
        "opcodes": ["SYS", "MOD", "ITOF", "FTOI", "BNOT"],
        "prompts": [
            "Write NML to print a number using SYS with subcode 0.",
            "Write NML to compute 17 modulo 5 using MOD.",
            "Write NML to convert an integer to float with ITOF and float to integer with FTOI.",
        ],
    },
    {
        "id": "vision",
        "opcodes": ["CONV", "POOL", "UPSC", "PADZ"],
        "prompts": [
            "Write NML for convolution with CONV followed by max pooling with POOL. Include the .nml.data file with a 4x4 image and 3x3 kernel.",
            "Write NML to zero-pad a tensor with PADZ, then upscale with UPSC. Include the .nml.data file.",
        ],
    },
    {
        "id": "transformer",
        "opcodes": ["ATTN", "NORM", "EMBD"],
        "prompts": [
            "Write NML for scaled dot-product attention using ATTN on query, key, value tensors. Include the .nml.data file.",
            "Write NML for layer normalization using NORM, followed by GELU. Include the .nml.data file.",
            "Write NML to look up embeddings from a table using EMBD. Include the .nml.data file with a 4x3 embedding table and index vector.",
        ],
    },
    {
        "id": "reduction",
        "opcodes": ["RDUC", "WHER", "CLMP", "CMPR"],
        "prompts": [
            "Write NML to compute sum, mean, max, and min of a tensor using RDUC. Include the .nml.data file.",
            "Write NML to create a comparison mask with CMPR, then use WHER to select values. Include the .nml.data file.",
            "Write NML to clamp values between 0.0 and 1.0 using CLMP. Include the .nml.data file.",
        ],
    },
    {
        "id": "signal",
        "opcodes": ["FFT", "FILT"],
        "prompts": [
            "Write NML to compute the FFT of a signal using FFT. Include the .nml.data file with an 8-sample signal.",
            "Write NML to apply a FIR filter to a signal using FILT. Include the .nml.data file with signal and filter coefficients.",
        ],
    },
    {
        "id": "m2m",
        "opcodes": ["META", "VOTE", "PROJ", "DIST", "GATH", "SCAT"],
        "prompts": [
            "Write NML with META headers declaring name, version, and domain.",
            "Write NML to compute consensus of agent values using VOTE. Include the .nml.data file with 5 agent results.",
            "Write NML to project features into embedding space with PROJ and compute distance with DIST. Include the .nml.data file.",
            "Write NML to look up tensor elements by index using GATH. Include the .nml.data file with a tensor and index.",
        ],
    },
    {
        "id": "training",
        "opcodes": ["BKWD", "WUPD", "LOSS", "TNET"],
        "prompts": [
            "Write NML for a training loop: forward pass with MMUL/RELU, compute LOSS, backpropagate with BKWD, update weights with WUPD. Include the .nml.data file with weights, input, and target.",
            "Write NML to train a neural network using the TNET opcode with 1000 epochs and learning rate 0.01. Include the .nml.data file with w1, b1, w2, b2, input, and target.",
        ],
    },
    {
        "id": "symbolic",
        "opcodes": [],
        "prompts": [
            "Write NML using only symbolic syntax (Unicode opcodes and Greek registers) to add two tensors and store the result. Include the .nml.data file.",
            "Write NML in symbolic syntax for a neural network layer: load, matrix multiply, add bias, apply ReLU, store. Include the .nml.data file.",
        ],
    },
]

SYMBOLIC_MAP = {
    "×": "MMUL", "⊕": "MADD", "⊖": "MSUB", "⊗": "EMUL", "⊘": "EDIV",
    "·": "SDOT", "∗": "SCLR", "÷": "SDIV",
    "⌐": "RELU", "σ": "SIGM", "τ": "TANH", "Σ": "SOFT", "ℊ": "GELU",
    "↓": "LD", "↑": "ST", "←": "MOV", "□": "ALLC", "∎": "LEAF",
    "⊞": "RSHP", "⊤": "TRNS", "⊢": "SPLT", "⊣": "MERG",
    "⋈": "CMPF", "≶": "CMP", "≺": "CMPI", "ϟ": "CMPI",
    "↗": "JMPT", "↘": "JMPF", "→": "JUMP", "↻": "LOOP", "↺": "ENDP",
    "⇒": "CALL", "⇐": "RET", "∑": "TACC",
    "◼": "HALT", "⚠": "TRAP", "⏸": "SYNC", "⚙": "SYS",
    "⊛": "CONV", "⊓": "POOL", "⊔": "UPSC", "⊡": "PADZ",
    "⊙": "ATTN", "‖": "NORM", "⊏": "EMBD",
    "⊥": "RDUC", "ϛ": "RDUC", "⊻": "WHER", "⊧": "CLMP", "⊜": "CMPR",
    "∿": "FFT", "⋐": "FILT",
    "§": "META", "◆": "FRAG", "◇": "ENDF", "⚖": "VOTE",
    "⟐": "PROJ", "⟂": "DIST", "⊃": "GATH", "⊂": "SCAT",
    "∇": "BKWD", "⟳": "WUPD", "△": "LOSS", "⥁": "TNET",
    "✦": "SIGN", "✓": "VRFY",
}

ALL_OPCODES = {
    "MMUL", "MADD", "MSUB", "EMUL", "EDIV", "SDOT", "DOT", "SCLR", "SDIV",
    "RELU", "SIGM", "TANH", "SOFT", "GELU",
    "LD", "ST", "MOV", "ALLC", "LEAF",
    "RSHP", "TRNS", "SPLT", "MERG",
    "CMP", "CMPI", "CMPF",
    "JMPT", "JMPF", "JUMP", "LOOP", "ENDP", "CALL", "RET",
    "TACC", "HALT", "SYNC", "TRAP",
    "SYS", "MOD", "ITOF", "FTOI", "BNOT",
    "CONV", "POOL", "UPSC", "PADZ",
    "ATTN", "NORM", "EMBD",
    "RDUC", "WHER", "CLMP", "CMPR",
    "FFT", "FILT",
    "META", "FRAG", "ENDF", "LINK", "PTCH", "SIGN", "VRFY",
    "VOTE", "PROJ", "DIST", "GATH", "SCAT", "SCTR",
    "BKWD", "WUPD", "LOSS", "TNET",
}


def extract_nml_and_data(response: str) -> tuple:
    """Pull NML code and optional .nml.data from a model response.
    Returns (nml_code, data_content) where data_content may be empty."""
    lines = response.split("\n")
    code_blocks = []
    current_block = []
    in_code = False

    for line in lines:
        if line.strip().startswith("```"):
            if in_code:
                code_blocks.append("\n".join(current_block))
                current_block = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            current_block.append(line)
    if current_block:
        code_blocks.append("\n".join(current_block))

    nml_code = ""
    data_content = ""

    if len(code_blocks) >= 2:
        nml_code = code_blocks[0]
        for block in code_blocks[1:]:
            if "@" in block and "shape=" in block:
                data_content = block
                break
    elif len(code_blocks) == 1:
        block = code_blocks[0]
        if "@" in block and "shape=" in block and ("LD " in block or "↓ " in block or "HALT" in block):
            nml_lines = []
            data_lines = []
            for line in block.split("\n"):
                stripped = line.strip()
                if stripped.startswith("@") and "shape=" in stripped:
                    data_lines.append(line)
                else:
                    nml_lines.append(line)
            nml_code = "\n".join(nml_lines)
            data_content = "\n".join(data_lines)
        else:
            nml_code = block

    if not nml_code:
        nml_lines = []
        data_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("@") and "shape=" in stripped:
                data_lines.append(line)
                continue
            if not stripped or stripped.startswith(";") or stripped.startswith("#"):
                nml_lines.append(line)
                continue
            first = stripped.split()[0] if stripped.split() else ""
            if first.upper() in ALL_OPCODES or first in SYMBOLIC_MAP or first.startswith("META") or first.startswith("§"):
                nml_lines.append(line)
        nml_code = "\n".join(nml_lines).strip()
        data_content = "\n".join(data_lines).strip()

    return nml_code.strip(), data_content.strip()


def extract_nml(response: str) -> str:
    """Backward-compatible wrapper."""
    code, _ = extract_nml_and_data(response)
    return code


def extract_opcodes_used(nml_code: str) -> set:
    """Find which opcodes appear in the code (classic + symbolic)."""
    found = set()
    for line in nml_code.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        first = stripped.split()[0]
        upper = first.upper()
        if upper in ALL_OPCODES:
            found.add(upper)
        if first in SYMBOLIC_MAP:
            found.add(SYMBOLIC_MAP[first])
        if upper == "DOT":
            found.add("SDOT")
        if upper == "SCTR":
            found.add("SCAT")
    return found


def validate_grammar(nml_code: str) -> dict:
    try:
        import nml_grammar
        report = nml_grammar.validate_grammar(nml_code)
        return report.to_dict()
    except Exception as e:
        return {"valid": False, "errors": [{"message": str(e)}]}


def execute_nml(nml_code: str, runtime: str) -> dict:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".nml", delete=False) as f:
        f.write(nml_code)
        path = f.name
    try:
        r = subprocess.run(
            [runtime, path, "--max-cycles", "100000"],
            capture_output=True, text=True, timeout=10,
        )
        return {"success": r.returncode == 0, "stdout": r.stdout[:500], "stderr": r.stderr[:300]}
    except subprocess.TimeoutExpired:
        return {"success": False, "stderr": "timeout"}
    finally:
        Path(path).unlink(missing_ok=True)


def generate_data_for_program(nml_code: str) -> str:
    """Auto-generate a .nml.data file for programs that use LD."""
    ld_names = set()
    st_names = set()
    for line in nml_code.split("\n"):
        stripped = line.strip()
        tokens = stripped.split()
        if not tokens:
            continue
        op = tokens[0].upper()
        if op in ("LD", "LOAD") or tokens[0] == "↓":
            for t in tokens:
                if t.startswith("@"):
                    ld_names.add(t[1:])
        if op in ("ST", "STORE") or tokens[0] == "↑":
            for t in tokens:
                if t.startswith("@"):
                    st_names.add(t[1:])

    inputs = ld_names - st_names
    if not inputs:
        return ""

    shape_hints = {
        "w1": "shape=1,4 dtype=f64 data=0.5,-0.3,0.2,-0.1",
        "b1": "shape=1,4 dtype=f64 data=0.1,0.1,0.1,0.1",
        "w2": "shape=4,1 dtype=f64 data=0.4,-0.2,0.3,0.1",
        "b2": "shape=1,1 dtype=f64 data=0.0",
        "weights1": "shape=1,4 dtype=f64 data=0.5,-0.3,0.2,-0.1",
        "bias1": "shape=1,4 dtype=f64 data=0.1,0.1,0.1,0.1",
        "weights2": "shape=4,1 dtype=f64 data=0.4,-0.2,0.3,0.1",
        "bias2": "shape=1,1 dtype=f64 data=0.0",
    }
    defaults_by_keyword = [
        (["weight", "kernel", "matrix", "proj"], "shape=4,4 data=" + ",".join(["0.1"] * 16)),
        (["bias", "beta", "offset"], "shape=1,4 data=0.1,0.1,0.1,0.1"),
        (["target", "label", "expected", "train_y"], "shape=1,1 data=7.0"),
        (["input", "feature", "sample", "data", "x", "train"], "shape=1,4 data=0.5,0.3,0.8,0.1"),
        (["gamma"], "shape=4 data=1.0,1.0,1.0,1.0"),
        (["embed", "table"], "shape=4,3 data=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2"),
        (["index", "indices", "idx"], "shape=2 data=0,2"),
        (["agent", "vote", "consensus"], "shape=5 data=100.0,100.01,99.98,100.0,95.0"),
        (["image", "img"], "shape=4,4 data=" + ",".join(["0.5"] * 16)),
        (["signal", "wave"], "shape=8 data=0.0,0.707,1.0,0.707,0.0,-0.707,-1.0,-0.707"),
        (["coeff", "filter"], "shape=3 data=0.25,0.5,0.25"),
        (["query", "key", "value", "q", "k", "v"], "shape=2,4 data=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8"),
        (["tensor", "vector", "array", "values"], "shape=4 data=10.0,20.0,30.0,40.0"),
    ]

    lines = []
    for name in sorted(inputs):
        if name in shape_hints:
            lines.append(f"@{name} {shape_hints[name]}")
            continue
        matched = False
        for keywords, spec in defaults_by_keyword:
            if any(kw in name.lower() for kw in keywords):
                lines.append(f"@{name} {spec}")
                matched = True
                break
        if not matched:
            lines.append(f"@{name} shape=1,4 data=0.5,0.3,0.8,0.1")

    return "\n".join(lines)


def execute_nml_with_data(nml_code: str, runtime: str, llm_data: str = "") -> dict:
    """Execute NML with LLM-generated data, falling back to auto-generated data."""
    data_content = llm_data if llm_data else generate_data_for_program(nml_code)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".nml", delete=False) as f:
        f.write(nml_code)
        nml_path = f.name

    data_path = None
    if data_content:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".nml.data", delete=False) as f:
            f.write(data_content)
            data_path = f.name

    try:
        cmd = [runtime, nml_path]
        if data_path:
            cmd.append(data_path)
        cmd.extend(["--max-cycles", "100000"])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return {"success": r.returncode == 0, "stdout": r.stdout[:500], "stderr": r.stderr[:300]}
    except subprocess.TimeoutExpired:
        return {"success": False, "stderr": "timeout"}
    finally:
        Path(nml_path).unlink(missing_ok=True)
        if data_path:
            Path(data_path).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="LLM Opcode Coverage Test")
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime", default="./nml")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--retries", type=int, default=2, help="Retries on grammar failure (M2M mode)")
    parser.add_argument("--output", default=None, help="Save raw results as JSON")
    args = parser.parse_args()

    print("=" * 70)
    print("  NML LLM Opcode Coverage Test")
    print("=" * 70)
    print(f"  Model:   {args.model}")
    print(f"  Runtime: {args.runtime}")
    print(f"  Retries: {args.retries}")
    print()
    print("  Loading model...")

    from mlx_lm import load, generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    model, tokenizer = load(args.model)
    print("  Model loaded.\n")

    results = []
    opcode_grammar = {}
    opcode_exec = {}
    opcode_generated = {}
    total_retries_used = 0

    total_prompts = sum(len(cat["prompts"]) for cat in OPCODE_PROMPTS)
    prompt_idx = 0

    for cat in OPCODE_PROMPTS:
        cat_id = cat["id"]
        cat_opcodes = set(cat["opcodes"])

        for prompt in cat["prompts"]:
            prompt_idx += 1
            sys.stdout.write(f"\r  [{prompt_idx}/{total_prompts}] {cat_id:<14} ")
            sys.stdout.flush()

            best_nml = None
            best_grammar = None
            best_grammar_ok = False
            attempts = 0

            best_data = ""
            retry_temps = [0.0, 0.5, 0.8, 1.0]
            for attempt in range(1 + args.retries):
                attempts = attempt + 1
                temp = retry_temps[min(attempt, len(retry_temps) - 1)]
                sampler = make_sampler(temp=temp)
                formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                try:
                    response = mlx_generate(
                        model, tokenizer, prompt=formatted,
                        max_tokens=args.max_tokens, verbose=False,
                        sampler=sampler,
                    )
                except Exception as e:
                    continue

                nml_code, llm_data = extract_nml_and_data(response)
                if not nml_code.strip():
                    continue

                grammar = validate_grammar(nml_code)
                grammar_ok = grammar.get("valid", False)

                if grammar_ok:
                    best_nml = nml_code
                    best_grammar = grammar
                    best_grammar_ok = True
                    best_data = llm_data
                    if attempt > 0:
                        total_retries_used += attempt
                    break
                elif best_nml is None:
                    best_nml = nml_code
                    best_grammar = grammar
                    best_data = llm_data

            if best_nml is None:
                results.append({"cat": cat_id, "prompt": prompt, "error": "empty extraction", "attempts": attempts})
                continue

            nml_code = best_nml
            grammar_ok = best_grammar_ok
            grammar = best_grammar or {}

            opcodes_used = extract_opcodes_used(nml_code)
            for op in opcodes_used:
                opcode_generated.setdefault(op, 0)
                opcode_generated[op] += 1

            for op in opcodes_used:
                if grammar_ok:
                    opcode_grammar[op] = opcode_grammar.get(op, 0) + 1

            exec_ok = False
            if grammar_ok:
                exec_result = execute_nml_with_data(nml_code, args.runtime, llm_data=best_data)
                exec_ok = exec_result.get("success", False)
                if exec_ok:
                    for op in opcodes_used:
                        opcode_exec[op] = opcode_exec.get(op, 0) + 1

            results.append({
                "cat": cat_id,
                "prompt": prompt,
                "grammar": grammar_ok,
                "execute": exec_ok,
                "opcodes": sorted(opcodes_used),
                "errors": grammar.get("errors", []) if not grammar_ok else [],
                "nml": nml_code[:300],
                "attempts": attempts,
            })

    print("\n")

    # Summary
    total = len(results)
    g_pass = sum(1 for r in results if r.get("grammar"))
    e_pass = sum(1 for r in results if r.get("execute"))
    errors = sum(1 for r in results if "error" in r and r.get("error"))

    multi_attempt = sum(1 for r in results if r.get("attempts", 1) > 1 and r.get("grammar"))

    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  Total prompts:     {total}")
    print(f"  Grammar pass:      {g_pass}/{total} ({g_pass/total*100:.0f}%)")
    print(f"  Execution pass:    {e_pass}/{total} ({e_pass/total*100:.0f}%)")
    print(f"  Generation errors: {errors}")
    if args.retries > 0:
        print(f"  Retries used:      {total_retries_used} (saved {multi_attempt} prompts)")
    print()

    # Per-category
    print("  Per-Category Results:")
    print(f"  {'Category':<14} {'Prompts':>7} {'Grammar':>9} {'Execute':>9}")
    print("  " + "─" * 43)
    for cat in OPCODE_PROMPTS:
        cat_id = cat["id"]
        cat_results = [r for r in results if r["cat"] == cat_id]
        n = len(cat_results)
        g = sum(1 for r in cat_results if r.get("grammar"))
        e = sum(1 for r in cat_results if r.get("execute"))
        print(f"  {cat_id:<14} {n:>7} {g:>5}/{n:<3} {e:>5}/{n:<3}")
    print()

    # Per-opcode coverage
    print("  Per-Opcode Coverage (generated at least once):")
    print(f"  {'Opcode':<8} {'Generated':>10} {'Grammar OK':>11} {'Exec OK':>9}")
    print("  " + "─" * 42)
    all_ops_sorted = sorted(ALL_OPCODES)
    covered = 0
    grammar_covered = 0
    exec_covered = 0
    for op in all_ops_sorted:
        gen = opcode_generated.get(op, 0)
        gok = opcode_grammar.get(op, 0)
        eok = opcode_exec.get(op, 0)
        if gen > 0:
            covered += 1
        if gok > 0:
            grammar_covered += 1
        if eok > 0:
            exec_covered += 1
        marker = "  " if gen > 0 else "  ✗"
        if gen > 0:
            marker = " ✓" if eok > 0 else " ~" if gok > 0 else " ✗"
        print(f"  {op:<8} {gen:>10} {gok:>11} {eok:>9} {marker}")

    print()
    n_ops = len(ALL_OPCODES)
    print(f"  Opcodes generated:        {covered}/{n_ops}")
    print(f"  Opcodes grammar-valid:    {grammar_covered}/{n_ops}")
    print(f"  Opcodes execution-valid:  {exec_covered}/{n_ops}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  Raw results saved to {args.output}")

    # Print any failed generations for debugging
    failures = [r for r in results if not r.get("grammar") and not r.get("error")]
    if failures:
        print(f"\n  Grammar failures ({len(failures)}):")
        for r in failures[:5]:
            print(f"    [{r['cat']}] {r['prompt'][:60]}...")
            for err in r.get("errors", [])[:2]:
                msg = err.get("message", err) if isinstance(err, dict) else str(err)
                print(f"      → {msg}")


if __name__ == "__main__":
    main()
