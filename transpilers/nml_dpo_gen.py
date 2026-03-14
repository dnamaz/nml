#!/usr/bin/env python3
"""
NML DPO Preference Pair Generator

Generates preference pairs for Direct Preference Optimization by using
the NML grammar validator + C runtime as an automated reward model.

For each prompt, generates multiple candidate programs at temperature > 0,
scores them (grammar valid? executes?), and picks the best as "chosen"
and worst as "rejected".

Output format (MLX DPO compatible):
    {"prompt": "...", "chosen": "...", "rejected": "..."}

Usage:
    python3 nml_dpo_gen.py \
        --model ../domain/output/model/nml-boost-combined-merged \
        --runtime ../nml \
        --output ../domain/output/training/nml_dpo_pairs.jsonl \
        --count 2000 --candidates 4
"""

import json
import random
import argparse
import subprocess
import tempfile
import sys
from pathlib import Path

random.seed(42)

sys.path.insert(0, str(Path(__file__).parent))
from nml_verify_gen import validate_nml_grammar, execute_nml, generate_data_file

PROMPTS = [
    "Write NML to add two vectors element-wise.",
    "Write NML to multiply a matrix by a vector.",
    "Write NML to apply ReLU activation to a tensor.",
    "Write NML to apply sigmoid activation.",
    "Write NML to compute the softmax of a vector.",
    "Write NML to scale a value by 3.5.",
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
    "Write NML to apply a FIR filter to a signal.",
    "Write NML to find consensus of values using VOTE.",
    "Write NML to print a number using SYS.",
    "Write NML for absolute value using conditional branching.",
    "Write NML to return the larger of two values.",
    "Write NML for a three-tier rate schedule.",
    "Write NML for layer normalization followed by GELU activation.",
    "Write NML for a feedforward layer with GELU activation.",
    "Write NML to create a comparison mask for values greater than 50.",
    "Write NML to gather elements from a tensor by index.",
    "Write NML to zero-pad a tensor then convolve with a kernel.",
    "Write NML for an autoencoder with ReLU in the bottleneck.",
    "Write NML to upscale a tensor by 2x.",
    "Write NML to accumulate a value over 20 iterations in a loop.",
    "Write NML to convert an integer to float using ITOF.",
    "Write NML for a weighted sum: multiply element-wise then reduce.",
    "Write NML for a linear regression model: y = Wx + b.",
    "Write NML to compute compound growth over 12 periods.",
    "Write NML for a 2-layer CNN with ReLU and pooling.",
    "Write NML to subtract two tensors using MSUB.",
    "Write NML to reshape a tensor using RSHP.",
    "Write NML for element-wise division using EDIV.",
    "Write NML to apply tanh activation to a tensor.",
    "Write NML to compute bitwise NOT using BNOT.",
    # Backward opcodes
    "Write NML to compute the ReLU backward pass using RELUBK.",
    "Write NML for sigmoid backward using SIGMBK.",
    "Write NML for tanh backward using TANHBK.",
    "Write NML for GELU backward using GELUBK.",
    "Write NML for softmax backward using SOFTBK.",
    "Write NML for matmul backward using MMULBK to get d_input and d_weight.",
    "Write NML for convolution backward using CONVBK.",
    "Write NML for max pool backward using POOLBK.",
    "Write NML for layer norm backward using NORMBK.",
    "Write NML for attention backward using ATTNBK to get dQ, dK, dV.",
    "Write NML to train a 2-layer network using TNDEEP with Adam.",
    "Write NML for a training loop: forward, LOSS, RELUBK, MMULBK, WUPD.",
    "Write NML for CNN training with CONVBK, RELUBK, POOLBK backward pass.",
    # Symbolic
    "Write symbolic NML to add two tensors.",
    "Write symbolic NML for a dense layer with sigmoid.",
    "Write symbolic NML to compute ReLU backward.",
    # Verbose
    "Write verbose NML to load, scale, and store a value.",
    "Write verbose NML for a neural network layer with ReLU.",
]


def score_candidate(code, runtime_path):
    """Score a candidate: 2 = grammar+executes, 1 = grammar only, 0 = fails grammar."""
    grammar = validate_nml_grammar(code)
    if not grammar.get("valid", False):
        return 0, "grammar_fail"

    data_path = generate_data_file(code)
    exec_result = execute_nml(code, runtime_path, data_file=data_path)
    if data_path:
        Path(data_path).unlink(missing_ok=True)

    if exec_result["success"]:
        return 2, "pass"
    return 1, f"exec_fail: {exec_result['stderr'][:100]}"


def main():
    parser = argparse.ArgumentParser(description="Generate NML DPO preference pairs")
    parser.add_argument("--model", required=True, help="Path to fine-tuned model")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter")
    parser.add_argument("--runtime", default=str(Path(__file__).parent.parent / "nml"))
    parser.add_argument("--output", default=str(
        Path(__file__).parent.parent / "domain" / "output" / "training" / "nml_dpo_pairs.jsonl"))
    parser.add_argument("--count", type=int, default=2000, help="Number of prompts to use")
    parser.add_argument("--candidates", type=int, default=4, help="Candidates per prompt")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    runtime = Path(args.runtime).resolve()

    print("=" * 60)
    print("  NML DPO Preference Pair Generator")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Runtime:    {runtime}")
    print(f"  Prompts:    {args.count}")
    print(f"  Candidates: {args.candidates} per prompt")
    print(f"  Temp:       {args.temperature}")
    print()

    print("Loading model...")
    try:
        from mlx_lm import load, generate as mlx_generate
        import mlx.core as mx
        model, tokenizer = load(args.model, adapter_path=args.adapter)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return

    def make_sampler(temp):
        """Create a temperature sampler for mlx_lm generate."""
        def sampler(logits):
            if temp <= 0:
                return mx.argmax(logits, axis=-1)
            scaled = logits / temp
            return mx.random.categorical(scaled)
        return sampler

    prompts = []
    while len(prompts) < args.count:
        prompts.extend(random.sample(PROMPTS, min(len(PROMPTS), args.count - len(prompts))))
    prompts = prompts[:args.count]

    pairs = []
    stats = {"total": 0, "pairs_generated": 0, "both_pass": 0, "both_fail": 0, "no_variance": 0}

    for idx, prompt in enumerate(prompts):
        stats["total"] += 1
        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{args.count}] pairs={stats['pairs_generated']}")

        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        candidates = []
        for ci in range(args.candidates):
            try:
                temp = args.temperature if ci > 0 else 0.0
                response = mlx_generate(
                    model, tokenizer, prompt=formatted,
                    max_tokens=args.max_tokens, verbose=False,
                    sampler=make_sampler(temp),
                )
                code = response.strip()
                if code and not code.startswith("ERROR"):
                    score, reason = score_candidate(code, str(runtime))
                    candidates.append((code, score, reason))
            except Exception as e:
                continue

        if len(candidates) < 2:
            continue

        candidates.sort(key=lambda x: x[1], reverse=True)
        best = candidates[0]
        worst = candidates[-1]

        if best[1] == worst[1]:
            stats["no_variance"] += 1
            if best[1] == 2:
                stats["both_pass"] += 1
            elif best[1] == 0:
                stats["both_fail"] += 1
            continue

        pairs.append({
            "prompt": prompt,
            "chosen": best[0],
            "rejected": worst[0],
        })
        stats["pairs_generated"] += 1

    print(f"\n{'─' * 60}")
    print(f"  Results:")
    print(f"    Total prompts:     {stats['total']}")
    print(f"    Preference pairs:  {stats['pairs_generated']}")
    print(f"    Both passed:       {stats['both_pass']} (no preference signal)")
    print(f"    Both failed:       {stats['both_fail']} (no preference signal)")
    print(f"    No variance:       {stats['no_variance']}")
    print(f"{'─' * 60}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    random.shuffle(pairs)

    # Write DPO format for MLX
    dpo_dir = out_path.parent / "mlx-dpo"
    dpo_dir.mkdir(parents=True, exist_ok=True)

    split = int(len(pairs) * 0.9)
    train_pairs = pairs[:split]
    valid_pairs = pairs[split:]

    with open(dpo_dir / "train.jsonl", "w") as f:
        for p in train_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(dpo_dir / "valid.jsonl", "w") as f:
        for p in valid_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n  Written: {out_path}")
    print(f"  DPO train: {dpo_dir / 'train.jsonl'} ({len(train_pairs)} pairs)")
    print(f"  DPO valid: {dpo_dir / 'valid.jsonl'} ({len(valid_pairs)} pairs)")


if __name__ == "__main__":
    main()
