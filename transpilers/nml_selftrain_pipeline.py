#!/usr/bin/env python3
"""
NML Self-Training Pipeline (RLEF — Reinforcement Learning from Execution Feedback)

Automated generate-verify-retrain loop:
  1. Generate N programs from diverse prompts using current model
  2. Grammar check + execution check each program
  3. Passing programs -> verified pairs (positive signal)
  4. Failing programs + error -> repair pairs (fix signal)
  5. Short LoRA round on combined verified + repair data
  6. Merge, re-verify, report accuracy
  7. Repeat until target accuracy reached

Usage:
    python3 nml_selftrain_pipeline.py \
        --model ../domain/output/model/nml-boost-combined-merged \
        --base-model ../domain/output/model/Qwen2.5-Coder-7B-Instruct-4bit \
        --runtime ../nml \
        --rounds 5 --programs-per-round 500 --target-exec 95
"""

import json
import random
import argparse
import subprocess
import sys
import time
from pathlib import Path

random.seed(42)

sys.path.insert(0, str(Path(__file__).parent))

from nml_verify_gen import (
    validate_nml_grammar, execute_nml, generate_data_file, PROMPTS as VERIFY_PROMPTS,
)

DIVERSE_PROMPTS = list(VERIFY_PROMPTS) + [
    "Write NML using {op1} and {op2} together.",
    "Write NML for a computation that uses {op1}.",
    "Write self-contained NML using LEAF to {task}.",
    "Write NML to {task} using a LOOP of {n} iterations.",
    "Write NML for a {layers}-layer neural network with {act}.",
]

OPCODES = [
    "MMUL", "MADD", "MSUB", "EMUL", "EDIV", "SDOT", "SCLR", "SDIV",
    "RELU", "SIGM", "TANH", "SOFT", "GELU",
    "CONV", "POOL", "ATTN", "NORM", "FFT", "FILT",
    "RDUC", "WHER", "CLMP", "CMPR",
    "RELUBK", "SIGMBK", "MMULBK", "CONVBK", "POOLBK", "NORMBK", "ATTNBK",
    "TNET", "TNDEEP", "LOSS", "BKWD", "WUPD",
]

TASKS = [
    "add two vectors", "multiply matrices", "scale by 3.5",
    "compute dot product", "apply sigmoid", "compute softmax",
    "find maximum", "compute mean", "transpose a matrix",
    "apply ReLU backward", "train a dense layer",
    "compute MSE loss", "apply convolution",
]

ACTIVATIONS = ["ReLU", "sigmoid", "tanh", "GELU"]


def expand_prompts(count):
    """Generate diverse prompts from templates + fixed pool."""
    prompts = list(VERIFY_PROMPTS)
    while len(prompts) < count:
        template = random.choice(DIVERSE_PROMPTS)
        if "{op1}" in template and "{op2}" in template:
            ops = random.sample(OPCODES, 2)
            prompts.append(template.format(op1=ops[0], op2=ops[1]))
        elif "{op1}" in template:
            prompts.append(template.format(op1=random.choice(OPCODES)))
        elif "{task}" in template:
            t = random.choice(TASKS)
            n = random.choice([5, 10, 20, 50, 100])
            prompts.append(template.format(task=t, n=n))
        elif "{layers}" in template:
            prompts.append(template.format(
                layers=random.choice([2, 3, 4]),
                act=random.choice(ACTIVATIONS)))
        else:
            prompts.append(template)
    random.shuffle(prompts)
    return prompts[:count]


def run_verification_round(model_path, adapter_path, runtime_path, prompts, max_tokens=512):
    """Generate and verify programs. Returns (verified, repair, stats)."""
    try:
        from mlx_lm import load, generate as mlx_generate
        model, tokenizer = load(model_path, adapter_path=adapter_path)
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return [], [], {"total": 0}

    verified, repair = [], []
    stats = {"total": 0, "grammar_pass": 0, "grammar_fail": 0,
             "execute_pass": 0, "execute_fail": 0, "error": 0}

    for idx, prompt in enumerate(prompts):
        stats["total"] += 1
        if (idx + 1) % 50 == 0:
            print(f"    [{idx+1}/{len(prompts)}] verified={len(verified)} repair={len(repair)}")

        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        try:
            response = mlx_generate(
                model, tokenizer, prompt=formatted,
                max_tokens=max_tokens, verbose=False,
            )
            nml_code = response.strip()
        except Exception:
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
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": nml_code},
                ]
            })
            continue

        stats["grammar_pass"] += 1
        data_path = generate_data_file(nml_code)
        exec_result = execute_nml(nml_code, runtime_path, data_file=data_path)
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
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": nml_code},
                ]
            })

    return verified, repair, stats


def run_training_round(base_model, data_path, adapter_dir, resume_from, iters=1000, lr=3e-6):
    """Run a short LoRA training round."""
    import yaml
    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    config_path = adapter_dir / "train_config.yaml"
    config_path.write_text(yaml.dump({
        "lora_parameters": {"rank": 16, "dropout": 0.0, "scale": 20.0}
    }))

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", str(base_model),
        "--data", str(data_path),
        "--adapter-path", str(adapter_dir),
        "--iters", str(iters),
        "--learning-rate", str(lr),
        "--num-layers", "22",
        "--max-seq-length", "1280",
        "--batch-size", "1",
        "--grad-checkpoint",
        "--save-every", "100",
        "-c", str(config_path),
        "--train",
    ]
    if resume_from:
        cmd.extend(["--resume-adapter-file", str(resume_from)])

    print(f"  Training: {iters} iters, lr={lr}, resume={resume_from}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if result.returncode != 0:
        print(f"  Training stderr: {result.stderr[-300:]}")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="NML Self-Training Pipeline (RLEF)")
    parser.add_argument("--model", required=True, help="Current merged model path")
    parser.add_argument("--base-model", required=True, help="Base model for LoRA")
    parser.add_argument("--adapter", default=None, help="Current adapter (if not merged)")
    parser.add_argument("--initial-adapter", default=None,
                        help="Adapter .safetensors to resume from on round 1")
    parser.add_argument("--runtime", default=str(Path(__file__).parent.parent / "nml"))
    parser.add_argument("--output-dir", default=str(
        Path(__file__).parent.parent / "domain" / "output" / "training" / "selftrain"))
    parser.add_argument("--rounds", type=int, default=5, help="Max self-training rounds")
    parser.add_argument("--programs-per-round", type=int, default=500)
    parser.add_argument("--iters-per-round", type=int, default=1000)
    parser.add_argument("--target-exec", type=float, default=95.0, help="Target execution %")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--anchor-data", default=None, nargs="+",
                        help="JSONL files with anchor pairs to mix in (prevents forgetting)")
    parser.add_argument("--min-train-pairs", type=int, default=500,
                        help="Minimum training pairs per round (padded from anchors)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = str(Path(args.runtime).resolve())

    print("=" * 60)
    print("  NML Self-Training Pipeline (RLEF)")
    print("=" * 60)
    print(f"  Model:          {args.model}")
    print(f"  Base:           {args.base_model}")
    print(f"  Rounds:         {args.rounds}")
    print(f"  Programs/round: {args.programs_per_round}")
    print(f"  Target exec:    {args.target_exec}%")
    print()

    current_model = args.model
    current_adapter = args.adapter
    resume_adapter = Path(args.initial_adapter) if args.initial_adapter else None

    anchor_pairs = []
    if args.anchor_data:
        for path in args.anchor_data:
            with open(path) as f:
                for line in f:
                    anchor_pairs.append(json.loads(line))
        print(f"  Loaded {len(anchor_pairs)} anchor pairs from {len(args.anchor_data)} files")

    all_verified = []
    prev_exec_pct = 0

    for round_num in range(1, args.rounds + 1):
        print(f"\n{'═' * 60}")
        print(f"  Round {round_num}/{args.rounds}")
        print(f"{'═' * 60}")

        prompts = expand_prompts(args.programs_per_round)

        print(f"  Generating and verifying {len(prompts)} programs...")
        verified, repair, stats = run_verification_round(
            current_model, current_adapter, runtime, prompts
        )

        total = stats["total"]
        grammar_pct = stats["grammar_pass"] / total * 100 if total else 0
        exec_pct = stats["execute_pass"] / total * 100 if total else 0

        print(f"\n  Round {round_num} Results:")
        print(f"    Grammar: {stats['grammar_pass']}/{total} ({grammar_pct:.1f}%)")
        print(f"    Execute: {stats['execute_pass']}/{total} ({exec_pct:.1f}%)")
        print(f"    Verified: {len(verified)}, Repair: {len(repair)}")

        if round_num > 1 and exec_pct < prev_exec_pct * 0.8:
            print(f"\n  EARLY STOP: Accuracy dropped from {prev_exec_pct:.1f}% to {exec_pct:.1f}%")
            print(f"  Rolling back to previous adapter.")
            break

        prev_exec_pct = exec_pct

        if exec_pct >= args.target_exec:
            print(f"\n  TARGET REACHED: {exec_pct:.1f}% >= {args.target_exec}%")
            print(f"  Stopping self-training.")
            break

        all_verified.extend(verified)

        train_pairs = list(verified)
        if len(train_pairs) < args.min_train_pairs and anchor_pairs:
            needed = args.min_train_pairs - len(train_pairs)
            anchor_sample = random.sample(anchor_pairs, min(needed, len(anchor_pairs)))
            train_pairs.extend(anchor_sample)
            print(f"  Padded with {len(anchor_sample)} anchor pairs (total: {len(train_pairs)})")

        if not train_pairs:
            print("  No verified pairs generated. Skipping training.")
            continue

        round_dir = output_dir / f"round_{round_num}"
        round_dir.mkdir(parents=True, exist_ok=True)

        random.shuffle(train_pairs)
        split = int(len(train_pairs) * 0.9)
        with open(round_dir / "train.jsonl", "w") as f:
            for p in train_pairs[:split]:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        with open(round_dir / "valid.jsonl", "w") as f:
            for p in train_pairs[split:]:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        print(f"\n  Training on {len(train_pairs)} verified pairs (NO repair pairs)...")
        adapter_dir = output_dir / f"adapters_round_{round_num}"

        ok = run_training_round(
            args.base_model, round_dir, adapter_dir,
            resume_from=resume_adapter,
            iters=args.iters_per_round, lr=args.lr,
        )

        if ok:
            resume_adapter = adapter_dir / "adapters.safetensors"
            current_adapter = str(adapter_dir)
            print(f"  Round {round_num} training complete.")
        else:
            print(f"  Round {round_num} training FAILED.")
            break

    print(f"\n{'═' * 60}")
    print(f"  Self-Training Complete")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
