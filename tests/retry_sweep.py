#!/usr/bin/env python3
"""
Retry sweep — test each model at retry counts 0, 1, 2, 3 to find optimal retry count.
Uses the two best models: tnet-incr and gap-fresh.
"""

import json
import subprocess
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "domain" / "output" / "model"
TEST_SCRIPT = Path(__file__).parent / "llm_opcode_test.py"
RUNTIME = Path(__file__).parent.parent / "nml"

MODELS = [
    ("tnet-incr", "nml-tnet-merged"),
    ("gap-fresh", "nml-gap-fresh-merged"),
]
RETRIES = [0, 1, 2, 3]


def run_test(model_name, model_dir, retries):
    tag = f"{model_name}_r{retries}"
    output_file = Path(__file__).parent / f"retry_{tag}.json"
    cmd = [
        sys.executable, str(TEST_SCRIPT),
        "--model", str(model_dir),
        "--runtime", str(RUNTIME),
        "--retries", str(retries),
        "--output", str(output_file),
    ]
    sys.stdout.write(f"  {model_name} retries={retries} ... ")
    sys.stdout.flush()

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if output_file.exists():
        results = json.load(open(output_file))
        total = len(results)
        grammar = sum(1 for r in results if r.get("grammar"))
        execute = sum(1 for r in results if r.get("execute"))
        retries_used = sum(r.get("attempts", 1) - 1 for r in results if r.get("grammar"))
        multi = sum(1 for r in results if r.get("attempts", 1) > 1 and r.get("grammar"))
        print(f"grammar={grammar}/{total} ({grammar*100//total}%)  exec={execute}/{total} ({execute*100//total}%)  retries_used={retries_used}  saved={multi}")
        return {
            "model": model_name, "retries": retries,
            "grammar": grammar, "execute": execute, "total": total,
            "grammar_pct": round(grammar / total * 100),
            "execute_pct": round(execute / total * 100),
            "retries_used": retries_used, "saved": multi,
        }
    else:
        print("FAILED")
        return None


def main():
    all_results = []

    for name, folder in MODELS:
        model_path = MODEL_DIR / folder
        if not model_path.exists():
            print(f"  SKIP: {model_path}")
            continue
        print(f"\n{'='*60}")
        print(f"  Model: {name}")
        print(f"{'='*60}")
        for retries in RETRIES:
            r = run_test(name, model_path, retries)
            if r:
                all_results.append(r)

    print(f"\n\n{'='*70}")
    print("  RETRY SWEEP RESULTS")
    print(f"{'='*70}\n")

    for name, _ in MODELS:
        model_results = [r for r in all_results if r["model"] == name]
        if not model_results:
            continue
        print(f"  {name}:")
        print(f"  {'Retries':>8} {'Grammar':>12} {'Execution':>12} {'Retries Used':>14} {'Prompts Saved':>14}")
        print(f"  {'─'*62}")
        for r in model_results:
            print(f"  {r['retries']:>8} {r['grammar']}/{r['total']} ({r['grammar_pct']}%){'':<3} {r['execute']}/{r['total']} ({r['execute_pct']}%){'':<3} {r['retries_used']:>14} {r['saved']:>14}")
        
        if len(model_results) >= 2:
            base = model_results[0]
            best = max(model_results, key=lambda x: x["grammar"])
            gain = best["grammar"] - base["grammar"]
            print(f"  Gain from retries: +{gain} grammar ({base['grammar_pct']}% → {best['grammar_pct']}%)")
        print()

    summary_path = Path(__file__).parent / "retry_sweep_summary.json"
    json.dump(all_results, open(summary_path, "w"), indent=2)
    print(f"  Saved to {summary_path}")


if __name__ == "__main__":
    main()
