#!/usr/bin/env python3
"""
Compare all available NML models on the opcode coverage test.
Runs each model through the same 41 prompts and reports side-by-side results.
"""

import json
import subprocess
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "domain" / "output" / "model"
TEST_SCRIPT = Path(__file__).parent / "llm_opcode_test.py"
RUNTIME = Path(__file__).parent.parent / "nml"

MODELS = [
    ("Qwen-base", "Qwen2.5-Coder-7B-Instruct-4bit"),
    ("equalized", "nml-equalized-merged"),
    ("gap-incr", "nml-gap-merged"),
    ("tnet-incr", "nml-tnet-merged"),
    ("gap-fresh", "nml-gap-fresh-merged"),
]

def run_test(model_name, model_dir):
    output_file = Path(__file__).parent / f"compare_{model_name}.json"
    cmd = [
        sys.executable, str(TEST_SCRIPT),
        "--model", str(model_dir),
        "--runtime", str(RUNTIME),
        "--retries", "0",
        "--output", str(output_file),
    ]
    print(f"\n{'='*60}")
    print(f"  Testing: {model_name}")
    print(f"  Model:   {model_dir}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr[-200:])

    if output_file.exists():
        results = json.load(open(output_file))
        total = len(results)
        grammar = sum(1 for r in results if r.get("grammar"))
        execute = sum(1 for r in results if r.get("execute"))
        errors = sum(1 for r in results if r.get("error"))

        cats = {}
        for r in results:
            c = r["cat"]
            if c not in cats:
                cats[c] = {"total": 0, "grammar": 0, "execute": 0}
            cats[c]["total"] += 1
            if r.get("grammar"):
                cats[c]["grammar"] += 1
            if r.get("execute"):
                cats[c]["execute"] += 1

        return {
            "model": model_name,
            "total": total,
            "grammar": grammar,
            "execute": execute,
            "errors": errors,
            "grammar_pct": round(grammar / total * 100) if total else 0,
            "execute_pct": round(execute / total * 100) if total else 0,
            "categories": cats,
        }
    return None


def main():
    all_results = []

    for name, model_folder in MODELS:
        model_path = MODEL_DIR / model_folder
        if not model_path.exists():
            print(f"  SKIP: {model_path} not found")
            continue
        try:
            result = run_test(name, model_path)
            if result:
                all_results.append(result)
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT: {name}")
        except Exception as e:
            print(f"  ERROR: {name}: {e}")

    if not all_results:
        print("No results collected.")
        return

    print("\n")
    print("=" * 80)
    print("  MODEL COMPARISON — SIDE BY SIDE")
    print("=" * 80)

    header = f"  {'Metric':<20}"
    for r in all_results:
        header += f" {r['model']:>12}"
    print(header)
    print("  " + "─" * (20 + 13 * len(all_results)))

    row = f"  {'Grammar':<20}"
    for r in all_results:
        row += f" {r['grammar']}/{r['total']} ({r['grammar_pct']}%)".rjust(13)
    print(row)

    row = f"  {'Execution':<20}"
    for r in all_results:
        row += f" {r['execute']}/{r['total']} ({r['execute_pct']}%)".rjust(13)
    print(row)

    row = f"  {'Errors':<20}"
    for r in all_results:
        row += f" {r['errors']}".rjust(13)
    print(row)

    print()
    print("  Per-Category Grammar Pass:")
    print(f"  {'Category':<16}", end="")
    for r in all_results:
        print(f" {r['model']:>12}", end="")
    print()
    print("  " + "─" * (16 + 13 * len(all_results)))

    all_cats = sorted(set(c for r in all_results for c in r["categories"]))
    for cat in all_cats:
        row = f"  {cat:<16}"
        for r in all_results:
            cd = r["categories"].get(cat, {"grammar": 0, "total": 0})
            row += f" {cd['grammar']}/{cd['total']}".rjust(13)
        print(row)

    print()
    print("  Per-Category Execution Pass:")
    print(f"  {'Category':<16}", end="")
    for r in all_results:
        print(f" {r['model']:>12}", end="")
    print()
    print("  " + "─" * (16 + 13 * len(all_results)))

    for cat in all_cats:
        row = f"  {cat:<16}"
        for r in all_results:
            cd = r["categories"].get(cat, {"execute": 0, "total": 0})
            row += f" {cd['execute']}/{cd['total']}".rjust(13)
        print(row)

    summary_path = Path(__file__).parent / "compare_summary.json"
    json.dump(all_results, open(summary_path, "w"), indent=2)
    print(f"\n  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
