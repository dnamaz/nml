#!/usr/bin/env python3
"""
Driver script: run all NML training data generators.

Outputs all JSONL files to domain/output/training/raw/, then prints a
summary table of pair counts per file.

Usage:
    python3 transpilers/run_all_generators.py [--dry-run]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "domain" / "output" / "training" / "raw"

# ---------------------------------------------------------------------------
# Generator registry
# Each entry: (module_path, output_filename, extra_args)
# Generators that hardcode their output path are listed with output=None;
# the driver copies their file into raw/ after the run.
# ---------------------------------------------------------------------------
GENERATORS = [
    # (script, output_filename, extra_cli_args)
    ("transpilers/nml_core_training_gen.py",    "nml_core_pairs.jsonl",            []),
    ("transpilers/general_purpose_training_gen.py", "general_purpose_pairs.jsonl", []),
    ("transpilers/nml_selftrain_gen.py",         "nml_selftrain_pairs.jsonl",       []),
    ("transpilers/nml_syntax_gen.py",            "nml_syntax_pairs.jsonl",          []),
    ("transpilers/gen_extension_training.py",    "extension_pairs.jsonl",           []),
    ("transpilers/nml_realworld_gen.py",         "nml_realworld_pairs.jsonl",       []),
    ("transpilers/nml_backward_gen.py",          "nml_backward_pairs.jsonl",        []),
    ("transpilers/nml_boost_gen.py",             "nml_boost_pairs.jsonl",           []),
    ("transpilers/nml_cascade_conv_gen.py",      "nml_cascade_conv_pairs.jsonl",    []),
    ("transpilers/nml_cmpi_fix_gen.py",          "nml_cmpi_fix_pairs.jsonl",        []),
    ("transpilers/nml_tensor_table_gen.py",      "nml_tensor_table_pairs.jsonl",    []),
    ("transpilers/nml_equalize_gen.py",          "nml_equalize_pairs.jsonl",        []),
    ("transpilers/nml_rebalance_gen.py",         "nml_rebalance_pairs.jsonl",       []),
    ("transpilers/nml_library_gen.py",           "nml_library_pairs.jsonl",         []),
    # These three hardcode their output path to domain/output/training/ directly
    # (not raw/); list them last so the directory already exists.
    ("transpilers/nml_jump_fix_gen.py",          "nml_jump_fix_pairs.jsonl",        None),
    ("transpilers/nml_tnet_fix_gen.py",          "nml_tnet_fix_pairs.jsonl",        None),
    ("transpilers/nml_gap_training_gen.py",      "nml_gap_fix_pairs.jsonl",         None),
]

# Hardcoded-output scripts write here; driver moves them into raw/
HARDCODED_DIR = ROOT / "domain" / "output" / "training"


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def run_generator(script: str, out_file: str, extra_args, dry_run: bool) -> tuple[bool, int, float]:
    """Run one generator. Returns (success, pair_count, elapsed_seconds)."""
    out_path = RAW_DIR / out_file

    if extra_args is None:
        # Hardcoded-output generator — runs without --output flag
        cmd = [sys.executable, str(ROOT / script)]
        dest = out_path
        src  = HARDCODED_DIR / out_file
    else:
        cmd = [sys.executable, str(ROOT / script), "--output", str(out_path)] + extra_args
        dest = out_path
        src  = None

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Running: {script}")
    if dry_run:
        return True, 0, 0.0

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"          # force UTF-8 stdout + default file encoding
    env["PYTHONIOENCODING"] = "utf-8"

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=False, text=True, env=env)
    elapsed = time.time() - t0

    # For hardcoded-output generators, move file into raw/
    if src and src.exists() and src != dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        src.replace(dest)  # replace() overwrites existing dest (safe on Windows)

    success = result.returncode == 0 and dest.exists()
    pairs = count_lines(dest) if success else 0
    return success, pairs, elapsed


def main():
    parser = argparse.ArgumentParser(description="Run all NML training data generators")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="Run only generators whose output filename contains NAME")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    generators = GENERATORS
    if args.only:
        generators = [g for g in GENERATORS
                      if any(name in g[1] for name in args.only)]
        if not generators:
            print(f"No generators matched: {args.only}")
            sys.exit(1)

    results = []
    total_pairs = 0

    for script, out_file, extra_args in generators:
        success, pairs, elapsed = run_generator(script, out_file, extra_args, args.dry_run)
        results.append((script, out_file, success, pairs, elapsed))
        if success:
            total_pairs += pairs

    # Summary table
    SEP = "-" * 72
    print("\n" + SEP)
    print(f"  {'Script':<42} {'Pairs':>8}  {'Time':>7}  {'Status'}")
    print(SEP)
    for script, out_file, success, pairs, elapsed in results:
        name = Path(script).stem
        status = "OK" if success else "FAIL"
        time_str = f"{elapsed:.1f}s" if elapsed else "-"
        print(f"  {name:<42} {pairs:>8,}  {time_str:>7}  {status}")
    print(SEP)
    print(f"  {'TOTAL':<42} {total_pairs:>8,}")
    print(SEP)
    print(f"\nOutput directory: {RAW_DIR}")

    failures = [r for r in results if not r[2]]
    if failures:
        print(f"\nFailed ({len(failures)}):")
        for r in failures:
            print(f"  {r[0]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
