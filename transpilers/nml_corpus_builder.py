#!/usr/bin/env python3
"""
NML Corpus Builder — normalize, validate, deduplicate, audit, and split
all generator output into clean train/valid/test JSONL files.

Handles two input formats:
  - Chat format:   {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
  - Alpaca format: {"instruction": "...", "input": "...", "output": "..."}

Both are normalized to chat format with a condensed NML system prompt.

Steps:
  1. Collect all .jsonl files from the raw input directory
  2. Normalize to chat format
  3. Grammar-validate assistant responses that look like NML code
  4. Execution-validate self-contained programs (no external LD) via runtime
  5. Deduplicate by (user, assistant) hash
  6. Opcode coverage audit — warn on opcodes with < MIN_OPCODE_EXAMPLES
  7. Stratified split: 80% train / 10% valid / 10% test
  8. Write domain/output/training/{train,valid,test}.jsonl

Usage:
    python3 nml_corpus_builder.py
    python3 nml_corpus_builder.py --raw-dir /path/to/raw --runtime ../nml
    python3 nml_corpus_builder.py --no-exec-validate  # skip runtime checks
    python3 nml_corpus_builder.py --anchor-dir ../programs --anchor-weight 3
"""

import argparse
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# Add transpilers directory to path so we can import nml_grammar
sys.path.insert(0, str(Path(__file__).parent))
from nml_grammar import validate_grammar, _CLASSIC_OPCODES

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
DEFAULT_RAW_DIR = ROOT / "domain" / "output" / "training" / "raw"
DEFAULT_OUT_DIR = ROOT / "domain" / "output" / "training"
DEFAULT_ANCHOR_DIRS = [ROOT / "programs", ROOT / "tests"]

# Resolve runtime binary: NML_RUNTIME env var overrides auto-detection.
# On Windows the binary is nml.exe; on other platforms it is nml.
_NML_EXE = "nml.exe" if platform.system() == "Windows" else "nml"
DEFAULT_RUNTIME = Path(os.environ.get("NML_RUNTIME", str(ROOT / _NML_EXE)))

TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO  = 0.10

MIN_OPCODE_EXAMPLES = 200   # warn if an opcode appears fewer times than this
ANCHOR_WEIGHT = 3           # replicate anchor pairs this many times in train set
EXEC_SAMPLE_RATE = 1.0      # fraction of self-contained programs to execution-test

# ─────────────────────────────────────────────────────────────────────────────
# System prompt (condensed from docs/NML_LLM_PROMPT.md)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert NML (Neural Machine Language) programmer.

NML is a minimal deterministic machine language for AI workloads with 32 tensor registers and 85 opcodes.

## Syntax variants (all equivalent)
- Classic:  MMUL R0 R1 R2
- Symbolic: × ι κ λ   (Unicode, token-efficient)
- Verbose:  MATRIX_MULTIPLY dest=R0 left=R1 right=R2

## Key registers
R0=ι(input), R8(loss), R9=ς(labels), RA=α(accumulator), RD=δ(counter),
RE=φ(flag/condition), RV(arch descriptor for TRAIN/INFER).
R1-R6: weight/bias pairs for TRAIN/INFER (R1=w1, R2=b1, ..., R6=b3).

## Core opcodes
Memory:     LD Rd @name | ST Rs @name | MOV Rd Rs | ALLC Rd #[r,c] | LEAF Rd #val
Arithmetic: MMUL Rd Ra Rb | MADD Rd Ra Rb | MSUB | EMUL | EDIV | SDOT | SCLR Rd Rs #v | SDIV
Activation: RELU Rd Rs | SIGM | TANH | SOFT | GELU
Data flow:  RSHP | TRNS | SPLT | MERG
Compare:    CMP Rd Rs | CMPI Rd Rs #v | CMPF Rd Ra Rb
Control:    JMPT offset | JMPF offset | JUMP offset | LOOP [Rd [#n]] | ENDP
Subroutine: CALL label | RET
System:     SYNC | HALT | TRAP [#code]
Extensions: CONV POOL UPSC PADZ | ATTN NORM EMBD | RDUC WHER CLMP CMPR | FFT FILT
M2M:        META FRAG ENDF LINK PTCH SIGN VRFY VOTE PROJ DIST GATH SCAT
Training:   BKWD WUPD LOSS TRAIN INFER | RELUBK SIGMBK TANHBK GELUBK SOFTBK MMULBK CONVBK POOLBK NORMBK ATTNBK
Config:     TLOG WDECAY | Legacy aliases: TNET TNDEEP (redirect to TRAIN)
General:    SYS Rs Rd | MOD Rd Ra Rb | ITOF Rd Rs | FTOI Rd Rs | BNOT Rd Rs

## Data file format
@name shape=rows,cols dtype=f32 data=v1,v2,...

Always end programs with HALT."""


# ─────────────────────────────────────────────────────────────────────────────
# Format detection and normalization
# ─────────────────────────────────────────────────────────────────────────────

def _is_chat_format(record: dict) -> bool:
    return "messages" in record and isinstance(record["messages"], list)


def _is_alpaca_format(record: dict) -> bool:
    return "instruction" in record and "output" in record


def normalize_record(record: dict) -> dict | None:
    """Normalize a raw record to chat format. Returns None if unrecognizable."""
    if _is_chat_format(record):
        msgs = record["messages"]
        # Ensure there is a system message; inject if missing
        if msgs and msgs[0].get("role") == "system":
            return record
        return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + msgs}

    if _is_alpaca_format(record):
        instruction = record["instruction"].strip()
        inp = record.get("input", "").strip()
        output = record.get("output", "").strip()
        user_content = f"{instruction}\n{inp}".strip() if inp else instruction
        return {
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": output},
            ]
        }

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Extract user / assistant text from a normalized record
# ─────────────────────────────────────────────────────────────────────────────

def _get_assistant_text(record: dict) -> str:
    for msg in record["messages"]:
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def _get_user_text(record: dict) -> str:
    for msg in record["messages"]:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Detect NML code in assistant response
# ─────────────────────────────────────────────────────────────────────────────

# Matches classic opcodes at the start of a line (with optional leading whitespace)
_NML_LINE_RE = re.compile(
    r"^\s*(?:" + "|".join(re.escape(op) for op in _CLASSIC_OPCODES) + r")\b",
    re.MULTILINE,
)
# Symbolic opcode starters
_SYMBOLIC_STARTS = set("↓↑←□×⊕⊖⊗⊘·∗÷∎∑⨁⌐στΣℊ⊞⊤⊢⊣⊂⊃⋈≶≺↗↘→↻↺⏸◼⇒⇐⚠⊛⊓⊔⊡⊙‖⊏⊥⊻⊧⊜∿⋐§◆◇⊿✦✓⚖⟐⟂⚙%⊶⊷¬∇⟳△⥁⧖⟴⟶ω")


def _looks_like_nml(text: str) -> bool:
    """Return True if the text appears to contain NML instructions."""
    if _NML_LINE_RE.search(text):
        return True
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and stripped[0] in _SYMBOLIC_STARTS:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Grammar validation
# ─────────────────────────────────────────────────────────────────────────────

def grammar_valid(nml_text: str) -> bool:
    """Return True if the NML text passes grammar validation."""
    report = validate_grammar(nml_text)
    return report.valid


# ─────────────────────────────────────────────────────────────────────────────
# Self-contained detection (no external LD @addr)
# ─────────────────────────────────────────────────────────────────────────────

_LD_EXTERNAL_RE = re.compile(r"^\s*(?:LD|↓|LOAD)\s+\S+\s+@\w+", re.MULTILINE)


def _is_self_contained(nml_text: str) -> bool:
    """Return True if the program has no LD @addr instructions (no external data needed)."""
    return not bool(_LD_EXTERNAL_RE.search(nml_text))


# ─────────────────────────────────────────────────────────────────────────────
# Execution validation
# ─────────────────────────────────────────────────────────────────────────────

def execution_valid(nml_text: str, runtime_path: Path) -> bool:
    """
    Run nml_text through the C runtime. Returns True if it exits cleanly (HALTED).
    Only call for self-contained programs.
    """
    if not runtime_path.exists():
        return False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".nml", delete=False, encoding="utf-8"
        ) as f:
            f.write(nml_text)
            tmp_path = f.name
        result = subprocess.run(
            [str(runtime_path), tmp_path],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Opcode coverage audit
# ─────────────────────────────────────────────────────────────────────────────

def audit_opcode_coverage(records: list[dict]) -> dict[str, int]:
    """Count how many assistant responses contain each canonical opcode."""
    counts: dict[str, int] = defaultdict(int)
    for rec in records:
        text = _get_assistant_text(rec)
        for op in _CLASSIC_OPCODES:
            if re.search(r"\b" + re.escape(op) + r"\b", text):
                counts[op] += 1
    return dict(counts)


# ─────────────────────────────────────────────────────────────────────────────
# Load anchor NML files (hand-crafted programs and tests)
# ─────────────────────────────────────────────────────────────────────────────

def load_anchor_records(anchor_dirs: list[Path]) -> list[dict]:
    """
    Load hand-crafted .nml files as training records.
    Each file becomes a chat pair: user = "Explain and reproduce this NML program",
    assistant = the program text.
    """
    records = []
    for d in anchor_dirs:
        if not d.exists():
            continue
        for nml_file in sorted(d.glob("**/*.nml")):
            text = nml_file.read_text(encoding="utf-8").strip()
            if not text:
                continue
            # Generate a user message from the file name and first comment
            first_comment = ""
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith(";"):
                    first_comment = stripped.lstrip("; ").strip()
                    break
            fname = nml_file.stem.replace("_", " ")
            user = f"Write an NML program for: {first_comment or fname}"
            records.append({
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": user},
                    {"role": "assistant", "content": text},
                ],
                "_anchor": True,
            })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Stratified split
# ─────────────────────────────────────────────────────────────────────────────

def stratified_split(
    records: list[dict],
    train_ratio: float = TRAIN_RATIO,
    valid_ratio: float = VALID_RATIO,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split records into train/valid/test.
    Anchor records always go to train (replicated by ANCHOR_WEIGHT).
    Non-anchor records are shuffled then split.
    """
    anchors = [r for r in records if r.get("_anchor")]
    non_anchors = [r for r in records if not r.get("_anchor")]

    random.shuffle(non_anchors)
    n = len(non_anchors)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train = non_anchors[:n_train]
    valid = non_anchors[n_train:n_train + n_valid]
    test  = non_anchors[n_train + n_valid:]

    # Replicate anchor records into train set
    for rec in anchors:
        clean = {k: v for k, v in rec.items() if k != "_anchor"}
        train.extend([clean] * ANCHOR_WEIGHT)

    return train, valid, test


# ─────────────────────────────────────────────────────────────────────────────
# Write JSONL
# ─────────────────────────────────────────────────────────────────────────────

def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            # Strip internal metadata keys before writing
            clean = {k: v for k, v in rec.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_corpus(
    raw_dir: Path,
    out_dir: Path,
    anchor_dirs: list[Path],
    runtime_path: Path,
    exec_validate: bool = True,
    verbose: bool = True,
) -> None:
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    # ── Step 1: Collect raw JSONL files ──────────────────────────────────────
    raw_files = sorted(raw_dir.glob("**/*.jsonl")) if raw_dir.exists() else []
    log(f"[1/7] Found {len(raw_files)} raw JSONL file(s) in {raw_dir}")

    # ── Step 2: Load and normalize ───────────────────────────────────────────
    raw_records: list[dict] = []
    for jf in raw_files:
        with open(jf, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                normalized = normalize_record(rec)
                if normalized:
                    raw_records.append(normalized)

    log(f"[2/7] Normalized {len(raw_records):,} records")

    # ── Step 3: Grammar validation ───────────────────────────────────────────
    grammar_passed: list[dict] = []
    grammar_skipped = 0
    grammar_failed  = 0
    for rec in raw_records:
        asst = _get_assistant_text(rec)
        if _looks_like_nml(asst):
            if grammar_valid(asst):
                grammar_passed.append(rec)
            else:
                grammar_failed += 1
        else:
            # Non-NML responses (Q&A, explanations) — keep without validation
            grammar_skipped += 1
            grammar_passed.append(rec)

    log(
        f"[3/7] Grammar validation: {len(grammar_passed):,} kept, "
        f"{grammar_failed:,} discarded, {grammar_skipped:,} non-NML (kept as-is)"
    )

    # ── Step 4: Execution validation (self-contained programs only) ──────────
    exec_verified: list[dict] = []
    exec_ok = 0
    exec_fail = 0
    exec_skip = 0
    for rec in grammar_passed:
        asst = _get_assistant_text(rec)
        if exec_validate and _looks_like_nml(asst) and _is_self_contained(asst):
            if random.random() < EXEC_SAMPLE_RATE:
                if execution_valid(asst, runtime_path):
                    rec = dict(rec, _exec_verified=True)
                    exec_ok += 1
                else:
                    exec_fail += 1
                    continue  # discard execution failures for self-contained programs
            else:
                exec_skip += 1
        else:
            exec_skip += 1
        exec_verified.append(rec)

    log(
        f"[4/7] Execution validation: {exec_ok:,} verified OK, "
        f"{exec_fail:,} failed (discarded), {exec_skip:,} skipped"
    )

    # ── Step 5: Deduplication ────────────────────────────────────────────────
    seen: set[str] = set()
    deduped: list[dict] = []
    dup_count = 0
    for rec in exec_verified:
        key = hashlib.md5(
            (_get_user_text(rec) + "\x00" + _get_assistant_text(rec)).encode()
        ).hexdigest()
        if key in seen:
            dup_count += 1
        else:
            seen.add(key)
            deduped.append(rec)

    log(f"[5/7] Deduplication: {dup_count:,} duplicates removed, {len(deduped):,} unique records")

    # ── Step 6: Load anchor data ─────────────────────────────────────────────
    anchor_records = load_anchor_records(anchor_dirs)
    log(f"       Anchor records from hand-crafted programs/tests: {len(anchor_records):,}")
    all_records = deduped + anchor_records

    # ── Step 7: Opcode coverage audit ────────────────────────────────────────
    coverage = audit_opcode_coverage(all_records)
    low_coverage = {op: cnt for op, cnt in coverage.items() if cnt < MIN_OPCODE_EXAMPLES}
    missing = [op for op in _CLASSIC_OPCODES if op not in coverage]

    if low_coverage:
        log(f"\n[6/7] COVERAGE WARNING — opcodes below {MIN_OPCODE_EXAMPLES} examples:")
        for op, cnt in sorted(low_coverage.items(), key=lambda x: x[1]):
            log(f"       {op:12s} {cnt:5d} examples  ← run nml_equalize_gen.py / nml_rebalance_gen.py")
    if missing:
        log(f"       MISSING from corpus: {', '.join(missing)}")
    if not low_coverage and not missing:
        log(f"[6/7] Opcode coverage OK — all {len(_CLASSIC_OPCODES)} opcodes present")

    # ── Step 8: Split and write ───────────────────────────────────────────────
    train, valid, test = stratified_split(all_records)
    random.shuffle(train)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    write_jsonl(out_dir / "test.jsonl",  test)

    total_size = sum(
        (out_dir / f).stat().st_size
        for f in ("train.jsonl", "valid.jsonl", "test.jsonl")
        if (out_dir / f).exists()
    )
    log(
        f"\n[7/7] Split complete:\n"
        f"       train: {len(train):,} records → {out_dir / 'train.jsonl'}\n"
        f"       valid: {len(valid):,} records → {out_dir / 'valid.jsonl'}\n"
        f"       test:  {len(test):,}  records → {out_dir / 'test.jsonl'}\n"
        f"       total size: {total_size / 1024 / 1024:.1f} MB"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build NML training corpus from generator output."
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
        help=f"Directory containing raw .jsonl files (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Output directory for train/valid/test.jsonl (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--runtime", type=Path, default=DEFAULT_RUNTIME,
        help=f"Path to NML runtime binary (default: {DEFAULT_RUNTIME})",
    )
    parser.add_argument(
        "--no-exec-validate", action="store_true",
        help="Skip execution validation via C runtime",
    )
    parser.add_argument(
        "--anchor-dir", type=Path, action="append", dest="anchor_dirs",
        help="Additional anchor directory (can be specified multiple times)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    anchor_dirs = list(DEFAULT_ANCHOR_DIRS)
    if args.anchor_dirs:
        anchor_dirs.extend(args.anchor_dirs)

    build_corpus(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        anchor_dirs=anchor_dirs,
        runtime_path=args.runtime,
        exec_validate=not args.no_exec_validate,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
