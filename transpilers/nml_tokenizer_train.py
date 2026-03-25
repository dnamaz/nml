#!/usr/bin/env python3
"""
NML BPE Tokenizer Trainer — train a 4,096-vocab tokenizer on the NML corpus.

Uses the HuggingFace `tokenizers` library to train a Byte-Pair Encoding
tokenizer with NML-specific special tokens pre-seeded as atomic units,
ensuring every opcode, register, and structural punctuation token is
represented as a single token rather than fragmented sub-words.

Special token categories (~500 total, always atomic):
  - 85 canonical opcodes (classic names)
  - All symbolic opcode aliases (Unicode single-char and multi-char)
  - All verbose opcode aliases
  - 32 numbered registers (R0-RV)
  - 32 Greek register aliases (ι, κ, λ, ...)
  - Verbose register names (INPUT, KERNEL, ACCUMULATOR, ...)
  - Structural punctuation (@, #, [, ], ,, ;, ¶, :, =)
  - Data type keywords (f32, f64, i32, shape=, dtype=, float, ...)
  - Chat control tokens matching Qwen2.5's template

Corpus:
  - All .nml files under programs/ and tests/
  - Assistant-side text from train.jsonl / valid.jsonl (if already built)
  - All .nml files under tests/opcode_coverage/

Output: domain/output/tokenizer/nml_bpe_4096/
  - tokenizer.json        — HuggingFace fast tokenizer
  - tokenizer_config.json — metadata
  - special_tokens.txt    — one special token per line (for reference)

Usage:
    python3 nml_tokenizer_train.py
    python3 nml_tokenizer_train.py --vocab-size 8192
    python3 nml_tokenizer_train.py --corpus-dir /path/to/nml/files
    python3 nml_tokenizer_train.py --jsonl ../domain/output/training/train.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Imports — check early so the user gets a clear error message
# ─────────────────────────────────────────────────────────────────────────────

try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from tokenizers.normalizers import NFD, StripAccents, Sequence as NormSequence
except ImportError:
    print(
        "ERROR: `tokenizers` library not found.\n"
        "Install it with:  pip install tokenizers",
        file=sys.stderr,
    )
    sys.exit(1)

# Add transpilers directory to path
sys.path.insert(0, str(Path(__file__).parent))
from nml_grammar import (
    _CLASSIC_OPCODES,
    _SYMBOLIC_TO_CANONICAL,
    _VERBOSE_TO_CANONICAL,
    _ALIAS_TO_CANONICAL,
    _NUMBERED_LIST,
    _GREEK_LIST,
    _VERBOSE_REG_LIST,
)

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
DEFAULT_OUT_DIR  = ROOT / "domain" / "output" / "tokenizer" / "nml_bpe_4096"
DEFAULT_NML_DIRS = [
    ROOT / "programs",
    ROOT / "tests",
    ROOT / "tests" / "opcode_coverage",
    ROOT / "docs",
]
DEFAULT_JSONL_DIRS = [
    ROOT / "domain" / "output" / "training",
]

DEFAULT_VOCAB_SIZE = 4096

# ─────────────────────────────────────────────────────────────────────────────
# Build the special-token list
# ─────────────────────────────────────────────────────────────────────────────

def build_special_tokens() -> list[str]:
    """
    Return the ordered list of special tokens to pre-seed in the tokenizer.
    These are always represented as single atomic tokens regardless of BPE merges.
    """
    tokens: list[str] = []
    seen: set[str] = set()

    def add(t: str) -> None:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            tokens.append(t)

    # ── Chat control tokens (Qwen2.5 template) ────────────────────────────────
    for ct in ["<|im_start|>", "<|im_end|>", "<|pad|>", "[UNK]", "[PAD]",
               "<|user|>", "<|assistant|>", "<|system|>", "<s>", "</s>"]:
        add(ct)

    # ── Canonical (classic) opcodes ───────────────────────────────────────────
    for op in _CLASSIC_OPCODES:
        add(op)

    # ── Symbolic opcode aliases ────────────────────────────────────────────────
    for sym in _SYMBOLIC_TO_CANONICAL:
        add(sym)

    # ── Verbose opcode aliases ─────────────────────────────────────────────────
    for verb in _VERBOSE_TO_CANONICAL:
        add(verb)

    # ── Extra aliases ──────────────────────────────────────────────────────────
    for alias in _ALIAS_TO_CANONICAL:
        add(alias)

    # ── Registers — numbered ──────────────────────────────────────────────────
    for r in _NUMBERED_LIST:
        add(r)

    # ── Registers — Greek aliases ─────────────────────────────────────────────
    for r in _GREEK_LIST:
        add(r)

    # ── Registers — verbose names ─────────────────────────────────────────────
    for r in _VERBOSE_REG_LIST:
        add(r)

    # ── Structural punctuation (as standalone tokens) ─────────────────────────
    for p in ["@", "#", "[", "]", ",", ";", "¶", ":", "=", ".", "(", ")"]:
        add(p)

    # ── Data type keywords ─────────────────────────────────────────────────────
    for kw in [
        "f32", "f64", "i32",
        "shape=", "dtype=", "data=",
        "float", "currency", "ratio", "category",
        "count", "bool", "embedding", "probability",
    ]:
        add(kw)

    # ── Common register-prefixed patterns ─────────────────────────────────────
    # Verbose keyword= argument patterns used in verbose syntax
    for kw in ["dest=", "left=", "right=", "src=", "rows=", "cols=",
               "stride=", "scale=", "mode=", "axis=", "heads=", "eps="]:
        add(kw)

    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# Corpus collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_corpus_text(
    nml_dirs: list[Path],
    jsonl_dirs: list[Path],
    verbose: bool = True,
) -> list[str]:
    """
    Collect raw text lines for tokenizer training from:
      - .nml source files
      - assistant-side content in .jsonl files
    Returns a list of file paths (strings) for the tokenizers trainer.
    We write everything to a single temporary corpus file and return its path.
    """
    import tempfile, os

    lines: list[str] = []

    # ── .nml source files ─────────────────────────────────────────────────────
    nml_count = 0
    for d in nml_dirs:
        if not d.exists():
            continue
        for nml_file in sorted(d.glob("**/*.nml")):
            try:
                text = nml_file.read_text(encoding="utf-8")
                lines.extend(text.splitlines())
                nml_count += 1
            except Exception:
                continue

    if verbose:
        print(f"  Loaded {nml_count} .nml files")

    # ── Assistant-side text from JSONL files ──────────────────────────────────
    jsonl_count = 0
    for d in jsonl_dirs:
        if not d.exists():
            continue
        for jf in sorted(d.glob("*.jsonl")):
            try:
                with open(jf, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        # Chat format — extract assistant content
                        if "messages" in rec:
                            for msg in rec["messages"]:
                                if msg.get("role") in ("assistant", "user", "system"):
                                    lines.extend(msg.get("content", "").splitlines())
                        # Alpaca format
                        elif "output" in rec:
                            lines.extend(rec["output"].splitlines())
                            if "instruction" in rec:
                                lines.extend(rec["instruction"].splitlines())
                jsonl_count += 1
            except Exception:
                continue

    if verbose:
        print(f"  Loaded {jsonl_count} .jsonl files")

    # Write to a temporary corpus file (tokenizers trainer needs file paths)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    )
    for line in lines:
        tmp.write(line + "\n")
    tmp.close()

    if verbose:
        size_kb = Path(tmp.name).stat().st_size / 1024
        print(f"  Corpus: {len(lines):,} lines, {size_kb:.1f} KB → {tmp.name}")

    return [tmp.name]


# ─────────────────────────────────────────────────────────────────────────────
# Train the tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def train_tokenizer(
    corpus_files: list[str],
    special_tokens: list[str],
    vocab_size: int,
    out_dir: Path,
    verbose: bool = True,
) -> Tokenizer:
    """Train a BPE tokenizer and save it to out_dir."""

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log(f"\nTraining BPE tokenizer (vocab_size={vocab_size}, special_tokens={len(special_tokens)})...")

    # ── Model: BPE with unknown token ─────────────────────────────────────────
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # ── Pre-tokenizer: whitespace split, preserving all characters ────────────
    # ByteLevel pre-tokenizer handles unicode properly and ensures every byte
    # can be represented; add_prefix_space=False so "@name" stays as "@" + "name"
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # ── Decoder matching the ByteLevel pre-tokenizer ──────────────────────────
    tokenizer.decoder = decoders.ByteLevel()

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=verbose,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    tokenizer.train(corpus_files, trainer)
    log(f"Training complete. Vocabulary size: {tokenizer.get_vocab_size()}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_dir / "tokenizer.json"))
    log(f"Saved tokenizer.json → {out_dir / 'tokenizer.json'}")

    # ── Save tokenizer_config.json (HuggingFace transformers compatible) ──────
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "model_max_length": 512,
        "special_tokens_map_file": None,
        "tokenizer_file": "tokenizer.json",
        "name_or_path": "nml-bpe-4096",
        "description": (
            "NML (Neural Machine Language) BPE tokenizer. "
            "Pre-seeded with all 85 NML opcodes, register aliases, and structural tokens."
        ),
    }
    with open(out_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    log(f"Saved tokenizer_config.json → {out_dir / 'tokenizer_config.json'}")

    # ── Save special tokens list ───────────────────────────────────────────────
    with open(out_dir / "special_tokens.txt", "w", encoding="utf-8") as f:
        for tok in special_tokens:
            f.write(tok + "\n")
    log(f"Saved special_tokens.txt → {out_dir / 'special_tokens.txt'} ({len(special_tokens)} tokens)")

    return tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Fragmentation analysis (optional diagnostic)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_fragmentation(tokenizer: Tokenizer, verbose: bool = True) -> None:
    """
    Show how the trained tokenizer encodes key NML tokens.
    A well-trained tokenizer should represent each NML token as 1 token.
    """
    test_cases = [
        # Opcode samples
        "MMUL", "MATRIX_MULTIPLY", "×", "⊕", "⌐", "◼",
        # Register samples
        "R0", "ι", "κ", "ACCUMULATOR",
        # Structural
        "@sensor_data", "#0.062", "#[3,4]",
        # Full instruction
        "MMUL R0 R1 R2",
        "× ι κ λ",
        "LD R0 @input",
    ]
    if verbose:
        print("\nFragmentation analysis (tokens per item):")
        print(f"  {'Input':<30} {'Tokens':<8} {'Ids'}")
        print(f"  {'-'*30} {'-'*8} {'-'*30}")
        for tc in test_cases:
            enc = tokenizer.encode(tc)
            ids_str = str(enc.ids[:8]) + ("..." if len(enc.ids) > 8 else "")
            print(f"  {tc:<30} {len(enc.ids):<8} {ids_str}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer on the NML corpus."
    )
    parser.add_argument(
        "--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE,
        help=f"BPE vocabulary size (default: {DEFAULT_VOCAB_SIZE})",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--nml-dir", type=Path, action="append", dest="nml_dirs",
        help="Directory to scan for .nml files (can repeat; default: programs/ tests/)",
    )
    parser.add_argument(
        "--jsonl-dir", type=Path, action="append", dest="jsonl_dirs",
        help="Directory to scan for .jsonl files (can repeat)",
    )
    parser.add_argument(
        "--no-analysis", action="store_true",
        help="Skip fragmentation analysis after training",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    verbose = not args.quiet

    nml_dirs  = args.nml_dirs  or list(DEFAULT_NML_DIRS)
    jsonl_dirs = args.jsonl_dirs or list(DEFAULT_JSONL_DIRS)

    # ── Build special tokens ──────────────────────────────────────────────────
    special_tokens = build_special_tokens()
    if verbose:
        print(f"Special tokens: {len(special_tokens)}")
        print(f"  First 10: {special_tokens[:10]}")

    # ── Collect corpus ────────────────────────────────────────────────────────
    if verbose:
        print("\nCollecting corpus...")
    corpus_files = collect_corpus_text(nml_dirs, jsonl_dirs, verbose=verbose)

    # ── Train ─────────────────────────────────────────────────────────────────
    tokenizer = train_tokenizer(
        corpus_files=corpus_files,
        special_tokens=special_tokens,
        vocab_size=args.vocab_size,
        out_dir=args.out_dir,
        verbose=verbose,
    )

    # ── Cleanup temp file ──────────────────────────────────────────────────────
    import os
    for p in corpus_files:
        try:
            os.unlink(p)
        except Exception:
            pass

    # ── Fragmentation analysis ─────────────────────────────────────────────────
    if not args.no_analysis:
        analyze_fragmentation(tokenizer, verbose=verbose)

    if verbose:
        print(f"\nTokenizer ready at: {args.out_dir}")
        print(
            "\nNext step: resize the base model's embedding table:\n"
            "  model.resize_token_embeddings(tokenizer.get_vocab_size())\n"
            "  # New NML-specific rows start randomly initialized — this is intentional."
        )


if __name__ == "__main__":
    main()
