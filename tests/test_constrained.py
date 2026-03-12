#!/usr/bin/env python3
"""
Test constrained decoding with Outlines CFG on the NML model.
Compares constrained vs unconstrained generation.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "transpilers"))

from nml_lark_grammar import NML_GRAMMAR

PROMPTS = [
    "Write NML to add two values and store the result.",
    "Write NML for a neural network layer with ReLU activation.",
    "Write NML to compare a value to 100 and branch.",
    "Write NML to sum 1 to 10 using a loop.",
    "Write NML to train with TNET for 500 epochs.",
    "Write NML using symbolic syntax to scale a value by 2.",
]


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "domain/output/model/nml-gap-fresh-merged"

    print("=" * 60)
    print("  Constrained Decoding Test")
    print("=" * 60)

    print("\n  Loading model for unconstrained...")
    from mlx_lm import load, generate as mlx_generate
    model_raw, tokenizer = load(model_path)

    print("  Loading model for constrained (Outlines)...")
    import outlines
    import mlx_lm
    from outlines.types import CFG

    outlines_model = outlines.from_mlxlm(model_raw, tokenizer)
    nml_cfg = CFG(NML_GRAMMAR)

    print("  Ready.\n")

    import nml_grammar

    for prompt in PROMPTS:
        print(f"  PROMPT: {prompt}")
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        # Unconstrained
        t0 = time.time()
        raw = mlx_generate(model_raw, tokenizer, prompt=formatted, max_tokens=256, verbose=False)
        t_raw = time.time() - t0

        # Extract NML from raw
        lines = raw.strip().split("\n")
        nml_raw = "\n".join(l for l in lines if not l.strip().startswith("```"))
        grammar_raw = nml_grammar.validate_grammar(nml_raw)

        # Constrained
        t0 = time.time()
        constrained = outlines_model(formatted, output_type=nml_cfg, max_tokens=256)
        t_cfg = time.time() - t0

        grammar_cfg = nml_grammar.validate_grammar(constrained)

        print(f"    Unconstrained: grammar={'PASS' if grammar_raw.valid else 'FAIL'}  {t_raw:.1f}s")
        if not grammar_raw.valid:
            for e in grammar_raw.errors[:2]:
                print(f"      ERR: {e.message}")
        print(f"    Code: {nml_raw[:80]}")
        print(f"    Constrained:   grammar={'PASS' if grammar_cfg.valid else 'FAIL'}  {t_cfg:.1f}s")
        print(f"    Code: {constrained[:80]}")
        print()


if __name__ == "__main__":
    main()
