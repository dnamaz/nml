#!/usr/bin/env python3
"""
Generate ~1K CMPI threshold pattern training pairs.

CMPI has non-obvious semantics: flag = (value < immediate).
JMPF fires when flag is false, i.e. when value >= immediate.
The model must learn the correct branch ordering.

Sub-generators:
  1. gen_cmpi_threshold  (~400): binary decision with varied thresholds
  2. gen_cmpi_range      (~200): two CMPI checks for low/mid/high
  3. gen_cmpi_explanation (~200): Q&A explaining the semantics
  4. gen_cmpi_symbolic    (~200): symbolic syntax variants

Output: domain/output/training/nml_cmpi_fix_pairs.jsonl
"""

import json
import random
import argparse
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent.parent / "domain" / "output" / "training" / "nml_cmpi_fix_pairs.jsonl"

random.seed(2026_03_16)

from nml_core_training_gen import (
    _fmt, _pair,
    apply_syntax, pick_syntax, syntax_tag,
)


def _thresh():
    return round(random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9]), 1)


def _val(label):
    return round(random.uniform(0.0, 100.0), 1)


SCORE_NAMES = ["score", "probability", "risk_score", "confidence", "fraud_score",
               "anomaly_score", "churn_prob", "spam_score", "quality"]
DECISION_NAMES = ["decision", "flag", "alert", "result", "is_positive",
                  "approved", "blocked", "flagged"]
INPUT_NAMES = ["input", "value", "data", "x", "features", "reading", "measurement"]


def gen_cmpi_threshold(count=400):
    """Binary decision: compute something, then CMPI threshold."""
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        thresh = _thresh()
        score = random.choice(SCORE_NAMES)
        decision = random.choice(DECISION_NAMES)
        inp = random.choice(INPUT_NAMES)
        hi_val = round(random.choice([1.0, 1.0, 2.0, 100.0]), 1)
        lo_val = 0.0

        variant = random.randint(0, 3)

        if variant == 0:
            lines = [
                _fmt("LD", "R0", f"@{inp}"),
                _fmt("LD", "R1", "@w1"),
                _fmt("LD", "R2", "@b1"),
                _fmt("MMUL", "R3", "R0", "R1"),
                _fmt("MADD", "R3", "R3", "R2"),
                _fmt("SIGM", "RA", "R3"),
                _fmt("ST", "RA", f"@{score}"),
                _fmt("CMPI", "RE", "RA", f"#{thresh}"),
                _fmt("JMPF", "#2"),
                _fmt("LEAF", "RB", f"#{lo_val}"),
                _fmt("JUMP", "#1"),
                _fmt("LEAF", "RB", f"#{hi_val}"),
                _fmt("ST", "RB", f"@{decision}"),
                "HALT",
            ]
            prompt = random.choice([
                f"Write NML to classify {inp} with sigmoid output, then flag as {hi_val} if score >= {thresh} using CMPI.",
                f"NML: forward pass with sigmoid, store {score}, then binary {decision} at threshold {thresh}.",
                f"Write NML to flag {inp} if sigmoid output >= {thresh}. Use CMPI and JMPF.",
            ])

        elif variant == 1:
            lines = [
                _fmt("LD", "R0", f"@{inp}"),
                _fmt("SIGM", "RA", "R0"),
                _fmt("ST", "RA", f"@{score}"),
                _fmt("CMPI", "RE", "RA", f"#{thresh}"),
                _fmt("JMPF", "#2"),
                _fmt("LEAF", "RB", f"#{lo_val}"),
                _fmt("JUMP", "#1"),
                _fmt("LEAF", "RB", f"#{hi_val}"),
                _fmt("ST", "RB", f"@{decision}"),
                "HALT",
            ]
            prompt = f"Write NML to apply sigmoid to {inp} and flag as {hi_val} if result >= {thresh}."

        elif variant == 2:
            lines = [
                _fmt("LD", "R0", f"@{inp}"),
                _fmt("CMPI", "RE", "R0", f"#{thresh}"),
                _fmt("JMPF", "#2"),
                _fmt("LEAF", "RB", f"#{lo_val}"),
                _fmt("JUMP", "#1"),
                _fmt("LEAF", "RB", f"#{hi_val}"),
                _fmt("ST", "RB", f"@{decision}"),
                "HALT",
            ]
            prompt = random.choice([
                f"Write NML using CMPI to check if {inp} >= {thresh} and set {decision} to {hi_val}.",
                f"NML: if {inp} >= {thresh} then {decision}={hi_val} else {decision}={lo_val}. Use CMPI.",
                f"Write NML for a threshold check: output {hi_val} when {inp} >= {thresh}.",
            ])

        else:
            lo_label = round(random.uniform(0.0, 10.0), 1)
            hi_label = round(random.uniform(50.0, 100.0), 1)
            lines = [
                _fmt("LD", "R0", f"@{inp}"),
                _fmt("LD", "R1", "@weights"),
                _fmt("MMUL", "R2", "R0", "R1"),
                _fmt("RELU", "R2", "R2"),
                _fmt("CMPI", "RE", "R2", f"#{thresh}"),
                _fmt("JMPF", "#2"),
                _fmt("LEAF", "RA", f"#{lo_label}"),
                _fmt("JUMP", "#1"),
                _fmt("LEAF", "RA", f"#{hi_label}"),
                _fmt("ST", "RA", f"@{decision}"),
                "HALT",
            ]
            prompt = f"Write NML: matmul + ReLU, then if result >= {thresh} output {hi_label}, else {lo_label}."

        prompt += syntax_tag(syntax)
        pairs.append(_pair(prompt, apply_syntax(lines, syntax)))
    return pairs


def gen_cmpi_range(count=200):
    """Two CMPI checks for low/mid/high classification."""
    pairs = []
    for _ in range(count):
        syntax = pick_syntax()
        lo_thresh = round(random.uniform(0.2, 0.4), 1)
        hi_thresh = round(random.uniform(0.6, 0.8), 1)
        inp = random.choice(INPUT_NAMES)

        # CMPI flag = (val < imm). JMPF jumps when val >= imm.
        # Check high first, then low.
        # if val >= hi_thresh: HIGH (2.0)
        # elif val >= lo_thresh: MID (1.0)
        # else: LOW (0.0)
        lines = [
            _fmt("LD", "R0", f"@{inp}"),
            _fmt("CMPI", "RE", "R0", f"#{hi_thresh}"),
            _fmt("JMPF", "#2"),            # val >= hi: jump to HIGH
            _fmt("CMPI", "RE", "R0", f"#{lo_thresh}"),
            _fmt("JMPF", "#4"),            # val >= lo: jump to MID
            # val < lo_thresh: LOW
            _fmt("LEAF", "RA", "#0.0"),
            _fmt("JUMP", "#3"),
            # val >= lo_thresh but < hi_thresh: MID
            _fmt("LEAF", "RA", "#1.0"),
            _fmt("JUMP", "#1"),
            # val >= hi_thresh: HIGH
            _fmt("LEAF", "RA", "#2.0"),
            _fmt("ST", "RA", "@category"),
            "HALT",
        ]

        prompt = random.choice([
            f"Write NML to classify {inp} into 3 levels: LOW (<{lo_thresh}), MID ({lo_thresh}-{hi_thresh}), HIGH (>={hi_thresh}). Use CMPI.",
            f"NML range classifier: output 0 if {inp} < {lo_thresh}, 1 if < {hi_thresh}, 2 if >= {hi_thresh}.",
            f"Write NML for three-way classification using two CMPI thresholds at {lo_thresh} and {hi_thresh}.",
        ]) + syntax_tag(syntax)

        pairs.append(_pair(prompt, apply_syntax(lines, syntax)))
    return pairs


def gen_cmpi_explanation(count=200):
    """Q&A pairs explaining CMPI semantics."""
    pairs = []
    explanations = [
        (
            "How does CMPI work in NML? What does the flag mean?",
            "CMPI RE Rs #imm sets the condition flag to 1 if the register value is LESS THAN the immediate, "
            "and 0 if it is GREATER THAN OR EQUAL. So `CMPI RE RA #0.5` sets flag=1 when RA < 0.5, flag=0 when RA >= 0.5.\n\n"
            "JMPF (jump if false) jumps when flag=0, meaning the value is >= the immediate.\n"
            "JMPT (jump if true) jumps when flag=1, meaning the value is < the immediate."
        ),
        (
            "Explain the CMPI threshold pattern for flagging values above a threshold.",
            "To flag values >= threshold:\n\n"
            "```\nCMPI  RE RA #0.5    ; flag = (RA < 0.5)\n"
            "JMPF  #2            ; if RA >= 0.5, jump to FLAGGED\n"
            "LEAF  RB #0.0       ; RA < 0.5: not flagged\n"
            "JUMP  #1            ; skip past flagged branch\n"
            "LEAF  RB #1.0       ; RA >= 0.5: FLAGGED\n"
            "ST    RB @decision\n```\n\n"
            "Key: JMPF #2 skips 2 instructions (LEAF + JUMP) to land on the FLAGGED branch. "
            "JUMP #1 skips 1 instruction (the second LEAF) to land on ST."
        ),
        (
            "Why are the branches swapped in CMPI compared to what I expect?",
            "CMPI sets flag = (value < immediate), which is the OPPOSITE of what most people expect. "
            "When you write CMPI RE RA #0.5:\n"
            "- flag=1 (true) means RA < 0.5\n"
            "- flag=0 (false) means RA >= 0.5\n\n"
            "JMPF jumps when flag=0, so JMPF jumps when value >= threshold. "
            "This means the 'then' branch (no jump) runs when value < threshold, "
            "and the 'else' branch (after JMPF jump) runs when value >= threshold.\n\n"
            "To output 1.0 when value >= 0.5:\n"
            "CMPI  RE RA #0.5\nJMPF  #2\nLEAF  RB #0.0    ; value < 0.5\n"
            "JUMP  #1\nLEAF  RB #1.0    ; value >= 0.5\nST    RB @result"
        ),
        (
            "What jump offsets should I use with CMPI for a binary decision?",
            "For a standard binary decision pattern:\n\n"
            "Line N:   CMPI  RE RA #threshold\n"
            "Line N+1: JMPF  #2            ; jump to N+1+2+1 = N+4\n"
            "Line N+2: LEAF  RB #lo_value   ; value < threshold\n"
            "Line N+3: JUMP  #1            ; jump to N+3+1+1 = N+5\n"
            "Line N+4: LEAF  RB #hi_value   ; value >= threshold\n"
            "Line N+5: ST    RB @decision\n\n"
            "JMPF #2 and JUMP #1. Both paths converge at the ST instruction. "
            "The offset formula is: target = PC + offset + 1."
        ),
        (
            "How do I use CMPI to make a binary fraud detection decision?",
            "After computing a fraud score with sigmoid (0-1 range):\n\n"
            "SIGM  RA R6              ; fraud probability\n"
            "ST    RA @fraud_score\n"
            "CMPI  RE RA #0.5         ; flag = (RA < 0.5)\n"
            "JMPF  #2                 ; jump to FLAGGED if RA >= 0.5\n"
            "LEAF  RB #0.0            ; score < 0.5: legitimate\n"
            "JUMP  #1                 ; skip to ST\n"
            "LEAF  RB #1.0            ; score >= 0.5: FRAUD\n"
            "ST    RB @fraud_flag\n\n"
            "The JMPF fires when the flag is false (RA >= 0.5), reaching the FRAUD branch."
        ),
    ]

    while len(pairs) < count:
        q, a = random.choice(explanations)
        variation = random.choice(["", " in NML", " for NML programs", ""])
        pairs.append(_pair(q + variation, a))
    return pairs[:count]


def gen_cmpi_symbolic(count=200):
    """CMPI threshold patterns in symbolic syntax."""
    pairs = []
    for _ in range(count):
        thresh = _thresh()
        score = random.choice(SCORE_NAMES)
        decision = random.choice(DECISION_NAMES)
        inp = random.choice(INPUT_NAMES)

        variant = random.randint(0, 1)

        if variant == 0:
            code = (
                f"↓  ι  @{inp}\n"
                f"↓  κ  @weights\n"
                f"×  λ  ι  κ\n"
                f"σ  α  λ\n"
                f"↑  α  @{score}\n"
                f"≺  φ  α  #{thresh}\n"
                f"↘  #2\n"
                f"∎  β  #0.0\n"
                f"→  #1\n"
                f"∎  β  #1.0\n"
                f"↑  β  @{decision}\n"
                f"◼"
            )
            prompt = f"Write symbolic NML to classify {inp}: matmul, sigmoid, then flag if score >= {thresh}."
        else:
            code = (
                f"↓  ι  @{inp}\n"
                f"≺  φ  ι  #{thresh}\n"
                f"↘  #2\n"
                f"∎  β  #0.0\n"
                f"→  #1\n"
                f"∎  β  #1.0\n"
                f"↑  β  @{decision}\n"
                f"◼"
            )
            prompt = f"Symbolic NML: if {inp} >= {thresh} then {decision}=1, else 0. Use ≺ (CMPI) and ↘ (JMPF)."

        pairs.append(_pair(prompt, code))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate CMPI threshold fix pairs")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    print("Generating CMPI threshold fix pairs...")
    print(f"{'─' * 60}")

    generators = [
        ("cmpi_threshold", gen_cmpi_threshold, 400),
        ("cmpi_range", gen_cmpi_range, 200),
        ("cmpi_explanation", gen_cmpi_explanation, 200),
        ("cmpi_symbolic", gen_cmpi_symbolic, 200),
    ]

    all_pairs = []
    for name, gen_fn, count in generators:
        p = gen_fn(count)
        all_pairs.extend(p)
        print(f"  {name:<28} {len(p):>5} pairs")

    random.shuffle(all_pairs)

    print(f"{'─' * 60}")
    print(f"  TOTAL:                             {len(all_pairs):>5} pairs")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nWritten to: {out_path}")

    # Validate non-explanation pairs
    print("\n  Validating sample...")
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import nml_grammar

    valid = 0
    checked = 0
    for p in random.sample(all_pairs, min(200, len(all_pairs))):
        code = p["messages"][1]["content"]
        if "CMPI sets" in code or "flag =" in code or "```" in code:
            continue
        try:
            report = nml_grammar.validate_grammar(code)
            if report.valid:
                valid += 1
            checked += 1
        except Exception:
            checked += 1

    if checked > 0:
        print(f"  Sample validation: {valid}/{checked} ({valid/checked*100:.0f}%) grammar-valid")


if __name__ == "__main__":
    main()
