#!/usr/bin/env python3
"""
NML Jump/Branch Fix Training Data — targets the last remaining LLM failures:
  1. Jump offset math (JMPT #N with correct offset calculation)
  2. Self-contained programs (LEAF-based, no LD required)
  3. LOOP with proper count setup

Output: domain/output/training/nml_jump_fix_pairs.jsonl
"""

import json
import random
from pathlib import Path

random.seed(123)
OUTPUT = Path(__file__).parent.parent / "domain" / "output" / "training" / "nml_jump_fix_pairs.jsonl"

def pair(prompt, code):
    return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": code}]}

def rand_val():
    return round(random.uniform(1.0, 100.0), 1)


# ═══════════════════════════════════════════════════════════════
# 1. Jump offset math — correct JMPT/JMPF/JUMP patterns
# ═══════════════════════════════════════════════════════════════

def gen_jump_if_else():
    """if/else with CMPI + JMPT/JMPF + JUMP — correct offset math."""
    pairs = []
    for _ in range(300):
        thresh = rand_val()
        val_then = rand_val()
        val_else = rand_val()

        code = f"""; if value < {thresh} then {val_then} else {val_else}
LEAF  R0 #{rand_val()}
CMPI  RE R0 #{thresh}
JMPF  #3
LEAF  RA #{val_then}
JUMP  #2
LEAF  RA #{val_else}
ST    RA @result
HALT"""
        prompts = [
            f"Write NML: if a value is less than {thresh}, store {val_then}, otherwise store {val_else}.",
            f"NML with CMPI and JMPF for if/else branching. Then branch: {val_then}, else: {val_else}.",
            f"Write NML using JMPF #3 and JUMP #2 for conditional logic.",
        ]
        pairs.append(pair(random.choice(prompts), code))

    for _ in range(200):
        thresh = rand_val()
        code = f"""LEAF  R0 #{rand_val()}
CMPI  RE R0 #{thresh}
JMPT  #2
LEAF  RA #0.0
JUMP  #1
LEAF  RA #1.0
ST    RA @result
HALT"""
        prompt = f"Write NML using JMPT to branch if value < {thresh}, with JUMP to skip the else."
        pairs.append(pair(prompt, code))

    for _ in range(100):
        thresh = rand_val()
        code = f"""↓  ι  @value
≺  φ  ι  #{thresh}
↘  #3
∎  α  #1.0
→  #2
∎  α  #0.0
↑  α  @result
◼"""
        prompt = f"Symbolic NML: if value < {thresh} then 1.0, else 0.0. Use ≺, ↘, →."
        pairs.append(pair(prompt, code))

    return pairs


def gen_jump_offset_drill():
    """Programs that specifically drill offset = target - current - 1."""
    pairs = []

    for _ in range(200):
        code = f"""LEAF  R0 #{rand_val()}
JUMP  #2
LEAF  R0 #999.0
LEAF  R0 #888.0
SCLR  R1 R0 #2.0
ST    R1 @result
HALT"""
        prompt = "Write NML that uses JUMP #2 to skip over two instructions."
        pairs.append(pair(prompt, code))

    for _ in range(100):
        code = f"""LEAF  R0 #{rand_val()}
LEAF  R1 #{rand_val()}
CMP   R0 R1
JMPT  #1
MOV   R0 R1
ST    R0 @result
HALT"""
        prompt = "Write NML: compare two values with CMP, use JMPT #1 to skip one instruction."
        pairs.append(pair(prompt, code))

    for _ in range(100):
        code = f"""LEAF  RA #0.0
LEAF  RD #1.0
LEAF  RC #1.0
TACC  RA RA RD
TACC  RD RD RC
CMPI  RE RD #6.0
JMPT  #-5
ST    RA @result
HALT"""
        prompt = "Write NML: backward jump loop summing 1 to 5 using JMPT with negative offset."
        pairs.append(pair(prompt, code))

    return pairs


# ═══════════════════════════════════════════════════════════════
# 2. Self-contained LEAF-based programs (no LD required)
# ═══════════════════════════════════════════════════════════════

def gen_self_contained():
    pairs = []

    for _ in range(100):
        a, b = rand_val(), rand_val()
        code = f"""LEAF  R0 #{a}
LEAF  R1 #{b}
MADD  R2 R0 R1
ST    R2 @result
HALT"""
        prompt = f"Write a self-contained NML program to add {a} and {b}."
        pairs.append(pair(prompt, code))

    for _ in range(100):
        a, b = rand_val(), rand_val()
        code = f"""LEAF  R0 #{a}
LEAF  R1 #{b}
EMUL  R2 R0 R1
ST    R2 @result
HALT"""
        prompt = f"Write NML to multiply {a} by {b} using EMUL."
        pairs.append(pair(prompt, code))

    for _ in range(50):
        v = rand_val()
        s = round(random.uniform(0.1, 5.0), 2)
        code = f"""LEAF  R0 #{v}
SCLR  R1 R0 #{s}
ST    R1 @result
HALT"""
        prompt = f"Write NML to scale {v} by {s}."
        pairs.append(pair(prompt, code))

    for _ in range(50):
        v = rand_val()
        code = f"""LEAF  R0 #{v}
RELU  R1 R0
ST    R1 @result
HALT"""
        prompt = f"Write NML to apply ReLU to {v}."
        pairs.append(pair(prompt, code))

    for _ in range(50):
        v = rand_val()
        code = f"""LEAF  R0 #{v}
SIGM  R1 R0
ST    R1 @result
HALT"""
        prompt = f"Write NML to compute sigmoid of {v}."
        pairs.append(pair(prompt, code))

    for _ in range(50):
        a, b = rand_val(), rand_val()
        code = f"""LEAF  R0 #{a}
LEAF  R1 #{b}
SDOT  R2 R0 R1
ST    R2 @result
HALT"""
        prompt = f"Write NML for dot product of scalar {a} and {b} using SDOT."
        pairs.append(pair(prompt, code))

    for _ in range(50):
        a, b = rand_val(), rand_val()
        code = f"""∎  ι  #{a}
∎  κ  #{b}
⊕  λ  ι  κ
↑  λ  @result
◼"""
        prompt = f"Symbolic NML to add {a} and {b}."
        pairs.append(pair(prompt, code))

    for _ in range(50):
        v, s = rand_val(), round(random.uniform(0.01, 3.0), 2)
        code = f"""∎  ι  #{v}
∗  κ  ι  #{s}
↑  κ  @result
◼"""
        prompt = f"Symbolic NML to scale {v} by {s}."
        pairs.append(pair(prompt, code))

    return pairs


# ═══════════════════════════════════════════════════════════════
# 3. LOOP with proper count setup
# ═══════════════════════════════════════════════════════════════

def gen_loop_fix():
    pairs = []

    for _ in range(100):
        n = random.randint(3, 20)
        code = f"""ALLC  RA #[1]
LEAF  R2 #1.0
LOOP  #{n}
TACC  RA RA R2
ENDP
ST    RA @result
HALT"""
        prompt = f"Write NML to sum 1+1+...+1 for {n} iterations using LOOP #{n}."
        pairs.append(pair(prompt, code))

    for _ in range(50):
        n = random.randint(3, 20)
        code = f"""LEAF  RD #{n}.0
ALLC  RA #[1]
LEAF  R2 #1.0
LOOP  RD
TACC  RA RA R2
ENDP
ST    RA @result
HALT"""
        prompt = f"Write NML: set RD to {n}, then use LOOP RD to iterate."
        pairs.append(pair(prompt, code))

    for _ in range(50):
        n = random.randint(3, 15)
        base = rand_val()
        rate = round(random.uniform(1.01, 1.1), 3)
        code = f"""LEAF  R0 #{base}
LEAF  R1 #{rate}
LOOP  #{n}
EMUL  R0 R0 R1
ENDP
ST    R0 @result
HALT"""
        prompt = f"Write NML for compound growth: start at {base}, multiply by {rate} for {n} periods."
        pairs.append(pair(prompt, code))

    return pairs


def main():
    all_pairs = []

    generators = [
        ("Jump if/else", gen_jump_if_else),
        ("Jump offset drill", gen_jump_offset_drill),
        ("Self-contained", gen_self_contained),
        ("LOOP fix", gen_loop_fix),
    ]

    for name, gen_fn in generators:
        p = gen_fn()
        all_pairs.extend(p)
        print(f"  {name:<22} {len(p):>5} pairs")

    random.shuffle(all_pairs)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n  Total: {len(all_pairs)} pairs")
    print(f"  Written to: {OUTPUT}")

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import nml_grammar
    valid = 0
    for p in all_pairs:
        try:
            r = nml_grammar.validate_grammar(p["messages"][1]["content"])
            if r.valid:
                valid += 1
        except:
            pass
    print(f"  Grammar validation: {valid}/{len(all_pairs)} ({valid/len(all_pairs)*100:.0f}%)")


if __name__ == "__main__":
    main()
